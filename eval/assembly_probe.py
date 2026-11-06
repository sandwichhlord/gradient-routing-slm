"""What does each assembly respond to? Reads a saved partition and a checkpoint, and scores it on held-out data.

Outputs (next to the assignments file, or in --save_dir):
  assembly_probe.csv            per assembly: size, top 20 tokens by mean activation, correlation of its activity with
                                the token's position in the window and with the token's log frequency in the training data
  assembly_domain_profile.csv   per assembly: fraction of story, math and code tokens with relative activity r > 1.5, plus
                                its size and top 10 tokens. r is activity / the assembly's mean activity on the same
                                data, with domains weighted by the training mix

Activity is the assembly's mean |h| at a token, the quantity plasticity is computed from. Position is the position inside
a random window, so it mostly reflects how much context the token has, not where it sits in a document. Tokens seen fewer
than --min_count times are left out of the top lists, or rare tokens would dominate them. Domain labels are used to report
results and nothing else.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import tiktoken
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablate_assemblies import domain_mix  # noqa: E402
from common import DOMAINS, ROOT, autocast_ctx, get_batch, load_checkpoint, load_val, open_bin, pick_device  # noqa: E402
from model.assemblies import LEFTOVER  # noqa: E402

VOCAB = 50304


def default_assignments(out_dir):
    for path in (os.path.join(out_dir, "assemblies_final.json"), os.path.join(out_dir, "posthoc", "assemblies_posthoc.json")):
        if os.path.exists(path):
            with open(path) as f:
                if all(lab is not None for lab in json.load(f)["layers"]):
                    return path
    raise SystemExit(f"no usable assignments in {out_dir}: run eval/ablate_assemblies.py first")


def pearson(n, sx, sy, sxx, syy, sxy):
    """Correlations from running sums; sx, sxx and sxy are per assembly, the y terms are scalars."""
    denom = np.sqrt(np.maximum(n * sxx - sx ** 2, 0.0) * max(n * syy - sy ** 2, 0.0))
    return np.divide(n * sxy - sx * sy, denom, out=np.full_like(sx, np.nan), where=denom > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"), help="run folder holding ckpt.pt")
    p.add_argument("--ckpt", default=None, help="checkpoint path; default out_dir/ckpt.pt")
    p.add_argument("--assignments", default=None, help="default: assemblies_final.json, else posthoc/assemblies_posthoc.json")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "out"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batches", type=int, default=20, help="batches per domain")
    p.add_argument("--batch_size", type=int, default=None, help="default: training batch size")
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--threshold", type=float, default=1.5, help="relative activity counted as active")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt or os.path.join(args.out_dir, "ckpt.pt"), device)
    cfg = model.config
    bs = args.batch_size or ckpt["config"]["batch_size"]
    path = args.assignments or default_assignments(args.out_dir)
    with open(path) as f:
        layers = [np.asarray(lab) for lab in json.load(f)["layers"]]
    save_dir = args.save_dir or os.path.dirname(path)
    os.makedirs(save_dir, exist_ok=True)

    val = load_val(args.data_dir)
    val_score = {d: val[d][len(val[d]) // 2:] for d in DOMAINS}
    freq = np.bincount(np.asarray(open_bin(args.data_dir, "train.bin")), minlength=VOCAB).astype(np.float64)
    log_freq = torch.as_tensor(np.log1p(freq), dtype=torch.float64)
    enc = tiktoken.get_encoding("gpt2")

    groups = [np.unique(lab[lab != LEFTOVER]) for lab in layers]
    pos_of = [torch.as_tensor(np.searchsorted(g, np.where(lab == LEFTOVER, g[0], lab)), dtype=torch.long, device=device)
              for g, lab in zip(groups, layers)]
    keep = [torch.as_tensor(lab != LEFTOVER, device=device) for lab in layers]
    sizes = [np.array([(lab == g).sum() for g in gs], dtype=np.float64) for gs, lab in zip(groups, layers)]
    captured = [None] * len(layers)
    hooks = []
    for layer, mlp in enumerate(model.mlps()):
        def hook(_m, _i, out, layer=layer):
            h = out.float().abs() * keep[layer]
            G = len(groups[layer])
            a = torch.zeros(*h.shape[:-1], G, device=h.device).index_add_(-1, pos_of[layer], h)
            captured[layer] = (a / torch.as_tensor(sizes[layer], dtype=a.dtype, device=a.device)).double().cpu()
        hooks.append(mlp.gelu.register_forward_hook(hook))

    def batches():
        rng = np.random.default_rng(1234)
        for d in range(len(DOMAINS)):
            for _ in range(args.batches):
                yield d, get_batch(val_score[DOMAINS[d]], bs, cfg.block_size, device, rng)[0]

    model.eval()
    n_layer = len(layers)
    token_sum = [torch.zeros(VOCAB, len(g), dtype=torch.float64) for g in groups]
    token_count = torch.zeros(VOCAB, dtype=torch.float64)
    sx = [np.zeros(len(g)) for g in groups]
    sxx = [np.zeros(len(g)) for g in groups]
    sxp = [np.zeros(len(g)) for g in groups]
    sxf = [np.zeros(len(g)) for g in groups]
    n, sp, spp, sf, sff = 0, 0.0, 0.0, 0.0, 0.0
    dom_mean = [np.zeros((len(DOMAINS), len(g))) for g in groups]
    dom_tokens = np.zeros(len(DOMAINS))
    with torch.no_grad(), autocast_ctx(device):
        for d, x in batches():
            model(x)
            x_cpu = x.cpu().view(-1)
            token_count.index_add_(0, x_cpu, torch.ones(x_cpu.shape[0], dtype=torch.float64))
            pos = torch.arange(x.shape[1], dtype=torch.float64).repeat(x.shape[0])
            f = log_freq[x_cpu]
            n += x_cpu.shape[0]
            sp += pos.sum().item()
            spp += pos.pow(2).sum().item()
            sf += f.sum().item()
            sff += f.pow(2).sum().item()
            dom_tokens[d] += x_cpu.shape[0]
            for layer in range(n_layer):
                a = captured[layer].view(-1, len(groups[layer]))
                token_sum[layer].index_add_(0, x_cpu, a)
                sx[layer] += a.sum(dim=0).numpy()
                sxx[layer] += a.pow(2).sum(dim=0).numpy()
                sxp[layer] += (a * pos.unsqueeze(1)).sum(dim=0).numpy()
                sxf[layer] += (a * f.unsqueeze(1)).sum(dim=0).numpy()
                dom_mean[layer][d] += a.sum(dim=0).numpy()

    # relative activity needs each assembly's overall mean first, hence a second pass
    mix = np.array(domain_mix(args.data_dir))
    baseline = [(dom_mean[layer] / dom_tokens[:, None] * mix[:, None]).sum(axis=0) for layer in range(n_layer)]
    active = [np.zeros((len(DOMAINS), len(g))) for g in groups]
    with torch.no_grad(), autocast_ctx(device):
        for d, x in batches():
            model(x)
            for layer in range(n_layer):
                a = captured[layer].view(-1, len(groups[layer])).numpy()
                active[layer][d] += (a / np.maximum(baseline[layer], 1e-12) > args.threshold).sum(axis=0)
    for h in hooks:
        h.remove()

    probe_rows, profile_rows = [], []
    seen = (token_count >= args.min_count).numpy()
    for layer in range(n_layer):
        means = (token_sum[layer] / token_count.clamp_min(1).unsqueeze(1)).numpy()
        corr_pos = pearson(n, sx[layer], sp, sxx[layer], spp, sxp[layer])
        corr_freq = pearson(n, sx[layer], sf, sxx[layer], sff, sxf[layer])
        for i, g in enumerate(groups[layer]):
            order = np.argsort(-np.where(seen, means[:, i], -np.inf))[:20]
            top = [enc.decode([int(t)]) for t in order if seen[t] and t < enc.n_vocab]
            probe_rows.append([layer, int(g), int(sizes[layer][i]), f"{corr_pos[i]:.4f}", f"{corr_freq[i]:.4f}",
                               json.dumps(top, ensure_ascii=False)])
            profile_rows.append([layer, int(g), int(sizes[layer][i]),
                                 *[f"{active[layer][d, i] / dom_tokens[d]:.4f}" for d in range(len(DOMAINS))],
                                 json.dumps(top[:10], ensure_ascii=False)])
    with open(os.path.join(save_dir, "assembly_probe.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "assembly", "size", "corr_position", "corr_log_token_freq", "top20_tokens"])
        w.writerows(probe_rows)
    with open(os.path.join(save_dir, "assembly_domain_profile.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "assembly", "size", *[f"frac_active_{d}" for d in DOMAINS], "top10_tokens"])
        w.writerows(profile_rows)
    print(f"[probe] wrote assembly_probe.csv and assembly_domain_profile.csv to {save_dir} ({len(probe_rows)} assemblies)")


if __name__ == "__main__":
    main()
