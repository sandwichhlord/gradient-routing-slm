"""Zero each MLP block (layer, block) in turn and measure val loss on every domain.

Outputs (in --out_dir):
  ablation.csv            one row per (layer, block): ablated losses, damage (= loss increase), specialization S
  ablation_heatmap.png    rows = layer.block, cols = story/math/code, colour = loss increase over unablated
  ablation_summary.txt    unablated losses, mean S, most specialised blocks
  block_norms.csv         mean per-token L2 norm of each block's activation, per domain (unablated model)

Specialization per block: p = max(damage, 0) / sum(max(damage, 0)) over the 3 domains,
  S = 1 - H(p) / log(3)   (0 = damage spread evenly, 1 = damage confined to one domain).
Blocks whose ablation does no damage on any domain have undefined S and are left out of the mean.
All conditions are evaluated on the SAME fixed val windows, so differences are paired.
"""
import argparse
import csv
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (DOMAINS, ROOT, autocast_ctx, eval_domains, eval_loss, get_batch,  # noqa: E402
                    load_checkpoint, load_val, pick_device)

INK, MUTED = "#0b0b0b", "#52514e"
# diverging blue (ablation helped) <-> neutral gray <-> red (ablation hurt)
CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def specialization(damage):
    d = np.clip(np.asarray(damage, dtype=np.float64), 0.0, None)
    total = d.sum()
    if total <= 1e-12:
        return float("nan")
    p = d / total
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    return 1.0 - h / math.log(len(p))


@torch.no_grad()
def check_mask_works(model, val, block_size, device):
    """Asserts: ablating a block changes the logits; an all-ones mask and clearing restore the dense model."""
    x, y = get_batch(val["story"], 4, block_size, device, np.random.default_rng(0))
    with autocast_ctx(device):
        base = model(x, y)[0].float()
        model.set_block_mask(0, torch.ones(model.config.n_blocks))
        ones = model(x, y)[0].float()
        model.set_ablation(0, 0)
        abl = model(x, y)[0].float()
        model.clear_ablation()
        again = model(x, y)[0].float()
    diff = (abl - base).abs().max().item()
    assert diff > 1e-4, f"ablating layer 0 block 0 did not change logits (max |diff| = {diff})"
    assert torch.allclose(ones, base, atol=1e-3), "all-ones mask should equal the dense model"
    assert torch.allclose(again, base, atol=1e-3), "clear_ablation() did not restore the dense model"
    print(f"[ablate] mask check passed: ablating L0.B0 changes logits by up to {diff:.4f}; "
          f"all-ones mask and clear_ablation() reproduce the dense model")


def plot_heatmap(names, dmg, out_path, base):
    L = float(np.nanmax(np.abs(dmg))) or 1e-6
    fig_h = max(3.0, 0.24 * len(names) + 1.6)
    fig, ax = plt.subplots(figsize=(5.2, fig_h), dpi=150)
    im = ax.imshow(dmg, cmap=CMAP, norm=Normalize(-L, L), aspect="auto")
    ax.set_xticks(range(len(DOMAINS)), [f"{d}\n(base {base[d]:.2f})" for d in DOMAINS], fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), names, fontsize=6 if len(names) > 24 else 8, color=INK)
    ax.set_xlabel("validation domain", color=INK)
    ax.set_ylabel("ablated MLP block (layer.block)", color=INK)
    ax.set_title("Loss increase when one MLP block is zeroed", color=INK, fontsize=10, loc="left")
    # thin surface-coloured gaps between cells
    ax.set_xticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(names)):
        for j in range(len(DOMAINS)):
            v = dmg[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=5.5 if len(names) > 24 else 7,
                    color="white" if abs(v) > 0.8 * L else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Δ val loss vs unablated (nats)", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"))
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "out"))
    p.add_argument("--device", default="auto")
    p.add_argument("--eval_iters", type=int, default=20, help="batches per domain per condition")
    p.add_argument("--batch_size", type=int, default=None, help="default: training batch size")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), device)
    cfg = model.config
    bs = args.batch_size or ckpt["config"]["batch_size"]
    val = load_val(args.data_dir)
    check_mask_works(model, val, cfg.block_size, device)

    def run():
        return eval_domains(model, val, args.eval_iters, bs, cfg.block_size, device)

    t0 = time.time()
    base, norms = {}, {}
    for d in DOMAINS:  # unablated pass per domain, also recording per-block activation norms
        model.record_block_norms(True)  # (re)sets the accumulators
        base[d] = eval_loss(model, val[d], args.eval_iters, bs, cfg.block_size, device)
        norms[d] = model.get_block_norms().numpy()
    model.record_block_norms(False)
    print(f"[ablate] unablated: " + " | ".join(f"{d} {base[d]:.4f}" for d in DOMAINS), flush=True)

    rows, names, dmg = [], [], []
    n_total = cfg.n_layer * cfg.n_blocks
    for layer in range(cfg.n_layer):
        for block in range(cfg.n_blocks):
            model.set_ablation(layer, block)
            losses = run()
            damage = [losses[d] - base[d] for d in DOMAINS]
            s = specialization(damage)
            names.append(f"{layer}.{block}")
            dmg.append(damage)
            rows.append([layer, block, f"{layer}.{block}", *[f"{losses[d]:.5f}" for d in DOMAINS],
                         *[f"{x:.5f}" for x in damage], "" if math.isnan(s) else f"{s:.4f}"])
            done = len(rows)
            eta = (time.time() - t0) / done * (n_total - done)
            print(f"[ablate] {done}/{n_total} L{layer}.B{block}: "
                  + " ".join(f"{d} {x:+.4f}" for d, x in zip(DOMAINS, damage))
                  + f" | S={s:.3f} | ETA {eta:.0f}s", flush=True)
    model.clear_ablation()
    dmg = np.array(dmg)

    with open(os.path.join(args.out_dir, "ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "block", "name", *[f"loss_{d}" for d in DOMAINS], *[f"dmg_{d}" for d in DOMAINS],
                    "specialization"])
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "block_norms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "block", *[f"norm_{d}" for d in DOMAINS]])
        for layer in range(cfg.n_layer):
            for block in range(cfg.n_blocks):
                w.writerow([layer, block, *[f"{norms[d][layer, block]:.5f}" for d in DOMAINS]])
    plot_heatmap(names, dmg, os.path.join(args.out_dir, "ablation_heatmap.png"), base)

    S = np.array([specialization(r) for r in dmg])
    valid = ~np.isnan(S)
    mean_s = float(S[valid].mean()) if valid.any() else float("nan")
    top = np.argsort(np.where(valid, -S, np.inf))[:5]
    summary = [
        f"checkpoint iter: {ckpt['iter']}",
        f"eval: {args.eval_iters} batches x {bs} x {cfg.block_size} tokens per domain per condition",
        "unablated val loss: " + ", ".join(f"{d}={base[d]:.4f}" for d in DOMAINS),
        f"mean specialization S over {int(valid.sum())}/{len(S)} blocks with positive damage: {mean_s:.4f}",
        "most specialised blocks: " + ", ".join(
            f"{names[i]} (S={S[i]:.3f}, worst={DOMAINS[int(np.argmax(dmg[i]))]})" for i in top if valid[i]),
        f"time: {time.time() - t0:.0f}s",
    ]
    with open(os.path.join(args.out_dir, "ablation_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("[ablate] " + "\n[ablate] ".join(summary))
    print(f"[ablate] wrote ablation.csv, ablation_heatmap.png, ablation_summary.txt, block_norms.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
