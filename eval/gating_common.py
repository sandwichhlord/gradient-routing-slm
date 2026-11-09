"""Shared pieces of the gated-mode evaluations (damage_table, failure_trace, unlearn, context_sensitivity).

Groups: a gated checkpoint brings its own (trained fixed or evolved groups, gates applied). For any other checkpoint
(dense) the groups are computed post hoc: co-activation + Leiden on the FIRST half of each val set, reduced to n_groups
per layer (model.gating.posthoc_groups), cached in out_dir/groups_posthoc.json, and selections are computed with
inference-only gating (GateRouter apply=False: the forward pass stays dense). Scoring always uses the SECOND half of
val, or generated held-out problems, never the data the groups came from.
"""
import json
import os
import sys

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from common import DOMAINS, ROOT, autocast_ctx, load_checkpoint, load_val, pick_device  # noqa: E402,F401
from deconf import TAGS  # noqa: E402
from model.gating import GateRouter, parse_layers, posthoc_groups  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
NEWLINE_IDS = None


def add_args(p, windows=96):
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"), help="run folder holding ckpt.pt; outputs go here")
    p.add_argument("--ckpt", default=None, help="default out_dir/ckpt.pt")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "out"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--windows", type=int, default=windows, help="eval windows per token category and per domain")
    p.add_argument("--seed", type=int, default=0)
    # post-hoc groups (dense checkpoints only; a gated checkpoint uses its own settings)
    p.add_argument("--gated_layers", default="2,3,4,5")
    p.add_argument("--n_groups", type=int, default=8)
    p.add_argument("--gate_k", type=int, default=3)
    p.add_argument("--route_smooth", type=float, default=0.9)
    p.add_argument("--gate_score", default="share", choices=["share", "relative"])
    p.add_argument("--posthoc_batches", type=int, default=200, help="batches of the first val half for clustering")
    p.add_argument("--knn", type=int, default=32)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min_size", type=int, default=16)


def load(args):
    """(model, ckpt, router, source, device). source is "trained-fixed", "trained-evolved" or "posthoc"."""
    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt or os.path.join(args.out_dir, "ckpt.pt"), device)
    if model.gate_router is not None:
        r = model.gate_router
        return model, ckpt, r, f"trained-{r.groups}", device
    layers = parse_layers(args.gated_layers, model.config.n_layer)
    settings = {"layers": layers, "n_groups": args.n_groups, "knn": args.knn, "resolution": args.resolution,
                "min_size": args.min_size, "posthoc_batches": args.posthoc_batches, "seed": args.seed,
                "ckpt_iter": ckpt.get("iter")}
    path = os.path.join(args.out_dir, "groups_posthoc.json")
    cached = None
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("settings") != settings:
            cached = None
    if cached is None:
        print(f"[groups] post-hoc groups: {args.posthoc_batches} batches of the first val half, layers {layers}", flush=True)
        val = load_val(args.data_dir)
        rng = np.random.default_rng(args.seed)
        T = model.config.block_size
        half = {d: val[d][:len(val[d]) // 2] for d in DOMAINS}
        per = max(1, args.batch_size // len(DOMAINS))
        batches = (torch.as_tensor(np.stack([half[d][i:i + T].astype(np.int64) for d in DOMAINS
                                             for i in rng.integers(0, len(half[d]) - T - 1, per)]), device=device)
                   for _ in range(args.posthoc_batches))
        with torch.no_grad(), autocast_ctx(device):
            groups = posthoc_groups(model, batches, layers, args.n_groups, args.knn, args.resolution, args.min_size,
                                    args.seed)
        cached = {"settings": settings, "layers": {str(k): [int(x) for x in v[0]] for k, v in groups.items()},
                  "neuron_ema": {str(k): [float(x) for x in v[1]] for k, v in groups.items()}}
        with open(path, "w") as f:
            json.dump(cached, f)
    router = GateRouter(model, layers=layers, n_groups=args.n_groups, k=args.gate_k, smooth=args.route_smooth,
                        groups="fixed", cluster_start=0, apply=False, score=args.gate_score)
    for layer in layers:
        router.set_labels(layer, cached["layers"][str(layer)])
        router.set_neuron_ema(layer, cached["neuron_ema"][str(layer)])
    sizes = {layer: np.bincount(cached["layers"][str(layer)], minlength=args.n_groups).tolist() for layer in layers}
    print(f"[groups] post-hoc group sizes: {sizes}", flush=True)
    return model, ckpt, router, "posthoc", device


def group_list(router):
    """[(layer, group, neuron indices)] for every group of every gated layer."""
    return [(layer, g, np.flatnonzero(router.labels[i] == g)) for i, layer in enumerate(router.layers) for g in range(router.G)]


def masks_for(router, deleted):
    """{layer: (hidden,) float mask} that zeroes the neuron sets in `deleted` = [(layer, neuron indices)]."""
    masks = {}
    for layer, idx in deleted:
        m = masks.setdefault(layer, np.ones(router.hidden, dtype=np.float32))
        m[idx] = 0.0
    return masks


# ------------------------------------------------------------------------------ tagged eval sets
def eval_sets(data_dir, block_size, windows, seed=0):
    """{name: (x, y, mask)} from the second half of val. For each tag: windows containing at least one target token of
    that tag, and the mask selects those targets. For each domain: random windows, mask all targets."""
    rng = np.random.default_rng(seed + 7)
    out = {}
    tag_tokens = {}
    for d in DOMAINS:
        data = np.memmap(os.path.join(data_dir, f"val_{d}.bin"), dtype=np.uint16, mode="r")
        tags = np.memmap(os.path.join(data_dir, f"val_{d}_tags.bin"), dtype=np.uint8, mode="r")
        lo, hi = len(data) // 2, len(data) - block_size - 1
        starts = rng.integers(lo, hi, windows)
        out[f"domain_{d}"] = _windows(data, tags, starts, block_size, None)
        t = np.asarray(tags[lo:])
        for c, name in enumerate(TAGS):
            pos = np.flatnonzero(t == c) + lo
            pos = pos[(pos - 1 >= lo) & (pos - 1 <= hi)]
            if len(pos):
                tag_tokens.setdefault(name, []).append((data, tags, pos))
    for c, name in enumerate(TAGS):
        sources = tag_tokens.get(name, [])
        if not sources:
            continue
        total = sum(len(s[2]) for s in sources)
        xs, ys, ms = [], [], []
        for data, tags, pos in sources:
            n = int(round(windows * len(pos) / total))
            if n == 0:
                continue
            ends = rng.choice(pos, size=n)                          # target position of a token with this tag
            offs = rng.integers(0, block_size, n)
            starts = np.clip(ends - 1 - offs, len(data) // 2, len(data) - block_size - 1)
            x, y, m = _windows(data, tags, starts, block_size, c)
            xs.append(x), ys.append(y), ms.append(m)
        out[f"tag_{name}"] = (torch.cat(xs), torch.cat(ys), torch.cat(ms))
    return out


def _windows(data, tags, starts, T, tag):
    x = torch.as_tensor(np.stack([data[s:s + T].astype(np.int64) for s in starts]))
    y = torch.as_tensor(np.stack([data[s + 1:s + T + 1].astype(np.int64) for s in starts]))
    if tag is None:
        m = torch.ones_like(y, dtype=torch.bool)
    else:
        m = torch.as_tensor(np.stack([tags[s + 1:s + T + 1] == tag for s in starts]))
    return x, y, m


@torch.no_grad()
def masked_losses(model, sets, device, batch_size=16):
    """{name: mean next-token loss over the masked targets}."""
    out = {}
    for name, (x, y, m) in sets.items():
        total, count = 0.0, 0
        for s in range(0, len(x), batch_size):
            xb, yb, mb = x[s:s + batch_size].to(device), y[s:s + batch_size].to(device), m[s:s + batch_size].to(device)
            with autocast_ctx(device):
                logits, _ = model(xb, yb)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none")
            total += float(loss[mb.reshape(-1)].sum())
            count += int(mb.sum())
        out[name] = total / max(count, 1)
    return out


# ------------------------------------------------------------------------------ arithmetic
def _newline_ids():
    global NEWLINE_IDS
    if NEWLINE_IDS is None:
        NEWLINE_IDS = {i for i in range(ENC.n_vocab) if "\n" in ENC.decode([i])} | {EOT}
    return NEWLINE_IDS


@torch.no_grad()
def solve(model, problems, device, batch_size=256, max_new=7, router=None):
    """Greedy answers to [(prompt, answer)]. Returns dict with correct (n,) bool, preds (list of str) and, with a router,
    active (n, n_layers, G): for each gated layer and group, the share of the problem's answer positions where the group
    was selected. Answer positions are the ones that emitted the answer: the ":" after "A" and each answer token but
    the last."""
    stop = _newline_ids()
    prompts = [[EOT] + ENC.encode_ordinary(p) for p, _ in problems]
    by_len = {}
    for i, pr in enumerate(prompts):
        by_len.setdefault(len(pr), []).append(i)
    n = len(problems)
    correct, preds = np.zeros(n, dtype=bool), [""] * n
    active = None if router is None else np.zeros((n, len(router.layers), router.G), dtype=np.float32)
    model.eval()
    for L, members in by_len.items():
        for s in range(0, len(members), batch_size):
            ids = members[s:s + batch_size]
            x = torch.as_tensor([prompts[i] for i in ids], device=device)
            with autocast_ctx(device):
                for _ in range(max_new):
                    logits, _ = model(x)
                    x = torch.cat([x, logits[:, -1].float().argmax(-1, keepdim=True)], dim=1)
                if router is not None:
                    model(x)        # one pass over the whole sequence to read every position's selection
            gen = x[:, L:].tolist()
            for row, i in enumerate(ids):
                cut = next((j for j, t in enumerate(gen[row]) if t in stop), len(gen[row]))
                preds[i] = ENC.decode(gen[row][:cut]).strip()
                correct[i] = preds[i] == problems[i][1]
                if router is not None:
                    n_pos = max(cut, 1)
                    for li in range(len(router.layers)):
                        active[i, li] = router.last_selected[li][row, L - 1:L - 1 + n_pos].float().mean(0).cpu().numpy()
    return {"correct": correct, "preds": preds, "active": active}


@torch.no_grad()
def selection_rate(model, router, sets, names, device, batch_size=16):
    """(n_layers, G) share of all tokens in the given eval sets where each group was selected."""
    total, count = np.zeros((len(router.layers), router.G)), 0
    model.eval()
    for name in names:
        x = sets[name][0]
        for s in range(0, len(x), batch_size):
            with autocast_ctx(device):
                model(x[s:s + batch_size].to(device))
            for li in range(len(router.layers)):
                total[li] += router.last_selected[li].float().sum(dim=(0, 1)).cpu().numpy()
            count += x[s:s + batch_size].numel()
    return total / max(count, 1)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
