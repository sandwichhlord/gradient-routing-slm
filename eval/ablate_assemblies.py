"""Ablate neuron assemblies: zero each one's neurons and measure the val loss increase on every domain.

Works on any checkpoint:
  --source final    use assemblies_final.json saved by an assembly-mode run
  --source posthoc  compute assemblies now: run the co-activation + Leiden procedure on held-out data. On a dense
                    checkpoint this is the key control: how much specialization appears with no routing at all?
  --source auto     final if the run saved assignments, otherwise posthoc

The val sets are split in two: post-hoc clustering only sees the first half of each, and ablation is always scored on the
second half. Scoring on the data used to form the groups would flatter them, and both sources use the same scoring data so
they can be compared. These S values use assemblies, not the 8 fixed blocks of eval/ablate.py, and a different val subset,
so compare them with each other and not with eval/ablate.py.

Random baseline: for each real assembly, --random_sets random sets of the same size from the same layer are ablated, so
any S or damage per neuron that only reflects "removing this many neurons" shows up in both. Bootstrap CIs resample
assemblies. They do not cover variation between seeds, and assemblies in one model are not independent.

Assemblies in degenerate layers (see model.assemblies.partition_health) are still ablated and listed but left out of the
headline numbers, because a layer-sized "assembly" says nothing about specialization. Dead-neuron groups are not ablated.

Outputs (in --save_dir; default out_dir for final, out_dir/posthoc for posthoc):
  assembly_ablation.csv, assembly_random.csv, assembly_summary.json, assembly_summary.txt, assembly_heatmap.png,
  assembly_sizes.png, and assemblies_posthoc.json for a post-hoc run.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablate import plot_heatmap, specialization  # noqa: E402
from common import (DOMAINS, ROOT, autocast_ctx, eval_loss, get_batch, load_checkpoint, load_val,  # noqa: E402
                    pick_device)
from model.assemblies import LEFTOVER, CoactivationTracker, cluster_correlation, partition_health  # noqa: E402

INK, MUTED, GRID, SERIES = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"


def domain_mix(data_dir):
    """Windows per domain follow the training mix when meta.json has it, otherwise they are equal."""
    path = os.path.join(data_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            counts = json.load(f).get("train_tokens_per_domain")
        if counts:
            total = sum(counts.values())
            return [counts[d] / total for d in DOMAINS]
    return [1.0 / len(DOMAINS)] * len(DOMAINS)


@torch.no_grad()
def posthoc_labels(model, val_cluster, mix, args, device, batch_size):
    """The training procedure on held-out data: EMA of correlations over `steps` batches, then Leiden per layer."""
    mlps = model.mlps()
    hidden = mlps[0].n_blocks * mlps[0].block_dim
    tracker = CoactivationTracker(len(mlps), hidden, decay=0.99, sample_tokens=args.sample_tokens, seed=args.seed,
                                  device=device)
    hooks = [mlp.gelu.register_forward_hook(lambda _m, _i, out, layer=layer: tracker.update(layer, out))
             for layer, mlp in enumerate(mlps)]
    rng = np.random.default_rng(args.seed)
    counts = [max(1, round(p * batch_size)) for p in mix]
    model.eval()
    with autocast_ctx(device):
        for step in range(args.posthoc_steps):
            xs = [get_batch(val_cluster[d], n, model.config.block_size, device, rng)[0] for d, n in zip(DOMAINS, counts)]
            model(torch.cat(xs))
    for h in hooks:
        h.remove()
    return [cluster_correlation(tracker.corr[layer].cpu().numpy(), args.knn, args.resolution, args.min_size,
                                args.seed + layer, dead=tracker.dead(layer)) for layer in range(len(mlps))]


def bootstrap_ci(values, n_boot=10000, seed=0):
    """Percentile 95% CI of the mean, resampling the values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def weighted(values, weights):
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else float("nan")


def plot_sizes(rows, n_layer, path):
    fig, ax = plt.subplots(figsize=(8, 0.5 * n_layer + 1.6), dpi=150)
    rng = np.random.default_rng(0)
    for layer in range(n_layer):
        sizes = np.array([r["size"] for r in rows if r["layer"] == layer])
        ax.scatter(sizes, layer + rng.uniform(-0.18, 0.18, len(sizes)), s=22, color=SERIES, alpha=0.8, linewidths=0)
        ax.text(1.01, layer, f"{len(sizes)} assemblies", transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                color=INK)
    ax.set_xscale("log")
    ax.set_yticks(range(n_layer), [f"layer {i}" for i in range(n_layer)], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("assembly size (neurons, log scale)", color=INK, fontsize=9)
    ax.set_title("Assembly sizes per layer", color=INK, fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, axis="x", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"), help="run folder holding ckpt.pt")
    p.add_argument("--ckpt", default=None, help="checkpoint path; default out_dir/ckpt.pt")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "out"))
    p.add_argument("--device", default="auto")
    p.add_argument("--source", choices=["auto", "final", "posthoc"], default="auto")
    p.add_argument("--eval_iters", type=int, default=20, help="batches per domain per assembly")
    p.add_argument("--batch_size", type=int, default=None, help="default: training batch size")
    p.add_argument("--posthoc_steps", type=int, default=500, help="batches for the post-hoc correlation EMA")
    p.add_argument("--knn", type=int, default=32)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min_size", type=int, default=16)
    p.add_argument("--sample_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random_sets", type=int, default=5, help="random same-size neuron sets per assembly (0 to skip)")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt or os.path.join(args.out_dir, "ckpt.pt"), device)
    cfg = model.config
    bs = args.batch_size or ckpt["config"]["batch_size"]
    val = load_val(args.data_dir)
    val_cluster = {d: val[d][:len(val[d]) // 2] for d in DOMAINS}
    val_score = {d: val[d][len(val[d]) // 2:] for d in DOMAINS}

    final_path = os.path.join(args.out_dir, "assemblies_final.json")
    final = None
    if args.source != "posthoc" and os.path.exists(final_path):
        with open(final_path) as f:
            final = json.load(f)
        if any(lab is None for lab in final["layers"]):
            final = None    # the run ended before its first clustering
    if args.source == "final" and final is None:
        raise SystemExit(f"{final_path} has no assignments; use --source posthoc")
    source = "final" if final is not None else "posthoc"
    save_dir = args.save_dir or (args.out_dir if source == "final" else os.path.join(args.out_dir, "posthoc"))
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    if source == "final":
        labels = [np.asarray(lab) for lab in final["layers"]]
    else:
        print(f"[assemblies] post-hoc clustering on held-out data ({args.posthoc_steps} batches)", flush=True)
        labels = posthoc_labels(model, val_cluster, domain_mix(args.data_dir), args, device, bs)
        with open(os.path.join(save_dir, "assemblies_posthoc.json"), "w") as f:
            json.dump({"resolution": args.resolution, "knn": args.knn, "min_size": args.min_size,
                       "posthoc_steps": args.posthoc_steps, "layers": [[int(x) for x in lab] for lab in labels]}, f)
    health = [partition_health(lab, args.min_size) for lab in labels]
    degenerate = [h["degenerate"] for h in health]
    print(f"[assemblies] source={source}, assemblies per layer: {[h['n_assemblies'] for h in health]}, "
          f"degenerate layers: {[i for i, d in enumerate(degenerate) if d]}", flush=True)

    def losses():
        return {d: eval_loss(model, val_score[d], args.eval_iters, bs, cfg.block_size, device) for d in DOMAINS}

    base = losses()
    print("[assemblies] unablated: " + " | ".join(f"{d} {base[d]:.4f}" for d in DOMAINS), flush=True)

    # the mask must really change the output, and clearing it must restore the model
    x, y = get_batch(val_score[DOMAINS[0]], 2, cfg.block_size, device, np.random.default_rng(0))
    with torch.no_grad(), autocast_ctx(device):
        before = model(x, y)[0].float()
        first = labels[0][labels[0] != LEFTOVER][0]
        model.set_neuron_mask(0, torch.as_tensor(labels[0] != first, dtype=torch.float32))
        masked = model(x, y)[0].float()
        model.clear_ablation()
        after = model(x, y)[0].float()
    assert (masked - before).abs().max() > 1e-4, "ablating an assembly did not change the logits"
    assert torch.allclose(after, before, atol=1e-3), "clear_ablation() did not restore the model"

    rows, total = [], sum(h["n_assemblies"] for h in health)
    for layer, lab in enumerate(labels):
        for g in np.unique(lab[lab != LEFTOVER]):
            mask = torch.as_tensor(lab != g, dtype=torch.float32)
            model.set_neuron_mask(layer, mask)
            got = losses()
            dmg = [got[d] - base[d] for d in DOMAINS]
            rows.append({"layer": layer, "assembly": int(g), "size": int((lab == g).sum()), "name": f"{layer}.{g}",
                         "losses": got, "dmg": dmg, "S": specialization(dmg), "degenerate": degenerate[layer]})
            eta = (time.time() - t0) / len(rows) * (total - len(rows))
            print(f"[assemblies] {len(rows)}/{total} L{layer}.A{g} size {rows[-1]['size']}: "
                  + " ".join(f"{d} {x:+.4f}" for d, x in zip(DOMAINS, dmg)) + f" | S={rows[-1]['S']:.3f} | ETA {eta:.0f}s",
                  flush=True)

    rng = np.random.default_rng(args.seed)
    random_rows = []
    counted = [r for r in rows if not r["degenerate"]]
    if args.random_sets > 0:
        for i, r in enumerate(counted):
            pool = np.flatnonzero(labels[r["layer"]] != LEFTOVER)
            for k in range(args.random_sets):
                mask = np.ones(len(labels[r["layer"]]), dtype=np.float32)
                mask[rng.choice(pool, size=r["size"], replace=False)] = 0.0
                model.set_neuron_mask(r["layer"], torch.as_tensor(mask))
                got = losses()
                dmg = [got[d] - base[d] for d in DOMAINS]
                random_rows.append({"layer": r["layer"], "assembly": r["assembly"], "set": k, "size": r["size"], "dmg": dmg,
                                    "S": specialization(dmg)})
            print(f"[assemblies] random baseline {i + 1}/{len(counted)}, {time.time() - t0:.0f}s", flush=True)
    model.clear_ablation()

    with open(os.path.join(save_dir, "assembly_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "assembly", "size", *[f"loss_{d}" for d in DOMAINS], *[f"dmg_{d}" for d in DOMAINS],
                    "specialization", "degenerate_layer"])
        for r in rows:
            w.writerow([r["layer"], r["assembly"], r["size"], *[f"{r['losses'][d]:.5f}" for d in DOMAINS],
                        *[f"{x:.5f}" for x in r["dmg"]], "" if math.isnan(r["S"]) else f"{r['S']:.4f}",
                        int(r["degenerate"])])
    with open(os.path.join(save_dir, "assembly_random.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "assembly", "set", "size", *[f"dmg_{d}" for d in DOMAINS], "specialization"])
        for r in random_rows:
            w.writerow([r["layer"], r["assembly"], r["set"], r["size"], *[f"{x:.5f}" for x in r["dmg"]],
                        "" if math.isnan(r["S"]) else f"{r['S']:.4f}"])

    # headline numbers leave out degenerate layers
    S = np.array([r["S"] for r in counted], dtype=np.float64)
    valid = ~np.isnan(S)
    sizes = np.array([r["size"] for r in counted], dtype=np.float64)
    damage = np.clip(np.array([r["dmg"] for r in counted], dtype=np.float64).reshape(-1, len(DOMAINS)), 0.0, None).sum(axis=1)
    per_neuron = damage / np.maximum(sizes, 1.0)
    random_S, random_per_neuron, paired = float("nan"), float("nan"), []
    random_S_ci, paired_ci = [float("nan")] * 2, [float("nan")] * 2
    if random_rows:
        by_assembly = {}
        for r in random_rows:
            total_dmg = sum(max(x, 0.0) for x in r["dmg"])
            by_assembly.setdefault((r["layer"], r["assembly"]), []).append((r["S"], total_dmg / r["size"]))
        means = {key: (np.nanmean([v[0] for v in vals]) if any(not math.isnan(v[0]) for v in vals) else float("nan"),
                       float(np.mean([v[1] for v in vals]))) for key, vals in by_assembly.items()}
        rand_values = [means[(r["layer"], r["assembly"])][0] for r in counted]
        random_S = float(np.nanmean(rand_values)) if not all(math.isnan(v) for v in rand_values) else float("nan")
        random_S_ci = bootstrap_ci(rand_values, seed=args.seed + 1)
        random_per_neuron = float(np.mean([means[(r["layer"], r["assembly"])][1] for r in counted]))
        paired = [r["S"] - means[(r["layer"], r["assembly"])][0] for r in counted]
        paired_ci = bootstrap_ci(paired, seed=args.seed + 2)
    summary = {"source": source, "checkpoint_iter": ckpt["iter"], "routing_mode": ckpt["config"].get("routing_mode", "n/a"),
               "n_assemblies": len(rows), "n_counted": len(counted), "n_with_damage": int(valid.sum()),
               "degenerate_layers": [i for i, d in enumerate(degenerate) if d],
               "n_dead_per_layer": [h["n_dead"] for h in health],
               "mean_S": float(S[valid].mean()) if valid.any() else float("nan"),
               "mean_S_ci95": bootstrap_ci(S, seed=args.seed),
               "random_mean_S": random_S, "random_mean_S_ci95": random_S_ci,
               "real_minus_random_S_ci95": paired_ci, "random_sets": args.random_sets,
               "damage_per_neuron": float(per_neuron.mean()) if len(per_neuron) else float("nan"),
               "random_damage_per_neuron": random_per_neuron,
               "size_weighted_S": weighted(S[valid], sizes[valid]),
               "damage_weighted_S": weighted(S[valid], damage[valid]),
               "mean_damage": float(damage.mean()) if len(damage) else float("nan"),
               "max_damage": float(damage.max()) if len(damage) else float("nan"),
               "unablated": base, "eval_iters": args.eval_iters, "batch_size": bs,
               "per_layer": [{"layer": layer, "n_assemblies": sum(r["layer"] == layer for r in rows),
                              "degenerate": degenerate[layer],
                              "mean_S": float(np.nanmean([r["S"] for r in rows if r["layer"] == layer] or [float("nan")]))}
                             for layer in range(len(labels))]}
    if source == "posthoc":
        summary.update({"posthoc_steps": args.posthoc_steps, "resolution": args.resolution})
    with open(os.path.join(save_dir, "assembly_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    text = [f"source: {source} (mode of checkpoint: {summary['routing_mode']}, iter {ckpt['iter']})",
            f"assemblies: {len(rows)}, per layer {[s['n_assemblies'] for s in summary['per_layer']]}",
            f"degenerate layers, left out below: {summary['degenerate_layers']}; dead neurons per layer: "
            f"{summary['n_dead_per_layer']}",
            f"mean S: {summary['mean_S']:.4f}, bootstrap 95% CI [{summary['mean_S_ci95'][0]:.4f}, {summary['mean_S_ci95'][1]:.4f}]",
            f"random same-size sets, mean S: {random_S:.4f}, 95% CI [{random_S_ci[0]:.4f}, {random_S_ci[1]:.4f}]",
            f"real minus random S, 95% CI: [{paired_ci[0]:.4f}, {paired_ci[1]:.4f}]",
            f"damage per neuron: real {summary['damage_per_neuron']:.6f}, random {random_per_neuron:.6f} nats",
            f"size-weighted mean S: {summary['size_weighted_S']:.4f}",
            f"damage-weighted mean S: {summary['damage_weighted_S']:.4f}",
            f"mean damage per assembly: {summary['mean_damage']:.4f} nats, max {summary['max_damage']:.4f}",
            "unablated val loss (second half of val): " + ", ".join(f"{d}={base[d]:.4f}" for d in DOMAINS),
            "per-layer mean S: " + ", ".join(f"L{s['layer']}={s['mean_S']:.3f}" for s in summary["per_layer"]),
            f"time: {time.time() - t0:.0f}s"]
    with open(os.path.join(save_dir, "assembly_summary.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print("[assemblies] " + "\n[assemblies] ".join(text))

    names = [f"{r['name']} (n={r['size']})" for r in rows]
    plot_heatmap(names, np.array([r["dmg"] for r in rows]), os.path.join(save_dir, "assembly_heatmap.png"), base,
                 title=f"Loss increase when one assembly is zeroed ({source})", ylabel="ablated assembly (layer.id, size)")
    plot_sizes(rows, len(labels), os.path.join(save_dir, "assembly_sizes.png"))
    print(f"[assemblies] wrote results to {save_dir}")
    print("[assemblies] note: these CIs resample assemblies within one model; they do not cover variation between seeds")


if __name__ == "__main__":
    main()
