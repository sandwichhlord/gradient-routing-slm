"""Number of assemblies and clustering stability over training, per layer: assemblies.csv -> assembly_health.png"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"))
    args = p.parse_args()
    path = os.path.join(args.out_dir, "assemblies.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "clustering"]
    if not rows:
        raise SystemExit(f"{path} has no clustering rows")
    layers = sorted({int(r["layer"]) for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=150, sharex=True)
    for layer in layers:
        col = COLORS[layer % len(COLORS)]
        lr = [r for r in rows if int(r["layer"]) == layer]
        xs = [int(r["iter"]) for r in lr]
        axes[0].plot(xs, [int(r["n_assemblies"]) for r in lr], "-o", ms=4, lw=2, color=col, label=f"layer {layer}")
        bad = [r for r in lr if r["degenerate"] == "1"]
        axes[0].scatter([int(r["iter"]) for r in bad], [int(r["n_assemblies"]) for r in bad], marker="x", s=70, color=INK,
                        zorder=3, label="degenerate" if layer == layers[0] and bad else None)
        ari = [r for r in lr if r["stability_ari"] != ""]
        axes[1].plot([int(r["iter"]) for r in ari], [float(r["stability_ari"]) for r in ari], "-o", ms=4, lw=2, color=col)
    axes[0].set_ylabel("assemblies", color=INK)
    axes[0].set_title("Number of assemblies at each clustering", color=INK, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, ncol=4)
    axes[1].set_ylabel("adjusted Rand index", color=INK)
    axes[1].set_xlabel("training iteration", color=INK)
    axes[1].set_title("Stability against the previous clustering (1 = identical, 0 = unrelated)", color=INK, loc="left", fontsize=10)
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "assembly_health.png")
    fig.savefig(out)
    print(f"[assembly_health_plot] wrote {out}")


if __name__ == "__main__":
    main()
