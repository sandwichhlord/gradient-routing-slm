"""Which blocks does each domain's tokens select? routing_stats.csv -> routing_by_domain.png

One heatmap per layer: blocks across, domains down, colour = fraction of that domain's tokens that selected the block
during the logging window ending at --iter (default: the last one). Each row sums to k in token mode.
"""
import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DOMAINS, ROOT  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"))
    p.add_argument("--iter", type=int, default=None, help="logging window to plot; default is the last one")
    args = p.parse_args()

    path = os.path.join(args.out_dir, "routing_stats.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "frac_story" not in rows[0]:
        raise SystemExit(f"{path} has no frac_story column: it was written by old (v0) code, which did not log "
                         f"per-domain selection, so there is nothing to plot")
    it = args.iter if args.iter is not None else max(int(r["iter"]) for r in rows)
    rows = [r for r in rows if int(r["iter"]) == it]
    if not rows:
        raise SystemExit(f"{path} has no rows for iter {it}")
    n_layer = max(int(r["layer"]) for r in rows) + 1
    n_blocks = max(int(r["block"]) for r in rows) + 1
    grid = np.full((n_layer, len(DOMAINS), n_blocks), np.nan)
    for r in rows:
        for d, name in enumerate(DOMAINS):
            if r[f"frac_{name}"] != "":
                grid[int(r["layer"]), d, int(r["block"])] = float(r[f"frac_{name}"])

    cols = min(n_layer, 2)
    nrows = math.ceil(n_layer / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5.2 * cols + 1.0, 1.75 * nrows + 1.0), dpi=150, squeeze=False,
                             constrained_layout=True)
    vmax = max(float(np.nanmax(grid)), 1e-6)
    im = None
    for layer, ax in enumerate(axes.flat):
        if layer >= n_layer:
            ax.set_visible(False)
            continue
        im = ax.imshow(grid[layer], cmap=CMAP, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(f"layer {layer}", color=INK, fontsize=9, loc="left")
        ax.set_xticks(range(n_blocks), range(n_blocks), fontsize=7, color=MUTED)
        ax.set_yticks(range(len(DOMAINS)), DOMAINS, fontsize=8, color=INK)
        ax.set_xticks(np.arange(-0.5, n_blocks, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
        ax.grid(which="minor", color="#fcfcfb", lw=1.0)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for d in range(len(DOMAINS)):
            for b in range(n_blocks):
                v = grid[layer, d, b]
                if not np.isnan(v):
                    ax.text(b, d, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.55 * vmax else INK)
    fig.suptitle(f"Fraction of each domain's tokens selecting each block (window ending at iter {it})", color=INK,
                 fontsize=10, x=0.02, ha="left")
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("block", color=MUTED, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    out = os.path.join(args.out_dir, "routing_by_domain.png")
    fig.savefig(out)
    print(f"[routing_plot] wrote {out}")


if __name__ == "__main__":
    main()
