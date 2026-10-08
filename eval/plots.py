"""Plot train loss + per-domain val loss from out/metrics.csv -> out/loss_curves.png."""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DOMAINS, ROOT  # noqa: E402

# categorical slots in fixed order (train = neutral ink, domains = slots 1-3)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COLORS = {"train": "#52514e", "story": "#2a78d6", "math": "#eb6834", "code": "#1baf7a"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"))
    args = p.parse_args()
    path = os.path.join(args.out_dir, "metrics.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    # a resumed run can repeat iters; keep the last value for each
    by_iter = {int(r["iter"]): r for r in rows}
    its = sorted(by_iter)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    series = [("train", "train_loss", "train (running mean)", "--")] + [(d, f"val_{d}", f"val {d}", "-") for d in DOMAINS]
    for key, col, label, ls in series:
        ys = [float(by_iter[i][col]) for i in its]
        ax.plot(its, ys, ls, color=COLORS[key], lw=2, label=f"{label}: final {ys[-1]:.3f}",
                marker="o" if len(its) <= 30 else None, ms=3)
    ax.set_xlabel("training iteration", color=INK)
    ax.set_ylabel("cross-entropy loss (nats / token)", color=INK)
    ax.set_title("Train and per-domain validation loss", color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "loss_curves.png")
    fig.savefig(out)
    print(f"[plots] wrote {out}")


if __name__ == "__main__":
    main()
