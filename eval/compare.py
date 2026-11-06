"""Compare every run in results/<run>/ -> results/comparison.csv and results/comparison.png

A run is any folder holding metrics.csv. Folders named <group>_s<seed> or <group>_<n> are grouped, and a group's rows
report mean and std across its runs. Mean S is the mean of the `specialization` column of ablation.csv, the number
eval/ablate.py reports. tokens_per_sec is the median over the run; old runs did not log it, so it is taken from the
median ms/it in run.log.
"""
import argparse
import csv
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DOMAINS, ROOT  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COLORS = {"story": "#2a78d6", "math": "#eb6834", "code": "#1baf7a"}
NEUTRAL = "#52514e"
ORDER = ["dense_v0", "v0_batch_routing", "dense", "batch", "token", "label"]
METRICS = [*[f"val_{d}" for d in DOMAINS], "mean_S", "mean_damage", "tokens_per_sec"]


def group_of(run):
    m = re.fullmatch(r"(.+)_s\d+", run) or re.fullmatch(r"(.+)_\d+", run)
    return m.group(1) if m else run


def read_tokens_per_sec(folder, rows):
    vals = [float(r["tokens_per_sec"]) for r in rows if r.get("tokens_per_sec")]
    if vals:
        return statistics.median(vals)
    log, summary = os.path.join(folder, "run.log"), os.path.join(folder, "ablation_summary.txt")
    if not (os.path.exists(log) and os.path.exists(summary)):
        return None
    with open(summary) as f:
        shape = re.search(r"x (\d+) x (\d+) tokens", f.read())
    with open(log, encoding="utf-8", errors="replace") as f:
        ms = [int(x) for x in re.findall(r"(\d+)ms/it", f.read())]
    if not shape or not ms:
        return None
    return int(shape.group(1)) * int(shape.group(2)) / (statistics.median(ms) / 1000)


def read_run(folder):
    with open(os.path.join(folder, "metrics.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {f"val_{d}": float(rows[-1][f"val_{d}"]) for d in DOMAINS}
    out["final_iter"] = int(rows[-1]["iter"])
    out["tokens_per_sec"] = read_tokens_per_sec(folder, rows)
    out["mean_S"] = out["mean_damage"] = None
    abl = os.path.join(folder, "ablation.csv")
    if os.path.exists(abl):
        with open(abl, newline="") as f:
            a = list(csv.DictReader(f))
        s = [float(r["specialization"]) for r in a if r["specialization"] != ""]
        out["mean_S"] = sum(s) / len(s) if s else None
        out["mean_damage"] = sum(sum(max(float(r[f"dmg_{d}"]), 0.0) for d in DOMAINS) for r in a) / len(a) if a else None
    return out


def summarise(runs):
    """{metric: (mean, std or None, n)} for one group."""
    out = {}
    for m in METRICS:
        vals = [r[m] for r in runs if r[m] is not None]
        if vals:
            out[m] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else None, len(vals))
        else:
            out[m] = (None, None, 0)
    return out


def fmt(v, digits=4):
    return "" if v is None else f"{v:.{digits}f}"


def plot(groups, summary, path):
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(max(11, 2.0 * len(groups) + 6), 4.2), dpi=150,
                             gridspec_kw={"width_ratios": [2, 1, 1]})

    def err(g, m):
        s = summary[g][m][1]
        return 0.0 if s is None else s

    def val(g, m):
        v = summary[g][m][0]
        return np.nan if v is None else v

    width = 0.8 / len(DOMAINS)
    for j, d in enumerate(DOMAINS):
        m = f"val_{d}"
        axes[0].bar(x + (j - 1) * width, [val(g, m) for g in groups], width * 0.9, color=COLORS[d], label=d,
                    yerr=[err(g, m) for g in groups], ecolor=INK, capsize=2)
    axes[0].set_title("Final validation loss (lower is better)", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("nats / token", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    axes[0].margins(y=0.2)
    for ax, m, title, ylabel in [(axes[1], "mean_S", "Mean specialization S", "S (0 to 1)"),
                                 (axes[2], "tokens_per_sec", "Training speed", "tokens / s")]:
        ax.bar(x, [val(g, m) for g in groups], 0.6, color=NEUTRAL, yerr=[err(g, m) for g in groups], ecolor=INK, capsize=2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [f"{g}\n(n={max(summary[g][m][2] for m in METRICS)})" for g in groups], fontsize=7, color=INK)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--out_dir", default=None, help="where to write comparison.csv/png; default: results_dir")
    args = p.parse_args()
    out_dir = args.out_dir or args.results_dir

    runs = {}
    for name in sorted(os.listdir(args.results_dir)):
        folder = os.path.join(args.results_dir, name)
        if os.path.isfile(os.path.join(folder, "metrics.csv")):
            r = read_run(folder)
            if r:
                runs[name] = r
    if not runs:
        raise SystemExit(f"no runs (folders with metrics.csv) in {args.results_dir}")

    by_group = {}
    for name, r in runs.items():
        by_group.setdefault(group_of(name), []).append(r)
    groups = sorted(by_group, key=lambda g: (ORDER.index(g) if g in ORDER else len(ORDER), g))
    summary = {g: summarise(by_group[g]) for g in groups}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "n_runs", *[c for m in METRICS for c in (m, f"{m}_std")]])
        for g in groups:
            row = ["group", g, len(by_group[g])]
            for m in METRICS:
                digits = 0 if m == "tokens_per_sec" else 4
                row += [fmt(summary[g][m][0], digits), fmt(summary[g][m][1], digits)]
            w.writerow(row)
        for name, r in runs.items():
            row = ["run", name, 1]
            for m in METRICS:
                row += [fmt(r[m], 0 if m == "tokens_per_sec" else 4), ""]
            w.writerow(row)
    plot(groups, summary, os.path.join(out_dir, "comparison.png"))

    print(f"{'group':20s} {'n':>2s} " + " ".join(f"{m:>16s}" for m in METRICS))
    for g in groups:
        cells = []
        for m in METRICS:
            mean, std, _ = summary[g][m]
            digits = 0 if m == "tokens_per_sec" else 3
            cells.append(f"{fmt(mean, digits) + (' +- ' + fmt(std, digits) if std is not None else ''):>16s}")
        print(f"{g:20s} {len(by_group[g]):2d} " + " ".join(cells))
    print(f"[compare] wrote comparison.csv and comparison.png to {out_dir}")


if __name__ == "__main__":
    main()
