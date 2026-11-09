"""Compare every run in results/<run>/ -> results/comparison.csv and results/comparison.png

A run is any folder holding metrics.csv. Folders named <group>_s<seed> or <group>_<n> are grouped, and a group's rows
report mean and std across its runs. Mean S is the mean of the `specialization` column of ablation.csv, the number
eval/ablate.py reports. tokens_per_sec is the median over the run; old runs did not log it, so it is taken from the
median ms/it in run.log. The asm_* columns come from eval/ablate_assemblies.py (assembly_summary.json in the run folder)
and the posthoc_* columns from its post-hoc mode (posthoc/assembly_summary.json).

Gated-mode table (results/comparison_gated.csv): one row per group of runs that have the v3 evaluations (dense,
gated_fixed, gated_evolved, ...): final val loss, damage-table classes and mean damage per category
(damage_summary.json), held-out arithmetic accuracy and failure-trace numbers (failure_trace_summary.json), unlearning
before/after (unlearn_summary.json) and context-sensitivity effects at route_smooth 0 and 0.9 (context_sensitivity.json).
"""
import argparse
import csv
import json
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
ASM_FIELDS = ["mean_S", "random_mean_S", "size_weighted_S", "damage_weighted_S", "n_assemblies"]
METRICS = [*[f"val_{d}" for d in DOMAINS], "mean_S", "mean_damage", "tokens_per_sec",
           *[f"asm_{f}" for f in ASM_FIELDS], *[f"posthoc_{f}" for f in ASM_FIELDS]]


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
    for prefix, path in (("asm", "assembly_summary.json"), ("posthoc", os.path.join("posthoc", "assembly_summary.json"))):
        summary_path = os.path.join(folder, path)
        found = {}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                found = json.load(f)
        for field in ASM_FIELDS:
            out[f"{prefix}_{field}"] = found.get(field)
    abl = os.path.join(folder, "ablation.csv")
    if os.path.exists(abl):
        with open(abl, newline="") as f:
            a = list(csv.DictReader(f))
        s = [float(r["specialization"]) for r in a if r["specialization"] != ""]
        out["mean_S"] = sum(s) / len(s) if s else None
        out["mean_damage"] = sum(sum(max(float(r[f"dmg_{d}"]), 0.0) for d in DOMAINS) for r in a) / len(a) if a else None
    return out


GATED = [
    ("val_story", None), ("val_math", None), ("val_code", None),
    ("n_lexical", ("damage_summary.json", lambda j: j["classes"]["lexical"])),
    ("n_functional", ("damage_summary.json", lambda j: j["classes"]["functional"])),
    ("dmg_arith_digit", ("damage_summary.json", lambda j: j["mean_damage"].get("tag_digit_in_arith_answer"))),
    ("dmg_story_digit", ("damage_summary.json", lambda j: j["mean_damage"].get("tag_digit_in_story"))),
    ("dmg_wp_word", ("damage_summary.json", lambda j: j["mean_damage"].get("tag_number_word_in_word_problem"))),
    ("arith_acc", ("failure_trace_summary.json", lambda j: j["accuracy"])),
    ("trace_consistency", ("failure_trace_summary.json", lambda j: j["mean_top_consistency"])),
    ("trace_ablate_d_acc", ("failure_trace_summary.json", lambda j: j["joint_accuracy_change"])),
    ("trace_ablate_d_story", ("failure_trace_summary.json", lambda j: j["joint_d_story"])),
    ("trace_ablate_d_code", ("failure_trace_summary.json", lambda j: j["joint_d_code"])),
    ("unlearn_d_acc", ("unlearn_summary.json", lambda j: j["accuracy_change"])),
    ("unlearn_rand_d_acc", ("unlearn_summary.json", lambda j: j["random_accuracy_change"])),
    ("unlearn_d_story", ("unlearn_summary.json", lambda j: j["d_story"])),
    ("unlearn_d_code", ("unlearn_summary.json", lambda j: j["d_code"])),
    ("ctx_digits_s0", ("context_sensitivity.json", lambda j: j["mean_effect"]["0.0"]["digits"])),
    ("ctx_digits_s0.9", ("context_sensitivity.json", lambda j: j["mean_effect"]["0.9"]["digits"])),
    ("ctx_words_s0", ("context_sensitivity.json", lambda j: j["mean_effect"]["0.0"]["number_words"])),
    ("ctx_words_s0.9", ("context_sensitivity.json", lambda j: j["mean_effect"]["0.9"]["number_words"])),
]
GATED_FILES = {"damage_summary.json", "failure_trace_summary.json", "unlearn_summary.json", "context_sensitivity.json"}


def read_gated(folder, run):
    """The v3 evaluation numbers of one run, or None if it has none."""
    if not any(os.path.exists(os.path.join(folder, f)) for f in GATED_FILES):
        return None
    cache, out = {}, {}
    for name, src in GATED:
        if src is None:
            out[name] = run[name]
            continue
        path = os.path.join(folder, src[0])
        if path not in cache:
            cache[path] = json.load(open(path)) if os.path.exists(path) else None
        try:
            v = src[1](cache[path]) if cache[path] is not None else None
        except (KeyError, TypeError):
            v = None
        out[name] = None if v is None or (isinstance(v, float) and v != v) else float(v)
    return out


def write_gated_table(runs, folders, out_dir):
    rows = {name: read_gated(folders[name], r) for name, r in runs.items()}
    rows = {k: v for k, v in rows.items() if v is not None}
    if not rows:
        return
    by_group = {}
    for name, r in rows.items():
        by_group.setdefault(group_of(name), []).append(r)
    order = ["dense", "gated_fixed", "gated_evolved"]
    groups = sorted(by_group, key=lambda g: (order.index(g) if g in order else len(order), g))
    names = [m for m, _ in GATED]
    with open(os.path.join(out_dir, "comparison_gated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_runs", *[c for m in names for c in (m, f"{m}_std")]])
        for g in groups:
            row = [g, len(by_group[g])]
            for m in names:
                vals = [r[m] for r in by_group[g] if r[m] is not None]
                row += [fmt(statistics.mean(vals)) if vals else "", fmt(statistics.stdev(vals)) if len(vals) > 1 else ""]
            w.writerow(row)
    print()
    print(f"{'metric':22s}" + "".join(f"{g:>18s}" for g in groups))
    for m in names:
        cells = []
        for g in groups:
            vals = [r[m] for r in by_group[g] if r[m] is not None]
            cells.append(f"{(fmt(statistics.mean(vals), 3) + (f' (n={len(vals)})' if len(vals) > 1 else '')) if vals else '-':>18s}")
        print(f"{m:22s}" + "".join(cells))
    print(f"[compare] wrote comparison_gated.csv to {out_dir}")


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

    runs, folders = {}, {}
    for name in sorted(os.listdir(args.results_dir)):
        folder = os.path.join(args.results_dir, name)
        if os.path.isfile(os.path.join(folder, "metrics.csv")):
            r = read_run(folder)
            if r:
                runs[name], folders[name] = r, folder
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
    write_gated_table(runs, folders, out_dir)


if __name__ == "__main__":
    main()
