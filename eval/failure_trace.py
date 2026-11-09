"""Failure trace: which groups are active when the model gets held-out arithmetic wrong?

1. --n_problems held-out problems (deconf.heldout_problems; never in training with --holdout_arith), greedy answers,
   marked right or wrong.
2. For each gated layer and group: its selection rate on the answer positions (the ":" after "A" and every answer token
   but the last) of wrong vs right problems. Groups are ranked by wrong-minus-right rate.
   consistency = share of the failures in which the group is active (selected on at least half of the answer positions).
   If no answer is right (an untrained model), groups are ranked by their rate on wrong answers alone, and the summary
   says so.
3. The top --top_n groups are ablated (each alone, then all together): arithmetic accuracy vs story/math/code loss change.

Works on dense (post-hoc groups, inference-only gating), gated-fixed and gated-evolved checkpoints.
Outputs (in out_dir): failure_trace.csv, failure_trace_ablation.csv, failure_trace_active.npz (per-problem selection
rates), failure_trace_summary.json, failure_trace_summary.txt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deconf import heldout_problems  # noqa: E402
from gating_common import (DOMAINS, add_args, eval_sets, load, masked_losses, masks_for, solve,  # noqa: E402
                           write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p, windows=48)
    p.add_argument("--n_problems", type=int, default=2000)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--problem_seed", type=int, default=12345)
    args = p.parse_args()
    t0 = time.time()
    model, ckpt, router, source, device = load(args)
    problems = heldout_problems(args.n_problems, args.problem_seed)
    res = solve(model, problems, device, router=router)
    correct, active = res["correct"], res["active"]
    wrong = ~correct
    acc = float(correct.mean())
    print(f"[trace] source={source}: accuracy {acc:.3f} on {len(problems)} held-out problems ({int(wrong.sum())} wrong)",
          flush=True)
    np.savez_compressed(os.path.join(args.out_dir, "failure_trace_active.npz"), active=active, correct=correct,
                        layers=np.array(router.layers))

    rows = []
    for li, layer in enumerate(router.layers):
        for g in range(router.G):
            a = active[:, li, g]
            rate_w = float(a[wrong].mean()) if wrong.any() else float("nan")
            rate_r = float(a[correct].mean()) if correct.any() else float("nan")
            rows.append({"layer": layer, "group": g, "size": int((router.labels[li] == g).sum()), "rate_wrong": rate_w,
                         "rate_right": rate_r, "diff": rate_w - rate_r if correct.any() else rate_w,
                         "consistency": float((a[wrong] >= 0.5).mean()) if wrong.any() else float("nan"),
                         "consistency_right": float((a[correct] >= 0.5).mean()) if correct.any() else float("nan")})
    ranked = sorted([r for r in rows if np.isfinite(r["diff"])], key=lambda r: -r["diff"])
    top = ranked[:args.top_n]
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    with open(os.path.join(args.out_dir, "failure_trace.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "group", "size", "rank", "rate_wrong", "rate_right", "diff",
                                          "consistency", "consistency_right"], restval="")
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    sets = {k: v for k, v in eval_sets(args.data_dir, model.config.block_size, args.windows, args.seed).items()
            if k.startswith("domain_")}
    base = masked_losses(model, sets, device, args.batch_size)
    ablations = []
    if top:
        plans = [([r], f"L{r['layer']}.G{r['group']}") for r in top] + ([(top, "top together")] if len(top) > 1 else [])
        for chosen, name in plans:
            deleted = [(r["layer"], np.flatnonzero(router.labels[router.pos[r["layer"]]] == r["group"])) for r in chosen]
            model.set_neuron_masks(masks_for(router, deleted))
            a = float(solve(model, problems, device)["correct"].mean())
            losses = masked_losses(model, sets, device, args.batch_size)
            model.clear_ablation()
            ablations.append({"ablated": name, "accuracy": a, "accuracy_change": a - acc,
                              **{f"d_{d}": losses[f"domain_{d}"] - base[f"domain_{d}"] for d in DOMAINS}})
            print(f"[trace] ablate {name}: accuracy {a:.3f} ({a - acc:+.3f}) | " + " | ".join(
                f"{d} {ablations[-1]['d_' + d]:+.4f}" for d in DOMAINS), flush=True)
    with open(os.path.join(args.out_dir, "failure_trace_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ablated", "accuracy", "accuracy_change", *[f"d_{d}" for d in DOMAINS]])
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in ablations)

    joint = ablations[-1] if ablations else {}
    basis = "rate on wrong minus rate on right" if correct.any() else "rate on wrong (no right answers)"
    summary = {"source": source, "checkpoint_iter": ckpt.get("iter"), "n_problems": len(problems), "accuracy": acc,
               "ranked_by": basis,
               "n_wrong": int(wrong.sum()), "top_groups": [f"{r['layer']}.{r['group']}" for r in top],
               "top_diff": [r["diff"] for r in top], "top_consistency": [r["consistency"] for r in top],
               "top_consistency_right": [r["consistency_right"] for r in top],
               "mean_top_consistency": float(np.mean([r["consistency"] for r in top])) if top else float("nan"),
               "ablation": ablations, "joint_accuracy_change": joint.get("accuracy_change", float("nan")),
               **{f"joint_d_{d}": joint.get(f"d_{d}", float("nan")) for d in DOMAINS},
               "unablated_loss": base, "examples_wrong": [(problems[i][0], problems[i][1], res["preds"][i])
                                                          for i in np.flatnonzero(wrong)[:10]],
               "time_s": round(time.time() - t0)}
    write_json(os.path.join(args.out_dir, "failure_trace_summary.json"), summary)
    lines = [f"source: {source} (checkpoint iter {ckpt.get('iter')})",
             f"held-out arithmetic accuracy: {acc:.3f} ({len(problems)} problems, {int(wrong.sum())} wrong)",
             f"top groups by {basis}; consistency = share of failures where active:"]
    lines += [f"  L{r['layer']}.G{r['group']} size {r['size']}: wrong {r['rate_wrong']:.3f} right {r['rate_right']:.3f} "
              f"diff {r['diff']:+.3f} consistency {r['consistency']:.3f} (on right answers {r['consistency_right']:.3f})"
              for r in top]
    lines += [f"ablate {a['ablated']}: accuracy {a['accuracy']:.3f} ({a['accuracy_change']:+.3f}), loss change "
              + ", ".join(f"{d} {a['d_' + d]:+.4f}" for d in DOMAINS) for a in ablations]
    lines.append(f"time: {summary['time_s']}s")
    with open(os.path.join(args.out_dir, "failure_trace_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
