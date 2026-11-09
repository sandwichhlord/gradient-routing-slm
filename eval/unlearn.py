"""Unlearning by deletion: remove the groups traced to arithmetic and measure what is lost.

Trace: for each gated layer and group, its selection rate on the answer positions of --n_problems held-out arithmetic
problems minus its selection rate on story and code tokens (second val half). The --top_n groups with the largest
difference are deleted (their neurons zeroed, all at once). Reported before and after: held-out arithmetic accuracy and
story / math / code loss. Baseline: --random_sets deletions of random neuron sets with the same sizes in the same layers.

Works on dense (post-hoc groups, inference-only gating), gated-fixed and gated-evolved checkpoints. In a gated model a
deleted group has zero activity, so it stops winning the k-WTA and other groups take its place.
Outputs (in out_dir): unlearn_trace.csv, unlearn_summary.json, unlearn_summary.txt
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
from gating_common import (DOMAINS, add_args, eval_sets, load, masked_losses, masks_for, selection_rate,  # noqa: E402
                           solve, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p, windows=48)
    p.add_argument("--n_problems", type=int, default=2000)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--random_sets", type=int, default=3)
    p.add_argument("--problem_seed", type=int, default=12345)
    args = p.parse_args()
    t0 = time.time()
    model, ckpt, router, source, device = load(args)
    problems = heldout_problems(args.n_problems, args.problem_seed)
    sets = {k: v for k, v in eval_sets(args.data_dir, model.config.block_size, args.windows, args.seed).items()
            if k.startswith("domain_")}

    before = solve(model, problems, device, router=router)
    arith_rate = before["active"].mean(axis=0)                                   # (n_layers, G)
    other_rate = selection_rate(model, router, sets, ["domain_story", "domain_code"], device, args.batch_size)
    score = arith_rate - other_rate
    rows = [{"layer": layer, "group": g, "size": int((router.labels[li] == g).sum()), "arith_rate": float(arith_rate[li, g]),
             "story_code_rate": float(other_rate[li, g]), "score": float(score[li, g])}
            for li, layer in enumerate(router.layers) for g in range(router.G)]
    top = sorted(rows, key=lambda r: -r["score"])[:args.top_n]
    with open(os.path.join(args.out_dir, "unlearn_trace.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    base_loss = masked_losses(model, sets, device, args.batch_size)
    base_acc = float(before["correct"].mean())

    def after(deleted):
        model.set_neuron_masks(masks_for(router, deleted))
        acc = float(solve(model, problems, device)["correct"].mean())
        loss = masked_losses(model, sets, device, args.batch_size)
        model.clear_ablation()
        return {"accuracy": acc, **{d: loss[f"domain_{d}"] for d in DOMAINS}}

    deleted = [(r["layer"], np.flatnonzero(router.labels[router.pos[r["layer"]]] == r["group"])) for r in top]
    traced = after(deleted)
    rng = np.random.default_rng(args.seed)
    randoms = [after([(layer, rng.choice(router.hidden, size=len(idx), replace=False)) for layer, idx in deleted])
               for _ in range(args.random_sets)]
    rand = {k: float(np.mean([r[k] for r in randoms])) for k in traced} if randoms else {}
    base = {"accuracy": base_acc, **{d: base_loss[f"domain_{d}"] for d in DOMAINS}}

    summary = {"source": source, "checkpoint_iter": ckpt.get("iter"), "n_problems": len(problems),
               "deleted_groups": [f"{r['layer']}.{r['group']}" for r in top], "deleted_sizes": [r["size"] for r in top],
               "deleted_scores": [r["score"] for r in top], "before": base, "after": traced, "random_after": rand,
               "random_sets": args.random_sets, "accuracy_change": traced["accuracy"] - base_acc,
               "random_accuracy_change": rand.get("accuracy", float("nan")) - base_acc,
               **{f"d_{d}": traced[d] - base[d] for d in DOMAINS},
               **{f"random_d_{d}": rand.get(d, float("nan")) - base[d] for d in DOMAINS},
               "time_s": round(time.time() - t0)}
    write_json(os.path.join(args.out_dir, "unlearn_summary.json"), summary)
    lines = [f"source: {source} (checkpoint iter {ckpt.get('iter')})",
             f"deleted (most arithmetic-selective): {summary['deleted_groups']}, sizes {summary['deleted_sizes']}",
             f"{'':22s} {'before':>8s} {'after':>8s} {'random':>8s}"]
    lines += [f"{k:22s} {base[k]:8.4f} {traced[k]:8.4f} {rand.get(k, float('nan')):8.4f}"
              for k in ["accuracy", *DOMAINS]]
    lines.append(f"(accuracy = held-out arithmetic, {len(problems)} problems; story/math/code = val loss; random = "
                 f"mean of {args.random_sets} same-size random deletions)")
    lines.append(f"time: {summary['time_s']}s")
    with open(os.path.join(args.out_dir, "unlearn_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
