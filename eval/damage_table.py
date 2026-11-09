"""Damage table: ablate each group of each gated layer and report the loss increase per token category.

Works on dense (post-hoc groups), gated-fixed and gated-evolved checkpoints (see eval/gating_common.py). Categories
come from data/prepare.py's tags (deconf.TAGS); each category is scored on windows of the second val half that contain
it, counting only targets of that category. Domain losses use random windows.

Baseline: --random_sets size-matched random neuron sets per group, from the same layer. A group "damages" a category
when its loss increase there is above BOTH an absolute floor (--min_damage, default 0.01 nats) and the 95th percentile
of the random sets' increases for that category in the same layer. Classes:
  lexical     damages digit_in_arith_answer AND digit_in_story
  functional  damages digit_in_arith_answer AND number_word_in_word_problem, NOT digit_in_story
  other       everything else
The floor and the percentile are this script's choices, not part of the spec.

Outputs (in out_dir): damage_table.csv, damage_random.csv, damage_summary.json, damage_summary.txt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gating_common import (DOMAINS, TAGS, add_args, eval_sets, group_list, load, masked_losses, masks_for,  # noqa: E402
                           write_json)

ARITH, STORY, WORDS = "tag_digit_in_arith_answer", "tag_digit_in_story", "tag_number_word_in_word_problem"


def classify(damaged):
    if damaged.get(ARITH) and damaged.get(STORY):
        return "lexical"
    if damaged.get(ARITH) and damaged.get(WORDS) and not damaged.get(STORY):
        return "functional"
    return "other"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--random_sets", type=int, default=3, help="size-matched random sets per group")
    p.add_argument("--min_damage", type=float, default=0.01)
    args = p.parse_args()
    t0 = time.time()
    model, ckpt, router, source, device = load(args)
    sets = eval_sets(args.data_dir, model.config.block_size, args.windows, args.seed)
    names = list(sets)
    base = masked_losses(model, sets, device, args.batch_size)
    print(f"[damage] source={source}, sets: " + ", ".join(f"{k} {int(v[2].sum())} targets" for k, v in sets.items()),
          flush=True)
    print("[damage] unablated: " + " | ".join(f"{k} {v:.3f}" for k, v in base.items()), flush=True)

    def damage(deleted):
        model.set_neuron_masks(masks_for(router, deleted))
        got = masked_losses(model, sets, device, args.batch_size)
        model.clear_ablation()
        return {k: got[k] - base[k] for k in names}

    groups = group_list(router)
    rows, rand_rows = [], []
    rng = np.random.default_rng(args.seed)
    for n, (layer, g, idx) in enumerate(groups):
        rows.append({"layer": layer, "group": g, "size": len(idx), **damage([(layer, idx)])})
        for r in range(args.random_sets):
            ridx = rng.choice(router.hidden, size=len(idx), replace=False)
            rand_rows.append({"layer": layer, "group": g, "set": r, "size": len(idx), **damage([(layer, ridx)])})
        eta = (time.time() - t0) / (n + 1) * (len(groups) - n - 1)
        print(f"[damage] {n + 1}/{len(groups)} L{layer}.G{g} size {len(idx)}: "
              + " ".join(f"{k.split('_', 1)[1][:14]} {rows[-1][k]:+.3f}" for k in names if k.startswith("tag_"))
              + f" | ETA {eta:.0f}s", flush=True)

    thresholds = {}
    for layer in router.layers:
        rr = [r for r in rand_rows if r["layer"] == layer]
        thresholds[layer] = {k: max(args.min_damage, float(np.percentile([r[k] for r in rr], 95)) if rr else args.min_damage)
                             for k in names}
    for r in rows:
        damaged = {k: r[k] > thresholds[r["layer"]][k] for k in names}
        r["class"] = classify(damaged)
        r["damaged"] = ";".join(k.split("_", 1)[1] for k in names if damaged[k])
        rr = [x for x in rand_rows if x["layer"] == r["layer"] and x["group"] == r["group"]]
        for k in (ARITH, STORY, WORDS):
            if k in names:
                r[f"excess_{k[4:]}"] = r[k] - (float(np.mean([x[k] for x in rr])) if rr else 0.0)

    fields = ["layer", "group", "size", "class", "damaged", *names, *[f"excess_{k[4:]}" for k in (ARITH, STORY, WORDS)
                                                                       if k in names]]
    with open(os.path.join(args.out_dir, "damage_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    with open(os.path.join(args.out_dir, "damage_random.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "group", "set", "size", *names])
        w.writeheader()
        w.writerows({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rand_rows)

    counts = {c: sum(r["class"] == c for r in rows) for c in ("lexical", "functional", "other")}
    mean = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs and k in names else float("nan")  # noqa: E731
    summary = {"source": source, "checkpoint_iter": ckpt.get("iter"), "n_groups": len(rows), "classes": counts,
               "random_sets": args.random_sets, "min_damage": args.min_damage,
               "unablated": base, "targets": {k: int(v[2].sum()) for k, v in sets.items()},
               "mean_damage": {k: mean(rows, k) for k in names}, "mean_random_damage": {k: mean(rand_rows, k) for k in names},
               "max_damage": {k: float(max(r[k] for r in rows)) for k in names},
               "functional_groups": [f"{r['layer']}.{r['group']}" for r in rows if r["class"] == "functional"],
               "lexical_groups": [f"{r['layer']}.{r['group']}" for r in rows if r["class"] == "lexical"],
               "thresholds": {str(k): v for k, v in thresholds.items()}, "time_s": round(time.time() - t0)}
    write_json(os.path.join(args.out_dir, "damage_summary.json"), summary)
    lines = [f"source: {source} (checkpoint iter {ckpt.get('iter')}), {len(rows)} groups, {args.random_sets} random sets each",
             f"classes: lexical {counts['lexical']}, functional {counts['functional']}, other {counts['other']}",
             f"functional: {summary['functional_groups']}", f"lexical: {summary['lexical_groups']}",
             "mean loss increase per category (groups | random same-size sets):"]
    lines += [f"  {k:42s} {summary['mean_damage'][k]:+.4f} | {summary['mean_random_damage'][k]:+.4f}  "
              f"(max {summary['max_damage'][k]:+.4f}, {summary['targets'][k]} targets)" for k in names]
    lines.append(f"time: {summary['time_s']}s")
    with open(os.path.join(args.out_dir, "damage_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
