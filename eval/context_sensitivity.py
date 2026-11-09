"""Context sensitivity of group selection: does the same token pick the same groups in arithmetic and in stories?

For digit tokens (arithmetic lines vs number stories) and number words (word problems vs number stories), per gated
layer: the Jaccard distance between the selected-group sets of the SAME token id in the two contexts (cross), and
between two occurrences in the same context (within: the noise floor). Distances are averaged over occurrence pairs,
then over token ids. effect = cross - within: about 0 means selection is lexical (the token decides), larger means the
context changes which groups are picked. Probe texts come from deconf.build_context_probes (generated, fixed seed).

Runs at each --route_smooth value in --smooth_values (the causal EMA of scores along the sequence). Zero training
needed: on a dense checkpoint it uses post-hoc groups and inference-only gating (the forward pass stays dense). On a
gated checkpoint the gates are applied, so changing route_smooth changes the forward pass.
Outputs (in out_dir): context_sensitivity.json, context_sensitivity.csv, context_sensitivity.txt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deconf import PROBE_PAIRS, build_context_probes  # noqa: E402
from gating_common import ENC, add_args, load, write_json  # noqa: E402
from model.gating import context_sensitivity  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--smooth_values", default="0,0.9")
    p.add_argument("--n_seqs", type=int, default=128, help="probe sequences per context")
    p.add_argument("--max_occ", type=int, default=50, help="occurrences per token id and context")
    args = p.parse_args()
    t0 = time.time()
    model, ckpt, router, source, device = load(args)
    probes = build_context_probes(ENC, n_seqs=args.n_seqs, seq_len=min(model.config.block_size, 128))
    trained_smooth = router.smooth
    results, rows = {}, []
    for s in [float(x) for x in args.smooth_values.split(",")]:
        router.smooth = s
        res = context_sensitivity(model, router, probes, PROBE_PAIRS, max_occ=args.max_occ, seed=args.seed)
        results[str(s)] = {name: {str(layer): v for layer, v in per.items()} for name, per in res.items()}
        for name, per in res.items():
            for layer, v in per.items():
                rows.append({"route_smooth": s, "tokens": name, "layer": layer, **v})
    router.smooth = trained_smooth

    summary = {"source": source, "checkpoint_iter": ckpt.get("iter"), "k": router.k, "n_groups": router.G,
               "trained_route_smooth": trained_smooth if source != "posthoc" else None, "results": results,
               "mean_effect": {s: {name: float(np.nanmean([v["effect"] for v in per.values()])) for name, per in r.items()}
                               for s, r in results.items()},
               "time_s": round(time.time() - t0)}
    write_json(os.path.join(args.out_dir, "context_sensitivity.json"), summary)
    with open(os.path.join(args.out_dir, "context_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["route_smooth", "tokens", "layer", "cross", "within", "effect", "n_tokens"])
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    lines = [f"source: {source} (checkpoint iter {ckpt.get('iter')}), k={router.k} of {router.G} groups, "
             f"layers {router.layers}",
             "Jaccard distance of selected groups, same token: arithmetic vs story (cross), two occurrences in one "
             "context (within); effect = cross - within"]
    for r in rows:
        lines.append(f"  smooth {r['route_smooth']:<4} {r['tokens']:13s} L{r['layer']}: cross {r['cross']:.3f} within "
                     f"{r['within']:.3f} effect {r['effect']:+.3f} ({r['n_tokens']} token ids)")
    lines.append(f"time: {summary['time_s']}s")
    with open(os.path.join(args.out_dir, "context_sensitivity.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
