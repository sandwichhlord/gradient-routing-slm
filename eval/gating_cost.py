"""Loss cost of inference-only gating on an ungated (dense) checkpoint.

Post-hoc groups (as in the other v3 evaluations) in --gated_layers, then the soft k-WTA gates are APPLIED to the forward
pass, which the model never trained with, for each k in --k_values and each score in --scores, with route_smooth from
--route_smooth. Reports per-domain val loss (second val half) against the ungated model. A small cost means the dense
model already concentrates its useful MLP output in a few groups per token; a large one means gating has to be learned.
Outputs (in out_dir): gating_cost.json, gating_cost.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gating_common import DOMAINS, add_args, eval_sets, load, masked_losses, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p, windows=64)
    p.add_argument("--k_values", default="2,3,4")
    p.add_argument("--scores", default="relative,share")
    args = p.parse_args()
    t0 = time.time()
    model, ckpt, router, source, device = load(args)
    if source != "posthoc":
        raise SystemExit("gating_cost.py is for ungated checkpoints; a gated model is already gated")
    sets = {k: v for k, v in eval_sets(args.data_dir, model.config.block_size, args.windows, args.seed).items()
            if k.startswith("domain_")}
    router.apply = False
    base = {d: masked_losses(model, sets, device, args.batch_size)[f"domain_{d}"] for d in DOMAINS}
    rows = []
    router.apply = True
    for score in args.scores.split(","):
        router.score = score
        for k in [int(x) for x in args.k_values.split(",")]:
            router.k = k
            got = masked_losses(model, sets, device, args.batch_size)
            loss = {d: got[f"domain_{d}"] for d in DOMAINS}
            rows.append({"score": score, "k": k, "loss": loss, "rel_change": {d: loss[d] / base[d] - 1 for d in DOMAINS}})
            print(f"[cost] {score} k={k}: " + " | ".join(f"{d} {loss[d]:.4f} ({loss[d] / base[d] - 1:+.1%})"
                                                       for d in DOMAINS), flush=True)
    router.apply = False
    summary = {"source": source, "checkpoint_iter": ckpt.get("iter"), "layers": router.layers, "n_groups": router.G,
               "route_smooth": router.smooth, "tau": router.tau, "ungated": base, "gated": rows,
               "windows_per_domain": args.windows, "time_s": round(time.time() - t0)}
    write_json(os.path.join(args.out_dir, "gating_cost.json"), summary)
    lines = [f"inference-only gating on {source} groups, layers {router.layers}, {router.G} groups, tau {router.tau}, "
             f"route_smooth {router.smooth}, {args.windows} windows per domain",
             f"{'':16s}" + "".join(f"{d:>18s}" for d in DOMAINS),
             f"{'ungated':16s}" + "".join(f"{base[d]:18.4f}" for d in DOMAINS)]
    lines += [f"{r['score'] + ' k=' + str(r['k']):16s}" + "".join(f"{r['loss'][d]:10.4f} ({r['rel_change'][d]:+6.1%})"
                                                                for d in DOMAINS) for r in rows]
    lines.append(f"time: {summary['time_s']}s")
    with open(os.path.join(args.out_dir, "gating_cost.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
