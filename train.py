"""Train the GPT on the mixed 3-domain corpus.

Config precedence: built-in DEFAULTS < --config file (e.g. configs/baseline.py) < explicit CLI flags.
  python train.py --config configs/baseline.py
  python train.py --config configs/baseline.py --max_iters 2000 --batch_size 16
  python train.py --config configs/baseline.py --resume     # continue from out_dir/ckpt.pt

fp16 autocast + GradScaler on CUDA (no bf16, no torch.compile); fp32 on CPU.
Writes out_dir/metrics.csv (iter, train_loss, val_story, val_math, val_code, lr, elapsed_s) and out_dir/ckpt.pt.
--routing_mode picks the method: dense (baseline), batch (v0), token (v1), label (oracle baseline), all in
model/routing.py, assembly (v2, model/assemblies.py), which writes out_dir/assemblies.csv and assemblies_final.json,
or gated (v3, model/gating.py; --groups fixed|evolved), which writes out_dir/gating.csv, gating_layers.csv (with the
context-sensitivity metric at every eval) and groups_final.json. The other routed modes write out_dir/routing_stats.csv. --routing is the old spelling of --routing_mode batch.
train_loss is always the LM loss alone. tokens_per_sec excludes eval and checkpoint time.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DOMAINS, ROOT, PureWindowSampler, autocast_ctx, eval_domains, get_batch,  # noqa: E402
                    load_config_file, load_val, make_grad_scaler, open_bin, open_domains, pick_device)
from model.model import GPT, GPTConfig  # noqa: E402
from model.assemblies import AssemblyRouter  # noqa: E402
from model.gating import context_sensitivity, router_from_config  # noqa: E402
from deconf import PROBE_PAIRS, build_context_probes  # noqa: E402
from model.routing import GradientRouter, TokenRouter  # noqa: E402

ROUTING_MODES = ("dense", "batch", "token", "label", "assembly", "gated")

DEFAULTS = dict(
    # io
    data_dir=os.path.join(ROOT, "data", "out"),
    out_dir=os.path.join(ROOT, "out"),
    device="auto",
    seed=1337,
    resume=False,
    # model
    n_layer=6, n_head=6, n_embd=384, block_size=256, n_blocks=8, dropout=0.0, bias=False,
    # optimisation
    batch_size=32, grad_accum=1, learning_rate=6e-4, min_lr=6e-5, warmup_iters=200, max_iters=5000,
    weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    # logging / eval
    eval_interval=250, eval_iters=50, log_interval=50, ckpt_interval=1000,
    # gradient routing (dense by default). route_eps, route_lambda and route_mask_aux are batch mode only;
    # route_gamma (bias step) is token mode only; route_ema is the frequency EMA in batch and the energy EMA in token
    routing_mode="dense", routing=False, route_k=2, route_eps=0.0, route_lambda=0.01, route_ema=0.99,
    route_mask_aux=False, route_gamma=0.001,
    # assembly mode (route_ema is its correlation and activity EMA)
    cluster_start=500, cluster_every=500, knn=32, resolution=1.0, min_size=16, floor=0.1, alpha=4.0,
    plasticity="rank", consolidate=False, beta=0.5, assembly_sample_tokens=512,
    # gated mode (v3). Shares cluster_start (gating starts, dense before), cluster_every, knn, resolution, min_size,
    # route_gamma (balancing bias step), route_ema (activity and correlation EMA) and assembly_sample_tokens
    groups="fixed", n_groups=8, gate_k=3, gate_tau=0.1, route_smooth=0.9, gated_layers="2,3,4,5", max_move=0.05,
    # scales the whole learning rate schedule, e.g. a dense run matched to an assembly run's mean plasticity
    lr_scale=1.0,
)


def parse_bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {s}")


def get_config(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="python config file, e.g. configs/baseline.py")
    for k, v in DEFAULTS.items():
        typ = parse_bool if isinstance(v, bool) else type(v)
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=typ, nargs="?", const=True, default=None)
        else:
            p.add_argument(f"--{k}", type=typ, default=None)
    args = p.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        file_cfg = load_config_file(args.config)
        unknown = set(file_cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown keys in {args.config}: {sorted(unknown)}")
        cfg.update(file_cfg)
    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    if cfg["routing"] and cfg["routing_mode"] == "dense":
        cfg["routing_mode"] = "batch"
    if cfg["plasticity"] not in ("rank", "sigmoid"):
        raise ValueError(f"plasticity={cfg['plasticity']!r}, expected rank or sigmoid")
    if cfg["groups"] not in ("fixed", "evolved"):
        raise ValueError(f"groups={cfg['groups']!r}, expected fixed or evolved")
    if cfg["routing_mode"] not in ROUTING_MODES:
        raise ValueError(f"routing_mode={cfg['routing_mode']!r}, expected one of {ROUTING_MODES}")
    cfg["routing"] = cfg["routing_mode"] == "batch"
    return cfg


def lr_at(it, c):
    return c["lr_scale"] * _base_lr(it, c)


def _base_lr(it, c):
    if it < c["warmup_iters"]:
        return c["learning_rate"] * (it + 1) / (c["warmup_iters"] + 1)
    if it >= c["max_iters"]:
        return c["min_lr"]
    ratio = (it - c["warmup_iters"]) / max(1, c["max_iters"] - c["warmup_iters"])
    return c["min_lr"] + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (c["learning_rate"] - c["min_lr"])


def fmt_time(s):
    s = int(max(s, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def main(argv=None):
    c = get_config(argv)
    device = pick_device(c["device"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True  # no-op on T4, harmless elsewhere
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(c["out_dir"], exist_ok=True)

    meta_path = os.path.join(c["data_dir"], "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            mode = json.load(f).get("mode")
        if mode == "synthetic_fallback":
            print("[train] WARNING: data/out was built from the SYNTHETIC FALLBACK (HF download failed).", flush=True)

    train_data = open_bin(c["data_dir"], "train.bin")
    val_data = load_val(c["data_dir"])
    print(f"[train] train tokens: {len(train_data):,} | " + " | ".join(f"val_{d}: {len(val_data[d]):,}" for d in DOMAINS))

    model_args = {k: c[k] for k in ("n_layer", "n_head", "n_embd", "block_size", "n_blocks", "dropout", "bias")}
    model_args["vocab_size"] = 50304
    model = GPT(GPTConfig(**model_args)).to(device)
    optimizer = model.configure_optimizers(c["weight_decay"], c["learning_rate"], (c["beta1"], c["beta2"]))
    scaler = make_grad_scaler(device)
    ctx = autocast_ctx(device)
    mode = c["routing_mode"]
    router, sampler, train_domains = None, None, None
    if mode == "batch":
        router = GradientRouter(model, k=c["route_k"], epsilon=c["route_eps"], aux_lambda=c["route_lambda"],
                                ema=c["route_ema"], mask_aux=c["route_mask_aux"])
        print(f"[train] gradient routing ON: top-{c['route_k']} of {c['n_blocks']} blocks per layer, "
              f"eps={c['route_eps']}, lambda={c['route_lambda']}, ema={c['route_ema']}, mask_aux={c['route_mask_aux']}")
    elif mode in ("token", "label"):
        router = TokenRouter(model, mode=mode, k=c["route_k"], gamma=c["route_gamma"], ema=c["route_ema"])
        print(f"[train] per-token routing ON: mode={mode}, k={c['route_k']}, gamma={c['route_gamma']}, "
              f"ema={c['route_ema']}")
    elif mode == "assembly":
        router = AssemblyRouter(model, cluster_start=c["cluster_start"], cluster_every=c["cluster_every"], knn=c["knn"],
                                resolution=c["resolution"], min_size=c["min_size"], floor=c["floor"], alpha=c["alpha"],
                                plasticity=c["plasticity"], ema=c["route_ema"], consolidate=c["consolidate"], beta=c["beta"],
                                sample_tokens=c["assembly_sample_tokens"], seed=c["seed"])
        print(f"[train] assembly routing ON: dense until step {c['cluster_start']}, re-cluster every "
              f"{c['cluster_every']}, knn={c['knn']}, resolution={c['resolution']}, min_size={c['min_size']}, "
              f"floor={c['floor']}, plasticity={c['plasticity']}, alpha={c['alpha']}, consolidate={c['consolidate']}")
    elif mode == "gated":
        router = router_from_config(model, c)
        print(f"[train] gated ON: groups={c['groups']}, layers {router.layers}, {c['n_groups']} groups, k={c['gate_k']}, "
              f"tau={c['gate_tau']}, route_smooth={c['route_smooth']}, gamma={c['route_gamma']}, dense until step "
              f"{c['cluster_start']}" + (f", re-cluster every {c['cluster_every']} moving at most {c['max_move']:.0%}"
                                         if c["groups"] == "evolved" else ""))
    if mode == "label":
        sampler = PureWindowSampler(c["data_dir"], c["block_size"])
        print("[train] label mode: domain-pure windows, domain mix " + ", ".join(
            f"{d} {p:.3f}" for d, p in zip(DOMAINS, sampler.p)))
    elif mode in ("batch", "token", "assembly", "gated"):
        train_domains = open_domains(c["data_dir"], len(train_data))  # logging only

    def next_batch():
        """(x, y, domains); domains is None in dense mode."""
        if sampler is not None:
            return sampler.get_batch(c["batch_size"], device)
        if train_domains is not None:
            return get_batch(train_data, c["batch_size"], c["block_size"], device, domains=train_domains)
        return (*get_batch(train_data, c["batch_size"], c["block_size"], device), None)

    print(f"[train] device={device} amp={'fp16' if device.startswith('cuda') else 'off'} "
          f"params={model.num_params() / 1e6:.2f}M (non-embedding {(model.num_params() - model.transformer.wte.weight.numel()) / 1e6:.2f}M)")

    ckpt_path = os.path.join(c["out_dir"], "ckpt.pt")
    metrics_path = os.path.join(c["out_dir"], "metrics.csv")
    routing_path = os.path.join(c["out_dir"], {"assembly": "assemblies.csv", "gated": "gating.csv"}.get(mode, "routing_stats.csv"))
    assignments_path = os.path.join(c["out_dir"], "groups_final.json" if mode == "gated" else "assemblies_final.json")
    probes = None
    if mode == "gated":
        import tiktoken
        probes = build_context_probes(tiktoken.get_encoding("gpt2"), n_seqs=32, seq_len=min(c["block_size"], 128))
    if mode == "assembly":
        router.log_path = routing_path
    start_iter = 0
    if c["resume"] and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt["iter"]
        if router is not None and ckpt.get("router"):
            router.load_state_dict(ckpt["router"])
        print(f"[train] resumed from {ckpt_path} at iter {start_iter}")
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "train_loss", *[f"val_{d}" for d in DOMAINS], "lr", "elapsed_s",
                                    "tokens_per_sec", *[f"act_norm_l{i}" for i in range(c["n_layer"])]])
        for path in (routing_path, routing_path.replace(".csv", "_layers.csv")):
            if os.path.exists(path):
                os.remove(path)

    def save_ckpt(it):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "model_args": model_args, "config": c, "iter": it,
                    "router": router.state_dict() if router is not None else None}, tmp)
        os.replace(tmp, ckpt_path)  # atomic: a crash mid-save never corrupts the last good checkpoint
        if mode in ("assembly", "gated"):
            router.write_assignments(assignments_path)
            router.write_summary(os.path.join(c["out_dir"], "summary.txt"))
        print(f"[train] saved checkpoint @ iter {it} -> {ckpt_path}", flush=True)

    def evaluate(it, train_loss, lr, tokens_per_sec=None):
        model.record_block_norms(True)   # per-layer activation size on the val windows, so a blow-up is visible
        losses = eval_domains(model, val_data, c["eval_iters"], c["batch_size"], c["block_size"], device)
        act_norms = model.get_block_norms().mean(dim=1).tolist()
        model.record_block_norms(False)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([it, f"{train_loss:.5f}", *[f"{losses[d]:.5f}" for d in DOMAINS], f"{lr:.3e}",
                                    f"{time.time() - t_start:.1f}",
                                    "" if tokens_per_sec is None else f"{tokens_per_sec:.0f}",
                                    *[f"{a:.3f}" for a in act_norms]])
        print(f"[eval] iter {it}: train {train_loss:.4f} | " + " | ".join(f"{d} {losses[d]:.4f}" for d in DOMAINS), flush=True)
        if mode == "gated" and router.steps > 0:
            extra = {}
            if all(router.active(i) for i in range(len(router.layers))):
                ctx = context_sensitivity(model, router, probes, PROBE_PAIRS, max_occ=30)
                extra = {layer: {f"ctx_{name}_{k}": f"{v[layer][k]:.4f}" for name, v in ctx.items()
                                 for k in ("cross", "within", "effect")} for layer in router.layers}
            router.log(it, routing_path, extra)
            print(f"[route] iter {it}: {router.summary()}", flush=True)
        elif router is not None and router.steps > 0:
            router.log(it, routing_path)
            print(f"[route] iter {it}: {router.summary()}", flush=True)

    model.train()
    t_start = time.time()
    t_last = t_start
    running = []
    train_seconds, train_tokens = 0.0, 0   # since the previous eval, for tokens_per_sec
    if start_iter == 0:
        x0, y0 = get_batch(train_data, c["batch_size"], c["block_size"], device)
        with torch.no_grad(), ctx:
            init_loss = model(x0, y0)[1].item()
        evaluate(0, init_loss, lr_at(0, c))

    x, y, dom = next_batch()
    for it in range(start_iter, c["max_iters"]):
        t_iter = time.time()
        lr = lr_at(it, c)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for _ in range(c["grad_accum"]):
            if mode in ("token", "label", "assembly", "gated"):
                router.set_batch(dom)
            with ctx:
                _, loss = model(x, y)
                aux = router.route() if mode == "batch" else None
            if mode == "batch":
                router.observe_domains(dom)
            x, y, dom = next_batch()  # prefetch next
            if mode == "batch":
                router.backward(loss / c["grad_accum"], aux / c["grad_accum"], scaler.scale)
            else:
                scaler.scale(loss / c["grad_accum"]).backward()
        if c["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        if mode in ("token", "label", "assembly", "gated"):
            router.step(optimizer, scaler)
        else:
            scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise FloatingPointError(f"non-finite loss {loss_val} at iter {it}")
        running.append(loss_val)
        done = it + 1
        train_seconds += time.time() - t_iter
        train_tokens += c["batch_size"] * c["block_size"] * c["grad_accum"]

        if done % c["log_interval"] == 0 or done == c["max_iters"]:
            now = time.time()
            per_iter = (now - t_start) / (done - start_iter)  # includes eval/ckpt overhead
            eta = per_iter * (c["max_iters"] - done)
            print(f"[train] iter {done}/{c['max_iters']} loss {loss_val:.4f} lr {lr:.2e} "
                  f"{1000 * (now - t_last) / c['log_interval']:.0f}ms/it | elapsed {fmt_time(now - t_start)} "
                  f"| ETA {fmt_time(eta)}", flush=True)
            t_last = now
        if done % c["eval_interval"] == 0 or done == c["max_iters"]:
            evaluate(done, float(np.mean(running)), lr, train_tokens / max(train_seconds, 1e-9))
            running = []
            train_seconds, train_tokens = 0.0, 0
        if done % c["ckpt_interval"] == 0 or done == c["max_iters"]:
            save_ckpt(done)

    if start_iter >= c["max_iters"]:
        print(f"[train] checkpoint already at iter {start_iter} >= max_iters; nothing to do")
    if router is not None:
        print(f"[route] final: {router.summary()}")
    print(f"[train] finished in {fmt_time(time.time() - t_start)}")
    return {"model": model, "router": router, "config": c}


if __name__ == "__main__":
    main()
