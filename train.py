"""Train the GPT on the mixed 3-domain corpus.

Config precedence: built-in DEFAULTS < --config file (e.g. configs/baseline.py) < explicit CLI flags.
  python train.py --config configs/baseline.py
  python train.py --config configs/baseline.py --max_iters 2000 --batch_size 16
  python train.py --config configs/baseline.py --resume     # continue from out_dir/ckpt.pt

fp16 autocast + GradScaler on CUDA (no bf16, no torch.compile); fp32 on CPU.
Writes out_dir/metrics.csv (iter, train_loss, val_story, val_math, val_code, lr, elapsed_s) and out_dir/ckpt.pt.
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
from common import (DOMAINS, ROOT, autocast_ctx, eval_domains, get_batch, load_config_file,  # noqa: E402
                    load_val, make_grad_scaler, open_bin, pick_device)
from model.model import GPT, GPTConfig  # noqa: E402

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
    return cfg


def lr_at(it, c):
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
    print(f"[train] device={device} amp={'fp16' if device.startswith('cuda') else 'off'} "
          f"params={model.num_params() / 1e6:.2f}M (non-embedding {(model.num_params() - model.transformer.wte.weight.numel()) / 1e6:.2f}M)")

    ckpt_path = os.path.join(c["out_dir"], "ckpt.pt")
    metrics_path = os.path.join(c["out_dir"], "metrics.csv")
    start_iter = 0
    if c["resume"] and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt["iter"]
        print(f"[train] resumed from {ckpt_path} at iter {start_iter}")
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "train_loss", *[f"val_{d}" for d in DOMAINS], "lr", "elapsed_s"])

    def save_ckpt(it):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "model_args": model_args, "config": c, "iter": it}, tmp)
        os.replace(tmp, ckpt_path)  # atomic: a crash mid-save never corrupts the last good checkpoint
        print(f"[train] saved checkpoint @ iter {it} -> {ckpt_path}", flush=True)

    def evaluate(it, train_loss, lr):
        losses = eval_domains(model, val_data, c["eval_iters"], c["batch_size"], c["block_size"], device)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([it, f"{train_loss:.5f}", *[f"{losses[d]:.5f}" for d in DOMAINS], f"{lr:.3e}",
                                    f"{time.time() - t_start:.1f}"])
        print(f"[eval] iter {it}: train {train_loss:.4f} | " + " | ".join(f"{d} {losses[d]:.4f}" for d in DOMAINS), flush=True)

    model.train()
    t_start = time.time()
    t_last = t_start
    running = []
    if start_iter == 0:
        x0, y0 = get_batch(train_data, c["batch_size"], c["block_size"], device)
        with torch.no_grad(), ctx:
            init_loss = model(x0, y0)[1].item()
        evaluate(0, init_loss, lr_at(0, c))

    x, y = get_batch(train_data, c["batch_size"], c["block_size"], device)
    for it in range(start_iter, c["max_iters"]):
        lr = lr_at(it, c)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for _ in range(c["grad_accum"]):
            with ctx:
                _, loss = model(x, y)
                loss = loss / c["grad_accum"]
            x, y = get_batch(train_data, c["batch_size"], c["block_size"], device)  # prefetch next
            scaler.scale(loss).backward()
        if c["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * c["grad_accum"]
        if not math.isfinite(loss_val):
            raise FloatingPointError(f"non-finite loss {loss_val} at iter {it}")
        running.append(loss_val)
        done = it + 1

        if done % c["log_interval"] == 0 or done == c["max_iters"]:
            now = time.time()
            per_iter = (now - t_start) / (done - start_iter)  # includes eval/ckpt overhead
            eta = per_iter * (c["max_iters"] - done)
            print(f"[train] iter {done}/{c['max_iters']} loss {loss_val:.4f} lr {lr:.2e} "
                  f"{1000 * (now - t_last) / c['log_interval']:.0f}ms/it | elapsed {fmt_time(now - t_start)} "
                  f"| ETA {fmt_time(eta)}", flush=True)
            t_last = now
        if done % c["eval_interval"] == 0 or done == c["max_iters"]:
            evaluate(done, float(np.mean(running)), lr)
            running = []
        if done % c["ckpt_interval"] == 0 or done == c["max_iters"]:
            save_ckpt(done)

    if start_iter >= c["max_iters"]:
        print(f"[train] checkpoint already at iter {start_iter} >= max_iters; nothing to do")
    print(f"[train] finished in {fmt_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()
