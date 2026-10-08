"""Shared helpers: device/AMP setup, data batches, deterministic per-domain eval, checkpoint loading."""
import contextlib
import os
import runpy

import numpy as np
import torch

from model.model import GPT, GPTConfig

DOMAINS = ["story", "math", "code"]
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config_file(path):
    """Read a python config file (plain `name = value` lines) into a dict."""
    return {k: v for k, v in runpy.run_path(path).items() if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}


def pick_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def autocast_ctx(device):
    """fp16 autocast on CUDA (T4: no bf16), plain fp32 on CPU."""
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device):
    enabled = device.startswith("cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def open_bin(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run data/prepare.py first")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device, rng=None):
    """Random contiguous windows. `rng` (np.random.Generator) makes it deterministic."""
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"split has {len(data)} tokens, need > block_size+1={block_size + 1}")
    ix = rng.integers(0, n, size=batch_size) if rng is not None else np.random.randint(0, n, size=batch_size)
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, n_batches, batch_size, block_size, device, seed=1234):
    """Mean loss over a FIXED set of windows (same seed -> same windows), so runs are comparable."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    with autocast_ctx(device):
        for _ in range(n_batches):
            x, y = get_batch(data, batch_size, block_size, device, rng)
            _, loss = model(x, y)
            losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def eval_domains(model, val_data, n_batches, batch_size, block_size, device):
    return {d: eval_loss(model, val_data[d], n_batches, batch_size, block_size, device) for d in DOMAINS}


def load_val(data_dir):
    return {d: open_bin(data_dir, f"val_{d}.bin") for d in DOMAINS}


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run train.py first")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
