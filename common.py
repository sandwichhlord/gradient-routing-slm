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


def _windows(data, ix, offset, block_size, device):
    t = torch.from_numpy(np.stack([data[i + offset:i + offset + block_size].astype(np.int64) for i in ix]))
    if device.startswith("cuda"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def get_batch(data, batch_size, block_size, device, rng=None, domains=None):
    """Random contiguous windows. `rng` (np.random.Generator) makes it deterministic.
    With `domains` (the train_domain.bin memmap) also returns each input token's domain id."""
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"split has {len(data)} tokens, need > block_size+1={block_size + 1}")
    ix = rng.integers(0, n, size=batch_size) if rng is not None else np.random.randint(0, n, size=batch_size)
    x, y = _windows(data, ix, 0, block_size, device), _windows(data, ix, 1, block_size, device)
    if domains is None:
        return x, y
    return x, y, _windows(domains, ix, 0, block_size, device)


class PureWindowSampler:
    """Windows that lie entirely inside one domain, for label mode. The domain of each window is drawn in proportion
    to its share of training tokens, so the domain mix matches the shuffled stream. Mixed windows would break label
    routing: through attention, a math token's loss would reach the story blocks used by story tokens before it."""

    def __init__(self, data_dir, block_size):
        self.data = open_bin(data_dir, "train.bin")
        self.domains = open_domains(data_dir, len(self.data))
        self.block_size = block_size
        dom = np.asarray(self.domains)
        change = np.flatnonzero(np.diff(dom)) + 1
        starts, ends = np.concatenate(([0], change)), np.concatenate((change, [len(dom)]))
        run_dom = dom[starts]
        self.starts, self.cum = [], []
        shares = np.zeros(len(DOMAINS))
        for d in range(len(DOMAINS)):
            shares[d] = (ends - starts)[run_dom == d].sum()
            ok = (run_dom == d) & (ends - starts > block_size)  # a window needs block_size + 1 tokens
            if not ok.any():
                raise ValueError(f"no {DOMAINS[d]} run of more than {block_size} tokens in the training data")
            self.starts.append(starts[ok])
            self.cum.append(np.cumsum((ends - starts)[ok] - block_size))
        self.p = shares / shares.sum()

    def get_batch(self, batch_size, device):
        ix = np.empty(batch_size, dtype=np.int64)
        for i, d in enumerate(np.random.choice(len(DOMAINS), size=batch_size, p=self.p)):
            cum = self.cum[d]
            r = np.random.randint(cum[-1])
            j = np.searchsorted(cum, r, side="right")
            ix[i] = self.starts[d][j] + r - (cum[j - 1] if j > 0 else 0)
        return (_windows(self.data, ix, 0, self.block_size, device), _windows(self.data, ix, 1, self.block_size, device),
                _windows(self.domains, ix, 0, self.block_size, device))


def open_domains(data_dir, expected_len):
    """Per-token domain ids (0 story, 1 math, 2 code), parallel to train.bin."""
    path = os.path.join(data_dir, "train_domain.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - rebuild the data with data/prepare.py")
    domains = np.memmap(path, dtype=np.uint8, mode="r")
    if len(domains) != expected_len:
        raise ValueError(f"train_domain.bin has {len(domains)} entries but train.bin has {expected_len} tokens")
    return domains


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
    """Model in eval mode. A gated checkpoint comes with its GateRouter attached (model.gate_router), because the gates
    are part of its forward pass."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run train.py first")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if ckpt.get("config", {}).get("routing_mode") == "gated":
        from model.gating import router_from_config
        router_from_config(model, ckpt["config"]).load_state_dict(ckpt["router"])
    return model, ckpt
