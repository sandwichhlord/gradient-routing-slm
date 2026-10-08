"""nanoGPT-style decoder-only GPT.

The MLP hidden layer (width 4*n_embd) is logically split into `n_blocks` equal contiguous blocks.
Two inference-time hooks, both OFF by default (so the baseline is an ordinary dense model):

  * ablation mask: `model.set_ablation(layer, block)` zeroes that block's hidden activations, which
    removes exactly that block's contribution to the MLP output. `model.set_block_mask(layer, mask)`
    accepts an arbitrary (n_blocks,) mask; `model.clear_ablation()` restores the dense model.
  * norm recording: `model.record_block_norms(True)`, run forward passes, then
    `model.get_block_norms()` -> (n_layer, n_blocks) tensor of mean per-token L2 norm of each
    block's hidden activation, averaged over all forward passes since the last reset.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304  # GPT-2's 50257 padded up to a multiple of 64
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_blocks: int = 8
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class BlockMLP(nn.Module):
    """Standard GELU MLP whose hidden layer is viewed as n_blocks equal blocks."""

    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.n_embd
        assert hidden % cfg.n_blocks == 0, f"4*n_embd={hidden} not divisible by n_blocks={cfg.n_blocks}"
        self.n_blocks, self.block_dim = cfg.n_blocks, hidden // cfg.n_blocks
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        # hooks (off by default)
        self.block_mask = None          # None or float tensor (n_blocks,)
        self.record_norms = False
        self.capture_energy = False     # in training mode, keep differentiable per-block energy (routing)
        self.energy = None              # (n_blocks,) mean over tokens of ||h_b||_2, set by forward
        self.energy_detach_input = False  # compute energy on a side graph from x.detach() (routing's default)
        self.reset_norms()

    def reset_norms(self):
        self.norm_sum = torch.zeros(self.n_blocks)
        self.norm_count = 0

    def _energy(self, h):
        """(n_blocks,) mean over tokens of each block's L2 norm."""
        return h.float().view(*h.shape[:-1], self.n_blocks, self.block_dim).norm(dim=-1).reshape(-1, self.n_blocks).mean(0)

    def forward(self, x):
        h = self.gelu(self.c_fc(x))
        capture = self.capture_energy and self.training
        if capture and self.energy_detach_input:
            # separate graph whose only parameters are this layer's c_fc (so its gradient can bypass routing masks)
            self.energy = self._energy(self.gelu(F.linear(x.detach(), self.c_fc.weight, self.c_fc.bias)))
        if self.block_mask is not None or self.record_norms or capture:
            hb = h.view(*h.shape[:-1], self.n_blocks, self.block_dim)
            if self.record_norms or (capture and not self.energy_detach_input):
                norms = hb.float().norm(dim=-1)  # (B, T, n_blocks)
                if capture and not self.energy_detach_input:
                    self.energy = norms.reshape(-1, self.n_blocks).mean(0)
                if self.record_norms:
                    norms = norms.detach()
                    self.norm_sum += norms.reshape(-1, self.n_blocks).sum(0).cpu()
                    self.norm_count += norms.numel() // self.n_blocks
            if self.block_mask is not None:
                mask = self.block_mask.to(device=h.device, dtype=h.dtype)
                hb = hb * mask.view(*([1] * (hb.dim() - 2)), self.n_blocks, 1)
                h = hb.view(h.shape)
        return self.dropout(self.c_proj(h))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = BlockMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    # ------------------------------------------------------------------ block hooks
    def mlps(self):
        return [b.mlp for b in self.transformer.h]

    def set_block_mask(self, layer, mask):
        self.mlps()[layer].block_mask = torch.as_tensor(mask, dtype=torch.float32)

    def set_ablation(self, layer, block):
        """Zero exactly one MLP block (and clear every other mask)."""
        self.clear_ablation()
        mask = torch.ones(self.config.n_blocks)
        mask[block] = 0.0
        self.set_block_mask(layer, mask)

    def clear_ablation(self):
        for m in self.mlps():
            m.block_mask = None

    def record_block_norms(self, on=True):
        for m in self.mlps():
            m.record_norms = on
            m.reset_norms()

    def get_block_norms(self):
        """(n_layer, n_blocks) mean per-token L2 norm of each block's hidden activation."""
        return torch.stack([m.norm_sum / max(m.norm_count, 1) for m in self.mlps()])

    # ------------------------------------------------------------------ training / sampling
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_token is not None and bool((idx_next == stop_token).all()):
                break
        return idx
