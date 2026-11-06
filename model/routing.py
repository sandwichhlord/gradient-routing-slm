"""Activation-based gradient routing over the MLP blocks of a GPT (see README "Method").

Two routers live here:
  GradientRouter  v0, `--routing_mode batch`: one block selection per layer per micro-batch (described below).
  TokenRouter     v1, `--routing_mode token` / `label`: one selection per token (see the class docstring).

GradientRouter, per training micro-batch, for every layer l:
  1. energy   a_{l,b} = mean over tokens of ||h_{l,b}||_2        (h = post-GELU MLP hidden, block b)
  2. select   S_l = top-k blocks by a_{l,b}                       (k=2 of 8 by default)
  3. mask     gradients of block b's c_fc rows and c_proj columns are multiplied by
              1 if b in S_l else epsilon                          (epsilon=0 -> hard routing)
  4. balance  f_{l,b} <- ema * f_{l,b} + (1 - ema) * 1[b in S_l]  (selection-frequency EMA, no grad)
              aux = lambda * sum_{l,b} f_{l,b} * a_{l,b} / sum_b' a_{l,b'}

The mask in step 3 always applies to the LM-loss gradient. By default (mask_aux=False) the aux loss is
computed on a side graph a_{l,b}(x.detach(); c_fc of layer l) and its gradient is NOT masked, so it can
raise the energy of blocks that are not currently selected - the same role the balancing loss plays for
an MoE router. With mask_aux=True everything (LM + aux) goes through the mask; at lambda=0.01 this lets
blocks die (rich-get-richer: unselected blocks never get a gradient that could make them win).

Usage (train.py does this when --routing is set):
    router = GradientRouter(model, k=2)
    _, loss = model(x, y)
    aux = router.route()             # picks blocks, sets gradient masks, returns aux loss
    router.backward(loss, aux)       # aux backward (unmasked by default), then masked LM backward

The masks are applied with tensor hooks on the weight leaves, i.e. to each backward's contribution
before it is accumulated into .grad, so gradient accumulation routes every micro-batch separately.
Routing only changes training; a routed checkpoint loads and evaluates as a plain GPT.
"""
import csv
import os

import torch

DOMAINS = ["story", "math", "code"]
# label mode: fixed blocks per domain (Gradient Routing 2024 style); blocks 6 and 7 are shared
LABEL_BLOCKS = {0: (0, 1), 1: (2, 3), 2: (4, 5)}
LABEL_SHARED = (6, 7)


def _domain_fracs(dom_sel, dom_tokens, layer, b):
    """Per-domain selection fraction as CSV strings (blank when that domain had no tokens)."""
    return [f"{dom_sel[d, layer, b] / dom_tokens[d]:.5f}" if dom_tokens[d] > 0 else "" for d in range(len(DOMAINS))]


class GradientRouter:
    def __init__(self, model, k=2, epsilon=0.0, aux_lambda=0.01, ema=0.99, mask_aux=False):
        self.mlps = model.mlps()
        self.n_layer, self.n_blocks = len(self.mlps), model.config.n_blocks
        self.block_dim = self.mlps[0].block_dim
        assert 1 <= k <= self.n_blocks, f"route_k={k} must be in [1, n_blocks={self.n_blocks}]"
        assert 0.0 <= epsilon <= 1.0, f"route_eps={epsilon} must be in [0, 1]"
        self.k, self.epsilon, self.aux_lambda, self.ema, self.mask_aux = k, epsilon, aux_lambda, ema, mask_aux
        self._masking = True
        device = model.lm_head.weight.device
        self.freq = torch.full((self.n_layer, self.n_blocks), k / self.n_blocks, device=device)
        self.counts = torch.zeros(self.n_layer, self.n_blocks, dtype=torch.long, device=device)
        self.last_selected = torch.full((self.n_layer, self.n_blocks), -1, dtype=torch.long, device=device)
        self.steps = 0
        self.last_energy = torch.zeros(self.n_layer, self.n_blocks, device=device)
        self.selected = None              # (n_layer, n_blocks) 0/1 of the last route() call
        # logging only, reset at each log(): how many tokens of each domain were in batches that selected a block
        self.dom_sel = torch.zeros(len(DOMAINS), self.n_layer, self.n_blocks, dtype=torch.long, device=device)
        self.dom_tokens = torch.zeros(len(DOMAINS), dtype=torch.long, device=device)
        self._grad_masks = [None] * self.n_layer   # per layer: (hidden,) multiplier
        self._handles = []
        for layer, mlp in enumerate(self.mlps):
            mlp.capture_energy = True
            mlp.energy_detach_input = not mask_aux
            # c_fc.weight is (hidden, n_embd): a block owns rows; c_proj.weight is (n_embd, hidden): columns
            self._handles.append(mlp.c_fc.weight.register_hook(self._hook(layer, dim=0)))
            self._handles.append(mlp.c_proj.weight.register_hook(self._hook(layer, dim=1)))
            if mlp.c_fc.bias is not None:  # c_fc's bias entries also belong to blocks (c_proj's bias does not)
                self._handles.append(mlp.c_fc.bias.register_hook(self._hook(layer, dim=0)))

    def _hook(self, layer, dim):
        def hook(grad):
            mask = self._grad_masks[layer]
            if mask is None or not self._masking:
                return grad
            shape = [1] * grad.dim()
            shape[dim] = -1
            return grad * mask.to(grad.dtype).view(shape)
        return hook

    def route(self, update_stats=True):
        """Call after the forward pass, before backward. Returns the load-balancing aux loss."""
        energies = []
        for mlp in self.mlps:
            if mlp.energy is None:
                raise RuntimeError("no MLP energy recorded - call route() right after a training-mode forward")
            energies.append(mlp.energy)
            mlp.energy = None  # never reuse a stale energy
        energy = torch.stack(energies)                     # (L, n_blocks), differentiable
        a = energy.detach()
        top = a.topk(self.k, dim=1).indices
        sel = torch.zeros_like(a).scatter_(1, top, 1.0)
        mult = sel + self.epsilon * (1.0 - sel)
        for layer in range(self.n_layer):
            self._grad_masks[layer] = mult[layer].repeat_interleave(self.block_dim)
        if update_stats:
            self.freq.mul_(self.ema).add_((1.0 - self.ema) * sel)
            self.counts += sel.long()
            self.steps += 1
            self.last_selected[sel.bool()] = self.steps
        self.selected, self.last_energy = sel, a
        share = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return self.aux_lambda * (self.freq * share).sum()

    def backward(self, loss, aux, scale=None):
        """Backprop LM loss (masked) and aux loss (unmasked unless mask_aux). `scale` e.g. GradScaler.scale."""
        scale = scale or (lambda t: t)
        if self.mask_aux:
            scale(loss + aux).backward()
            return
        self._masking = False
        try:
            scale(aux).backward()  # side graph: only reaches the c_fc weights
        finally:
            self._masking = True
        scale(loss).backward()

    def observe_domains(self, domains):
        """Logging only, call after route(): `domains` is (B, T) token labels; never affects training."""
        counts = torch.bincount(domains.view(-1), minlength=len(DOMAINS))
        self.dom_sel += counts.view(-1, 1, 1) * self.selected.long()
        self.dom_tokens += counts

    def clear(self):
        """Drop the current gradient masks (subsequent backwards are dense)."""
        self._grad_masks = [None] * self.n_layer

    def remove(self):
        """Detach all hooks: the model goes back to a plain dense GPT."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self.clear()
        for mlp in self.mlps:
            mlp.capture_energy, mlp.energy, mlp.energy_detach_input = False, None, False

    # ------------------------------------------------------------------ stats / checkpointing
    def dead_blocks(self, window=None):
        """(layer, block) pairs never selected, or (with `window`) not selected in the last `window` steps."""
        dead = self.counts == 0
        if window is not None:
            dead |= (self.steps - self.last_selected) >= window
        return [tuple(ix) for ix in dead.nonzero().tolist()]

    def log(self, it, path):
        """Append one row per (layer, block) to a long-format CSV."""
        new = not os.path.exists(path)
        freq, counts, energy = self.freq.cpu(), self.counts.cpu(), self.last_energy.cpu()
        dom_sel, dom_tokens = self.dom_sel.cpu(), self.dom_tokens.cpu()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["iter", "layer", "block", "freq_ema", "selected_count", "selected_frac", "energy",
                            *[f"frac_{d}" for d in DOMAINS]])
            for layer in range(self.n_layer):
                for b in range(self.n_blocks):
                    c = int(counts[layer, b])
                    w.writerow([it, layer, b, f"{freq[layer, b]:.5f}", c, f"{c / max(self.steps, 1):.5f}",
                                f"{energy[layer, b]:.5f}", *_domain_fracs(dom_sel, dom_tokens, layer, b)])
        self.dom_sel.zero_()
        self.dom_tokens.zero_()

    def summary(self):
        frac = (self.counts.float() / max(self.steps, 1)).cpu()
        return (f"routing: k={self.k}/{self.n_blocks} eps={self.epsilon} mask_aux={self.mask_aux} | selection frac "
                f"min {frac.min():.3f} max {frac.max():.3f} (uniform {self.k / self.n_blocks:.3f}) | "
                f"dead blocks: {len(self.dead_blocks())}")

    def state_dict(self):
        return {"freq": self.freq.cpu(), "counts": self.counts.cpu(), "last_selected": self.last_selected.cpu(),
                "steps": self.steps}

    def load_state_dict(self, sd):
        self.freq.copy_(sd["freq"])
        self.counts.copy_(sd["counts"])
        self.last_selected.copy_(sd["last_selected"])
        self.steps = sd["steps"]


class TokenRouter:
    """Per-token routing for `--routing_mode token` and `label`. The forward output stays equal to the dense model.

    Every training forward, for every layer, each token gets a 0/1 mask over the blocks (BlockMLP applies it to both
    projections, see BlockMLP._token_routed_output):
      token  score(t, b) = ||h_b(t)||_2 / EMA_b + bias_b, keep the top k blocks. EMA_b is a running mean of the block's
             average energy, so scores measure relative activity. bias_b only affects selection and is nudged after each
             step towards a selection fraction of k / n_blocks (DeepSeek-V3 style). There is no aux loss in this mode.
      label  fixed blocks per domain label (LABEL_BLOCKS, LABEL_SHARED). An oracle baseline, which needs domain-pure
             windows (common.PureWindowSampler) because attention carries a later token's loss back through earlier
             tokens' blocks.

    Domain labels drive the masks only in label mode. In token mode they are only used for the logged fractions.

    Usage:
        router = TokenRouter(model, mode="token")
        router.set_batch(domains)          # (B, T) labels, or None
        loss = model(x, y)[1]; loss.backward()
        router.step(optimizer, scaler)     # skips blocks with exactly zero gradient, then updates the bias
        scaler.update()
    """

    def __init__(self, model, mode="token", k=2, gamma=0.001, ema=0.99):
        assert mode in ("token", "label"), mode
        self.mode, self.k, self.gamma, self.ema_decay = mode, k, gamma, ema
        self.mlps = model.mlps()
        self.n_layer, self.n_blocks = len(self.mlps), model.config.n_blocks
        self.block_dim = self.mlps[0].block_dim
        assert 1 <= k <= self.n_blocks, f"route_k={k} must be in [1, n_blocks={self.n_blocks}]"
        device = model.lm_head.weight.device
        shape = (self.n_layer, self.n_blocks)
        self.ema = torch.zeros(shape, device=device)
        self.ema_ready = torch.zeros(self.n_layer, dtype=torch.bool, device=device)
        self.bias = torch.zeros(shape, device=device)
        self.domains = None
        self.last_mask = [None] * self.n_layer      # (B, T, n_blocks) of the last forward
        self.steps = 0
        self.skipped_slices = 0                     # block slices whose optimizer update was skipped
        # per-step fraction of tokens selecting each block, for the bias update
        self._step_frac = torch.zeros(shape, device=device)
        self._step_passes = 0
        # cumulative counters
        self.total_sel = torch.zeros(shape, dtype=torch.long, device=device)
        self.total_tokens = 0
        # logging window, reset at each log()
        self.dom_sel = torch.zeros(len(DOMAINS), *shape, dtype=torch.long, device=device)
        self.dom_tokens = torch.zeros(len(DOMAINS), dtype=torch.long, device=device)
        self.win_sel = torch.zeros(shape, dtype=torch.long, device=device)
        self.win_tokens = 0
        self._label_table = None
        if mode == "label":
            assert self.n_blocks == 8, "label mode assigns blocks for n_blocks=8"
            table = torch.zeros(len(DOMAINS), self.n_blocks, device=device)
            for d, blocks in LABEL_BLOCKS.items():
                table[d, list(blocks) + list(LABEL_SHARED)] = 1.0
            self._label_table = table
        self._slices = []                           # (param, dim) whose block slices may skip the optimizer
        for layer, mlp in enumerate(self.mlps):
            mlp.token_router, mlp.layer_idx = self, layer
            self._slices += [(mlp.c_fc.weight, 0), (mlp.c_proj.weight, 1)]
            if mlp.c_fc.bias is not None:
                self._slices.append((mlp.c_fc.bias, 0))

    def set_batch(self, domains):
        """Domain labels (B, T) for the coming forward. Required in label mode; token mode only logs with them."""
        self.domains = domains

    def select(self, layer, hb):
        """Called by BlockMLP. hb is detached (B, T, n_blocks, block_dim). Returns the (B, T, n_blocks) 0/1 mask."""
        if self.mode == "label":
            if self.domains is None:
                raise RuntimeError("label mode needs router.set_batch(domains) before each forward")
            m = self._label_table[self.domains]
        else:
            norms = hb.float().norm(dim=-1)         # (B, T, n_blocks)
            avg = norms.mean(dim=(0, 1))
            if self.ema_ready[layer]:
                self.ema[layer].mul_(self.ema_decay).add_((1.0 - self.ema_decay) * avg)
            else:
                self.ema[layer].copy_(avg)
                self.ema_ready[layer] = True
            score = norms / self.ema[layer].clamp_min(1e-8) + self.bias[layer]
            top = score.topk(self.k, dim=-1).indices
            m = torch.zeros_like(score).scatter_(-1, top, 1.0)
        self._record(layer, m)
        return m

    def _record(self, layer, m):
        self.last_mask[layer] = m
        n_tokens = m.shape[0] * m.shape[1]
        sel = m.sum(dim=(0, 1)).long()
        self._step_frac[layer] += sel.float() / n_tokens
        self.total_sel[layer] += sel
        self.win_sel[layer] += sel
        if layer == 0:
            self._step_passes += 1
            self.total_tokens += n_tokens
            self.win_tokens += n_tokens
        if self.domains is not None:
            d = self.domains.view(-1)
            self.dom_sel[:, layer].index_add_(0, d, m.view(-1, self.n_blocks).long())
            if layer == 0:
                self.dom_tokens += torch.bincount(d, minlength=len(DOMAINS))

    # ------------------------------------------------------------------ optimizer step
    def _save_idle_slices(self, optimizer):
        """Copy parameters and Adam moments of block slices with exactly zero gradient."""
        saved, n_idle = [], 0
        for p, dim in self._slices:
            if p.grad is None:
                continue
            g = p.grad if dim == 0 else p.grad.movedim(1, 0)
            active = (g.reshape(self.n_blocks, -1) != 0).any(dim=1)
            if bool(active.all()):
                continue
            n_idle += int((~active).sum())
            mask = (~active).repeat_interleave(self.block_dim)
            tensors = [p.data] + [v for v in optimizer.state.get(p, {}).values()
                                  if torch.is_tensor(v) and v.shape == p.shape]
            for t in tensors:
                saved.append((t, dim, mask, t[mask].clone() if dim == 0 else t[:, mask].clone()))
        self.skipped_slices += n_idle
        return saved

    @staticmethod
    def _restore_slices(saved):
        for t, dim, mask, values in saved:
            if dim == 0:
                t[mask] = values
            else:
                t[:, mask] = values

    def step(self, optimizer, scaler=None):
        """optimizer.step() that leaves zero-gradient block slices bit-identical (no update, no weight decay, Adam
        moments frozen), then updates the selection bias. The caller still runs scaler.update()."""
        saved = self._save_idle_slices(optimizer)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        self._restore_slices(saved)
        self.after_step()

    def after_step(self):
        if self.mode == "token" and self._step_passes > 0:
            frac = self._step_frac / self._step_passes
            self.bias += torch.where(frac > self.k / self.n_blocks, -self.gamma, self.gamma)
        self._step_frac.zero_()
        self._step_passes = 0
        self.steps += 1

    # ------------------------------------------------------------------ stats / checkpointing
    def dead_blocks(self):
        """(layer, block) pairs never selected by any token so far."""
        return [tuple(ix) for ix in (self.total_sel == 0).nonzero().tolist()]

    def selection_fracs(self):
        """(n_layer, n_blocks) fraction of all tokens so far that selected each block."""
        return (self.total_sel.float() / max(self.total_tokens, 1)).cpu()

    def log(self, it, path):
        """Append one row per (layer, block). frac_* cover the tokens since the previous log."""
        new = not os.path.exists(path)
        dom_sel, dom_tokens = self.dom_sel.cpu(), self.dom_tokens.cpu()
        win, cum, bias, ema = self.win_sel.cpu(), self.selection_fracs(), self.bias.cpu(), self.ema.cpu()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["iter", "layer", "block", *[f"frac_{d}" for d in DOMAINS], "frac_all", "cum_frac_all",
                            "bias", "ema"])
            for layer in range(self.n_layer):
                for b in range(self.n_blocks):
                    w.writerow([it, layer, b, *_domain_fracs(dom_sel, dom_tokens, layer, b),
                                f"{win[layer, b] / max(self.win_tokens, 1):.5f}", f"{cum[layer, b]:.5f}",
                                f"{bias[layer, b]:.5f}", f"{ema[layer, b]:.5f}"])
        self.dom_sel.zero_()
        self.dom_tokens.zero_()
        self.win_sel.zero_()
        self.win_tokens = 0

    def summary(self):
        frac = self.selection_fracs()
        return (f"routing: mode={self.mode} k={self.k}/{self.n_blocks} | selection frac min {frac.min():.3f} "
                f"max {frac.max():.3f} (uniform {self.k / self.n_blocks:.3f}) | dead blocks: {len(self.dead_blocks())} "
                f"| skipped slices: {self.skipped_slices}")

    def remove(self):
        """Detach from the model: it goes back to the plain dense path."""
        for mlp in self.mlps:
            mlp.token_router = None

    def state_dict(self):
        return {"mode": self.mode, "ema": self.ema.cpu(), "ema_ready": self.ema_ready.cpu(), "bias": self.bias.cpu(),
                "total_sel": self.total_sel.cpu(), "total_tokens": self.total_tokens, "steps": self.steps,
                "skipped_slices": self.skipped_slices}

    def load_state_dict(self, sd):
        self.ema.copy_(sd["ema"])
        self.ema_ready.copy_(sd["ema_ready"])
        self.bias.copy_(sd["bias"])
        self.total_sel.copy_(sd["total_sel"])
        self.total_tokens, self.steps, self.skipped_slices = sd["total_tokens"], sd["steps"], sd["skipped_slices"]
