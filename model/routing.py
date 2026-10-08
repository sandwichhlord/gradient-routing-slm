"""Activation-based gradient routing over the MLP blocks of a GPT (see README "Method").

Per training micro-batch, for every layer l:
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
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["iter", "layer", "block", "freq_ema", "selected_count", "selected_frac", "energy"])
            for layer in range(self.n_layer):
                for b in range(self.n_blocks):
                    c = int(counts[layer, b])
                    w.writerow([it, layer, b, f"{freq[layer, b]:.5f}", c, f"{c / max(self.steps, 1):.5f}",
                                f"{energy[layer, b]:.5f}"])

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
