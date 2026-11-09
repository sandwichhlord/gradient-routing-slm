"""Gated mode (`--routing_mode gated`, `--groups fixed|evolved`): soft k-winners-take-all over neuron groups.

In each gated layer (--gated_layers, default 2,3,4,5; other layers stay dense) the MLP hidden neurons are split into
n_groups groups. For every token:

    r_g(t)  = mean |h| over group g at token t / baseline_g        relative activity; baseline_g is the mean over the
                                                                   group's members of each neuron's running mean |h|
    s_g(t)  = beta * s_g(t-1) + (1 - beta) * r_g(t)                causal EMA along the sequence (beta = route_smooth),
                                                                   restarted at the sequence start and after <|endoftext|>
    score   = s_g(t) + bias_g                                      bias_g: DeepSeek-style balancing, nudged by gamma
                                                                   after each step towards a selection share of k / G
    gate_g  = sigmoid((score_g - theta) / tau)                     theta: midway between the k-th and (k+1)-th score
    h      <- h * gate (per neuron, from its group)

--gate_score share replaces r_g(t) with each group's share of the layer's activation: G * ||h_g(t)|| / sum_j ||h_j(t)||
(L2 norms; multiplied by G so the mean score is 1, the same scale as relative activity, which keeps tau meaningful).
It needs no running baseline, so the gate cannot be raised by inflating every activation.

Selected groups are the top k by score (gate > 0.5). No gradient masking or scaling is added. With gate_grad (the
default) the gate is differentiable: gradient flows through r_g(t) (and the smoothing) into the activations, so a group
can learn to raise its activity on tokens where its output helps. It does not flow through the baseline (a running
mean), the balancing bias or the threshold theta (both treated as constants). With gate_grad off, scores come from
h.detach() and the gate is a constant multiplier. With k = n_groups the (k+1)-th score is -inf, every gate is exactly 1
and the layer is the dense layer.

Groups:
  fixed    contiguous equal-size groups
  evolved  Hebbian co-activation + Leiden (as in model/assemblies.py), reduced or split to exactly n_groups. The first
           clustering (at cluster_start) installs the partition; later ones (every cluster_every) move at most
           max_move of the layer's neurons towards Leiden's proposal, highest gain in mean correlation first, and never
           shrink a group below min_size. Groups keep their IDs and can grow or shrink.
Both kinds train dense before cluster_start and share every other line of code.

Inference-only use (post-hoc groups on a dense model): GateRouter(..., apply=False) computes and records selections
without changing the forward pass.
"""
import csv
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from model.assemblies import LEFTOVER, CoactivationTracker, adjusted_rand_index, cluster_correlation

DOMAINS = ["story", "math", "code"]
GROUP_FIELDS = ["iter", "layer", "group", "size", "sel_frac", *[f"sel_{d}" for d in DOMAINS], "bias", "mean_gate"]
CTX_NAMES = ["number_words", "prose_digits", "qa_digits"]   # deconf.PROBE_PAIRS; qa_digits is the "format" comparison
LAYER_FIELDS = ["iter", "layer", "n_clusterings", "moved_last", "moved_total", "stability_ari", "size_min", "size_max",
                "largest_frac", *[f"ctx_{n}_{k}" for n in CTX_NAMES for k in ("cross", "within", "effect")]]


def parse_layers(spec, n_layer):
    layers = sorted({int(x) for x in str(spec).replace(" ", "").split(",") if x != ""})
    bad = [x for x in layers if not 0 <= x < n_layer]
    if bad:
        raise ValueError(f"gated_layers {bad} out of range for n_layer={n_layer}")
    return layers


def fixed_groups(hidden, n_groups):
    assert hidden % n_groups == 0, f"hidden={hidden} not divisible by n_groups={n_groups}"
    return np.repeat(np.arange(n_groups), hidden // n_groups)


# ------------------------------------------------------------------------------ partitions with exactly G groups
def _mean_corr_to_groups(corr, labels, G):
    """(n, G) mean correlation of each neuron with each group's members, excluding itself."""
    c = np.nan_to_num(np.array(corr, dtype=np.float64))
    np.fill_diagonal(c, 0.0)
    onehot = np.eye(G)[labels]
    sums = c @ onehot
    sizes = onehot.sum(axis=0)
    denom = np.maximum(sizes[None, :] - onehot, 1.0)  # own group: size - 1
    return sums / denom


def partition_to_n(corr, n_groups, knn=32, resolution=1.0, min_size=16, seed=0, dead=None):
    """Leiden partition reduced (merge the smallest group into its most correlated neighbour) or split (largest group,
    at the median of its leading eigenvector) to exactly n_groups. Dead neurons go to the smallest group."""
    n = corr.shape[0]
    c = np.nan_to_num(np.array(corr, dtype=np.float64))
    labels = cluster_correlation(c, knn, resolution, min_size, seed, dead=dead)
    alive = labels != LEFTOVER
    if alive.sum() < 2 * n_groups:
        return fixed_groups(n, n_groups) if n % n_groups == 0 else np.arange(n) % n_groups
    lab = np.unique(labels[alive], return_inverse=True)[1]
    sub = c[np.ix_(alive, alive)]
    while lab.max() + 1 > n_groups:
        G = lab.max() + 1
        sizes = np.bincount(lab, minlength=G)
        g = int(np.argmin(sizes))
        mc = _mean_corr_to_groups(sub, lab, G)[lab == g].mean(axis=0)
        mc[g] = -np.inf
        lab[lab == g] = int(np.argmax(mc))
        lab = np.unique(lab, return_inverse=True)[1]
    while lab.max() + 1 < n_groups:
        G = lab.max() + 1
        g = int(np.argmax(np.bincount(lab, minlength=G)))
        members = np.flatnonzero(lab == g)
        _, vecs = np.linalg.eigh(sub[np.ix_(members, members)])
        lead = vecs[:, -1]
        upper = members[lead > np.median(lead)]
        if len(upper) == 0 or len(upper) == len(members):
            upper = members[: len(members) // 2]
        lab[upper] = G
    out = np.empty(n, dtype=np.int64)
    out[alive] = lab
    if (~alive).any():
        out[~alive] = int(np.argmin(np.bincount(lab, minlength=n_groups)))
    return out


def match_to(proposal, current, G):
    """Rename proposal's groups to current's IDs by maximum overlap (Hungarian, full assignment of G to G)."""
    overlap = np.zeros((G, G), dtype=np.int64)
    np.add.at(overlap, (proposal, current), 1)
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    mapping = np.empty(G, dtype=np.int64)
    mapping[rows] = cols
    return mapping[proposal]


def capped_move(current, proposal, corr, max_frac, min_size, dead=None):
    """Move at most max_frac of the neurons from `current` towards `proposal` (both 0..G-1): only neurons the proposal
    puts elsewhere, largest gain in mean correlation first, only with positive gain, and never below min_size.
    Returns (new labels, number moved)."""
    G = int(max(current.max(), proposal.max())) + 1
    target = match_to(proposal, current, G)
    mc = _mean_corr_to_groups(corr, current, G)
    idx = np.arange(len(current))
    gain = mc[idx, target] - mc[idx, current]
    cand = np.flatnonzero((target != current) & (gain > 0) & (np.ones_like(current, bool) if dead is None else ~dead))
    cand = cand[np.argsort(-gain[cand], kind="stable")]
    budget = int(np.floor(max_frac * len(current)))
    sizes = np.bincount(current, minlength=G)
    new, moved = current.copy(), 0
    for i in cand:
        if moved >= budget:
            break
        if sizes[new[i]] - 1 < min_size:
            continue
        sizes[new[i]] -= 1
        sizes[target[i]] += 1
        new[i] = target[i]
        moved += 1
    return new, moved


# ------------------------------------------------------------------------------ causal smoothing
def smoothing_weights(idx, beta, eot):
    """(B, T, T) weights W with (W @ r)[t] = the causal EMA of r at t, restarted at position 0 and after each eot."""
    B, T = idx.shape
    reset = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
    reset[:, 0] = True
    reset[:, 1:] = idx[:, :-1] == eot
    pos = torch.arange(T, device=idx.device)
    seg = torch.where(reset, pos, torch.zeros_like(pos)).cummax(dim=1).values      # (B, T) start of each segment
    t, j = pos.view(1, T, 1), pos.view(1, 1, T)
    valid = (j <= t) & (j >= seg.unsqueeze(-1))
    power = torch.full((), beta, device=idx.device, dtype=torch.float32) ** (t - j).clamp_min(0).float()
    w = power * torch.where(j == seg.unsqueeze(-1), torch.ones((), device=idx.device), torch.full((), 1.0 - beta, device=idx.device))
    return torch.where(valid, w, torch.zeros((), device=idx.device))


# ------------------------------------------------------------------------------ router
class GateRouter:
    """Usage in training:
        router = GateRouter(model, layers=[2, 3, 4, 5], groups="evolved")
        router.set_batch(domains)            # (B, T) labels, logging only
        loss = model(x, y)[1]; loss.backward()
        router.step(optimizer, scaler)       # optimizer step, bias update, re-clustering when due
    The model calls router.begin_forward(idx) and router.gate(layer, h) itself.
    """

    def __init__(self, model, layers=(2, 3, 4, 5), n_groups=8, k=4, tau=0.1, gamma=0.001, smooth=0.9, groups="fixed",
                 cluster_start=500, cluster_every=500, knn=32, resolution=1.0, min_size=16, max_move=0.05, ema=0.99,
                 sample_tokens=512, seed=0, apply=True, eot=50256, gate_grad=True, score="relative"):
        assert score in ("relative", "share"), score
        assert groups in ("fixed", "evolved"), groups
        assert 1 <= k <= n_groups, f"gate_k={k} must be in [1, n_groups={n_groups}]"
        assert 0.0 <= smooth < 1.0, f"route_smooth={smooth} must be in [0, 1)"
        mlps = model.mlps()
        self.layers = list(layers)
        self.pos = {layer: i for i, layer in enumerate(self.layers)}
        self.hidden = mlps[0].n_blocks * mlps[0].block_dim
        self.G, self.k, self.tau, self.gamma, self.smooth = n_groups, k, tau, gamma, smooth
        self.groups, self.apply, self.eot, self.gate_grad, self.score = groups, apply, eot, gate_grad, score
        self.cluster_start, self.cluster_every = cluster_start, cluster_every
        self.knn, self.resolution, self.min_size, self.max_move, self.seed = knn, resolution, min_size, max_move, seed
        self.device = model.lm_head.weight.device
        L = len(self.layers)
        self.tracker = CoactivationTracker(len(mlps), self.hidden, ema, sample_tokens, seed, self.device)
        self.labels = [fixed_groups(self.hidden, n_groups) if groups == "fixed" else None for _ in range(L)]
        self.index = [None] * L
        self.member = [None] * L         # (hidden, G) one-hot membership: group sums are a matmul, which is
        self.sizes = [None] * L          # deterministic on GPU (index_add_ uses atomics, so it is not)
        for i, lab in enumerate(self.labels):
            if lab is not None:
                self._install(i, lab)
        self.bias = torch.zeros(L, n_groups, device=self.device)
        self.steps, self.n_clusterings = 0, 0
        self.moved_last, self.moved_total, self.stability = [0] * L, [0] * L, [None] * L
        self.domains = None
        self.record = True
        self.last_selected = [None] * L    # (B, T, G) bool, from the last forward
        self.last_gate = [None] * L        # (B, T, G) float
        self._idx, self._W, self._training = None, None, False
        self._step_sel = torch.zeros(L, n_groups, device=self.device)
        self._step_passes = torch.zeros(L, device=self.device)
        self.win_sel = torch.zeros(len(DOMAINS), L, n_groups, device=self.device)
        self.win_tokens = torch.zeros(len(DOMAINS), device=self.device)
        self.win_gate = torch.zeros(L, n_groups, device=self.device)
        self.win_passes = torch.zeros(L, device=self.device)
        model.gate_router = self
        for layer in self.layers:
            mlps[layer].gate_router, mlps[layer].layer_idx = self, layer
        self._model = model

    # ------------------------------------------------------------------ groups
    def _install(self, i, labels):
        labels = np.asarray(labels, dtype=np.int64)
        assert labels.shape == (self.hidden,) and labels.min() >= 0 and labels.max() < self.G, "need labels 0..G-1"
        self.labels[i] = labels
        self.index[i] = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        self.member[i] = torch.nn.functional.one_hot(self.index[i], self.G).float()
        self.sizes[i] = torch.bincount(self.index[i], minlength=self.G).float().clamp_min(1.0)

    def set_labels(self, layer, labels):
        self._install(self.pos[layer], labels)

    def set_neuron_ema(self, layer, values):
        self.tracker.neuron_ema[layer].copy_(torch.as_tensor(values, dtype=torch.float32))
        self.tracker.ready[layer] = True

    def active(self, i):
        return self.labels[i] is not None and self.steps >= self.cluster_start and self.tracker.ready[self.layers[i]]

    # ------------------------------------------------------------------ forward
    def set_batch(self, domains):
        self.domains = domains

    def begin_forward(self, idx):
        self._idx, self._W = idx, None
        self._training = self._model.training and torch.is_grad_enabled()

    def _smoothing(self):
        if self._W is None:
            self._W = smoothing_weights(self._idx, self.smooth, self.eot)
        return self._W

    def scores(self, i, h):
        """(B, T, G) smoothed relative activity plus bias, in float32."""
        layer = self.layers[i]
        member, sizes = self.member[i], self.sizes[i]
        hf = (h if self.gate_grad else h.detach()).float()
        if self.score == "share":
            sq = (hf * hf) @ member
            norms = (sq + 1e-12).sqrt()                       # eps: finite gradient for a silent group
            r = self.G * norms / norms.sum(dim=-1, keepdim=True)
        else:
            activity = (hf.abs() @ member) / sizes
            baseline = (self.tracker.neuron_ema[layer].detach() @ member) / sizes
            r = activity / baseline.clamp_min(1e-8)
        if self.smooth > 0:
            r = self._smoothing() @ r
        return r + self.bias[i]

    def soft_kwta(self, s):
        """(gate, selected) for scores s (..., G)."""
        if self.k >= self.G:
            return torch.ones_like(s), torch.ones_like(s, dtype=torch.bool)
        top = s.detach().topk(self.k + 1, dim=-1).values          # the threshold is a constant for the gradient
        kth = top[..., self.k - 1:self.k]
        theta = 0.5 * (kth + top[..., self.k:self.k + 1])
        return torch.sigmoid((s - theta) / self.tau), s.detach() >= kth

    def gate(self, layer, h):
        """Called by BlockMLP for gated layers. Returns the (B, T, hidden) multiplier, or None for the dense path."""
        i = self.pos[layer]
        out = None
        if self.active(i):
            with torch.autocast(device_type=h.device.type, enabled=False):
                gate, sel = self.soft_kwta(self.scores(i, h))
            if self.record:
                self.last_selected[i], self.last_gate[i] = sel, gate.detach()
            if self._training:
                self._note(i, gate.detach(), sel)
            if self.apply:
                out = (gate @ self.member[i].T).to(h.dtype)    # exact copy of each group's gate to its neurons
        if self._training:
            self.tracker.update(layer, h)   # after the gate: scores never see the current batch's statistics
        return out

    def _note(self, i, gate, sel):
        flat = sel.reshape(-1, self.G).float()
        self._step_sel[i] += flat.mean(dim=0)
        self._step_passes[i] += 1
        self.win_gate[i] += gate.reshape(-1, self.G).mean(dim=0)
        self.win_passes[i] += 1
        if self.domains is not None:
            d = self.domains.reshape(-1)
            self.win_sel[:, i] += torch.nn.functional.one_hot(d, len(DOMAINS)).float().T @ flat
            if i == 0:
                self.win_tokens += torch.bincount(d, minlength=len(DOMAINS)).float()

    # ------------------------------------------------------------------ optimizer step
    def step(self, optimizer, scaler=None):
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        self.after_step()

    def after_step(self):
        busy = self._step_passes > 0
        if busy.any():
            frac = self._step_sel / self._step_passes.clamp_min(1).unsqueeze(1)
            delta = torch.where(frac > self.k / self.G, -self.gamma, self.gamma)
            self.bias += torch.where(busy.unsqueeze(1), delta, torch.zeros_like(delta))
        self._step_sel.zero_()
        self._step_passes.zero_()
        self.steps += 1
        if (self.groups == "evolved" and self.steps >= self.cluster_start
                and (self.steps - self.cluster_start) % self.cluster_every == 0):
            self.recluster()

    def recluster(self):
        self.n_clusterings += 1
        for i, layer in enumerate(self.layers):
            corr = self.tracker.corr[layer].cpu().numpy()
            dead = self.tracker.dead(layer)
            proposal = partition_to_n(corr, self.G, self.knn, self.resolution, self.min_size, self.seed + layer, dead)
            prev = self.labels[i]
            if prev is None:
                new, moved = proposal, self.hidden
            else:
                new, moved = capped_move(prev, proposal, corr, self.max_move, self.min_size, dead)
                self.stability[i] = adjusted_rand_index(new, prev)
            self.moved_last[i] = moved
            self.moved_total[i] += moved if prev is not None else 0
            self._install(i, new)

    # ------------------------------------------------------------------ logging
    def layer_stats(self, i):
        sizes = np.bincount(self.labels[i], minlength=self.G) if self.labels[i] is not None else np.zeros(self.G, int)
        return {"n_clusterings": self.n_clusterings, "moved_last": self.moved_last[i], "moved_total": self.moved_total[i],
                "stability_ari": "" if self.stability[i] is None else f"{self.stability[i]:.4f}",
                "size_min": int(sizes.min()), "size_max": int(sizes.max()),
                "largest_frac": f"{sizes.max() / self.hidden:.4f}"}

    def log(self, it, path, layer_extra=None):
        """gating.csv: one row per (layer, group) with selection shares since the previous log (overall and per domain
        label), bias and mean gate. gating_layers.csv: one row per layer (moves, stability, sizes, layer_extra)."""
        sel, tok = self.win_sel.cpu().numpy(), self.win_tokens.cpu().numpy()
        gate = (self.win_gate / self.win_passes.clamp_min(1).unsqueeze(1)).cpu().numpy()
        bias = self.bias.cpu().numpy()
        rows, lrows = [], []
        for i, layer in enumerate(self.layers):
            sizes = np.bincount(self.labels[i], minlength=self.G) if self.labels[i] is not None else np.zeros(self.G, int)
            for g in range(self.G):
                per = {f"sel_{d}": f"{sel[j, i, g] / tok[j]:.4f}" for j, d in enumerate(DOMAINS) if tok[j] > 0}
                total = sel[:, i, g].sum() / tok.sum() if tok.sum() > 0 else float("nan")
                rows.append({"iter": it, "layer": layer, "group": g, "size": int(sizes[g]), "sel_frac": f"{total:.4f}",
                             **per, "bias": f"{bias[i, g]:.5f}", "mean_gate": f"{gate[i, g]:.4f}"})
            lrows.append({"iter": it, "layer": layer, **self.layer_stats(i), **((layer_extra or {}).get(layer, {}))})
        for p, fields, data in [(path, GROUP_FIELDS, rows), (path.replace(".csv", "_layers.csv"), LAYER_FIELDS, lrows)]:
            new = not os.path.exists(p)
            with open(p, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerows(data)
        self.win_sel.zero_()
        self.win_tokens.zero_()
        self.win_gate.zero_()
        self.win_passes.zero_()

    def summary(self):
        if not any(self.active(i) for i in range(len(self.layers))):
            return f"gated ({self.groups}): dense until step {self.cluster_start} (step {self.steps})"
        sizes = [sorted(np.bincount(lab, minlength=self.G).tolist()) if lab is not None else [] for lab in self.labels]
        return (f"gated ({self.groups}): k={self.k}/{self.G} tau={self.tau} smooth={self.smooth} | clusterings "
                f"{self.n_clusterings} | moved last {self.moved_last} | sizes (min, max) "
                f"{[(s[0], s[-1]) if s else None for s in sizes]}")

    def assignments(self):
        return {"groups": self.groups, "n_groups": self.G, "k": self.k, "tau": self.tau, "smooth": self.smooth,
                "steps": self.steps, "n_clusterings": self.n_clusterings,
                "layers": {str(layer): None if self.labels[i] is None else [int(x) for x in self.labels[i]]
                           for i, layer in enumerate(self.layers)},
                "neuron_ema": {str(layer): [float(x) for x in self.tracker.neuron_ema[layer].cpu()] for layer in self.layers}}

    def write_assignments(self, path):
        with open(path, "w") as f:
            json.dump(self.assignments(), f)

    def write_summary(self, path):
        lines = [f"mode: gated, groups={self.groups}, layers={self.layers}, n_groups={self.G}, k={self.k}, tau={self.tau}, "
                 f"route_smooth={self.smooth}, gamma={self.gamma}, gate_grad={self.gate_grad}, gate_score={self.score}",
                 f"steps: {self.steps}, gating from step {self.cluster_start}, clusterings: {self.n_clusterings}"]
        for i, layer in enumerate(self.layers):
            sizes = np.bincount(self.labels[i], minlength=self.G).tolist() if self.labels[i] is not None else None
            lines.append(f"layer {layer}: sizes {sizes}, moved total {self.moved_total[i]}, bias "
                         f"{[round(float(b), 3) for b in self.bias[i].cpu()]}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def remove(self):
        self._model.gate_router = None
        for layer in self.layers:
            self._model.mlps()[layer].gate_router = None

    def state_dict(self):
        """Correlation matrices are left out to keep checkpoints small; they refill within about 100 steps."""
        return {"labels": [None if lab is None else lab.tolist() for lab in self.labels], "bias": self.bias.cpu(),
                "steps": self.steps, "n_clusterings": self.n_clusterings, "moved_last": self.moved_last,
                "moved_total": self.moved_total, "stability": self.stability,
                "neuron_ema": [self.tracker.neuron_ema[layer].cpu() for layer in self.layers],
                "ready": [self.tracker.ready[layer] for layer in self.layers]}

    def load_state_dict(self, sd):
        for i, lab in enumerate(sd["labels"]):
            if lab is not None:
                self._install(i, lab)
            layer = self.layers[i]
            self.tracker.neuron_ema[layer].copy_(sd["neuron_ema"][i])
            self.tracker.ready[layer] = sd["ready"][i]
        self.bias.copy_(sd["bias"])
        self.steps, self.n_clusterings = sd["steps"], sd["n_clusterings"]
        self.moved_last, self.moved_total, self.stability = list(sd["moved_last"]), list(sd["moved_total"]), list(sd["stability"])


def router_from_config(model, c, apply=True):
    """GateRouter built from a train.py config dict."""
    return GateRouter(model, layers=parse_layers(c["gated_layers"], c["n_layer"]), n_groups=c["n_groups"], k=c["gate_k"],
                      tau=c["gate_tau"], gamma=c["route_gamma"], smooth=c["route_smooth"], groups=c["groups"],
                      cluster_start=c["cluster_start"], cluster_every=c["cluster_every"], knn=c["knn"],
                      resolution=c["resolution"], min_size=c["min_size"], max_move=c["max_move"], ema=c["route_ema"],
                      sample_tokens=c["assembly_sample_tokens"], seed=c["seed"], apply=apply,
                      gate_grad=bool(c.get("gate_grad", True)), score=c.get("gate_score", "relative"))


# ------------------------------------------------------------------------------ post-hoc groups (dense models)
@torch.no_grad()
def posthoc_groups(model, batches, layers, n_groups=8, knn=32, resolution=1.0, min_size=16, seed=0, sample_tokens=512):
    """Co-activation + Leiden on held-out batches, reduced to n_groups per layer. Returns {layer: (labels, neuron_ema)}.
    neuron_ema is the plain mean |h| per neuron over the batches (the baseline for relative activity)."""
    mlps = model.mlps()
    hidden = mlps[0].n_blocks * mlps[0].block_dim
    tracker = CoactivationTracker(len(mlps), hidden, decay=0.99, sample_tokens=sample_tokens, seed=seed,
                                  device=model.lm_head.weight.device)
    sums = {layer: torch.zeros(hidden, device=model.lm_head.weight.device) for layer in layers}
    counts = {layer: 0 for layer in layers}

    def hook(layer):
        def f(_m, _i, out):
            tracker.update(layer, out)
            flat = out.detach().float().abs().reshape(-1, hidden)
            sums[layer] += flat.sum(dim=0)
            counts[layer] += flat.shape[0]
        return f

    handles = [mlps[layer].gelu.register_forward_hook(hook(layer)) for layer in layers]
    was_training = model.training
    model.eval()
    for x in batches:
        model(x)
    for h in handles:
        h.remove()
    model.train(was_training)
    return {layer: (partition_to_n(tracker.corr[layer].cpu().numpy(), n_groups, knn, resolution, min_size, seed + layer,
                                   tracker.dead(layer)), (sums[layer] / max(counts[layer], 1)).cpu().numpy())
            for layer in layers}


# ------------------------------------------------------------------------------ context sensitivity
def jaccard_distance_matrix(a, b):
    """(na, nb) Jaccard distances between rows of boolean matrices a (na, G) and b (nb, G)."""
    a, b = a.float(), b.float()
    inter = a @ b.T
    union = a.sum(1, keepdim=True) + b.sum(1).unsqueeze(0) - inter
    return 1.0 - inter / union.clamp_min(1.0)


@torch.no_grad()
def selected_sets(model, router, tokens, batch_size=32):
    """{layer: (N, T, G) bool} selected groups for every token of `tokens` (N, T)."""
    out = {layer: [] for layer in router.layers}
    was_training, record = model.training, router.record
    model.eval()
    router.record = True
    dev = router.device
    for s in range(0, len(tokens), batch_size):
        x = torch.as_tensor(tokens[s:s + batch_size], device=dev)
        model(x)
        for i, layer in enumerate(router.layers):
            out[layer].append(router.last_selected[i].cpu())
    model.train(was_training)
    router.record = record
    return {layer: torch.cat(v) for layer, v in out.items()}


def context_sensitivity(model, router, probes, pairs, max_occ=50, min_occ=2, seed=0):
    """For each (name, context_a, context_b[, kind]) and gated layer: the Jaccard distance between selected-group sets of the
    SAME token id in context a vs context b (cross), and between two occurrences in the same context (within, the noise
    floor), each averaged over occurrence pairs, then over token ids. effect = cross - within.
    Returns {name: {layer: {"cross", "within", "effect", "n_tokens"}}}."""
    sets = {ctx: (selected_sets(model, router, toks), toks, mask) for ctx, (toks, mask) in probes.items()}
    rng = np.random.default_rng(seed)
    result = {}
    for name, ca, cb, *_ in pairs:
        sel_a, tok_a, mask_a = sets[ca]
        sel_b, tok_b, mask_b = sets[cb]
        ids_a, ids_b = tok_a[mask_a], tok_b[mask_b]
        common = sorted(set(np.unique(ids_a).tolist()) & set(np.unique(ids_b).tolist()))
        per_layer = {}
        for layer in router.layers:
            fa, fb = sel_a[layer][torch.as_tensor(mask_a)], sel_b[layer][torch.as_tensor(mask_b)]
            cross, within = [], []
            for v in common:
                ia, ib = np.flatnonzero(ids_a == v), np.flatnonzero(ids_b == v)
                if len(ia) < min_occ or len(ib) < min_occ:
                    continue
                ia = rng.permutation(ia)[:max_occ]
                ib = rng.permutation(ib)[:max_occ]
                A, B = fa[torch.as_tensor(ia)], fb[torch.as_tensor(ib)]
                cross.append(float(jaccard_distance_matrix(A, B).mean()))
                w = []
                for M in (A, B):
                    d = jaccard_distance_matrix(M, M)
                    n = len(M)
                    w.append(float((d.sum() - d.diagonal().sum()) / (n * (n - 1))))
                within.append(float(np.mean(w)))
            per_layer[layer] = {"cross": float(np.mean(cross)) if cross else float("nan"),
                                "within": float(np.mean(within)) if within else float("nan"),
                                "effect": float(np.mean(cross) - np.mean(within)) if cross else float("nan"),
                                "n_tokens": len(cross)}
        result[name] = per_layer
    return result
