"""Routing v2 (`--routing_mode assembly`): neuron assemblies from Hebbian co-activation, with graded plasticity.

Per layer, an EMA of the correlation matrix of the MLP hidden activations is tracked from a small token sample. Every so
often a kNN graph of that matrix is clustered (Leiden) into assemblies. Each token then gives each assembly a plasticity
between floor and 1 according to how active it is relative to its usual level, and BlockMLP scales the gradient of both
projections by it. The forward output stays equal to the dense model. Labels are only used for logging.

This is Hebbian co-activation clustering. Cannistraci-Hebb Training is related inspiration, not the method.

Assembly IDs are >= 0. LEFTOVER (-1) marks dead or constant neurons: they are left out of the graph and of the plasticity
competition, and they keep plasticity 1 (as in dense training). A layer whose clustering is degenerate (see
partition_health) falls back to dense training until a later clustering recovers.
"""
import csv
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DOMAINS = ["story", "math", "code"]
LEFTOVER = -1
DEAD_VARIANCE = 1e-8          # a neuron with less activation variance than this is dead or constant
DEGENERATE_LARGEST = 0.5      # degenerate if the largest assembly holds more than this fraction of the layer's neurons
DEGENERATE_MIN_GROUPS = 4     # degenerate if there are fewer assemblies than this
SPLIT_MERGE_SHARE = 0.25      # how much of a group must move for that move to count as a split or merge
P_HIST_BINS = 10
CSV_FIELDS = ["iter", "kind", "layer", "clustering", "n_assemblies", "size_min", "size_median", "size_max", "largest_frac",
              "frac_below_2x_min", "n_dead", "dead_handling", "splits", "merges", "stability_ari", "degenerate",
              "dense_fallback", "mean_p", *[f"p_hist_{i}" for i in range(P_HIST_BINS)], "assembly", "size", "ceiling",
              *[f"plast_{d}" for d in DOMAINS]]
DEAD_HANDLING = "excluded from graph, assigned to leftover assembly -1 with plasticity 1"


# ------------------------------------------------------------------------------ clustering
def knn_edges(corr, knn):
    """Keep each neuron's top `knn` positive correlations and symmetrize. Returns (edges (E, 2), weights (E,))."""
    n = corr.shape[0]
    c = np.nan_to_num(np.array(corr, dtype=np.float64))
    np.fill_diagonal(c, -np.inf)
    k = min(knn, n - 1)
    top = np.argpartition(-c, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = top.ravel()
    keep = c[rows, cols] > 0
    a, b = rows[keep], cols[keep]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    codes = np.unique(lo * n + hi)
    lo, hi = codes // n, codes % n
    return np.stack([lo, hi], axis=1), np.maximum(c[lo, hi], c[hi, lo])


def _leiden(n, edges, weights, resolution, seed):
    import igraph
    import leidenalg
    g = igraph.Graph(n=n, edges=edges.tolist(), edge_attrs={"weight": weights.tolist()})
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed)
    return np.asarray(part.membership, dtype=np.int64)


def _louvain(n, edges, weights, resolution, seed):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_weighted_edges_from((int(a), int(b), float(w)) for (a, b), w in zip(edges, weights))
    labels = np.zeros(n, dtype=np.int64)
    for i, community in enumerate(nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)):
        labels[list(community)] = i
    return labels


def merge_small(labels, corr, min_size):
    """Merge communities below min_size into their most correlated neighbour, smallest first. Returns labels 0..K-1."""
    labels = np.unique(labels, return_inverse=True)[1]
    c = np.nan_to_num(np.array(corr, dtype=np.float64))
    while True:
        counts = np.bincount(labels)
        present = np.flatnonzero(counts)
        small = present[counts[present] < min_size]
        if len(small) == 0 or len(present) == 1:
            break
        g = small[np.argmin(counts[small])]
        mean_corr = np.bincount(labels, weights=c[labels == g].mean(axis=0), minlength=len(counts))
        mean_corr[present] /= counts[present]
        others = present[present != g]
        labels[labels == g] = others[np.argmax(mean_corr[others])]
    return np.unique(labels, return_inverse=True)[1]


def cluster_correlation(corr, knn=32, resolution=1.0, min_size=16, seed=0, backend="auto", dead=None):
    """Partition neurons from a correlation matrix. Returns labels 0..K-1 with no group below min_size. Neurons flagged
    in `dead` (boolean array) are left out of the graph and get the label LEFTOVER."""
    n = corr.shape[0]
    alive = np.ones(n, dtype=bool) if dead is None else ~np.asarray(dead, dtype=bool)
    out = np.full(n, LEFTOVER, dtype=np.int64)
    if alive.sum() < 2:
        out[alive] = 0
        return out
    sub = np.asarray(corr)[np.ix_(alive, alive)]
    edges, weights = knn_edges(sub, knn)
    if backend == "auto":
        try:
            import igraph  # noqa: F401
            import leidenalg  # noqa: F401
            backend = "leiden"
        except ImportError:
            print("[assembly] leidenalg/igraph not available, falling back to Louvain (networkx)", flush=True)
            backend = "louvain"
    labels = (_leiden if backend == "leiden" else _louvain)(int(alive.sum()), edges, weights, resolution, seed)
    out[alive] = merge_small(labels, sub, min_size)
    return out


def match_labels(new, prev, next_id):
    """Rename the groups in `new` to the IDs of `prev` by maximum neuron overlap (Hungarian). Unmatched groups get fresh
    IDs from next_id, and LEFTOVER stays LEFTOVER. Returns (labels, next_id)."""
    out = np.full_like(new, LEFTOVER)
    new_groups = np.unique(new[new != LEFTOVER])
    prev_groups = np.unique(prev[prev != LEFTOVER])
    if len(new_groups) == 0:
        return out, next_id
    mapping = {}
    if len(prev_groups) > 0:
        overlap = np.zeros((len(new_groups), len(prev_groups)), dtype=np.int64)
        for i, g in enumerate(new_groups):
            members = prev[(new == g) & (prev != LEFTOVER)]
            overlap[i] = np.bincount(np.searchsorted(prev_groups, members), minlength=len(prev_groups))
        rows, cols = linear_sum_assignment(overlap, maximize=True)
        mapping = {new_groups[i]: prev_groups[j] for i, j in zip(rows, cols) if overlap[i, j] > 0}
    for g in new_groups:
        if g not in mapping:
            mapping[g] = next_id
            next_id += 1
        out[new == g] = mapping[g]
    return out, next_id


def adjusted_rand_index(a, b):
    """Adjusted Rand index of two partitions: 1 for identical, about 0 for unrelated."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    table = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).astype(np.float64)
    table = table.reshape(a.max() + 1, b.max() + 1)

    def comb2(x):
        return x * (x - 1) / 2

    sum_ij = comb2(table).sum()
    sum_a, sum_b = comb2(table.sum(axis=1)).sum(), comb2(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(float(len(a)))
    maximum = (sum_a + sum_b) / 2
    return 1.0 if maximum == expected else float((sum_ij - expected) / (maximum - expected))


def count_splits_merges(new, prev):
    """A split: an old assembly gives at least SPLIT_MERGE_SHARE of its neurons to each of 2+ new ones. A merge: a new
    assembly takes at least SPLIT_MERGE_SHARE of its neurons from each of 2+ old ones."""
    both = (new != LEFTOVER) & (prev != LEFTOVER)
    if not both.any():
        return 0, 0
    n_groups, n_idx = np.unique(new[both], return_inverse=True)
    p_groups, p_idx = np.unique(prev[both], return_inverse=True)
    table = np.bincount(n_idx * len(p_groups) + p_idx, minlength=len(n_groups) * len(p_groups)).reshape(len(n_groups), -1)
    prev_sizes = np.array([(prev == g).sum() for g in p_groups], dtype=np.float64)
    new_sizes = np.array([(new == g).sum() for g in n_groups], dtype=np.float64)
    splits = int(((table / prev_sizes >= SPLIT_MERGE_SHARE).sum(axis=0) >= 2).sum())
    merges = int(((table / new_sizes[:, None] >= SPLIT_MERGE_SHARE).sum(axis=1) >= 2).sum())
    return splits, merges


def partition_health(labels, min_size):
    """Health of one layer's partition. LEFTOVER neurons count towards the layer's size but are not an assembly."""
    real = labels[labels != LEFTOVER]
    sizes = np.bincount(np.unique(real, return_inverse=True)[1]) if len(real) else np.zeros(0, dtype=np.int64)
    n = len(sizes)
    largest = float(sizes.max() / len(labels)) if n else 0.0
    return {"n_assemblies": n, "size_min": int(sizes.min()) if n else 0, "size_median": float(np.median(sizes)) if n else 0.0,
            "size_max": int(sizes.max()) if n else 0, "largest_frac": largest,
            "frac_below_2x_min": float(sizes[sizes < 2 * min_size].sum() / len(labels)),
            "n_dead": int((labels == LEFTOVER).sum()),
            "degenerate": bool(largest > DEGENERATE_LARGEST or n < DEGENERATE_MIN_GROUPS)}


# ------------------------------------------------------------------------------ plasticity formulas
def sigmoid_plasticity(r, floor, ceiling, alpha):
    return floor + (ceiling - floor) * torch.sigmoid(alpha * (r - 1.0))


def rank_plasticity(r, floor, ceiling):
    """r is (..., G). Within each token the most active assembly gets its ceiling and the least active gets floor. With
    fewer than 2 assemblies there is nothing to rank, so plasticity is the ceiling."""
    G = r.shape[-1]
    if G < 2:
        return ceiling * torch.ones_like(r)
    rank = r.argsort(dim=-1, descending=True).argsort(dim=-1).to(r.dtype)
    return floor + (ceiling - floor) * (1.0 - rank / (G - 1))


# ------------------------------------------------------------------------------ co-activation tracking
class CoactivationTracker:
    """Per layer: EMAs of the neuron correlation matrix (from a token sample), of each neuron's mean |h| and of each
    neuron's activation variance."""

    def __init__(self, n_layer, hidden, decay=0.99, sample_tokens=512, seed=0, device="cpu"):
        self.decay, self.sample_tokens = decay, sample_tokens
        self.corr = [torch.zeros(hidden, hidden, device=device) for _ in range(n_layer)]
        self.neuron_ema = [torch.zeros(hidden, device=device) for _ in range(n_layer)]
        self.variance = [torch.zeros(hidden, device=device) for _ in range(n_layer)]
        self.ready = [False] * n_layer
        # a private generator, so the tracker never touches the random streams of the training run
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self, n_tokens):
        if n_tokens <= self.sample_tokens:
            return torch.arange(n_tokens)
        return torch.randperm(n_tokens, generator=self.gen)[:self.sample_tokens]

    @torch.no_grad()
    def update(self, layer, h):
        """h is (..., hidden). Autocast is off here: fp16 would lose too much precision in the correlation."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.detach().float()
            flat = h.view(-1, h.shape[-1])
            mean_abs = flat.abs().mean(dim=0)
            x = flat[self.sample(flat.shape[0]).to(h.device)]
            x = x - x.mean(dim=0, keepdim=True)
            var = x.pow(2).sum(dim=0) / max(x.shape[0] - 1, 1)
            std = var.sqrt().unsqueeze(0)
            live = var > DEAD_VARIANCE
            z = torch.where(live.unsqueeze(0), x / std.clamp_min(1e-8), torch.zeros_like(x))
            c = z.T @ z / max(x.shape[0] - 1, 1)
        if self.ready[layer]:
            d = self.decay
            self.corr[layer].mul_(d).add_((1.0 - d) * c)
            self.neuron_ema[layer].mul_(d).add_((1.0 - d) * mean_abs)
            self.variance[layer].mul_(d).add_((1.0 - d) * var)
        else:
            self.corr[layer].copy_(c)
            self.neuron_ema[layer].copy_(mean_abs)
            self.variance[layer].copy_(var)
            self.ready[layer] = True

    def dead(self, layer):
        return (self.variance[layer] < DEAD_VARIANCE).cpu().numpy()


# ------------------------------------------------------------------------------ router
class AssemblyRouter:
    """Trains dense until cluster_start, then scales gradients per token and per neuron by plasticity P.

        r(t, g) = mean |h| over assembly g at token t / baseline_g
        rank     p = floor + (ceiling_g - floor) * (1 - rank(r) / (G - 1))       (default; rank 0 is the most active)
        sigmoid  p = floor + (ceiling_g - floor) * sigmoid(alpha * (r - 1))

    baseline_g is the mean over the assembly's current members of each neuron's recent activity (a per-neuron EMA), so
    after a re-clustering every assembly's baseline comes from its members, whatever ID it was matched to. ceiling_g is 1
    without --consolidate. With it, the ceiling is 1 / (1 + beta * age), never below floor, where age counts consecutive
    clusterings in which the assembly kept a membership overlap (Jaccard) above 0.8 with itself.

    Usage:
        router = AssemblyRouter(model)
        router.log_path = "out/assemblies.csv"   # optional
        router.set_batch(domains)                # (B, T) labels for logging only, or None
        loss = model(x, y)[1]; loss.backward()
        router.step(optimizer, scaler)           # optimizer step, then re-clusters when due
        scaler.update()
    """

    def __init__(self, model, cluster_start=500, cluster_every=500, knn=32, resolution=1.0, min_size=16, floor=0.1,
                 alpha=4.0, plasticity="rank", ema=0.99, consolidate=False, beta=0.5, sample_tokens=512, seed=0):
        assert 0.0 < floor <= 1.0, f"floor={floor} must be in (0, 1]: no neuron may ever have zero plasticity"
        assert plasticity in ("rank", "sigmoid"), plasticity
        self.mlps = model.mlps()
        self.n_layer, self.hidden = len(self.mlps), self.mlps[0].n_blocks * self.mlps[0].block_dim
        self.cluster_start, self.cluster_every = cluster_start, cluster_every
        self.knn, self.resolution, self.min_size, self.seed = knn, resolution, min_size, seed
        self.floor, self.alpha, self.plasticity_kind = floor, alpha, plasticity
        self.consolidate, self.beta = consolidate, beta
        self.device = model.lm_head.weight.device
        self.tracker = CoactivationTracker(self.n_layer, self.hidden, ema, sample_tokens, seed, self.device)
        self.log_path = None
        n = self.n_layer
        self.labels = [None] * n          # (hidden,) numpy assembly IDs
        self.index = [None] * n           # (hidden,) tensor of positions 0..G-1 (LEFTOVER has a position too)
        self.real_pos = [None] * n        # positions of real assemblies, without LEFTOVER
        self.ids = [None] * n             # position -> assembly ID
        self.sizes = [None] * n           # float tensor, by position
        self.ceiling = [None] * n         # float tensor, by real position
        self.age = [{} for _ in range(n)]
        self.next_id = [0] * n
        self.stability = [None] * n       # adjusted Rand index against the previous clustering
        self.health = [None] * n          # partition_health plus splits and merges
        self.degenerate = [False] * n     # True: this layer trains dense until a later clustering recovers
        self.last_P = [None] * n          # (B, T, hidden) of the last plastic forward
        self.domains = None
        self.steps = 0
        self.n_clusterings = 0
        self.dom_plast = torch.zeros(len(DOMAINS), n, self.hidden, device=self.device)
        self.dom_count = torch.zeros(len(DOMAINS), n, device=self.device)
        self.p_hist = torch.zeros(n, P_HIST_BINS, device=self.device)    # since the previous log
        # mean p: per layer, [sum of mean p, number of forwards], with p = 1 whenever a layer trained dense
        self.p_total = np.zeros((n, 2))
        self.p_routed = np.zeros((n, 2))  # the same, counted only after the first clustering
        for layer, mlp in enumerate(self.mlps):
            mlp.assembly_router, mlp.layer_idx = self, layer

    def set_batch(self, domains):
        """Domain labels (B, T) of the coming forward, used for logging only."""
        self.domains = domains

    def _note_p(self, layer, mean_p):
        self.p_total[layer] += (mean_p, 1)
        if self.labels[layer] is not None:
            self.p_routed[layer] += (mean_p, 1)

    def plasticity(self, layer, h):
        """Called by BlockMLP with detached h. Returns the (B, T, hidden) multiplier P, or None for the plain dense path
        (before the first clustering, or in a degenerate layer)."""
        self.tracker.update(layer, h)
        if self.labels[layer] is None or self.degenerate[layer]:
            self._note_p(layer, 1.0)
            return None
        idx, sizes, real = self.index[layer], self.sizes[layer], self.real_pos[layer]
        habs = h.float().abs()
        activity = torch.zeros(*habs.shape[:-1], len(sizes), device=h.device).index_add_(-1, idx, habs) / sizes
        baseline = torch.zeros(len(sizes), device=h.device).index_add_(0, idx, self.tracker.neuron_ema[layer]) / sizes
        r = (activity / baseline.clamp_min(1e-8)).index_select(-1, real)
        if self.plasticity_kind == "rank":
            p_real = rank_plasticity(r, self.floor, self.ceiling[layer])
        else:
            p_real = sigmoid_plasticity(r, self.floor, self.ceiling[layer], self.alpha)
        p = torch.ones(*habs.shape[:-1], len(sizes), device=h.device)
        p[..., real] = p_real
        P = p[..., idx]
        self.last_P[layer] = P
        s = self.tracker.sample(P.shape[0] * P.shape[1]).to(h.device)
        sampled = P.view(-1, self.hidden)[s]
        self.p_hist[layer] += torch.histc(sampled, bins=P_HIST_BINS, min=0.0, max=1.0)
        self._note_p(layer, float(P.mean()))
        if self.domains is not None:
            d = self.domains.view(-1)[s]
            self.dom_plast[:, layer].index_add_(0, d, sampled)
            self.dom_count[:, layer] += torch.bincount(d, minlength=len(DOMAINS)).float()
        return P

    def step(self, optimizer, scaler=None):
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        self.after_step()

    def after_step(self):
        self.steps += 1
        if self.steps >= self.cluster_start and (self.steps - self.cluster_start) % self.cluster_every == 0:
            self.recluster()

    # ------------------------------------------------------------------ clustering
    def recluster(self):
        self.n_clusterings += 1
        for layer in range(self.n_layer):
            corr = self.tracker.corr[layer].cpu().numpy()
            new = cluster_correlation(corr, self.knn, self.resolution, self.min_size, self.seed + layer,
                                      dead=self.tracker.dead(layer))
            self.set_labels(layer, new)
            self._report_clustering(layer)

    def set_labels(self, layer, new):
        """Install a partition: keep IDs stable against the previous one, then update health, ages and ceilings."""
        prev = self.labels[layer]
        if prev is None:
            labels = new.copy()
            self.next_id[layer] = int(new.max()) + 1 if (new != LEFTOVER).any() else 0
            self.stability[layer], splits, merges = None, 0, 0
        else:
            labels, self.next_id[layer] = match_labels(new, prev, self.next_id[layer])
            self.stability[layer] = adjusted_rand_index(labels, prev)
            splits, merges = count_splits_merges(labels, prev)
        ids, index, counts = np.unique(labels, return_inverse=True, return_counts=True)
        real = np.flatnonzero(ids != LEFTOVER)
        age = {}
        for g in ids[real]:
            age[int(g)] = 0
            if prev is not None:
                now, before = labels == g, prev == g
                if (now & before).sum() / (now | before).sum() > 0.8:
                    age[int(g)] = self.age[layer].get(int(g), 0) + 1
        self.age[layer] = age
        ceiling = [1.0 / (1.0 + self.beta * age[int(g)]) if self.consolidate else 1.0 for g in ids[real]]
        self.labels[layer] = labels
        self.ids[layer] = [int(g) for g in ids]
        self.index[layer] = torch.as_tensor(index, dtype=torch.long, device=self.device)
        self.real_pos[layer] = torch.as_tensor(real, dtype=torch.long, device=self.device)
        self.sizes[layer] = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        self.ceiling[layer] = torch.as_tensor(ceiling, dtype=torch.float32, device=self.device).clamp_min(self.floor)
        self.health[layer] = {**partition_health(labels, self.min_size), "splits": splits, "merges": merges}
        self.degenerate[layer] = self.health[layer]["degenerate"]

    def _report_clustering(self, layer):
        h = self.health[layer]
        if h["degenerate"]:
            print(f"[assembly] WARNING: layer {layer} clustering {self.n_clusterings} at step {self.steps} is degenerate "
                  f"({h['n_assemblies']} assemblies, largest {h['largest_frac']:.2f} of neurons). This layer trains dense "
                  f"until a later clustering recovers.", flush=True)
        if h["n_dead"] > 0:
            print(f"[assembly] layer {layer}: {h['n_dead']} dead or constant neurons, {DEAD_HANDLING}", flush=True)
        self._write_rows([self._layer_row(self.steps, "clustering", layer)])

    def _layer_row(self, it, kind, layer):
        h = self.health[layer]
        return {"iter": it, "kind": kind, "layer": layer, "clustering": self.n_clusterings,
                "n_assemblies": h["n_assemblies"], "size_min": h["size_min"], "size_median": f"{h['size_median']:.1f}",
                "size_max": h["size_max"], "largest_frac": f"{h['largest_frac']:.4f}",
                "frac_below_2x_min": f"{h['frac_below_2x_min']:.4f}", "n_dead": h["n_dead"],
                "dead_handling": DEAD_HANDLING if h["n_dead"] else "", "splits": h["splits"], "merges": h["merges"],
                "stability_ari": "" if self.stability[layer] is None else f"{self.stability[layer]:.4f}",
                "degenerate": int(h["degenerate"]), "dense_fallback": int(self.degenerate[layer])}

    def _write_rows(self, rows, path=None):
        path = path or self.log_path
        if not path or not rows:
            return
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="")
            if new:
                w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------------ logging
    def log(self, it, path=None):
        """Eval-interval rows: one `eval_layer` row per layer (with the plasticity histogram over [0, 1] in 10 equal
        bins, as fractions) and one `eval_assembly` row per assembly. plast_* is the mean plasticity of that domain's
        tokens since the previous log, averaged over the assembly's current neurons."""
        if all(lab is None for lab in self.labels):
            return
        plast, count = self.dom_plast.cpu().numpy(), self.dom_count.cpu().numpy()
        hist = self.p_hist.cpu().numpy()
        rows = []
        for layer in range(self.n_layer):
            if self.labels[layer] is None:
                continue
            row = self._layer_row(it, "eval_layer", layer)
            if hist[layer].sum() > 0:
                frac = hist[layer] / hist[layer].sum()
                row.update({f"p_hist_{i}": f"{frac[i]:.4f}" for i in range(P_HIST_BINS)})
                row["mean_p"] = f"{self.p_routed[layer][0] / max(self.p_routed[layer][1], 1):.4f}"
            rows.append(row)
            labels = self.labels[layer]
            for pos, g in enumerate(self.ids[layer]):
                if g == LEFTOVER:
                    continue
                members = labels == g
                cpos = self.ids[layer][:pos].count(LEFTOVER)
                vals = {f"plast_{d}": f"{plast[i, layer][members].mean() / count[i, layer]:.4f}"
                        for i, d in enumerate(DOMAINS) if count[i, layer] > 0}
                rows.append({"iter": it, "kind": "eval_assembly", "layer": layer, "clustering": self.n_clusterings,
                             "assembly": g, "size": int(members.sum()),
                             "ceiling": f"{float(self.ceiling[layer][pos - cpos]):.4f}", **vals})
        self._write_rows(rows, path)
        self.dom_plast.zero_()
        self.dom_count.zero_()
        self.p_hist.zero_()

    def mean_p(self):
        """(whole run, after the first clustering), each per layer, with p = 1 wherever a layer trained dense."""
        return (self.p_total[:, 0] / np.maximum(self.p_total[:, 1], 1), self.p_routed[:, 0] / np.maximum(self.p_routed[:, 1], 1))

    def write_summary(self, path):
        whole, routed = self.mean_p()
        lines = [f"mode: assembly, plasticity={self.plasticity_kind}, floor={self.floor}, resolution={self.resolution}",
                 f"steps: {self.steps}, clusterings: {self.n_clusterings}",
                 f"mean p, whole run (p=1 while dense): {whole.mean():.4f}",
                 f"mean p, after the first clustering: {routed.mean():.4f}",
                 "mean p per layer, whole run: " + ", ".join(f"L{i}={v:.4f}" for i, v in enumerate(whole)),
                 f"lr_scale for a matched dense run: {whole.mean():.4f}  (ROUTING_MODE=dense LR_SCALE={whole.mean():.4f})",
                 "assemblies per layer: " + str([h["n_assemblies"] if h else 0 for h in self.health]),
                 "degenerate layers (trained dense at the end): " + str([i for i, d in enumerate(self.degenerate) if d])]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def summary(self):
        if self.labels[0] is None:
            return f"assembly: dense until step {self.cluster_start} (step {self.steps})"
        counts = [h["n_assemblies"] for h in self.health]
        aris = [f"{a:.2f}" for a in self.stability if a is not None]
        return (f"assembly: clusterings {self.n_clusterings} | assemblies per layer {counts} | stability {aris or 'n/a'} "
                f"| degenerate layers {[i for i, d in enumerate(self.degenerate) if d]}")

    def assignments(self):
        return {"resolution": self.resolution, "knn": self.knn, "min_size": self.min_size, "floor": self.floor,
                "alpha": self.alpha, "plasticity": self.plasticity_kind, "consolidate": self.consolidate,
                "steps": self.steps, "n_clusterings": self.n_clusterings, "leftover_id": LEFTOVER,
                "layers": [None if lab is None else [int(x) for x in lab] for lab in self.labels]}

    def write_assignments(self, path):
        with open(path, "w") as f:
            json.dump(self.assignments(), f)

    def remove(self):
        for mlp in self.mlps:
            mlp.assembly_router = None

    def state_dict(self):
        """The correlation matrices are left out to keep checkpoints small; they refill within about 100 steps."""
        return {"labels": self.labels, "next_id": self.next_id, "age": self.age, "stability": self.stability,
                "steps": self.steps, "n_clusterings": self.n_clusterings, "p_total": self.p_total, "p_routed": self.p_routed,
                "neuron_ema": [e.cpu() for e in self.tracker.neuron_ema],
                "variance": [v.cpu() for v in self.tracker.variance]}

    def load_state_dict(self, sd):
        self.steps, self.n_clusterings = sd["steps"], sd["n_clusterings"]
        self.p_total, self.p_routed = sd["p_total"], sd["p_routed"]
        for layer, labels in enumerate(sd["labels"]):
            self.tracker.neuron_ema[layer].copy_(sd["neuron_ema"][layer])
            self.tracker.variance[layer].copy_(sd["variance"][layer])
            if labels is not None:
                self.labels[layer] = None
                self.set_labels(layer, labels)      # labels already carry stable IDs
                self.age[layer], self.stability[layer] = sd["age"][layer], sd["stability"][layer]
            self.next_id[layer] = sd["next_id"][layer]
