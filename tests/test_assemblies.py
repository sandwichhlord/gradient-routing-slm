"""Tests for routing v2 (assembly mode). Run: python -m pytest tests/test_assemblies.py -v"""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import train  # noqa: E402
from ablate_assemblies import bootstrap_ci  # noqa: E402
from model.assemblies import (LEFTOVER, AssemblyRouter, CoactivationTracker, adjusted_rand_index,  # noqa: E402
                              cluster_correlation, count_splits_merges, match_labels, partition_health, rank_plasticity,
                              sigmoid_plasticity)
from model.model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=64, n_blocks=8)
HIDDEN = 256
SMOKE = os.path.join(ROOT, "configs", "smoke.py")
GOLDEN = os.path.join(ROOT, "tests", "golden")


def fresh_model(cfg=CFG):
    torch.manual_seed(0)
    return GPT(cfg)


def tokens(batch=2, length=CFG.block_size, seed=0):
    return torch.randint(0, 50257, (batch, length), generator=torch.Generator().manual_seed(seed))


def striped_labels(hidden=HIDDEN, groups=8):
    return np.repeat(np.arange(groups), hidden // groups)


def block_corr(sizes, within=0.6, between=0.02, seed=0):
    """A correlation matrix with clear groups of the given sizes, and a little noise."""
    n = sum(sizes)
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    c = np.where(labels[:, None] == labels[None, :], within, between) + rng.normal(0, 0.01, (n, n))
    c = (c + c.T) / 2
    np.fill_diagonal(c, 1.0)
    return c, labels


def routed_model(plasticity="rank", floor=0.1, cfg=CFG, labels=None):
    model = fresh_model(cfg).train()
    router = AssemblyRouter(model, cluster_start=10 ** 9, plasticity=plasticity, floor=floor)
    x = tokens()
    model(x, x)                 # fills the activity baselines
    hidden = model.mlps()[0].n_blocks * model.mlps()[0].block_dim
    for layer in range(router.n_layer):
        router.set_labels(layer, striped_labels(hidden) if labels is None else labels)
    return model, router


# ---------------------------------------------------------------------------------- forward and gradients
@pytest.mark.parametrize("plasticity", ["rank", "sigmoid"])
@pytest.mark.parametrize("bias", [False, True])
def test_forward_equals_dense(plasticity, bias):
    cfg = GPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=64, n_blocks=8, bias=bias)
    x = tokens(batch=4)
    want = fresh_model(cfg).train()(x, x)[0]
    model, router = routed_model(plasticity, cfg=cfg)
    got = model(x, x)[0]
    assert router.last_P[0] is not None, "the graded path must have run"
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-5)


def test_forward_is_exactly_dense_before_the_first_clustering():
    x = tokens()
    dense = fresh_model().train()
    model = fresh_model().train()
    router = AssemblyRouter(model, cluster_start=10 ** 9)
    assert torch.equal(model(x, x)[0], dense(x, x)[0])
    assert router.last_P[0] is None


@pytest.mark.parametrize("plasticity", ["rank", "sigmoid"])
def test_last_layer_token_gradient_is_scaled_by_exactly_p_in_both_projections(plasticity):
    x = tokens()
    b, t = 1, CFG.block_size - 1
    dense = fresh_model().train()
    logits = dense(x, x)[0]
    F.cross_entropy(logits[b, t].unsqueeze(0), x[b, t].unsqueeze(0)).backward()
    model, router = routed_model(plasticity)
    logits = model(x, x)[0]
    F.cross_entropy(logits[b, t].unsqueeze(0), x[b, t].unsqueeze(0)).backward()
    P = router.last_P[-1][b, t]
    assert P.unique().numel() > 1, "the test needs neurons with different plasticity"
    want, got = dense.mlps()[-1], model.mlps()[-1]
    torch.testing.assert_close(got.c_fc.weight.grad, want.c_fc.weight.grad * P.unsqueeze(1), atol=1e-6, rtol=1e-3)
    torch.testing.assert_close(got.c_proj.weight.grad, want.c_proj.weight.grad * P.unsqueeze(0), atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("plasticity", ["rank", "sigmoid"])
def test_no_neuron_ever_gets_zero_plasticity(plasticity):
    model, router = routed_model(plasticity, floor=0.1)
    x = tokens(batch=4)
    model(x, x)[1].backward()
    for layer, mlp in enumerate(model.mlps()):
        P = router.last_P[layer]
        assert P.min() >= 0.1 - 1e-6 and P.max() <= 1.0 + 1e-6
        assert (mlp.c_fc.weight.grad.abs().sum(dim=1) > 0).all()
        assert (mlp.c_proj.weight.grad.abs().sum(dim=0) > 0).all()
    silent = torch.zeros(1, 1, 8)
    assert sigmoid_plasticity(silent, 0.1, torch.ones(8), 4.0).min() >= 0.1
    with pytest.raises(AssertionError):
        AssemblyRouter(fresh_model(), floor=0.0)


def test_rank_formula_gives_top_one_and_bottom_floor():
    r = torch.tensor([[0.5, 3.0, 1.0, 2.0]])
    p = rank_plasticity(r, 0.1, torch.ones(4))
    torch.testing.assert_close(p, torch.tensor([[0.1, 1.0, 0.4, 0.7]]))


@pytest.mark.parametrize("groups", [0, 1])
def test_rank_formula_is_guarded_below_two_assemblies(groups):
    p = rank_plasticity(torch.ones(2, 3, groups), 0.1, torch.ones(groups))
    assert p.shape == (2, 3, groups) and not p.isnan().any()
    if groups == 1:
        assert (p == 1.0).all()


def test_labels_do_not_change_training():
    def run(labels):
        model = fresh_model().train()
        router = AssemblyRouter(model, cluster_start=2, cluster_every=2, seed=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(5):
            x = tokens(seed=step)
            router.set_batch(None if labels is None else torch.randint(0, 3, x.shape, generator=torch.Generator().manual_seed(step)))
            model(x, x)[1].backward()
            router.step(opt)
            opt.zero_grad(set_to_none=True)
        return model

    for a, b in zip(run(None).parameters(), run(True).parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------------- clustering
@pytest.mark.parametrize("backend", ["leiden", "louvain"])
def test_clustering_is_a_partition_of_all_1536_neurons_with_no_small_group(backend):
    corr, _ = block_corr([700, 500, 300, 30, 6])
    labels = cluster_correlation(corr, knn=32, resolution=1.0, min_size=16, seed=0, backend=backend)
    assert labels.shape == (1536,)
    assert (labels >= 0).all()
    assert np.bincount(labels).min() >= 16


def test_clustering_recovers_planted_groups():
    corr, truth = block_corr([400, 400, 400, 336])
    labels = cluster_correlation(corr, knn=32)
    assert adjusted_rand_index(labels, truth) == pytest.approx(1.0)


def test_ids_are_stable_under_an_unchanged_correlation_matrix():
    model = fresh_model().train()
    router = AssemblyRouter(model, cluster_start=10 ** 9, min_size=16)
    corr, _ = block_corr([64] * 4)
    router.tracker.corr[0].copy_(torch.as_tensor(corr, dtype=torch.float32))
    router.tracker.corr[1].copy_(torch.as_tensor(corr, dtype=torch.float32))
    router.tracker.variance[0].fill_(1.0)
    router.tracker.variance[1].fill_(1.0)
    router.recluster()
    first = router.labels[0].copy()
    router.recluster()
    assert np.array_equal(router.labels[0], first)
    assert router.stability[0] == pytest.approx(1.0)
    assert router.health[0]["splits"] == 0 and router.health[0]["merges"] == 0


def test_match_labels_keeps_ids_when_the_new_clustering_numbers_groups_differently():
    prev = np.repeat([7, 3, 12, 5], 10)
    shuffled = np.repeat([0, 1, 2, 3], 10)
    labels, next_id = match_labels(shuffled, prev, next_id=13)
    assert np.array_equal(labels, prev) and next_id == 13


def test_match_labels_gives_new_groups_fresh_ids():
    prev = np.repeat([7, 3], 20)
    new = np.repeat([0, 1, 2], [20, 10, 10])
    labels, next_id = match_labels(new, prev, next_id=8)
    assert set(labels.tolist()) == {7, 3, 8} and next_id == 9


def test_splits_and_merges_are_counted():
    prev = np.repeat([0, 1, 2], [40, 20, 20])
    split = np.repeat([0, 1, 2, 3], [20, 20, 20, 20])
    assert count_splits_merges(split, prev) == (1, 0)
    merged = np.repeat([0, 1], [40, 40])
    assert count_splits_merges(merged, prev) == (0, 1)


def test_adjusted_rand_index():
    a = np.repeat([0, 1], 50)
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert adjusted_rand_index(a, 1 - a) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert abs(adjusted_rand_index(rng.integers(0, 4, 2000), rng.integers(0, 4, 2000))) < 0.02


def test_baseline_after_reclustering_comes_from_member_neurons():
    """Two old assemblies with very different activity merge. Whatever ID the merged group is matched to, its baseline must
    be the mean over all its members, not the old baseline of that ID."""
    model, router = routed_model("sigmoid")
    router.tracker.neuron_ema[0][:32] = 1.0
    router.tracker.neuron_ema[0][32:64] = 100.0
    labels = router.labels[0].copy()
    labels[:64] = 0
    router.set_labels(0, labels)
    h = torch.zeros(1, 1, HIDDEN)
    h[..., :64] = 50.5                           # exactly the mean of the members' usual activity
    router.plasticity(0, h)
    want = 0.1 + 0.9 * 0.5                       # r = 1, so sigmoid(0) = 0.5
    assert float(router.last_P[0][0, 0, 0]) == pytest.approx(want, abs=1e-4)
    assert float(router.last_P[0][0, 0, 40]) == pytest.approx(want, abs=1e-4)


# ---------------------------------------------------------------------------------- health checks
def test_zero_variance_neurons_are_excluded_and_assigned_to_a_leftover_group():
    corr, _ = block_corr([64] * 4)
    dead = np.zeros(256, dtype=bool)
    dead[[3, 70, 200]] = True
    corr[dead] = 0.0
    corr[:, dead] = 0.0
    labels = cluster_correlation(corr, knn=32, min_size=16, dead=dead)
    assert (labels[dead] == LEFTOVER).all() and (labels[~dead] >= 0).all()
    health = partition_health(labels, 16)
    assert health["n_dead"] == 3 and health["n_assemblies"] == 4 and not health["degenerate"]


def test_tracker_flags_constant_and_dead_neurons_without_nan():
    tracker = CoactivationTracker(1, 32, sample_tokens=64)
    h = torch.randn(4, 16, 32)
    h[..., 5] = 0.0
    h[..., 9] = 0.7
    tracker.update(0, h)
    dead = tracker.dead(0)
    assert dead[5] and dead[9] and dead.sum() == 2
    assert not tracker.corr[0].isnan().any()
    assert tracker.corr[0][5].abs().sum() == 0


def test_dead_neurons_keep_plasticity_one_and_stay_out_of_the_ranking():
    labels = striped_labels().copy()
    labels[:5] = LEFTOVER
    model, router = routed_model("rank", labels=labels)
    x = tokens()
    model(x, x)
    P = router.last_P[0]
    assert (P[..., :5] == 1.0).all()
    assert (P[..., 5:].unique().numel()) == 8     # 8 ranks among the real assemblies
    assert P[..., 5:].min() == pytest.approx(0.1)


def test_all_dead_layer_does_not_crash():
    labels = cluster_correlation(np.zeros((32, 32)), dead=np.ones(32, dtype=bool))
    assert (labels == LEFTOVER).all()
    assert partition_health(labels, 16)["degenerate"]


@pytest.mark.parametrize("labels", [np.repeat([0, 1, 2], [100, 78, 78]),         # fewer than 4 assemblies
                                    np.repeat([0, 1, 2, 3, 4], [200, 14, 14, 14, 14])])  # largest above 0.5
def test_degenerate_layer_falls_back_to_dense_and_warns(labels, tmp_path, capsys):
    model = fresh_model().train()
    router = AssemblyRouter(model, cluster_start=10 ** 9)
    router.log_path = str(tmp_path / "assemblies.csv")
    x = tokens()
    model(x, x)
    router.set_labels(0, labels)
    router.n_clusterings = 1
    router._report_clustering(0)
    assert "WARNING" in capsys.readouterr().out
    with open(router.log_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["degenerate"] == "1" and row["dense_fallback"] == "1" and row["kind"] == "clustering"
    router.last_P = [None, None]
    assert router.plasticity(0, torch.ones(1, 2, HIDDEN)) is None, "a degenerate layer must take the dense path"
    dense = fresh_model().train()
    assert torch.equal(model.mlps()[0](torch.ones(1, 2, 64)), dense.mlps()[0](torch.ones(1, 2, 64)))


def test_healthy_layer_recovers_after_a_degenerate_one():
    model = fresh_model().train()
    router = AssemblyRouter(model, cluster_start=10 ** 9)
    router.set_labels(0, np.zeros(HIDDEN, dtype=np.int64))
    assert router.degenerate[0]
    router.set_labels(0, striped_labels())
    assert not router.degenerate[0]


# ---------------------------------------------------------------------------------- documented limitation
def test_adam_cancels_a_constant_gradient_scale():
    """Why mean p is not an effective learning rate (docs/findings.md, section 8): scaling a parameter's gradient by a
    constant barely changes its Adam update. Variation between tokens is what survives."""
    def updates(scale):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(16))
        opt = torch.optim.Adam([p], lr=1e-3)
        for _ in range(20):
            p.grad = torch.randn(16) * scale
            opt.step()
        return p.detach().clone()

    torch.testing.assert_close(updates(1.0), updates(0.1), rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_contains_the_mean_and_ignores_nan():
    lo, hi = bootstrap_ci([0.1, 0.2, float("nan"), 0.3, 0.4])
    assert lo < 0.25 < hi
    assert all(np.isnan(bootstrap_ci([float("nan")])))


# ---------------------------------------------------------------------------------- training runs
@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "prepare.py"), "--synthetic",
                    "--synthetic_tokens", "50000", "--deconf_frac", "0", "--holdout_arith", "0", "--out_dir", str(d)],
                   check=True, capture_output=True)
    return d


def read_metrics(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def smoke_args(data, out, *extra):
    return ["--config", SMOKE, "--data_dir", str(data), "--out_dir", str(out), "--device", "cpu", *extra]


def test_assembly_mode_before_clustering_reproduces_dense_exactly(smoke_data, tmp_path):
    """Tracking must not disturb training: with no clustering in the run, the metrics match the v0 dense golden file."""
    train.main(smoke_args(smoke_data, tmp_path, "--routing_mode", "assembly", "--seed", "1337", "--cluster_start", "100000"))
    got = [r[:6] for r in read_metrics(tmp_path / "metrics.csv")]
    assert got == read_metrics(os.path.join(GOLDEN, "dense_smoke_metrics.csv"))


def test_lr_scale_scales_the_schedule(smoke_data, tmp_path):
    train.main(smoke_args(smoke_data, tmp_path, "--routing_mode", "dense", "--seed", "1337", "--lr_scale", "0.5",
                          "--max_iters", "25"))
    golden = {r[0]: float(r[5]) for r in read_metrics(os.path.join(GOLDEN, "dense_smoke_metrics.csv"))[1:]}
    # the 25-iter schedule differs from the golden 100-iter one after warmup, so compare at iter 0
    got = {r[0]: float(r[5]) for r in read_metrics(tmp_path / "metrics.csv")[1:]}
    assert got["0"] == pytest.approx(golden["0"] * 0.5, rel=1e-3)


def test_assembly_training_logs_health_and_mean_p(smoke_data, tmp_path):
    train.main(smoke_args(smoke_data, tmp_path, "--routing_mode", "assembly", "--seed", "0", "--max_iters", "60"))
    with open(tmp_path / "assemblies.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    clustering = [r for r in rows if r["kind"] == "clustering"]
    assert {r["iter"] for r in clustering} == {"30", "60"} and len(clustering) == 4
    for r in clustering:
        for field in ("n_assemblies", "largest_frac", "frac_below_2x_min", "n_dead", "splits", "merges", "degenerate"):
            assert r[field] != ""
    assert clustering[0]["stability_ari"] == "" and clustering[-1]["stability_ari"] != ""
    layer_rows = [r for r in rows if r["kind"] == "eval_layer" and r["iter"] == "50"]
    assert len(layer_rows) == 2
    hist = [float(layer_rows[0][f"p_hist_{i}"]) for i in range(10)]
    assert sum(hist) == pytest.approx(1.0, abs=1e-3) and sum(h > 0 for h in hist) >= 2, "p must actually spread"
    assert any(r["kind"] == "eval_assembly" and r["plast_math"] != "" for r in rows)
    text = (tmp_path / "summary.txt").read_text()
    assert "mean p, whole run" in text and "lr_scale for a matched dense run" in text
    with open(tmp_path / "assemblies_final.json") as f:
        final = json.load(f)
    assert len(final["layers"]) == 2 and len(final["layers"][0]) == 256
    metrics = list(csv.DictReader(open(tmp_path / "metrics.csv", newline="")))
    assert float(metrics[-1]["act_norm_l0"]) > 0 and float(metrics[-1]["act_norm_l1"]) > 0


# ---------------------------------------------------------------------------------- full pipeline
def find_bash():
    if sys.platform == "win32":  # avoid the WSL launcher in System32
        for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if os.path.exists(p):
                return p
    return shutil.which("bash")


def run_pipeline(tmp_path, **env_extra):
    out = tmp_path / "out"
    env = {**os.environ, "SMOKE": "1", "SEED": "0", "DEVICE": "cpu", "PYTHON": sys.executable, "OUT_DIR": out.as_posix(),
           "DATA_DIR": (tmp_path / "data").as_posix(), **env_extra}
    r = subprocess.run([find_bash(), "run_all.sh"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out


def test_smoke_run_all_assembly_mode(tmp_path):
    out = run_pipeline(tmp_path, ROUTING_MODE="assembly")
    for name in ["assemblies.csv", "assemblies_final.json", "summary.txt", "assembly_ablation.csv", "assembly_random.csv",
                 "assembly_summary.json", "assembly_heatmap.png", "assembly_sizes.png", "assembly_health.png",
                 "assembly_probe.csv", "assembly_domain_profile.csv", "ablation.csv", "metrics.csv"]:
        assert (out / name).stat().st_size > 0, name
    with open(out / "assembly_summary.json") as f:
        s = json.load(f)
    assert s["source"] == "final" and s["random_sets"] == 2
    for field in ("mean_S", "mean_S_ci95", "random_mean_S", "size_weighted_S", "damage_weighted_S", "damage_per_neuron",
                  "random_damage_per_neuron", "real_minus_random_S_ci95"):
        assert field in s
    with open(out / "assembly_probe.csv", newline="", encoding="utf-8") as f:
        probe = list(csv.DictReader(f))
    assert len(probe) == s["n_assemblies"] and len(json.loads(probe[0]["top20_tokens"])) == 20
    with open(out / "assembly_domain_profile.csv", newline="", encoding="utf-8") as f:
        profile = list(csv.DictReader(f))
    assert 0.0 <= float(profile[0]["frac_active_math"]) <= 1.0 and len(json.loads(profile[0]["top10_tokens"])) == 10


def test_smoke_dense_with_posthoc_control(tmp_path):
    out = run_pipeline(tmp_path, ROUTING_MODE="dense", POSTHOC="1")
    assert not (out / "assemblies.csv").exists()
    with open(out / "posthoc" / "assembly_summary.json") as f:
        s = json.load(f)
    assert s["source"] == "posthoc" and s["routing_mode"] == "dense"
    for name in ["assemblies_posthoc.json", "assembly_ablation.csv", "assembly_random.csv", "assembly_heatmap.png",
                 "assembly_sizes.png", "assembly_probe.csv", "assembly_domain_profile.csv"]:
        assert (out / "posthoc" / name).stat().st_size > 0, name


def test_non_default_settings_get_their_own_output_folder():
    env = {**os.environ, "SMOKE": "1", "ROUTING_MODE": "assembly", "RESOLUTION": "0.5", "FLOOR": "0.2", "PYTHON": "false"}
    env.pop("OUT_DIR", None)
    r = subprocess.run([find_bash(), "run_all.sh"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert "outputs in out_assembly_r0.5_f0.2_s0/" in r.stdout
