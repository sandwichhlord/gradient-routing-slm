"""Tests for routing v1 (token and label modes) and for v0 staying intact. Run: python -m pytest tests/ -v

The golden files were produced by the code at tag v0-batch-routing (torch 2.6.0 on CPU). Floats can differ across
machines and torch versions, so a mismatch elsewhere should first be checked by regenerating them from that tag.
"""
import csv
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
import train  # noqa: E402
from common import PureWindowSampler  # noqa: E402
from model.model import GPT, GPTConfig  # noqa: E402
from model.routing import LABEL_BLOCKS, LABEL_SHARED, TokenRouter  # noqa: E402

CFG = GPTConfig(block_size=32, n_layer=3, n_head=2, n_embd=64, n_blocks=8)
STORY, MATH, CODE = 0, 1, 2
SMOKE = os.path.join(ROOT, "configs", "smoke.py")
GOLDEN = os.path.join(ROOT, "tests", "golden")


def fresh_model(cfg=CFG):
    torch.manual_seed(0)
    return GPT(cfg)


def tokens(batch=2, length=CFG.block_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 50257, (batch, length), generator=g)


def block_grads(mlp):
    """(up, down): per-block grad magnitude on W_up rows and W_down columns."""
    n, bd = mlp.n_blocks, mlp.block_dim
    up = mlp.c_fc.weight.grad.view(n, bd, -1).abs().sum(dim=(1, 2))
    down = mlp.c_proj.weight.grad.view(-1, n, bd).abs().sum(dim=(0, 2))
    return up, down


def assert_only_blocks(mlp, allowed, msg=""):
    allowed = set(allowed)
    up, down = block_grads(mlp)
    for b in range(mlp.n_blocks):
        if b in allowed:
            assert up[b] > 0 and down[b] > 0, f"{msg} block {b} should have gradient (up {up[b]}, down {down[b]})"
        else:
            assert up[b] == 0 and down[b] == 0, f"{msg} block {b} should have none (up {up[b]}, down {down[b]})"


def token_loss(logits, targets, b, t):
    return F.cross_entropy(logits[b, t].unsqueeze(0), targets[b, t].unsqueeze(0))


# ---------------------------------------------------------------------------------- token mode gradients
@pytest.mark.parametrize("length", [1, CFG.block_size])
def test_token_mode_single_token_loss_reaches_only_its_blocks(length):
    """Token 0 sees only itself (causal attention), so its loss must land on its selected blocks in every layer,
    in both W_up rows and W_down columns."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="token", k=2)
    x = tokens(length=length)
    logits, _ = model(x, x)
    token_loss(logits, x, 1, 0).backward()
    for layer, mlp in enumerate(model.mlps()):
        selected = router.last_mask[layer][1, 0].nonzero().view(-1).tolist()
        assert len(selected) == 2
        assert_only_blocks(mlp, selected, f"layer {layer}:")


def test_token_mode_later_token_reaches_earlier_tokens_blocks_only_through_attention():
    """A later token's loss also flows through the earlier tokens it attends to. The last layer has no attention
    after it, so there the loss reaches exactly that token's blocks; earlier layers get the union over tokens <= t."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="token", k=2)
    x = tokens()
    t = CFG.block_size - 1
    logits, _ = model(x, x)
    token_loss(logits, x, 0, t).backward()
    last = model.mlps()[-1]
    assert_only_blocks(last, router.last_mask[-1][0, t].nonzero().view(-1).tolist(), "last layer:")
    for layer, mlp in enumerate(model.mlps()[:-1]):
        union = router.last_mask[layer][0, :t + 1].sum(dim=0) > 0
        up, down = block_grads(mlp)
        assert (up[~union] == 0).all() and (down[~union] == 0).all(), f"layer {layer}: gradient outside the union"


def test_naive_masking_of_h_alone_leaks_into_w_down():
    """Why both projections need handling: if only h is masked, W_down is multiplied by the full activation, so it
    receives gradient from every block. BlockMLP._token_routed_output detaches W_down on the unselected part."""
    torch.manual_seed(0)
    W_up, W_down = torch.randn(8, 4, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    x = torch.randn(1, 4)
    h = F.gelu(x @ W_up.T)
    m = torch.tensor([[1.0, 1.0, 0, 0, 0, 0, 0, 0]])
    naive = F.linear(h * m + (h * (1 - m)).detach(), W_down)
    naive.sum().backward()
    assert (W_down.grad[:, 2:].abs().sum(dim=0) > 0).all(), "the unselected columns do get gradient from naive masking"
    assert (W_up.grad[2:] == 0).all(), "W_up is fine either way"


# ---------------------------------------------------------------------------------- forward equality
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mode", ["token", "label"])
def test_forward_matches_dense(mode, bias):
    cfg = GPTConfig(block_size=32, n_layer=3, n_head=2, n_embd=64, n_blocks=8, bias=bias)
    x = tokens(batch=4)
    dense = fresh_model(cfg).train()
    want = dense(x, x)[0]
    routed = fresh_model(cfg).train()
    router = TokenRouter(routed, mode=mode)
    router.set_batch(torch.randint(0, 3, x.shape))
    got = routed(x, x)[0]
    assert router.last_mask[0] is not None, "the routed path must have run"
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-5)
    routed.eval()
    dense.eval()
    assert torch.equal(routed(x, x)[0], dense(x, x)[0]), "eval mode must use the plain dense path"


def test_mode_works_under_autocast():
    """The T4 runs fp16, which is not available on CPU; bf16 autocast exercises the same dtype handling."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="token")
    x = tokens()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(x, x)[1]
    loss.backward()
    assert router.total_tokens > 0
    assert_only = block_grads(model.mlps()[0])
    assert all(g.isfinite().all() for g in assert_only)


# ---------------------------------------------------------------------------------- token mode internals
def test_token_mode_does_not_use_labels():
    """Same weights after several steps whether labels are absent, right, or scrambled."""
    def run(labels):
        model = fresh_model().train()
        router = TokenRouter(model, mode="token")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(4):
            x = tokens(seed=step)
            router.set_batch(None if labels is None else labels(x, step))
            model(x, x)[1].backward()
            router.step(opt)
            opt.zero_grad(set_to_none=True)
        return model, router

    base, _ = run(None)
    other, router = run(lambda x, step: torch.randint(0, 3, x.shape, generator=torch.Generator().manual_seed(step)))
    for a, b in zip(base.parameters(), other.parameters()):
        assert torch.equal(a, b)
    assert router.dom_tokens.sum() > 0, "the labels were still counted for logging"


def test_bias_moves_against_over_selected_blocks_and_is_not_a_parameter():
    model = fresh_model().train()
    router = TokenRouter(model, mode="token", k=2, gamma=0.001)
    assert all(p is not router.bias for p in model.parameters())
    x = tokens()
    model(x, x)
    frac = router._step_frac.clone()
    router.after_step()
    expected = torch.where(frac > 0.25, -0.001, 0.001)
    torch.testing.assert_close(router.bias, expected)
    assert router.bias.requires_grad is False


def test_scores_are_relative_to_each_blocks_ema():
    """A block that is uniformly 100x louder must not win every token: the score divides by its EMA."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="token", k=2)
    hb = torch.randn(4, 16, 8, 32)
    hb[:, :, 3] *= 100.0
    m = router.select(0, hb)
    assert m[:, :, 3].mean() < 0.9
    # without the EMA division block 3 would be picked by every token
    assert hb.norm(dim=-1).argmax(dim=-1).eq(3).all()


# ---------------------------------------------------------------------------------- label mode
def test_label_mode_math_tokens_never_touch_story_blocks():
    model = fresh_model().train()
    router = TokenRouter(model, mode="label")
    x = tokens()
    router.set_batch(torch.full(x.shape, MATH))
    model(x, x)[1].backward()
    for layer, mlp in enumerate(model.mlps()):
        assert_only_blocks(mlp, [*LABEL_BLOCKS[MATH], *LABEL_SHARED], f"layer {layer}:")
        up, down = block_grads(mlp)
        assert (up[list(LABEL_BLOCKS[STORY])] == 0).all() and (down[list(LABEL_BLOCKS[STORY])] == 0).all()


def test_label_mode_math_loss_in_a_mixed_batch_stays_in_math_blocks():
    """A batch mixes windows of different domains. Each window's loss stays in its own blocks."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="label")
    x = tokens(batch=3)
    labels = torch.stack([torch.full((CFG.block_size,), d) for d in (STORY, MATH, CODE)])
    router.set_batch(labels)
    logits, _ = model(x, x)
    F.cross_entropy(logits[1].view(-1, logits.size(-1)), x[1].view(-1)).backward()
    for mlp in model.mlps():
        assert_only_blocks(mlp, [*LABEL_BLOCKS[MATH], *LABEL_SHARED])


def test_mixed_domain_window_would_leak_which_is_why_label_mode_uses_pure_windows():
    """A math token after story tokens reaches story blocks through attention, so training never uses such windows."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="label")
    x = tokens(batch=1)
    labels = torch.full(x.shape, STORY)
    labels[0, -1] = MATH
    router.set_batch(labels)
    logits, _ = model(x, x)
    token_loss(logits, x, 0, CFG.block_size - 1).backward()
    up, _ = block_grads(model.mlps()[0])
    assert (up[list(LABEL_BLOCKS[STORY])] > 0).all()


@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory):
    """The v0-v2 smoke corpus (no deconfounding data, no held-out arithmetic), which the golden metrics were produced on."""
    d = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "prepare.py"), "--synthetic",
                    "--synthetic_tokens", "50000", "--deconf_frac", "0", "--holdout_arith", "0", "--out_dir", str(d)],
                   check=True, capture_output=True)
    return d


def test_pure_window_sampler_never_mixes_domains(smoke_data):
    np.random.seed(0)
    sampler = PureWindowSampler(str(smoke_data), 64)
    seen = set()
    for _ in range(20):
        x, y, d = sampler.get_batch(16, "cpu")
        assert (d.min(dim=1).values == d.max(dim=1).values).all()
        assert torch.equal(x[:, 1:], y[:, :-1])
        seen |= set(d[:, 0].tolist())
    assert seen == {STORY, MATH, CODE}
    assert abs(sampler.p.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------------- optimizer skip
def test_zero_gradient_slices_are_bit_identical_across_a_step():
    model = fresh_model().train()
    router = TokenRouter(model, mode="label")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    x = tokens(batch=3)
    # first a step with all domains, so the Adam moments are non-zero everywhere
    router.set_batch(torch.stack([torch.full((CFG.block_size,), d) for d in (STORY, MATH, CODE)]))
    model(x, x)[1].backward()
    router.step(opt)
    opt.zero_grad(set_to_none=True)
    # blocks 6 and 7 are shared, block 0-1 and 4-5 are idle on this math-only step
    idle = [*LABEL_BLOCKS[STORY], *LABEL_BLOCKS[CODE]]
    bd = model.mlps()[0].block_dim
    rows = torch.tensor([i for b in idle for i in range(b * bd, (b + 1) * bd)])
    active = torch.tensor([i for b in (*LABEL_BLOCKS[MATH], *LABEL_SHARED) for i in range(b * bd, (b + 1) * bd)])

    def snapshot():
        out = []
        for mlp in model.mlps():
            fc, proj = mlp.c_fc.weight, mlp.c_proj.weight
            for p, dim in ((fc, 0), (proj, 1)):
                st = opt.state[p]
                out.append((p, dim, p.data.clone(), st["exp_avg"].clone(), st["exp_avg_sq"].clone()))
        return out

    before = snapshot()
    router.set_batch(torch.full(x.shape, MATH))
    model(x, x)[1].backward()
    router.step(opt)
    for p, dim, value, avg, avg_sq in before:
        st = opt.state[p]
        for now, old in ((p.data, value), (st["exp_avg"], avg), (st["exp_avg_sq"], avg_sq)):
            assert torch.equal(now.index_select(dim, rows), old.index_select(dim, rows)), "idle slice changed"
        assert not torch.equal(p.data.index_select(dim, active), value.index_select(dim, active)), "active slice frozen"
    assert router.skipped_slices > 0


def test_without_the_skip_idle_slices_would_move():
    """Control for the test above: plain AdamW moves zero-gradient weights through weight decay and momentum."""
    model = fresh_model().train()
    router = TokenRouter(model, mode="label")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    x = tokens(batch=3)
    router.set_batch(torch.stack([torch.full((CFG.block_size,), d) for d in (STORY, MATH, CODE)]))
    model(x, x)[1].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    w = model.mlps()[0].c_fc.weight
    bd = model.mlps()[0].block_dim
    before = w.data[:bd].clone()
    router.set_batch(torch.full(x.shape, MATH))
    model(x, x)[1].backward()
    assert (w.grad[:bd] == 0).all()
    opt.step()
    assert not torch.equal(w.data[:bd], before)


# ---------------------------------------------------------------------------------- v0 reproduced exactly
@pytest.mark.parametrize("mode", ["dense", "batch"])
def test_mode_reproduces_v0_smoke_metrics(mode, smoke_data, tmp_path):
    train.main(["--config", SMOKE, "--data_dir", str(smoke_data), "--out_dir", str(tmp_path), "--device", "cpu",
                "--routing_mode", mode, "--seed", "1337"])
    with open(tmp_path / "metrics.csv", newline="") as f:
        got = [r[:6] for r in csv.reader(f)]
    with open(os.path.join(GOLDEN, f"{mode}_smoke_metrics.csv"), newline="") as f:
        want = list(csv.reader(f))
    assert got == want


# ---------------------------------------------------------------------------------- token mode training health
def test_token_mode_no_dead_blocks_and_balanced_after_300_steps(smoke_data, tmp_path):
    out = train.main(["--config", SMOKE, "--data_dir", str(smoke_data), "--out_dir", str(tmp_path), "--device", "cpu",
                      "--routing_mode", "token", "--seed", "0", "--max_iters", "300", "--eval_interval", "100",
                      "--eval_iters", "2", "--ckpt_interval", "300"])
    router = out["router"]
    assert router.steps == 300
    assert not router.dead_blocks(), f"dead blocks: {router.dead_blocks()}"
    frac = router.selection_fracs()
    assert frac.min() >= 0.1 and frac.max() <= 0.4, f"selection fractions out of range:\n{frac}"
    with open(tmp_path / "routing_stats.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 2 * 8
    last = [r for r in rows if r["iter"] == "300"]
    for name in ("story", "math", "code"):       # each domain's tokens pick k blocks per layer
        total = sum(float(r[f"frac_{name}"]) for r in last if r["layer"] == "0")
        assert abs(total - 2.0) < 1e-3
    with open(tmp_path / "metrics.csv", newline="") as f:
        metrics = list(csv.DictReader(f))
    assert metrics[0]["tokens_per_sec"] == "" and float(metrics[-1]["tokens_per_sec"]) > 0


# ---------------------------------------------------------------------------------- full pipeline
def find_bash():
    if sys.platform == "win32":  # avoid the WSL launcher in System32
        for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if os.path.exists(p):
                return p
    return shutil.which("bash")


@pytest.mark.parametrize("mode", ["dense", "batch", "token", "label"])
def test_smoke_run_all_completes(mode, tmp_path):
    bash = find_bash()
    assert bash, "bash not found"
    out = tmp_path / "out"
    env = {**os.environ, "SMOKE": "1", "ROUTING_MODE": mode, "SEED": "0", "DEVICE": "cpu", "PYTHON": sys.executable,
           "OUT_DIR": out.as_posix(), "DATA_DIR": (tmp_path / "data").as_posix()}
    r = subprocess.run([bash, "run_all.sh"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    expected = ["metrics.csv", "ckpt.pt", "samples.txt", "loss_curves.png", "ablation.csv", "ablation_heatmap.png",
                "ablation_summary.txt", "block_norms.csv"]
    if mode != "dense":
        expected += ["routing_stats.csv", "routing_by_domain.png"]
    else:  # the v3 evaluations, on post-hoc groups
        expected += ["groups_posthoc.json", "context_sensitivity.json", "damage_table.csv", "damage_summary.json",
                     "failure_trace_summary.json", "unlearn_summary.json"]
    for name in expected:
        assert (out / name).stat().st_size > 0, name
    assert not (mode == "dense" and (out / "routing_stats.csv").exists())


def test_default_output_dir_is_named_after_mode_and_seed():
    with open(os.path.join(ROOT, "run_all.sh"), encoding="utf-8") as f:
        assert 'OUT="${OUT_DIR:-out_${NAME}${TAG}_s${SEED}}"' in f.read()   # NAME is the mode, or gated_<groups>


def test_compare_reads_old_and_new_runs(tmp_path):
    results = tmp_path / "results"
    for name in ("dense_v0", "v0_batch_routing"):
        shutil.copytree(os.path.join(ROOT, "results", name), results / name,
                        ignore=shutil.ignore_patterns("ckpt.pt", "*.png"))
    for seed, loss in ((0, "3.0"), (1, "3.2")):
        d = results / f"token_s{seed}"
        d.mkdir()
        (d / "metrics.csv").write_text("iter,train_loss,val_story,val_math,val_code,lr,elapsed_s,tokens_per_sec\n"
                                       f"100,1,{loss},2.0,2.5,1e-4,10,40000\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "eval", "compare.py"), "--results_dir", str(results)], check=True,
                   capture_output=True)
    with open(results / "comparison.csv", newline="") as f:
        rows = {(r["kind"], r["name"]): r for r in csv.DictReader(f)}
    assert (results / "comparison.png").stat().st_size > 0
    dense = rows[("group", "dense_v0")]
    assert dense["mean_S"] == "0.0510" and dense["val_story"] == "2.1201"
    assert abs(float(dense["tokens_per_sec"]) - 8192 / 0.186) < 8192 / 0.186 * 0.1  # about 186 ms/it in run.log
    assert rows[("group", "v0_batch_routing")]["mean_S"] == "0.1104"
    token = rows[("group", "token")]
    assert token["n_runs"] == "2" and token["val_story"] == "3.1000" and float(token["val_story_std"]) > 0
    assert token["mean_S"] == ""
