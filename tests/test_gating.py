"""Tests for v3: deconfounding data and tags, gated mode (fixed / evolved groups), its evaluations.
Run: python -m pytest tests/test_gating.py -v"""
import copy
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import tiktoken
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import train  # noqa: E402
from deconf import (TAG_ID, TAGS, heldout_problems, is_heldout, math_line, tag_tokens)  # noqa: E402
from model.gating import (GateRouter, capped_move, fixed_groups, partition_to_n, smoothing_weights)  # noqa: E402
from model.model import GPT, GPTConfig  # noqa: E402
from test_assemblies import block_corr, find_bash, read_metrics  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
CFG = GPTConfig(block_size=32, n_layer=3, n_head=2, n_embd=64, n_blocks=8)
HIDDEN = 256
SMOKE = os.path.join(ROOT, "configs", "smoke.py")
GOLDEN = os.path.join(ROOT, "tests", "golden")


def fresh_model(cfg=CFG):
    torch.manual_seed(0)
    return GPT(cfg)


def tokens(batch=2, length=CFG.block_size, seed=0, eot_at=(10,)):
    x = torch.randint(0, 50000, (batch, length), generator=torch.Generator().manual_seed(seed))
    for t in eot_at:
        x[:, t] = EOT
    return x


def gated(model, groups="fixed", k=3, smooth=0.9, layers=(1, 2), labels=None, **kw):
    """A router whose gating is on from the first forward, with activity baselines filled from one batch."""
    r = GateRouter(model, layers=layers, k=k, smooth=smooth, groups=groups, cluster_start=0, **kw)
    was = model.training
    model.train()
    model(tokens(seed=99))              # a training forward fills the per-neuron baselines (gating needs them)
    model.train(was)
    assert all(r.tracker.ready[layer] for layer in layers)
    for layer in layers:
        if labels is not None or groups == "evolved":
            r.set_labels(layer, fixed_groups(HIDDEN, 8) if labels is None else labels)
    return r


# ---------------------------------------------------------------------------------- forward pass
@pytest.mark.parametrize("score", ["relative", "share"])
@pytest.mark.parametrize("groups", ["fixed", "evolved"])
@pytest.mark.parametrize("smooth", [0.0, 0.9])
@pytest.mark.parametrize("training", [False, True])
def test_k_equal_to_all_groups_is_exactly_ungated(groups, smooth, training, score):
    model, ref = fresh_model(), fresh_model()
    rng = np.random.default_rng(0)
    gated(model, groups, k=8, smooth=smooth, labels=rng.integers(0, 8, HIDDEN) if groups == "evolved" else None,
          score=score)
    model.train(training)
    ref.train(training)
    x = tokens()
    assert torch.equal(model(x, x)[0], ref(x, x)[0])


def test_k_below_all_groups_changes_the_output():
    model, ref = fresh_model().eval(), fresh_model().eval()
    gated(model, k=3)
    x = tokens()
    assert not torch.equal(model(x, x)[0], ref(x, x)[0])


def test_fixed_and_evolved_differ_only_in_group_assignment():
    a, b = fresh_model().eval(), fresh_model().eval()
    ra, rb = gated(a, "fixed"), gated(b, "evolved")       # evolved gets the fixed partition
    x = tokens()
    la, lb = a(x, x)[0], b(x, x)[0]
    assert torch.equal(la, lb)
    for i in range(2):
        assert torch.equal(ra.last_selected[i], rb.last_selected[i]) and torch.equal(ra.last_gate[i], rb.last_gate[i])
    perm = np.random.default_rng(1).permutation(HIDDEN)
    for layer in rb.layers:
        rb.set_labels(layer, fixed_groups(HIDDEN, 8)[perm])
    assert not torch.equal(b(x, x)[0], la)


def test_soft_kwta_selects_exactly_k_with_gates_split_at_one_half():
    model = fresh_model().eval()
    r = gated(model, k=3)
    model(tokens())
    for i in range(2):
        sel, gate = r.last_selected[i], r.last_gate[i]
        assert (sel.sum(-1) == 3).all()
        assert (gate[sel] > 0.5).all() and (gate[~sel] < 0.5).all()


@pytest.mark.parametrize("training", [False, True])
def test_gates_are_causal(training):
    """Changing token t+1 must not change any gate at positions <= t (with smoothing and a document boundary)."""
    model = fresh_model()
    gated(model, "evolved", smooth=0.9)
    x = tokens(eot_at=(6, 20))
    t = 14
    x2 = x.clone()
    x2[:, t + 1] = (x2[:, t + 1] + 123) % 50000
    m1, m2 = copy.deepcopy(model).train(training), copy.deepcopy(model).train(training)
    m1(x, x)
    m2(x2, x2)
    r1, r2 = m1.gate_router, m2.gate_router
    for i in range(2):
        assert torch.equal(r1.last_gate[i][:, :t + 1], r2.last_gate[i][:, :t + 1])
        assert not torch.equal(r1.last_gate[i][:, t + 1:], r2.last_gate[i][:, t + 1:])


def test_smoothing_is_the_causal_ema_restarted_after_endoftext():
    x = tokens(batch=3, eot_at=(0, 5, 17))
    r = torch.randn(3, CFG.block_size, 8)
    s = smoothing_weights(x, 0.9, EOT) @ r
    ref = torch.zeros_like(r)
    for b in range(3):
        for t in range(CFG.block_size):
            ref[b, t] = r[b, t] if (t == 0 or x[b, t - 1] == EOT) else 0.9 * ref[b, t - 1] + 0.1 * r[b, t]
    assert torch.allclose(s, ref, atol=1e-5)
    assert torch.equal(smoothing_weights(x, 0.0, EOT), torch.eye(CFG.block_size).expand(3, -1, -1))


def test_layers_not_listed_stay_dense():
    model, ref = fresh_model().eval(), fresh_model().eval()
    gated(model, layers=(2,))
    assert model.mlps()[0].gate_router is None and model.mlps()[1].gate_router is None
    outs = {}
    for name, m in (("gated", model), ("dense", ref)):
        h = m.mlps()[1].register_forward_hook(lambda _m, _i, out, name=name: outs.__setitem__(name, out))
        x = tokens()
        m(x, x)
        h.remove()
    assert torch.equal(outs["gated"], outs["dense"])


@pytest.mark.parametrize("score", ["relative", "share"])
def test_no_nan_with_a_silent_group_and_dead_neurons(score):
    model = fresh_model().train()
    r = gated(model, "evolved", score=score)
    lab = fixed_groups(HIDDEN, 8)
    model.set_neuron_masks({1: (lab != 3).astype(np.float32)})           # group 3 of layer 1 is silent
    r.set_neuron_ema(2, torch.zeros(HIDDEN))                             # every baseline of layer 2 is zero
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(5):
        x = tokens(seed=np.random.randint(1000))
        loss = model(x, x)[1]
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        r.step(opt)
        opt.zero_grad()
        assert all(torch.isfinite(g).all() for g in r.last_gate)
    assert not r.last_selected[0][..., 3].any()        # a silent group never wins


def test_gate_gradient_reaches_activations_only_with_gate_grad():
    """With gate_grad the gate itself is differentiable (the c_fc gradient changes); without it, the gate is a constant
    multiplier. The threshold, baseline and bias never receive gradient."""
    grads = {}
    for gg in (True, False):
        model = fresh_model().train()
        r = gated(model, k=3, gate_grad=gg)
        x = tokens()
        model(x, x)[1].backward()
        grads[gg] = model.mlps()[2].c_fc.weight.grad.clone()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert not r.bias.requires_grad and not r.tracker.neuron_ema[2].requires_grad
        assert not r.last_gate[1].requires_grad      # the recorded copy is detached
    assert not torch.allclose(grads[True], grads[False])


def test_share_score_is_each_groups_share_of_the_layer_norm():
    model = fresh_model().eval()
    r = gated(model, k=3, smooth=0.0, score="share")
    h = torch.randn(2, 5, HIDDEN)
    s = r.scores(0, h)
    norms = h.view(2, 5, 8, 32).norm(dim=-1)
    assert torch.allclose(s, 8 * norms / norms.sum(-1, keepdim=True), atol=1e-5)
    assert torch.allclose(s.mean(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.allclose(r.scores(0, 7 * h), s, atol=1e-5)          # scaling every activation changes nothing


def test_gate_grad_does_not_change_the_forward_pass():
    a, b = fresh_model().eval(), fresh_model().eval()
    gated(a, gate_grad=True)
    gated(b, gate_grad=False)
    x = tokens()
    assert torch.equal(a(x, x)[0], b(x, x)[0])


def test_bias_moves_against_over_selected_groups_and_is_not_a_parameter():
    model = fresh_model().train()
    r = gated(model, k=3)
    r.tracker.neuron_ema[1][fixed_groups(HIDDEN, 8) == 0] *= 1e-3      # group 0 of layer 1 looks hugely active
    x = tokens()
    model(x, x)
    sel = r.last_selected[0].float().mean(dim=(0, 1))
    r.after_step()
    over = sel > 3 / 8
    assert over[0] and (r.bias[0][over] < 0).all() and (r.bias[0][~over] > 0).all()
    assert not any(p is r.bias for p in model.parameters())


# ---------------------------------------------------------------------------------- groups
@pytest.mark.parametrize("sizes", [[64, 64, 64, 64], [20] * 12 + [16]])
def test_partition_has_exactly_n_groups(sizes):
    corr, _ = block_corr(sizes)
    labels = partition_to_n(corr, 8, min_size=16)
    assert labels.shape == (sum(sizes),) and set(labels.tolist()) == set(range(8))


def test_capped_move_moves_at_most_max_frac_and_keeps_min_size():
    corr, planted = block_corr([32] * 8)
    current = fixed_groups(HIDDEN, 8)[np.random.default_rng(0).permutation(HIDDEN)]  # unrelated to the planted groups
    new, moved = capped_move(current, planted, corr, 0.05, 16)
    assert moved == int(0.05 * HIDDEN) == (new != current).sum()
    assert np.bincount(new, minlength=8).min() >= 16
    assert len(set(np.bincount(new, minlength=8).tolist())) > 1        # groups grow and shrink
    agree = lambda lab: (lab[:, None] == lab[None, :])[planted[:, None] == planted[None, :]].mean()  # noqa: E731
    assert agree(new) > agree(current)                                   # moves go towards the proposal
    same, n = capped_move(planted, planted, corr, 0.05, 16)
    assert n == 0 and np.array_equal(same, planted)


def test_evolved_router_reclusters_and_caps_moves():
    model = fresh_model().train()
    r = GateRouter(model, layers=(1, 2), groups="evolved", cluster_start=3, cluster_every=3, min_size=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(9):
        assert (r.labels[0] is None) == (step < 3)
        x = tokens(seed=step)
        model(x, x)[1].backward()
        r.step(opt)
        opt.zero_grad()
    assert r.n_clusterings == 3 and r.moved_last[0] <= int(0.05 * HIDDEN) and r.stability[0] is not None
    for lab in r.labels:
        assert set(np.unique(lab).tolist()) == set(range(8))


# ---------------------------------------------------------------------------------- data and tags
def test_tags_on_known_text():
    def tags(text, domain):
        ids = ENC.encode_ordinary(text)
        return [(ENC.decode([i]), TAGS[t]) for i, t in zip(ids, tag_tokens(np.array(ids), domain, ENC))]
    math = dict(tags("Q: 37 + 48 = A: 85\nQ: 7 - 12 = A: -5", "math"))
    assert math[" 85"] == math["5"] == "digit_in_arith_answer"
    assert math[" 37"] == math[" 48"] == math[" 12"] == "other"
    prose = tags("Q: 1 + 2 = A: 3\nMia added 37 and 48 and got 85.\nQ: 4 + 5 = A: 9", "math")
    assert [t for tok, t in prose if tok in (" 37", " 48", " 85")] == ["digit_in_prose_arith"] * 3
    assert dict(prose)[" 1"] == "other" and dict(prose)[" 9"] == "digit_in_arith_answer"
    wp = dict(tags("Three plus four makes seven.", "math"))
    assert wp["Three"] == wp[" four"] == wp[" seven"] == "number_word_in_word_problem" and wp[" plus"] == "other"
    story = dict(tags("Mia was 7 and had three cats.", "story"))
    assert story[" 7"] == "digit_in_story" and story[" was"] == "story_word" and story[" three"] == "other"
    code = tags("def f(x):\n    y = 37 + 48\n\treturn y", "code")
    kw = {tok for tok, t in code if t == "code_keyword_or_indent"}
    assert {"def", "return", " ", "\t"} <= kw and dict(code)[" 37"] == "other"


def test_heldout_problems_never_appear_in_training_lines():
    import random
    rng = random.Random(0)
    for _ in range(20000):
        q = math_line(rng, holdout=True)
        a, op, b = q.split()[1:4]
        assert not is_heldout(int(a), op, int(b))
    probs = heldout_problems(500, seed=3)
    assert len({p for p, _ in probs}) == 500
    for prompt, ans in probs:
        a, op, b = prompt.split()[1:4]
        assert is_heldout(int(a), op, int(b)) and prompt.endswith("= A:")


def test_prepare_writes_deconf_data_and_tags(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "prepare.py"), "--synthetic", "--synthetic_tokens", "20000",
                    "--out_dir", str(tmp_path)], check=True, capture_output=True)
    meta = json.load(open(tmp_path / "meta.json"))
    assert all(v > 0 for v in meta["deconf_tokens_per_domain"].values()) and meta["tags"] == TAGS
    n = os.path.getsize(tmp_path / "train.bin") // 2
    assert os.path.getsize(tmp_path / "train_tags.bin") == n == os.path.getsize(tmp_path / "train_domain.bin")
    found = set()
    for d in ("story", "math", "code"):
        assert os.path.getsize(tmp_path / f"val_{d}_tags.bin") == os.path.getsize(tmp_path / f"val_{d}.bin") // 2
        found |= set(meta["val_tag_counts"][d])
    assert found == set(TAGS)
    dom = np.fromfile(tmp_path / "train_domain.bin", dtype=np.uint8)
    tags = np.fromfile(tmp_path / "train_tags.bin", dtype=np.uint8)
    assert (dom[tags == TAG_ID["digit_in_story"]] == 0).all() and (dom[tags == TAG_ID["digit_in_arith_answer"]] == 1).all()


# ---------------------------------------------------------------------------------- training runs
@pytest.fixture(scope="module")
def old_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "prepare.py"), "--synthetic", "--synthetic_tokens", "50000",
                    "--deconf_frac", "0", "--holdout_arith", "0", "--out_dir", str(d)], check=True, capture_output=True)
    return d


@pytest.mark.parametrize("groups", ["fixed", "evolved"])
def test_gated_mode_before_gating_reproduces_dense_exactly(groups, old_corpus, tmp_path):
    """Fixed and evolved share the dense phase, and their tracking does not disturb training."""
    train.main(["--config", SMOKE, "--data_dir", str(old_corpus), "--out_dir", str(tmp_path), "--device", "cpu",
                "--routing_mode", "gated", "--groups", groups, "--seed", "1337", "--cluster_start", "100000"])
    got = [r[:6] for r in read_metrics(tmp_path / "metrics.csv")]
    assert got == read_metrics(os.path.join(GOLDEN, "dense_smoke_metrics.csv"))


@pytest.fixture(scope="module")
def gated_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("runs")
    outs = {}
    for groups in ("fixed", "evolved"):
        out = base / f"gated_{groups}_s0"
        env = {**os.environ, "SMOKE": "1", "SEED": "0", "DEVICE": "cpu", "PYTHON": sys.executable, "OUT_DIR": out.as_posix(),
               "DATA_DIR": (base / "data").as_posix(), "ROUTING_MODE": "gated", "GATE_GROUPS": groups}
        r = subprocess.run([find_bash(), "run_all.sh"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        outs[groups] = out
    return outs


@pytest.mark.parametrize("groups", ["fixed", "evolved"])
def test_smoke_run_all_gated(groups, gated_runs):
    out = gated_runs[groups]
    for name in ["metrics.csv", "gating.csv", "gating_layers.csv", "groups_final.json", "summary.txt", "ablation.csv",
                 "context_sensitivity.json", "damage_table.csv", "damage_random.csv", "damage_summary.json",
                 "failure_trace.csv", "failure_trace_summary.json", "failure_trace_active.npz", "unlearn_summary.json"]:
        assert (out / name).stat().st_size > 0, name
    metrics = list(csv.DictReader(open(out / "metrics.csv", newline="")))
    assert all(np.isfinite(float(r[f"val_{d}"])) for r in metrics for d in ("story", "math", "code"))
    layers = list(csv.DictReader(open(out / "gating_layers.csv", newline="")))
    last = [r for r in layers if r["iter"] == metrics[-1]["iter"]]
    assert len(last) == 2 and all(r[f"ctx_{n}_effect"] != "" for r in last
                                  for n in ("number_words", "prose_digits", "qa_digits"))
    if groups == "evolved":
        assert all(int(r["moved_last"]) <= int(0.05 * 256) for r in last) and int(last[0]["n_clusterings"]) >= 2
    damage = json.load(open(out / "damage_summary.json"))
    assert damage["source"] == f"trained-{groups}" and damage["n_groups"] == 16
    assert sum(damage["classes"].values()) == 16 == sum(damage["format_classes"].values())
    assert "tag_digit_in_prose_arith" in damage["mean_damage"]
    trace = json.load(open(out / "failure_trace_summary.json"))
    assert trace["n_problems"] == 100 and len(trace["top_groups"]) == 3 and len(trace["ablation"]) == 4
    unlearn = json.load(open(out / "unlearn_summary.json"))
    assert len(unlearn["deleted_groups"]) == 3 and set(unlearn["before"]) == {"accuracy", "story", "math", "code"}
    ctx = json.load(open(out / "context_sensitivity.json"))
    assert set(ctx["results"]) == {"0.0", "0.9"} and ctx["trained_route_smooth"] == 0.9
    assert ctx["pair_kind"] == {"number_words": "matched", "prose_digits": "matched", "qa_digits": "format"}


def test_load_checkpoint_restores_the_gates(gated_runs):
    from common import load_checkpoint
    model, ckpt = load_checkpoint(str(gated_runs["evolved"] / "ckpt.pt"), "cpu")
    r = model.gate_router
    assert r is not None and r.groups == "evolved" and r.steps == ckpt["iter"] and r.active(0)
    saved = json.load(open(gated_runs["evolved"] / "groups_final.json"))
    assert r.labels[0].tolist() == saved["layers"]["0"]


def test_compare_writes_the_gated_table(gated_runs, tmp_path):
    import shutil
    results = tmp_path / "results"
    for groups, out in gated_runs.items():
        shutil.copytree(out, results / out.name, ignore=shutil.ignore_patterns("ckpt.pt", "*.png", "*.npz"))
    subprocess.run([sys.executable, os.path.join(ROOT, "eval", "compare.py"), "--results_dir", str(results)], check=True,
                   capture_output=True)
    rows = {r["group"]: r for r in csv.DictReader(open(results / "comparison_gated.csv", newline=""))}
    assert set(rows) == {"gated_fixed", "gated_evolved"}
    for r in rows.values():
        assert r["val_story"] != "" and r["arith_acc"] != "" and r["ctx_prose_digits_s0.9"] != ""
        assert r["unlearn_d_story"] != "" and r["format_ctx_qa_digits_s0.9"] != "" and r["format_n_functional"] != ""
    assert rows["gated_fixed"]["groups_near_equal"] == "1.0000"          # fixed groups are exactly equal
    sizes = list(csv.DictReader(open(results / "group_sizes.csv", newline="")))
    assert {r["run"] for r in sizes} == {"gated_fixed_s0", "gated_evolved_s0"} and len(sizes) == 4


def test_compare_flags_evolved_groups_that_stay_near_equal(tmp_path):
    results = tmp_path / "results"
    for name, labels in (("gated_evolved_s0", np.repeat(np.arange(8), 32)),
                         ("gated_evolved_s1", np.concatenate([np.zeros(100), np.repeat(np.arange(1, 8), 22)[:156]]))):
        d = results / name
        d.mkdir(parents=True)
        (d / "metrics.csv").write_text("iter,train_loss,val_story,val_math,val_code,lr,elapsed_s,tokens_per_sec\n"
                                       "100,1,3.0,2.0,2.5,1e-4,10,40000\n")
        (d / "damage_summary.json").write_text("{}")
        (d / "groups_final.json").write_text(json.dumps({"n_groups": 8, "layers": {"0": labels.astype(int).tolist()}}))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "compare.py"), "--results_dir", str(results)],
                       check=True, capture_output=True, text=True)
    assert "FLAG: gated_evolved_s0" in r.stdout and "FLAG: gated_evolved_s1" not in r.stdout


def test_smoothing_and_groups_name_the_output_folder():
    env = {**os.environ, "SMOKE": "1", "ROUTING_MODE": "gated", "GATE_GROUPS": "evolved", "ROUTE_SMOOTH": "0",
           "PYTHON": "false"}
    env.pop("OUT_DIR", None)
    r = subprocess.run([find_bash(), "run_all.sh"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert "outputs in out_gated_evolved_sm0_s0/" in r.stdout
