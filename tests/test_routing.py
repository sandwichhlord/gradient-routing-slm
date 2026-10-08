"""Gradient-routing tests. Run: python -m pytest tests/ -v"""
import csv
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.model import GPT, GPTConfig  # noqa: E402
from model.routing import GradientRouter  # noqa: E402

CFG = GPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=64, n_blocks=8)


def fresh_model():
    torch.manual_seed(0)
    return GPT(CFG)


def batch():
    torch.manual_seed(0)
    return torch.randint(0, 50257, (4, CFG.block_size))


def one_step(model, router=None):
    x = batch()
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    if router is None:
        loss.backward()
    else:
        router.backward(loss, router.route())


def block_grad_norms(model):
    """(n_layer, n_blocks) grad norm over each block's c_fc rows + c_proj columns."""
    out = []
    for mlp in model.mlps():
        bd = mlp.block_dim
        g_fc = mlp.c_fc.weight.grad.view(mlp.n_blocks, bd, -1)       # rows per block
        g_proj = mlp.c_proj.weight.grad.view(-1, mlp.n_blocks, bd)   # columns per block
        out.append(torch.sqrt(g_fc.pow(2).sum(dim=(1, 2)) + g_proj.pow(2).sum(dim=(0, 2))))
    return torch.stack(out)


def unselected_rows(router, layer):
    return ~router.selected[layer].bool().repeat_interleave(router.block_dim)


# (a) only the top-k blocks per layer get gradient ------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("mask_aux,aux_lambda", [(True, 0.01), (False, 0.0)])
def test_only_topk_blocks_get_gradients(k, mask_aux, aux_lambda):
    """Literal check of the whole MLP gradient: with mask_aux=True (aux on), and in the default mode for
    the task gradient (aux off; the aux part is checked separately below)."""
    model = fresh_model()
    router = GradientRouter(model, k=k, mask_aux=mask_aux, aux_lambda=aux_lambda)
    one_step(model, router)
    nonzero = block_grad_norms(model) > 0
    assert (nonzero.sum(dim=1) == k).all(), f"expected exactly {k} blocks per layer with grads, got {nonzero.sum(1)}"
    assert torch.equal(nonzero.float(), router.selected)
    top = router.last_energy.topk(k, dim=1).indices
    assert torch.equal(router.selected, torch.zeros_like(router.selected).scatter_(1, top, 1.0))
    for blk in model.transformer.h:  # non-MLP weights are not routed
        assert blk.attn.c_attn.weight.grad.abs().sum() > 0


def test_default_mode_unselected_blocks_get_only_aux_gradient():
    """Default mode: total grad = masked LM grad + unmasked aux grad. Non-selected blocks get exactly the
    aux gradient, only on their c_fc rows; the aux side graph reaches no other weights."""
    model = fresh_model()
    router = GradientRouter(model, k=2, aux_lambda=0.01)
    one_step(model, router)

    lm_model = fresh_model()                       # same step, LM term only
    lm_router = GradientRouter(lm_model, k=2, aux_lambda=0.0)
    one_step(lm_model, lm_router)
    assert torch.equal(lm_router.selected, router.selected)

    aux_model = fresh_model()                      # same step, aux term only (no masking)
    aux_router = GradientRouter(aux_model, k=2, aux_lambda=0.01)
    aux_model.train()
    x = batch()
    aux_model(x, x)
    aux_router.route().backward()

    lm_p, aux_p = dict(lm_model.named_parameters()), dict(aux_model.named_parameters())
    for name, p in model.named_parameters():
        aux_g = aux_p[name].grad if aux_p[name].grad is not None else torch.zeros_like(p)
        torch.testing.assert_close(p.grad, lm_p[name].grad + aux_g)
    for layer, (mlp, lm_mlp) in enumerate(zip(model.mlps(), lm_model.mlps())):
        rows = unselected_rows(router, layer)
        assert lm_mlp.c_fc.weight.grad[rows].abs().max() == 0, "LM gradient must skip unselected blocks"
        assert mlp.c_proj.weight.grad[:, rows].abs().max() == 0
        assert (mlp.c_fc.weight.grad[rows].abs().sum(dim=1) > 0).all(), "unselected blocks must get aux gradient"
    for blk in aux_model.transformer.h:
        assert blk.attn.c_attn.weight.grad is None, "aux side graph must not reach attention weights"


def test_masked_grads_equal_dense_grads_on_selected_blocks():
    """Routing only removes gradient from non-selected blocks; selected blocks' gradients are unchanged."""
    dense = fresh_model()
    one_step(dense)
    routed = fresh_model()
    router = GradientRouter(routed, k=2, aux_lambda=0.0)
    one_step(routed, router)
    for layer, (md, mr) in enumerate(zip(dense.mlps(), routed.mlps())):
        rows = ~unselected_rows(router, layer)
        torch.testing.assert_close(mr.c_fc.weight.grad[rows], md.c_fc.weight.grad[rows])
        torch.testing.assert_close(mr.c_proj.weight.grad[:, rows], md.c_proj.weight.grad[:, rows])
        assert mr.c_fc.weight.grad[~rows].abs().max() == 0
        assert mr.c_proj.weight.grad[:, ~rows].abs().max() == 0


def test_soft_epsilon_scales_unselected_grads():
    dense = fresh_model()
    one_step(dense)
    routed = fresh_model()
    router = GradientRouter(routed, k=2, epsilon=0.1, aux_lambda=0.0)
    one_step(routed, router)
    for layer, (md, mr) in enumerate(zip(dense.mlps(), routed.mlps())):
        rows = unselected_rows(router, layer)
        torch.testing.assert_close(mr.c_fc.weight.grad[rows], 0.1 * md.c_fc.weight.grad[rows])
        torch.testing.assert_close(mr.c_proj.weight.grad[:, rows], 0.1 * md.c_proj.weight.grad[:, rows])


# (b) routing off -> dense gradients ---------------------------------------------------------------
def test_routing_off_gradients_are_dense():
    model = fresh_model()
    assert all(not m.capture_energy for m in model.mlps()), "routing hooks must be off by default"
    one_step(model)
    assert (block_grad_norms(model) > 0).all(), "without routing every MLP block must receive gradient"


def test_router_remove_restores_dense():
    model = fresh_model()
    router = GradientRouter(model, k=2)
    one_step(model, router)
    router.remove()
    one_step(model)
    ref = fresh_model()
    one_step(ref)
    for a, b in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


# (c) no dead blocks after 100 smoke iterations ------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "prepare.py"), "--synthetic",
                    "--synthetic_tokens", "30000", "--out_dir", str(d)], check=True, capture_output=True)
    return d


def smoke_train(data_dir, out_dir, *extra):
    import train
    return train.main(["--config", os.path.join(ROOT, "configs", "smoke.py"), "--data_dir", str(data_dir),
                       "--out_dir", str(out_dir), "--device", "cpu", "--routing", "--max_iters", "100", *extra])


def test_no_dead_blocks_after_100_smoke_iters(smoke_data, tmp_path):
    """Dead = never selected in 100 iterations, or (currently unselected and) receiving no gradient at all,
    i.e. unable to ever win selection again."""
    out = smoke_train(smoke_data, tmp_path)
    router, model = out["router"], out["model"]
    assert router.steps == 100
    assert not router.dead_blocks(), f"never-selected blocks (layer, block): {router.dead_blocks()}\n{router.summary()}"
    assert (router.counts.sum(dim=1) == router.k * router.steps).all()

    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randint(0, 50257, (8, out["config"]["block_size"]))
    router.backward(model(x, x)[1], router.route())
    for layer, mlp in enumerate(model.mlps()):
        rows = unselected_rows(router, layer)
        assert (mlp.c_fc.weight.grad[rows].abs().sum(dim=1) > 0).all(), f"layer {layer}: unselected block has no grad"

    with open(tmp_path / "routing_stats.csv") as f:
        rows = list(csv.DictReader(f))
    n = out["config"]["n_layer"] * out["config"]["n_blocks"]
    assert rows and len(rows) % n == 0
    assert {(int(r["layer"]), int(r["block"])) for r in rows} == {(l, b) for l in range(2) for b in range(8)}


def test_literal_masking_lets_blocks_die(smoke_data, tmp_path):
    """Documents why mask_aux defaults to False: if the aux gradient is masked too (lambda=0.01), routing
    collapses onto a fixed set of blocks and the rest never get selected (nor any MLP gradient)."""
    out = smoke_train(smoke_data, tmp_path, "--route_mask_aux")
    assert out["router"].dead_blocks(), "expected collapse with literal masking; if this changed, revisit the default"
