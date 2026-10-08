"""Quick CPU checks of the block-split MLP hooks. Run: python -m pytest tests/ (or python tests/test_model.py)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import GPT, GPTConfig  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=64, n_blocks=8)
    model = GPT(cfg).eval()
    x = torch.randint(0, 50257, (2, 32))
    with torch.no_grad():
        base = model(x, x)[0]
        # hooks are off by default
        assert all(m.block_mask is None and not m.record_norms for m in model.mlps())
        # every single-block ablation changes the logits
        for layer in range(cfg.n_layer):
            for block in range(cfg.n_blocks):
                model.set_ablation(layer, block)
                assert (model(x, x)[0] - base).abs().max() > 1e-6, (layer, block)
        # zeroing a block == zeroing the matching slice of c_proj's input columns
        model.set_ablation(1, 3)
        abl = model(x, x)[0]
        model.clear_ablation()
        ref = GPT(cfg).eval()
        ref.load_state_dict(model.state_dict())
        bd = 4 * cfg.n_embd // cfg.n_blocks
        ref.transformer.h[1].mlp.c_proj.weight[:, 3 * bd:4 * bd] = 0
        assert torch.allclose(ref(x, x)[0], abl, atol=1e-5), "ablation != zeroing c_proj columns"
        # clearing restores the dense model exactly
        assert torch.equal(model(x, x)[0], base)
        # norm recording
        model.record_block_norms(True)
        model(x, x)
        norms = model.get_block_norms()
        assert norms.shape == (cfg.n_layer, cfg.n_blocks) and (norms > 0).all()
        h = model.transformer.h[0]
        hid = h.mlp.gelu(h.mlp.c_fc(h.ln_2(model.transformer.wte(x) + model.transformer.wpe(torch.arange(32))
                                            + h.attn(h.ln_1(model.transformer.wte(x) + model.transformer.wpe(torch.arange(32)))))))
        expect = hid.view(2, 32, cfg.n_blocks, bd).norm(dim=-1).mean(dim=(0, 1))
        assert torch.allclose(norms[0], expect, atol=1e-5), "layer-0 block norms mismatch"
        model.record_block_norms(False)
        assert torch.equal(model(x, x)[0], base)
    print("tests/test_model.py: all checks passed")


def test_block_hooks():
    main()


if __name__ == "__main__":
    main()
