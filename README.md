# gradient-routing-slm

Train a small GPT (nanoGPT-style) on a 3-domain corpus (stories, arithmetic, Python code), then ablate MLP blocks one at a time to see which domain each block serves.

The MLP hidden layer in every transformer layer is split into `n_blocks` (8) equal blocks. The baseline trains as an ordinary dense model. The block hooks (ablation mask and per-block activation norms) only switch on at inference, and are there for later gradient-routing experiments.

## Quick start

```bash
pip install -r requirements.txt
SMOKE=1 bash run_all.sh   # ~1 min on CPU: synthetic data, 2 layers, 64 dim, 100 iters
bash run_all.sh           # full run: real data + configs/baseline.py (needs internet; use a GPU)
ROUTED=1 bash run_all.sh  # full run with gradient routing (configs/routed.py) -> out_routed/
python -m pytest tests/   # block hooks + routing tests (~1 min on CPU)
```

`run_all.sh` runs `prepare -> train -> sample -> plots -> ablate` and stops at the first error. You can pass options through environment variables: `ROUTED=1`, `DEVICE=cpu|cuda`, `PYTHON=python3`, `TRAIN_ARGS="--max_iters 2000"`, `PREP_ARGS="--force"`.

## Kaggle (T4)

1. Push this repo to GitHub (public, or add your token to the URL).
2. Upload `kaggle_run.ipynb` to Kaggle and set `REPO_URL` in the first cell.
3. Settings: **Accelerator: GPU T4**, **Internet: On**. Click **Run All**.
4. Results are written to `/kaggle/working/out/`, along with `run.log`. Set `ROUTED = True` in the first cell to train the routed model instead (outputs in `/kaggle/working/out_routed/`).

Expected wall time is about 20–30 min (see below). The notebook prints `DATA MODE: real` at the end. Any other value means the HuggingFace download failed and the run used the synthetic fallback.

## Files

| Path | What it does |
|---|---|
| `data/prepare.py` | Builds the GPT-2-tokenized corpus in `data/out/`: `train.bin` (documents from all domains shuffled together), `train_domain.bin` (domain id per token: 0 story, 1 math, 2 code), `val_{story,math,code}.bin` (about 0.5M tokens each), and `meta.json`. If data matching the same arguments already exists, it skips the rebuild. |
| `model/model.py` | The GPT. `set_ablation(layer, block)`, `set_block_mask(layer, mask)`, `clear_ablation()`, `record_block_norms(on)`, `get_block_norms()` |
| `train.py` | Training loop with fp16 AMP and GradScaler on CUDA, fp32 on CPU, warmup plus cosine LR, eval on each domain, ETA printout, `--resume` |
| `model/routing.py` | `GradientRouter`: activation-based gradient routing over MLP blocks (see Method). Only active with `--routing` |
| `configs/baseline.py`, `configs/routed.py`, `configs/smoke.py` | Configs. Precedence: defaults < config file < CLI flags. `routed.py` = baseline + routing |
| `eval/sample.py` | `out/samples.txt`: 3 samples for each of 3 prompts |
| `eval/plots.py` | `out/loss_curves.png` |
| `eval/ablate.py` | `out/ablation.csv`, `ablation_heatmap.png`, `ablation_summary.txt`, `block_norms.csv`. Before the sweep it asserts that the mask actually changes the logits. |

### Data
- **Stories:** `roneneldan/TinyStories`, streamed (`--story_tokens`, default 20M).
- **Math:** synthetic lines such as `Q: 347 x 58 = A: 20126`. Operators are + - x, with 1–3 digit operands, grouped 20 lines per document (`--math_tokens`, default 10M).
- **Code:** `codeparrot/codeparrot-clean-valid`, field `content`, streamed. Files are capped at 4096 tokens (`--code_tokens`, default 10M).
- `--synthetic` generates tiny template-based fake data for all 3 domains. The script also falls back to this automatically, with a loud warning, if the HuggingFace download fails.

### Ablation metric
For each block, damage on domain d is `loss_ablated[d] - loss_unablated[d]`. All conditions are scored on the same fixed validation windows. The specialization score is `S = 1 - H(p)/log 3`, where `p = max(damage,0) / sum(max(damage,0))`. S is 0 when damage is spread evenly across domains and 1 when it falls on a single domain. Blocks that do no damage anywhere have no defined S and are left out of the reported mean.

## Method: activation-based gradient routing

With `--routing`, every training step decides, separately for each layer, which MLP blocks get to learn from the batch.

Let $h_{l,b}(t) \in \mathbb{R}^{d/n}$ be the post-GELU hidden activation of block $b$ in layer $l$ at token $t$, where $n=8$ blocks and $d = 4\,n_\text{embd}$.

1. **Energy.** $a_{l,b} = \frac{1}{|T|}\sum_{t \in T} \lVert h_{l,b}(t)\rVert_2$, averaged over all tokens in the micro-batch.
2. **Selection.** $S_l = \operatorname{top\text{-}k}_b\, a_{l,b}$, with $k=2$ by default (`--route_k`).
3. **Gradient mask.** Block $b$ owns rows $[b\,d/n, (b+1)\,d/n)$ of `c_fc.weight` (the up-projection) and the same columns of `c_proj.weight` (the down-projection). Their task-loss gradient is multiplied by $m_{l,b} = 1$ if $b \in S_l$, else $\epsilon$. The default $\epsilon=0$ (`--route_eps`) is hard routing. The mask is applied with tensor hooks on the weights, so each micro-batch is routed separately. The forward pass is unchanged, and attention, embeddings and LayerNorms train densely.
4. **Load balancing.** Each block keeps an EMA of its selection frequency, $f_{l,b} \leftarrow \beta f_{l,b} + (1-\beta)\,\mathbb{1}[b \in S_l]$, with $\beta = 0.99$ and initial value $k/n$. Its activation share is $p_{l,b} = a_{l,b} / \sum_{b'} a_{l,b'}$. The training objective becomes
   $$\mathcal{L} = \mathcal{L}_\text{LM} + \lambda \sum_{l,b} f_{l,b}\, p_{l,b}, \qquad \lambda = 0.01.$$
   $f$ is treated as a constant. The aux gradient lowers the energy share of blocks that are selected often and raises it for blocks that are selected rarely. At uniform balance the aux term equals $\lambda\, n_\text{layer}\, k/n$.

**Why the aux gradient bypasses the mask (`route_mask_aux=False`, the default).** In the default mode, $a_{l,b}$ for the aux term is recomputed on a side graph from `x.detach()`, so its gradient reaches only that layer's `c_fc` rows, and it is not masked. This is the same role the balancing loss plays for an MoE router.

If the aux gradient is masked as well (`--route_mask_aux`), training collapses. A block that loses selection gets no gradient that could make it win again, while AdamW weight decay shrinks it, so routing locks onto a fixed $k$ blocks per layer. In the 100-iteration smoke run at $\lambda=0.01$, 9 of 16 blocks were never selected. Over 300 iterations, 12 of 16 were dead (that is, exactly $k$ alive per layer). Even $\lambda=1$ left 6 of 16 idle. `tests/test_routing.py::test_literal_masking_lets_blocks_die` records this behaviour.

In the default mode, idle blocks keep receiving balancing gradient and do come back into selection. One block per layer still tends to win a slot on most steps.

**Caveat.** A block whose gradient is zeroed still moves slightly under AdamW: its momentum carries on for a few steps, and weight decay keeps applying. Routing controls what each block learns *from*, not whether its weights stay frozen.

`out_routed/routing_stats.csv` logs, at every eval, `freq_ema`, the cumulative `selected_count` / `selected_frac`, and the last `energy` for each (layer, block).

Tests (`python -m pytest tests/`) check the following:
- (a) Only the top-$k$ blocks per layer receive MLP gradient. This holds for the total gradient with `mask_aux`, and for the task gradient in the default mode. In the default mode, non-selected blocks receive exactly the aux gradient and nothing else.
- (b) With routing off, gradients are dense, and `router.remove()` restores dense gradients bit-for-bit.
- (c) After 100 smoke iterations, no block is dead: every block has been selected, and every currently unselected block still receives gradient.

## Runtime on a T4 (estimate)

| Stage | Time |
|---|---|
| pip + clone | ~1 min |
| prepare (41.5M tokens streamed + tokenized) | ~3–6 min (network bound; measured ~150–200K tok/s locally) |
| train (5000 iters × 8192 tokens, ~30M params incl. tied embedding, fp16) | ~10–15 min |
| sample + plots | <1 min |
| ablate (48 blocks × 3 domains × 20 batches) | ~2–3 min |

`train.py` prints a measured ETA every 50 iterations. The ETA includes eval and checkpoint overhead.

A routed run costs a little more per iteration. It re-runs each layer's `c_fc` on detached inputs and does a second, small backward pass for the aux loss, adding roughly 10% (estimated, not measured on a T4).
