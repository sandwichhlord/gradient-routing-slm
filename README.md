# gradient-routing-slm

Train a small GPT (nanoGPT-style) on a 3-domain corpus (stories, arithmetic, Python code), then ablate MLP blocks one at a time to see which domain each block serves.

The MLP hidden layer in every transformer layer is split into `n_blocks` (8) equal blocks. The baseline trains as an ordinary dense model. The block hooks (ablation mask and per-block activation norms) only switch on at inference, and are there for later gradient-routing experiments.

## Quick start

```bash
pip install -r requirements.txt
SMOKE=1 bash run_all.sh   # ~1 min on CPU: synthetic data, 2 layers, 64 dim, 100 iters
bash run_all.sh           # full run: real data + configs/baseline.py (needs internet; use a GPU)
python tests/test_model.py  # unit checks for the block ablation / norm hooks
```

`run_all.sh` runs `prepare -> train -> sample -> plots -> ablate` and stops at the first error. You can pass options through environment variables: `DEVICE=cpu|cuda`, `PYTHON=python3`, `TRAIN_ARGS="--max_iters 2000"`, `PREP_ARGS="--force"`.

## Kaggle (T4)

1. Push this repo to GitHub (public, or add your token to the URL).
2. Upload `kaggle_run.ipynb` to Kaggle and set `REPO_URL` in the first cell.
3. Settings: **Accelerator: GPU T4**, **Internet: On**. Click **Run All**.
4. Results are written to `/kaggle/working/out/`, along with `run.log`.

Expected wall time is about 20–30 min (see below). The notebook prints `DATA MODE: real` at the end. Any other value means the HuggingFace download failed and the run used the synthetic fallback.

## Files

| Path | What it does |
|---|---|
| `data/prepare.py` | Builds the GPT-2-tokenized corpus in `data/out/`: `train.bin` (documents from all domains shuffled together), `train_domain.bin` (domain id per token: 0 story, 1 math, 2 code), `val_{story,math,code}.bin` (about 0.5M tokens each), and `meta.json`. If data matching the same arguments already exists, it skips the rebuild. |
| `model/model.py` | The GPT. `set_ablation(layer, block)`, `set_block_mask(layer, mask)`, `clear_ablation()`, `record_block_norms(on)`, `get_block_norms()` |
| `train.py` | Training loop with fp16 AMP and GradScaler on CUDA, fp32 on CPU, warmup plus cosine LR, eval on each domain, ETA printout, `--resume` |
| `configs/baseline.py`, `configs/smoke.py` | Configs. Precedence: defaults < config file < CLI flags |
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

## Runtime on a T4 (estimate)

| Stage | Time |
|---|---|
| pip + clone | ~1 min |
| prepare (41.5M tokens streamed + tokenized) | ~3–6 min (network bound; measured ~150–200K tok/s locally) |
| train (5000 iters × 8192 tokens, ~30M params incl. tied embedding, fp16) | ~10–15 min |
| sample + plots | <1 min |
| ablate (48 blocks × 3 domains × 20 batches) | ~2–3 min |

`train.py` prints a measured ETA every 50 iterations. The ETA includes eval and checkpoint overhead.
