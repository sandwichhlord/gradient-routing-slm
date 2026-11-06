# Project status (refresher)

Written for someone who has forgotten the details. Last audited 2026-11-06.

## What this project is

We train a small GPT (10.6M parameters, plus a word table) on three kinds of text: children's stories, arithmetic, and Python code. The MLP in each layer is split into 8 blocks. The question: if only some blocks are allowed to learn from each piece of text, chosen by how active the blocks are and with no human labels, do blocks end up specialising by domain?

We measure this by switching off one block at a time (ablation) and seeing which domain's loss goes up. The specialization score S runs from 0 (damage spread evenly across domains) to 1 (damage in one domain).

## What exists

| Path | What it is |
|---|---|
| `data/prepare.py` | Builds `data/out/`: `train.bin`, `train_domain.bin` (domain id per token), `val_{story,math,code}.bin` |
| `model/model.py` | The GPT and the block hooks (ablation mask, activation norms) |
| `model/routing.py` | `GradientRouter`, the v0 routing method |
| `train.py` | Training loop. `configs/` holds the settings |
| `eval/` | `sample.py`, `plots.py`, `ablate.py` (ablation and S) |
| `tests/` | pytest suite |
| `run_all.sh` | Runs everything in order |
| `kaggle_run.ipynb` | Runs `run_all.sh` on a Kaggle T4 GPU |
| `docs/` | `findings.md` (what we learned), `plan.md` (roadmap), `literature.md`, `architecture.md`, `viva_notes.md` |
| `presentation/` | Mid-term deck and the script that built it |

## How to run things

```bash
pip install -r requirements.txt
python -m pytest tests/      # about a minute on CPU
SMOKE=1 bash run_all.sh      # tiny synthetic run, about a minute on CPU
bash run_all.sh              # full run; needs internet and really needs a GPU
```

For Kaggle, import `kaggle_run.ipynb`, turn on the T4 GPU and Internet, and click Run All.

## What each run produced

There are two real runs, both on a Kaggle T4, both 5000 iterations with seed 1337 on the same data (40.6M training tokens).

| | `results/dense_v0/` | `results/v0_batch_routing/` |
|---|---|---|
| Method | Plain dense model | v0 batch routing: top-2 of 8 blocks per layer per batch, lambda 0.01, unmasked balancing gradient |
| Val loss story / math / code | 2.120 / 1.467 / 2.497 | 2.270 / 1.510 / 2.659 |
| Mean S | 0.051 | 0.110 |
| Mean damage per ablated block | 0.038 nats | 0.405 nats |
| Training time | 19m25s | 20m51s |
| Checkpoint | `ckpt.pt` (345 MB, not in git) | `ckpt.pt` (345 MB, not in git) |

One seed per run, so the S difference is not established. See `docs/findings.md`.

## How the old results were classified

The old outputs were in `results/v0/` (not `results/_unsorted/`), in folders `baseline`, `results`, `results (2)` and `routed`. `routed` is the folder that was meant by `routed_results`. Everything was classified with certainty, so nothing was left unsorted.

| Old folder | Verdict | How we decided | What happened to it |
|---|---|---|---|
| `results/` (with `out/`) | **Dense run** | `ckpt.pt` config has no routing keys and the checkpoint has no `router` entry; `run.log` has no routing line; no `routing_stats.csv`; Tesla T4, 19m25s | Moved to `results/dense_v0/` |
| `results (2)/` (with `out_routed/`) | **v0 batch-routing run** | `ckpt.pt` config has `routing=True, route_k=2, route_lambda=0.01, route_mask_aux=False` and router state at step 5000; `run.log` says "gradient routing ON"; `routing_stats.csv` has `freq_ema` columns (6 x 8 x 20 rows); 20m51s | Moved to `results/v0_batch_routing/` |
| `baseline/` | Duplicate of the dense run | 9 of 9 files match `results/out/` by hash after normalising line endings (4 text files differ only in CRLF vs LF, from a git checkout on Windows); also identical to the copy in git at `results/baseline/` in commit `840901c` | Deleted |
| `results/baseline/` | Duplicate of the dense run | 9 of 9 files byte-identical to `results/out/` and `results/run.log` (a copy made on 2026-10-10) | Deleted |
| `routed/` | Duplicate of the v0 run | 10 of 10 files byte-identical to `results (2)/out_routed/` and its `run.log` | Deleted |
| `results*/__results___files/` | Rendered notebook images | Byte-identical to `loss_curves.png` and `ablation_heatmap.png` of the same run | Deleted |

Neither run was partial or failed: both have a checkpoint at iter 5000 and a log ending in "All 5 stages finished". The file timestamps are download times, so they do not say when a run happened.

`results/v1/` was an empty folder and was left alone. The deleted copies can still be recovered from git (commit `840901c`), and both checkpoints were kept.

## Git

- Tag `v0-batch-routing` marks the code as it was before routing v1.
- Uncommitted files in `presentation/` (`Minor_Group_55.pptx` and an edited `midterm_deck.pptx`) are yours and were left alone.

## Update: routing v1 (added after the tag)

- `--routing_mode {dense,batch,token,label}`, set through `ROUTING_MODE` in `run_all.sh`. `dense` and `batch` reproduce the v0 smoke metrics exactly (`tests/golden/`).
- `SEED` (default 0) controls init and batch order. The dataset and val sets stay fixed. Old runs used seed 1337.
- Outputs go to `out_<mode>_s<seed>/`. Put downloaded results in `results/<mode>_s<seed>/`, then run `python eval/compare.py`.
- `python -m pytest tests/`: 41 tests, about 10 minutes on CPU.
- **Token and label modes have only run on CPU.** No full-scale v1 result exists yet. See `docs/findings.md` and `docs/plan.md`.
