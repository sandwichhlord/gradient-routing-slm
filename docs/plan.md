# Roadmap

**Research question:** can a model organise its own domain-specialised, ablatable groups of neurons during training, with no domain labels?

## Done

- Data pipeline for three domains with per-domain validation sets and a per-token domain id file (`train_domain.bin`).
- Dense baseline GPT with the MLP split into 8 blocks, an ablation mask and activation-norm recording.
- Training with fp16 on a T4, sampling, loss curves, and the ablation matrix with specialization score S.
- Routing v0 (`batch` mode), with one full-scale dense run and one full-scale routed run on the T4 at seed 1337. Routed mean S was 0.110 against 0.051 for dense, at a 3-7% loss cost. Tag `v0-batch-routing` marks this code.
- Routing v1 (`token` mode) and the oracle baseline (`label` mode). Per-domain routing logs, `eval/routing_plot.py`, `eval/compare.py`, and seeds through `SEED`. One full token run, which failed its health check.
- Routing v2 (`assembly` mode), its evaluation (`eval/ablate_assemblies.py`, `eval/assembly_probe.py`) and health checks, tested on CPU only.

## Remaining, in order

### 1. Assembly runs on the T4 (next)

Set these in the notebook's first cell. Unzip each result into `results/<zip name>/`, then run `python eval/compare.py`.

| Order | Run | Notebook settings | Estimated time |
|---|---|---|---|
| 1 | `dense_s0` with the post-hoc control | `ROUTING_MODE="dense"`, `POSTHOC=True` | about 50 min |
| 2 | `assembly_s0` | `ROUTING_MODE="assembly"`, `POSTHOC=False` | about 50 min |
| 3 | If healthy: `assembly_r0.5_s0`, `assembly_r2.0_s0` | `RESOLUTION=0.5`, then `RESOLUTION=2.0` | about 50 min and 60-90 min |
| 4 | 3 seeds of dense and assembly | `SEED=1`, then `SEED=2`, for each | about 50 min each |
| 5 | Unlearning and continual learning | See task 4 | |
| 6 | Consolidation | See task 5 | |

Run 1 does both steps of "dense_s0, then post-hoc on dense_s0" in one session. Post hoc can also run alone on an uploaded checkpoint with `POSTHOC_CKPT`. An optional extra control after `assembly_s0` is a dense run with `LR_SCALE` set from its `summary.txt` (see the Adam concern in `findings.md`, section 8).

**Check on `assembly_s0` before going further, in order:**
1. `metrics.csv`: loss falling smoothly, no NaN, and `act_norm_l*` near the dense values (`token_s0` reached about 180 against about 4).
2. `assemblies.csv`: no `degenerate` flags, and ARI rising over training. `assembly_health.png` shows both.
3. `assembly_summary.txt`: real S above the random baseline, and above `dense_s0/posthoc`.
4. `assembly_domain_profile.csv` and `assembly_probe.csv`: do assemblies respond to domains, or only to frequent tokens?

**Success criterion (to agree on):** assembly S clearly above the dense post-hoc S across seeds, at a validation loss within a few percent of dense, with no blow-up.

### 2. Older modes
`token_s0` failed its health check (`findings.md`, section 6) and further token seeds are on hold. `label_s0` was not run, so the cause of that failure is not separated. It would still answer that question if it becomes relevant, and it remains the oracle baseline for any later comparison.

### 3. Domain probes
- Train linear probes that predict the domain from each block's activations, per layer and block, using the existing activation hooks.
- Compare dense, assembly and label. Probe accuracy shows whether a block encodes domain information even when ablating it does little damage, which also helps with the S confound below.
- Controls: shuffled labels, and a randomly initialised model.

### 4. Unlearning and continual-learning experiments
- **Unlearning:** ablate the blocks that matter most for domain X, then measure the loss increase on X against the other domains. Then run a relearning attack (a short fine-tune on X) and measure how quickly X comes back, for dense, token and label models. This follows the Gradient Routing evaluation.
- **Continual learning:** train on domains in sequence. Measure forgetting of earlier domains for dense, token, and dense with EWC (EWC has to be implemented).

### 5. Consolidation (last)
Consolidation means groups hardening as they mature, to reduce forgetting. Starting a routed run from the dense checkpoint is a different idea and is not what this item means. `--consolidate` exists but is off by default, and Adam probably cancels it as implemented (`findings.md`, section 8), so it needs rethinking before use.

## Risks and open questions

- **S confound.** S compares shares of damage, so it cannot tell "more domain-specific" from "more critical overall". In v0, ablation damage grew about 10x. `eval/compare.py` reports `mean_damage` next to S for this reason.
- **Mixed windows in token mode.** Token mode trains on the same mixed windows as dense and batch, so a token's loss also reaches the blocks of earlier tokens from other domains through attention. Label mode avoids this with pure windows. If token mode's domain separation is weak, trying pure windows there is a cheap experiment.
- **Label mode samples differently.** Pure windows never cross a domain boundary, and they cover some tokens less (on a real-data sample: 62% of story, 29% of math and 89% of code tokens can start a pure window). Keep this in mind when comparing its losses with the other modes.
- **Selection may not be token-specific.** With a small gamma, selection can swing as a group between blocks. This is the first thing to check in the token run.
- **S is noisy** when ablation damage is tiny. Report damage-weighted S alongside the plain mean if it matters, and consider more than the default 20 evaluation batches.
- **Scale.** About 10M non-embedding parameters on 40M tokens is small. Results may not carry over to larger models.
- **fp16 is untested locally.** The new paths ran on CPU, including bf16 autocast, but the first GPU run is the first real fp16 test. The token run is first in the order partly for this reason.
