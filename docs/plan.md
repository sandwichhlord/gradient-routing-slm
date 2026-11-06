# Roadmap

**Research question:** can a model organise its own domain-specialised, ablatable MLP blocks during training, with no domain labels?

## Done

- Data pipeline for three domains with per-domain validation sets and a per-token domain id file (`train_domain.bin`).
- Dense baseline GPT with the MLP split into 8 blocks, an ablation mask and activation-norm recording.
- Training with fp16 on a T4, sampling, loss curves, and the ablation matrix with specialization score S.
- Routing v0 (`batch` mode), with one full-scale dense run and one full-scale routed run on the T4 at seed 1337. Routed mean S was 0.110 against 0.051 for dense, at a 3-7% loss cost. Tag `v0-batch-routing` marks this code.
- Routing v1 (`token` mode) and the oracle baseline (`label` mode), tested on CPU only. Per-domain routing logs, `eval/routing_plot.py`, `eval/compare.py`, and seeds through `SEED`.

## Remaining, in order

### 1. Full-scale runs on the T4 (next)

Run these in order, one Kaggle session each, using `ROUTING_MODE` and `SEED` in the notebook's first cell:

| Order | Mode | Seed | Why now |
|---|---|---|---|
| 1 | `token` | 0 | The first real test of v1. If it collapses or does not learn, stop and fix it before spending more GPU time |
| 2 | `dense` | 0 | The comparison point on the new seed scheme |
| 3 | `label` | 0 | The upper bound: how much S is possible with labels |
| 4-7 | `token`, `dense`, `label`, `batch` | 1 | Second seed |
| 8-11 | `token`, `dense`, `label`, `batch` | 2 | Third seed. Without several seeds, S gaps between modes cannot be trusted |
| 12 | `batch` | 0 | Completes seed 0. Lowest priority, since a v0 run at seed 1337 already exists |

Unzip each result into `results/<mode>_s<seed>/`, then run `python eval/compare.py`.

**Check first on the token run:**
- `routing_by_domain.png`: do domains select different blocks? If the three rows of each layer look the same, routing is not separating domains, whatever S says.
- `frac_all` in `routing_stats.csv`: is selection stable, or does it swing between blocks? The smoke runs showed bursts.
- The bias range: if it grows much larger than the spread of the scores, the bias is deciding selection and not the tokens.

**Success criterion (to agree on):** token S clearly above dense S across seeds, per-domain selection visibly different, and label S above token S, at a validation loss within a few percent of dense.

### 2. Tune v1 if needed
A small sweep of `route_gamma` and `route_k` on shortened runs (`MAX_ITERS=1000`) is cheap. Only do this after the first token run shows what needs fixing.

### 3. Domain probes
- Train linear probes that predict the domain from each block's activations, per layer and block, using the existing activation hooks.
- Compare dense, token and label. Probe accuracy shows whether a block encodes domain information even when ablating it does little damage, which also helps with the S confound below.
- Controls: shuffled labels, and a randomly initialised model.

### 4. Consolidation
**Not yet defined, so this needs your decision.** Working definition: after the routing phase, freeze or slow down the blocks that have become specialised, then keep training so that general capacity can absorb new data. A simpler reading is to initialise the routed run from the dense checkpoint, which only needs an `--init_from` flag. I would start with the simpler one.

### 5. Unlearning and continual-learning experiments
- **Unlearning:** ablate the blocks that matter most for domain X, then measure the loss increase on X against the other domains. Then run a relearning attack (a short fine-tune on X) and measure how quickly X comes back, for dense, token and label models. This follows the Gradient Routing evaluation.
- **Continual learning:** train on domains in sequence. Measure forgetting of earlier domains for dense, token, and dense with EWC (EWC has to be implemented).

## Risks and open questions

- **S confound.** S compares shares of damage, so it cannot tell "more domain-specific" from "more critical overall". In v0, ablation damage grew about 10x. `eval/compare.py` reports `mean_damage` next to S for this reason.
- **Mixed windows in token mode.** Token mode trains on the same mixed windows as dense and batch, so a token's loss also reaches the blocks of earlier tokens from other domains through attention. Label mode avoids this with pure windows. If token mode's domain separation is weak, trying pure windows there is a cheap experiment.
- **Label mode samples differently.** Pure windows never cross a domain boundary, and they cover some tokens less (on a real-data sample: 62% of story, 29% of math and 89% of code tokens can start a pure window). Keep this in mind when comparing its losses with the other modes.
- **Selection may not be token-specific.** With a small gamma, selection can swing as a group between blocks. This is the first thing to check in the token run.
- **S is noisy** when ablation damage is tiny. Report damage-weighted S alongside the plain mean if it matters, and consider more than the default 20 evaluation batches.
- **Scale.** About 10M non-embedding parameters on 40M tokens is small. Results may not carry over to larger models.
- **fp16 is untested locally.** The new paths ran on CPU, including bf16 autocast, but the first GPU run is the first real fp16 test. The token run is first in the order partly for this reason.
