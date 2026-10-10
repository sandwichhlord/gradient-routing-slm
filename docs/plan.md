# Roadmap

**Research question:** does routing gradients by each block's own activation energy produce domain-specialised, ablatable MLP blocks without domain labels?

Compute figures below are estimates from the T4 numbers in the README (about 20-30 min per full run). None of them has been measured on a GPU yet.

## Done

- Data pipeline for three domains (stories, math, code) with per-domain validation sets and a per-token domain id file (`data/out/train_domain.bin`).
- Dense baseline GPT with the MLP split into 8 blocks, an ablation mask and activation-norm recording.
- Training with fp16 on a T4, sampling, loss curves, and the ablation matrix with specialization score S.
- Activation-based gradient routing (`model/routing.py`, `--routing`) with tests. Only smoke-tested on CPU, never at full scale.

## Remaining, in order

### 1. Full dense and routed runs (first, everything else depends on it)
- Run `bash run_all.sh` (dense) and `ROUTED=1 bash run_all.sh` (routed) at full scale on the T4.
- **Check first:** the routed run's `routing_stats.csv`. In the smoke test one block per layer won a slot on nearly every step. If the full run does the same, the load-balancing weight (lambda) and k need a sweep before anything else is worth running. A small sweep over lambda in {0.01, 0.1, 1} and k in {1, 2, 3} on shortened runs (1000 iterations) is cheap.
- **Output:** dense vs routed per-domain loss, S, and how concentrated the ablation damage is.
- **Success criterion (to agree on):** routed S clearly above dense S across seeds (task 6), at similar validation loss.
- **Cost:** about 1 hour.

### 2. Hard label-routing baseline
- Route by the known domain label, as in Gradient Routing: a fixed set of blocks per domain (for example 3 / 3 / 2 blocks, or one shared block). This is the label-supervised upper bound for specialisation.
- **Design decision:** our training batches are random windows of a mixed stream, so one window can span documents from different domains. Per-sequence label masks need domain-pure batches. Plan: draw each micro-batch from a single domain and cycle through the domains with gradient accumulation. Dense and routed runs keep their current sampling.
- Add `--routing label` to `train.py`, reusing the hooks in `GradientRouter` with a fixed mask per micro-batch. Tests: each domain's gradient reaches only its own blocks.
- **Cost:** about half a day of work, then one run.

### 3. Consolidation
- **Not yet defined, so this needs your decision.** My working definition: after the routing phase, freeze (or lower the learning rate of) the blocks that have become specialised (high per-block S), then continue training so that general capacity can still absorb new data. A second reading is "initialise the routed run from the dense checkpoint" (the dashed edge in `architecture.md`).
- Start with the second reading because it needs no new mechanism, only a `--init_from` flag. Compare routed-from-scratch against routed-from-dense.

### 4. Domain probes
- Train linear probes that predict the domain of a token window from each block's activations (use the existing activation hooks), per layer and block.
- Compare dense, routed and label-routed. Probe accuracy shows whether a block encodes domain information even when ablating it does little damage.
- Control: probe on shuffled labels, and the same probes on a randomly initialised model.
- **Cost:** one script, minutes to run.

### 5. Unlearning and continual-learning experiments
- **Unlearning:** ablate the blocks that matter most for domain X, then measure the loss increase on X against the other domains. Then run a relearning attack (a short fine-tune on X) and measure how quickly X comes back, for dense, routed and label-routed models. This follows the Gradient Routing evaluation.
- **Continual learning:** train on domains in sequence (stories, then math, then code). Measure forgetting of earlier domains for dense, routed, and dense with EWC (Fisher penalty, as in Kirkpatrick et al.). EWC has to be implemented (`eval/` or `train.py --ewc`).
- **Cost:** a few hours of work plus roughly 2-3 extra runs per arm.

### 6. Multi-seed runs
- At least 3 seeds per arm (dense, routed, label-routed, plus EWC where relevant), reporting mean and standard deviation. Without this, S differences between arms cannot be trusted. In the CPU smoke runs, mean S was 0.05 (dense) vs 0.08 (routed) on one seed with tiny evaluation sets, which says nothing yet.
- The `--seed` flag exists. Data preparation uses its own seed, so keep the data fixed across seeds unless data variance is also being measured.
- **Cost:** roughly 4 arms x 3 seeds x about 25 min = 5 hours of GPU time, spread over several Kaggle sessions. Check the weekly GPU quota first.
- Add a small aggregation script that reads the `out*/` folders and produces one table and plot.

## Risks and open questions

- **Block dominance.** One block per layer may always win selection, so the "specialisation" we measure would just be that block mattering. Check the per-block selection frequencies, and measure S with the dominant blocks excluded.
- **S is noisy** when ablation damage is tiny (blocks with near-zero damage have an unstable S). Report damage-weighted S alongside the plain mean, and use more evaluation batches than the default 20.
- **Scale.** About 10M non-embedding parameters on 40M tokens is small. Results may not carry over to larger models, so say so in any write-up.
- **Evaluation noise from synthetic math.** Math loss depends on the generator's number distribution, so only compare it between runs that use the same data.
