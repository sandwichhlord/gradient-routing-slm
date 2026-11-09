# Findings

## 1. Routing collapse when the balancing gradient is masked

**Scope.** All numbers in this section come from CPU smoke runs: synthetic data, 2 layers x 8 blocks = 16 blocks, `n_embd` 64, one seed, 100-300 iterations. They are a debugging record, not a result at full scale. The full-scale T4 run is in section 2.

**What happened.** The first implementation applied the gradient mask to everything, including the load-balancing loss (`--route_mask_aux`). Test (c) "no dead blocks after 100 iterations" failed. A "dead" block is one that is never selected and so never receives an MLP gradient.

| Setup (smoke scale) | Blocks with no selection |
|---|---|
| Masked aux, lambda 0.01, 100 iters (never selected at all) | 9 of 16 |
| Masked aux, lambda 0.1, 100 iters | 4 of 16 |
| Masked aux, lambda 1 and 10, 100 iters | 0 of 16 |
| Masked aux, lambda 0.01, 300 iters (idle in the last 50 iters) | 12 of 16, exactly 2 alive per layer |
| Masked aux, lambda 1, 300 iters (idle in the last 50 iters) | 6 of 16 |

So a larger lambda only delays the collapse, it does not prevent it. At lambda 1 the 100-iteration test passes and the 300-iteration run still collapses.

**Likely mechanism (hypothesis, not isolated by an experiment).** Selection is by activation energy, and only selected blocks get gradient, so selected blocks learn to produce larger activations and keep winning. A block that loses gets no gradient that could change its energy, so nothing can bring it back. AdamW weight decay and momentum decay probably shrink idle blocks further. We did not run an experiment that separates these effects.

**Fix.** The balancing loss is computed on a side graph from `x.detach()`, so its gradient reaches only the `c_fc` rows of its own layer, and it is not masked (`route_mask_aux=False`, the default). The task-loss gradient is still masked to the top-k blocks. Idle blocks therefore keep receiving a gradient that raises their energy share. `tests/test_routing.py` checks that non-selected blocks get exactly the aux gradient and nothing else, and `test_literal_masking_lets_blocks_die` keeps the collapse as a regression test.

**What the fix does not solve.** With the fix, no block is permanently dead, but selection is still uneven:

| Default mode (aux gradient unmasked), 300 iters | Blocks unselected, per 50-iter window |
|---|---|
| lambda 0.01 | 0, 5, 4, 5, 4, 7 (of 16) |
| lambda 0.1 | 0, 3, 1, 5, 2, 4 |
| lambda 1 | 0, 3, 3, 9, 3, 7 |

In the lambda 0.01 run, one block in each layer (0.5 and 1.7) was selected in 49-50 of the 50 iterations in each of the last two windows. So at smoke scale routing was bursty and a few blocks dominated. At full scale this did not persist (section 2).

**Cost of routing at smoke scale** (300 iters, lambda 0.01, single seed, final validation loss):

| Domain | Dense | Routed (default mode) | Difference |
|---|---|---|---|
| story | 1.795 | 2.030 | +0.235 |
| math | 3.010 | 3.111 | +0.101 |
| code | 2.171 | 2.318 | +0.147 |

Routing costs loss at this scale, as expected from training only 2 of 8 blocks per layer on each step. At full scale the cost was smaller in relative terms, 3-7% (section 2).

## 2. Full-scale runs on the T4 (5000 iterations, one seed)

Full tables and plots are in "Results so far" in the README. Dense and routed runs share the seed, data and config, apart from `--routing`.

| | Dense | Routed |
|---|---|---|
| Mean S | 0.051 | 0.110 |
| Damage-weighted S | 0.059 | 0.160 |
| Blocks with S above 0.2 | 0 | 8 |
| Val loss story / math / code | 2.120 / 1.467 / 2.497 | 2.270 / 1.510 / 2.659 |
| Mean damage per ablated block (nats) | 0.038 | 0.405 |

**Dense.** S is low, and damage differs mostly by layer (layer 0 blocks matter 5-9x more than later layers), not by domain.

**Routed.**
- **Specialization went up, modestly.** Mean S roughly doubled and 8 blocks passed S = 0.2, but the median only moved from 0.04 to 0.07. The five highest-S blocks all hurt math most, so much of the effect is math versus the rest.
- **Ablation damage grew about 10x.** Routing seems to have concentrated function into fewer blocks (for example block 5.0 costs 1.52 nats on math, and layer 0 block 6 costs 1.36 on story). S compares shares of damage, so it cannot tell "more domain-specific" from "more critical overall". The correlation between a block's S and its total damage is 0.31 in the routed run and 0.18 in the dense run, so high-S blocks tend to be high-damage blocks.
- **Selection was balanced and no block died.** 47 of 48 blocks were selected 15-31% of the time over the run. Layer 0 block 6 was selected 70% of the time (39% in the last 1000 iterations). So the smoke-scale dominance did not carry over, except in that one block.
- **Loss cost 3-7%** (math +0.043, story +0.150, code +0.162 nats). Training took 7% longer.
- **Activations did not become more domain-specific.** The spread of each block's mean activation norm across the three domains, (max - min) / mean, averaged 0.295 for dense and 0.306 for routed. The specialization shows up in ablation damage and not in how strongly blocks fire per domain.

**Limits.** One seed per arm, 20 evaluation batches per ablation, a small model, and no label-routed upper bound yet. The S difference between the two runs is not established until it holds across seeds.

## 3. Why v0 was replaced: its failure modes

- **One choice per batch.** A batch mixes stories, math and code, so one selection averages over all three. Selection cannot follow the domain of each token, which is the thing we want it to learn.
- **Selection rewards magnitude.** Raw energy favours blocks that are already large. On the T4, layer 0 block 6 was selected 70% of the time, and ablation damage grew about 10x.
- **The balancing loss competes with the task.** It needed its gradient to bypass the mask (section 1), so idle blocks were never truly isolated.
- **AdamW moved masked blocks.** Zero gradient does not stop momentum or weight decay.
- **Gradient still reached earlier layers through unselected blocks,** because v0 masks at the weights.

## 4. The W_down masking problem

Moving to per-token masks needs masking on activations, since weight hooks only see a gradient already summed over tokens. The obvious way, `h*m + (h*(1-m)).detach()`, is wrong: W_down is multiplied by the full activation, so it still receives gradient from every block. `test_naive_masking_of_h_alone_leaks_into_w_down` demonstrates this.

**v0 did not have this bug.** Its weight hooks zero both W_up rows and W_down columns, per batch. The bug is a trap for per-token masking, and v1 avoids it by computing `F.linear(h_sel, W_down) + F.linear(h_rest, W_down.detach())`.

## 5. v1 design and what we have measured so far

Design: per-token selection on relative scores (norm / EMA), bias balancing with no aux loss, masking on both projections, and truly skipped optimizer updates for zero-gradient slices. Details are in the README Method section.

**A limit we found while testing:** "a token's loss only reaches its own blocks" is true only in the last layer, or for the first token. Attention carries a later token's loss back through earlier tokens, so earlier layers receive the union of blocks selected by the token and those before it. Label mode would leak across domains on mixed windows, so it trains on domain-pure windows. Token mode still uses mixed windows.

**CPU measurements only (2 layers, 64 dim, one seed). None of this is a full-scale result.**

| Check | Result |
|---|---|
| Smoke data, 300 steps, cumulative selection | 0.17-0.34 (seed 1337), 0.18-0.32 (seed 0), no dead blocks |
| Same runs, last 50-step window | 0.016-0.37 and 0.005-0.34: bursty |
| Real-data sample, 500 steps, last 100-step window | 0.15-0.43 |
| Real-data sample, 500 steps, per-domain selection gap (max - min across domains, mean over blocks) | token 0.48, batch 0.17 |
| Same run, val loss story / math / code | token 5.62 / 2.64 / 5.53, batch 5.36 / 2.55 / 5.44 |

The token model's tokens from different domains pick visibly different blocks, with some math cells at 0.00 where story is above 0.5. This is encouraging but is not specialization: selection differing by domain does not show that ablating a block hurts only that domain. Part of it may be simple input statistics. The bias was still growing at step 500 (-0.28 to +0.32).

Speed at full model size on CPU (batch 4, two 40-iteration passes, tokens/s): dense 330 and 330, batch 328 and 302, token 363 and 343, label 369 and 335. Overhead is within noise on CPU. It has not been measured on a GPU.

## 6. First full-scale token run (token_s0): separates domains, but training is unhealthy

T4, 5000 iterations, seed 0. One run. fp16 is not the cause: Adam took 4998 of 5000 steps.

| Check | Result | Verdict |
|---|---|---|
| Loss falls smoothly, no NaN | No NaN, but the best point is near iteration 1250-3750 and the loss jumps around (final train 3.31, dense 2.04) | **Fail** |
| Domains pick different blocks | Mean per-domain gap 0.34 (0.42 at its peak). Clear in layers 0-2, weak in layers 3-5 | Pass |
| No block stuck at 0% | Whole-run fractions 0.246-0.256, last window 0.240-0.259, no dead blocks | Pass |
| S against dense_v0 | Mean S 0.208 vs 0.051, but see below | Not trustworthy yet |

| Final val loss | dense_v0 | v0_batch_routing | token_s0 |
|---|---|---|---|
| story | 2.120 | 2.270 | 3.649 (+72% on dense) |
| math | 1.467 | 1.510 | 1.929 (+32%) |
| code | 2.497 | 2.659 | 3.961 (+59%) |

**Activations blew up.** Measured from the checkpoints on real validation windows:

| | dense_v0 | v0_batch_routing | token_s0 |
|---|---|---|---|
| Block activation norm, layer 3 | 3.1 | 4.2 | 178 |
| Block activation norm, layer 5 | 4.1 | 5.5 | 180 |
| MLP output / MLP input norm, layer 5 | 0.60 | 0.54 | 66 |
| Cancellation ratio, layer 3: norm of summed block outputs / sum of block output norms (1 = none) | 0.52 | 0.34 | 0.075 |
| Mean damage per ablated block (nats) | 0.038 | 0.405 | 7.3 |
| Largest damage to one domain (nats) | 0.10 | 1.52 | 29.7 |

Blocks became very large and mostly cancel each other. **Hypothesis, not tested:** each token can only train its 2 selected blocks, so those learn to offset the output of the 6 it cannot change, and this feeds on itself. Whether the cause is the masking or the changing selection is unknown. Label mode uses the same masking with fixed selection, so it can separate the two.

**How to read S here.** Mean S rose to 0.208, but damage-weighted S is 0.079, below v0's 0.160. Damage in layers 4-5 is huge and shared across domains (block 5.5 costs +24 to +30 nats on all three). The real signal is in the early layers: mean S is 0.46 and 0.36 in layers 0-1, and damage follows selection there (correlation between selection share and damage share per domain: +0.86, +0.89, +0.80 in layers 0-2, against -0.03, +0.20, +0.49 in layers 3-5). For example, blocks 0.2 and 0.3 are selected by 76% of math tokens and mostly hurt math. So early-layer domain separation looks real in this one run, in a model that trains badly.

Speed: training took 21m06s and about 39.7K tokens/s (+9% time on dense). The whole pipeline took 25m56s.

## 7. Routing v2: self-organizing assemblies with graded plasticity

**Why we pivoted.** The 8 x 192 blocks are arbitrary: nothing in the network makes those neurons belong together. And `token_s0` showed that hard per-token masking can go with blocks that grow huge and cancel each other out (section 6). v2 lets neuron groups form from Hebbian co-activation, and it uses graded per-group plasticity in place of hard masks, so no neuron is ever cut off from a token's gradient.

**What we did not establish.** `label_s0` was not run, so section 6 never separated whether the blow-up came from the masking or from the changing selection. v2 changes both, so if it trains well we will not know which change fixed it. `metrics.csv` now logs per-layer activation norms in every mode, so a blow-up is visible during a run.

**Terminology.** This is Hebbian co-activation clustering. Cannistraci-Hebb Training (CHT) is related inspiration only, and is not the method.

**Method summary** (full version in the README):
- Per layer, an EMA of the correlation matrix of the hidden activations, from about 512 sampled tokens per step.
- From step 500, every 500 steps: kNN graph, Leiden, merge small groups, keep IDs stable with Hungarian matching.
- Per token, each assembly's plasticity depends on its relative activity, and it scales the gradient on both projections with the forward output unchanged.
- Default plasticity is `rank`: the most active assembly at a token gets 1 and the least active gets the floor. `sigmoid` was the first design. It stays available, but rank guarantees a spread of plasticity, where sigmoid could leave nearly every assembly near the same value.

**Health checks.** After every clustering, each layer logs its number of assemblies, largest fraction, fraction of neurons in small assemblies, dead neurons, splits, merges and ARI. Dead or constant neurons (variance below 1e-8) are left out of the graph and put in a leftover group (ID -1) with plasticity 1, outside the ranking. A layer with fewer than 4 assemblies, or one holding more than half the neurons, is flagged degenerate and trains dense until a later clustering recovers. These thresholds are judgement calls.

**First real data: post-hoc control on `dense_v0`.** CPU, with a much reduced evaluation: 150 batches for clustering, and 8 windows per domain per ablation, on a small real-data sample whose validation windows come from the same sources as this model's training data and may overlap it. Treat these as preliminary. A run on the real val sets replaces them.

| | Real assemblies | Random sets of the same size |
|---|---|---|
| Assemblies per layer | 6, 7, 8, 8, 8, 9 (none degenerate, no dead neurons) | |
| Mean S | 0.153, bootstrap 95% CI [0.104, 0.209] | 0.090, CI [0.066, 0.115] |
| Real minus random S | CI [0.011, 0.119] | |
| Damage per neuron (nats) | 0.0021 | 0.00025 |

So a model trained with **no routing** already has co-activation groups that are more specialised than random sets, with about 8 times the damage per neuron. For example, layer 0 assembly 2's top tokens are digits and `=`. This is why the post-hoc dense control matters: assembly mode only shows something if it beats this number, not if it beats 0.051 from the fixed blocks.

## 8. Reviewer concerns

What a sceptical reviewer would ask, and where we stand. "Open" means we have no answer yet.

| Concern | Status |
|---|---|
| **Does the clustering find specialization that already exists?** Dense models have domain-aligned groups too (section 7). | Addressed by design: `eval/ablate_assemblies.py --source posthoc` on the same dense checkpoint. Assembly mode must beat that, not the fixed-block number. |
| **Is S just a side effect of group size, or of removing many neurons?** | Addressed: each assembly is compared with 5 random neuron sets of the same size, reporting S and damage per neuron. |
| **Is S confounded with how critical a group is?** Section 6 had high S with enormous shared damage. | Partly addressed: size-weighted and damage-weighted S, damage per neuron, and activation norms in `metrics.csv`. S still cannot separate the two on its own. |
| **Are clusters flattered by being scored on the data that formed them?** | Addressed: post-hoc clustering uses the first half of each val set, and scoring always uses the second half. |
| **Do the error bars mean anything?** | Partly. Bootstrap CIs resample assemblies inside one model. They ignore variation between seeds, and assemblies in a model are not independent. Only the 3-seed runs answer this. |
| **Do assemblies encode domain, or just token frequency and position?** | Measured, not settled: `eval/assembly_probe.py` reports top tokens and correlations with position and log frequency. On the preliminary dense control, mean absolute correlation was 0.12 with position and 0.21 with log frequency, and up to about 0.67 with frequency in layer 0. Some assemblies are at least partly frequency detectors. |
| **Does graded plasticity do anything under Adam?** Adam divides by a running gradient scale, so scaling a parameter's gradient by a constant leaves its update almost unchanged (`test_adam_cancels_a_constant_gradient_scale` shows this). What survives is the variation: a neuron learns more from tokens where its plasticity is high, compared with its other tokens. So the mean of p is not an effective learning rate. | Open. The `LR_SCALE` dense control therefore tests a stronger change than routing actually makes. Read it as a conservative comparison, not an exact match. |
| **Does consolidation harden anything under Adam?** A slowly decaying ceiling is close to a constant scale, so Adam would largely undo it. | Open, and important before anyone uses `--consolidate`. A real version probably has to scale the parameter update rather than the gradient. It is off by default and deferred. |
| **Do partitions stay stable enough to matter?** If membership churns, no group accumulates a specialty. | Measured per clustering (ARI, splits, merges, `assembly_health.png`). Smoke ARI was low (0.2-0.5), on a tiny model that is still changing fast. Real values are unknown. |
| **Is the correlation estimate reliable?** A sample of 512 tokens for 1536 neurons gives a rank-deficient matrix per step. | Mitigated: the EMA pools about 100 steps, so roughly 50K tokens. Not tested for sensitivity to sample size. |
| **Does a token's plasticity really apply to that token only?** | No. The scaling is exact only in the last layer. Attention carries a later token's loss back through earlier tokens, which have their own plasticity, and training windows mix domains. The test covers the last layer only. |
| **Rank plasticity ignores how active a group is.** The top group gets 1 even when every group is near baseline. | Open by design. Compare with `PLASTICITY=sigmoid` if rank looks wrong. |
| **Thresholds and settings are arbitrary:** resolution, kNN, min size, floor, the degenerate limits, the 0.25 split and merge share. | Partly addressed: resolutions 0.5 and 2.0 are in the run order. The others are not swept. |
| **Positive correlations only.** Anti-correlated neurons never link, so opposing pairs that work together are ignored. | Accepted limitation. |
| **Does training with p < 1 simply train less?** | Measured: the loss curves compare directly with dense, and `summary.txt` records mean p. On smoke runs mean p after clustering was about 0.55, as rank predicts. |

## 9. First assembly run (assembly_s0) against the dense post-hoc control (dense_s0)

Both on the T4, seed 0, 5000 iterations, real data. Files in `results/dense_s0_v2data/` (renamed from `dense_s0`, see `docs/plan.md`) and `results/assembly_s0/` (checkpoints not in git).

**Health: fine.** Final val loss 2.139 / 1.479 / 2.508 (story / math / code) against dense 2.118 / 1.457 / 2.470, so 1-1.5% worse. No activation blow-up (largest final norm 4.3 against 4.7 for dense; `token_s0` reached 178). No degenerate layers, no dead neurons. Layer 4's largest assembly reached 48% of the layer at the last clustering, just under the 50% limit. Training took 23 min, the whole pipeline about 50 min, as estimated.

**Specialization: no significant difference.** Assembly S values below come from assemblies tracked during training. The dense values come from post-hoc clustering of the finished model on validation data, so the two groupings were built differently.

| Metric | assembly_s0 | dense_s0 post hoc | Difference (95% CI) |
|---|---|---|---|
| Mean S | 0.187 | 0.158 | +0.029 (-0.035 to +0.092) |
| Mean S above own random sets | 0.125 | 0.105 | +0.020 (-0.041 to +0.082) |
| Size-weighted S | 0.173 | 0.152 | +0.020 (-0.041 to +0.083) |
| Damage-weighted S | 0.141 | 0.152 | -0.011 (-0.275 to +0.192) |

The CIs bootstrap assemblies within each model (one seed each). Both models beat their random sets. The same kinds of group appear in both, for example a layer 4 math group on `x`, `+` and `-` (S 0.42 in assembly, 0.29 in dense). Stability (ARI) rose from 0.21 to about 0.67 over training, but the learning rate fell tenfold over the same period and dense has no matching measurement. A likely reason for the null result, which we did not test: per-domain plasticity was nearly uniform (mean 0.557 / 0.550 / 0.545 for story / math / code; S computed on each assembly's per-domain plasticity was about 0.011 at the end). Rank plasticity seems to sort by token more than by domain.

**Consequence.** Scope was cut to the minimum in `docs/plan.md`: the assembly sweeps and seeds are now future work, and v3 (section 10) asks a narrower question.

## 10. v3: deconfounding data, gated mode, and a zero-training diagnostic

**The confound.** In the v0-v2 corpus, digits appear almost only in arithmetic and arithmetic only as digits. In 300K tokens of real TinyStories text (the sample used below), 48 tokens are digits. So a group that "serves math" cannot be told apart from a group that serves digit tokens, and S cannot separate function from vocabulary. v3 adds about 10% synthetic deconfounding data (word problems in words, stories with numbers, arithmetic inside code) and per-token category tags for evaluation (`deconf.py`).

**Gated mode.** A forward-pass soft k-winners-take-all over neuron groups, with no gradient masking or scaling (README, Method). Fixed and evolved groups share every line of code except the group assignment. `route_smooth` (default 0.9) is a causal EMA of the group scores along the sequence, so selection can depend on the context and not only on the current token.

**Diagnostic before any GPU run: context sensitivity on `dense_s0`.** `eval/context_sensitivity.py`, CPU, on the existing `dense_s0` checkpoint. Groups are post hoc: co-activation and Leiden on the first half of a small real-data sample (200 batches), reduced to 8 groups per layer in layers 2-5. Gating is inference-only: selections are computed (k = 3 of 8, bias 0) and the forward pass stays dense. For each token id that occurs in both contexts, we take the Jaccard distance between the selected-group sets of two occurrences. Cross pairs one occurrence from each context. Within pairs two occurrences from the same context, which gives the noise floor. Effect = cross - within. Results are averaged over occurrence pairs (up to 50 per context), then over token ids: 99 digit ids, 11 number-word ids.

| Layer | Digits, smooth 0: cross / within / effect | Digits, smooth 0.9 | Number words, smooth 0 | Number words, smooth 0.9 |
|---|---|---|---|---|
| 2 | 0.680 / 0.510 / **+0.170** | 0.643 / 0.363 / **+0.280** | 0.582 / 0.501 / +0.081 | 0.615 / 0.522 / +0.094 |
| 3 | 0.717 / 0.411 / **+0.306** | 0.928 / 0.341 / **+0.587** | 0.309 / 0.272 / +0.036 | 0.186 / 0.179 / +0.007 |
| 4 | 0.624 / 0.422 / **+0.202** | 0.916 / 0.298 / **+0.618** | 0.290 / 0.285 / +0.006 | 0.367 / 0.356 / +0.012 |
| 5 | 0.666 / 0.486 / **+0.180** | 0.922 / 0.321 / **+0.600** | 0.429 / 0.393 / +0.036 | 0.440 / 0.428 / +0.012 |
| Mean | +0.214 | +0.521 | +0.040 | +0.031 |

Contexts: digits in arithmetic lines (`Q: 37 + 48 = A: 85`) against digits in number stories ("Mia was 7 and had 3 cats"). Number words in word problems ("three plus four makes seven") against number words in number stories.

**What this shows.**
- In the dense model, a digit picks different groups in arithmetic than in a story, well beyond the within-context noise. Smoothing more than doubles this: at 0.9, in layers 3-5, the top-3 sets of the same digit in the two contexts are almost disjoint (cross 0.92-0.93).
- Number words pick nearly the same groups in word problems as in stories (effect 0.01-0.09), with or without smoothing.
- So selection follows the surface context. The digit contexts differ in format (Q/A lines against prose) and the number-word contexts do not (both prose). This does **not** show that selection follows arithmetic function. If it did, word problems would separate from stories too.

**Caveats.** `dense_s0` never trained on word problems or number stories. The probes are generated text from `deconf.py`. The clustering sample is the start of the HuggingFace streams, which `dense_s0` trained on. Selections come from a model that never trained with gates, with zero bias. One checkpoint, 11 number-word ids. Files are in `results/dense_s0_v2data/context_diag/`.

**What to look for in the trained runs.** The number-word effect is the informative one. If gated training with deconfounding data makes word problems route away from stories while digits in stories do not, that is functional routing. The damage table checks the same thing by ablation (functional against lexical groups). The `gated_evolved` run at `route_smooth=0` tests whether smoothing matters during training, not only at inference.
