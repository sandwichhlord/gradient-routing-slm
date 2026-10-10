# Findings

## 1. Routing collapse when the balancing gradient is masked

**Scope.** All numbers here come from CPU smoke runs: synthetic data, 2 layers x 8 blocks = 16 blocks, `n_embd` 64, one seed, 100-300 iterations. They are a debugging record, not a result at full scale. No routed run has been done on the T4 yet.

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

In the lambda 0.01 run, one block in each layer (0.5 and 1.7) was selected in 49-50 of the 50 iterations in each of the last two windows. So routing is bursty and a few blocks dominate. Whether this persists at full scale is an open question (`plan.md`, task 1).

**Cost of routing at smoke scale** (300 iters, lambda 0.01, single seed, final validation loss):

| Domain | Dense | Routed (default mode) | Difference |
|---|---|---|---|
| story | 1.795 | 2.030 | +0.235 |
| math | 3.010 | 3.111 | +0.101 |
| code | 2.171 | 2.318 | +0.147 |

Routing costs loss at this scale, as expected from training only 2 of 8 blocks per layer on each step. The size of the cost at full scale is unknown.

## 2. Dense baseline on the T4 (5000 iterations)

See "Results so far" in the README. Short version: mean specialization S = 0.051, damage differs mostly by layer (layer 0 blocks matter 5-9x more than later layers) and not by domain.
