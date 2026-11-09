# Roadmap (minimum scope)

**Research question:** with the digit/arithmetic confound removed, do groups of MLP neurons that compete per token (gated mode) separate function from vocabulary better than fixed groups, or than a dense model's own groups? And can the arithmetic groups be deleted without hurting stories and code?

## Done

- Data pipeline for three domains, per-domain validation sets, per-token domain ids. v3 adds about 10% deconfounding data and per-token category tags (`deconf.py`); `--deconf_frac 0 --holdout_arith 0` rebuilds the old corpus exactly.
- Dense baseline with block ablation and the specialization score S.
- Routing v0 (batch), v1 (token, label) and v2 (assembly), all kept as baselines. One full run each of dense, batch, token and assembly. Assembly did not beat the dense post-hoc control (`findings.md`, section 9).
- v3: `--routing_mode gated` with `--groups fixed|evolved`, `--gated_layers` and `--route_smooth`. Four evaluations that work on dense (post-hoc groups) and gated checkpoints: `eval/damage_table.py`, `eval/failure_trace.py`, `eval/unlearn.py`, `eval/context_sensitivity.py`, and the summary table in `eval/compare.py`. Tested on CPU only.
- The zero-training diagnostic on `dense_s0` (`findings.md`, section 10): in the dense model, digits route by surface context and number words do not.

## Runs (seed 0, in this order)

Set these in the notebook's first cell, Run All, and unzip each result into `results/<zip name>/`. Then run `python eval/compare.py`, which writes `results/comparison_gated.csv`.

| Order | Run | Notebook settings | Estimate on a T4 |
|---|---|---|---|
| 1 | `dense_s0` (new data) | `ROUTING_MODE="dense"` | about 35 min |
| 2 | `gated_fixed_s0` | `ROUTING_MODE="gated"`, `GROUPS="fixed"` | about 40 min |
| 3 | `gated_evolved_s0` | `ROUTING_MODE="gated"`, `GROUPS="evolved"` | about 40 min |
| 4 | `gated_evolved_sm0_s0` | as 3, with `ROUTE_SMOOTH=0.0` | about 40 min |

These times are estimates; nothing in v3 has run on a GPU. The new data changes the corpus, so run 1 replaces the old dense run as the baseline. The old one is in `results/dense_s0_v2data/` (findings sections 9 and 10), so the new `dense_s0.zip` unzips into `results/dense_s0/` with no clash.

**Check each run, in order:**
1. `metrics.csv`: loss falls smoothly with no NaN, `act_norm_l*` stays near dense, and there is no lasting jump at step 500 when gating starts.
2. `gating.csv`: selection shares near k/G = 0.375 per group; no group stuck near 0 or 1. Evolved: `gating_layers.csv` shows `moved_last` at or below 76 (5% of 1536) and sizes that change slowly.
3. `gating_layers.csv`: the `ctx_*_effect` columns over training.
4. `damage_summary.txt`, `failure_trace_summary.txt`, `unlearn_summary.txt`, `context_sensitivity.txt`, then `results/comparison_gated.csv` and `results/group_sizes.csv`. Functional claims use only the format-matched rows; `format_*` rows are context.
5. Health against dense, per the pre-registered criteria below. The gate default was chosen before any GPU run (next section).

## Pre-GPU design decision: gate score and gate gradient (2026-11-09)

**Rule, set before the comparison was run:** default to `--gate_grad 1 --gate_score share` if, in a 300-step smoke run, its per-domain val loss is within 5% of dense and its activation norms stay within 2x dense in every layer. Otherwise default to `--gate_grad 0`.

**Smoke comparison:** CPU, `configs/smoke.py` (2 layers, both gated, 64 dim), synthetic data, seed 0, 300 iterations, `--groups evolved`. The dense, gate_grad 0 and gate_grad 1 + relative rows are from the earlier run of the same comparison (same data, seed and code paths); the share row was run with them unchanged.

| Run | Val loss story / math / code (vs dense) | Final act norm L0 / L1 | Max ratio to dense over all evals, L0 / L1 |
|---|---|---|---|
| dense | 2.847 / 3.390 / 2.738 | 0.783 / 0.622 | 1 / 1 |
| gate_grad 0 (relative) | 2.790 / 3.348 / 2.665 (-2.0%, -1.2%, -2.6%) | 0.635 / 0.491 | 1.00 / 1.00 |
| gate_grad 1 + relative | 3.543 / 3.605 / 2.992 (+24.5%, +6.3%, +9.3%) | 2.853 / 1.179 | 3.73 / 1.90 |
| gate_grad 1 + share | 2.958 / 3.394 / 2.700 (+3.9%, +0.1%, -1.4%) | 0.856 / 0.579 | 1.22 / 1.16 |

**Decision:** `--gate_grad 1 --gate_score share` meets both conditions (worst loss +3.9%, worst norm ratio 1.22x), so it is the default for all gated runs. `GATE_SCORE=relative` and `GATE_GRAD=0` remain available. With the relative score, the gate gradient rewards raising activity against a lagging baseline, which matches the norm growth seen; the share score is unchanged when every activation is scaled. This is a smoke run on a tiny model and synthetic data, not evidence about the full-scale result. The post-hoc groups of dense runs also use the share score, so dense and gated are scored the same way.

## Pre-registered success criteria (2026-11-09, committed before any v3 result exists)

These are fixed before the first v3 GPU run and are not to be changed after seeing results.

1. **Health** (every run): no NaN; per-layer activation norms (`act_norm_l*` in `metrics.csv`) within 10x of dense; per-domain val loss within 15% of dense.
2. **Evolved beats fixed** only if BOTH: its failure-trace consistency is higher than fixed's, AND its unlearning collateral damage (story and code loss increase after deleting the arithmetic-traced groups) is lower than fixed's. Seed 0 only decides whether to run more seeds. Any claim needs 3 seeds with bootstrap CIs.
3. **"Functional" groups** are claimed only on format-matched categories (digits in prose arithmetic, number words in word problems, digits in stories), and only above both the within-context noise floor and size-matched random groups. Comparisons with the Q/A format (`Q: a op b = A: c`) are reported separately, labelled "format", and support no functional claim.
4. **Smoothing helped** only if `gated_evolved` (route_smooth 0.9) beats `gated_evolved_sm0` (route_smooth 0) on the format-matched context-sensitivity effect.

## Future work (cut from the current scope)

- More seeds (1 and 2) for the four runs above.
- Assembly mode: resolutions 0.5 and 2.0, three seeds, the `LR_SCALE` dense control, sigmoid plasticity and the floor setting, and the like-for-like post-hoc clustering of the assembly checkpoint.
- `label_s0` (the oracle baseline, and the only way to separate masking from selection in the `token_s0` blow-up).
- Linear domain probes per group.
- Relearning attacks after unlearning, and continual learning (sequential domains, EWC).
- Consolidation (groups hardening as they mature). `--consolidate` in assembly mode exists but Adam probably cancels it.
- Sweeps of the gated settings (k, tau, number of groups, max_move, gated layers).

## Risks and open questions

- **Gating changes the model at step 500.** Both gated runs train dense for 500 steps, then the gates switch on abruptly. Expect a loss bump; a long one would be a problem.
- **Soft gates leak.** With tau 0.1, an unselected group's gate is often 0.1-0.3, not 0. Deleting a group in the eval is a hard zero, so it differs from what training saw.
- **The bias limits selection.** With gamma 0.001, the bias can only move 5 per 5000 steps. If a group is never selected early on, it may stay that way.
- **Format against function.** The diagnostic showed context effects that follow surface format. The functional/lexical classes in the damage table need the deconfounding data to be in the training mix, which only the new runs have.
- **Thresholds are choices:** the damage floor (0.01 nats), the 95th percentile of random sets, "active" meaning selected on at least half the answer positions, the top 3 groups. All are set in the eval scripts.
- **One seed and small scale.** About 10M non-embedding parameters on about 44M tokens. Results may not carry over.
- **fp16.** The gated path was only tested on CPU. Scores and gates are computed in fp32, but the first GPU run is the first fp16 test.
