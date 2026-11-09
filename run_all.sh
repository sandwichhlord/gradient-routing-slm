#!/usr/bin/env bash
# Full pipeline: prepare -> train -> sample -> plots -> ablate -> routing / group evaluation. Stops on the first error.
#   ROUTING_MODE=gated GATE_GROUPS=evolved SEED=0 bash run_all.sh   # full run on real data; outputs in out_gated_evolved_s0/
#   SMOKE=1 ROUTING_MODE=gated bash run_all.sh                 # tiny CPU-friendly run on synthetic data (2 layers, 100 iters)
# Env:
#   ROUTING_MODE  dense (default), gated (v3), or the older batch (v0), token (v1), label (oracle), assembly (v2).
#                 ROUTED=1 means batch. dense and gated runs get the v3 evaluations (damage table, failure trace,
#                 unlearning, context sensitivity); a dense run uses post-hoc groups for them.
#   GATE_GROUPS   gated mode: fixed (default) or evolved (not GROUPS: bash reserves that name)
#   GATE_GRAD     gated mode: 1 (default) lets gradient flow through the group scores, 0 detaches them (folder tag _gg0)
#   ROUTE_SMOOTH  gated mode: causal EMA of group scores along the sequence (default 0.9); also the post-hoc gating of
#                 a dense run's evaluations. A non-default value changes the default OUT_DIR, e.g. out_gated_evolved_sm0_s0
#   RESOLUTION, FLOOR, PLASTICITY  assembly mode settings (defaults 1.0, 0.1, rank)
#   LR_SCALE      scales the learning rate, e.g. a dense run matched to an assembly run's mean p (see its summary.txt)
#                 Non-default values of these four change the default OUT_DIR.
#   POSTHOC=1     also compute assemblies post hoc on the finished checkpoint and ablate them (the dense control)
#   SEED          default 0: model init and batch order. The dataset and val sets stay fixed across seeds.
#   MAX_ITERS     override the config's max_iters
#   OUT_DIR       default out_<mode>_s<seed>
#   DATA_DIR      default data/out
#   PYTHON=python3  DEVICE=cpu|cuda  TRAIN_ARGS="--batch_size 16"  PREP_ARGS="--force"
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
export PYTHONUNBUFFERED=1

MODE="${ROUTING_MODE:-dense}"
if [[ "${ROUTED:-0}" == "1" && -z "${ROUTING_MODE:-}" ]]; then MODE=batch; fi
case "$MODE" in dense|batch|token|label|assembly|gated) ;; *) echo "unknown ROUTING_MODE=$MODE"; exit 2;; esac
SEED="${SEED:-0}"
NAME="$MODE"
if [[ "$MODE" == "gated" ]]; then
  GATE_GROUPS="${GATE_GROUPS:-fixed}"
  case "$GATE_GROUPS" in fixed|evolved) ;; *) echo "unknown GATE_GROUPS=$GATE_GROUPS"; exit 2;; esac
  NAME="gated_$GATE_GROUPS"
fi
# runs with non-default settings get their own folder, e.g. out_assembly_r0.5_s0 or out_dense_lr0.55_s0
TAG=""
if [[ -n "${RESOLUTION:-}" && "$RESOLUTION" != "1.0" ]]; then TAG+="_r$RESOLUTION"; fi
if [[ -n "${FLOOR:-}" && "$FLOOR" != "0.1" ]]; then TAG+="_f$FLOOR"; fi
if [[ -n "${PLASTICITY:-}" && "$PLASTICITY" != "rank" ]]; then TAG+="_$PLASTICITY"; fi
if [[ -n "${LR_SCALE:-}" && "$LR_SCALE" != "1.0" ]]; then TAG+="_lr$LR_SCALE"; fi
if [[ -n "${ROUTE_SMOOTH:-}" && "$ROUTE_SMOOTH" != "0.9" ]]; then TAG+="_sm$ROUTE_SMOOTH"; fi
if [[ -n "${GATE_GRAD:-}" && "$GATE_GRAD" != "1" ]]; then TAG+="_gg$GATE_GRAD"; fi
OUT="${OUT_DIR:-out_${NAME}${TAG}_s${SEED}}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">>> SMOKE mode: synthetic data, configs/smoke.py"
  PREP=(--synthetic --synthetic_tokens 50000)
  CONFIG=configs/smoke.py
  ABLATE=(--eval_iters 2)
  SAMPLE=(--max_new_tokens 40)
  ASSEMBLY=(--eval_iters 2 --posthoc_steps 20 --random_sets 2)
  PROBE=(--batches 2)
  V3=(--gated_layers 0,1 --posthoc_batches 10 --windows 8)
  DAMAGE=(--random_sets 1)
  TRACE=(--n_problems 100)
  UNLEARN=(--n_problems 100 --random_sets 1)
else
  PREP=()
  CONFIG=configs/baseline.py
  ABLATE=(--eval_iters 20)
  SAMPLE=()
  ASSEMBLY=(--eval_iters 20)
  PROBE=()
  V3=()
  DAMAGE=()
  TRACE=()
  UNLEARN=()
fi
TRAIN=(--config "$CONFIG" --routing_mode "$MODE" --seed "$SEED" --device "$DEVICE" --out_dir "$OUT")
EVAL_DATA=()
if [[ -n "${DATA_DIR:-}" ]]; then
  PREP+=(--out_dir "$DATA_DIR")
  TRAIN+=(--data_dir "$DATA_DIR")
  EVAL_DATA=(--data_dir "$DATA_DIR")
fi
if [[ -n "${MAX_ITERS:-}" ]]; then TRAIN+=(--max_iters "$MAX_ITERS"); fi
if [[ -n "${RESOLUTION:-}" ]]; then TRAIN+=(--resolution "$RESOLUTION"); ASSEMBLY+=(--resolution "$RESOLUTION"); fi
if [[ -n "${FLOOR:-}" ]]; then TRAIN+=(--floor "$FLOOR"); fi
if [[ -n "${PLASTICITY:-}" ]]; then TRAIN+=(--plasticity "$PLASTICITY"); fi
if [[ -n "${LR_SCALE:-}" ]]; then TRAIN+=(--lr_scale "$LR_SCALE"); fi
if [[ "$MODE" == "gated" ]]; then TRAIN+=(--groups "$GATE_GROUPS"); fi
if [[ -n "${GATE_GRAD:-}" ]]; then TRAIN+=(--gate_grad "$GATE_GRAD"); fi
if [[ -n "${ROUTE_SMOOTH:-}" ]]; then TRAIN+=(--route_smooth "$ROUTE_SMOOTH"); V3+=(--route_smooth "$ROUTE_SMOOTH"); fi
echo ">>> mode=$MODE seed=$SEED outputs in $OUT/"

T0=$SECONDS
stage() { echo; echo "=================== [$1/6] $2  (t=$((SECONDS - T0))s) ==================="; }

stage 1 "prepare data"
# shellcheck disable=SC2086
"$PY" data/prepare.py "${PREP[@]}" ${PREP_ARGS:-}

stage 2 "train"
# shellcheck disable=SC2086
"$PY" train.py "${TRAIN[@]}" ${TRAIN_ARGS:-}

stage 3 "sample"
"$PY" eval/sample.py --device "$DEVICE" --out_dir "$OUT" "${SAMPLE[@]}"

stage 4 "plots"
"$PY" eval/plots.py --out_dir "$OUT"

stage 5 "ablate"
"$PY" eval/ablate.py --device "$DEVICE" --out_dir "$OUT" "${EVAL_DATA[@]}" "${ABLATE[@]}"

stage 6 "routing, assembly and group evaluation"
V3_ARGS=(--device "$DEVICE" --out_dir "$OUT" "${EVAL_DATA[@]}" "${V3[@]}")
case "$MODE" in
  dense|gated)
    "$PY" eval/context_sensitivity.py "${V3_ARGS[@]}"
    "$PY" eval/damage_table.py "${V3_ARGS[@]}" "${DAMAGE[@]}"
    "$PY" eval/failure_trace.py "${V3_ARGS[@]}" "${TRACE[@]}"
    "$PY" eval/unlearn.py "${V3_ARGS[@]}" "${UNLEARN[@]}"
    ;;
  assembly)
    "$PY" eval/assembly_health_plot.py --out_dir "$OUT"
    "$PY" eval/ablate_assemblies.py --device "$DEVICE" --out_dir "$OUT" --source final "${EVAL_DATA[@]}" "${ASSEMBLY[@]}"
    "$PY" eval/assembly_probe.py --device "$DEVICE" --out_dir "$OUT" --assignments "$OUT/assemblies_final.json" "${EVAL_DATA[@]}" "${PROBE[@]}"
    ;;
  *) "$PY" eval/routing_plot.py --out_dir "$OUT" ;;
esac
if [[ "${POSTHOC:-0}" == "1" ]]; then
  "$PY" eval/ablate_assemblies.py --device "$DEVICE" --out_dir "$OUT" --source posthoc "${EVAL_DATA[@]}" "${ASSEMBLY[@]}"
  "$PY" eval/assembly_probe.py --device "$DEVICE" --out_dir "$OUT" --assignments "$OUT/posthoc/assemblies_posthoc.json" "${EVAL_DATA[@]}" "${PROBE[@]}"
fi

echo
echo "All stages finished in $((SECONDS - T0))s. Outputs:"
ls -la "$OUT/"
