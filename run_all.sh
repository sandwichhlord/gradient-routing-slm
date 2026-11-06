#!/usr/bin/env bash
# Full pipeline: prepare -> train -> sample -> plots -> ablate (-> routing plot). Stops on the first error.
#   ROUTING_MODE=token SEED=0 bash run_all.sh     # full run on real data; outputs in out_token_s0/
#   SMOKE=1 ROUTING_MODE=label bash run_all.sh    # tiny CPU-friendly run on synthetic data (2 layers, 64 dim, 100 iters)
# Env:
#   ROUTING_MODE  dense (default), batch (v0), token (v1), label (oracle baseline). ROUTED=1 means batch.
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
case "$MODE" in dense|batch|token|label) ;; *) echo "unknown ROUTING_MODE=$MODE"; exit 2;; esac
SEED="${SEED:-0}"
OUT="${OUT_DIR:-out_${MODE}_s${SEED}}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">>> SMOKE mode: synthetic data, configs/smoke.py"
  PREP=(--synthetic --synthetic_tokens 50000)
  CONFIG=configs/smoke.py
  ABLATE=(--eval_iters 2)
  SAMPLE=(--max_new_tokens 40)
else
  PREP=()
  CONFIG=configs/baseline.py
  ABLATE=(--eval_iters 20)
  SAMPLE=()
fi
TRAIN=(--config "$CONFIG" --routing_mode "$MODE" --seed "$SEED" --device "$DEVICE" --out_dir "$OUT")
EVAL_DATA=()
if [[ -n "${DATA_DIR:-}" ]]; then
  PREP+=(--out_dir "$DATA_DIR")
  TRAIN+=(--data_dir "$DATA_DIR")
  EVAL_DATA=(--data_dir "$DATA_DIR")
fi
if [[ -n "${MAX_ITERS:-}" ]]; then TRAIN+=(--max_iters "$MAX_ITERS"); fi
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

stage 6 "routing plot"
if [[ "$MODE" == "dense" ]]; then
  echo "dense mode: no routing stats, skipping"
else
  "$PY" eval/routing_plot.py --out_dir "$OUT"
fi

echo
echo "All stages finished in $((SECONDS - T0))s. Outputs:"
ls -la "$OUT/"
