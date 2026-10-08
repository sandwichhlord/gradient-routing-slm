#!/usr/bin/env bash
# Full pipeline: prepare -> train -> sample -> plots -> ablate. Stops on the first error.
#   bash run_all.sh               # full run (real data, configs/baseline.py; GPU if available)
#   SMOKE=1 bash run_all.sh       # tiny CPU-friendly run on synthetic data (2 layers, 64 dim, 100 iters)
#   ROUTED=1 bash run_all.sh      # same, with gradient routing (configs/routed.py), outputs in out_routed/
# Optional env: PYTHON=python3  DEVICE=cpu|cuda  TRAIN_ARGS="--max_iters 2000"  PREP_ARGS="--force"
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
export PYTHONUNBUFFERED=1

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
OUT=out
ROUTE=()
if [[ "${ROUTED:-0}" == "1" ]]; then
  OUT=out_routed
  if [[ "${SMOKE:-0}" == "1" ]]; then ROUTE=(--routing); else CONFIG=configs/routed.py; fi
  echo ">>> ROUTED mode: gradient routing on, outputs in $OUT/"
fi

T0=$SECONDS
stage() { echo; echo "=================== [$1/5] $2  (t=$((SECONDS - T0))s) ==================="; }

stage 1 "prepare data"
# shellcheck disable=SC2086
"$PY" data/prepare.py "${PREP[@]}" ${PREP_ARGS:-}

stage 2 "train"
# shellcheck disable=SC2086
"$PY" train.py --config "$CONFIG" --device "$DEVICE" --out_dir "$OUT" "${ROUTE[@]}" ${TRAIN_ARGS:-}

stage 3 "sample"
"$PY" eval/sample.py --device "$DEVICE" --out_dir "$OUT" "${SAMPLE[@]}"

stage 4 "plots"
"$PY" eval/plots.py --out_dir "$OUT"

stage 5 "ablate"
"$PY" eval/ablate.py --device "$DEVICE" --out_dir "$OUT" "${ABLATE[@]}"

echo
echo "All 5 stages finished in $((SECONDS - T0))s. Outputs:"
ls -la "$OUT/"
