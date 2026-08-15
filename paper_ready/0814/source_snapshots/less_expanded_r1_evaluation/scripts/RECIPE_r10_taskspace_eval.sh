#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
EXPANSION=${EXPANSION:?set EXPANSION}
EVAL_OUT=${EVAL_OUT:?set EVAL_OUT}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 1 or 3}
TARGET_ROUND=${TARGET_ROUND:-11}
EVALUATION_ROUNDS=${EVALUATION_ROUNDS:-$TARGET_ROUND}
EXPANSION_MANIFEST=${EXPANSION_MANIFEST:-}
LAB_TASK_CONFIG=${LAB_TASK_CONFIG:-}

if [[ "$HELIOS_GPU" != "1" && "$HELIOS_GPU" != "3" ]]; then
  echo "HELIOS_GPU must be 1 or 3; GPU0 is evacuated and GPU2 is prohibited" >&2
  exit 2
fi
read -r -a evaluation_rounds <<< "$EVALUATION_ROUNDS"

manifest_args=()
if [[ -n "$EXPANSION_MANIFEST" ]]; then
  manifest_args=(--expansion-manifest "$EXPANSION_MANIFEST")
fi
task_config_args=()
if [[ -n "$LAB_TASK_CONFIG" ]]; then
  task_config_args=(--lab-task-config "$LAB_TASK_CONFIG")
fi

set +u
python scripts/research_evaluate_ball_expansion.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --expansion "$EXPANSION" \
  "${manifest_args[@]}" \
  "${task_config_args[@]}" \
  --evaluation-output "$EVAL_OUT" \
  --device mps \
  --episodes 40 \
  --probe-samples 16 \
  --flow-nfe 12 \
  --evaluation-rounds "${evaluation_rounds[@]}" \
  --seed 91000 \
  --fixed-scene-rollouts 10 \
  --fixed-scene-seed 191000 \
  --gallery-rounds "$TARGET_ROUND" \
  --gallery-view head_on \
  --save-raw-trajectories \
  --screening-only
set -u
