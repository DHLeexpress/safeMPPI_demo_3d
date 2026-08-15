#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
OUT=${OUT:?set OUT}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 1 or 3}
ROUNDS=${ROUNDS:-5}
CYLINDER_WEIGHT=${CYLINDER_WEIGHT:?set CYLINDER_WEIGHT}
SEED=${SEED:-82310}
EVAL_GPU=${EVAL_GPU:-$HELIOS_GPU}
START_ROUND=${START_ROUND:-0}

PRE="$PRE" \
OUT="$OUT" \
HELIOS_GPU="$HELIOS_GPU" \
ROUNDS="$ROUNDS" \
SEED="$SEED" \
CYLINDER_WEIGHT="$CYLINDER_WEIGHT" \
CYLINDER_RADIUS="${CYLINDER_RADIUS:-1.1}" \
TASKSPACE_WEIGHT="${TASKSPACE_WEIGHT:-500}" \
TASKSPACE_TARGET="${TASKSPACE_TARGET:-0.15}" \
bash scripts/RECIPE_axis_cylinder_trunk3.sh

evaluation_rounds=()
round_i=$START_ROUND
while [[ "$round_i" -le "$ROUNDS" ]]; do
  evaluation_rounds+=("$round_i")
  round_i=$((round_i + 1))
done

PRE="$PRE" \
EXPANSION="$OUT" \
EVAL_OUT="$OUT/fixed_eval_r$(printf '%03d' "$ROUNDS")" \
HELIOS_GPU="$EVAL_GPU" \
TARGET_ROUND="$ROUNDS" \
EVALUATION_ROUNDS="${evaluation_rounds[*]}" \
bash scripts/RECIPE_r10_taskspace_eval.sh
