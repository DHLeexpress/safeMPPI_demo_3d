#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
BASE=${BASE:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_e15_multipair_t48}
ARM_RECIPE=$ROOT/scripts/RECIPE_multisphere_pre2_e15_multipair_inner_arm.sh
CURVE_RECIPE=$ROOT/scripts/RECIPE_multisphere_m8_e15_raw_curve.sh
STATUS=$BASE/pipeline_status.log

mkdir -p "$BASE"

record() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$STATUS"
}

inner10=${BASE}_inner10
inner50=${BASE}_inner50
inner100=${BASE}_inner100

launch_arm() {
  local inner_steps=$1
  local gpu=$2
  local seed=$3
  local arm=$4
  local log=$5
  env \
    SUCCESS_TRAJECTORIES_PER_ROUND=48 \
    SUCCESS_TRAJECTORIES_PER_GAMMA=12 \
    PARALLEL_EPISODES=24 \
    INNER_STEPS="$inner_steps" \
    HELIOS_GPU="$gpu" \
    ROUNDS=5 \
    SEED="$seed" \
    OUT="$arm" \
    bash "$ARM_RECIPE" > "$log" 2>&1 &
  arm_pid=$!
}

record "LAUNCH inner10=$inner10 inner50=$inner50"
launch_arm 10 1 81550 "$inner10" "$BASE/launcher_inner10.log"
inner10_pid=$arm_pid
printf '%s\n' "$inner10_pid" > "$BASE/launcher_inner10.pid"
launch_arm 50 3 81551 "$inner50" "$BASE/launcher_inner50.log"
inner50_pid=$arm_pid
printf '%s\n' "$inner50_pid" > "$BASE/launcher_inner50.pid"

if ! wait "$inner50_pid"; then
  record "FAILED inner50=$inner50"
  exit 1
fi
record "COMPLETE inner50; LAUNCH inner100"

launch_arm 100 3 81552 "$inner100" "$BASE/launcher_inner100.log"
inner100_pid=$arm_pid
printf '%s\n' "$inner100_pid" > "$BASE/launcher_inner100.pid"

if ! wait "$inner10_pid"; then
  record "FAILED inner10=$inner10"
  exit 1
fi
record "COMPLETE inner10"
if ! wait "$inner100_pid"; then
  record "FAILED inner100=$inner100"
  exit 1
fi
record "COMPLETE inner100"

run_curve() {
  local arm=$1
  local gpu=$2
  local label=$3
  local log=$4
  env \
    EXPANSION="$arm" \
    HELIOS_GPU="$gpu" \
    ARM_LABEL="$label" \
    bash "$CURVE_RECIPE" > "$log" 2>&1 &
  curve_pid=$!
}

run_curve "$inner10" 1 'inner-steps 10 · M8 E15' "$BASE/eval_inner10.log"
eval10_pid=$curve_pid
run_curve "$inner50" 3 'inner-steps 50 · M8 E15' "$BASE/eval_inner50.log"
eval50_pid=$curve_pid
wait "$eval10_pid"
wait "$eval50_pid"
record "EVALUATED inner10 inner50"

run_curve "$inner100" 3 'inner-steps 100 · M8 E15' "$BASE/eval_inner100.log"
eval100_pid=$curve_pid
wait "$eval100_pid"
record "EVALUATED inner100; PIPELINE_COMPLETE"
