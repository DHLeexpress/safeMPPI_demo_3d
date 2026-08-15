#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
ARM_RECIPE=$ROOT/scripts/RECIPE_multisphere_pre2_e15_multipair_inner_arm.sh
SUCCESS_TRAJECTORIES_PER_ROUND=${SUCCESS_TRAJECTORIES_PER_ROUND:-48}
SUCCESS_TRAJECTORIES_PER_GAMMA=${SUCCESS_TRAJECTORIES_PER_GAMMA:-12}
ROUNDS=${ROUNDS:-5}
BASE_SEED=${BASE_SEED:-81550}
OUT_BASE=${OUT_BASE:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_e15_multipair_t48}

launch() {
  local inner_steps=$1
  local gpu=$2
  local seed=$3
  local output=$4
  local log=$5
  env \
    SUCCESS_TRAJECTORIES_PER_GAMMA="$SUCCESS_TRAJECTORIES_PER_GAMMA" \
    SUCCESS_TRAJECTORIES_PER_ROUND="$SUCCESS_TRAJECTORIES_PER_ROUND" \
    INNER_STEPS="$inner_steps" \
    HELIOS_GPU="$gpu" \
    ROUNDS="$ROUNDS" \
    SEED="$seed" \
    OUT="$output" \
    bash "$ARM_RECIPE" >"$log" 2>&1 &
  launched_pid=$!
}

mkdir -p "$OUT_BASE"
launch 10 1 "$BASE_SEED" "${OUT_BASE}_inner10" "$OUT_BASE/launcher_inner10.log"
pid10=$launched_pid
launch 50 3 "$((BASE_SEED + 1))" "${OUT_BASE}_inner50" "$OUT_BASE/launcher_inner50.log"
pid50=$launched_pid
printf '%s\n' "$pid10" > "$OUT_BASE/launcher_inner10.pid"
printf '%s\n' "$pid50" > "$OUT_BASE/launcher_inner50.pid"

# GPU 3 takes the inner100 arm after inner50's detached transport has returned.
wait "$pid50"
launch 100 3 "$((BASE_SEED + 2))" "${OUT_BASE}_inner100" "$OUT_BASE/launcher_inner100.log"
pid100=$launched_pid
printf '%s\n' "$pid100" > "$OUT_BASE/launcher_inner100.pid"

wait "$pid10"
wait "$pid100"
