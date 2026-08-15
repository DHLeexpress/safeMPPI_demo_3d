#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
EXPANSION=${EXPANSION:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_dense_z_relaxedy_multipair_q10_trunk3}
SCENE_BANK=${SCENE_BANK:-$EXPANSION/eval_m50_r0_r1/raw_eval.json}
CHECKPOINT_ROUND=${CHECKPOINT_ROUND:-0}
SAMPLES_PER_STEP=${SAMPLES_PER_STEP:-8}
SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE:-1.0}
EPISODES=${EPISODES:-50}
DEVICE=${DEVICE:-mps}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0811_e15_raw_cost_deploy_pre_m8_t1_replay}

cd "$ROOT"
python scripts/evaluate_multisphere_min_cost_deployment.py \
  --helios \
  --helios-gpu "${HELIOS_GPU:-1}" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --expansion "$EXPANSION" \
  --checkpoint-round "$CHECKPOINT_ROUND" \
  --scene-bank-json "$SCENE_BANK" \
  --evaluation-output "$OUT" \
  --device "$DEVICE" \
  --episodes "$EPISODES" \
  --samples-per-step "$SAMPLES_PER_STEP" \
  --sampling-temperature "$SAMPLING_TEMPERATURE" \
  --execution-clearance-exp-weight 15 \
  --execution-clearance-target-m 0.6 \
  --execution-clearance-exp-temperature 0.15 \
  --execution-taskspace-quadratic-weight 250 \
  --execution-taskspace-quadratic-target-m 0.15 \
  --execution-axis-cylinder-quadratic-weight 5 \
  --execution-axis-cylinder-radius-m 1.1 \
  --execution-control-weight 0.05 \
  --seed 91000

# Selection contract:
#   - sample exactly M raw flow plans at every step;
#   - never call or consult the verifier for selection;
#   - add only E15 obstacle cost from the other task;
#   - retain all round-1 native, wall, axis-cylinder, and control costs;
#   - score SR/CR/OOB/timeout/window-validity after closed-loop execution.
