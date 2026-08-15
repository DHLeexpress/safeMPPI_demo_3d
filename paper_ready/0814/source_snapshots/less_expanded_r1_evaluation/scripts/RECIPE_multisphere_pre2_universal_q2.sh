#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK_CONFIG=${TASK_CONFIG:-$ROOT/configs/lab_clutter_spheres_double_hourglass_n6_relaxed_z01_17_v1.json}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0810_pre2_universal_q2_r1}
HELIOS_GPU=${HELIOS_GPU:-1}
ROUNDS=${ROUNDS:-1}
SEED=${SEED:-81231}
TRUNK_LAYERS=${TRUNK_LAYERS:-1}
WALL_WEIGHT=${WALL_WEIGHT:-250}
WALL_TARGET_M=${WALL_TARGET_M:-0.15}
AXIS_WEIGHT=${AXIS_WEIGHT:-5}
CONTROL_WEIGHT=${CONTROL_WEIGHT:-0.05}
EVENT_LOG=${EVENT_LOG:-full}

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac

cd "$ROOT"
python scripts/research_multisphere_expansion_pre2.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --pretrain-dir "$PRE" \
  --lab-task-config "$TASK_CONFIG" \
  --output "$OUT" \
  --device cuda:0 \
  --rounds "$ROUNDS" \
  --flow-nfe 16 \
  --flow-base-std 1.0 \
  --flow-base-std-schedule none \
  --candidate-perturb-std 0.0 \
  --fa-alloc none \
  --parallel-episodes 8 \
  --verifier-workers 16 \
  --max-retry-batches 20 \
  --retry-exhaustion-policy abort \
  --successful-trajectories-per-gamma 2 \
  --K 16 \
  --B 8 \
  --retry-B 8 \
  --beta 0.1 \
  --archive-rule successful_executed_windows \
  --successful-trajectory-selector random_success \
  --replay-acceptance execution_eligible \
  --replay-scope sliding \
  --replay-rounds 3 \
  --replay-batch-sampler row_permutation \
  --replay-passes-per-round 1 \
  --inner-steps 1 \
  --batch-size 128 \
  --no-optimizer-steps-total \
  --replay-top-fraction 1.0 \
  --replay-selector uniform \
  --learning-rate 2e-5 \
  --gradient-clip-norm 1.0 \
  --trainable-trunk-layers "$TRUNK_LAYERS" \
  --freeze-visual-encoder-during-expansion \
  --negative-alpha 0 \
  --execution-rule min_cost \
  --execution-taskspace-quadratic-weight "$WALL_WEIGHT" \
  --execution-taskspace-quadratic-target-m "$WALL_TARGET_M" \
  --execution-axis-cylinder-quadratic-weight "$AXIS_WEIGHT" \
  --execution-axis-cylinder-radius-m 1.1 \
  --execution-control-weight "$CONTROL_WEIGHT" \
  --execution-z-bias-mode none \
  --acquisition-feature learned_phi \
  --gp-buffer-cap 1536 \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --coverage-replay none \
  --replay-augmentation none \
  --paired-noised-representation \
  --event-log "$EVENT_LOG" \
  --seed "$SEED"

# Deliberately absent:
#   --sample-update-mode
#   --retry-verify-all-fast-path
