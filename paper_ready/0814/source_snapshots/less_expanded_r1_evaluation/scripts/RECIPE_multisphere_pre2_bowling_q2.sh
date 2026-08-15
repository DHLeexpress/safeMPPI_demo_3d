#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK_CONFIG=${TASK_CONFIG:-$ROOT/configs/lab_clutter_spheres_double_hourglass_n6_dense_z0711_v1.json}
HELIOS_GPU=${HELIOS_GPU:-3}
ROUNDS=${ROUNDS:-1}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_bowling123_q2_r${ROUNDS}}
SEED=${SEED:-81312}

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac

cd "$ROOT"
python scripts/research_multisphere_expansion_paired.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config "$TASK_CONFIG" \
  --output "$OUT" \
  --device cuda:0 \
  --rounds "$ROUNDS" \
  --fixed-scene-layout bowling_123 \
  --flow-nfe 16 \
  --flow-base-std 1.1 \
  --flow-base-std-schedule cosine \
  --flow-base-std-final 1.0 \
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
  --trainable-trunk-layers 1 \
  --freeze-visual-encoder-during-expansion \
  --negative-alpha 0 \
  --execution-rule min_cost \
  --execution-taskspace-quadratic-weight 250 \
  --execution-taskspace-quadratic-target-m 0.15 \
  --execution-axis-cylinder-quadratic-weight 5 \
  --execution-axis-cylinder-radius-m 1.1 \
  --execution-control-weight 0.05 \
  --execution-z-bias-mode none \
  --acquisition-feature learned_phi \
  --gp-buffer-cap 1536 \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --coverage-replay none \
  --replay-augmentation none \
  --paired-noised-representation \
  --event-log committed_success \
  --seed "$SEED"

# Round 1 uses std=1.1.  With ROUNDS=5, cosine scheduling reaches exactly
# std=1.0 at round 5.  Start that five-round schedule in its fresh default
# r5 output; do not resume the diagnostic r1 output under a changed horizon.
# No retry fast path is enabled.
