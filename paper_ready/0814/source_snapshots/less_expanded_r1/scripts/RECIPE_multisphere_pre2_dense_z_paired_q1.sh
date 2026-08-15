#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK_CONFIG=${TASK_CONFIG:-$ROOT/configs/lab_clutter_spheres_double_hourglass_n6_dense_z0711_v1.json}
HELIOS_GPU=${HELIOS_GPU:-1}
ROUNDS=${ROUNDS:-1}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_dense_z_axis180_pair_q1_r${ROUNDS}}
SEED=${SEED:-81311}
MAX_RETRY_BATCHES=${MAX_RETRY_BATCHES:-20}
K=${K:-16}
B=${B:-8}
RETRY_B=${RETRY_B:-$B}
BETA=${BETA:-0.1}
ACQUISITION_TAIL_RESERVE=${ACQUISITION_TAIL_RESERVE:-0}
PAIRED_SCENE_SEED_OFFSET=${PAIRED_SCENE_SEED_OFFSET:-0}
PAIRED_SCENE_SEED_OFFSET_START_ROUND=${PAIRED_SCENE_SEED_OFFSET_START_ROUND:-1}
PAIRED_SUCCESS_MIRROR_PROPOSAL=${PAIRED_SUCCESS_MIRROR_PROPOSAL:-1}

resume=()
if [[ -s "$OUT/resume_state_latest.pt" && -s "$OUT/resume_state.json" ]]; then
  resume=(--resume-from "$OUT")
fi

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac
case "$PAIRED_SUCCESS_MIRROR_PROPOSAL" in
  0) mirror_proposal=() ;;
  1) mirror_proposal=(--paired-success-mirror-proposal) ;;
  *) echo "PAIRED_SUCCESS_MIRROR_PROPOSAL must be 0 or 1" >&2; exit 2 ;;
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
  "${resume[@]}" \
  --device cuda:0 \
  --rounds "$ROUNDS" \
  --paired-scene-rotation start_goal_axis_180 \
  --paired-scene-seed-offset "$PAIRED_SCENE_SEED_OFFSET" \
  --paired-scene-seed-offset-start-round "$PAIRED_SCENE_SEED_OFFSET_START_ROUND" \
  "${mirror_proposal[@]}" \
  --flow-nfe 16 \
  --flow-base-std 1.0 \
  --flow-base-std-schedule none \
  --candidate-perturb-std 0.0 \
  --fa-alloc none \
  --parallel-episodes 8 \
  --verifier-workers 16 \
  --max-retry-batches "$MAX_RETRY_BATCHES" \
  --retry-exhaustion-policy abort \
  --successful-trajectories-per-gamma 1 \
  --K "$K" \
  --B "$B" \
  --retry-B "$RETRY_B" \
  --acquisition-tail-reserve "$ACQUISITION_TAIL_RESERVE" \
  --beta "$BETA" \
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

# Deliberately absent: --sample-update-mode, --retry-verify-all-fast-path.
# The default paired-success proposal still verifies only B candidates: B-1
# ordinary uncertainty acquisitions plus one mirrored terminal-success plan.
# parallel-episodes=8 means four original and four axis-180 attempts per
# gamma/retry batch.  q1 means one complete pair = two committed trajectories.
# To extend r5->r10 exactly, pass the same explicit OUT with ROUNDS=10.  This
# recipe auto-loads the last committed model, Adam, RNG, replay, and GP state.
