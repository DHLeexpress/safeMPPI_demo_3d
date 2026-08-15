#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK_CONFIG=${TASK_CONFIG:-$ROOT/configs/lab_clutter_spheres_double_hourglass_n6_dense_z0711_goalspace_yminus04_v1.json}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_dense_z_relaxedy_multipair_q10_trunk3}
HELIOS_GPU=${HELIOS_GPU:-3}
ROUNDS=${ROUNDS:-1}
SEED=${SEED:-81411}

SUCCESS_TRAJECTORIES_PER_GAMMA=${SUCCESS_TRAJECTORIES_PER_GAMMA:-10}
PARALLEL_EPISODES=${PARALLEL_EPISODES:-20}
MAX_RETRY_BATCHES=${MAX_RETRY_BATCHES:-20}
K=${K:-16}
B=${B:-8}
RETRY_B=${RETRY_B:-$B}
GP_BUFFER_CAP=${GP_BUFFER_CAP:-1536}
PAIRED_SUCCESS_MIRROR_PROPOSAL=${PAIRED_SUCCESS_MIRROR_PROPOSAL:-1}
PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES=${PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES:-8}
PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT=${PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT:-1}

EXECUTION_CONTROL_WEIGHT=${EXECUTION_CONTROL_WEIGHT:-0.05}

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac
case "$PAIRED_SUCCESS_MIRROR_PROPOSAL" in
  0) mirror_proposal=() ;;
  1) mirror_proposal=(--paired-success-mirror-proposal) ;;
  *) echo "PAIRED_SUCCESS_MIRROR_PROPOSAL must be 0 or 1" >&2; exit 2 ;;
esac
if (( SUCCESS_TRAJECTORIES_PER_GAMMA <= 0 || SUCCESS_TRAJECTORIES_PER_GAMMA % 2 != 0 )); then
  echo "SUCCESS_TRAJECTORIES_PER_GAMMA must be a positive even number" >&2
  exit 2
fi
if (( PARALLEL_EPISODES % SUCCESS_TRAJECTORIES_PER_GAMMA != 0 )); then
  echo "PARALLEL_EPISODES must be a multiple of SUCCESS_TRAJECTORIES_PER_GAMMA" >&2
  exit 2
fi
if (( PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES < 1 )); then
  echo "PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES must be positive" >&2
  exit 2
fi
if (( PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT < 1 )); then
  echo "PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT must be positive" >&2
  exit 2
fi

resume=()
if [[ -s "$OUT/resume_state_latest.pt" && -s "$OUT/resume_state.json" ]]; then
  resume=(--resume-from "$OUT")
fi

cd "$ROOT"
python scripts/research_multisphere_expansion_multipair.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config "$TASK_CONFIG" \
  --output "$OUT" \
  ${resume[@]+"${resume[@]}"} \
  --device cuda:0 \
  --rounds "$ROUNDS" \
  --paired-scene-rotation start_goal_axis_180 \
  --paired-scene-replace-after-retry-batches "$PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES" \
  --paired-scene-max-replacements-per-slot "$PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT" \
  ${mirror_proposal[@]+"${mirror_proposal[@]}"} \
  --flow-nfe 16 \
  --batched-rollout-sampling \
  --flow-base-std 1.0 \
  --flow-base-std-schedule none \
  --candidate-perturb-std 0.0 \
  --fa-alloc none \
  --parallel-episodes "$PARALLEL_EPISODES" \
  --verifier-workers 16 \
  --max-retry-batches "$MAX_RETRY_BATCHES" \
  --retry-exhaustion-policy abort \
  --successful-trajectories-per-gamma "$SUCCESS_TRAJECTORIES_PER_GAMMA" \
  --K "$K" \
  --B "$B" \
  --retry-B "$RETRY_B" \
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
  --trainable-trunk-layers 3 \
  --freeze-visual-encoder-during-expansion \
  --negative-alpha 0 \
  --execution-rule min_cost \
  --execution-taskspace-quadratic-weight 250 \
  --execution-taskspace-quadratic-target-m 0.15 \
  --execution-axis-cylinder-quadratic-weight 5 \
  --execution-axis-cylinder-radius-m 1.1 \
  --execution-control-weight "$EXECUTION_CONTROL_WEIGHT" \
  --execution-z-bias-mode none \
  --acquisition-feature learned_phi \
  --gp-buffer-cap "$GP_BUFFER_CAP" \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --coverage-replay none \
  --replay-augmentation none \
  --paired-noised-representation \
  --event-log committed_success \
  --seed "$SEED"

# Scientific contract:
# - Q=10 means five distinct deterministic scene pairs per gamma and exactly
#   one committed original + one committed axis-180 trajectory per pair.
# - P=20 launches two attempts for every one of the ten quota labels in each
#   whole retry batch. Uncertainty-tilted K/B acquisition remains active.
# - At retry batch 8, a slot with commit-capable success on only one member is
#   replaced once; both old member candidates are discarded and recollected.
# - Round 1 has no causal GP evidence. Its committed windows become the
#   trajectory-uniform, per-gamma capped GP support used from round 2 onward.
# - OUT is deliberately independent of ROUNDS. Re-run with the same OUT and a
#   larger ROUNDS value to restore model, Adam, RNG, replay, and GP state.
