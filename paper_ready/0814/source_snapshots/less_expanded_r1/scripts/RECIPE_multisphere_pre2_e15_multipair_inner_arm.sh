#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK_CONFIG=${TASK_CONFIG:-$ROOT/configs/lab_clutter_spheres_double_hourglass_n6_dense_z0711_goalspace_yminus04_v1.json}

SUCCESS_TRAJECTORIES_PER_ROUND=${SUCCESS_TRAJECTORIES_PER_ROUND:-48}
SUCCESS_TRAJECTORIES_PER_GAMMA=${SUCCESS_TRAJECTORIES_PER_GAMMA:-12}
INNER_STEPS=${INNER_STEPS:-10}
HELIOS_GPU=${HELIOS_GPU:-1}
ROUNDS=${ROUNDS:-5}
SEED=${SEED:-81550}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_e15_multipair_t${SUCCESS_TRAJECTORIES_PER_ROUND}_inner${INNER_STEPS}}

PARALLEL_EPISODES=${PARALLEL_EPISODES:-24}
MAX_RETRY_BATCHES=${MAX_RETRY_BATCHES:-20}
K=${K:-16}
B=${B:-8}
RETRY_B=${RETRY_B:-$B}
GP_BUFFER_CAP=${GP_BUFFER_CAP:-1536}
PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES=${PAIRED_SCENE_REPLACE_AFTER_RETRY_BATCHES:-2}
PAIRED_SCENE_REPLACE_INTERVAL_BATCHES=${PAIRED_SCENE_REPLACE_INTERVAL_BATCHES:-4}
PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT=${PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT:-5}

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac
if (( SUCCESS_TRAJECTORIES_PER_GAMMA <= 0 || SUCCESS_TRAJECTORIES_PER_GAMMA % 2 != 0 )); then
  echo "SUCCESS_TRAJECTORIES_PER_GAMMA must be a positive even number" >&2
  exit 2
fi
if (( SUCCESS_TRAJECTORIES_PER_ROUND != 48 )); then
  echo "this paper arm fixes SUCCESS_TRAJECTORIES_PER_ROUND=48" >&2
  exit 2
fi
if (( PARALLEL_EPISODES % SUCCESS_TRAJECTORIES_PER_GAMMA != 0 )); then
  echo "PARALLEL_EPISODES must be a multiple of SUCCESS_TRAJECTORIES_PER_GAMMA" >&2
  exit 2
fi
if (( INNER_STEPS != 10 && INNER_STEPS != 50 && INNER_STEPS != 100 )); then
  echo "INNER_STEPS must be one of 10, 50, or 100" >&2
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
  --paired-scene-replace-interval-batches "$PAIRED_SCENE_REPLACE_INTERVAL_BATCHES" \
  --paired-scene-max-replacements-per-slot "$PAIRED_SCENE_MAX_REPLACEMENTS_PER_SLOT" \
  --paired-success-mirror-proposal \
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
  --successful-trajectories-per-round "$SUCCESS_TRAJECTORIES_PER_ROUND" \
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
  --inner-steps "$INNER_STEPS" \
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
  --execution-clearance-exp-weight 15 \
  --execution-clearance-target-m 0.6 \
  --execution-clearance-exp-temperature 0.15 \
  --execution-taskspace-quadratic-weight 250 \
  --execution-taskspace-quadratic-target-m 0.15 \
  --execution-axis-cylinder-quadratic-weight 5 \
  --execution-axis-cylinder-radius-m 1.1 \
  --execution-control-weight 0.05 \
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
# - Each round commits exactly 48 trajectories: 12 per gamma, represented as
#   six exact start-goal-axis-180 mirror pairs. R5/R10 contain 240/480 total
#   trajectories, with exactly 60/120 trajectories per gamma.
# - At retry batches 2,6,10,14,18, an incomplete slot may be replaced; both
#   old member successes are discarded before collecting the fresh exact pair.
# - E15 is part of min_cost execution ranking only. Acquisition and full-H
#   GREEN verification remain unchanged.
# - --inner-steps repeats every replay microbatch. The manifest records actual
#   optimizer steps and loss for each committed round.
# - ROUNDS defaults to 5. Re-run the same OUT with ROUNDS=10 to restore model,
#   Adam, RNG, replay, and GP state and continue exactly from checkpoint 005.
