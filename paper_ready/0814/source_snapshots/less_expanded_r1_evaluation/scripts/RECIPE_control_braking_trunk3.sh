#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
OUT=${OUT:?set OUT}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 1 or 3}
ROUNDS=${ROUNDS:-5}
SEED=${SEED:-82410}
RETRY_RESAMPLE_BATCH_CAP=${RETRY_RESAMPLE_BATCH_CAP:-4096}
CONTROL_WEIGHT=${CONTROL_WEIGHT:?set CONTROL_WEIGHT}
TERMINAL_WEIGHT=${TERMINAL_WEIGHT:-80}
BRAKING_WEIGHT=${BRAKING_WEIGHT:-0}
FINITE_SEGMENT=${FINITE_SEGMENT:-0}
TASKSPACE_WEIGHT=${TASKSPACE_WEIGHT:-500}
TASKSPACE_TARGET=${TASKSPACE_TARGET:-0.15}
AXIS_WEIGHT=${AXIS_WEIGHT:-5}
AXIS_RADIUS=${AXIS_RADIUS:-1.1}
CLEARANCE_WEIGHT=${CLEARANCE_WEIGHT:-2500}
CLEARANCE_TARGET=${CLEARANCE_TARGET:-0.6}
EXECUTION_RULE=${EXECUTION_RULE:-quadratic_cost}
CLEARANCE_EXP_WEIGHT=${CLEARANCE_EXP_WEIGHT:-0}
CLEARANCE_EXP_TEMPERATURE=${CLEARANCE_EXP_TEMPERATURE:-0.15}
CLEARANCE_EXP_TARGET=${CLEARANCE_EXP_TARGET:-$CLEARANCE_TARGET}
CLEARANCE_EXP_AGGREGATION=${CLEARANCE_EXP_AGGREGATION:-mean}
OPTIMIZER_STEPS_PER_ROUND=${OPTIMIZER_STEPS_PER_ROUND:-2500}
SUCCESS_QUOTA=${SUCCESS_QUOTA:-12}
SAMPLE_MODES=${SAMPLE_MODES:-0,0,0,1,1,1,2,2,2,3,3,3}
SAMPLE_UPDATE_SUBMODES=${SAMPLE_UPDATE_SUBMODES:-none}
FAITHFUL_RETRY=${FAITHFUL_RETRY:-0}
GOAL_BOX_WEIGHT=${GOAL_BOX_WEIGHT:-0}
GOAL_BOX_HALF_EXTENT=${GOAL_BOX_HALF_EXTENT:-0.2}
GOAL_BOX_TEMPERATURE=${GOAL_BOX_TEMPERATURE:-0.5}
FULL_H_TASKSPACE=${FULL_H_TASKSPACE:-0}
STOPPING_MARGIN=${STOPPING_MARGIN:-}
LAB_TASK_CONFIG=${LAB_TASK_CONFIG:-configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json}
GOAL_SIDE_WALL_WEIGHT=${GOAL_SIDE_WALL_WEIGHT:-0}
GOAL_SIDE_WALL_TARGET=${GOAL_SIDE_WALL_TARGET:-0.6}

if [[ "$HELIOS_GPU" != "1" && "$HELIOS_GPU" != "3" ]]; then
  echo "HELIOS_GPU must be 1 or 3; GPU0 is evacuated and GPU2 is prohibited" >&2
  exit 2
fi
if [[ "$FINITE_SEGMENT" != "0" && "$FINITE_SEGMENT" != "1" ]]; then
  echo "FINITE_SEGMENT must be 0 or 1" >&2
  exit 2
fi
if [[ "$FAITHFUL_RETRY" != "0" && "$FAITHFUL_RETRY" != "1" ]]; then
  echo "FAITHFUL_RETRY must be 0 or 1" >&2
  exit 2
fi
if [[ "$FULL_H_TASKSPACE" != "0" && "$FULL_H_TASKSPACE" != "1" ]]; then
  echo "FULL_H_TASKSPACE must be 0 or 1" >&2
  exit 2
fi
if [[ "$EXECUTION_RULE" != "quadratic_cost" && "$EXECUTION_RULE" != "exponential_cost" ]]; then
  echo "EXECUTION_RULE must be quadratic_cost or exponential_cost" >&2
  exit 2
fi
if [[ "$CLEARANCE_EXP_AGGREGATION" != "mean" && "$CLEARANCE_EXP_AGGREGATION" != "max" && "$CLEARANCE_EXP_AGGREGATION" != "top3_mean" ]]; then
  echo "CLEARANCE_EXP_AGGREGATION must be mean, max, or top3_mean" >&2
  exit 2
fi
if (( OPTIMIZER_STEPS_PER_ROUND < 1 )); then
  echo "OPTIMIZER_STEPS_PER_ROUND must be positive" >&2
  exit 2
fi
if [[ "$SAMPLE_UPDATE_SUBMODES" != "none" && "$SAMPLE_UPDATE_SUBMODES" != "angular8" ]]; then
  echo "SAMPLE_UPDATE_SUBMODES must be none or angular8" >&2
  exit 2
fi

finite_args=()
if [[ "$FINITE_SEGMENT" == "1" ]]; then
  finite_args=(--execution-axis-cylinder-finite-segment)
fi
retry_args=(--retry-B 16 --retry-verify-all-fast-path)
if [[ "$FAITHFUL_RETRY" == "1" ]]; then
  # Preserve the ordinary B-of-K GP acquisition ranking on retries.
  retry_args=(--retry-B 8)
fi
full_h_args=()
if [[ "$FULL_H_TASKSPACE" == "1" ]]; then
  full_h_args=(--verifier-full-h-taskspace)
fi
stopping_args=()
if [[ -n "$STOPPING_MARGIN" ]]; then
  stopping_args=(--verifier-taskspace-stopping-margin-m "$STOPPING_MARGIN")
fi
submode_args=()
if [[ "$SAMPLE_UPDATE_SUBMODES" == "angular8" ]]; then
  submode_args=(--sample-update-submodes angular8)
fi
obstacle_args=(
  --execution-rule quadratic_cost
  --execution-clearance-quadratic-weight "$CLEARANCE_WEIGHT"
  --execution-clearance-quadratic-target-m "$CLEARANCE_TARGET"
)
if [[ "$EXECUTION_RULE" == "exponential_cost" ]]; then
  obstacle_args=(
    --execution-rule exponential_cost
    --execution-clearance-exp-weight "$CLEARANCE_EXP_WEIGHT"
    --execution-clearance-exp-temperature "$CLEARANCE_EXP_TEMPERATURE"
    --execution-clearance-exp-aggregation "$CLEARANCE_EXP_AGGREGATION"
    --execution-clearance-target-m "$CLEARANCE_EXP_TARGET"
  )
fi

process_lock="${OUT}.control_braking_process.lock"
if ! mkdir "$process_lock" 2>/dev/null; then
  echo "another wrapper already owns $process_lock" >&2
  exit 75
fi
echo "$$" > "$process_lock/pid"
release_process_lock() {
  rm -f "$process_lock/pid"
  rmdir "$process_lock" 2>/dev/null || true
}
trap release_process_lock EXIT INT TERM

resume=()
if [[ -s "$OUT/resume_state_latest.pt" && -s "$OUT/resume_state.json" ]]; then
  resume=(--resume-from "$OUT")
fi

set +u
python scripts/research_ball_expansion_optimization.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config "$LAB_TASK_CONFIG" \
  --output "$OUT" \
  "${resume[@]}" \
  --device mps \
  --rounds "$ROUNDS" \
  --flow-nfe 12 \
  --batched-rollout-sampling \
  --fa-alloc none \
  --flow-base-std 1 \
  --flow-base-std-schedule none \
  --flow-base-std-final 1 \
  --candidate-perturb-std 0 \
  --candidate-perturb-scope coherent_horizon \
  --learning-rate 5e-5 \
  --learning-rate-final 5e-5 \
  --learning-rate-decay-steps 1 \
  --gradient-clip-norm 5 \
  --round-learning-rate-warmup-power 0 \
  --trainable-trunk-layers 3 \
  --freeze-visual-encoder-during-expansion \
  --beta 0.1 \
  --parallel-episodes 16 \
  --verifier-workers 8 \
  --max-retry-batches 32 \
  --retry-exhaustion-policy resample_scene \
  --retry-resample-batch-cap "$RETRY_RESAMPLE_BATCH_CAP" \
  --successful-trajectories-per-gamma "$SUCCESS_QUOTA" \
  --sample-update-mode "$SAMPLE_MODES" \
  "${submode_args[@]}" \
  --sample-update-cohorts unguided_only \
  --K 16 \
  --B 8 \
  "${retry_args[@]}" \
  --inner-steps 1 \
  --optimizer-steps-per-round "$OPTIMIZER_STEPS_PER_ROUND" \
  --no-optimizer-steps-total \
  --batch-size 256 \
  --replay-scope cumulative \
  --replay-batch-sampler mode_gamma_stratified \
  --replay-top-fraction 1 \
  --replay-selector uniform \
  --replay-rounds 100 \
  --gp-buffer-cap 1536 \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --negative-alpha 0 \
  --archive-rule successful_executed_windows \
  --successful-trajectory-selector random_success \
  --replay-acceptance execution_eligible \
  "${obstacle_args[@]}" \
  --execution-taskspace-quadratic-weight "$TASKSPACE_WEIGHT" \
  --execution-taskspace-quadratic-target-m "$TASKSPACE_TARGET" \
  --execution-goal-side-wall-quadratic-weight "$GOAL_SIDE_WALL_WEIGHT" \
  --execution-goal-side-wall-target-m "$GOAL_SIDE_WALL_TARGET" \
  --execution-goal-box-exp-weight "$GOAL_BOX_WEIGHT" \
  --execution-goal-box-half-extent-m "$GOAL_BOX_HALF_EXTENT" \
  --execution-goal-box-exp-temperature-m "$GOAL_BOX_TEMPERATURE" \
  --execution-axis-cylinder-quadratic-weight "$AXIS_WEIGHT" \
  --execution-axis-cylinder-radius-m "$AXIS_RADIUS" \
  "${finite_args[@]}" \
  --execution-control-weight "$CONTROL_WEIGHT" \
  --execution-terminal-goal-weight "$TERMINAL_WEIGHT" \
  --execution-goal-braking-weight "$BRAKING_WEIGHT" \
  --execution-goal-braking-distance-m 0.6 \
  --execution-goal-braking-temperature-m 0.15 \
  --execution-step-margin-weight 0 \
  --acquisition-feature learned_phi \
  --coverage-replay none \
  --replay-augmentation none \
  --execution-z-bias-mode none \
  --tight-corridor \
  --verifier-mode full_polytope \
  --verifier-solver analytic \
  "${full_h_args[@]}" \
  "${stopping_args[@]}" \
  --event-log committed_success \
  --paired-noised-representation \
  --seed "$SEED"
