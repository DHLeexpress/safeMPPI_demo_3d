#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
OUT=${OUT:?set OUT}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 0, 1, or 3}
EXECUTION_ARM=${EXECUTION_ARM:?exp_balanced, quad_fast, or quad_high_sr}
UPDATE_SCOPE=${UPDATE_SCOPE:?head, trunk1, or trunk3}
ROUNDS=${ROUNDS:-5}
SEED=${SEED:-82100}
BETA=${BETA:-0.1}
FLOW_BASE_STD=${FLOW_BASE_STD:-1.0}
FLOW_BASE_STD_FINAL=${FLOW_BASE_STD_FINAL:-1.0}
FLOW_BASE_STD_SCHEDULE=${FLOW_BASE_STD_SCHEDULE:-none}
MAX_RETRY_BATCHES=${MAX_RETRY_BATCHES:-32}
RETRY_RESAMPLE_BATCH_CAP=${RETRY_RESAMPLE_BATCH_CAP:-64}
QUAD_HIGH_SR_WEIGHT=${QUAD_HIGH_SR_WEIGHT:-2500}
OPTIMIZER_STEPS_PER_ROUND=${OPTIMIZER_STEPS_PER_ROUND:-2500}
REPLAY_SCOPE=${REPLAY_SCOPE:-cumulative}
REPLAY_ROUNDS=${REPLAY_ROUNDS:-100}

process_lock="${OUT}.paper_arm_process.lock"
if ! mkdir "$process_lock" 2>/dev/null; then
  echo "another expansion wrapper already owns $process_lock" >&2
  exit 75
fi
echo "$$" > "$process_lock/pid"
release_process_lock() {
  rm -f "$process_lock/pid"
  rmdir "$process_lock" 2>/dev/null || true
}
trap release_process_lock EXIT INT TERM

if [[ "$HELIOS_GPU" != "0" && "$HELIOS_GPU" != "1" && "$HELIOS_GPU" != "3" ]]; then
  echo "HELIOS_GPU must be 0, 1, or 3" >&2
  exit 2
fi

case "$EXECUTION_ARM" in
  exp_balanced)
    execution=(
      --execution-rule exponential_cost
      --execution-clearance-exp-weight 100
      --execution-clearance-exp-temperature 0.3
      --execution-clearance-target-m 0.6
    )
    ;;
  quad_fast)
    execution=(
      --execution-rule quadratic_cost
      --execution-clearance-quadratic-weight 1200
      --execution-clearance-quadratic-target-m 0.6
    )
    ;;
  quad_high_sr)
    execution=(
      --execution-rule quadratic_cost
      --execution-clearance-quadratic-weight "$QUAD_HIGH_SR_WEIGHT"
      --execution-clearance-quadratic-target-m 0.6
    )
    ;;
  *) echo "unknown EXECUTION_ARM=$EXECUTION_ARM" >&2; exit 2 ;;
esac

case "$UPDATE_SCOPE" in
  head) scope=(--head-only-expansion) ;;
  trunk1) scope=(--trainable-trunk-layers 1) ;;
  trunk3) scope=(--trainable-trunk-layers 3) ;;
  *) echo "unknown UPDATE_SCOPE=$UPDATE_SCOPE" >&2; exit 2 ;;
esac

resume=()
if [[ -s "$OUT/resume_state_latest.pt" && -s "$OUT/resume_state.json" ]]; then
  resume=(--resume-from "$OUT")
fi

# macOS Bash 3.2 treats an empty array expansion as unbound under `set -u`.
# Every required scalar has already been validated above; disable nounset only
# for the command assembly so a fresh run can omit the optional resume pair.
set +u
python scripts/research_ball_expansion_optimization.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json \
  --output "$OUT" \
  "${resume[@]}" \
  --device mps \
  --rounds "$ROUNDS" \
  --flow-nfe 12 \
  --batched-rollout-sampling \
  --fa-alloc none \
  --flow-base-std "$FLOW_BASE_STD" \
  --flow-base-std-schedule "$FLOW_BASE_STD_SCHEDULE" \
  --flow-base-std-final "$FLOW_BASE_STD_FINAL" \
  --candidate-perturb-std 0 \
  --candidate-perturb-scope coherent_horizon \
  --learning-rate 5e-5 \
  --learning-rate-final 5e-5 \
  --learning-rate-decay-steps 1 \
  --gradient-clip-norm 5 \
  --round-learning-rate-warmup-power 0 \
  "${scope[@]}" \
  --freeze-visual-encoder-during-expansion \
  --beta "$BETA" \
  --parallel-episodes 16 \
  --verifier-workers 8 \
  --max-retry-batches "$MAX_RETRY_BATCHES" \
  --retry-exhaustion-policy resample_scene \
  --retry-resample-batch-cap "$RETRY_RESAMPLE_BATCH_CAP" \
  --successful-trajectories-per-gamma 12 \
  --sample-update-mode "0,0,0,1,1,1,2,2,2,3,3,3" \
  --sample-update-cohorts unguided_only \
  --K 16 \
  --B 8 \
  --retry-B 16 \
  --retry-verify-all-fast-path \
  --inner-steps 1 \
  --optimizer-steps-per-round "$OPTIMIZER_STEPS_PER_ROUND" \
  --no-optimizer-steps-total \
  --batch-size 256 \
  --replay-scope "$REPLAY_SCOPE" \
  --replay-batch-sampler mode_gamma_stratified \
  --replay-top-fraction 1 \
  --replay-selector uniform \
  --replay-rounds "$REPLAY_ROUNDS" \
  --gp-buffer-cap 1536 \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --negative-alpha 0 \
  --archive-rule successful_executed_windows \
  --successful-trajectory-selector random_success \
  --replay-acceptance execution_eligible \
  "${execution[@]}" \
  --execution-step-margin-weight 0 \
  --acquisition-feature learned_phi \
  --coverage-replay none \
  --replay-augmentation none \
  --execution-z-bias-mode none \
  --tight-corridor \
  --verifier-mode full_polytope \
  --verifier-solver analytic \
  --event-log committed_success \
  --paired-noised-representation \
  --seed "$SEED"
