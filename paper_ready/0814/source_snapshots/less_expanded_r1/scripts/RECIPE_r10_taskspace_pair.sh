#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
OUT=${OUT:?set OUT to a prepared r10 fork}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 1 or 3}
TASKSPACE_WEIGHT=${TASKSPACE_WEIGHT:?set TASKSPACE_WEIGHT}
TASKSPACE_TARGET=${TASKSPACE_TARGET:-0.15}

if [[ "$HELIOS_GPU" != "1" && "$HELIOS_GPU" != "3" ]]; then
  echo "HELIOS_GPU must be 1 or 3; GPU0 is evacuated and GPU2 is prohibited" >&2
  exit 2
fi
if [[ ! -s "$OUT/resume_state_latest.pt" || ! -s "$OUT/resume_state.json" ]]; then
  echo "OUT must contain the prepared r10 resume fork" >&2
  exit 2
fi

process_lock="${OUT}.r10_taskspace_process.lock"
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

python scripts/research_ball_expansion_optimization.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json \
  --output "$OUT" \
  --resume-from "$OUT" \
  --device mps \
  --rounds 11 \
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
  --retry-resample-batch-cap 4096 \
  --successful-trajectories-per-gamma 12 \
  --sample-update-mode "0,0,0,1,1,1,2,2,2,3,3,3" \
  --sample-update-cohorts unguided_only \
  --K 16 \
  --B 8 \
  --retry-B 16 \
  --retry-verify-all-fast-path \
  --inner-steps 1 \
  --optimizer-steps-per-round 2500 \
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
  --execution-rule quadratic_cost \
  --execution-clearance-quadratic-weight 2500 \
  --execution-clearance-quadratic-target-m 0.6 \
  --execution-taskspace-quadratic-weight "$TASKSPACE_WEIGHT" \
  --execution-taskspace-quadratic-target-m "$TASKSPACE_TARGET" \
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
  --seed 82108
