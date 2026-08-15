#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:-results/stage1_single_ball_t128/pretrain_hp100_t128_d3_e52}
OUT=${OUT:-results/stage1_single_ball_t128/0808_reserve_g_online_q3_r21_warmup2_seed10}
HELIOS_GPU=${HELIOS_GPU:-1}
HELIOS_GPU_POLICY=${HELIOS_GPU_POLICY:-queue}
SEED=${SEED:-10}
CANDIDATE_K=${CANDIDATE_K:-16}
INITIAL_B=${INITIAL_B:-8}
RETRY_B=${RETRY_B:-$CANDIDATE_K}
RETRY_VERIFY_ALL_FAST_PATH=${RETRY_VERIFY_ALL_FAST_PATH:-1}
FA_RETRY_MAP=${FA_RETRY_MAP:-missing_quota}
FA_RETRY_BAND=${FA_RETRY_BAND:-1.0}
RESUME_FROM=${RESUME_FROM:-}

if [[ "$HELIOS_GPU" == "2" ]]; then
  echo "Helios physical GPU 2 is reserved; use GPU 0, 1, or 3" >&2
  exit 2
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  if [[ "$RESUME_FROM" != "$OUT" ]]; then
    echo "RESUME_FROM must equal OUT for an in-place committed-round resume" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume-from "$RESUME_FROM")
fi

FA_RETRY_BAND_ARGS=()
if [[ -n "$FA_RETRY_BAND" ]]; then
  FA_RETRY_BAND_ARGS=(--fa-alloc-retry-band "$FA_RETRY_BAND")
fi

RETRY_VERIFY_ALL_ARGS=()
if [[ "$RETRY_VERIFY_ALL_FAST_PATH" == "1" ]]; then
  RETRY_VERIFY_ALL_ARGS=(--retry-verify-all-fast-path)
elif [[ "$RETRY_VERIFY_ALL_FAST_PATH" != "0" ]]; then
  echo "RETRY_VERIFY_ALL_FAST_PATH must be 0 or 1" >&2
  exit 2
fi

if [[ "$HELIOS_GPU_POLICY" == "share" ]]; then
  HELIOS_GPU_POLICY_ARGS=(--helios-share-gpu)
elif [[ "$HELIOS_GPU_POLICY" == "queue" ]]; then
  HELIOS_GPU_POLICY_ARGS=(--helios-queue-gpu)
else
  echo "HELIOS_GPU_POLICY must be queue or share" >&2
  exit 2
fi

python scripts/research_ball_expansion_optimization.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  "${HELIOS_GPU_POLICY_ARGS[@]}" \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config configs/lab_ball_stage1_t128.json \
  --output "$OUT" \
  "${RESUME_ARGS[@]}" \
  --device mps \
  --rounds 21 \
  --flow-nfe 12 \
  --batched-rollout-sampling \
  --geofence-floor-z 0.1 \
  --fa-alloc steps \
  --fa-alloc-steps 30 \
  --fa-alloc-map "0,0,0,0,1,1,1,1" \
  --fa-alloc-retry-map "$FA_RETRY_MAP" \
  "${FA_RETRY_BAND_ARGS[@]}" \
  --fa-alloc-band 0.5 \
  --fa-alloc-band-below 1.0 \
  --fa-below-diagonal 0.0 \
  --flow-base-std 1.1 \
  --flow-base-std-schedule cosine \
  --flow-base-std-final 1.0 \
  --candidate-perturb-std 0.0 \
  --candidate-perturb-scope coherent_horizon \
  --learning-rate 5e-5 \
  --learning-rate-final 1e-6 \
  --learning-rate-decay-steps 50000 \
  --gradient-clip-norm 5 \
  --round-learning-rate-warmup-power 2 \
  --trainable-trunk-layers 3 \
  --freeze-visual-encoder-during-expansion \
  --beta 0.1 \
  --parallel-episodes 8 \
  --verifier-workers 32 \
  --max-retry-batches 32 \
  --retry-exhaustion-policy resample_scene \
  --retry-resample-batch-cap 512 \
  --successful-trajectories-per-gamma 12 \
  --sample-update-mode "0,0,0,1,1,1,2,2,2,3,3,3" \
  --K "$CANDIDATE_K" \
  --B "$INITIAL_B" \
  --retry-B "$RETRY_B" \
  "${RETRY_VERIFY_ALL_ARGS[@]}" \
  --inner-steps 1 \
  --optimizer-steps-total 50000 \
  --optimizer-step-allocation quadratic_round \
  --batch-size 256 \
  --replay-scope cumulative \
  --replay-batch-sampler mode_gamma_stratified \
  --replay-top-fraction 1.0 \
  --replay-selector uniform \
  --replay-rounds 21 \
  --gp-buffer-cap 1536 \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --negative-alpha 0 \
  --archive-rule successful_executed_windows \
  --successful-trajectory-selector random_success \
  --replay-acceptance execution_eligible \
  --execution-rule min_cost \
  --execution-step-margin-weight 30000 \
  --execution-taskspace-weight 0 \
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
