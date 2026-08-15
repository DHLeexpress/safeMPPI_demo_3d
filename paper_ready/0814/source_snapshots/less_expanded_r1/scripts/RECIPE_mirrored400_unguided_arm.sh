#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:-results/stage1_single_ball_t128/0810_pretrain_mirrored400_hp100_t128_d3}
OUT=${OUT:?set OUT}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 0, 1, or 3}
BETA=${BETA:?set BETA}
PARALLEL_EPISODES=${PARALLEL_EPISODES:?set PARALLEL_EPISODES}
SEED=${SEED:-81740}
ROUNDS=${ROUNDS:-3}
MAX_RETRY_BATCHES=${MAX_RETRY_BATCHES:-128}
OPTIMIZER_STEPS_TOTAL=${OPTIMIZER_STEPS_TOTAL:-8000}
EXECUTION_STEP_MARGIN_WEIGHT=${EXECUTION_STEP_MARGIN_WEIGHT:-30000}
FLOW_NFE=${FLOW_NFE:-12}
CANDIDATE_K=${CANDIDATE_K:-64}
INITIAL_B=${INITIAL_B:-32}
RETRY_B=${RETRY_B:-$CANDIDATE_K}
VERIFIER_MODE=${VERIFIER_MODE:-full_polytope}
VERIFIER_WORKERS=${VERIFIER_WORKERS:-16}

if [[ "$HELIOS_GPU" != "0" && "$HELIOS_GPU" != "1" && "$HELIOS_GPU" != "3" ]]; then
  echo "HELIOS_GPU must be 0, 1, or 3" >&2
  exit 2
fi

python scripts/research_ball_expansion_optimization.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --lab-task-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json \
  --output "$OUT" \
  --device mps \
  --rounds "$ROUNDS" \
  --flow-nfe "$FLOW_NFE" \
  --batched-rollout-sampling \
  --fa-alloc none \
  --flow-base-std 1.0 \
  --flow-base-std-schedule none \
  --flow-base-std-final 1.0 \
  --candidate-perturb-std 0.0 \
  --candidate-perturb-scope coherent_horizon \
  --learning-rate 5e-5 \
  --learning-rate-final 1e-6 \
  --learning-rate-decay-steps "$OPTIMIZER_STEPS_TOTAL" \
  --gradient-clip-norm 5 \
  --round-learning-rate-warmup-power 2 \
  --trainable-trunk-layers 3 \
  --freeze-visual-encoder-during-expansion \
  --beta "$BETA" \
  --parallel-episodes "$PARALLEL_EPISODES" \
  --verifier-workers "$VERIFIER_WORKERS" \
  --max-retry-batches "$MAX_RETRY_BATCHES" \
  --retry-exhaustion-policy abort \
  --successful-trajectories-per-gamma 12 \
  --sample-update-mode "0,0,0,1,1,1,2,2,2,3,3,3" \
  --sample-update-cohorts unguided_only \
  --K "$CANDIDATE_K" \
  --B "$INITIAL_B" \
  --retry-B "$RETRY_B" \
  --retry-verify-all-fast-path \
  --inner-steps 1 \
  --optimizer-steps-total "$OPTIMIZER_STEPS_TOTAL" \
  --optimizer-step-allocation quadratic_round \
  --batch-size 256 \
  --replay-scope cumulative \
  --replay-batch-sampler mode_gamma_stratified \
  --replay-top-fraction 1.0 \
  --replay-selector uniform \
  --replay-rounds "$ROUNDS" \
  --gp-buffer-cap 1536 \
  --gp-reference-mode sliding_success_per_gamma_current_phi \
  --gp-sliding-row-selector trajectory_uniform \
  --negative-alpha 0 \
  --archive-rule successful_executed_windows \
  --successful-trajectory-selector random_success \
  --replay-acceptance execution_eligible \
  --execution-rule min_cost \
  --execution-step-margin-weight "$EXECUTION_STEP_MARGIN_WEIGHT" \
  --execution-taskspace-weight 0 \
  --acquisition-feature learned_phi \
  --coverage-replay none \
  --replay-augmentation none \
  --execution-z-bias-mode none \
  --tight-corridor \
  --verifier-mode "$VERIFIER_MODE" \
  --verifier-solver analytic \
  --event-log committed_success \
  --paired-noised-representation \
  --seed "$SEED"
