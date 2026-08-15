#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:-results/stage1_single_ball_t128/0809_route_guided_assets/pretrain_with_route_prototypes}
OUT=${OUT:-results/stage1_single_ball_t128/0809_route_injection_q2_r2_seed32}
HELIOS_GPU=${HELIOS_GPU:-3}
SEED=${SEED:-32}
ROUNDS=${ROUNDS:-2}
RETRY_CAP=${RETRY_CAP:-64}

if [[ "$HELIOS_GPU" == "2" ]]; then
  echo "Helios physical GPU 2 is reserved; use GPU 0, 1, or 3" >&2
  exit 2
fi

python scripts/research_ball_expansion_route_guided.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --route-guide-prototypes route_prototypes_s19r10_s20r10_s28r9.pt \
  --route-guide-band 1.0 \
  --route-guide-velocity-weight 0.5 \
  --route-guide-cosine-weight 0.25 \
  --route-guide-time-decay 0.9 \
  --route-guide-proposal-strengths "1,1,0.85,0.85,0.7,0.7,0.5,0.35" \
  --lab-task-config configs/lab_ball_stage1_t128.json \
  --output "$OUT" \
  --device mps \
  --rounds "$ROUNDS" \
  --flow-nfe 12 \
  --batched-rollout-sampling \
  --geofence-floor-z 0.1 \
  --fa-alloc steps \
  --fa-alloc-map "0,0,1,1,2,2,3,3" \
  --fa-alloc-retry-map missing_quota \
  --flow-base-std 1.1 \
  --flow-base-std-schedule cosine \
  --flow-base-std-final 1.0 \
  --candidate-perturb-std 0.0 \
  --candidate-perturb-scope coherent_horizon \
  --learning-rate 5e-5 \
  --learning-rate-final 1e-6 \
  --learning-rate-decay-steps 128 \
  --gradient-clip-norm 5 \
  --round-learning-rate-warmup-power 2 \
  --trainable-trunk-layers 3 \
  --freeze-visual-encoder-during-expansion \
  --beta 0.1 \
  --parallel-episodes 8 \
  --verifier-workers 32 \
  --max-retry-batches 32 \
  --retry-exhaustion-policy resample_scene \
  --retry-resample-batch-cap "$RETRY_CAP" \
  --successful-trajectories-per-gamma 8 \
  --sample-update-mode "0,0,1,1,2,2,3,3" \
  --K 16 \
  --B 8 \
  --retry-B 16 \
  --retry-verify-all-fast-path \
  --inner-steps 1 \
  --optimizer-steps-total 128 \
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
  --seed "$SEED"
