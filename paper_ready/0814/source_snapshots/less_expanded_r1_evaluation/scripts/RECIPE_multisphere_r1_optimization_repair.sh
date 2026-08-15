#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
DATA=${DATA:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_dense_z_relaxedy_multipair_q10_trunk3}
OUT_BASE=${OUT_BASE:-$ROOT/results/stage2_multi_sphere_n6/0811_r1_optimization_repair}
DEVICE=${DEVICE:-mps}

cd "$ROOT"

# Preferred bounded arm: update the final 128-D H_P token projection plus the
# complete 12,926-parameter flow.  A complete original/axis-180 pair per gamma
# is held out to select the stopping step; training then restarts from PRE2 on
# all 40 committed lineages for exactly that many steps.
python scripts/repair_multisphere_r1_optimization.py \
  --pretrain-dir "$PRE" \
  --data-expansion "$DATA" \
  --output "${OUT_BASE}_projection" \
  --device "$DEVICE" \
  --encoder-scope last_projection \
  --maximum-steps 480 \
  --minimum-steps 72 \
  --audit-every 24 \
  --audit-draws 3 \
  --patience 8 \
  --batch-size 160 \
  --flow-learning-rate 1e-5 \
  --final-flow-learning-rate 1e-6 \
  --encoder-learning-rate 5e-7 \
  --final-encoder-learning-rate 1e-7 \
  --vector-distillation-weight 0.25 \
  --feature-distillation-weight 0.05 \
  --ema-decay 0.995 \
  --seed 20260811

# Canonical raw M=50 curve for the repaired one-round package.
python scripts/evaluate_sphere_clutter_expansion.py \
  --pretrain-dir "$PRE" \
  --expansion "${OUT_BASE}_projection" \
  --evaluation-output "${OUT_BASE}_projection/eval_raw_m50" \
  --device "$DEVICE" \
  --episodes 50 \
  --probe-samples 0 \
  --sampling-temperature 1.0 \
  --evaluation-rounds 0 1 \
  --video-rounds 0 1 \
  --fixed-scene-rollouts 10 \
  --fixed-scene-mode bowling_123 \
  --seed 91000 \
  --metrics-only

# Independent deployment ablation on the exact same 50-scene bank.  This is
# intentionally not substituted into the canonical raw curve.
python scripts/evaluate_multisphere_min_cost_deployment.py \
  --pretrain-dir "$PRE" \
  --expansion "${OUT_BASE}_projection" \
  --checkpoint-round 1 \
  --scene-bank-json "${OUT_BASE}_projection/eval_raw_m50/raw_eval.json" \
  --output "${OUT_BASE}_projection/eval_bestof8_t085" \
  --device "$DEVICE" \
  --episodes 50 \
  --samples-per-step 8 \
  --sampling-temperature 0.85 \
  --execution-clearance-exp-weight 15 \
  --execution-clearance-target-m 0.6 \
  --execution-clearance-exp-temperature 0.15 \
  --seed 91000
