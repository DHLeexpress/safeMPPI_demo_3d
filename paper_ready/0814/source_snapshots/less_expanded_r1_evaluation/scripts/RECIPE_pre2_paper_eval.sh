#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
EXPANSION=${EXPANSION:?set EXPANSION}
EVAL_OUT=${EVAL_OUT:?set EVAL_OUT}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 0, 1, or 3}
TARGET_ROUND=${TARGET_ROUND:?set TARGET_ROUND}

python scripts/research_evaluate_ball_expansion.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --expansion "$EXPANSION" \
  --evaluation-output "$EVAL_OUT" \
  --device mps \
  --episodes 40 \
  --probe-samples 16 \
  --flow-nfe 12 \
  --evaluation-rounds "$TARGET_ROUND" \
  --seed 91000 \
  --fixed-scene-rollouts 10 \
  --fixed-scene-seed 191000 \
  --gallery-rounds "$TARGET_ROUND" \
  --gallery-view head_on \
  --screening-only
