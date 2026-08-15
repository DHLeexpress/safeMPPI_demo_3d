#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=${PRE:?set PRE}
EXPANSION=${EXPANSION:?set EXPANSION}
EVAL_OUT=${EVAL_OUT:?set EVAL_OUT}
EXPANSION_MANIFEST=${EXPANSION_MANIFEST:?set EXPANSION_MANIFEST}
HELIOS_GPU=${HELIOS_GPU:?set HELIOS_GPU to 0, 1, or 3}
CHECKPOINT_ROUND=${CHECKPOINT_ROUND:?set CHECKPOINT_ROUND to 1 or 3}
COVERAGE_EPISODES=${COVERAGE_EPISODES:-80}

if [[ "$HELIOS_GPU" != "0" && "$HELIOS_GPU" != "1" && "$HELIOS_GPU" != "3" ]]; then
  echo "HELIOS_GPU must be 0, 1, or 3" >&2
  exit 2
fi
if (( CHECKPOINT_ROUND < 1 || CHECKPOINT_ROUND % 2 == 0 || CHECKPOINT_ROUND % 5 == 0 )); then
  echo "CHECKPOINT_ROUND must be a positive odd non-full-eval round" >&2
  exit 2
fi
if [[ ! -s "$EXPANSION/checkpoint_$(printf '%03d' "$CHECKPOINT_ROUND").pt" ]]; then
  echo "checkpoint $CHECKPOINT_ROUND is not fully published in $EXPANSION" >&2
  exit 3
fi

python scripts/research_evaluate_ball_expansion.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --expansion "$EXPANSION" \
  --expansion-manifest "$EXPANSION_MANIFEST" \
  --evaluation-output "$EVAL_OUT" \
  --device mps \
  --episodes "$COVERAGE_EPISODES" \
  --probe-samples 16 \
  --flow-nfe 12 \
  --evaluation-rounds "$CHECKPOINT_ROUND" \
  --seed 91000 \
  --metrics-only
