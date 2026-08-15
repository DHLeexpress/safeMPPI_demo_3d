#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
EXP=${EXP:?set EXP to the completed dense-z r5 expansion directory}
OUT_BASE=${OUT_BASE:-$EXP/eval_m50_r0_r5}
PART=${PART:?set PART to even or odd}

case "$PART" in
  even)
    HELIOS_GPU=${HELIOS_GPU:-1}
    rounds=(0 2 4)
    output="$OUT_BASE/even_r0_r2_r4"
    ;;
  odd)
    HELIOS_GPU=${HELIOS_GPU:-3}
    rounds=(1 3 5)
    output="$OUT_BASE/odd_r1_r3_r5"
    ;;
  *) echo "PART must be even or odd" >&2; exit 2 ;;
esac

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac

cd "$ROOT"
python scripts/evaluate_sphere_clutter_expansion.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --expansion "$EXP" \
  --evaluation-output "$output" \
  --device cuda:0 \
  --episodes 50 \
  --probe-samples 1 \
  --sampling-temperature 1.0 \
  --evaluation-rounds "${rounds[@]}" \
  --video-rounds "${rounds[@]}" \
  --fixed-scene-rollouts 50 \
  --fixed-scene-mode bowling_123 \
  --seed 91000 \
  --metrics-only

