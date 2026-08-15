#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
EXP=${EXP:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_dense_z_relaxedy_multipair_q10_trunk3}
OUT=${OUT:-$EXP/eval_m50_r0_r1}
HELIOS_GPU=${HELIOS_GPU:-3}

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
  --evaluation-output "$OUT" \
  --device cuda:0 \
  --episodes 50 \
  --probe-samples 0 \
  --sampling-temperature 1.0 \
  --evaluation-rounds 0 1 \
  --fixed-scene-rollouts 50 \
  --fixed-scene-mode bowling_123 \
  --seed 91000 \
  --metrics-only
