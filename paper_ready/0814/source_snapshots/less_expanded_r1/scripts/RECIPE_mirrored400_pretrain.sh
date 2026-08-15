#!/bin/bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

ROOT=results/stage1_single_ball_t128/0810_pretrain_mirrored_400pg
MERGED="$ROOT/demos_merged"
PRE=results/stage1_single_ball_t128/0810_pretrain_mirrored400_hp100_t128_d3

for gamma_dir in gamma_0p1 gamma_0p3 gamma_0p5 gamma_1p0; do
  while [ ! -s "$ROOT/$gamma_dir/manifest.json" ]; do
    sleep 30
  done
done

python scripts/merge_mirrored_pair_archives.py \
  --source "$ROOT/gamma_0p1" \
  --source "$ROOT/gamma_0p3" \
  --source "$ROOT/gamma_0p5" \
  --source "$ROOT/gamma_1p0" \
  --output "$MERGED" \
  --expected-successes-per-gamma 400

python scripts/pretrain_lab_reference_flow.py \
  --helios \
  --helios-gpu 3 \
  --demo-dir "$MERGED" \
  --output "$PRE" \
  --context-model uniform_hp100 \
  --grid-token-dim 128 \
  --trunk-depth 3 \
  --hidden 48 \
  --representation-dim 32 \
  --nfe 16 \
  --epochs 500 \
  --batch-size 32 \
  --max-windows-per-trajectory 32 \
  --cuda-amp \
  --learning-rate 3e-4 \
  --patience 50 \
  --min-delta 1e-4 \
  --min-epochs 100 \
  --device cuda:0 \
  --audit-episodes 100 \
  --audit-seed 91000 \
  --ood-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json \
  --ood-audit-episodes 100 \
  --ood-audit-seed 191000 \
  --seed 0
