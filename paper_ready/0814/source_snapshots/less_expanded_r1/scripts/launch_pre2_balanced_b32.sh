#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

python scripts/pretrain_lab_reference_flow.py \
  --helios \
  --helios-gpu 0 \
  --helios-share-gpu \
  --helios-detached \
  --demo-dir results/stage1_single_ball_t128/0810_pretrain_mirrored1000/demos_merged_v2 \
  --output results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_b32 \
  --context-model uniform_hp100 \
  --grid-token-dim 128 \
  --trunk-depth 3 \
  --hidden 48 \
  --representation-dim 32 \
  --nfe 16 \
  --epochs 500 \
  --max-windows-per-trajectory 32 \
  --cuda-amp \
  --batch-size 32 \
  --batch-sampler gamma_mirror_balanced \
  --optimizer adamw \
  --weight-decay 1e-5 \
  --warmup-epochs 10 \
  --gradient-clip-norm 1.0 \
  --ema-decay 0.999 \
  --learning-rate 3e-4 \
  --patience 75 \
  --min-delta 1e-4 \
  --min-epochs 150 \
  --deterministic-validation-seed 2000003 \
  --device cuda:0 \
  --audit-episodes 100 \
  --audit-seed 91000 \
  --ood-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json \
  --ood-audit-episodes 100 \
  --ood-audit-seed 191000 \
  --seed 0
