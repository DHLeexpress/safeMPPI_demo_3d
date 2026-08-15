#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

MERGED=${MERGED:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000/demos_merged_v2}
BASELINE=${BASELINE:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_baseline}
BALANCED=${BALANCED:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_noema}
PRE2=${PRE2:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema}

common=(
  --helios
  --helios-share-gpu
  --helios-detached
  --demo-dir "$MERGED"
  --context-model uniform_hp100
  --grid-token-dim 128
  --trunk-depth 3
  --hidden 48
  --representation-dim 32
  --nfe 16
  --epochs 500
  --max-windows-per-trajectory 32
  --cuda-amp
  --patience 50
  --min-delta 1e-4
  --min-epochs 100
  --deterministic-validation-seed 2000003
  --device cuda:0
  --audit-episodes 100
  --audit-seed 91000
  --ood-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json
  --ood-audit-episodes 100
  --ood-audit-seed 191000
  --seed 0
)

python scripts/pretrain_lab_reference_flow.py \
  "${common[@]}" \
  --helios-gpu 0 \
  --output "$BASELINE" \
  --batch-size 32 \
  --batch-sampler random \
  --optimizer adam \
  --learning-rate 3e-4 \
  > "${BASELINE}.launch.log" 2>&1 &
baseline_pid=$!

balanced_common=(
  --batch-size 256
  --batch-sampler gamma_mirror_balanced
  --optimizer adamw
  --weight-decay 1e-5
  --warmup-epochs 10
  --gradient-clip-norm 1.0
  --learning-rate 3e-4
  --min-epochs 150
  --patience 75
)

python scripts/pretrain_lab_reference_flow.py \
  "${common[@]}" \
  "${balanced_common[@]}" \
  --helios-gpu 1 \
  --output "$BALANCED" \
  > "${BALANCED}.launch.log" 2>&1 &
balanced_pid=$!

python scripts/pretrain_lab_reference_flow.py \
  "${common[@]}" \
  "${balanced_common[@]}" \
  --helios-gpu 3 \
  --output "$PRE2" \
  --ema-decay 0.999 \
  > "${PRE2}.launch.log" 2>&1 &
pre2_pid=$!

wait "$baseline_pid"
wait "$balanced_pid"
wait "$pre2_pid"

python scripts/compare_pretrain_training_runs.py \
  --arm "baseline=$BASELINE" \
  --arm "balanced_noema=$BALANCED" \
  --arm "balanced_ema=$PRE2" \
  --deployment-reference \
    "PRE400=results/stage1_single_ball_t128/0810_pretrain_mirrored400_hp100_t128_d3" \
  --output results/stage1_single_ball_t128/0810_pretrain_mirrored1000/PRE2_TRAINING_PROOF
