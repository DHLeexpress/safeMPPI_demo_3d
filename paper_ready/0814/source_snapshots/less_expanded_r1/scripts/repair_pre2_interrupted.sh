#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

root=results/stage1_single_ball_t128
demo=$root/0810_pretrain_mirrored1000/demos_merged_v2
baseline=$root/0810_pretrain_mirrored1000_hp100_t128_d3_baseline
noema_source=$root/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_noema/.best_training_state.pt
ema_source=$root/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema/.best_training_state.pt
noema=$root/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_noema_recovered
ema=$root/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered
proof=$root/0810_pretrain_mirrored1000/PRE2_TRAINING_PROOF_RECOVERED

for path in "$noema_source" "$ema_source"; do
  [[ -s "$path" ]] || { echo "missing recovery checkpoint: $path" >&2; exit 2; }
done
for directory in "$noema" "$ema"; do
  [[ ! -e "$directory" ]] || {
    echo "refusing to overwrite recovery output: $directory" >&2
    exit 2
  }
done

common=(
  --helios
  --helios-share-gpu
  --helios-detached
  --demo-dir "$demo"
  --context-model uniform_hp100
  --grid-token-dim 128
  --trunk-depth 3
  --hidden 48
  --representation-dim 32
  --nfe 16
  --epochs 500
  --max-windows-per-trajectory 32
  --cuda-amp
  --min-delta 1e-4
  --min-epochs 150
  --patience 75
  --deterministic-validation-seed 2000003
  --device cuda:0
  --audit-episodes 100
  --audit-seed 91000
  --ood-config configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json
  --ood-audit-episodes 100
  --ood-audit-seed 191000
  --seed 0
  --batch-size 256
  --batch-sampler gamma_mirror_balanced
  --optimizer adamw
  --weight-decay 1e-5
  --warmup-epochs 10
  --gradient-clip-norm 1.0
  --learning-rate 3e-4
  --recovered-observed-through-epoch 150
)

python scripts/pretrain_lab_reference_flow.py \
  "${common[@]}" \
  --helios-gpu 0 \
  --recover-best-checkpoint "$noema_source" \
  --output "$noema" \
  > "${noema}.launch.log" 2>&1 &
noema_pid=$!

python scripts/pretrain_lab_reference_flow.py \
  "${common[@]}" \
  --helios-gpu 3 \
  --ema-decay 0.999 \
  --recover-best-checkpoint "$ema_source" \
  --output "$ema" \
  > "${ema}.launch.log" 2>&1 &
ema_pid=$!

wait "$noema_pid"
wait "$ema_pid"

python scripts/compare_pretrain_training_runs.py \
  --arm "baseline=$baseline" \
  --arm "balanced_noema_recovered=$noema" \
  --arm "balanced_ema_recovered=$ema" \
  --deployment-reference \
    "PRE400=$root/0810_pretrain_mirrored400_hp100_t128_d3" \
  --output "$proof"

printf '%s complete: %s.json %s.md\n' \
  "$(date -u +%FT%TZ)" "$proof" "$proof" \
  > "$root/0810_pretrain_mirrored1000/PRE2_RECOVERY_STATUS.txt"
