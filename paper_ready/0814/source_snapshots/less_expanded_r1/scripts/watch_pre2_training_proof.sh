#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

root=results/stage1_single_ball_t128
baseline=$root/0810_pretrain_mirrored1000_hp100_t128_d3_baseline
balanced=$root/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_noema
ema=$root/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema
proof=$root/0810_pretrain_mirrored1000/PRE2_TRAINING_PROOF
status=$root/0810_pretrain_mirrored1000/PRE2_TRAINING_PROOF_STATUS.txt

while true; do
  missing=()
  for directory in "$baseline" "$balanced" "$ema"; do
    if [[ ! -s "$directory/pretrain_manifest.json" ]]; then
      missing+=("$(basename "$directory")")
    fi
  done
  if (( ${#missing[@]} == 0 )); then
    break
  fi
  printf '%s waiting: %s\n' "$(date -u +%FT%TZ)" "${missing[*]}" > "$status"
  sleep 60
done

python scripts/compare_pretrain_training_runs.py \
  --arm "baseline=$baseline" \
  --arm "balanced_noema=$balanced" \
  --arm "balanced_ema=$ema" \
  --deployment-reference \
    "PRE400=$root/0810_pretrain_mirrored400_hp100_t128_d3" \
  --output "$proof"

printf '%s complete: %s.json %s.md\n' \
  "$(date -u +%FT%TZ)" "$proof" "$proof" > "$status"
