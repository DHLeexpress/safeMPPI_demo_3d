#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
COLLECTOR=${COLLECTOR:-$ROOT/results/stage2_multi_sphere_n6/0812_axis180_expert_collector}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0812_axis180_expert_posttrain}
DEVICE=${DEVICE:-cuda:0}
STEPS_PER_PREFIX=${STEPS_PER_PREFIX:-2000}
DRY_RUN=${DRY_RUN:-1}

case "$DRY_RUN" in
  0) dry_run=() ;;
  1) dry_run=(--dry-run) ;;
  *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

cd "$ROOT"
python scripts/posttrain_multisphere_axis180_expert.py \
  --pretrain-dir "$PRE" \
  --collector "$COLLECTOR" \
  --output "$OUT" \
  --device "$DEVICE" \
  --prefix-trajectories 40 80 120 160 200 \
  --steps-per-prefix "$STEPS_PER_PREFIX" \
  --batch-size 256 \
  --context-batch-size 1024 \
  --learning-rate 5e-5 \
  --final-learning-rate 1e-6 \
  --gradient-clip-norm 1.0 \
  --seed 20260812 \
  "${dry_run[@]}"

# Safety/default contract:
# - DRY_RUN=1 performs the full manifest/NPZ replay, PRE2 context rebuild,
#   pair-prefix balance audit, and 12,926-parameter scope audit without writes.
# - Set DRY_RUN=0 only after the collector has exactly 200 admitted trajectories.
# - Run inside an allocated Helios GPU 1 or 3 shell; this recipe does not submit
#   or detach a remote job by itself.
# - checkpoint_000 is exact PRE2; 001..005 are cumulative 40..200 prefixes.
