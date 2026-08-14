#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEVICE=${DEVICE:-cuda:0}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/cfm_mppi_bowling}
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK=${TASK:-$ROOT/results/stage2_multi_sphere_n6/0812_pre2_speed400_faithful_gpu1/s4_bowling_visual/stability_expansion/task_config_resolved.json}

python "$ROOT/scripts/run_multisphere_cfm_mppi_bowling.py" \
  --pretrain-dir "$PRE" \
  --task-config "$TASK" \
  --output "$OUT" \
  --device "$DEVICE" \
  --trials 4 \
  --seed 271828 \
  --proposal-count 32 \
  --elite-count 8 \
  --copies-per-elite 32 \
  --mppi-sigma 0.20 \
  --mppi-lambda 0.10 \
  --alpha-cbf 0.5 \
  --cbf-margin-m 0.10 \
  --goal-coefficient-max 0.25 \
  --safety-coefficient-max 1.0 \
  --regimes-json '{"safety_dominant":{"goal":0.0,"safety":1.0},"reward_dominant":{"goal":1.0,"safety":0.0},"balanced":{"goal":0.5,"safety":1.0}}'
