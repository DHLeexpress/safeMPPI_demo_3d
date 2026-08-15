#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
TASK_CONFIG=${TASK_CONFIG:-$ROOT/configs/lab_clutter_spheres_double_hourglass_n6_relaxed_z01_17_v1.json}
PHASE=${PHASE:-calibration}
OUT=${OUT:-$ROOT/results/stage2_multi_sphere_n6/0810_pre2_${PHASE}_seed81231}
HELIOS_GPU=${HELIOS_GPU:-1}
SEED=${SEED:-81231}

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac

cd "$ROOT"
if [[ "$PHASE" == calibration ]]; then
  python scripts/run_preflight_multisphere_pre2_on_helios.py \
    --phase calibration \
    --pretrain-dir "$PRE" \
    --task-config "$TASK_CONFIG" \
    --output "$OUT" \
    --helios-gpu "$HELIOS_GPU" \
    --flow-nfe 16 \
    --K 16 \
    --B 8 \
    --calibration-scenes 2 \
    --arm base,0,0.15,0,1.1,0.05 \
    --arm wall250_axis5,250,0.15,5,1.1,0.05 \
    --arm wall500_axis5,500,0.15,5,1.1,0.05 \
    --arm wall250_axis5_ctrl02,250,0.15,5,1.1,0.2 \
    --arm wall500_axis5_ctrl05,500,0.15,5,1.1,0.5 \
    --seed "$SEED"
elif [[ "$PHASE" == preflight ]]; then
  ARM=${ARM:-wall250_axis5,250,0.15,5,1.1,0.05}
  python scripts/run_preflight_multisphere_pre2_on_helios.py \
    --phase preflight \
    --pretrain-dir "$PRE" \
    --task-config "$TASK_CONFIG" \
    --output "$OUT" \
    --helios-gpu "$HELIOS_GPU" \
    --flow-nfe 16 \
    --K 16 \
    --B 8 \
    --parallel-episodes 8 \
    --quota-per-gamma 2 \
    --max-retry-batches 20 \
    --gallery-scenes 5 \
    --capture-stride 40 \
    --arm "$ARM" \
    --seed "$SEED"
else
  echo "PHASE must be calibration or preflight" >&2
  exit 2
fi
