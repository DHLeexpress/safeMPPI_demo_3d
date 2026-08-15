#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

PRE=results/stage1_single_ball_t128/0810_pretrain_mirrored400_hp100_t128_d3
ROOT=results/stage1_single_ball_t128/0810_mirrored400_unguided_arms_r3

while [[ ! -s "$PRE/pretrained.pt" || ! -s "$PRE/pretrain_manifest.json" ]]; do
  sleep 30
done

mkdir -p "$ROOT"

GPU_ORDER=($(
  ssh dohyun@helios.robotics.caltech.edu \
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits" \
  | awk -F, '
      {
        for (i = 1; i <= 3; i++) gsub(/^ +| +$/, "", $i)
      }
      $1 == 0 || $1 == 1 || $1 == 3 {print $1, $2, $3}
    ' \
  | sort -k2,2n -k3,3n \
  | awk '{print $1}'
))

if [[ "${#GPU_ORDER[@]}" != "3" ]]; then
  echo "could not resolve Helios GPUs 0,1,3" >&2
  exit 2
fi

echo "GPU assignment: more_parallel=${GPU_ORDER[0]} "\
"control=${GPU_ORDER[1]} lower_beta=${GPU_ORDER[2]}"

screen -dmS mirrored400_arm_more_parallel bash -lc "
  cd /Users/dhl/Documents/safeMPPI_demo_3d &&
  PRE='$PRE' \
  OUT='$ROOT/more_parallel_beta05_p16' \
  HELIOS_GPU='${GPU_ORDER[0]}' BETA='0.5' PARALLEL_EPISODES='16' \
  bash scripts/RECIPE_mirrored400_unguided_arm.sh \
  > /tmp/mirrored400_arm_more_parallel.log 2>&1
"

screen -dmS mirrored400_arm_control bash -lc "
  cd /Users/dhl/Documents/safeMPPI_demo_3d &&
  PRE='$PRE' \
  OUT='$ROOT/control_beta05_p8' \
  HELIOS_GPU='${GPU_ORDER[1]}' BETA='0.5' PARALLEL_EPISODES='8' \
  bash scripts/RECIPE_mirrored400_unguided_arm.sh \
  > /tmp/mirrored400_arm_control.log 2>&1
"

screen -dmS mirrored400_arm_lower_beta bash -lc "
  cd /Users/dhl/Documents/safeMPPI_demo_3d &&
  PRE='$PRE' \
  OUT='$ROOT/lower_beta01_p8' \
  HELIOS_GPU='${GPU_ORDER[2]}' BETA='0.1' PARALLEL_EPISODES='8' \
  bash scripts/RECIPE_mirrored400_unguided_arm.sh \
  > /tmp/mirrored400_arm_lower_beta.log 2>&1
"
