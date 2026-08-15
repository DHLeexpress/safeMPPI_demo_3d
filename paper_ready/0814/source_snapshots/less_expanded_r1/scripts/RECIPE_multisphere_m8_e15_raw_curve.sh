#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/dhl/Documents/safeMPPI_demo_3d
PLOTTER_ROOT=/Users/dhl/Documents/safe_flow_expansion
PRE=${PRE:-$ROOT/results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered}
: "${EXPANSION:?set the completed expansion arm directory}"
ROUNDS=${ROUNDS:-0-5}
HELIOS_GPU=${HELIOS_GPU:-1}
SCENE_BANK=${SCENE_BANK:-$ROOT/results/stage2_multi_sphere_n6/0811_pre2_dense_z_relaxedy_multipair_q10_trunk3/eval_m50_r0_r1/raw_eval.json}
EVAL_OUT=${EVAL_OUT:-$EXPANSION/eval_m50_m8_e15_r0_r5}
CURVE_JSONL=${CURVE_JSONL:-$EVAL_OUT/dense_z_random_m50_m8_e15_r0_r5.jsonl}
PAPER_OUT=${PAPER_OUT:-$EVAL_OUT/paper}
STEM=${STEM:-dense_z_random_m50_m8_e15_r0_r5}
ARM_LABEL=${ARM_LABEL:-M8 E15}

case "$HELIOS_GPU" in
  1|3) ;;
  *) echo "HELIOS_GPU must be 1 or 3 (GPU 0/2 are out of scope)" >&2; exit 2 ;;
esac

cd "$ROOT"
python scripts/evaluate_multisphere_min_cost_deployment.py \
  --helios \
  --helios-gpu "$HELIOS_GPU" \
  --helios-share-gpu \
  --helios-detached \
  --pretrain-dir "$PRE" \
  --expansion "$EXPANSION" \
  --checkpoint-rounds "$ROUNDS" \
  --scene-bank-json "$SCENE_BANK" \
  --evaluation-output "$EVAL_OUT" \
  --device cuda:0 \
  --episodes 50 \
  --samples-per-step 8 \
  --sampling-temperature 1.0 \
  --execution-clearance-exp-weight 15 \
  --execution-clearance-target-m 0.6 \
  --execution-clearance-exp-temperature 0.15 \
  --execution-taskspace-quadratic-weight 250 \
  --execution-taskspace-quadratic-target-m 0.15 \
  --execution-axis-cylinder-quadratic-weight 5 \
  --execution-axis-cylinder-radius-m 1.1 \
  --execution-control-weight 0.05 \
  --seed 91000

python scripts/convert_multisphere_raw_curve.py \
  --raw-eval "$EVAL_OUT/raw_eval.json" \
  --output "$CURVE_JSONL" \
  --expected-rounds "$ROUNDS" \
  --expected-m 50 \
  --expected-temperature 1.0 \
  --task-config "$EXPANSION/task_config_resolved.json" \
  --expansion-manifest "$EXPANSION/manifest.json"

python "$PLOTTER_ROOT/scripts/paper_b1_margin50_trends.py" \
  --arm "$ARM_LABEL=$CURVE_JSONL" \
  --panel-heading 'D. 3D multi spheres' \
  --outdir "$PAPER_OUT" \
  --stem "$STEM"

# Raw deployment contract: fixed unseen 50-scene bank per gamma/round, exactly
# 8 unverified flow plans per closed-loop step, global E15/native min-cost plan
# selection, and only post-execution SR/CR/OOB/timeout/window-validity scoring.
