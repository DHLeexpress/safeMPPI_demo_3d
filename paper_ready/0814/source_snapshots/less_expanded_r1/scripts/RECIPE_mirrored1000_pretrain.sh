#!/usr/bin/env bash
set -euo pipefail

cd /Users/dhl/Documents/safeMPPI_demo_3d

OLD_ROOT=${OLD_ROOT:-results/stage1_single_ball_t128/0810_pretrain_mirrored_400pg}
ADD_ROOT=${ADD_ROOT:-results/stage1_single_ball_t128/0810_pretrain_mirrored_add600pg}
MERGED=${MERGED:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000/demos_merged_v2}
PRE2=${PRE2:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema}
PRE2_BASELINE=${PRE2_BASELINE:-results/stage1_single_ball_t128/0810_pretrain_mirrored1000_hp100_t128_d3_baseline}
DOMAIN_SEED=${DOMAIN_SEED:-91702}
ROLLOUT_SEED_START=${ROLLOUT_SEED_START:-917200}

mkdir -p "$ADD_ROOT"

collect_shard() {
  local gamma=$1
  local tag=$2
  local gpu=$3
  python scripts/collect_mirrored_pair_success_quota.py \
    --helios \
    --helios-gpu "$gpu" \
    --helios-share-gpu \
    --helios-detached \
    --config configs/lab_clutter_cylinders_path_midpoint_uniform_mirrored_z01_17_v1.json \
    --output "$ADD_ROOT/gamma_$tag" \
    --target-successes-per-gamma 600 \
    --max-pair-attempts-per-gamma 3000 \
    --gammas "$gamma" \
    --domain-seed "$DOMAIN_SEED" \
    --rollout-seed-start "$ROLLOUT_SEED_START" \
    --device mps
}

collect_shard 0.1 0p1 3 > "$ADD_ROOT/gamma_0p1.launch.log" 2>&1 &
pid_01=$!
collect_shard 0.3 0p3 0 > "$ADD_ROOT/gamma_0p3.launch.log" 2>&1 &
pid_03=$!
collect_shard 0.5 0p5 1 > "$ADD_ROOT/gamma_0p5.launch.log" 2>&1 &
pid_05=$!
collect_shard 1.0 1p0 3 > "$ADD_ROOT/gamma_1p0.launch.log" 2>&1 &
pid_10=$!

wait "$pid_01"
wait "$pid_03"
wait "$pid_05"
wait "$pid_10"

python scripts/merge_mirrored_pair_archives.py \
  --source "$OLD_ROOT/gamma_0p1" \
  --source "$ADD_ROOT/gamma_0p1" \
  --source "$OLD_ROOT/gamma_0p3" \
  --source "$ADD_ROOT/gamma_0p3" \
  --source "$OLD_ROOT/gamma_0p5" \
  --source "$ADD_ROOT/gamma_0p5" \
  --source "$OLD_ROOT/gamma_1p0" \
  --source "$ADD_ROOT/gamma_1p0" \
  --output "$MERGED" \
  --expected-successes-per-gamma 1000

pretrain_common=(
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

# Same-data control: preserves the PRE400 optimizer/batching recipe.
python scripts/pretrain_lab_reference_flow.py \
  "${pretrain_common[@]}" \
  --helios-gpu 0 \
  --output "$PRE2_BASELINE" \
  --batch-size 32 \
  --batch-sampler random \
  --optimizer adam \
  --learning-rate 3e-4 \
  > "${PRE2_BASELINE}.launch.log" 2>&1 &
baseline_pid=$!

# PRE2 candidate: exact gamma balance, complete source/mirror units, stable
# full-network AdamW, warmup, gradient clipping, and EMA checkpoint selection.
python scripts/pretrain_lab_reference_flow.py \
  "${pretrain_common[@]}" \
  --helios-gpu 3 \
  --output "$PRE2" \
  --batch-size 256 \
  --batch-sampler gamma_mirror_balanced \
  --optimizer adamw \
  --weight-decay 1e-5 \
  --warmup-epochs 10 \
  --gradient-clip-norm 1.0 \
  --ema-decay 0.999 \
  --learning-rate 3e-4 \
  --min-epochs 150 \
  --patience 75 \
  > "${PRE2}.launch.log" 2>&1 &
advanced_pid=$!

wait "$baseline_pid"
wait "$advanced_pid"
