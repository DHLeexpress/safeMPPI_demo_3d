#!/usr/bin/env bash
set -euo pipefail

BUNDLE=$(cd "$(dirname "$0")" && pwd)
DEVICE=${DEVICE:-cuda:0}
OUT=${OUT:-$BUNDLE/reproduced}

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
export PYTHONPATH="$BUNDLE/runtime_snapshot:$BUNDLE/source${PYTHONPATH:+:$PYTHONPATH}"

python "$BUNDLE/VERIFY.py"

python "$BUNDLE/source/reproduce_policy_site_rollouts.py" \
  --device "$DEVICE" \
  --model pre2 \
  --verify-frozen \
  --output "$OUT/pre2_site_rollouts.pt"

python "$BUNDLE/source/reproduce_policy_site_rollouts.py" \
  --device "$DEVICE" \
  --model expanded \
  --verify-frozen \
  --output "$OUT/expanded_site_rollouts.pt"

python "$BUNDLE/source/run_multisphere_cfm_mppi_bowling.py" \
  --pretrain-dir "$BUNDLE/checkpoints/pre2" \
  --task-config "$BUNDLE/config/task_config_resolved.json" \
  --output "$OUT/cfmmppi_safety" \
  --device "$DEVICE" --trials 4 --seed 271828 \
  --proposal-count 32 --elite-count 8 --copies-per-elite 32 \
  --mppi-sigma 0.20 --mppi-lambda 0.10 \
  --alpha-cbf 0.5 --cbf-margin-m 0.10 \
  --goal-coefficient-max 0.25 --safety-coefficient-max 1.0 \
  --regimes-json '{"safety_alpha05":{"goal":0.0,"safety":1.0}}'

python "$BUNDLE/source/run_multisphere_cfm_mppi_bowling.py" \
  --pretrain-dir "$BUNDLE/checkpoints/pre2" \
  --task-config "$BUNDLE/config/task_config_resolved.json" \
  --output "$OUT/cfmmppi_reward" \
  --device "$DEVICE" --trials 8 --seed 271828 \
  --proposal-count 32 --elite-count 8 --copies-per-elite 32 \
  --mppi-sigma 0.20 --mppi-lambda 0.10 \
  --alpha-cbf 0.5 --cbf-margin-m 0.10 \
  --goal-coefficient-max 0.25 --safety-coefficient-max 1.0 \
  --regimes-json '{"reward_dominant":{"goal":1.0,"safety":0.0}}'

python "$BUNDLE/source/run_multisphere_cfm_mppi_bowling.py" \
  --pretrain-dir "$BUNDLE/checkpoints/pre2" \
  --task-config "$BUNDLE/config/task_config_resolved.json" \
  --output "$OUT/cfmmppi_balanced" \
  --device "$DEVICE" --trials 4 --seed 271828 \
  --proposal-count 32 --elite-count 8 --copies-per-elite 32 \
  --mppi-sigma 0.20 --mppi-lambda 0.10 \
  --alpha-cbf 0.5 --cbf-margin-m 0.10 \
  --goal-coefficient-max 0.25 --safety-coefficient-max 1.0 \
  --regimes-json '{"balanced_alpha05":{"goal":0.5,"safety":1.0}}'

echo "Reproduction outputs: $OUT"
