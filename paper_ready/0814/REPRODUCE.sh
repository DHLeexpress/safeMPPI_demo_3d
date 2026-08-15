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

python "$BUNDLE/source/reproduce_policy_site_rollouts.py" \
  --device "$DEVICE" \
  --model less-expanded \
  --verify-frozen \
  --output "$OUT/less_expanded_site_rollouts.pt"

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

python "$BUNDLE/source/run_multisphere_cfm_mppi_bowling.py" \
  --pretrain-dir "$BUNDLE/checkpoints/pre2" \
  --task-config "$BUNDLE/config/task_config_resolved.json" \
  --output "$OUT/cfmmppi_fixed_g01_matched8" \
  --device "$DEVICE" --gammas 0.1 --trials 8 --seed 314159 \
  --proposal-count 32 --elite-count 8 --copies-per-elite 32 \
  --mppi-sigma 0.20 --mppi-lambda 0.10 \
  --alpha-cbf 0.5 --cbf-margin-m 0.10 \
  --goal-coefficient-max 0.25 --safety-coefficient-max 1.0 \
  --regimes-json '{"safety":{"goal":0.0,"safety":1.0},"balanced":{"goal":0.5,"safety":1.0},"performance":{"goal":1.0,"safety":0.0}}'

while IFS= read -r SEED; do
  python "$BUNDLE/source/collect_paper_ready_safemppi_bowling.py" \
    --source-root "$BUNDLE/runtime_snapshot" \
    --config "$BUNDLE/config/safemppi_exact_bowling_config.json" \
    --output "$OUT/safemppi_gamma01_seed_${SEED}" \
    --device "$DEVICE" --gammas 0.1 \
    --attempts-per-gamma 1 --seed-start "$SEED"
done < <(
  python -c 'import csv,sys; rows=csv.DictReader(open(sys.argv[1])); print("\n".join(r["seed"] for r in rows if r["group"] == "paper-ready-safemppi" and abs(float(r["gamma"])-0.1) < 1e-9))' \
    "$BUNDLE/SITE_TRAJECTORY_INDEX.csv"
)

echo "Reproduction outputs: $OUT"
