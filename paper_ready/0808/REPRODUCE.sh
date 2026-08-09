#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ABSOLUTE_NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$1"
PYTHON_BIN="${PYTHON_BIN:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  echo "output directory must be absolute: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p \
  "${OUTPUT_ROOT}/trajectories/expanded_quality_v2" \
  "${OUTPUT_ROOT}/trajectories/expanded_supplement_v1" \
  "${OUTPUT_ROOT}/trajectories/expanded_string_safe_v1" \
  "${OUTPUT_ROOT}/trajectories/pretrained_success" \
  "${OUTPUT_ROOT}/trajectories/pretrained_collisions" \
  "${OUTPUT_ROOT}/figures" \
  "${OUTPUT_ROOT}/quality"

export PYTHONPATH="${BUNDLE_ROOT}/runtime_snapshot"

CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" \
  "${BUNDLE_ROOT}/source/export_selected_ball_rollouts.py" \
  --checkpoint "${BUNDLE_ROOT}/checkpoints/expanded_v1_reserve_G_nfe12.pt" \
  --expected-checkpoint-sha256 c1a3c77fc956c57d02a0970c4e54fca942cee391a68275a58134361e00828056 \
  --expected-nfe 12 \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --expected-config-sha256 7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2 \
  --selections "${BUNDLE_ROOT}/selections/expanded_quality_v2.json" \
  --policy-label expanded_reserve_G_quality_v2 \
  --output "${OUTPUT_ROOT}/trajectories/expanded_quality_v2" \
  --device cuda:0 \
  --physical-gpu 2 \
  --source-id 5c8a57779f16-008acd883e14 \
  --sampling-temperature 1.0 \
  --repeat 2 \
  --theta-tolerance-deg 0.05

CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" \
  "${BUNDLE_ROOT}/source/export_selected_ball_rollouts.py" \
  --checkpoint "${BUNDLE_ROOT}/checkpoints/expanded_v1_reserve_G_nfe12.pt" \
  --expected-checkpoint-sha256 c1a3c77fc956c57d02a0970c4e54fca942cee391a68275a58134361e00828056 \
  --expected-nfe 12 \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --expected-config-sha256 7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2 \
  --selections "${BUNDLE_ROOT}/selections/expanded_gamma1_octants_and_side_above_v1.json" \
  --policy-label expanded_reserve_G_octant_supplement_v1 \
  --output "${OUTPUT_ROOT}/trajectories/expanded_supplement_v1" \
  --device cuda:0 \
  --physical-gpu 3 \
  --source-id 5c8a57779f16-008acd883e14 \
  --sampling-temperature 1.0 \
  --repeat 2 \
  --theta-tolerance-deg 0.05

CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" \
  "${BUNDLE_ROOT}/source/export_selected_ball_rollouts.py" \
  --checkpoint "${BUNDLE_ROOT}/checkpoints/expanded_v1_reserve_G_nfe12.pt" \
  --expected-checkpoint-sha256 c1a3c77fc956c57d02a0970c4e54fca942cee391a68275a58134361e00828056 \
  --expected-nfe 12 \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --expected-config-sha256 7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2 \
  --selections "${BUNDLE_ROOT}/selections/expanded_gamma1_side_above_string_safe_v1.json" \
  --policy-label expanded_reserve_G_string_safe_v1 \
  --output "${OUTPUT_ROOT}/trajectories/expanded_string_safe_v1" \
  --device cuda:0 \
  --physical-gpu 3 \
  --source-id 5c8a57779f16-008acd883e14 \
  --sampling-temperature 1.0 \
  --repeat 2 \
  --theta-tolerance-deg 0.05

CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" \
  "${BUNDLE_ROOT}/source/export_selected_ball_rollouts.py" \
  --checkpoint "${BUNDLE_ROOT}/checkpoints/pretrained_p0806_nfe16.pt" \
  --expected-checkpoint-sha256 cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff \
  --expected-nfe 16 \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --expected-config-sha256 7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2 \
  --selections "${BUNDLE_ROOT}/selections/pretrained.json" \
  --policy-label pretrained_p0806 \
  --output "${OUTPUT_ROOT}/trajectories/pretrained_success" \
  --device cuda:0 \
  --physical-gpu 3 \
  --source-id 5c8a57779f16-008acd883e14 \
  --sampling-temperature 1.0 \
  --repeat 2 \
  --theta-tolerance-deg 0.05

CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" \
  "${BUNDLE_ROOT}/source/export_pretrained_collisions.py" \
  --checkpoint "${BUNDLE_ROOT}/checkpoints/pretrained_p0806_nfe16.pt" \
  --expected-checkpoint-sha256 cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff \
  --expected-nfe 16 \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --expected-config-sha256 7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2 \
  --selections "${BUNDLE_ROOT}/selections/pretrained_collisions.json" \
  --output "${OUTPUT_ROOT}/trajectories/pretrained_collisions" \
  --device cuda:0 \
  --physical-gpu 2 \
  --source-id 5c8a57779f16-008acd883e14 \
  --sampling-temperature 1.0 \
  --repeat 2

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/validate_expanded_quality_selection.py" \
  --trajectories "${OUTPUT_ROOT}/trajectories/expanded_quality_v2" \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --output "${OUTPUT_ROOT}/quality/expanded_quality_v2_summary.json"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/validate_expanded_angular_supplement.py" \
  --bundle "${BUNDLE_ROOT}" \
  --existing "${OUTPUT_ROOT}/trajectories/expanded_quality_v2" \
  --supplement "${OUTPUT_ROOT}/trajectories/expanded_supplement_v1" \
  --output "${OUTPUT_ROOT}/quality/expanded_angular_supplement_v1_summary.json"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/validate_expanded_string_safe.py" \
  --bundle "${BUNDLE_ROOT}" \
  --trajectory "${OUTPUT_ROOT}/trajectories/expanded_string_safe_v1/gamma_1_mode_above_seed_131629.npz" \
  --output "${OUTPUT_ROOT}/quality/expanded_string_safe_v1_summary.json"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/export_flight_references.py" \
  --bundle "${BUNDLE_ROOT}" \
  --trajectories "${OUTPUT_ROOT}/trajectories" \
  --output "${OUTPUT_ROOT}/flight_references" \
  --reference-prefix flight_references

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/plot_trajectory_review.py" \
  --config "${BUNDLE_ROOT}/config/task_config_resolved.json" \
  --expanded "${OUTPUT_ROOT}/trajectories/expanded_quality_v2" \
  --pretrained "${OUTPUT_ROOT}/trajectories/pretrained_success" \
  --collisions "${OUTPUT_ROOT}/trajectories/pretrained_collisions" \
  --output "${OUTPUT_ROOT}/figures"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/plot_expanded_angular_supplement.py" \
  --bundle "${BUNDLE_ROOT}" \
  --expanded "${OUTPUT_ROOT}/trajectories/expanded_quality_v2" \
  --supplement "${OUTPUT_ROOT}/trajectories/expanded_supplement_v1" \
  --output "${OUTPUT_ROOT}/figures"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/source/plot_expanded_string_safe.py" \
  --bundle "${BUNDLE_ROOT}" \
  --replacement "${OUTPUT_ROOT}/trajectories/expanded_string_safe_v1/gamma_1_mode_above_seed_131629.npz" \
  --output "${OUTPUT_ROOT}/figures"

echo "reproduction complete: ${OUTPUT_ROOT}"
