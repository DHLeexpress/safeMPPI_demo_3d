#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

OUT=$1
DEVICE=${2:-mps}
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=$(cd "$HERE/.." && pwd)
REPO=$(cd "$BUNDLE/../.." && pwd)
SOURCE_SHA=9cafc00551e4964b9dbe559b1a4ba95104e9c88a
CONFIG_SHA=7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2
CONFIG="$BUNDLE/config/task_config_resolved.json"

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 2
fi
git -C "$REPO" cat-file -e "$SOURCE_SHA^{commit}"
git -C "$REPO" diff --quiet "$SOURCE_SHA" -- safe_mppi
mkdir -p "$OUT/recordings"

PYTHONDONTWRITEBYTECODE=1 python "$BUNDLE/source/screen_safemppi_modes.py" \
  --repo "$REPO" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-source-git-sha "$SOURCE_SHA" \
  --gammas 0.1 0.3 0.5 1.0 \
  --seed-start 0 \
  --seed-count 64 \
  --device "$DEVICE" \
  --output "$OUT/mode_screen"

for spec in 0.1:41 0.3:52 0.5:12 1.0:48; do
  gamma=${spec%%:*}
  seed=${spec##*:}
  PYTHONDONTWRITEBYTECODE=1 python \
    "$REPO/paper_ready/0806/inputs/0806_flight_demonstration_suite/renderer/p0806_record_safemppi.py" \
    --repo "$REPO" \
    --config "$CONFIG" \
    --expected-config-sha256 "$CONFIG_SHA" \
    --gammas "$gamma" \
    --seed "$seed" \
    --device "$DEVICE" \
    --output "$OUT/recordings/g${gamma}_s${seed}"
done

echo "SafeMPPI screen and four bit-identical recordings completed at $OUT"
