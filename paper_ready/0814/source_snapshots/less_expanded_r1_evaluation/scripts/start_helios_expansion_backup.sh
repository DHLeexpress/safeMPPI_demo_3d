#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REMOTE_OUTPUT LOCAL_BACKUP SCREEN_NAME" >&2
  exit 2
fi

REMOTE_OUTPUT=$1
LOCAL_BACKUP=$2
SCREEN_NAME=$3
ROOT=/Users/dhl/Documents/safeMPPI_demo_3d

if screen -ls 2>/dev/null | grep -F ".${SCREEN_NAME}" >/dev/null; then
  echo "screen session already exists: $SCREEN_NAME" >&2
  exit 2
fi

mkdir -p "$LOCAL_BACKUP"
(
  cd "$LOCAL_BACKUP"
  screen -dmS "$SCREEN_NAME" -L \
    python "$ROOT/scripts/watch_helios_expansion_backup.py" \
    --remote-output "$REMOTE_OUTPUT" \
    --local-backup "$LOCAL_BACKUP" \
    --interval-seconds 240
)

watcher_ready=0
for _attempt in {1..120}; do
  if screen -ls 2>/dev/null | grep -F ".${SCREEN_NAME}" >/dev/null; then
    watcher_ready=1
    break
  fi
  sleep 0.5
done
if [[ "$watcher_ready" -ne 1 ]]; then
  echo "backup watcher failed to stay running; inspect $LOCAL_BACKUP/screenlog.0" >&2
  exit 1
fi
echo "started $SCREEN_NAME -> $LOCAL_BACKUP"
