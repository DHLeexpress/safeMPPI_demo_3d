#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python VERIFY.py

if [[ "${REGENERATE_REFERENCES:-0}" == "1" ]]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  python source/export_real_flight_references.py \
    --bundle "$ROOT" \
    --output "$TMP_DIR/flight_references"
  python - "$ROOT/flight_references" "$TMP_DIR/flight_references" <<'PY'
from pathlib import Path
import hashlib
import sys

expected, actual = map(Path, sys.argv[1:])
for path in expected.rglob("*.npz"):
    relative = path.relative_to(expected)
    other = actual / relative
    if not other.exists():
        raise SystemExit(f"missing regenerated reference: {relative}")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(path) != digest(other):
        raise SystemExit(f"byte mismatch: {relative}")
print("OK: regenerated 40 flight-reference archives are byte-identical")
PY
fi
