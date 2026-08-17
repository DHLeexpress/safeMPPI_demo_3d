#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"SHA256 mismatch: {relative}")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = manifest["rows"]
    if len(rows) != 8 or {float(row["gamma"]) for row in rows} != {0.3}:
        raise RuntimeError("fixed gamma/count contract failed")
    if Counter(row["route_family"] for row in rows) != {
        "ALL_LEFT": 3, "ALL_RIGHT": 3, "MIDDLE": 2,
    }:
        raise RuntimeError("route-family roster changed")
    if Counter(row["signature"] for row in rows) != {
        "LLLLLL": 3, "RRRRRR": 3, "LRLRLR": 1, "RRLRLL": 1,
    }:
        raise RuntimeError("signature roster changed")
    expected_scene = np.asarray(manifest["scene"]["cylinders"], np.float32)
    if manifest["screening"]["trials"] != 384:
        raise RuntimeError("z=0.9 screening contract changed")
    for row in rows:
        raw = np.load(ROOT / row["raw_rollout_file"], allow_pickle=False)
        ref = np.load(ROOT / row["flight_reference_file"], allow_pickle=False)
        if not np.array_equal(raw["cylinders"], expected_scene):
            raise RuntimeError("raw rollout scene changed")
        if not np.array_equal(ref["cylinders"], expected_scene):
            raise RuntimeError("flight-reference scene changed")
        if not np.array_equal(raw["dense_positions"], ref["position_ref"]):
            raise RuntimeError("dense position/reference mismatch")
        if str(raw["signature"]) != row["signature"]:
            raise RuntimeError("signature metadata mismatch")
        if bool(ref["hardware_eligible"]):
            raise RuntimeError("simulation reference marked hardware eligible")
        mean_abs_z = float(np.mean(np.abs(raw["dense_positions"][:, 2] - 0.9)))
        if not np.isclose(mean_abs_z, row["mean_abs_z_minus_0p9_m"], atol=1.0e-7):
            raise RuntimeError("z=0.9 score metadata mismatch")
        if mean_abs_z > 0.05 or row["min_clearance_m"] < 0.012:
            raise RuntimeError("selected trajectory violates z/clearance curation")
        if max(
            row["max_dense_position_reconstruction_error_m"],
            row["max_knot_position_reconstruction_error_m"],
            row["max_knot_velocity_reconstruction_error_mps"],
        ) > 2.0e-6:
            raise RuntimeError("reference recurrence error too large")
    if expected_scene.shape != (6, 3):
        raise RuntimeError("fixed scene is not the six-cylinder episode")
    print("PASS: 8 fixed six-cylinder SafeMPPI gamma=0.3 references verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
