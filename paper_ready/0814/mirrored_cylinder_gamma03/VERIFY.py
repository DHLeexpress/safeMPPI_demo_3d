#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
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
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["gamma"] == 0.3
    assert manifest["trajectory_count"] == 8
    assert set(manifest["modes"]) == {
        "LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"
    }
    pairs: dict[str, set[str]] = {}
    for row in manifest["rows"]:
        assert np.isclose(row["gamma"], 0.3)
        assert row["max_dense_position_reconstruction_error_m"] <= 2e-6
        assert sha256(ROOT / row["raw_demo_file"]) == row["raw_demo_sha256"]
        assert sha256(ROOT / row["flight_reference_file"]) == row["flight_reference_sha256"]
        with np.load(ROOT / row["flight_reference_file"], allow_pickle=False) as ref:
            assert bool(ref["hardware_eligible"]) is False
            assert str(ref["mode"]) == row["mode"]
            assert int(ref["seed"]) == row["rollout_seed"]
            assert np.isclose(float(ref["gamma"]), 0.3)
            assert ref["position_ref"].shape == ref["velocity_ref"].shape
            assert ref["position_ref"].shape == ref["acceleration_ref"].shape
        pairs.setdefault(row["pair_id"], set()).add(row["pair_member"])
    assert len(pairs) == 4
    assert all(members == {"source", "mirror"} for members in pairs.values())

    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256(ROOT / relative) == expected, relative
    print("OK: gamma 0.3, 8/8 signatures, 4 complete mirrored pairs, hashes valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
