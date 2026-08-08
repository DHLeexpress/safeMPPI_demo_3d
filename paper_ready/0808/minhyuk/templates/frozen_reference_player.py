#!/usr/bin/env python3
"""Validated 100 Hz playback core for one frozen 0808 reference.

This file deliberately has no Crazyflie import.  Minhyuk must copy it into a
new run and connect ``send_full_state`` to the actual, versioned hardware
runner.  Running this file directly is validation-only and never sends a
command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(path: Path, expected_sha256: str | None = None) -> dict:
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"reference SHA mismatch: {actual_sha256}")
    with np.load(path, allow_pickle=False) as archive:
        reference = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "time_s",
        "position_ref",
        "velocity_ref",
        "acceleration_ref",
        "status",
        "checkpoint_sha256",
    }
    missing = sorted(required.difference(reference))
    if missing:
        raise ValueError(f"reference is missing {missing}")
    count = len(reference["time_s"])
    for key in ("position_ref", "velocity_ref", "acceleration_ref"):
        if reference[key].shape != (count, 3):
            raise ValueError(f"{key} must have shape ({count}, 3)")
    delta = np.diff(reference["time_s"])
    if len(delta) and not np.allclose(delta, 0.01, atol=1.0e-9, rtol=0.0):
        raise ValueError("reference is not uniformly sampled at 100 Hz")
    numeric_required = {
        "time_s", "position_ref", "velocity_ref", "acceleration_ref"
    }
    if any(not np.isfinite(reference[key]).all() for key in numeric_required):
        raise ValueError("reference contains non-finite values")
    reference["sha256"] = actual_sha256
    return reference


def stream_reference(
    reference: dict,
    send_full_state: Callable[[np.ndarray, np.ndarray, np.ndarray], None],
    *,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Send the stored arrays once, without interpolation or re-governing."""
    status = str(np.asarray(reference.get("status", "UNKNOWN")).item())
    if status != "SUCCESS":
        raise ValueError(
            f"hardware playback is forbidden for simulated status {status}"
        )
    start = now()
    for index, timestamp in enumerate(reference["time_s"]):
        target = start + float(timestamp)
        remaining = target - now()
        if remaining > 0:
            sleep(remaining)
        send_full_state(
            reference["position_ref"][index],
            reference["velocity_ref"][index],
            reference["acceleration_ref"][index],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    reference = load_reference(args.reference, args.expected_sha256)
    report = {
        "status": "VALIDATION_ONLY_PASS",
        "commands_sent": 0,
        "reference": str(args.reference.resolve()),
        "sha256": reference["sha256"],
        "samples": len(reference["time_s"]),
        "rate_hz": 100,
        "duration_s": float(reference["time_s"][-1]),
        "first_position": reference["position_ref"][0].tolist(),
        "last_position": reference["position_ref"][-1].tolist(),
        "simulated_status": str(np.asarray(reference["status"]).item()),
        "policy_checkpoint_sha256": str(
            np.asarray(reference["checkpoint_sha256"]).item()
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
