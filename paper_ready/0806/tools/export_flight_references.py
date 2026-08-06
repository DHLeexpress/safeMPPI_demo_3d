#!/usr/bin/env python3
"""Export frozen 100 Hz references from the approved 0806 trajectory archives.

This script never calls SafeMPPI, a flow policy, or ReferenceGovernor.  The
governor has already been applied exactly once in each source NPZ.  It only
checks that the stored recurrence is self-consistent and converts it into the
time/position/velocity/acceleration arrays expected by a reference player.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


POLICIES = ("safemppi", "pretrained")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _run_outcomes(scene: Path, policy: str) -> dict[tuple[float, int], dict]:
    summary = json.loads((scene / policy / "summary.json").read_text())
    return {
        (round(float(row["gamma"]), 6), int(row["seed"])): row
        for row in summary["runs"]
    }


def reconstruct_reference(
    source: Path,
    *,
    dt: float,
    substeps: int,
    max_speed: float,
    max_vertical_speed: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(source, allow_pickle=True) as archive:
        dense = np.asarray(archive["dense_positions"], np.float32)
        states = np.asarray(archive["states"], np.float32)
        raw = np.asarray(archive["controls"], np.float32)
        applied = np.asarray(archive["executed_controls"], np.float32)
        gamma = float(archive["gamma"])
        seed = int(archive["seed"])

    if raw.shape != applied.shape or applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError(f"control shape mismatch in {source}")
    expected_dense = 1 + len(applied) * substeps
    if dense.shape != (expected_dense, 3):
        raise ValueError(
            f"dense reference mismatch in {source}: {dense.shape} != "
            f"({expected_dense}, 3)"
        )
    if states.ndim != 2 or states.shape[1] < 6 or len(states) != len(applied) + 1:
        raise ValueError(f"state shape mismatch in {source}")

    dt_sub = float(dt) / int(substeps)
    positions = [states[0, :3].astype(np.float32).copy()]
    velocities = [states[0, 3:6].astype(np.float32).copy()]
    position = positions[0].copy()
    velocity = velocities[0].copy()
    for acceleration in applied:
        for _ in range(substeps):
            velocity = velocity + np.float32(dt_sub) * acceleration
            speed = float(np.linalg.norm(velocity))
            if speed > max_speed:
                velocity = velocity * np.float32(max_speed / speed)
            velocity[2] = np.clip(
                velocity[2], -max_vertical_speed, max_vertical_speed
            )
            position = position + np.float32(dt_sub) * velocity
            positions.append(position.astype(np.float32).copy())
            velocities.append(velocity.astype(np.float32).copy())

    reconstructed = np.asarray(positions, np.float32)
    velocity_ref = np.asarray(velocities, np.float32)
    position_error = float(np.max(np.abs(reconstructed - dense), initial=0.0))
    state_position_error = float(
        np.max(np.abs(reconstructed[::substeps] - states[:, :3]), initial=0.0)
    )
    state_velocity_error = float(
        np.max(np.abs(velocity_ref[::substeps] - states[:, 3:6]), initial=0.0)
    )
    tolerance = 2.0e-6
    if max(position_error, state_position_error, state_velocity_error) > tolerance:
        raise ValueError(
            f"stored governor recurrence does not reproduce {source}: "
            f"dense={position_error:.3e}, state_p={state_position_error:.3e}, "
            f"state_v={state_velocity_error:.3e}"
        )

    acceleration_ref = np.repeat(applied, substeps, axis=0)
    acceleration_ref = np.concatenate(
        [acceleration_ref, applied[-1:].copy()], axis=0
    ).astype(np.float32)
    time_s = (np.arange(len(dense), dtype=np.float64) * dt_sub).astype(np.float64)
    payload = {
        "time_s": time_s,
        "position_ref": dense,
        "velocity_ref": velocity_ref,
        "acceleration_ref": acceleration_ref,
        "raw_controls_10hz": raw,
        "executed_controls_10hz": applied,
        "gamma": np.float32(gamma),
        "seed": np.int64(seed),
        "reference_rate_hz": np.float32(1.0 / dt_sub),
        "source_control_rate_hz": np.float32(1.0 / dt),
    }
    diagnostics = {
        "max_dense_position_reconstruction_error_m": position_error,
        "max_knot_position_reconstruction_error_m": state_position_error,
        "max_knot_velocity_reconstruction_error_mps": state_velocity_error,
        "max_speed_mps": float(np.linalg.norm(velocity_ref, axis=1).max()),
        "max_vertical_speed_mps": float(np.abs(velocity_ref[:, 2]).max()),
        "max_executed_acceleration_component_mps2": float(np.abs(applied).max()),
        "duration_s": float(time_s[-1]),
    }
    return payload, diagnostics


def export_scene(scene: Path, *, force_empty: bool = False) -> list[dict]:
    config_path = scene / "concrete_config.json"
    config = json.loads(config_path.read_text())
    mppi = config["safemppi"]
    output = scene / "flight_references"
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing and not force_empty:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")

    rows: list[dict] = []
    for policy in POLICIES:
        outcomes = _run_outcomes(scene, policy)
        for source in sorted((scene / policy).glob("run_*.npz")):
            payload, diagnostics = reconstruct_reference(
                source,
                dt=float(mppi["dt"]),
                substeps=int(mppi["integration_substeps"]),
                max_speed=float(mppi["max_speed"]),
                max_vertical_speed=float(mppi["max_vertical_speed"]),
            )
            gamma = float(payload["gamma"])
            seed = int(payload["seed"])
            outcome = outcomes[(round(gamma, 6), seed)]
            name = (
                f"{policy}_gamma_{gamma_token(gamma)}_seed_{seed}_100hz.npz"
            )
            target = output / name
            np.savez_compressed(target, **payload)
            rows.append(
                {
                    "scene_id": scene.name,
                    "policy": policy,
                    "gamma": gamma,
                    "seed": seed,
                    "simulated_outcome": outcome["status"],
                    "source_archive": str(source.relative_to(scene)),
                    "source_archive_sha256": sha256_file(source),
                    "flight_reference": str(target.relative_to(scene)),
                    "flight_reference_sha256": sha256_file(target),
                    **diagnostics,
                }
            )
    if len(rows) != 8:
        raise ValueError(f"expected 8 flight cells in {scene}, got {len(rows)}")
    manifest = {
        "status": "P0806_FROZEN_100HZ_REFERENCES_COMPLETE",
        "scene_id": scene.name,
        "concrete_config_sha256": sha256_file(config_path),
        "contract": {
            "governor_application_count": 1,
            "policy_or_planner_called_by_exporter": False,
            "reference_player_must_not_reapply_governor": True,
            "position_semantics": "stored dense_positions, unchanged",
            "velocity_semantics": "exact governed recurrence at each 100 Hz knot",
            "acceleration_semantics": (
                "stored executed_controls held piecewise constant over each "
                "10 Hz control interval"
            ),
        },
        "runs": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="0806_flight_demonstration_suite directory",
    )
    args = parser.parse_args()
    all_rows: list[dict] = []
    scenes = sorted((args.suite / "scenes").iterdir())
    if [scene.name for scene in scenes] != [
        "symmetric_scene_inner",
        "symmetric_scene_outer",
    ]:
        raise ValueError("suite must contain exactly the approved inner/outer scenes")
    for scene in scenes:
        all_rows.extend(export_scene(scene))
    if len(all_rows) != 16:
        raise ValueError(f"expected 16 flight cells, got {len(all_rows)}")
    print(json.dumps({"status": "PASS", "references": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
