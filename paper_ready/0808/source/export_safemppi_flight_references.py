#!/usr/bin/env python3
"""Export the four frozen SafeMPPI recordings as exact 100 Hz references."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct(
    source: Path,
    *,
    dt: float,
    substeps: int,
    max_speed: float,
    max_vertical_speed: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    with np.load(source, allow_pickle=True) as archive:
        states = np.asarray(archive["states"], np.float32)
        raw = np.asarray(archive["controls"], np.float32)
        applied = np.asarray(archive["executed_controls"], np.float32)
        stored_dense = np.asarray(archive["dense_positions"], np.float32)
        gamma = float(archive["gamma"].item())
        seed = int(archive["seed"].item())

    if raw.shape != applied.shape or states.shape != (len(applied) + 1, 6):
        raise ValueError(f"state/control shape mismatch in {source}")
    if stored_dense.shape != (len(applied) * substeps + 1, 3):
        raise ValueError(f"dense-position shape mismatch in {source}")

    dt_sub = np.float32(dt / substeps)
    positions = [states[0, :3].copy()]
    velocities = [states[0, 3:6].copy()]
    position = positions[0].copy()
    velocity = velocities[0].copy()
    for acceleration in applied:
        for _ in range(substeps):
            velocity = velocity + dt_sub * acceleration
            speed = float(np.linalg.norm(velocity))
            if speed > max_speed:
                velocity = velocity * np.float32(max_speed / speed)
            velocity[2] = np.clip(
                velocity[2], -max_vertical_speed, max_vertical_speed
            )
            position = position + dt_sub * velocity
            positions.append(position.astype(np.float32).copy())
            velocities.append(velocity.astype(np.float32).copy())

    reconstructed_positions = np.asarray(positions, np.float32)
    reconstructed_velocities = np.asarray(velocities, np.float32)
    dense_error = float(np.max(np.abs(reconstructed_positions - stored_dense)))
    knot_position_error = float(
        np.max(np.abs(reconstructed_positions[::substeps] - states[:, :3]))
    )
    knot_velocity_error = float(
        np.max(np.abs(reconstructed_velocities[::substeps] - states[:, 3:6]))
    )
    if max(dense_error, knot_position_error, knot_velocity_error) > 2.0e-6:
        raise ValueError(
            f"governed recurrence mismatch in {source}: dense={dense_error:.3e}, "
            f"knot_p={knot_position_error:.3e}, knot_v={knot_velocity_error:.3e}"
        )

    acceleration_ref = np.concatenate(
        [np.repeat(applied, substeps, axis=0), applied[-1:]], axis=0
    ).astype(np.float32)
    time_s = np.arange(len(stored_dense), dtype=np.float64) * (dt / substeps)
    payload = {
        "time_s": time_s,
        "position_ref": stored_dense,
        "velocity_ref": reconstructed_velocities,
        "acceleration_ref": acceleration_ref,
        "raw_controls_10hz": raw,
        "applied_controls_10hz": applied,
        "status": np.str_("SUCCESS"),
        "gamma": np.float32(gamma),
        "seed": np.int64(seed),
        "planner": np.str_("SafeMPPI"),
        "checkpoint_sha256": np.str_("NOT_APPLICABLE_SAFEMPPI"),
        "source_git_sha": np.str_(
            "9cafc00551e4964b9dbe559b1a4ba95104e9c88a"
        ),
        "config_sha256": np.str_(
            "7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2"
        ),
        "reference_rate_hz": np.float32(1.0 / (dt / substeps)),
        "source_control_rate_hz": np.float32(1.0 / dt),
    }
    diagnostics: dict[str, float | int] = {
        "gamma": gamma,
        "seed": seed,
        "samples": len(stored_dense),
        "duration_s": float(time_s[-1]),
        "max_dense_position_reconstruction_error_m": dense_error,
        "max_knot_position_reconstruction_error_m": knot_position_error,
        "max_knot_velocity_reconstruction_error_mps": knot_velocity_error,
        "max_speed_mps": float(np.linalg.norm(reconstructed_velocities, axis=1).max()),
        "max_vertical_speed_mps": float(np.abs(reconstructed_velocities[:, 2]).max()),
        "max_applied_acceleration_component_mps2": float(np.abs(applied).max()),
    }
    return payload, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = (args.output or bundle / "safemppi" / "flight_references").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    config_path = bundle / "config" / "task_config_resolved.json"
    config = json.loads(config_path.read_text())
    mppi = config["safemppi"]
    selections = json.loads(
        (bundle / "safemppi" / "selection.json").read_text()
    )["representatives"]
    rows: list[dict[str, object]] = []
    for selection in selections:
        source = bundle / str(selection["recording"])
        payload, diagnostics = reconstruct(
            source,
            dt=float(mppi["dt"]),
            substeps=int(mppi["integration_substeps"]),
            max_speed=float(mppi["max_speed"]),
            max_vertical_speed=float(mppi["max_vertical_speed"]),
        )
        if not np.isclose(
            diagnostics["gamma"], float(selection["gamma"]), atol=1.0e-7, rtol=0.0
        ):
            raise ValueError(f"gamma mismatch in {source}")
        if diagnostics["seed"] != int(selection["seed"]):
            raise ValueError(f"seed mismatch in {source}")
        payload["mode"] = np.str_(selection["mode"])
        target = output / f"safemppi_g{selection['gamma']:g}_{selection['mode']}_s{selection['seed']}_100hz.npz"
        np.savez_compressed(target, **payload)
        rows.append({
            "flight_id": (
                f"0808_safemppi_g{selection['gamma']:g}_"
                f"{selection['mode']}_s{selection['seed']}"
            ),
            "group": "safemppi_prominent_modes",
            "policy": "SafeMPPI_0806_source",
            "gamma": selection["gamma"],
            "mode": selection["mode"],
            "seed": selection["seed"],
            "simulated_status": "SUCCESS",
            "hardware_eligibility": "REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL",
            "source_archive": source.relative_to(bundle).as_posix(),
            "source_archive_sha256": sha256_file(source),
            "flight_reference": target.relative_to(bundle).as_posix(),
            "flight_reference_sha256": sha256_file(target),
            **diagnostics,
        })

    manifest = {
        "schema": "paper_ready_0808_safemppi_100hz_references_v1",
        "status": "COMPLETE",
        "count": len(rows),
        "source_git_sha": "9cafc00551e4964b9dbe559b1a4ba95104e9c88a",
        "contract": {
            "planner_called_by_exporter": False,
            "governor_application_count": 1,
            "player_must_not_reapply_governor": True,
            "player_must_not_interpolate_or_smooth": True,
        },
        "runs": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    columns = [
        "flight_id", "group", "policy", "gamma", "mode", "seed",
        "simulated_status", "hardware_eligibility", "flight_reference",
        "flight_reference_sha256", "source_archive", "source_archive_sha256",
    ]
    with (output / "FLIGHT_INDEX.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "references": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
