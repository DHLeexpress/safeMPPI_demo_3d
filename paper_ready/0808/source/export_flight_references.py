#!/usr/bin/env python3
"""Convert the frozen 0808 rollout archives into 100 Hz flight references.

This exporter never calls SafeMPPI, a flow policy, or ReferenceGovernor.  Each
source rollout already contains the once-governed control and dense state
history.  The exporter verifies that recurrence and packages the exact
position, velocity, and acceleration arrays required by the flight player.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


GROUPS = (
    ("expanded_quality_v2", "expanded_reserve_G_nfe12"),
    ("expanded_supplement_v1", "expanded_reserve_G_nfe12"),
    ("expanded_string_safe_v1", "expanded_reserve_G_nfe12"),
    ("expanded_mirrored_above_v1", "expanded_reserve_G_nfe12"),
    ("pretrained_success", "pretrained_p0806_nfe16"),
    ("pretrained_gamma1_biased_left_v1", "pretrained_p0806_nfe16"),
    ("pretrained_collisions", "pretrained_p0806_nfe16"),
)

EXPECTED_COUNTS = {
    "expanded_quality_v2": 16,
    "expanded_supplement_v1": 6,
    "expanded_string_safe_v1": 1,
    "expanded_mirrored_above_v1": 1,
    "pretrained_success": 4,
    "pretrained_gamma1_biased_left_v1": 8,
    "pretrained_collisions": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_reference(
    source: Path,
    *,
    dt: float,
    substeps: int,
    max_speed: float,
    max_vertical_speed: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    with np.load(source, allow_pickle=False) as archive:
        states = np.asarray(archive["states"], np.float32)
        raw = np.asarray(archive["controls"], np.float32)
        applied = np.asarray(archive["applied_controls"], np.float32)
        dense_steps = np.asarray(archive["dense_steps"], np.float32)
        status = str(archive["status"].item())
        gamma = float(archive["gamma"].item())
        seed = int(archive["seed"].item())
        mode = (
            str(archive["mode"].item())
            if "mode" in archive.files
            else "collision"
        )
        checkpoint_sha256 = str(archive["checkpoint_sha256"].item())
        config_sha256 = str(archive["config_sha256"].item())

    if raw.shape != applied.shape or applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError(f"control shape mismatch in {source}")
    if states.shape != (len(applied) + 1, 6):
        raise ValueError(f"state shape mismatch in {source}: {states.shape}")
    if dense_steps.shape != (len(applied), substeps, 3):
        raise ValueError(f"dense-step shape mismatch in {source}: {dense_steps.shape}")

    dt_sub = np.float32(float(dt) / int(substeps))
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

    position_ref = np.asarray(positions, np.float32)
    velocity_ref = np.asarray(velocities, np.float32)
    stored_dense = np.concatenate(
        [states[:1, :3], dense_steps.reshape(-1, 3)], axis=0
    ).astype(np.float32)
    position_error = float(np.max(np.abs(position_ref - stored_dense), initial=0.0))
    knot_position_error = float(
        np.max(np.abs(position_ref[::substeps] - states[:, :3]), initial=0.0)
    )
    knot_velocity_error = float(
        np.max(np.abs(velocity_ref[::substeps] - states[:, 3:6]), initial=0.0)
    )
    if max(position_error, knot_position_error, knot_velocity_error) > 2.0e-6:
        raise ValueError(
            f"governed recurrence mismatch in {source}: dense={position_error:.3e}, "
            f"knot_p={knot_position_error:.3e}, knot_v={knot_velocity_error:.3e}"
        )

    acceleration_ref = np.repeat(applied, substeps, axis=0)
    acceleration_ref = np.concatenate(
        [acceleration_ref, applied[-1:].copy()], axis=0
    ).astype(np.float32)
    time_s = np.arange(len(position_ref), dtype=np.float64) * (float(dt) / substeps)
    payload = {
        "time_s": time_s,
        "position_ref": stored_dense,
        "velocity_ref": velocity_ref,
        "acceleration_ref": acceleration_ref,
        "raw_controls_10hz": raw,
        "applied_controls_10hz": applied,
        "status": np.str_(status),
        "gamma": np.float32(gamma),
        "seed": np.int64(seed),
        "mode": np.str_(mode),
        "checkpoint_sha256": np.str_(checkpoint_sha256),
        "config_sha256": np.str_(config_sha256),
        "reference_rate_hz": np.float32(1.0 / (float(dt) / substeps)),
        "source_control_rate_hz": np.float32(1.0 / float(dt)),
    }
    diagnostics: dict[str, float | int | str] = {
        "status": status,
        "gamma": gamma,
        "seed": seed,
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "samples": len(position_ref),
        "duration_s": float(time_s[-1]),
        "max_dense_position_reconstruction_error_m": position_error,
        "max_knot_position_reconstruction_error_m": knot_position_error,
        "max_knot_velocity_reconstruction_error_mps": knot_velocity_error,
        "max_speed_mps": float(np.linalg.norm(velocity_ref, axis=1).max()),
        "max_vertical_speed_mps": float(np.abs(velocity_ref[:, 2]).max()),
        "max_applied_acceleration_component_mps2": float(np.abs(applied).max()),
    }
    return payload, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--trajectories",
        type=Path,
        help="trajectory root; defaults to BUNDLE/trajectories",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-prefix",
        type=Path,
        help="manifest-relative prefix when OUTPUT is staged outside BUNDLE",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    trajectory_root = (
        args.trajectories.resolve()
        if args.trajectories is not None
        else bundle / "trajectories"
    )
    output = (args.output or (bundle / "flight_references")).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    config_path = bundle / "config" / "task_config_resolved.json"
    config = json.loads(config_path.read_text())
    mppi = config["safemppi"]
    expected_config_sha = sha256_file(config_path)
    rows: list[dict[str, object]] = []

    for group, policy in GROUPS:
        group_output = output / group
        group_output.mkdir(parents=True, exist_ok=True)
        sources = sorted((trajectory_root / group).glob("*.npz"))
        expected_count = EXPECTED_COUNTS[group]
        if len(sources) != expected_count:
            raise ValueError(f"expected {expected_count} {group} sources, got {len(sources)}")
        for source in sources:
            payload, diagnostics = reconstruct_reference(
                source,
                dt=float(mppi["dt"]),
                substeps=int(mppi["integration_substeps"]),
                max_speed=float(mppi["max_speed"]),
                max_vertical_speed=float(mppi["max_vertical_speed"]),
            )
            if diagnostics["config_sha256"] != expected_config_sha:
                raise ValueError(f"config SHA embedded in {source} does not match bundle")
            target = group_output / f"{source.stem}_100hz.npz"
            np.savez_compressed(target, **payload)
            known_collision = diagnostics["status"] == "COLLISION"
            try:
                source_archive = source.relative_to(bundle).as_posix()
            except ValueError:
                source_archive = source.resolve().as_posix()
            try:
                flight_reference = target.relative_to(bundle).as_posix()
            except ValueError:
                relative_target = target.relative_to(output)
                flight_reference = (
                    (args.reference_prefix / relative_target).as_posix()
                    if args.reference_prefix is not None
                    else relative_target.as_posix()
                )
            rows.append({
                "flight_id": f"0808_{group}_{source.stem}",
                "group": group,
                "policy": policy,
                "gamma": diagnostics["gamma"],
                "mode": diagnostics["mode"],
                "seed": diagnostics["seed"],
                "simulated_status": diagnostics["status"],
                "hardware_eligibility": (
                    "SIMULATION_ONLY_KNOWN_COLLISION"
                    if known_collision else "REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL"
                ),
                "source_archive": source_archive,
                "source_archive_sha256": sha256_file(source),
                "flight_reference": flight_reference,
                "flight_reference_sha256": sha256_file(target),
                **diagnostics,
            })

    if len(rows) != 38:
        raise ValueError(f"expected 38 flight references, got {len(rows)}")
    manifest = {
        "schema": "paper_ready_0808_frozen_100hz_references_v4",
        "status": "COMPLETE",
        "count": len(rows),
        "contract": {
            "policy_or_planner_called_by_exporter": False,
            "governor_application_count": 1,
            "player_must_not_reapply_governor": True,
            "player_must_not_interpolate_or_smooth": True,
            "expanded_checkpoint_is_generation_provenance_not_a_flight_dependency": True,
            "position_semantics": "stored dense_steps, unchanged",
            "velocity_semantics": "exact governed recurrence at each 100 Hz knot",
            "acceleration_semantics": "stored applied control held over each 10 Hz interval",
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
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "references": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
