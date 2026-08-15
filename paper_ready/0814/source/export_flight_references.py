#!/usr/bin/env python3
"""Export the frozen 0814 paper-ready trajectories as exact 100 Hz references.

The exporter never calls a policy, CFM-MPPI, SafeMPPI, or ReferenceGovernor.
It verifies the once-governed recurrence stored in the handoff and writes the
position, velocity, and acceleration arrays consumed by the flight player.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PAPER_GROUPS = (
    "paper-ready-pre2",
    "paper-ready-less-expanded",
    "paper-ready-expanded",
    "paper-ready-cfmmppi",
    "paper-ready-safemppi",
)
EXPECTED_COUNTS = {
    "paper-ready-pre2": 8,
    "paper-ready-less-expanded": 8,
    "paper-ready-expanded": 8,
    "paper-ready-cfmmppi": 24,
    "paper-ready-safemppi": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_rows(handoff: dict) -> list[tuple[str, int, dict]]:
    """Return the fixed-gamma paper roster, including all three CFM regimes."""
    selected: list[tuple[str, int, dict]] = []
    for group in PAPER_GROUPS:
        rows = handoff["groups"][group]
        if group == "paper-ready-safemppi":
            rows = [row for row in rows if np.isclose(float(row["gamma"]), 0.1)]
        if len(rows) != EXPECTED_COUNTS[group]:
            raise ValueError(
                f"{group} expected {EXPECTED_COUNTS[group]} rows, got {len(rows)}"
            )
        if {float(row["gamma"]) for row in rows} != {0.1}:
            raise ValueError(f"{group} is not fixed to gamma 0.1")
        selected.extend((group, index, row) for index, row in enumerate(rows))
    if len(selected) != 56:
        raise ValueError(f"expected 56 frozen paper trajectories, got {len(selected)}")
    return selected


def reconstruct_reference(
    row: dict,
    *,
    dt: float,
    substeps: int,
    max_speed: float,
    max_vertical_speed: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    states = np.asarray(row["states"], np.float32)
    raw = np.asarray(row["controls"], np.float32)
    applied = np.asarray(row["applied_controls"], np.float32)
    if "dense_positions" in row:
        stored_dense = np.asarray(row["dense_positions"], np.float32)
    else:
        dense_steps = np.asarray(row["dense_steps"], np.float32)
        stored_dense = np.concatenate(
            [states[:1, :3], dense_steps.reshape(-1, 3)], axis=0
        ).astype(np.float32)

    if raw.shape != applied.shape or applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError("raw/applied control shape mismatch")
    if states.shape != (len(applied) + 1, 6):
        raise ValueError(f"state shape mismatch: {states.shape}")
    if stored_dense.shape != (len(applied) * substeps + 1, 3):
        raise ValueError(f"dense-position shape mismatch: {stored_dense.shape}")

    dt_sub = np.float32(float(dt) / int(substeps))
    position = states[0, :3].copy()
    velocity = states[0, 3:6].copy()
    positions = [position.copy()]
    velocities = [velocity.copy()]
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
    velocity_ref = np.asarray(velocities, np.float32)
    dense_error = float(
        np.max(np.abs(reconstructed_positions - stored_dense), initial=0.0)
    )
    knot_position_error = float(
        np.max(
            np.abs(reconstructed_positions[::substeps] - states[:, :3]),
            initial=0.0,
        )
    )
    knot_velocity_error = float(
        np.max(
            np.abs(velocity_ref[::substeps] - states[:, 3:6]),
            initial=0.0,
        )
    )
    if max(dense_error, knot_position_error, knot_velocity_error) > 2.0e-6:
        raise ValueError(
            "once-governed recurrence mismatch: "
            f"dense={dense_error:.3e}, knot_p={knot_position_error:.3e}, "
            f"knot_v={knot_velocity_error:.3e}"
        )

    acceleration_ref = np.concatenate(
        [np.repeat(applied, substeps, axis=0), applied[-1:]], axis=0
    ).astype(np.float32)
    time_s = (
        np.arange(len(stored_dense), dtype=np.float64)
        * (float(dt) / int(substeps))
    )
    payload = {
        "time_s": time_s,
        "position_ref": stored_dense,
        "velocity_ref": velocity_ref,
        "acceleration_ref": acceleration_ref,
        "raw_controls_10hz": raw,
        "applied_controls_10hz": applied,
        # Compatibility alias used by the 0806 player/handoff.
        "executed_controls_10hz": applied,
        "status": np.str_(str(row["status"])),
        "gamma": np.float32(row["gamma"]),
        "seed": np.int64(row["rollout_seed"]),
        "episode": np.int64(row.get("episode", row.get("trial", -1))),
        "reference_rate_hz": np.float32(1.0 / (float(dt) / int(substeps))),
        "source_control_rate_hz": np.float32(1.0 / float(dt)),
    }
    diagnostics: dict[str, float | int] = {
        "samples": len(stored_dense),
        "duration_s": float(time_s[-1]),
        "max_dense_position_reconstruction_error_m": dense_error,
        "max_knot_position_reconstruction_error_m": knot_position_error,
        "max_knot_velocity_reconstruction_error_mps": knot_velocity_error,
        "max_speed_mps": float(np.linalg.norm(velocity_ref, axis=1).max()),
        "max_vertical_speed_mps": float(np.abs(velocity_ref[:, 2]).max()),
        "max_applied_acceleration_component_mps2": float(np.abs(applied).max()),
    }
    return payload, diagnostics


def _method(group: str) -> str:
    return {
        "paper-ready-pre2": "PRE2",
        "paper-ready-less-expanded": "Expanded_R1",
        "paper-ready-expanded": "Expanded_S4",
        "paper-ready-cfmmppi": "CFM-MPPI",
        "paper-ready-safemppi": "SafeMPPI",
    }[group]


def _slug(group: str) -> str:
    return group.removeprefix("paper-ready-").replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = (args.output or bundle / "flight_references").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    handoff_path = bundle / "trajectories/paper_ready_bowling_handoff.pt"
    handoff = torch.load(handoff_path, map_location="cpu", weights_only=False)
    bundle_manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    config_path = bundle / "config/task_config_resolved.json"
    config = json.loads(config_path.read_text())
    mppi = config["safemppi"]
    source_sha = sha256_file(handoff_path)
    config_sha = sha256_file(config_path)
    rows: list[dict[str, object]] = []

    for group, source_row, row in paper_rows(handoff):
        method = _method(group)
        regime = str(row.get("regime") or "")
        route = str((row.get("bowling_route") or {}).get("stable_code") or "")
        payload, diagnostics = reconstruct_reference(
            row,
            dt=float(mppi["dt"]),
            substeps=int(mppi["integration_substeps"]),
            max_speed=float(mppi["max_speed"]),
            max_vertical_speed=float(mppi["max_vertical_speed"]),
        )
        payload.update({
            "view": np.str_(group),
            "method": np.str_(method),
            "regime": np.str_(regime),
            "route": np.str_(route),
            "source_handoff_sha256": np.str_(source_sha),
            "config_sha256": np.str_(config_sha),
        })
        model_key = {
            "paper-ready-pre2": "pre2",
            "paper-ready-less-expanded": "less_expanded_r1",
            "paper-ready-expanded": "expanded_s4_r4",
            "paper-ready-cfmmppi": "pre2",
        }.get(group)
        checkpoint_sha = (
            bundle_manifest["models"][model_key]["sha256"]
            if model_key is not None else "NOT_APPLICABLE_SAFEMPPI"
        )
        payload["checkpoint_sha256"] = np.str_(checkpoint_sha)

        directory = output / _slug(group)
        if regime:
            directory = directory / regime
        directory.mkdir(parents=True, exist_ok=True)
        episode = int(payload["episode"])
        seed = int(payload["seed"])
        target = directory / f"gamma_0p1_e{episode}_seed_{seed}_100hz.npz"
        np.savez_compressed(target, **payload)
        status = str(payload["status"])
        try:
            reference_path = target.relative_to(bundle).as_posix()
        except ValueError:
            reference_path = target.relative_to(output).as_posix()
        rows.append({
            "flight_id": f"0814_{_slug(group)}_{regime or 'default'}_e{episode}_s{seed}",
            "view": group,
            "method": method,
            "regime": regime,
            "gamma": 0.1,
            "route": route,
            "episode": episode,
            "seed": seed,
            "simulated_status": status,
            "hardware_eligibility": (
                "REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL"
                if status == "SUCCESS"
                else f"SIMULATION_ONLY_KNOWN_{status}"
            ),
            "source_archive": handoff_path.relative_to(bundle).as_posix(),
            "source_archive_sha256": source_sha,
            "source_group": group,
            "source_row": source_row,
            "flight_reference": reference_path,
            "flight_reference_sha256": sha256_file(target),
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            **diagnostics,
        })

    manifest = {
        "schema": "paper_ready_0814_frozen_100hz_references_v1",
        "status": "COMPLETE",
        "count": len(rows),
        "visible_trajectory_count_per_cfm_regime": 40,
        "frozen_reference_count_all_cfm_regimes": 56,
        "contract": {
            "policy_planner_or_governor_called_by_exporter": False,
            "governor_application_count": 1,
            "player_must_not_reapply_governor": True,
            "player_must_not_interpolate_smooth_or_differentiate_position": True,
            "position_semantics": "stored dense trajectory, unchanged",
            "velocity_semantics": "exact governed recurrence at every 100 Hz knot",
            "acceleration_semantics": (
                "stored applied control held over each 10 Hz interval"
            ),
            "known_non_success_references_are_simulation_only": True,
        },
        "runs": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    columns = (
        "flight_id", "view", "method", "regime", "gamma", "route", "episode",
        "seed", "simulated_status", "hardware_eligibility", "flight_reference",
        "flight_reference_sha256", "source_archive", "source_archive_sha256",
        "source_group", "source_row", "checkpoint_sha256", "config_sha256",
    )
    with (output / "FLIGHT_INDEX.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "references": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
