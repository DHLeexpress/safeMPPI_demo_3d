#!/usr/bin/env python3
"""Reproduce the fixed-scene gamma=0.3 SafeMPPI reference bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parent
RUNTIME = BUNDLE.parent / "runtime_snapshot"
sys.path.insert(0, str(RUNTIME))

from safe_mppi.acquire import run_episode  # noqa: E402
from safe_mppi.config import ExperimentConfig, ObstacleConfig, load_config  # noqa: E402
from safe_mppi.controller import Mode1SafeMPPI  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402


GAMMA = 0.3
SCENE_NAME = "axis3_outer_pair"
CYLINDERS = np.asarray([
    [-1.4279999732971191, 0.7799999713897705, 0.20000000298023224],
    [-0.3490934669971466, 0.3275127708911896, 0.20000000298023224],
    [-1.0509065389633179, -0.3275127708911896, 0.20000000298023224],
    [0.02800000086426735, -0.7799999713897705, 0.20000000298023224],
], np.float32)
ROSTER = (
    ("ALL_LEFT", "LLLL", 819585),
    ("ALL_LEFT", "LLLL", 819574),
    ("ALL_LEFT", "LLLL", 819558),
    ("ALL_RIGHT", "RRRR", 819510),
    ("ALL_RIGHT", "RRRR", 819552),
    ("ALL_RIGHT", "RRRR", 819520),
    ("MIDDLE", "LRLR", 819546),
    ("MIDDLE", "RRLL", 819507),
)
EXPECTED_SOURCE_SHA256 = {
    "safe_mppi/controller.py": "dfc91a26ccac2818c902215bf4d9a06e405d5878e5c6af0be2f75c4f68106dad",
    "safe_mppi/acquire.py": "954b673c24897cbb6ab2b8255fbbca5532e1a679593767d4e35d57916cf6c050",
    "config/lab_clutter_cylinders_path_midpoint_uniform_mirrored_z01_17_v1.json": "af2e0c97f2c4079a906dc3588e8fd04c8093cb5214e631a7794c00ae54bcb881",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_hash(start: np.ndarray, goal: np.ndarray) -> str:
    payload = json.dumps({
        "name": SCENE_NAME,
        "start": np.asarray(start[:3], float).tolist(),
        "goal": np.asarray(goal, float).tolist(),
        "cylinders": CYLINDERS.astype(float).tolist(),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def signature(points: np.ndarray, start: np.ndarray, goal: np.ndarray) -> str:
    axis = goal[:2] - start[:2]
    length = float(np.linalg.norm(axis))
    along = axis / length
    left = np.asarray([-along[1], along[0]])
    path_rel = np.asarray(points)[:, :2] - start[:2]
    path_fraction = path_rel @ along / length
    path_lateral = path_rel @ left
    cylinder_rel = CYLINDERS[:, :2] - start[:2]
    cylinder_fraction = cylinder_rel @ along / length
    cylinder_lateral = cylinder_rel @ left
    labels = []
    for index in np.argsort(cylinder_fraction, kind="stable"):
        near = int(np.argmin(np.abs(path_fraction - cylinder_fraction[index])))
        labels.append("L" if path_lateral[near] >= cylinder_lateral[index] else "R")
    return "".join(labels)


def geometry(points: np.ndarray) -> dict[str, float]:
    segments = np.diff(np.asarray(points, float), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    directions = segments[lengths > 1.0e-8] / lengths[lengths > 1.0e-8, None]
    turns = (
        np.arccos(np.clip(np.sum(directions[1:] * directions[:-1], axis=1), -1.0, 1.0))
        if len(directions) > 1 else np.zeros(0)
    )
    return {
        "path_length_m": float(lengths.sum()),
        "total_turn_radians": float(turns.sum()),
        "mean_turn_radians": float(turns.mean()) if len(turns) else 0.0,
    }


def reconstruct_reference(states, raw, applied, dense, *, dt, substeps, max_speed, max_vertical_speed):
    dt_sub = np.float32(dt / substeps)
    position = np.asarray(states[0, :3], np.float32).copy()
    velocity = np.asarray(states[0, 3:6], np.float32).copy()
    positions = [position.copy()]
    velocities = [velocity.copy()]
    for acceleration in applied:
        for _ in range(substeps):
            velocity = velocity + dt_sub * acceleration
            speed = float(np.linalg.norm(velocity))
            if speed > max_speed:
                velocity = velocity * np.float32(max_speed / speed)
            velocity[2] = np.clip(velocity[2], -max_vertical_speed, max_vertical_speed)
            position = position + dt_sub * velocity
            positions.append(position.astype(np.float32).copy())
            velocities.append(velocity.astype(np.float32).copy())
    positions = np.asarray(positions, np.float32)
    velocities = np.asarray(velocities, np.float32)
    dense_error = float(np.max(np.abs(positions - dense), initial=0.0))
    knot_p_error = float(np.max(np.abs(positions[::substeps] - states[:, :3]), initial=0.0))
    knot_v_error = float(np.max(np.abs(velocities[::substeps] - states[:, 3:6]), initial=0.0))
    if max(dense_error, knot_p_error, knot_v_error) > 2.0e-6:
        raise RuntimeError("reference-governor recurrence mismatch")
    acceleration = np.concatenate([np.repeat(applied, substeps, axis=0), applied[-1:]], axis=0).astype(np.float32)
    time_s = np.arange(len(dense), dtype=np.float64) * (dt / substeps)
    return {
        "time_s": time_s,
        "position_ref": dense.astype(np.float32),
        "velocity_ref": velocities,
        "acceleration_ref": acceleration,
        "raw_controls_10hz": raw.astype(np.float32),
        "applied_controls_10hz": applied.astype(np.float32),
        "executed_controls_10hz": applied.astype(np.float32),
        "reference_rate_hz": np.float32(substeps / dt),
        "source_control_rate_hz": np.float32(1.0 / dt),
    }, {
        "max_dense_position_reconstruction_error_m": dense_error,
        "max_knot_position_reconstruction_error_m": knot_p_error,
        "max_knot_velocity_reconstruction_error_mps": knot_v_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw_rollouts"
    flight_dir = output / "flight_references"
    raw_dir.mkdir()
    flight_dir.mkdir()

    config_path = BUNDLE / "config/lab_clutter_cylinders_path_midpoint_uniform_mirrored_z01_17_v1.json"
    source_paths = {
        "safe_mppi/controller.py": RUNTIME / "safe_mppi/controller.py",
        "safe_mppi/acquire.py": RUNTIME / "safe_mppi/acquire.py",
        "config/lab_clutter_cylinders_path_midpoint_uniform_mirrored_z01_17_v1.json": config_path,
    }
    for key, path in source_paths.items():
        observed = sha256(path)
        if observed != EXPECTED_SOURCE_SHA256[key]:
            raise RuntimeError(f"source hash mismatch for {key}: {observed}")

    base = load_config(config_path)
    fixed = ExperimentConfig(
        base.taskspace,
        ObstacleConfig(spheres=np.zeros((0, 4), np.float32), cylinders=CYLINDERS),
        base.safemppi,
        base.data,
        base.raw,
    )
    env = TaskEnvironment(fixed)
    fixed_scene_hash = scene_hash(env.start, env.goal)
    rows = []
    for index, (family, expected_signature, seed) in enumerate(ROSTER, 1):
        controller = Mode1SafeMPPI(fixed.safemppi, env, device="cpu")
        row, arrays = run_episode(env, controller, GAMMA, seed, fixed.data.rollout_dynamics)
        accepted = bool(
            row["success"]
            and row["minimum_online_one_step_slack"] >= -1.0e-6
            and row["deployment_speed_compatible"]
        )
        observed_signature = signature(arrays["dense_positions"], env.start, env.goal)
        if not accepted or observed_signature != expected_signature:
            raise RuntimeError(
                f"seed {seed} changed: accepted={accepted}, signature={observed_signature}"
            )
        slug = family.lower()
        raw_path = raw_dir / f"gamma0p3_{index:02d}_{slug}_{observed_signature.lower()}_seed{seed}.npz"
        np.savez_compressed(
            raw_path,
            states=arrays["states"],
            controls=arrays["controls"],
            executed_controls=arrays["executed_controls"],
            dense_positions=arrays["dense_positions"],
            controller_one_step_slack=arrays["controller_one_step_slack"],
            online_one_step_slack=arrays["online_one_step_slack"],
            feasible_fraction=arrays["feasible_fraction"],
            all_infeasible=arrays["all_infeasible"],
            cylinders=CYLINDERS,
            gamma=np.float32(GAMMA),
            seed=np.int64(seed),
            route_family=np.str_(family),
            signature=np.str_(observed_signature),
            scene_name=np.str_(SCENE_NAME),
            scene_hash=np.str_(fixed_scene_hash),
        )
        reference, errors = reconstruct_reference(
            arrays["states"], arrays["controls"], arrays["executed_controls"], arrays["dense_positions"],
            dt=float(fixed.safemppi.dt),
            substeps=int(fixed.safemppi.integration_substeps),
            max_speed=float(fixed.safemppi.max_speed),
            max_vertical_speed=float(fixed.safemppi.max_vertical_speed),
        )
        reference.update({
            "gamma": np.float32(GAMMA),
            "seed": np.int64(seed),
            "route_family": np.str_(family),
            "signature": np.str_(observed_signature),
            "scene_name": np.str_(SCENE_NAME),
            "scene_hash": np.str_(fixed_scene_hash),
            "cylinders": CYLINDERS,
            "controller_sha256": np.str_(EXPECTED_SOURCE_SHA256["safe_mppi/controller.py"]),
            "hardware_eligible": np.bool_(False),
        })
        flight_path = flight_dir / f"gamma0p3_{index:02d}_{slug}_{observed_signature.lower()}_seed{seed}_100hz.npz"
        np.savez_compressed(flight_path, **reference)
        rows.append({
            "index": index,
            "route_family": family,
            "signature": observed_signature,
            "gamma": GAMMA,
            "seed": seed,
            "success": True,
            "collision": False,
            "taskspace_violation": False,
            "steps": int(row["steps"]),
            "time_to_goal_s": float(row["time_to_goal_s"]),
            "min_clearance_m": float(row["min_clearance_m"]),
            **geometry(arrays["dense_positions"]),
            **errors,
            "raw_rollout_file": raw_path.relative_to(output).as_posix(),
            "raw_rollout_sha256": sha256(raw_path),
            "flight_reference_file": flight_path.relative_to(output).as_posix(),
            "flight_reference_sha256": sha256(flight_path),
            "reference_samples": int(len(reference["time_s"])),
            "duration_s": float(reference["time_s"][-1]),
        })

    scene = {
        "name": SCENE_NAME,
        "scene_hash": fixed_scene_hash,
        "gamma": GAMMA,
        "start": np.asarray(env.start, float).tolist(),
        "goal": np.asarray(env.goal, float).tolist(),
        "bounds": np.asarray(env.bounds, float).tolist(),
        "cylinders": CYLINDERS.astype(float).tolist(),
        "description": "two axis cylinders at fractions 0.24/0.76 and one symmetric outer pair at fraction 0.50",
    }
    (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    manifest = {
        "schema": "fixed_scene_safemppi_gamma03_v1",
        "kind": "exact SafeMPPI rollouts on one fixed symmetric cylinder episode",
        "gamma": GAMMA,
        "screening": {
            "trials": 96,
            "nominal_safe_successes": 87,
            "all_left": 57,
            "all_right": 27,
            "middle": 3,
            "signature_counts": {"LLLL": 57, "RRRR": 27, "LRLR": 2, "RRLL": 1},
        },
        "frozen_roster": {"all_left": 3, "all_right": 3, "middle": 2, "total": 8},
        "scene": scene,
        "source": {
            "controller": "paper_ready/0814/runtime_snapshot/safe_mppi/controller.py",
            "episode_runner": "paper_ready/0814/runtime_snapshot/safe_mppi/acquire.py::run_episode",
            "config": "config/lab_clutter_cylinders_path_midpoint_uniform_mirrored_z01_17_v1.json",
            "sha256": EXPECTED_SOURCE_SHA256,
            "existing_safemppi_source_git_sha": "dabb5011dfc674864e1de275a1e1c2adab58f4af",
            "device": "cpu",
        },
        "rows": rows,
        "hardware_eligible": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    columns = list(rows[0])
    with (output / "FLIGHT_INDEX.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(output),
        "scene_hash": fixed_scene_hash,
        "rows": len(rows),
        "signatures": [row["signature"] for row in rows],
        "seeds": [row["seed"] for row in rows],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
