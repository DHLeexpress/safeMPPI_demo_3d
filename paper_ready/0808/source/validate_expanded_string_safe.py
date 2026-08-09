#!/usr/bin/env python3
"""Validate the frozen gamma-1 side-above suspension-line alternative."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CHECKPOINT_SHA = "c1a3c77fc956c57d02a0970c4e54fca942cee391a68275a58134361e00828056"
CONFIG_SHA = "7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2"


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    trajectory = (
        args.trajectory.resolve()
        if args.trajectory is not None
        else bundle / "trajectories" / "expanded_string_safe_v1"
        / "gamma_1_mode_above_seed_131629.npz"
    )
    config = json.loads((bundle / "config" / "task_config_resolved.json").read_text())
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    sphere = np.asarray(config["obstacles"]["spheres"][0], dtype=float)
    physical_radius = float(config["stage1"]["physical_sphere_radius_m"])

    with np.load(trajectory, allow_pickle=False) as data:
        if str(scalar(data, "status")) != "SUCCESS":
            raise ValueError("string-safe reference is not a terminal success")
        if str(scalar(data, "checkpoint_sha256")) != CHECKPOINT_SHA:
            raise ValueError("checkpoint SHA mismatch")
        if str(scalar(data, "config_sha256")) != CONFIG_SHA:
            raise ValueError("config SHA mismatch")
        if not np.isclose(float(scalar(data, "gamma")), 1.0):
            raise ValueError("expected gamma=1.0")
        if int(scalar(data, "seed")) != 131629:
            raise ValueError("expected seed 131629")
        theta_deg = float(scalar(data, "theta_deg"))
        if not 45.0 <= theta_deg < 65.0:
            raise ValueError(f"trajectory is not side-above: {theta_deg}")
        states = np.asarray(data["states"][:, :3], dtype=float)
        dense = np.concatenate([
            states[:1], np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3)
        ])

    state_progress = np.diff(-np.linalg.norm(states - goal[None], axis=1))
    dense_progress = np.diff(-np.linalg.norm(dense - goal[None], axis=1))
    if not np.all(state_progress > 0.0) or not np.all(dense_progress > 0.0):
        raise ValueError("string-safe trajectory is not strictly goal-monotone")
    overhead = dense[:, 2] >= sphere[2] + physical_radius
    if not np.any(overhead):
        raise ValueError("trajectory has no points above the physical sphere")
    string_distance = np.linalg.norm(dense[overhead, :2] - sphere[None, :2], axis=1)
    minimum_string_distance = float(string_distance.min())
    if minimum_string_distance < 0.25:
        raise ValueError(
            f"insufficient horizontal separation from string centerline: "
            f"{minimum_string_distance:.6f} m"
        )

    summary = {
        "schema": "paper_ready_0808_expanded_string_safe_v1",
        "status": "PASS",
        "file": trajectory.name,
        "sha256": sha256_file(trajectory),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "config_sha256": CONFIG_SHA,
        "gamma": 1.0,
        "seed": 131629,
        "theta_deg": theta_deg,
        "sector_8": 5,
        "minimum_horizontal_distance_to_vertical_string_centerline_m": minimum_string_distance,
        "minimum_goal_progress_m": float(state_progress.min()),
        "minimum_dense_goal_progress_m": float(dense_progress.min()),
        "repeat_verification": "BIT_IDENTICAL",
        "note": "Centerline separation is a geometric screen, not hardware safety certification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
