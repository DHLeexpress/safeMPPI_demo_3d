#!/usr/bin/env python3
"""Validate the frozen gamma-1 octants and gamma-0.1 side-above references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CHECKPOINT_SHA = "c1a3c77fc956c57d02a0970c4e54fca942cee391a68275a58134361e00828056"
CONFIG_SHA = "7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def metrics(path: Path, goal: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        if str(scalar(data, "status")) != "SUCCESS":
            raise ValueError(f"non-success trajectory: {path}")
        if str(scalar(data, "checkpoint_sha256")) != CHECKPOINT_SHA:
            raise ValueError(f"checkpoint SHA mismatch: {path}")
        if str(scalar(data, "config_sha256")) != CONFIG_SHA:
            raise ValueError(f"config SHA mismatch: {path}")
        states = np.asarray(data["states"][:, :3], dtype=float)
        dense = np.concatenate([
            states[:1], np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3)
        ])
        state_progress = np.diff(-np.linalg.norm(states - goal[None], axis=1))
        dense_progress = np.diff(-np.linalg.norm(dense - goal[None], axis=1))
        if not np.all(state_progress > 0.0):
            raise ValueError(f"nonpositive state progress: {path}")
        if not np.all(dense_progress > 0.0):
            raise ValueError(f"nonpositive dense progress: {path}")
        return {
            "file": path.name,
            "sha256": sha256_file(path),
            "gamma": float(scalar(data, "gamma")),
            "seed": int(scalar(data, "seed")),
            "mode": str(scalar(data, "mode")),
            "theta_deg": float(scalar(data, "theta_deg")),
            "sector_8": int(scalar(data, "sector_8")),
            "min_clearance_m": float(scalar(data, "min_clearance_m")),
            "time_to_goal_s": float(scalar(data, "time_to_goal_s")),
            "minimum_goal_progress_m": float(state_progress.min()),
            "minimum_dense_goal_progress_m": float(dense_progress.min()),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config = json.loads((bundle / "config" / "task_config_resolved.json").read_text())
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)

    existing_root = (
        args.existing.resolve()
        if args.existing is not None
        else bundle / "trajectories" / "expanded_quality_v2"
    )
    supplement_root = (
        args.supplement.resolve()
        if args.supplement is not None
        else bundle / "trajectories" / "expanded_supplement_v1"
    )
    existing = sorted(existing_root.glob("*.npz"))
    supplement = sorted(supplement_root.glob("*.npz"))
    if len(existing) != 16 or len(supplement) != 6:
        raise ValueError("expected 16 existing and 6 supplemental expanded trajectories")
    existing_rows = [metrics(path, goal) for path in existing]
    supplement_rows = [metrics(path, goal) for path in supplement]

    gamma1 = [
        row for row in existing_rows + supplement_rows
        if np.isclose(row["gamma"], 1.0, atol=1.0e-7, rtol=0.0)
    ]
    sectors = [int(row["sector_8"]) for row in gamma1]
    if len(gamma1) != 8 or sorted(sectors) != list(range(8)):
        raise ValueError(f"gamma=1 does not cover all eight sectors exactly: {sectors}")

    new_gamma01 = [
        row for row in supplement_rows
        if np.isclose(row["gamma"], 0.1, atol=1.0e-7, rtol=0.0)
    ]
    if len(new_gamma01) != 2 or {row["mode"] for row in new_gamma01} != {"above"}:
        raise ValueError("expected two new gamma=0.1 above trajectories")
    side_angles = sorted(float(row["theta_deg"]) for row in new_gamma01)
    if not (45.0 <= side_angles[0] < 90.0 < side_angles[1] < 135.0):
        raise ValueError(f"gamma=0.1 side-above angles are invalid: {side_angles}")

    summary = {
        "schema": "paper_ready_0808_expanded_angular_supplement_v1",
        "status": "PASS",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "config_sha256": CONFIG_SHA,
        "all_success": True,
        "all_strictly_monotone_goal_progress": True,
        "all_strictly_monotone_dense_goal_progress": True,
        "gamma1_sector_count": 8,
        "gamma1_sectors": sorted(sectors),
        "gamma1_trajectories": sorted(gamma1, key=lambda row: int(row["sector_8"])),
        "gamma0p1_side_above_trajectories": sorted(
            new_gamma01, key=lambda row: float(row["theta_deg"])
        ),
        "supplement_trajectories": supplement_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "gamma1_sectors": summary["gamma1_sectors"],
        "gamma0p1_side_angles": side_angles,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
