#!/usr/bin/env python3
"""Validate and summarize the proposed smooth expanded-policy selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GAMMAS = (0.1, 0.3, 0.5, 1.0)
MODES = ("below", "above", "left", "right")


def turn_angles_deg(displacements: np.ndarray) -> np.ndarray:
    first = displacements[:-1]
    second = displacements[1:]
    denom = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid = denom > 1.0e-10
    cosine = np.sum(first[valid] * second[valid], axis=1) / denom[valid]
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    records = []
    seen = set()
    for path in sorted(args.trajectories.glob("*.npz")):
        data = np.load(path)
        gamma_raw = float(np.asarray(data["gamma"]).item())
        gamma = min(GAMMAS, key=lambda value: abs(value - gamma_raw))
        mode = str(np.asarray(data["mode"]).item())
        key = (gamma, mode)
        if key in seen:
            raise SystemExit(f"duplicate stratum {key}")
        seen.add(key)
        if str(np.asarray(data["status"]).item()) != "SUCCESS":
            raise SystemExit(f"non-success {path}")

        states = np.asarray(data["states"][:, :3], dtype=float)
        dense = np.concatenate([
            states[:1],
            np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3),
        ])
        distance = np.linalg.norm(states - goal[None, :], axis=1)
        progress = distance[:-1] - distance[1:]
        dense_distance = np.linalg.norm(dense - goal[None, :], axis=1)
        dense_progress = dense_distance[:-1] - dense_distance[1:]
        displacements = np.diff(states, axis=0)
        turns = turn_angles_deg(displacements)
        tail_steps = min(10, len(displacements))
        tail = displacements[-tail_steps:]
        tail_length = float(np.linalg.norm(tail, axis=1).sum())
        tail_displacement = float(np.linalg.norm(states[-1] - states[-tail_steps - 1]))
        record = {
            "file": path.name,
            "gamma": gamma,
            "mode": mode,
            "seed": int(np.asarray(data["seed"]).item()),
            "status": "SUCCESS",
            "minimum_goal_progress_m": float(progress.min()),
            "monotone_goal_progress": bool(np.all(progress > 0.0)),
            "minimum_dense_goal_progress_m": float(dense_progress.min()),
            "monotone_dense_goal_progress": bool(np.all(dense_progress > 0.0)),
            "p95_turn_angle_deg": float(np.percentile(turns, 95.0)),
            "max_turn_angle_deg": float(turns.max()),
            "terminal_path_efficiency": (
                tail_displacement / tail_length if tail_length > 0.0 else 1.0
            ),
            "min_clearance_m": float(np.asarray(data["min_clearance_m"]).item()),
            "time_to_goal_s": float(np.asarray(data["time_to_goal_s"]).item()),
        }
        if not record["monotone_goal_progress"]:
            raise SystemExit(f"non-monotone goal progress: {path}")
        if not record["monotone_dense_goal_progress"]:
            raise SystemExit(f"non-monotone dense goal progress: {path}")
        records.append(record)

    expected = {(gamma, mode) for gamma in GAMMAS for mode in MODES}
    if seen != expected:
        raise SystemExit(f"missing strata: {sorted(expected - seen)}")

    per_gamma = []
    for gamma in GAMMAS:
        rows = [row for row in records if row["gamma"] == gamma]
        per_gamma.append({
            "gamma": gamma,
            "mean_min_clearance_m": float(np.mean([
                row["min_clearance_m"] for row in rows
            ])),
            "mean_time_to_goal_s": float(np.mean([
                row["time_to_goal_s"] for row in rows
            ])),
            "maximum_p95_turn_angle_deg": float(max(
                row["p95_turn_angle_deg"] for row in rows
            )),
            "minimum_terminal_path_efficiency": float(min(
                row["terminal_path_efficiency"] for row in rows
            )),
        })
    clearances = [row["mean_min_clearance_m"] for row in per_gamma]
    if not all(first > second for first, second in zip(clearances, clearances[1:])):
        raise SystemExit(f"clearance trend is not strictly decreasing: {clearances}")

    summary = {
        "schema": "paper_ready_expanded_quality_selection_summary_v1",
        "status": "PASS",
        "trajectory_count": len(records),
        "all_success": True,
        "all_strictly_monotone_goal_progress": True,
        "all_strictly_monotone_dense_goal_progress": True,
        "minimum_progress_over_all_trajectories_m": float(min(
            row["minimum_goal_progress_m"] for row in records
        )),
        "minimum_dense_progress_over_all_trajectories_m": float(min(
            row["minimum_dense_goal_progress_m"] for row in records
        )),
        "clearance_mean_strictly_decreases_with_gamma": True,
        "gamma_0p5_vs_1p0_mean_time_difference_s": abs(
            per_gamma[2]["mean_time_to_goal_s"]
            - per_gamma[3]["mean_time_to_goal_s"]
        ),
        "per_gamma": per_gamma,
        "trajectories": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
