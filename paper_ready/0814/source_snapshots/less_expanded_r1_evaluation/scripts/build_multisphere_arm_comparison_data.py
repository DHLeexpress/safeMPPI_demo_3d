#!/usr/bin/env python3
"""Build a compact dense/uniform, speed100/200 trajectory audit payload."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_clutter_expansion import sphere_scene_spec_from_config


ARM_SPECS = {
    "dense200": ("dense-z · speed200", 10, 200),
    "dense100": ("dense-z · speed100", 8, 100),
    "uniform200": ("uniform-z · speed200", 8, 200),
    "uniform100": ("uniform-z · speed100", 5, 100),
}
GAMMAS = (0.1, 0.3, 0.5, 1.0)


def _gamma_key(value: float) -> str:
    return f"{float(value):g}"


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _encode_path(path: np.ndarray, limit: int = 46) -> list[Any]:
    points = np.asarray(path, np.float64).reshape(-1, 3)
    if len(points) > limit:
        indices = np.unique(
            np.rint(np.linspace(0, len(points) - 1, limit)).astype(int)
        )
        points = points[indices]
    quantized = np.rint(points * 1000).astype(np.int32)
    return [
        quantized[0].tolist(),
        np.diff(quantized, axis=0).reshape(-1).tolist(),
    ]


def _axis_distance(points: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    direction = goal - start
    unit = direction / np.linalg.norm(direction)
    offsets = points - start
    return np.linalg.norm(offsets - np.outer(offsets @ unit, unit), axis=1)


def _select_three(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["z_excursion_m"])
    if len(ordered) <= 3:
        return ordered
    indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, 3)).astype(int))
    return [ordered[index] for index in indices]


def _arm_payload(root: Path, arm_id: str) -> dict[str, Any]:
    label, round_index, speed_weight = ARM_SPECS[arm_id]
    arm = root / arm_id
    config = load_config(arm / "task_config_resolved.json")
    env = TaskEnvironment(config)
    scene_spec = sphere_scene_spec_from_config(config)
    metric_rows = _read_metric_rows(arm / "metrics.jsonl")
    metric = next(row for row in metric_rows if int(row["round"]) == round_index)
    committed = {
        (float(gamma), int(episode))
        for gamma, detail in metric["successful_executed_commit_by_gamma"].items()
        for episode in detail["committed_episode_ids"]
    }
    events = torch.load(
        arm / f"events_round_{round_index:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = (float(event["gamma"]), int(event["episode"]))
        if key in committed:
            grouped[key].append(event)
    if set(grouped) != committed:
        raise RuntimeError(f"{arm_id}: committed traces are missing")

    trajectories = []
    for gamma, episode in sorted(committed):
        trace = sorted(grouped[(gamma, episode)], key=lambda row: int(row["step"]))
        if trace[-1].get("status") != "SUCCESS":
            raise RuntimeError(f"{arm_id}: committed trace is not terminal success")
        path = np.vstack([
            np.asarray(trace[0]["robot"], np.float64)[:3],
            *[np.asarray(row["robot_after"], np.float64)[:3] for row in trace],
        ])
        states = np.vstack([
            np.asarray(trace[0]["robot"], np.float64),
            *[np.asarray(row["robot_after"], np.float64) for row in trace],
        ])
        context = np.asarray(trace[0]["context"], np.float32).reshape(-1)
        spheres = np.asarray(
            scene_spec.unpack(env, context[-scene_spec.packed_dim:]), np.float64
        )
        clearance = np.min(
            np.linalg.norm(path[:, None, :] - spheres[None, :, :3], axis=2)
            - spheres[None, :, 3],
            axis=1,
        )
        speeds = np.linalg.norm(states[:, 3:6], axis=1)
        near = clearance < 0.6
        axis_distance = _axis_distance(path, env.start[:3], env.goal[:3])
        trajectory = {
            "id": f"{arm_id}:r{round_index}:g{gamma:g}:e{episode}",
            "arm": arm_id,
            "round": round_index,
            "gamma": gamma,
            "episode": episode,
            "path": _encode_path(path),
            "spheres": np.round(spheres, 4).tolist(),
            "scene_hash": str(trace[0].get("scene_hash", ""))[:12],
            "pair_member": trace[0].get("paired_scene_member_name"),
            "pair_slot": trace[0].get("paired_scene_pair_slot"),
            "steps": len(trace),
            "duration_s": round(len(trace) * float(config.safemppi.dt), 3),
            "path_length_m": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 4),
            "min_clearance_m": round(float(clearance.min()), 4),
            "mean_speed_mps": round(float(speeds.mean()), 4),
            "near_obstacle_speed_mps": round(float(speeds[near].mean()), 4) if near.any() else None,
            "max_speed_mps": round(float(speeds.max()), 4),
            "z_min_m": round(float(path[:, 2].min()), 4),
            "z_max_m": round(float(path[:, 2].max()), 4),
            "z_excursion_m": round(float(np.max(np.abs(path[:, 2] - 0.9))), 4),
            "axis_deviation_mean_m": round(float(axis_distance.mean()), 4),
            "axis_deviation_max_m": round(float(axis_distance.max()), 4),
        }
        trajectories.append(trajectory)

    selected = []
    for gamma in GAMMAS:
        selected.extend(_select_three([
            row for row in trajectories if row["gamma"] == gamma
        ]))

    def mean(key: str) -> float:
        values = [float(row[key]) for row in trajectories if row[key] is not None]
        return round(float(np.mean(values)), 4)

    z_centers = [sphere[2] for row in trajectories for sphere in row["spheres"]]
    details = metric["successful_executed_commit_by_gamma"]
    cumulative_success = {_gamma_key(gamma): 0 for gamma in GAMMAS}
    cumulative_committed = {_gamma_key(gamma): 0 for gamma in GAMMAS}
    for round_row in metric_rows:
        for gamma in GAMMAS:
            key = _gamma_key(gamma)
            detail = round_row["successful_executed_commit_by_gamma"][key]
            cumulative_success[key] += int(detail["success_episode_count"])
            cumulative_committed[key] += int(detail["committed_trajectory_count"])
    return {
        "id": arm_id,
        "label": label,
        "scene_law": "dense z∈[0.7,1.1]" if arm_id.startswith("dense") else "uniform z∈[0.6,1.2]",
        "speed_weight": speed_weight,
        "latest_committed_round": round_index,
        "trajectory_count": len(trajectories),
        "commit_by_gamma": {
            _gamma_key(gamma): int(details[_gamma_key(gamma)]["committed_trajectory_count"])
            for gamma in GAMMAS
        },
        "success_seen_by_gamma": {
            _gamma_key(gamma): int(details[_gamma_key(gamma)]["success_episode_count"])
            for gamma in GAMMAS
        },
        "completed_round_success_by_gamma": cumulative_success,
        "completed_round_committed_by_gamma": cumulative_committed,
        "summary": {
            "mean_min_clearance_m": mean("min_clearance_m"),
            "mean_speed_mps": mean("mean_speed_mps"),
            "mean_near_obstacle_speed_mps": mean("near_obstacle_speed_mps"),
            "mean_z_excursion_m": mean("z_excursion_m"),
            "mean_axis_deviation_m": mean("axis_deviation_mean_m"),
            "mean_duration_s": mean("duration_s"),
            "sphere_center_z_min_m": round(float(min(z_centers)), 4),
            "sphere_center_z_max_m": round(float(max(z_centers)), 4),
        },
        "trajectories": selected,
    }


def _evaluation_payload(root: Path) -> list[dict[str, Any]]:
    specs = (
        ("dense200_m1", "dense-z · speed200 · faithful M1", root / "dense200/faithful_m1_r0_r10/raw_eval.json"),
        ("dense200_m8", "dense-z · speed200 · hack M8", root / "dense200/hack_m8_r0_r10/raw_eval.json"),
        ("dense100_m8", "dense-z · speed100 · hack M8", root / "dense100/hack_m8_r0_r5/raw_eval.json"),
        ("uniform200_m8", "uniform-z · speed200 · hack M8", root / "uniform200/hack_m8_r0_r5/raw_eval.json"),
    )
    output = []
    for series_id, label, path in specs:
        payload = json.loads(path.read_text())
        rows = []
        for round_raw, summary in sorted(payload["summary"].items(), key=lambda item: int(item[0])):
            pooled = summary["pooled"]
            rows.append({
                "round": int(round_raw),
                "SR": pooled["SR"],
                "CR": pooled["CR"],
                "OOB": pooled["OOB"],
                "timeout": pooled["timeout"],
                "validity": pooled["window_validity"],
                "clearance": pooled["successful_min_clearance_m"],
                "ttg": pooled["successful_time_to_goal_s"],
            })
        output.append({"id": series_id, "label": label, "rows": rows})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = [_arm_payload(args.trajectory_root, arm_id) for arm_id in ARM_SPECS]
    first_config = load_config(args.trajectory_root / "dense200/task_config_resolved.json")
    env = TaskEnvironment(first_config)
    payload = {
        "start": np.round(env.start[:3], 4).tolist(),
        "goal": np.round(env.goal[:3], 4).tolist(),
        "bounds": np.round(env.bounds, 4).tolist(),
        "arms": arms,
        "evaluations": _evaluation_payload(args.evaluation_root),
        "evaluation_completeness": [
            {"arm": "dense200", "faithful": "R0–R10 complete", "hack": "R0–R10 complete"},
            {"arm": "dense100", "faithful": "missing", "hack": "R0–R5 only"},
            {"arm": "uniform200", "faithful": "missing", "hack": "R0–R5 only"},
            {"arm": "uniform100", "faithful": "missing", "hack": "missing"},
        ],
        "observed_terminal_successes": {
            "dense200": [110, 112, 102, 67],
            "dense100": [96, 114, 79, 71],
            "uniform200": [60, 51, 54, 45],
            "uniform100": [90, 82, 53, 143],
            "gammas": list(GAMMAS),
            "note": "includes successful attempts that were not commit-eligible under the exact paired quota",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "selected_trajectories": {row["id"]: len(row["trajectories"]) for row in arms},
        "summaries": {row["id"]: row["summary"] for row in arms},
    }, indent=2))


if __name__ == "__main__":
    main()
