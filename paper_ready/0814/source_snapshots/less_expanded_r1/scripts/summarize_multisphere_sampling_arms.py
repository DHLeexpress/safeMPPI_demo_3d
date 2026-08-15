#!/usr/bin/env python3
"""Summarize committed trajectories from the two PRE2 sampling sanity arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safe_mppi.lab_clutter_pre2_expansion import (
    rotate_points_180_about_start_goal_axis,
)
from safe_mppi.bowling_coverage import bowling_route_signature


GAMMAS = (0.1, 0.3, 0.5, 1.0)
PROGRESS_GRID = np.linspace(0.2, 0.7, 101)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _frame(start: np.ndarray, goal: np.ndarray):
    delta = goal - start
    length = float(np.linalg.norm(delta))
    forward = delta / length
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    vertical = np.asarray([0.0, 0.0, 1.0])
    return length, forward, lateral, vertical


def _profiles(path: np.ndarray, start: np.ndarray, goal: np.ndarray):
    length, forward, lateral, vertical = _frame(start, goal)
    offset = path - start[None]
    progress = offset @ forward / length
    lateral_values = offset @ lateral
    vertical_values = offset @ vertical
    order = np.argsort(progress, kind="stable")
    progress = progress[order]
    lateral_values = lateral_values[order]
    vertical_values = vertical_values[order]
    keep = np.r_[True, np.diff(progress) > 1.0e-9]
    progress = progress[keep]
    lateral_values = lateral_values[keep]
    vertical_values = vertical_values[keep]
    return np.column_stack([
        np.interp(PROGRESS_GRID, progress, lateral_values),
        np.interp(PROGRESS_GRID, progress, vertical_values),
    ])


def _bowling_signature(
    path: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    lane_delta: float = 0.345,
) -> dict[str, Any]:
    return bowling_route_signature(
        path, start, goal, lane_delta_m=lane_delta,
    )


def _scene_map(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for row in manifest.get("lab_scene_ledger", []):
        result[str(row["scene_hash"])] = np.asarray(
            row["spheres"], np.float64,
        )
    return result


def _committed_episode_keys(manifest: dict[str, Any]):
    keys = []
    for round_row in manifest["rounds"]:
        round_index = int(round_row["round"])
        for gamma_text, detail in round_row[
            "successful_executed_commit_by_gamma"
        ].items():
            for episode in detail["committed_episode_ids"]:
                keys.append((round_index, float(gamma_text), int(episode)))
    return set(keys)


def _load_arm(path: Path, arm: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    events = torch.load(
        path / "events.pt", map_location="cpu", weights_only=False,
    )
    committed = _committed_episode_keys(manifest)
    grouped: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            int(event["round"]),
            float(event["gamma"]),
            int(event["episode"]),
        )
        if key in committed:
            grouped.setdefault(key, []).append(event)
    if set(grouped) != committed:
        missing = sorted(committed - set(grouped))
        raise RuntimeError(f"missing committed event traces: {missing}")

    task_config = json.loads((path / "task_config_resolved.json").read_text())
    start = np.asarray(task_config["taskspace"]["start"][:3], np.float64)
    goal = np.asarray(task_config["taskspace"]["goal"], np.float64)
    bounds = np.column_stack([
        np.asarray(task_config["taskspace"]["origin"], np.float64),
        np.asarray(task_config["taskspace"]["origin"], np.float64)
        + np.asarray(task_config["taskspace"]["size"], np.float64),
    ])
    dt = float(task_config["safemppi"]["dt"])
    scenes = _scene_map(manifest)
    trajectories = []
    profiles: dict[float, list[np.ndarray]] = {gamma: [] for gamma in GAMMAS}
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["step"]))
        path_values = np.asarray(
            [rows[0]["robot"][:3]]
            + [row["robot_after"][:3] for row in rows],
            np.float64,
        )
        scene_hash = str(rows[0]["scene_hash"])
        spheres = scenes[scene_hash]
        clearance = np.linalg.norm(
            path_values[:, None, :] - spheres[None, :, :3], axis=2,
        ) - spheres[None, :, 3]
        gamma = float(key[1])
        profile = _profiles(path_values, start, goal)
        profiles[gamma].append(profile)
        row = {
            "round": key[0],
            "gamma": gamma,
            "episode": key[2],
            "scene_hash": scene_hash,
            "spheres": spheres,
            "path": path_values,
            "steps": len(path_values) - 1,
            "time_to_goal_s": dt * (len(path_values) - 1),
            "minimum_clearance_m": float(clearance.min()),
            "mean_z_m": float(path_values[:, 2].mean()),
            "minimum_z_m": float(path_values[:, 2].min()),
            "maximum_z_m": float(path_values[:, 2].max()),
            "flow_base_std": float(rows[0]["flow_base_std"]),
            "paired_scene_id": rows[0].get("paired_scene_id"),
            "paired_scene_member": rows[0].get("paired_scene_member"),
            "paired_scene_member_name": rows[0].get(
                "paired_scene_member_name"
            ),
            "fixed_scene_layout": rows[0].get("fixed_scene_layout"),
            "transverse_profile": profile,
        }
        if arm == "bowling":
            row["route"] = _bowling_signature(
                path_values, start, goal,
            )
        trajectories.append(row)

    round_row = manifest["rounds"][-1]
    attempts = int(round_row["attempted_episode_count"])
    known = (
        int(round_row["success"])
        + int(round_row["NVP"])
        + int(round_row["timeout"])
    )
    metrics = {
        "attempted_episodes": attempts,
        "terminal_successes": int(round_row["success"]),
        "NVP": int(round_row["NVP"]),
        "timeout": int(round_row["timeout"]),
        "collision_or_oob": attempts - known,
        "committed_trajectories": len(trajectories),
        "retry_batches_by_gamma": round_row["retry_batches_by_gamma"],
        "optimizer_steps": int(round_row["steps"]),
        "optimizer_step": int(round_row.get(
            "optimizer_step", round_row["steps"],
        )),
        "loss": float(round_row["positive_loss"]),
        "replay_positives": int(round_row["replay_positives"]),
        "ESS_over_K": float(round_row["ESS_over_K"]),
        "marginal_ESS_over_K": float(round_row["marginal_ESS_over_K"]),
        "uncertainty_uplift": round_row["uncertainty_uplift"],
        "gather_s": float(round_row["gather_s"]),
        "flow_sampling_s": float(round_row["flow_sampling_s"]),
        "acquisition_s": float(round_row["acquisition_s"]),
        "verifier_dispatch_s": float(round_row["verifier_dispatch_s"]),
        "update_s": float(round_row["update_s"]),
        "round_total_s": float(round_row["round_total_s"]),
        "gp_buffer": int(round_row["gp_buffer"]),
        "flow_base_std": float(trajectories[0]["flow_base_std"]),
    }

    diversity: dict[str, Any] = {}
    if arm == "paired":
        pair_rows = {}
        for gamma in GAMMAS:
            members = sorted(
                [row for row in trajectories if row["gamma"] == gamma],
                key=lambda row: int(row["paired_scene_member"]),
            )
            if [row["paired_scene_member"] for row in members] != [0, 1]:
                raise RuntimeError(f"gamma={gamma:g} lacks one exact pair")
            original = np.asarray(members[0]["path"], np.float64)
            rotated_back = rotate_points_180_about_start_goal_axis(
                np.asarray(members[1]["path"], np.float64), start, goal,
            )
            original_profile = _profiles(original, start, goal)
            recovered_profile = _profiles(rotated_back, start, goal)
            raw_profile = np.asarray(members[1]["transverse_profile"])
            raw_difference = (
                np.asarray(members[0]["transverse_profile"]) - raw_profile
            )
            equivariance_difference = original_profile - recovered_profile
            pair_rows[f"{gamma:g}"] = {
                "raw_transverse_rms_m": float(np.sqrt(
                    np.square(raw_difference).sum(axis=1).mean()
                )),
                "rotation_recovered_rms_m": float(np.sqrt(
                    np.square(equivariance_difference).sum(axis=1).mean()
                )),
                "scene_pair_id": members[0]["paired_scene_id"],
            }
        diversity["paired_by_gamma"] = pair_rows
        diversity["mean_raw_transverse_rms_m"] = float(np.mean([
            row["raw_transverse_rms_m"] for row in pair_rows.values()
        ]))
        diversity["mean_rotation_recovered_rms_m"] = float(np.mean([
            row["rotation_recovered_rms_m"]
            for row in pair_rows.values()
        ]))
    else:
        route_codes = [row["route"]["code"] for row in trajectories]
        stable_codes = [
            row["route"]["stable_code"] for row in trajectories
        ]
        within = {}
        centroids = {}
        for gamma in GAMMAS:
            gamma_profiles = [
                np.asarray(row["transverse_profile"])
                for row in trajectories if row["gamma"] == gamma
            ]
            if len(gamma_profiles) != 2:
                raise RuntimeError(f"gamma={gamma:g} does not have q2")
            z_difference = gamma_profiles[0][:, 1] - gamma_profiles[1][:, 1]
            transverse_difference = gamma_profiles[0] - gamma_profiles[1]
            within[f"{gamma:g}"] = {
                "z_rms_m": float(np.sqrt(np.square(z_difference).mean())),
                "transverse_rms_m": float(np.sqrt(
                    np.square(transverse_difference).sum(axis=1).mean()
                )),
            }
            centroids[gamma] = np.mean(gamma_profiles, axis=0)
        between_z, between_transverse = [], []
        for first_index, first in enumerate(GAMMAS):
            for second in GAMMAS[first_index + 1:]:
                difference = centroids[first] - centroids[second]
                between_z.append(float(np.sqrt(
                    np.square(difference[:, 1]).mean()
                )))
                between_transverse.append(float(np.sqrt(
                    np.square(difference).sum(axis=1).mean()
                )))
        diversity.update({
            "route_codes": route_codes,
            "stable_route_codes": stable_codes,
            "unique_route_codes": sorted(set(route_codes)),
            "unique_stable_route_codes": sorted(set(stable_codes)),
            "within_gamma": within,
            "mean_within_gamma_z_rms_m": float(np.mean([
                row["z_rms_m"] for row in within.values()
            ])),
            "mean_between_gamma_z_rms_m": float(np.mean(between_z)),
            "mean_between_gamma_transverse_rms_m": float(
                np.mean(between_transverse)
            ),
        })

    return {
        "arm": arm,
        "path": str(path.resolve()),
        "start": start,
        "goal": goal,
        "bounds": bounds,
        "metrics": metrics,
        "diversity": diversity,
        "trajectories": trajectories,
        "checkpoint": str((path / "checkpoint_001.pt").resolve()),
        "recipe": str((path / "RECIPE.sh").resolve()),
        "source_sha256": manifest.get("source_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--bowling", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "pre2_multisphere_sampling_sanity_v1",
        "paired": _load_arm(args.paired, "paired"),
        "bowling": _load_arm(args.bowling, "bowling"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_ready(result), indent=2, allow_nan=False) + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
