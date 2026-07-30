"""Unconditioned path-focused clutter demo collection.

The geometry bank is fixed before SafeMPPI runs.  Expert failure never removes
or replaces a scene; only successful trajectories enter the behavior-cloning
archive, while every attempted scene/gamma outcome remains in the manifest.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from .acquire import aggregate_metrics, run_episode
from .controller import Mode1SafeMPPI
from .environment import TaskEnvironment
from .lab_clutter import (
    _accepted_lab_demo,
    _scene_arrays,
    config_for_scene,
    fixed_lab_clutter_config,
    start_goal_path_diagnostics,
    summarize_start_goal_path_diagnostics,
)
from .path_focused_clutter import (
    PathFocusedClutterSpec,
    path_focused_scene_bank,
)


def _trajectory_geometry(
    dense_positions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    cylinders: np.ndarray,
    *,
    interaction_distance_m: float = 0.30,
) -> dict:
    path = np.asarray(dense_positions, np.float64).reshape(-1, 3)
    start = np.asarray(start, np.float64).reshape(-1)[:3]
    goal = np.asarray(goal, np.float64).reshape(-1)[:3]
    displacement = np.diff(path, axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    path_length = float(lengths.sum())
    direct_length = float(np.linalg.norm(goal - start))
    realized_displacement = float(np.linalg.norm(path[-1] - path[0]))
    direction = (goal - start) / direct_length
    along = (path - start[None]) @ direction
    closest = start[None] + along[:, None] * direction[None]
    transverse = path - closest
    transverse_rms = float(np.sqrt(np.mean(np.sum(
        transverse * transverse, axis=1,
    ))))

    horizontal_direction = direction[:2]
    horizontal_direction /= np.linalg.norm(horizontal_direction)
    horizontal_normal = np.asarray(
        [-horizontal_direction[1], horizontal_direction[0]],
        np.float64,
    )
    signed = (path[:, :2] - start[None, :2]) @ horizontal_normal
    signs = np.sign(signed[np.abs(signed) >= 0.02])
    sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    valid_displacements = displacement[lengths > 1.0e-8]
    total_turn = 0.0
    if len(valid_displacements) > 1:
        unit = valid_displacements / np.linalg.norm(
            valid_displacements, axis=1, keepdims=True,
        )
        cosines = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
        total_turn = float(np.arccos(cosines).sum())

    cylinders = np.asarray(cylinders, np.float64).reshape(-1, 3)
    interaction_count = 0
    if len(cylinders):
        surface_distance = (
            np.linalg.norm(
                path[:, None, :2] - cylinders[None, :, :2], axis=2,
            )
            - cylinders[None, :, 2]
        )
        interaction_count = int(np.sum(
            surface_distance.min(axis=0) <= float(interaction_distance_m)
        ))
    return {
        "path_length_m": path_length,
        "path_length_excess_ratio": (
            path_length / max(realized_displacement, 1.0e-12) - 1.0
        ),
        "transverse_rms_m": transverse_rms,
        "horizontal_side_sign_changes": sign_changes,
        "total_turn_radians": total_turn,
        "interacted_obstacle_count": interaction_count,
    }


def _behavior_summary(rows: list[dict], gammas: tuple[float, ...]) -> list[dict]:
    output = []
    for gamma in gammas:
        group = [
            row for row in rows
            if np.isclose(row["gamma"], gamma, rtol=0.0, atol=1.0e-7)
        ]
        successes = [row for row in group if row["success"]]

        def mean(key: str):
            values = [float(row[key]) for row in successes]
            return float(np.mean(values)) if values else None

        output.append({
            "gamma": float(gamma),
            "attempted_scenes": len(group),
            "successful_scenes": len(successes),
            "timeout_rate": float(np.mean([
                not row["success"]
                and not row["collision"]
                and not row["taskspace_violation"]
                for row in group
            ])),
            "mean_path_length_excess_ratio": mean(
                "path_length_excess_ratio"
            ),
            "mean_transverse_rms_m": mean("transverse_rms_m"),
            "mean_horizontal_side_sign_changes": mean(
                "horizontal_side_sign_changes"
            ),
            "mean_total_turn_radians": mean("total_turn_radians"),
            "mean_interacted_obstacle_count": mean(
                "interacted_obstacle_count"
            ),
        })
    return output


def collect_path_focused_clutter_demos(
    base_config,
    output_dir: str | Path,
    *,
    scene_count: int | None = None,
    domain_seed: int | None = None,
    rollout_seed_start: int | None = None,
    transverse_std_m: float | None = None,
    device: str = "cpu",
    episode_runner=run_episode,
    controller_factory=Mode1SafeMPPI,
) -> dict:
    """Evaluate one fixed geometry bank once per gamma and archive successes."""
    count = (
        int(base_config.data.episodes_per_gamma)
        if scene_count is None else int(scene_count)
    )
    if count < 1:
        raise ValueError("scene_count must be positive")
    template = fixed_lab_clutter_config(
        base_config, episodes_per_gamma=count,
    )
    raw = copy.deepcopy(template.raw)
    randomization = dict(raw.get("scene_randomization", {}))
    if transverse_std_m is not None:
        if not np.isfinite(transverse_std_m) or transverse_std_m <= 0.0:
            raise ValueError("transverse_std_m must be finite and positive")
        randomization["transverse_std_m"] = float(transverse_std_m)
    if domain_seed is not None:
        randomization["seed"] = int(domain_seed)
    raw["scene_randomization"] = randomization
    template = type(template)(
        template.taskspace,
        template.obstacles,
        template.safemppi,
        template.data,
        raw,
    )
    spec = PathFocusedClutterSpec.from_config(
        template, expected_family="vertical_cylinders",
    )
    scenes = path_focused_scene_bank(
        template,
        count,
        seed=spec.domain_seed,
    )

    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw["data"] = dict(raw["data"])
    raw["data"]["episodes_per_gamma"] = count
    if raw["data"].get("max_attempts_per_gamma") is not None:
        raw["data"]["max_attempts_per_gamma"] = max(
            count, int(raw["data"]["max_attempts_per_gamma"])
        )
    raw["scene_randomization"].update({
        "admission_mode": "geometry_only_no_expert_conditioning",
        "unconditioned_geometry_bank": True,
        "expert_rollouts_per_scene_gamma": 1,
    })
    (output / "resolved_config.json").write_text(
        json.dumps(raw, indent=2) + "\n"
    )

    seed0 = (
        int(template.data.seed_start)
        if rollout_seed_start is None else int(rollout_seed_start)
    )
    gammas = tuple(map(float, template.data.gammas))
    attempts: list[dict] = []
    runs: list[dict] = []
    scene_rows = []
    for scene in scenes:
        diagnostics = start_goal_path_diagnostics(
            scene,
            start=np.asarray(template.taskspace.start),
            goal=np.asarray(template.taskspace.goal),
            soft_clearance_target_m=template.safemppi.soft_clearance_target,
        )
        scene_rows.append({
            **scene.as_manifest_row(),
            "obstacle_count": len(scene.cylinders),
            "start_goal_path_diagnostics": diagnostics,
        })
        scene_config = config_for_scene(template, scene)
        env = TaskEnvironment(scene_config)
        rollout_seed = seed0 + 1009 * int(scene.index)
        for gamma in gammas:
            controller = controller_factory(
                scene_config.safemppi, env, device=device,
            )
            row, arrays = episode_runner(
                env,
                controller,
                gamma,
                rollout_seed,
                scene_config.data.rollout_dynamics,
            )
            accepted = _accepted_lab_demo(row)
            geometry = _trajectory_geometry(
                arrays["dense_positions"],
                env.start,
                env.goal,
                np.asarray(scene.cylinders, np.float32),
            )
            row = {
                **row,
                **scene.as_manifest_row(),
                "obstacle_count": len(scene.cylinders),
                "trajectory_accepted": accepted,
                "accepted": accepted,
                "file": None,
                **geometry,
            }
            if accepted:
                name = (
                    f"run_g{gamma:g}_{scene.scene_id}"
                    f"_s{rollout_seed}.npz"
                )
                np.savez_compressed(
                    output / name, **arrays, **_scene_arrays(scene),
                )
                row["file"] = name
                runs.append(row)
            attempts.append(row)

    metrics = aggregate_metrics(attempts, gammas)
    behavior = _behavior_summary(attempts, gammas)
    manifest = {
        "kind": (
            "Minhyuk lab path-focused randomized-cylinder SafeMPPI "
            "demonstrations"
        ),
        "schema_version": 3,
        "status": "COMPLETE_GEOMETRY_BANK_EVALUATED",
        "config": "resolved_config.json",
        "rollout_dynamics": template.data.rollout_dynamics,
        "acceptance": template.data.acceptance,
        "requested_scene_count": count,
        "evaluated_scene_count": len(scenes),
        "admitted_scene_count": len(scenes),
        "rejected_scene_count": 0,
        "gammas": list(gammas),
        "sampling_distribution": {
            "proposal": spec.scene_schema,
            "unconditioned_geometry": True,
            "expert_success_used_for_scene_admission": False,
            "failed_scenes_retained": True,
            "parameters": {
                "count_min": spec.count_min,
                "count_max": spec.count_max,
                "modeled_radius_m": spec.modeled_radius_m,
                "vehicle_inflation_m": spec.vehicle_inflation_m,
                "longitudinal_fraction": [
                    spec.longitudinal_min,
                    spec.longitudinal_max,
                ],
                "transverse_std_m": spec.transverse_std_m,
                "minimum_obstacle_surface_gap_m": (
                    spec.minimum_surface_gap_m
                ),
                "minimum_taskspace_wall_surface_clearance_m": (
                    spec.boundary_surface_gap_m
                ),
            },
        },
        "scene_bank": {
            "shared_across_gamma": True,
            "domain_seed": spec.domain_seed,
            "start_goal_path_summary": (
                summarize_start_goal_path_diagnostics([
                    row["start_goal_path_diagnostics"]
                    for row in scene_rows
                ])
            ),
            "scenes": scene_rows,
            "rejected_scenes": [],
        },
        "runs": runs,
        "attempts": attempts,
        "metrics": metrics,
        "behavior_metrics": behavior,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "metrics.json").write_text(
        json.dumps({
            "expert_outcomes": metrics,
            "behavior": behavior,
        }, indent=2) + "\n"
    )
    if metrics:
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(metrics[0]), lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(metrics)
    return manifest
