"""Unconditioned path-focused clutter demo collection.

The geometry bank is fixed before SafeMPPI runs.  Expert failure never removes
or replaces a scene; only successful trajectories enter the behavior-cloning
archive, while every attempted scene/gamma outcome remains in the manifest.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np

from .acquire import aggregate_metrics, run_episode
from .config import validate_config
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


SUCCESS_QUOTA_SCHEMA = "path_focused_success_quota_v1"


def path_focused_collection_config(
    base_config,
    *,
    gammas: tuple[float, ...] | None = None,
    episodes_per_gamma: int | None = None,
    max_attempts_per_gamma: int | None = None,
    centroid_gain: float | None = None,
    centroid_smooth: float | None = None,
    sigma_aniso: float | None = None,
):
    """Return one validated in-memory config with collection-only overrides."""
    gamma_values = (
        tuple(map(float, base_config.data.gammas))
        if gammas is None else tuple(map(float, gammas))
    )
    if not gamma_values:
        raise ValueError("at least one collection gamma is required")
    episodes = (
        int(base_config.data.episodes_per_gamma)
        if episodes_per_gamma is None else int(episodes_per_gamma)
    )
    attempts = (
        base_config.data.max_attempts_per_gamma
        if max_attempts_per_gamma is None else int(max_attempts_per_gamma)
    )
    if attempts is not None:
        attempts = max(int(attempts), episodes)

    gain = (
        float(base_config.safemppi.centroid_gain)
        if centroid_gain is None else float(centroid_gain)
    )
    smooth = (
        float(base_config.safemppi.centroid_smooth)
        if centroid_smooth is None else float(centroid_smooth)
    )
    anisotropy = (
        float(base_config.safemppi.sigma_aniso)
        if sigma_aniso is None else float(sigma_aniso)
    )
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("centroid_gain must be finite and nonnegative")
    if not np.isfinite(smooth) or not 0.0 <= smooth <= 1.0:
        raise ValueError("centroid_smooth must lie in [0,1]")
    if not np.isfinite(anisotropy) or anisotropy <= 0.0:
        raise ValueError("sigma_aniso must be finite and positive")

    mppi = replace(
        base_config.safemppi,
        centroid_gain=gain,
        centroid_smooth=smooth,
        sigma_aniso=anisotropy,
    )
    data = replace(
        base_config.data,
        gammas=gamma_values,
        episodes_per_gamma=episodes,
        max_attempts_per_gamma=attempts,
    )
    raw = copy.deepcopy(base_config.raw)
    raw["safemppi"] = dict(raw["safemppi"])
    raw["safemppi"].update({
        "centroid_gain": gain,
        "centroid_smooth": smooth,
        "sigma_aniso": anisotropy,
    })
    raw["data"] = dict(raw["data"])
    raw["data"].update({
        "gammas": list(gamma_values),
        "episodes_per_gamma": episodes,
        "max_attempts_per_gamma": attempts,
    })
    config = type(base_config)(
        base_config.taskspace,
        base_config.obstacles,
        mppi,
        data,
        raw,
    )
    validate_config(config)
    return config


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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


def collect_path_focused_success_quota(
    base_config,
    output_dir: str | Path,
    *,
    target_successes_per_gamma: int,
    max_scene_count: int,
    gammas: tuple[float, ...] | None = None,
    domain_seed: int | None = None,
    rollout_seed_start: int | None = None,
    centroid_gain: float | None = None,
    centroid_smooth: float | None = None,
    sigma_aniso: float | None = None,
    device: str = "cpu",
    episode_runner=run_episode,
    controller_factory=Mode1SafeMPPI,
) -> dict:
    """Collect the first exact success quota from one geometry-only stream.

    Each scene/gamma attempt is committed as an atomic JSON shard. Accepted
    trajectory arrays are atomically installed before the corresponding shard.
    Reinvoking the same contract skips committed shards and resumes at the first
    missing cell. A gamma is no longer evaluated after its exact quota is met.
    """
    target = int(target_successes_per_gamma)
    max_scenes = int(max_scene_count)
    if target < 1 or max_scenes < target:
        raise ValueError(
            "target_successes_per_gamma must be positive and cannot exceed "
            "max_scene_count"
        )
    template = fixed_lab_clutter_config(
        base_config, episodes_per_gamma=target,
    )
    template = path_focused_collection_config(
        template,
        gammas=gammas,
        episodes_per_gamma=target,
        max_attempts_per_gamma=max_scenes,
        centroid_gain=centroid_gain,
        centroid_smooth=centroid_smooth,
        sigma_aniso=sigma_aniso,
    )
    raw = copy.deepcopy(template.raw)
    randomization = dict(raw.get("scene_randomization", {}))
    if domain_seed is not None:
        randomization["seed"] = int(domain_seed)
    randomization.update({
        "admission_mode": "geometry_only_no_expert_conditioning",
        "unconditioned_geometry_bank": True,
        "expert_rollouts_per_scene_gamma": 1,
    })
    raw["scene_randomization"] = randomization
    raw["data"] = dict(raw["data"])
    raw["data"].update({
        "episodes_per_gamma": target,
        "max_attempts_per_gamma": max_scenes,
    })
    template = type(template)(
        template.taskspace,
        template.obstacles,
        template.safemppi,
        replace(
            template.data,
            episodes_per_gamma=target,
            max_attempts_per_gamma=max_scenes,
        ),
        raw,
    )
    validate_config(template)
    spec = PathFocusedClutterSpec.from_config(
        template, expected_family="vertical_cylinders",
    )
    scenes = path_focused_scene_bank(
        template, max_scenes, seed=spec.domain_seed,
    )
    gamma_values = tuple(map(float, template.data.gammas))
    seed0 = (
        int(template.data.seed_start)
        if rollout_seed_start is None else int(rollout_seed_start)
    )

    output = Path(output_dir).resolve()
    contract_path = output / "quota_contract.json"
    resolved_path = output / "resolved_config.json"
    shard_dir = output / "attempt_shards"
    contract = {
        "schema": SUCCESS_QUOTA_SCHEMA,
        "target_successes_per_gamma": target,
        "max_scene_count": max_scenes,
        "gammas": list(gamma_values),
        "domain_seed": int(spec.domain_seed),
        "rollout_seed_start": seed0,
        "resolved_config": raw,
    }
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("existing success-quota contract does not match")
        if not resolved_path.is_file():
            raise FileNotFoundError("resumable archive lacks resolved_config.json")
        if json.loads(resolved_path.read_text()) != raw:
            raise ValueError("resumable archive resolved config changed")
    else:
        if any(output.iterdir()):
            raise FileExistsError(
                f"nonempty output lacks a quota contract: {output}"
            )
        shard_dir.mkdir()
        _atomic_json(resolved_path, raw)
        _atomic_json(contract_path, contract)
    shard_dir.mkdir(exist_ok=True)

    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())

    attempts_by_cell: dict[tuple[int, int], dict] = {}
    for shard_path in sorted(shard_dir.glob("scene_*_gamma_*.json")):
        shard = json.loads(shard_path.read_text())
        if shard.get("schema") != SUCCESS_QUOTA_SCHEMA:
            raise ValueError(f"invalid attempt shard schema: {shard_path}")
        scene_index = int(shard["scene_index"])
        gamma_index = int(shard["gamma_index"])
        key = (scene_index, gamma_index)
        if key in attempts_by_cell:
            raise ValueError(f"duplicate attempt shard for cell {key}")
        if not 0 <= scene_index < max_scenes:
            raise ValueError(f"attempt shard scene index is out of range: {key}")
        if not 0 <= gamma_index < len(gamma_values):
            raise ValueError(f"attempt shard gamma index is out of range: {key}")
        if not np.isclose(
            float(shard["row"]["gamma"]), gamma_values[gamma_index],
            rtol=0.0, atol=1.0e-9,
        ):
            raise ValueError(f"attempt shard gamma mismatch: {key}")
        row = shard["row"]
        if row.get("accepted", False):
            run_file = output / str(row.get("file"))
            if not run_file.is_file():
                raise FileNotFoundError(
                    f"accepted attempt lacks trajectory file: {run_file}"
                )
        attempts_by_cell[key] = row

    def accepted_counts() -> dict[int, int]:
        return {
            gamma_index: sum(
                bool(row.get("accepted", False))
                for (scene_index, observed_gamma), row
                in attempts_by_cell.items()
                if observed_gamma == gamma_index
            )
            for gamma_index in range(len(gamma_values))
        }

    counts = accepted_counts()
    for scene in scenes:
        if all(counts[index] == target for index in counts):
            break
        scene_config = config_for_scene(template, scene)
        env = TaskEnvironment(scene_config)
        rollout_seed = seed0 + 1009 * int(scene.index)
        for gamma_index, gamma in enumerate(gamma_values):
            if counts[gamma_index] == target:
                continue
            key = (int(scene.index), gamma_index)
            if key in attempts_by_cell:
                continue
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
                "attempt_index": int(scene.index),
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
                _atomic_npz(
                    output / name,
                    {**arrays, **_scene_arrays(scene)},
                )
                row["file"] = name
            shard = {
                "schema": SUCCESS_QUOTA_SCHEMA,
                "scene_index": int(scene.index),
                "gamma_index": gamma_index,
                "row": row,
            }
            shard_path = shard_dir / (
                f"scene_{int(scene.index):09d}_gamma_{gamma_index:03d}.json"
            )
            _atomic_json(shard_path, shard)
            attempts_by_cell[key] = row
            if accepted:
                counts[gamma_index] += 1

    attempts = [
        attempts_by_cell[key] for key in sorted(attempts_by_cell)
    ]
    runs = [row for row in attempts if row.get("accepted", False)]
    used_scene_indices = sorted({
        int(row["scene_index"]) for row in attempts
    })
    scene_rows = []
    for scene_index in used_scene_indices:
        scene = scenes[scene_index]
        scene_rows.append({
            **scene.as_manifest_row(),
            "obstacle_count": len(scene.cylinders),
            "start_goal_path_diagnostics": start_goal_path_diagnostics(
                scene,
                start=np.asarray(template.taskspace.start),
                goal=np.asarray(template.taskspace.goal),
                soft_clearance_target_m=(
                    template.safemppi.soft_clearance_target
                ),
            ),
        })
    metrics = aggregate_metrics(attempts, gamma_values)
    behavior = _behavior_summary(attempts, gamma_values)
    complete = all(counts[index] == target for index in counts)
    manifest = {
        "kind": (
            "Minhyuk lab path-focused randomized-cylinder SafeMPPI "
            "success-quota demonstrations"
        ),
        "schema_version": 4,
        "status": (
            "COMPLETE_EXACT_SUCCESS_QUOTA"
            if complete else "FAILED_MAX_SCENE_BUDGET"
        ),
        "config": "resolved_config.json",
        "quota_contract": "quota_contract.json",
        "rollout_dynamics": template.data.rollout_dynamics,
        "acceptance": template.data.acceptance,
        "target_successes_per_gamma": target,
        "max_scene_count": max_scenes,
        "evaluated_scene_count": len(used_scene_indices),
        "accepted_counts_by_gamma": {
            f"{gamma_values[index]:.9g}": counts[index]
            for index in range(len(gamma_values))
        },
        "gammas": list(gamma_values),
        "sampling_distribution": {
            "proposal": spec.scene_schema,
            "unconditioned_geometry": True,
            "expert_success_used_for_scene_admission": False,
            "failed_scenes_retained": True,
            "stopping_rule": (
                "stop each gamma after its first exact accepted-trajectory "
                "quota; scene proposals never depend on expert outcomes"
            ),
        },
        "scene_bank": {
            "shared_across_gamma": False,
            "shared_deterministic_scene_prefix": True,
            "domain_seed": spec.domain_seed,
            "start_goal_path_summary": (
                summarize_start_goal_path_diagnostics([
                    row["start_goal_path_diagnostics"]
                    for row in scene_rows
                ])
                if scene_rows else None
            ),
            "scenes": scene_rows,
            "rejected_scenes": [],
        },
        "runs": runs,
        "attempts": attempts,
        "metrics": metrics,
        "behavior_metrics": behavior,
    }
    if not complete:
        _atomic_json(output / "FAILED_collection.json", manifest)
        raise RuntimeError(
            "success quota was not reached within max_scene_count: "
            f"{manifest['accepted_counts_by_gamma']}"
        )

    _atomic_json(output / "metrics.json", {
        "expert_outcomes": metrics,
        "behavior": behavior,
    })
    csv_path = output / "metrics.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(metrics[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metrics)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_csv, csv_path)
    _atomic_json(manifest_path, manifest)
    return manifest
