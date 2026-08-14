"""Gamma-exclusive mirrored-pair SafeMPPI demonstration collection.

Each trial owns one obstacle scene and its exact reflection about the fixed
start-goal line in x-y.  Both members are rolled out once at one gamma.  The
pair contributes two training trajectories only when both members satisfy the
existing nominal-safe-success acceptance rule; otherwise it contributes zero.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .acquire import aggregate_metrics, run_episode
from .config import ExperimentConfig, ObstacleConfig, validate_config
from .controller import Mode1SafeMPPI
from .environment import TaskEnvironment
from .lab_clutter import (
    ClutterScene,
    _accepted_lab_demo,
    _scene_arrays,
    config_for_scene,
    obstacle_scene_hash,
    start_goal_path_diagnostics,
)
from .path_focused_clutter import PathFocusedClutterSpec
from .path_focused_collection import (
    _atomic_json,
    _atomic_npz,
    _behavior_summary,
    _trajectory_geometry,
    path_focused_collection_config,
)


MIRRORED_PAIR_SCHEMA = "path_focused_gamma_exclusive_mirrored_pairs_v1"


def _gamma_key(gamma: float) -> int:
    """Stable integer namespace for one configured gamma value."""
    value = float(gamma)
    key = int(round(value * 1_000_000.0))
    if not np.isclose(value, key / 1_000_000.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("gamma must be representable to six decimal places")
    return key


def _seed32(*parts: int) -> int:
    return int(
        np.random.SeedSequence([int(part) for part in parts])
        .generate_state(1, dtype=np.uint32)[0]
    )


def relaxed_mirrored_collection_config(
    base_config: ExperimentConfig,
    *,
    gammas: tuple[float, ...] | None = None,
    target_successes_per_gamma: int | None = None,
    max_pair_attempts_per_gamma: int | None = None,
) -> ExperimentConfig:
    """Clear static obstacles while preserving the configured taskspace."""
    bounds = np.asarray(base_config.taskspace.bounds, np.float64)
    start = np.asarray(base_config.taskspace.start, np.float64)
    goal = np.asarray(base_config.taskspace.goal, np.float64)
    if not np.allclose(
        bounds,
        np.asarray([[-2.5, 1.3], [-1.7, 1.8], [0.1, 1.7]]),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "mirrored collection requires taskspace x=[-2.5,1.3], "
            "y=[-1.7,1.8], z=[0.1,1.7]"
        )
    if not np.allclose(
        start, [-2.1, 1.5, 0.9, 0.0, 0.0, 0.0], rtol=0.0, atol=1.0e-6,
    ) or not np.allclose(
        goal, [0.7, -1.5, 0.9], rtol=0.0, atol=1.0e-6,
    ):
        raise ValueError("mirrored collection requires the fixed lab start/goal")
    if not np.isclose(
        start[2] - bounds[2, 0], bounds[2, 1] - start[2],
        rtol=0.0, atol=1.0e-12,
    ):
        raise ValueError("taskspace z bounds are not symmetric about z=0.9")
    if base_config.data.rollout_dynamics != "minhyuk_reference_governor":
        raise ValueError("mirrored collection requires the Minhyuk governor")
    if base_config.data.acceptance != "nominal_safe_success":
        raise ValueError("mirrored collection requires safe-success acceptance")

    target = (
        int(base_config.data.episodes_per_gamma)
        if target_successes_per_gamma is None
        else int(target_successes_per_gamma)
    )
    if target < 2 or target % 2:
        raise ValueError("paired success target must be a positive even number")
    pair_budget = (
        int(base_config.data.max_attempts_per_gamma or target) // 2
        if max_pair_attempts_per_gamma is None
        else int(max_pair_attempts_per_gamma)
    )
    if pair_budget < target // 2:
        raise ValueError("pair-attempt budget cannot be smaller than target/2")

    configured = path_focused_collection_config(
        base_config,
        gammas=gammas,
        episodes_per_gamma=target,
        max_attempts_per_gamma=2 * pair_budget,
    )
    raw = copy.deepcopy(configured.raw)
    raw["obstacles"] = {"spheres": [], "cylinders": []}
    raw["safety"] = {
        "safe_min": np.round(bounds[:, 0], 12).tolist(),
        "safe_max": np.round(bounds[:, 1], 12).tolist(),
    }
    raw["safemppi"] = dict(raw["safemppi"])
    raw["safemppi"]["z_bias_weight"] = 0.0
    raw["data"] = dict(raw["data"])
    raw["data"].update({
        "episodes_per_gamma": target,
        "max_attempts_per_gamma": 2 * pair_budget,
    })
    raw["domain_randomization"] = {
        "obstacle_geometry_source": "per_attempt_npz_and_manifest",
        "gamma_exclusive_scene_streams": True,
        "scene_bank_shared_across_gamma": False,
        "mirrored_pair_admission": "both_success=>2; otherwise=>0",
        "static_obstacles_in_resolved_config": False,
    }
    normalized = ExperimentConfig(
        configured.taskspace,
        ObstacleConfig(spheres=(), cylinders=()),
        replace(configured.safemppi, z_bias_weight=0.0),
        replace(
            configured.data,
            episodes_per_gamma=target,
            max_attempts_per_gamma=2 * pair_budget,
        ),
        raw,
    )
    validate_config(normalized)
    return normalized


def reflect_xy_about_start_goal_line(
    points_xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> np.ndarray:
    """Reflect x-y points about the infinite line through start and goal."""
    points = np.asarray(points_xy, np.float64).reshape(-1, 2)
    start = np.asarray(start_xy, np.float64).reshape(2)
    goal = np.asarray(goal_xy, np.float64).reshape(2)
    direction = goal - start
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        raise ValueError("reflection axis requires distinct start and goal")
    direction /= length
    projection = start[None] + (
        (points - start[None]) @ direction
    )[:, None] * direction[None]
    return 2.0 * projection - points


def _mirrored_scene(
    scene: ClutterScene,
    *,
    mirrored_index: int,
    mirrored_seed: int,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: np.ndarray,
) -> ClutterScene | None:
    if scene.spheres:
        raise ValueError("mirrored pretraining currently requires cylinders")
    cylinders = np.asarray(scene.cylinders, np.float64).reshape(-1, 3)
    centers = reflect_xy_about_start_goal_line(
        cylinders[:, :2], start[:2], goal[:2],
    ).astype(np.float32).astype(np.float64)
    radii = cylinders[:, 2].astype(np.float32).astype(np.float64)
    low = bounds[:2, 0] + radii[:, None]
    high = bounds[:2, 1] - radii[:, None]
    if bool((centers < low).any() or (centers > high).any()):
        return None
    rows = tuple(
        (float(center[0]), float(center[1]), float(radius))
        for center, radius in zip(centers, radii)
    )
    mirrored = ClutterScene(
        index=int(mirrored_index),
        seed=int(mirrored_seed),
        spheres=(),
        cylinders=rows,
        scene_hash=obstacle_scene_hash(cylinders=rows),
    )
    if mirrored.scene_hash == scene.scene_hash:
        return None
    recovered = reflect_xy_about_start_goal_line(
        np.asarray(mirrored.cylinders, np.float64)[:, :2],
        start[:2],
        goal[:2],
    )
    if not np.allclose(
        recovered,
        np.asarray(scene.cylinders, np.float64)[:, :2],
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise RuntimeError("float32 mirrored geometry failed involution check")
    return mirrored


def sample_gamma_exclusive_mirrored_pair(
    config: ExperimentConfig,
    *,
    gamma: float,
    pair_index: int,
    domain_seed: int,
    max_geometry_proposals: int = 10_000,
) -> tuple[ClutterScene, ClutterScene, dict]:
    """Deterministically sample one mirrorable geometry pair."""
    if pair_index < 0 or max_geometry_proposals < 1:
        raise ValueError("pair index must be nonnegative and budget positive")
    spec = PathFocusedClutterSpec.from_config(
        config, expected_family="vertical_cylinders",
    )
    gamma_key = _gamma_key(gamma)
    start = np.asarray(config.taskspace.start, np.float64)
    goal = np.asarray(config.taskspace.goal, np.float64)
    bounds = np.asarray(config.taskspace.bounds, np.float64)
    base_index = gamma_key * 1_000_000 + 2 * int(pair_index)
    for proposal_index in range(int(max_geometry_proposals)):
        source_seed = _seed32(
            int(domain_seed), gamma_key, int(pair_index), proposal_index, 0xA17,
        )
        original = spec.sample_scene(
            scene_index=base_index,
            scene_seed=source_seed,
            bounds=bounds,
            start=start,
            goal=goal,
        )
        mirror_seed = _seed32(source_seed, 1, 0xB17)
        mirrored = _mirrored_scene(
            original,
            mirrored_index=base_index + 1,
            mirrored_seed=mirror_seed,
            start=start,
            goal=goal,
            bounds=bounds,
        )
        if mirrored is not None:
            pair_id = f"g{gamma:g}_pair_{pair_index:06d}"
            return original, mirrored, {
                "pair_id": pair_id,
                "pair_index": int(pair_index),
                "gamma": float(gamma),
                "gamma_key": gamma_key,
                "geometry_proposal_index": proposal_index,
                "domain_seed": int(domain_seed),
                "reflection_axis_start_xy": start[:2].tolist(),
                "reflection_axis_goal_xy": goal[:2].tolist(),
                "member_scene_hashes": [
                    original.scene_hash, mirrored.scene_hash,
                ],
            }
    raise RuntimeError(
        f"could not sample a mirrorable pair for gamma={gamma:g}, "
        f"pair_index={pair_index}"
    )


def _attempt_path(
    output: Path, gamma: float, pair_index: int, member_index: int,
) -> Path:
    return output / "attempt_shards" / (
        f"g{gamma:g}_pair_{pair_index:06d}_member_{member_index}.npz"
    )


def _load_attempt(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        row = json.loads(str(archive["attempt_row_json"].item()))
        arrays = {
            key: archive[key]
            for key in archive.files
            if key != "attempt_row_json"
        }
    return row, arrays


def _write_progress(
    output: Path,
    gamma_values: tuple[float, ...],
    attempts: dict[tuple[int, int, int], dict],
    target: int,
) -> None:
    counts = {}
    pair_attempts = {}
    for gamma_index, gamma in enumerate(gamma_values):
        admitted = 0
        observed_pairs = set()
        for key, row in attempts.items():
            if key[0] != gamma_index:
                continue
            observed_pairs.add(key[1])
            if row.get("pair_admitted"):
                admitted += 1
        counts[f"{gamma:.9g}"] = admitted
        pair_attempts[f"{gamma:.9g}"] = len(observed_pairs)
    _atomic_json(output / "progress.json", {
        "schema": MIRRORED_PAIR_SCHEMA,
        "target_successes_per_gamma": target,
        "accepted_counts_by_gamma": counts,
        "attempted_pairs_by_gamma": pair_attempts,
    })


def collect_mirrored_pair_success_quota(
    base_config: ExperimentConfig,
    output_dir: str | Path,
    *,
    target_successes_per_gamma: int,
    max_pair_attempts_per_gamma: int,
    gammas: tuple[float, ...] | None = None,
    domain_seed: int | None = None,
    rollout_seed_start: int | None = None,
    device: str = "cpu",
    episode_runner=run_episode,
    controller_factory=Mode1SafeMPPI,
) -> dict:
    """Resume until every gamma has an exact paired-success trajectory quota."""
    target = int(target_successes_per_gamma)
    pair_budget = int(max_pair_attempts_per_gamma)
    template = relaxed_mirrored_collection_config(
        base_config,
        gammas=gammas,
        target_successes_per_gamma=target,
        max_pair_attempts_per_gamma=pair_budget,
    )
    spec = PathFocusedClutterSpec.from_config(
        template, expected_family="vertical_cylinders",
    )
    gamma_values = tuple(map(float, template.data.gammas))
    if len({_gamma_key(gamma) for gamma in gamma_values}) != len(gamma_values):
        raise ValueError("configured gamma namespaces collide")
    domain_seed_value = (
        int(spec.domain_seed) if domain_seed is None else int(domain_seed)
    )
    rollout_seed0 = (
        int(template.data.seed_start)
        if rollout_seed_start is None else int(rollout_seed_start)
    )

    output = Path(output_dir).resolve()
    contract_path = output / "pair_quota_contract.json"
    resolved_path = output / "resolved_config.json"
    shard_dir = output / "attempt_shards"
    contract = {
        "schema": MIRRORED_PAIR_SCHEMA,
        "target_successes_per_gamma": target,
        "max_pair_attempts_per_gamma": pair_budget,
        "gammas": list(gamma_values),
        "domain_seed": domain_seed_value,
        "rollout_seed_start": rollout_seed0,
        "resolved_config": template.raw,
    }
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("existing mirrored-pair contract does not match")
        if json.loads(resolved_path.read_text()) != template.raw:
            raise ValueError("resumable mirrored-pair resolved config changed")
    else:
        if any(output.iterdir()):
            raise FileExistsError(
                f"nonempty output lacks a mirrored-pair contract: {output}"
            )
        shard_dir.mkdir()
        _atomic_json(resolved_path, template.raw)
        _atomic_json(contract_path, contract)
    shard_dir.mkdir(exist_ok=True)

    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())

    attempts: dict[tuple[int, int, int], dict] = {}
    for gamma_index, gamma in enumerate(gamma_values):
        for pair_index in range(pair_budget):
            for member_index in (0, 1):
                path = _attempt_path(output, gamma, pair_index, member_index)
                if not path.is_file():
                    continue
                row, _ = _load_attempt(path)
                if (
                    not np.isclose(row["gamma"], gamma, rtol=0.0, atol=1.0e-9)
                    or int(row["pair_index"]) != pair_index
                    or int(row["pair_member_index"]) != member_index
                ):
                    raise ValueError(f"attempt shard identity mismatch: {path}")
                row["file"] = str(path.relative_to(output))
                attempts[(gamma_index, pair_index, member_index)] = row

    def reconcile_pair(gamma_index: int, pair_index: int) -> bool | None:
        keys = [
            (gamma_index, pair_index, 0),
            (gamma_index, pair_index, 1),
        ]
        if any(key not in attempts for key in keys):
            return None
        admitted = all(bool(attempts[key]["individual_accepted"]) for key in keys)
        for key in keys:
            attempts[key]["pair_admitted"] = admitted
            attempts[key]["accepted"] = admitted
        return admitted

    for gamma_index in range(len(gamma_values)):
        for pair_index in range(pair_budget):
            reconcile_pair(gamma_index, pair_index)

    _write_progress(output, gamma_values, attempts, target)
    for gamma_index, gamma in enumerate(gamma_values):
        accepted_count = sum(
            bool(row.get("pair_admitted"))
            for (observed_gamma, _, _), row in attempts.items()
            if observed_gamma == gamma_index
        )
        for pair_index in range(pair_budget):
            if accepted_count == target:
                break
            original, mirrored, pair_meta = sample_gamma_exclusive_mirrored_pair(
                template,
                gamma=gamma,
                pair_index=pair_index,
                domain_seed=domain_seed_value,
            )
            pair_scenes = (original, mirrored)
            for member_index, scene in enumerate(pair_scenes):
                key = (gamma_index, pair_index, member_index)
                if key in attempts:
                    continue
                scene_config = config_for_scene(template, scene)
                env = TaskEnvironment(scene_config)
                controller = controller_factory(
                    scene_config.safemppi, env, device=device,
                )
                rollout_seed = _seed32(
                    rollout_seed0,
                    _gamma_key(gamma),
                    pair_index,
                    member_index,
                    0xC17,
                )
                row, arrays = episode_runner(
                    env,
                    controller,
                    gamma,
                    rollout_seed,
                    scene_config.data.rollout_dynamics,
                )
                individual_accepted = _accepted_lab_demo(row)
                geometry = _trajectory_geometry(
                    arrays["dense_positions"],
                    env.start,
                    env.goal,
                    np.asarray(scene.cylinders, np.float32),
                )
                path = _attempt_path(
                    output, gamma, pair_index, member_index,
                )
                row = {
                    **row,
                    **scene.as_manifest_row(),
                    **pair_meta,
                    "pair_member_index": member_index,
                    "pair_member": "source" if member_index == 0 else "mirror",
                    "obstacle_count": len(scene.cylinders),
                    "individual_accepted": individual_accepted,
                    "trajectory_accepted": individual_accepted,
                    "pair_admitted": False,
                    "accepted": False,
                    "file": str(path.relative_to(output)),
                    **geometry,
                }
                _atomic_npz(path, {
                    **arrays,
                    **_scene_arrays(scene),
                    "attempt_row_json": np.asarray(
                        json.dumps(row, sort_keys=True, allow_nan=False),
                    ),
                })
                attempts[key] = row
            admitted = reconcile_pair(gamma_index, pair_index)
            if admitted:
                accepted_count += 2
            print(
                "[mirrored-pair] "
                f"gamma={gamma:g} pair={pair_index} "
                f"members={int(attempts[(gamma_index, pair_index, 0)]['individual_accepted'])}/"
                f"{int(attempts[(gamma_index, pair_index, 1)]['individual_accepted'])} "
                f"pair_admitted={bool(admitted)} quota={accepted_count}/{target}",
                flush=True,
            )
            _write_progress(output, gamma_values, attempts, target)

    counts = {
        gamma_index: sum(
            bool(row.get("pair_admitted"))
            for (observed_gamma, _, _), row in attempts.items()
            if observed_gamma == gamma_index
        )
        for gamma_index in range(len(gamma_values))
    }
    all_rows = [attempts[key] for key in sorted(attempts)]
    runs = [row for row in all_rows if row.get("pair_admitted")]
    complete = all(counts[index] == target for index in counts)

    hashes_by_gamma = {}
    observed_hashes: set[str] = set()
    for gamma_index, gamma in enumerate(gamma_values):
        hashes = {
            str(row["scene_hash"])
            for (observed_gamma, _, _), row in attempts.items()
            if observed_gamma == gamma_index
        }
        if observed_hashes.intersection(hashes):
            raise RuntimeError("a randomized scene was reused across gamma")
        observed_hashes.update(hashes)
        hashes_by_gamma[f"{gamma:.9g}"] = len(hashes)

    pair_rows = []
    for gamma_index, gamma in enumerate(gamma_values):
        pair_indices = sorted({
            pair_index
            for observed_gamma, pair_index, _ in attempts
            if observed_gamma == gamma_index
        })
        for pair_index in pair_indices:
            members = [
                attempts.get((gamma_index, pair_index, member))
                for member in (0, 1)
            ]
            pair_rows.append({
                "pair_id": members[0]["pair_id"],
                "pair_index": pair_index,
                "gamma": gamma,
                "member_scene_hashes": [
                    member["scene_hash"] for member in members if member
                ],
                "member_individual_accepted": [
                    bool(member["individual_accepted"])
                    for member in members if member
                ],
                "pair_admitted": bool(
                    len(members) == 2
                    and all(member is not None for member in members)
                    and all(member["individual_accepted"] for member in members)
                ),
            })

    metrics = aggregate_metrics(all_rows, gamma_values)
    behavior = _behavior_summary(all_rows, gamma_values)
    manifest = {
        "kind": (
            "Minhyuk lab gamma-exclusive mirrored-pair randomized-cylinder "
            "SafeMPPI demonstrations"
        ),
        "schema_version": 1,
        "schema": MIRRORED_PAIR_SCHEMA,
        "status": (
            "COMPLETE_EXACT_PAIRED_SUCCESS_QUOTA"
            if complete else "FAILED_MAX_PAIR_BUDGET"
        ),
        "config": "resolved_config.json",
        "quota_contract": "pair_quota_contract.json",
        "rollout_dynamics": template.data.rollout_dynamics,
        "acceptance": template.data.acceptance,
        "target_successes_per_gamma": target,
        "max_pair_attempts_per_gamma": pair_budget,
        "accepted_counts_by_gamma": {
            f"{gamma_values[index]:.9g}": counts[index]
            for index in range(len(gamma_values))
        },
        "gammas": list(gamma_values),
        "taskspace": {
            "bounds": template.taskspace.bounds.tolist(),
            "z_symmetry_plane_m": 0.9,
            "z_distance_each_side_m": 0.8,
        },
        "sampling_distribution": {
            "proposal": spec.scene_schema,
            "gamma_exclusive_scene_streams": True,
            "scene_reuse_across_gamma": False,
            "expert_rollouts_per_pair": 2,
            "rollout_retries_on_same_pair": 0,
            "failed_pair_training_contribution": 0,
            "successful_pair_training_contribution": 2,
            "mirror_axis": "infinite xy line through fixed start and goal",
            "unconditioned_geometry": True,
            "expert_success_used_only_for_pair_admission": True,
        },
        "scene_bank": {
            "shared_across_gamma": False,
            "unique_scene_hash_counts_by_gamma": hashes_by_gamma,
            "domain_seed": domain_seed_value,
            "pairs": pair_rows,
            "scenes": [
                {
                    "scene_index": int(row["scene_index"]),
                    "scene_id": str(row["scene_id"]),
                    "scene_seed": int(row["scene_seed"]),
                    "scene_hash": str(row["scene_hash"]),
                    "spheres": row["spheres"],
                    "cylinders": row["cylinders"],
                    "gamma": float(row["gamma"]),
                    "pair_id": str(row["pair_id"]),
                    "pair_member": str(row["pair_member"]),
                }
                for row in all_rows
            ],
        },
        "runs": runs,
        "attempts": all_rows,
        "metrics": metrics,
        "behavior_metrics": behavior,
    }
    if not complete:
        _atomic_json(output / "FAILED_collection.json", manifest)
        raise RuntimeError(
            "paired success quota was not reached within the pair budget: "
            f"{manifest['accepted_counts_by_gamma']}"
        )
    _atomic_json(output / "metrics.json", {
        "expert_outcomes": metrics,
        "behavior": behavior,
    })
    _atomic_json(manifest_path, manifest)
    return manifest
