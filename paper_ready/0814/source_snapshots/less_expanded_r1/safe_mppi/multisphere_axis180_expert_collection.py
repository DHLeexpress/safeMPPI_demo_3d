"""Resumable SafeMPPI expert collection on paired dense-z sphere scenes.

Each gamma owns a deterministic stream of randomized six-sphere scenes.  One
scene and its exact 180-degree rotation about the three-dimensional start-goal
axis form an admission pair.  Both members are rolled out once with SafeMPPI;
the pair contributes exactly two trajectories only when both members satisfy
the existing nominal-safe-success acceptance rule.

This module is intentionally isolated from the older cylinder/XY-reflection
collector.  In particular, ``axis_180`` flips both transverse dimensions,
including height about the start-goal axis, while the older reflection leaves
height unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .acquire import aggregate_metrics, run_episode
from .config import ExperimentConfig, ObstacleConfig, validate_config
from .controller import Mode1SafeMPPI
from .environment import TaskEnvironment
from .lab_clutter import (
    _accepted_lab_demo,
    _scene_arrays,
    config_for_scene,
    obstacle_scene_hash,
)
from .lab_clutter_expansion import PathFocusedVariableSphereScene
from .lab_clutter_pre2_expansion import (
    rotate_points_180_about_start_goal_axis,
)
from .path_focused_clutter import (
    PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION,
    PathFocusedClutterSpec,
)
from .path_focused_collection import (
    _atomic_json,
    _atomic_npz,
    _behavior_summary,
    path_focused_collection_config,
)


AXIS180_EXPERT_PAIR_SCHEMA = "multisphere_axis180_expert_pairs_v1"
EXPECTED_BOUNDS = np.asarray(
    [[-2.5, 1.3], [-2.1, 1.8], [0.1, 1.7]], np.float64,
)
EXPECTED_START = np.asarray(
    [-2.1, 1.5, 0.9, 0.0, 0.0, 0.0], np.float64,
)
EXPECTED_GOAL = np.asarray([0.7, -1.5, 0.9], np.float64)
EXPECTED_SPHERE_COUNT = 6
EXPECTED_PHYSICAL_RADIUS_M = 0.1905
EXPECTED_VEHICLE_INFLATION_M = 0.05
EXPECTED_MODELED_RADIUS_M = 0.2405
EXPECTED_Z_CENTER_RANGE_M = (0.7, 1.1)


@dataclass(frozen=True)
class Axis180SphereScene:
    """One immutable sphere scene with an unambiguous sphere scene id."""

    index: int
    seed: int
    spheres: tuple[tuple[float, float, float, float], ...]
    cylinders: tuple[tuple[float, float, float], ...]
    scene_hash: str

    @property
    def scene_id(self) -> str:
        return f"spheres_{self.index:012d}"

    def as_manifest_row(self) -> dict:
        return {
            "scene_index": int(self.index),
            "scene_id": self.scene_id,
            "scene_seed": int(self.seed),
            "scene_hash": self.scene_hash,
            "spheres": [list(row) for row in self.spheres],
            "cylinders": [],
        }


def _gamma_key(gamma: float) -> int:
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


def _validate_dense_relaxed_contract(config: ExperimentConfig) -> None:
    bounds = np.asarray(config.taskspace.bounds, np.float64)
    start = np.asarray(config.taskspace.start, np.float64)
    goal = np.asarray(config.taskspace.goal, np.float64)
    if not np.allclose(bounds, EXPECTED_BOUNDS, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "axis-180 expert collection requires taskspace "
            "x=[-2.5,1.3], y=[-2.1,1.8], z=[0.1,1.7]"
        )
    if not np.allclose(start, EXPECTED_START, rtol=0.0, atol=1.0e-6):
        raise ValueError("axis-180 expert collection requires the fixed lab start")
    if not np.allclose(goal, EXPECTED_GOAL, rtol=0.0, atol=1.0e-6):
        raise ValueError("axis-180 expert collection requires the fixed lab goal")
    if not np.isclose(
        start[2] - bounds[2, 0], bounds[2, 1] - start[2],
        rtol=0.0, atol=1.0e-12,
    ):
        raise ValueError("taskspace z bounds must be symmetric about z=0.9")
    if config.data.rollout_dynamics != "minhyuk_reference_governor":
        raise ValueError("axis-180 expert collection requires the Minhyuk governor")
    if config.data.acceptance != "nominal_safe_success":
        raise ValueError("axis-180 expert collection requires safe-success acceptance")
    if config.safemppi.z_bias_weight != 0.0:
        raise ValueError("axis-180 expert collection forbids demonstration z bias")
    if config.safemppi.robot_radius != 0.0:
        raise ValueError(
            "sphere rows already include vehicle inflation; robot_radius must be zero"
        )

    spec = PathFocusedClutterSpec.from_config(
        config, expected_family="spheres",
    )
    if spec.distribution != PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION:
        raise ValueError("axis-180 expert collection requires the double-hourglass law")
    if (spec.count_min, spec.count_max) != (
        EXPECTED_SPHERE_COUNT, EXPECTED_SPHERE_COUNT,
    ):
        raise ValueError("axis-180 expert collection requires exactly six spheres")
    expected_scalars = (
        ("physical radius", spec.physical_radius_m, EXPECTED_PHYSICAL_RADIUS_M),
        ("vehicle inflation", spec.vehicle_inflation_m, EXPECTED_VEHICLE_INFLATION_M),
        ("modeled radius", spec.modeled_radius_m, EXPECTED_MODELED_RADIUS_M),
    )
    for name, observed, expected in expected_scalars:
        if not np.isclose(observed, expected, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"unexpected dense-z sphere {name}: {observed}")
    if not np.allclose(
        [spec.sphere_z_lower_m, spec.sphere_z_upper_m],
        EXPECTED_Z_CENTER_RANGE_M,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("dense-z sphere centers must use z range [0.7,1.1]")


def axis180_expert_collection_config(
    base_config: ExperimentConfig,
    *,
    gammas: tuple[float, ...] | None = None,
    target_trajectories_per_gamma: int = 50,
    max_pairs_per_gamma: int,
) -> ExperimentConfig:
    """Return the validated obstacle-free template for expert collection."""
    _validate_dense_relaxed_contract(base_config)
    target = int(target_trajectories_per_gamma)
    pair_budget = int(max_pairs_per_gamma)
    if target < 2 or target % 2:
        raise ValueError("target trajectories per gamma must be positive and even")
    if pair_budget < target // 2:
        raise ValueError("max pairs per gamma cannot be smaller than target/2")

    configured = path_focused_collection_config(
        base_config,
        gammas=gammas,
        episodes_per_gamma=target,
        max_attempts_per_gamma=2 * pair_budget,
    )
    bounds = np.asarray(configured.taskspace.bounds, np.float64)
    raw = copy.deepcopy(configured.raw)
    raw["obstacles"] = {"spheres": [], "cylinders": []}
    raw["safety"] = {
        "safe_min": np.round(bounds[:, 0], 12).tolist(),
        "safe_max": np.round(bounds[:, 1], 12).tolist(),
    }
    raw["data"] = dict(raw["data"])
    raw["data"].update({
        "episodes_per_gamma": target,
        "max_attempts_per_gamma": 2 * pair_budget,
    })
    raw["expert_demo_collection"] = {
        "schema": AXIS180_EXPERT_PAIR_SCHEMA,
        "obstacle_geometry_source": "per_attempt_npz_and_manifest",
        "gamma_exclusive_scene_streams": True,
        "scene_bank_shared_across_gamma": False,
        "pair_transform": "start_goal_axis_180",
        "pair_admission": "both_nominal_safe_success=>2; otherwise=>0",
        "static_obstacles_in_resolved_config": False,
    }
    normalized = ExperimentConfig(
        configured.taskspace,
        ObstacleConfig(spheres=(), cylinders=()),
        configured.safemppi,
        replace(
            configured.data,
            episodes_per_gamma=target,
            max_attempts_per_gamma=2 * pair_budget,
        ),
        raw,
    )
    validate_config(normalized)
    _validate_dense_relaxed_contract(normalized)
    return normalized


def _sphere_rows(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    rows = np.asarray(values, np.float32).reshape(-1, 4)
    return tuple(tuple(map(float, row)) for row in rows)


def _sphere_scene(
    *,
    index: int,
    seed: int,
    spheres: np.ndarray,
) -> Axis180SphereScene:
    rows = _sphere_rows(spheres)
    return Axis180SphereScene(
        index=int(index),
        seed=int(seed),
        spheres=rows,
        cylinders=(),
        scene_hash=obstacle_scene_hash(spheres=rows),
    )


def sample_gamma_exclusive_axis180_pair(
    config: ExperimentConfig,
    *,
    gamma: float,
    pair_index: int,
    domain_seed: int,
    max_geometry_proposals: int = 10_000,
) -> tuple[Axis180SphereScene, Axis180SphereScene, dict]:
    """Deterministically sample one source/axis-180 sphere pair."""
    if pair_index < 0 or max_geometry_proposals < 1:
        raise ValueError("pair index must be nonnegative and budget positive")
    spec = PathFocusedClutterSpec.from_config(
        config, expected_family="spheres",
    )
    validator = PathFocusedVariableSphereScene.from_config(config)
    empty_env = TaskEnvironment(config)
    gamma_key = _gamma_key(gamma)
    start = np.asarray(config.taskspace.start, np.float64)
    goal = np.asarray(config.taskspace.goal, np.float64)
    bounds = np.asarray(config.taskspace.bounds, np.float64)
    base_index = gamma_key * 1_000_000 + 2 * int(pair_index)
    for proposal_index in range(int(max_geometry_proposals)):
        source_seed = _seed32(
            int(domain_seed), gamma_key, int(pair_index), proposal_index, 0xA180,
        )
        sampled = spec.sample_scene(
            scene_index=base_index,
            scene_seed=source_seed,
            bounds=bounds,
            start=start,
            goal=goal,
        )
        source_values = validator.validate(
            empty_env, np.asarray(sampled.spheres, np.float32),
        )
        rotated_values = source_values.copy()
        rotated_values[:, :3] = rotate_points_180_about_start_goal_axis(
            source_values[:, :3], start, goal,
        ).astype(np.float32)
        try:
            rotated_values = validator.validate(empty_env, rotated_values)
        except ValueError:
            continue
        mirror_seed = _seed32(source_seed, 1, 0xB180)
        source = _sphere_scene(
            index=base_index, seed=source_seed, spheres=source_values,
        )
        axis_180 = _sphere_scene(
            index=base_index + 1, seed=mirror_seed, spheres=rotated_values,
        )
        if source.scene_hash == axis_180.scene_hash:
            continue

        recovered = axis_180.spheres
        recovered_values = np.asarray(recovered, np.float32)
        recovered_values[:, :3] = rotate_points_180_about_start_goal_axis(
            recovered_values[:, :3], start, goal,
        ).astype(np.float32)
        recovered_values = validator.validate(empty_env, recovered_values)
        if not np.allclose(
            recovered_values,
            np.asarray(source.spheres, np.float32),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise RuntimeError("axis-180 sphere rotation failed involution")
        pair_id = f"g{float(gamma):g}_axis180_pair_{pair_index:06d}"
        return source, axis_180, {
            "pair_id": pair_id,
            "pair_index": int(pair_index),
            "gamma": float(gamma),
            "gamma_key": gamma_key,
            "geometry_proposal_index": int(proposal_index),
            "domain_seed": int(domain_seed),
            "pair_transform": "start_goal_axis_180",
            "rotation_axis_start_xyz": start[:3].tolist(),
            "rotation_axis_goal_xyz": goal[:3].tolist(),
            "member_scene_hashes": [source.scene_hash, axis_180.scene_hash],
        }
    raise RuntimeError(
        f"could not sample a valid axis-180 pair for gamma={gamma:g}, "
        f"pair_index={pair_index}"
    )


def _trajectory_geometry(
    dense_positions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    spheres: np.ndarray,
    *,
    interaction_distance_m: float = 0.30,
) -> dict:
    path = np.asarray(dense_positions, np.float64).reshape(-1, 3)
    start3 = np.asarray(start, np.float64).reshape(-1)[:3]
    goal3 = np.asarray(goal, np.float64).reshape(-1)[:3]
    displacement = np.diff(path, axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    path_length = float(lengths.sum())
    direct = goal3 - start3
    direct_length = float(np.linalg.norm(direct))
    realized_displacement = float(np.linalg.norm(path[-1] - path[0]))
    direction = direct / direct_length
    along = (path - start3[None]) @ direction
    closest = start3[None] + along[:, None] * direction[None]
    transverse = path - closest
    transverse_rms = float(np.sqrt(np.mean(np.sum(
        transverse * transverse, axis=1,
    ))))

    horizontal = direction[:2]
    horizontal /= np.linalg.norm(horizontal)
    normal = np.asarray([-horizontal[1], horizontal[0]], np.float64)
    signed = (path[:, :2] - start3[None, :2]) @ normal
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

    sphere_rows = np.asarray(spheres, np.float64).reshape(-1, 4)
    interaction_count = 0
    if len(sphere_rows):
        surface_distance = (
            np.linalg.norm(
                path[:, None, :] - sphere_rows[None, :, :3], axis=2,
            )
            - sphere_rows[None, :, 3]
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


def _attempt_path(
    output: Path,
    gamma: float,
    pair_index: int,
    member_index: int,
) -> Path:
    return output / "attempt_shards" / (
        f"g{gamma:g}_pair_{pair_index:06d}_member_{member_index}.npz"
    )


def _load_attempt(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["attempt_row_json"].item()))


def _write_progress(
    output: Path,
    gamma_values: tuple[float, ...],
    attempts: dict[tuple[int, int, int], dict],
    target: int,
) -> None:
    accepted = {}
    attempted_pairs = {}
    for gamma_index, gamma in enumerate(gamma_values):
        rows = [
            row for (observed_gamma, _, _), row in attempts.items()
            if observed_gamma == gamma_index
        ]
        accepted[f"{gamma:.9g}"] = sum(
            bool(row.get("pair_admitted")) for row in rows
        )
        attempted_pairs[f"{gamma:.9g}"] = len({
            int(row["pair_index"]) for row in rows
        })
    _atomic_json(output / "progress.json", {
        "schema": AXIS180_EXPERT_PAIR_SCHEMA,
        "target_trajectories_per_gamma": int(target),
        "target_pairs_per_gamma": int(target // 2),
        "accepted_trajectories_by_gamma": accepted,
        "attempted_pairs_by_gamma": attempted_pairs,
    })


def collect_multisphere_axis180_expert_demos(
    base_config: ExperimentConfig,
    output_dir: str | Path,
    *,
    target_trajectories_per_gamma: int = 50,
    max_pairs_per_gamma: int,
    gammas: tuple[float, ...] | None = None,
    domain_seed: int | None = None,
    rollout_seed_start: int | None = None,
    device: str = "cpu",
    episode_runner=run_episode,
    controller_factory=Mode1SafeMPPI,
) -> dict:
    """Resume until every gamma reaches its exact paired-success quota."""
    target = int(target_trajectories_per_gamma)
    pair_budget = int(max_pairs_per_gamma)
    template = axis180_expert_collection_config(
        base_config,
        gammas=gammas,
        target_trajectories_per_gamma=target,
        max_pairs_per_gamma=pair_budget,
    )
    spec = PathFocusedClutterSpec.from_config(
        template, expected_family="spheres",
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
        "schema": AXIS180_EXPERT_PAIR_SCHEMA,
        "target_trajectories_per_gamma": target,
        "target_pairs_per_gamma": target // 2,
        "max_pairs_per_gamma": pair_budget,
        "gammas": list(gamma_values),
        "domain_seed": domain_seed_value,
        "rollout_seed_start": rollout_seed0,
        "pair_transform": "start_goal_axis_180",
        "resolved_config": template.raw,
    }
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("existing axis-180 expert contract does not match")
        if not resolved_path.is_file():
            raise FileNotFoundError("resumable archive lacks resolved_config.json")
        if json.loads(resolved_path.read_text()) != template.raw:
            raise ValueError("resumable archive resolved config changed")
    else:
        if any(output.iterdir()):
            raise FileExistsError(
                f"nonempty output lacks an axis-180 expert contract: {output}"
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
                row = _load_attempt(path)
                if row.get("schema") != AXIS180_EXPERT_PAIR_SCHEMA:
                    raise ValueError(f"attempt shard schema mismatch: {path}")
                if (
                    not np.isclose(
                        row["gamma"], gamma, rtol=0.0, atol=1.0e-9,
                    )
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
        admitted = all(
            bool(attempts[key]["individual_accepted"]) for key in keys
        )
        for key in keys:
            attempts[key]["pair_admitted"] = admitted
            attempts[key]["accepted"] = admitted
        return admitted

    def accepted_trajectory_count(gamma_index: int) -> int:
        return sum(
            bool(row.get("pair_admitted"))
            for (observed_gamma, _, _), row in attempts.items()
            if observed_gamma == gamma_index
        )

    for gamma_index in range(len(gamma_values)):
        for pair_index in range(pair_budget):
            reconcile_pair(gamma_index, pair_index)

    _write_progress(output, gamma_values, attempts, target)
    for gamma_index, gamma in enumerate(gamma_values):
        accepted_count = accepted_trajectory_count(gamma_index)
        for pair_index in range(pair_budget):
            if accepted_count == target:
                break
            original, axis_180, pair_meta = (
                sample_gamma_exclusive_axis180_pair(
                    template,
                    gamma=gamma,
                    pair_index=pair_index,
                    domain_seed=domain_seed_value,
                )
            )
            for member_index, scene in enumerate((original, axis_180)):
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
                    0xC180,
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
                    np.asarray(scene.spheres, np.float32),
                )
                path = _attempt_path(
                    output, gamma, pair_index, member_index,
                )
                row = {
                    **row,
                    **scene.as_manifest_row(),
                    **pair_meta,
                    "schema": AXIS180_EXPERT_PAIR_SCHEMA,
                    "pair_member_index": member_index,
                    "pair_member": (
                        "original" if member_index == 0 else "axis_180"
                    ),
                    "obstacle_count": len(scene.spheres),
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
                _write_progress(output, gamma_values, attempts, target)
            admitted = reconcile_pair(gamma_index, pair_index)
            accepted_count = accepted_trajectory_count(gamma_index)
            print(
                "[axis180-expert] "
                f"gamma={gamma:g} pair={pair_index} "
                f"members="
                f"{int(attempts[(gamma_index, pair_index, 0)]['individual_accepted'])}/"
                f"{int(attempts[(gamma_index, pair_index, 1)]['individual_accepted'])} "
                f"pair_admitted={bool(admitted)} "
                f"quota={accepted_count}/{target}",
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
            raise RuntimeError("a randomized sphere scene was reused across gamma")
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
                attempts.get((gamma_index, pair_index, member_index))
                for member_index in (0, 1)
            ]
            present = [member for member in members if member is not None]
            pair_rows.append({
                "pair_id": present[0]["pair_id"],
                "pair_index": pair_index,
                "gamma": gamma,
                "pair_transform": "start_goal_axis_180",
                "member_scene_hashes": [
                    member["scene_hash"] for member in present
                ],
                "member_individual_accepted": [
                    bool(member["individual_accepted"]) for member in present
                ],
                "pair_admitted": bool(
                    len(present) == 2
                    and all(member["individual_accepted"] for member in present)
                ),
            })

    metrics = aggregate_metrics(all_rows, gamma_values)
    behavior = _behavior_summary(all_rows, gamma_values)
    manifest = {
        "kind": (
            "Minhyuk lab gamma-exclusive axis-180 paired randomized-sphere "
            "SafeMPPI expert demonstrations"
        ),
        "schema_version": 1,
        "schema": AXIS180_EXPERT_PAIR_SCHEMA,
        "status": (
            "COMPLETE_EXACT_AXIS180_EXPERT_PAIR_QUOTA"
            if complete else "FAILED_MAX_PAIR_BUDGET"
        ),
        "config": "resolved_config.json",
        "quota_contract": "pair_quota_contract.json",
        "rollout_dynamics": template.data.rollout_dynamics,
        "acceptance": template.data.acceptance,
        "target_trajectories_per_gamma": target,
        "target_pairs_per_gamma": target // 2,
        "max_pairs_per_gamma": pair_budget,
        "accepted_counts_by_gamma": {
            f"{gamma_values[index]:.9g}": counts[index]
            for index in range(len(gamma_values))
        },
        "accepted_pair_counts_by_gamma": {
            f"{gamma_values[index]:.9g}": counts[index] // 2
            for index in range(len(gamma_values))
        },
        "gammas": list(gamma_values),
        "taskspace": {
            "bounds": template.taskspace.bounds.tolist(),
            "rotation_axis_start_xyz": list(map(
                float, template.taskspace.start[:3],
            )),
            "rotation_axis_goal_xyz": list(map(
                float, template.taskspace.goal,
            )),
        },
        "sampling_distribution": {
            "proposal": spec.scene_schema,
            "unconditioned_geometry": True,
            "gamma_exclusive_scene_streams": True,
            "scene_reuse_across_gamma": False,
            "expert_rollouts_per_pair": 2,
            "rollout_retries_on_same_pair": 0,
            "failed_pair_training_contribution": 0,
            "successful_pair_training_contribution": 2,
            "pair_transform": "start_goal_axis_180",
            "expert_success_used_for_pair_admission": True,
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
                    "cylinders": [],
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
            "axis-180 expert pair quota was not reached within max pairs: "
            f"{manifest['accepted_counts_by_gamma']}"
        )
    _atomic_json(output / "metrics.json", {
        "expert_outcomes": metrics,
        "behavior": behavior,
    })
    _atomic_json(manifest_path, manifest)
    return manifest
