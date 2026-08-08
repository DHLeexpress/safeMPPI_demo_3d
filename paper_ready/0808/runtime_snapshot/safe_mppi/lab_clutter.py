"""Deterministic Minhyuk-lab cylinder scenes and SafeMPPI demo collection.

The checked-in controller/configuration stack models a vertical cylinder as
``[x, y, radius]`` spanning the full taskspace height.  This module keeps that
contract explicit, fixes the lab task geometry, and stores the randomized scene
with every trajectory so downstream visual-context loaders need not guess which
obstacles generated a run.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .acquire import aggregate_metrics, run_episode
from .config import ExperimentConfig, ObstacleConfig
from .controller import Mode1SafeMPPI
from .environment import TaskEnvironment


LAB_BOUNDS = np.asarray(
    [[-2.5, 1.3], [-1.7, 1.8], [0.4, 2.0]], dtype=np.float64,
)
LAB_START = np.asarray(
    [-2.1, 1.5, 0.9, 0.0, 0.0, 0.0], dtype=np.float64,
)
LAB_GOAL = np.asarray([0.7, -1.5, 0.9], dtype=np.float64)
DEFAULT_CYLINDER_COUNT = 3
DEFAULT_CYLINDER_PHYSICAL_RADIUS_M = 0.10
DEFAULT_VEHICLE_INFLATION_M = 0.125
DEFAULT_CYLINDER_RADIUS_M = (
    DEFAULT_CYLINDER_PHYSICAL_RADIUS_M + DEFAULT_VEHICLE_INFLATION_M
)
DEFAULT_MIN_SURFACE_GAP_M = 0.20


def _rows(values: np.ndarray, width: int) -> tuple[tuple[float, ...], ...]:
    # TaskEnvironment and the NPZ contract both use float32 obstacle arrays.
    # Hash that modeled/stored geometry, not sampler-only float64 tail bits.
    array = np.asarray(values, dtype=np.float32).reshape(-1, width)
    return tuple(tuple(map(float, row)) for row in array)


def obstacle_scene_hash(
    spheres: np.ndarray | tuple = (),
    cylinders: np.ndarray | tuple = (),
) -> str:
    """Return a stable SHA-256 over the exact modeled obstacle geometry."""
    payload = {
        "cylinders": [list(row) for row in _rows(cylinders, 3)],
        "cylinder_semantics": "full_height_vertical_[x,y,radius]_m",
        "spheres": [list(row) for row in _rows(spheres, 4)],
        "sphere_semantics": "[x,y,z,radius]_m",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ClutterScene:
    """One immutable randomized obstacle world."""

    index: int
    seed: int
    spheres: tuple[tuple[float, float, float, float], ...]
    cylinders: tuple[tuple[float, float, float], ...]
    scene_hash: str

    @property
    def scene_id(self) -> str:
        return f"cylinders_{self.index:05d}"

    def as_manifest_row(self) -> dict:
        return {
            "scene_index": int(self.index),
            "scene_id": self.scene_id,
            "scene_seed": int(self.seed),
            "scene_hash": self.scene_hash,
            "spheres": [list(row) for row in self.spheres],
            "cylinders": [list(row) for row in self.cylinders],
        }


def _validate_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _point_to_segment_distance(
    points: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Euclidean distance from each point to the closed start-goal segment."""
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    direction = goal - start
    squared_length = float(direction @ direction)
    if squared_length <= 0.0:
        return np.linalg.norm(points - start[None], axis=1)
    fraction = np.clip(
        ((points - start[None]) @ direction) / squared_length,
        0.0,
        1.0,
    )
    closest = start[None] + fraction[:, None] * direction[None]
    return np.linalg.norm(points - closest, axis=1)


def start_goal_path_diagnostics(
    scene: ClutterScene,
    *,
    start: np.ndarray = LAB_START,
    goal: np.ndarray = LAB_GOAL,
    soft_clearance_target_m: float,
) -> dict:
    """Describe obstacle relevance to the straight start-goal centerline.

    The distance is measured to the modeled obstacle surfaces: in 3-D for
    spheres and in x-y for full-height vertical cylinders. These values are
    diagnostics only and never participate in scene sampling or acceptance.
    """
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    if len(start) < 3 or len(goal) < 3:
        raise ValueError("start and goal must each provide at least 3 coordinates")
    start = start[:3]
    goal = goal[:3]
    soft_target = _validate_nonnegative(
        "soft_clearance_target_m", soft_clearance_target_m,
    )

    surface_distances = []
    spheres = np.asarray(scene.spheres, dtype=np.float64).reshape(-1, 4)
    if len(spheres):
        surface_distances.extend(
            _point_to_segment_distance(spheres[:, :3], start, goal)
            - spheres[:, 3]
        )
    cylinders = np.asarray(scene.cylinders, dtype=np.float64).reshape(-1, 3)
    if len(cylinders):
        surface_distances.extend(
            _point_to_segment_distance(
                cylinders[:, :2], start[:2], goal[:2],
            )
            - cylinders[:, 2]
        )

    minimum = (
        float(np.min(surface_distances)) if surface_distances else None
    )
    return {
        "minimum_obstacle_surface_distance_m": minimum,
        "modeled_hard_path_intersection": bool(
            minimum is not None and minimum <= 0.0
        ),
        "within_soft_clearance_tube": bool(
            minimum is not None and minimum <= soft_target
        ),
        "soft_clearance_target_m": soft_target,
        "used_for_scene_filtering": False,
    }


def summarize_start_goal_path_diagnostics(
    diagnostics: list[dict] | tuple[dict, ...],
) -> dict:
    """Aggregate scene-level path diagnostics without changing the scene bank."""
    if not diagnostics:
        raise ValueError("at least one scene diagnostic is required")
    targets = {
        float(row["soft_clearance_target_m"]) for row in diagnostics
    }
    if len(targets) != 1:
        raise ValueError("scene diagnostics use inconsistent clearance targets")
    distances = [
        float(row["minimum_obstacle_surface_distance_m"])
        for row in diagnostics
        if row["minimum_obstacle_surface_distance_m"] is not None
    ]
    hard_intersection_count = sum(
        bool(row["modeled_hard_path_intersection"]) for row in diagnostics
    )
    soft_count = sum(
        bool(row["within_soft_clearance_tube"]) for row in diagnostics
    )
    count = len(diagnostics)
    return {
        "definition": (
            "modeled obstacle-surface distance to the closed straight "
            "start-goal centerline segment"
        ),
        "scene_count": count,
        "soft_clearance_target_m": targets.pop(),
        "minimum_obstacle_surface_distance_m": {
            "minimum": float(np.min(distances)) if distances else None,
            "mean": float(np.mean(distances)) if distances else None,
            "maximum": float(np.max(distances)) if distances else None,
        },
        "modeled_hard_path_intersection_scene_count": int(
            hard_intersection_count
        ),
        "modeled_hard_path_intersection_scene_fraction": (
            hard_intersection_count / count
        ),
        "within_soft_clearance_tube_scene_count": int(soft_count),
        "within_soft_clearance_tube_scene_fraction": soft_count / count,
        "used_for_scene_filtering": False,
    }


def cylinder_scene_bank(
    count: int,
    *,
    seed: int = 0,
    bounds: np.ndarray = LAB_BOUNDS,
    start: np.ndarray = LAB_START,
    goal: np.ndarray = LAB_GOAL,
    cylinder_count: int = DEFAULT_CYLINDER_COUNT,
    cylinder_radius_m: float = DEFAULT_CYLINDER_RADIUS_M,
    min_surface_gap_m: float = DEFAULT_MIN_SURFACE_GAP_M,
    endpoint_surface_gap_m: float = DEFAULT_MIN_SURFACE_GAP_M,
    start_surface_gap_m: float | None = None,
    goal_surface_gap_m: float | None = None,
    boundary_surface_gap_m: float = 0.0,
    max_layout_attempts: int = 20_000,
) -> tuple[ClutterScene, ...]:
    """Sample a deterministic bank shared unchanged across every gamma.

    Pairwise and endpoint gaps are measured surface-to-surface using the modeled
    cylinder radius.  ``boundary_surface_gap_m`` is additional free space beyond
    fitting the complete circular cross-section inside the x-y task bounds.
    """
    if count < 1 or cylinder_count < 1 or max_layout_attempts < 1:
        raise ValueError(
            "count, cylinder_count, and max_layout_attempts must be positive"
        )
    # Obstacles enter TaskEnvironment as float32; enforce spacing on those exact
    # modeled coordinates so serialization cannot erode a requested gap.
    radius = float(np.float32(cylinder_radius_m))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("cylinder_radius_m must be finite and positive")
    pair_gap = _validate_nonnegative(
        "min_surface_gap_m", min_surface_gap_m,
    )
    endpoint_gap = _validate_nonnegative(
        "endpoint_surface_gap_m", endpoint_surface_gap_m,
    )
    start_gap = (
        endpoint_gap
        if start_surface_gap_m is None
        else _validate_nonnegative(
            "start_surface_gap_m", start_surface_gap_m,
        )
    )
    goal_gap = (
        endpoint_gap
        if goal_surface_gap_m is None
        else _validate_nonnegative(
            "goal_surface_gap_m", goal_surface_gap_m,
        )
    )
    boundary_gap = _validate_nonnegative(
        "boundary_surface_gap_m", boundary_surface_gap_m,
    )
    bounds = np.asarray(bounds, dtype=np.float64).reshape(3, 2)
    start = np.asarray(start, dtype=np.float64).reshape(6)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    if not np.allclose(bounds, LAB_BOUNDS, rtol=0.0, atol=1.0e-9):
        raise ValueError("clutter scenes require the fixed Minhyuk lab bounds")
    if not np.allclose(start, LAB_START, rtol=0.0, atol=1.0e-9):
        raise ValueError("clutter scenes require the fixed lab start")
    if not np.allclose(goal, LAB_GOAL, rtol=0.0, atol=1.0e-9):
        raise ValueError("clutter scenes require the fixed lab goal")

    low = bounds[:2, 0] + radius + boundary_gap
    high = bounds[:2, 1] - radius - boundary_gap
    if np.any(low >= high):
        raise ValueError("cylinder radius/boundary gap leaves no sampling area")

    scenes = []
    for scene_index in range(int(count)):
        scene_seed = int(
            np.random.SeedSequence([int(seed), scene_index])
            .generate_state(1, dtype=np.uint32)[0]
        )
        rng = np.random.default_rng(scene_seed)
        centers: list[np.ndarray] = []
        for _ in range(int(max_layout_attempts)):
            if len(centers) == cylinder_count:
                break
            candidate = rng.uniform(low, high).astype(np.float32).astype(np.float64)
            if (
                np.linalg.norm(candidate - start[:2]) - radius < start_gap
                or np.linalg.norm(candidate - goal[:2]) - radius < goal_gap
            ):
                continue
            if any(
                np.linalg.norm(candidate - center) - 2.0 * radius < pair_gap
                for center in centers
            ):
                continue
            centers.append(candidate)
        if len(centers) != cylinder_count:
            raise RuntimeError(
                f"could not place {cylinder_count} cylinders for scene "
                f"{scene_index} in {max_layout_attempts} proposals"
            )
        cylinders = tuple(
            (float(center[0]), float(center[1]), radius)
            for center in centers
        )
        scenes.append(ClutterScene(
            index=scene_index,
            seed=scene_seed,
            spheres=(),
            cylinders=cylinders,
            scene_hash=obstacle_scene_hash(cylinders=cylinders),
        ))
    return tuple(scenes)


def fixed_lab_clutter_config(
    config: ExperimentConfig,
    *,
    episodes_per_gamma: int | None = None,
) -> ExperimentConfig:
    """Return a static-free template with exact lab geometry and zero z bias."""
    if not np.allclose(
        config.taskspace.bounds, LAB_BOUNDS, rtol=0.0, atol=1.0e-9,
    ):
        raise ValueError("base config taskspace does not match the lab bounds")
    if not np.allclose(
        config.taskspace.start, LAB_START, rtol=0.0, atol=1.0e-6,
    ):
        raise ValueError("base config start does not match the fixed lab start")
    if not np.allclose(
        config.taskspace.goal, LAB_GOAL, rtol=0.0, atol=1.0e-6,
    ):
        raise ValueError("base config goal does not match the fixed lab goal")
    if config.data.rollout_dynamics != "minhyuk_reference_governor":
        raise ValueError("lab clutter collection requires the Minhyuk governor")
    if config.data.acceptance != "nominal_safe_success":
        raise ValueError("lab clutter collection requires safe-success acceptance")
    if episodes_per_gamma is not None and episodes_per_gamma < 1:
        raise ValueError("episodes_per_gamma must be positive")

    data = (
        config.data
        if episodes_per_gamma is None
        else replace(config.data, episodes_per_gamma=int(episodes_per_gamma))
    )
    raw = copy.deepcopy(config.raw)
    raw["taskspace"] = {
        "origin": LAB_BOUNDS[:, 0].tolist(),
        "size": (LAB_BOUNDS[:, 1] - LAB_BOUNDS[:, 0]).tolist(),
        "start": LAB_START.tolist(),
        "goal": LAB_GOAL.tolist(),
        "reach_radius": float(config.taskspace.reach_radius),
        "max_steps": int(config.taskspace.max_steps),
    }
    raw["safety"] = {
        "safe_min": LAB_BOUNDS[:, 0].tolist(),
        "safe_max": LAB_BOUNDS[:, 1].tolist(),
    }
    raw["obstacles"] = {"spheres": [], "cylinders": []}
    raw["safemppi"] = dict(raw["safemppi"])
    raw["safemppi"]["z_bias_weight"] = 0.0
    raw["data"] = dict(raw["data"])
    raw["data"]["episodes_per_gamma"] = int(data.episodes_per_gamma)
    raw["domain_randomization"] = {
        "obstacle_geometry_source": "per_run_npz_and_manifest",
        "scene_bank_shared_across_gamma": True,
        "static_obstacles_in_resolved_config": False,
    }
    normalized = ExperimentConfig(
        config.taskspace,
        ObstacleConfig(spheres=(), cylinders=()),
        replace(config.safemppi, z_bias_weight=0.0),
        data,
        raw,
    )
    if normalized.safemppi.z_bias_weight != 0.0:
        raise RuntimeError("lab clutter z-bias invariant was not enforced")
    return normalized


def _randomization_defaults(config: ExperimentConfig) -> dict:
    """Resolve optional cylinder-scene defaults from the raw config block."""
    spec = config.raw.get("scene_randomization", {})
    if not spec:
        return {
            "domain_seed": 0,
            "cylinder_count": DEFAULT_CYLINDER_COUNT,
            "cylinder_radius_m": DEFAULT_CYLINDER_RADIUS_M,
            "min_surface_gap_m": DEFAULT_MIN_SURFACE_GAP_M,
            "start_surface_gap_m": DEFAULT_MIN_SURFACE_GAP_M,
            "goal_surface_gap_m": DEFAULT_MIN_SURFACE_GAP_M,
            "boundary_surface_gap_m": 0.0,
        }
    if not isinstance(spec, dict):
        raise ValueError("scene_randomization must be an object")
    if spec.get("enabled") is not True:
        raise ValueError("scene_randomization must be enabled")
    if spec.get("obstacle_family") != "vertical_cylinders":
        raise ValueError(
            "clutter demo collector requires vertical_cylinders randomization"
        )
    physical_radius = float(
        spec.get(
            "physical_radius_m",
            DEFAULT_CYLINDER_PHYSICAL_RADIUS_M,
        )
    )
    vehicle_inflation = float(
        spec.get("vehicle_inflation_m", DEFAULT_VEHICLE_INFLATION_M)
    )
    effective_radius = float(
        spec.get("radius_m", DEFAULT_CYLINDER_RADIUS_M)
    )
    if (
        physical_radius <= 0.0
        or vehicle_inflation < 0.0
        or not np.isclose(
            physical_radius + vehicle_inflation,
            effective_radius,
            atol=1.0e-9,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "cylinder radius_m must equal physical_radius_m plus "
            "vehicle_inflation_m"
        )
    return {
        "domain_seed": int(spec.get("seed", 0)),
        "cylinder_count": int(
            spec.get("count", DEFAULT_CYLINDER_COUNT)
        ),
        "cylinder_radius_m": effective_radius,
        "min_surface_gap_m": float(spec.get(
            "minimum_obstacle_surface_gap_m",
            DEFAULT_MIN_SURFACE_GAP_M,
        )),
        "start_surface_gap_m": float(spec.get(
            "minimum_start_surface_clearance_m",
            DEFAULT_MIN_SURFACE_GAP_M,
        )),
        "goal_surface_gap_m": float(spec.get(
            "minimum_goal_surface_clearance_m",
            DEFAULT_MIN_SURFACE_GAP_M,
        )),
        "boundary_surface_gap_m": float(spec.get(
            "minimum_taskspace_wall_surface_clearance_m", 0.0,
        )),
    }


def config_for_scene(
    template: ExperimentConfig,
    scene: ClutterScene,
) -> ExperimentConfig:
    """Bind one scene to the otherwise fixed collection template."""
    if template.safemppi.z_bias_weight != 0.0:
        raise ValueError("refusing clutter collection with a nonzero z bias")
    return ExperimentConfig(
        template.taskspace,
        ObstacleConfig(
            spheres=scene.spheres,
            cylinders=scene.cylinders,
        ),
        template.safemppi,
        template.data,
        template.raw,
    )


def _scene_arrays(scene: ClutterScene) -> dict[str, np.ndarray]:
    return {
        "spheres": np.asarray(scene.spheres, np.float32).reshape(-1, 4),
        "cylinders": np.asarray(scene.cylinders, np.float32).reshape(-1, 3),
        "scene_index": np.asarray(scene.index, np.int64),
        "scene_seed": np.asarray(scene.seed, np.int64),
        "scene_id": np.asarray(scene.scene_id),
        "scene_hash": np.asarray(scene.scene_hash),
    }


def _accepted_lab_demo(row: dict) -> bool:
    return bool(
        row["success"]
        and row["minimum_online_one_step_slack"] is not None
        and row["minimum_online_one_step_slack"] >= -1.0e-6
        and row["deployment_speed_compatible"]
    )


def collect_clutter_demos(
    base_config: ExperimentConfig,
    output_dir: str | Path,
    *,
    scene_count: int | None = None,
    domain_seed: int | None = None,
    rollout_seed_start: int | None = None,
    cylinder_count: int | None = None,
    cylinder_radius_m: float | None = None,
    min_surface_gap_m: float | None = None,
    endpoint_surface_gap_m: float | None = None,
    start_surface_gap_m: float | None = None,
    goal_surface_gap_m: float | None = None,
    boundary_surface_gap_m: float | None = None,
    max_layout_attempts: int = 20_000,
    max_rollout_attempts_per_scene: int | None = None,
    max_candidate_scenes: int | None = None,
    device: str = "cpu",
    episode_runner: Callable = run_episode,
    controller_factory: Callable = Mode1SafeMPPI,
) -> dict:
    """Admit scenes with one accepted SafeMPPI trajectory for every gamma.

    Candidate layouts are deterministic uniform proposals. A candidate is
    committed only after all configured gammas succeed within the finite
    per-gamma retry budget, so the resulting gamma-paired bank is explicitly
    conditioned on that observed expert-success event.
    """
    count = (
        int(base_config.data.episodes_per_gamma)
        if scene_count is None else int(scene_count)
    )
    rollout_attempt_limit = (
        int(base_config.data.max_attempts_per_gamma or 32)
        if max_rollout_attempts_per_scene is None
        else int(max_rollout_attempts_per_scene)
    )
    candidate_limit = (
        4 * count
        if max_candidate_scenes is None else int(max_candidate_scenes)
    )
    if count < 1 or rollout_attempt_limit < 1 or candidate_limit < 1:
        raise ValueError(
            "scene_count, max_rollout_attempts_per_scene, and "
            "max_candidate_scenes must be positive"
        )
    template = fixed_lab_clutter_config(
        base_config, episodes_per_gamma=count,
    )
    defaults = _randomization_defaults(base_config)
    domain_seed = (
        defaults["domain_seed"] if domain_seed is None else int(domain_seed)
    )
    cylinder_count = (
        defaults["cylinder_count"]
        if cylinder_count is None else int(cylinder_count)
    )
    cylinder_radius_m = (
        defaults["cylinder_radius_m"]
        if cylinder_radius_m is None else float(cylinder_radius_m)
    )
    min_surface_gap_m = (
        defaults["min_surface_gap_m"]
        if min_surface_gap_m is None else float(min_surface_gap_m)
    )
    if endpoint_surface_gap_m is not None:
        start_surface_gap_m = float(endpoint_surface_gap_m)
        goal_surface_gap_m = float(endpoint_surface_gap_m)
    else:
        start_surface_gap_m = (
            defaults["start_surface_gap_m"]
            if start_surface_gap_m is None else float(start_surface_gap_m)
        )
        goal_surface_gap_m = (
            defaults["goal_surface_gap_m"]
            if goal_surface_gap_m is None else float(goal_surface_gap_m)
        )
    boundary_surface_gap_m = (
        defaults["boundary_surface_gap_m"]
        if boundary_surface_gap_m is None
        else float(boundary_surface_gap_m)
    )
    candidate_scenes = cylinder_scene_bank(
        candidate_limit,
        seed=domain_seed,
        bounds=template.taskspace.bounds,
        start=np.asarray(template.taskspace.start),
        goal=np.asarray(template.taskspace.goal),
        cylinder_count=cylinder_count,
        cylinder_radius_m=cylinder_radius_m,
        min_surface_gap_m=min_surface_gap_m,
        endpoint_surface_gap_m=0.0,
        start_surface_gap_m=start_surface_gap_m,
        goal_surface_gap_m=goal_surface_gap_m,
        boundary_surface_gap_m=boundary_surface_gap_m,
        max_layout_attempts=max_layout_attempts,
    )
    candidate_path_rows = {}
    for scene in candidate_scenes:
        candidate_path_rows[scene.index] = {
            **scene.as_manifest_row(),
            "start_goal_path_diagnostics": start_goal_path_diagnostics(
                scene,
                start=np.asarray(template.taskspace.start),
                goal=np.asarray(template.taskspace.goal),
                soft_clearance_target_m=(
                    template.safemppi.soft_clearance_target
                ),
            ),
        }
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw = copy.deepcopy(template.raw)
    raw["scene_randomization"] = {
        **dict(raw.get("scene_randomization", {})),
        "enabled": True,
        "resample_frequency": (
            "fixed_scene_bank; fixed_across_controller_retries"
        ),
        "obstacle_family": "vertical_cylinders",
        "count": int(cylinder_count),
        "radius_m": float(cylinder_radius_m),
        "minimum_obstacle_surface_gap_m": float(min_surface_gap_m),
        "minimum_start_surface_clearance_m": float(start_surface_gap_m),
        "minimum_goal_surface_clearance_m": float(goal_surface_gap_m),
        "minimum_taskspace_wall_surface_clearance_m": float(
            boundary_surface_gap_m
        ),
        "seed": int(domain_seed),
        "admission_mode": (
            "all_configured_gammas_success_within_retry_budget"
        ),
        "max_candidate_scenes": candidate_limit,
        "distribution": (
            "uniform layout proposals conditioned on observed expert success "
            "within the finite retry budget for every configured gamma"
        ),
        "unconditioned_uniform": False,
    }
    raw["domain_randomization"].update({
        "kind": "three_full_height_vertical_cylinders",
        "domain_seed": int(domain_seed),
        "scene_count": count,
        "max_candidate_scenes": candidate_limit,
        "admission_mode": (
            "all_configured_gammas_success_within_retry_budget"
        ),
        "proposal_distribution": "deterministic uniform layout proposals",
        "effective_distribution": (
            "uniform proposals conditioned on observed expert success within "
            "the finite retry budget for every configured gamma"
        ),
        "unconditioned_uniform": False,
        "cylinder_count": int(cylinder_count),
        "cylinder_radius_m": float(cylinder_radius_m),
        "min_surface_gap_m": float(min_surface_gap_m),
        "minimum_start_surface_clearance_m": float(start_surface_gap_m),
        "minimum_goal_surface_clearance_m": float(goal_surface_gap_m),
        "boundary_surface_gap_m": float(boundary_surface_gap_m),
    })
    (output / "resolved_config.json").write_text(
        json.dumps(raw, indent=2) + "\n"
    )

    seed0 = (
        int(template.data.seed_start)
        if rollout_seed_start is None else int(rollout_seed_start)
    )
    gamma_values = list(map(float, template.data.gammas))
    runs: list[dict] = []
    attempts: list[dict] = []
    accepted_scene_rows: list[dict] = []
    rejected_scene_rows: list[dict] = []
    for scene in candidate_scenes:
        if len(accepted_scene_rows) == count:
            break
        scene_config = config_for_scene(template, scene)
        env = TaskEnvironment(scene_config)
        candidate_attempts: list[dict] = []
        pending_runs: list[tuple[dict, dict[str, np.ndarray]]] = []
        successful_gammas = []
        failed_gamma = None
        failed_gamma_index = None
        for gamma_index, gamma in enumerate(gamma_values):
            gamma_accepted = False
            for attempt in range(rollout_attempt_limit):
                rollout_seed = (
                    seed0
                    + scene.index * rollout_attempt_limit
                    + attempt
                )
                controller = controller_factory(
                    scene_config.safemppi, env, device=device,
                )
                row, arrays = episode_runner(
                    env,
                    controller,
                    float(gamma),
                    int(rollout_seed),
                    scene_config.data.rollout_dynamics,
                )
                trajectory_accepted = _accepted_lab_demo(row)
                row = {
                    **row,
                    **scene.as_manifest_row(),
                    "attempt_index": int(attempt),
                    "trajectory_accepted": trajectory_accepted,
                    "candidate_admitted": None,
                    "admitted_scene_index": None,
                    "accepted": False,
                    "file": None,
                }
                candidate_attempts.append(row)
                if trajectory_accepted:
                    pending_runs.append((row, arrays))
                    successful_gammas.append(float(gamma))
                    gamma_accepted = True
                    break
            if not gamma_accepted:
                failed_gamma = float(gamma)
                failed_gamma_index = gamma_index
                break

        candidate_admitted = failed_gamma is None
        admitted_scene_index = (
            len(accepted_scene_rows) if candidate_admitted else None
        )
        rejection_reason = (
            None
            if candidate_admitted else
            "no_accepted_trajectory_for_configured_gamma_within_attempt_limit"
        )
        attempt_start = len(attempts)
        for row in candidate_attempts:
            row["candidate_admitted"] = candidate_admitted
            row["admitted_scene_index"] = admitted_scene_index
            row["accepted"] = bool(
                candidate_admitted and row["trajectory_accepted"]
            )
            if rejection_reason is not None:
                row["candidate_rejection_reason"] = rejection_reason
        attempts.extend(candidate_attempts)
        attempt_range = {
            "start_inclusive": attempt_start,
            "stop_exclusive": len(attempts),
        }

        scene_row = {
            **candidate_path_rows[scene.index],
            "candidate_admitted": candidate_admitted,
            "admitted_scene_index": admitted_scene_index,
            "attempt_row_range": attempt_range,
        }
        if candidate_admitted:
            evidence = []
            for row, arrays in pending_runs:
                name = (
                    f"run_g{float(row['gamma']):g}_{scene.scene_id}"
                    f"_s{int(row['seed'])}.npz"
                )
                np.savez_compressed(
                    output / name, **arrays, **_scene_arrays(scene),
                )
                row["file"] = name
                runs.append(row)
                evidence.append({
                    "gamma": float(row["gamma"]),
                    "seed": int(row["seed"]),
                    "attempt_index": int(row["attempt_index"]),
                    "file": name,
                })
            scene_row["admission_evidence"] = evidence
            accepted_scene_rows.append(scene_row)
        else:
            scene_row.update({
                "rejection_reason": rejection_reason,
                "failed_gamma": failed_gamma,
                "max_attempts_for_failed_gamma": rollout_attempt_limit,
                "successful_gammas_before_rejection": successful_gammas,
                "unevaluated_gammas": gamma_values[
                    int(failed_gamma_index) + 1:
                ],
            })
            rejected_scene_rows.append(scene_row)

    collection_complete = len(accepted_scene_rows) == count
    evaluated_scene_rows = accepted_scene_rows + rejected_scene_rows
    path_summary = (
        summarize_start_goal_path_diagnostics([
            row["start_goal_path_diagnostics"]
            for row in accepted_scene_rows
        ])
        if accepted_scene_rows else None
    )
    evaluated_path_summary = summarize_start_goal_path_diagnostics([
        row["start_goal_path_diagnostics"] for row in evaluated_scene_rows
    ])
    metrics = (
        aggregate_metrics(runs, gamma_values) if runs else []
    )
    attempted_gammas = [
        gamma for gamma in gamma_values
        if any(np.isclose(row["gamma"], gamma) for row in attempts)
    ]
    attempt_metrics = (
        aggregate_metrics(attempts, attempted_gammas)
        if attempted_gammas else []
    )
    manifest = {
        "kind": "minhyuk lab randomized-cylinder SafeMPPI demonstrations",
        "schema_version": 2,
        "status": (
            "COMPLETE" if collection_complete
            else "FAILED_MAX_CANDIDATE_SCENES"
        ),
        "config": "resolved_config.json",
        "rollout_dynamics": template.data.rollout_dynamics,
        "acceptance": template.data.acceptance,
        "max_rollout_attempts_per_scene": rollout_attempt_limit,
        "requested_scene_count": count,
        "max_candidate_scenes": candidate_limit,
        "candidate_scenes_generated": len(candidate_scenes),
        "candidate_scenes_evaluated": len(evaluated_scene_rows),
        "admitted_scene_count": len(accepted_scene_rows),
        "rejected_scene_count": len(rejected_scene_rows),
        "z_bias_weight": 0.0,
        "gammas": gamma_values,
        "sampling_distribution": {
            "proposal": "deterministic uniform obstacle-layout proposals",
            "admission_condition": (
                "at least one accepted SafeMPPI trajectory within the finite "
                "configured per-gamma retry budget for every configured gamma"
            ),
            "effective": (
                "uniform proposals conditioned on observed expert success "
                "within the finite retry budget for every configured gamma"
            ),
            "unconditioned_uniform": False,
        },
        "scene_bank": {
            "shared_across_gamma": True,
            "domain_seed": int(domain_seed),
            "start_goal_path_summary": path_summary,
            "evaluated_candidate_start_goal_path_summary": (
                evaluated_path_summary
            ),
            "scenes": accepted_scene_rows,
            "rejected_scenes": rejected_scene_rows,
        },
        "obstacle_array_keys": [
            "spheres", "cylinders", "scene_index", "scene_seed",
            "scene_id", "scene_hash",
        ],
        "runs": runs,
        "attempts": attempts,
        "metrics": metrics,
        "attempt_metrics": attempt_metrics,
    }
    if not collection_complete:
        (output / "FAILED_collection.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        raise RuntimeError(
            f"admitted {len(accepted_scene_rows)}/{count} scenes after "
            f"{len(evaluated_scene_rows)} deterministic candidates"
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(metrics[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metrics)
    return manifest
