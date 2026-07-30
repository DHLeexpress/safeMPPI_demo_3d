"""Path-focused variable-count clutter distributions for the lab task.

Geometry is sampled independently of controller success.  The distribution
concentrates obstacle centers around the fixed start-goal segment while
enforcing only modeled-body non-overlap, endpoint non-intersection, and body
containment in the configured taskspace.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lab_clutter import (
    ClutterScene,
    LAB_BOUNDS,
    LAB_GOAL,
    LAB_START,
    obstacle_scene_hash,
)


PATH_FOCUSED_DISTRIBUTION = "path_focused_truncated_normal_v1"


def _orthogonal_basis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(direction @ reference)) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.stack([first, second], axis=1)


@dataclass(frozen=True)
class PathFocusedClutterSpec:
    obstacle_family: str
    count_min: int
    count_max: int
    physical_radius_m: float
    vehicle_inflation_m: float
    modeled_radius_m: float
    longitudinal_min: float
    longitudinal_max: float
    transverse_std_m: float
    minimum_surface_gap_m: float
    endpoint_surface_gap_m: float
    boundary_surface_gap_m: float
    domain_seed: int
    max_layout_attempts: int = 100_000

    def __post_init__(self) -> None:
        if self.obstacle_family not in {"vertical_cylinders", "spheres"}:
            raise ValueError("unsupported path-focused obstacle family")
        if self.count_min < 1 or self.count_max < self.count_min:
            raise ValueError("invalid path-focused obstacle count range")
        if (
            self.physical_radius_m <= 0.0
            or self.vehicle_inflation_m < 0.0
            or self.modeled_radius_m <= 0.0
            or not np.isclose(
                self.physical_radius_m + self.vehicle_inflation_m,
                self.modeled_radius_m,
                rtol=0.0,
                atol=1.0e-9,
            )
        ):
            raise ValueError(
                "modeled radius must equal physical radius plus vehicle inflation"
            )
        if not 0.0 < self.longitudinal_min < self.longitudinal_max < 1.0:
            raise ValueError("longitudinal fractions must lie strictly inside (0,1)")
        if (
            self.transverse_std_m <= 0.0
            or self.minimum_surface_gap_m < 0.0
            or self.endpoint_surface_gap_m < 0.0
            or self.boundary_surface_gap_m < 0.0
            or self.max_layout_attempts < self.count_max
        ):
            raise ValueError("path-focused scales/margins are invalid")

    @classmethod
    def from_config(
        cls,
        config,
        *,
        expected_family: str | None = None,
    ) -> "PathFocusedClutterSpec":
        raw = config.raw.get("scene_randomization")
        if not isinstance(raw, dict) or raw.get("enabled") is not True:
            raise ValueError("path-focused scene_randomization must be enabled")
        if raw.get("distribution") != PATH_FOCUSED_DISTRIBUTION:
            raise ValueError(
                f"scene distribution must be {PATH_FOCUSED_DISTRIBUTION!r}"
            )
        family = str(raw.get("obstacle_family"))
        if expected_family is not None and family != expected_family:
            raise ValueError(
                f"expected obstacle family {expected_family!r}, found {family!r}"
            )
        count = raw.get("count")
        count_min = int(raw.get("count_min", count if count is not None else -1))
        count_max = int(raw.get("count_max", count if count is not None else -1))
        longitudinal = raw.get("longitudinal_fraction", (0.15, 0.85))
        if len(longitudinal) != 2:
            raise ValueError("longitudinal_fraction must contain [minimum, maximum]")
        start_gap = float(raw.get("minimum_start_surface_clearance_m", 0.0))
        goal_gap = float(raw.get("minimum_goal_surface_clearance_m", 0.0))
        if not np.isclose(start_gap, goal_gap, rtol=0.0, atol=0.0):
            raise ValueError(
                "path-focused scenes currently require equal start/goal gaps"
            )
        return cls(
            obstacle_family=family,
            count_min=count_min,
            count_max=count_max,
            physical_radius_m=float(raw["physical_radius_m"]),
            vehicle_inflation_m=float(raw["vehicle_inflation_m"]),
            modeled_radius_m=float(raw["radius_m"]),
            longitudinal_min=float(longitudinal[0]),
            longitudinal_max=float(longitudinal[1]),
            transverse_std_m=float(raw["transverse_std_m"]),
            minimum_surface_gap_m=float(
                raw["minimum_obstacle_surface_gap_m"]
            ),
            endpoint_surface_gap_m=start_gap,
            boundary_surface_gap_m=float(
                raw["minimum_taskspace_wall_surface_clearance_m"]
            ),
            domain_seed=int(raw.get("seed", 0)),
            max_layout_attempts=int(raw.get("max_layout_attempts", 100_000)),
        )

    @property
    def dimensions(self) -> int:
        return 2 if self.obstacle_family == "vertical_cylinders" else 3

    @property
    def scene_schema(self) -> str:
        family = (
            "vertical_cylinders"
            if self.obstacle_family == "vertical_cylinders"
            else "spheres"
        )
        return f"lab_path_focused_variable_{family}_v2"

    def _sample_rows(
        self,
        *,
        scene_seed: int,
        bounds: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        dimensions = self.dimensions
        bounds = np.asarray(bounds, np.float64).reshape(3, 2)
        start = np.asarray(start, np.float64).reshape(-1)[:dimensions]
        goal = np.asarray(goal, np.float64).reshape(-1)[:dimensions]
        direction = goal - start
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-12:
            raise ValueError("path-focused scenes require distinct start and goal")
        direction /= length
        transverse = (
            np.asarray([[-direction[1]], [direction[0]]], np.float64)
            if dimensions == 2
            else _orthogonal_basis(direction)
        )
        # Perform every geometry test with the exact radius representation that
        # will be written into NPZ/context arrays.  Otherwise a layout can pass
        # in float64 and overlap by a few nanometers after float32 serialization.
        serialized_radius = float(np.float32(self.modeled_radius_m))
        lower = (
            bounds[:dimensions, 0]
            + serialized_radius
            + self.boundary_surface_gap_m
        )
        upper = (
            bounds[:dimensions, 1]
            - serialized_radius
            - self.boundary_surface_gap_m
        )
        if bool((lower >= upper).any()):
            raise ValueError("modeled body and wall margin leave no sampling area")

        effective_seed = int(np.random.SeedSequence([
            int(self.domain_seed), int(scene_seed),
        ]).generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(effective_seed)
        count = int(rng.integers(self.count_min, self.count_max + 1))
        endpoint_distance = (
            serialized_radius + self.endpoint_surface_gap_m
        )
        pair_distance = (
            2.0 * serialized_radius + self.minimum_surface_gap_m
        )
        centers: list[np.ndarray] = []
        for _ in range(self.max_layout_attempts):
            along = rng.uniform(
                self.longitudinal_min, self.longitudinal_max,
            )
            offset = rng.normal(
                0.0, self.transverse_std_m, size=transverse.shape[1],
            )
            candidate = (
                start + along * (goal - start) + transverse @ offset
            ).astype(np.float32).astype(np.float64)
            if bool((candidate < lower).any() or (candidate > upper).any()):
                continue
            if (
                float(np.linalg.norm(candidate - start)) < endpoint_distance
                or float(np.linalg.norm(candidate - goal)) < endpoint_distance
            ):
                continue
            if any(
                float(np.linalg.norm(candidate - center)) < pair_distance
                for center in centers
            ):
                continue
            centers.append(candidate)
            if len(centers) == count:
                break
        if len(centers) != count:
            raise RuntimeError(
                f"could not place {count} {self.obstacle_family} for "
                f"scene seed {scene_seed}"
            )
        radii = np.full(
            (count, 1), serialized_radius, dtype=np.float32,
        )
        return np.concatenate([
            np.asarray(centers, np.float32), radii,
        ], axis=1)

    def sample_scene(
        self,
        *,
        scene_index: int,
        scene_seed: int,
        bounds: np.ndarray = LAB_BOUNDS,
        start: np.ndarray = LAB_START,
        goal: np.ndarray = LAB_GOAL,
    ) -> ClutterScene:
        rows = self._sample_rows(
            scene_seed=scene_seed,
            bounds=bounds,
            start=start,
            goal=goal,
        )
        if self.obstacle_family == "vertical_cylinders":
            cylinders = tuple(tuple(map(float, row)) for row in rows)
            spheres = ()
        else:
            spheres = tuple(tuple(map(float, row)) for row in rows)
            cylinders = ()
        return ClutterScene(
            index=int(scene_index),
            seed=int(scene_seed),
            spheres=spheres,
            cylinders=cylinders,
            scene_hash=obstacle_scene_hash(
                spheres=spheres, cylinders=cylinders,
            ),
        )


def path_focused_scene_bank(
    config,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[ClutterScene, ...]:
    """Materialize a deterministic geometry-only scene bank."""
    if int(count) < 1:
        raise ValueError("scene count must be positive")
    spec = PathFocusedClutterSpec.from_config(config)
    bank_seed = spec.domain_seed if seed is None else int(seed)
    scenes = []
    for index in range(int(count)):
        scene_seed = int(np.random.SeedSequence([
            bank_seed, index,
        ]).generate_state(1, dtype=np.uint32)[0])
        scenes.append(spec.sample_scene(
            scene_index=index,
            scene_seed=scene_seed,
            bounds=config.taskspace.bounds,
            start=np.asarray(config.taskspace.start),
            goal=np.asarray(config.taskspace.goal),
        ))
    return tuple(scenes)
