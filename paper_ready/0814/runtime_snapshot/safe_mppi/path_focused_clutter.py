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
PATH_FOCUSED_MIDPOINT_UNIFORM_DISTRIBUTION = (
    "path_focused_midpoint_uniform_v2"
)
PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION = (
    "path_focused_double_hourglass_v1"
)
PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION = (
    "path_focused_center_anchored_truncnorm_v1"
)
PATH_FOCUSED_DISTRIBUTIONS = frozenset({
    PATH_FOCUSED_DISTRIBUTION,
    PATH_FOCUSED_MIDPOINT_UNIFORM_DISTRIBUTION,
    PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION,
    PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION,
})


def midpoint_uniform_halfwidth(
    longitudinal_fraction: float,
    scale_m: float,
) -> float:
    """Return the symmetric transverse proposal half-width in metres."""
    return float(scale_m) * (
        0.5 - abs(float(longitudinal_fraction) - 0.5)
    )


def double_hourglass_halfwidth(
    longitudinal_fraction: float,
    control_points: tuple[tuple[float, float], ...],
) -> float:
    """Linearly interpolate the configured lateral half-width profile."""
    points = np.asarray(control_points, dtype=np.float64)
    return float(np.interp(
        float(longitudinal_fraction), points[:, 0], points[:, 1],
    ))


def _sample_truncated_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    std: float,
    lower: float,
    upper: float,
) -> float:
    while True:
        value = float(rng.normal(mean, std))
        if lower <= value <= upper:
            return value


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
    distribution: str
    obstacle_family: str
    count_min: int
    count_max: int
    physical_radius_m: float
    vehicle_inflation_m: float
    modeled_radius_m: float
    longitudinal_min: float
    longitudinal_max: float
    transverse_std_m: float | None
    transverse_halfwidth_scale_m: float | None
    transverse_halfwidth_control_points: (
        tuple[tuple[float, float], ...] | None
    )
    sphere_z_mean_m: float | None
    sphere_z_std_m: float | None
    sphere_z_lower_m: float | None
    sphere_z_upper_m: float | None
    sphere_z_plane_m: float | None
    sphere_vertical_halfwidth_scale: float | None
    minimum_surface_gap_m: float
    endpoint_surface_gap_m: float
    boundary_surface_gap_m: float
    domain_seed: int
    max_layout_attempts: int = 100_000
    fixed_sphere_centers_m: (
        tuple[tuple[float, float, float], ...] | None
    ) = None

    def __post_init__(self) -> None:
        if self.distribution not in PATH_FOCUSED_DISTRIBUTIONS:
            raise ValueError("unsupported path-focused distribution")
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
            self.minimum_surface_gap_m < 0.0
            or self.endpoint_surface_gap_m < 0.0
            or self.boundary_surface_gap_m < 0.0
            or self.max_layout_attempts < self.count_max
        ):
            raise ValueError("path-focused scales/margins are invalid")
        if (
            self.fixed_sphere_centers_m is not None
            and self.distribution
            != PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION
        ):
            raise ValueError(
                "fixed sphere centers require the center-anchored distribution"
            )
        if self.distribution == PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION:
            if self.obstacle_family != "spheres":
                raise ValueError(
                    "center-anchored scenes currently require spheres"
                )
            if (
                self.transverse_std_m is not None
                or self.transverse_halfwidth_scale_m is None
                or self.transverse_halfwidth_scale_m <= 0.0
                or self.transverse_halfwidth_control_points is not None
                or self.sphere_z_plane_m is not None
                or self.sphere_vertical_halfwidth_scale is not None
            ):
                raise ValueError(
                    "center-anchored scenes require a positive half-width "
                    "scale and reject legacy/hourglass parameters"
                )
            if any(value is None for value in (
                self.sphere_z_mean_m, self.sphere_z_std_m,
                self.sphere_z_lower_m, self.sphere_z_upper_m,
            )) or not (
                self.sphere_z_std_m > 0.0
                and self.sphere_z_lower_m
                < self.sphere_z_mean_m
                < self.sphere_z_upper_m
            ):
                raise ValueError(
                    "center-anchored scenes require a truncated-normal z law"
                )
            if (
                not self.fixed_sphere_centers_m
                or any(
                    len(center) != 3 for center in self.fixed_sphere_centers_m
                )
            ):
                raise ValueError(
                    "center-anchored scenes require at least one fixed "
                    "sphere center [x, y, z]"
                )
            if len(self.fixed_sphere_centers_m) > self.count_min:
                raise ValueError(
                    "fixed sphere centers cannot exceed the minimum count"
                )
        elif self.distribution == PATH_FOCUSED_DISTRIBUTION:
            if self.transverse_std_m is None or self.transverse_std_m <= 0.0:
                raise ValueError("legacy path-focused scenes require positive std")
            if any(value is not None for value in (
                self.transverse_halfwidth_scale_m,
                self.sphere_z_mean_m,
                self.sphere_z_std_m,
                self.sphere_z_lower_m,
                self.sphere_z_upper_m,
            )):
                raise ValueError("legacy path-focused scenes reject v2 parameters")
        elif self.distribution == PATH_FOCUSED_MIDPOINT_UNIFORM_DISTRIBUTION:
            if (
                self.transverse_std_m is not None
                or self.transverse_halfwidth_scale_m is None
                or self.transverse_halfwidth_scale_m <= 0.0
                or self.transverse_halfwidth_control_points is not None
            ):
                raise ValueError(
                    "midpoint-uniform scenes require a positive half-width scale"
                )
            z_values = (
                self.sphere_z_mean_m,
                self.sphere_z_std_m,
                self.sphere_z_lower_m,
                self.sphere_z_upper_m,
            )
            if self.obstacle_family == "spheres":
                if any(value is None for value in z_values):
                    raise ValueError(
                        "midpoint-uniform spheres require a truncated-normal z law"
                    )
                if not (
                    self.sphere_z_std_m > 0.0
                    and self.sphere_z_lower_m
                    < self.sphere_z_mean_m
                    < self.sphere_z_upper_m
                ):
                    raise ValueError("invalid truncated-normal sphere z law")
            elif any(value is not None for value in z_values):
                raise ValueError("cylinder scenes reject sphere z parameters")
            if (
                self.sphere_z_plane_m is not None
                or self.sphere_vertical_halfwidth_scale is not None
            ):
                raise ValueError("midpoint-uniform scenes reject hourglass z parameters")
        else:
            if self.obstacle_family != "spheres":
                raise ValueError("double-hourglass scenes currently require spheres")
            if (
                self.transverse_std_m is not None
                or self.transverse_halfwidth_scale_m is not None
                or self.transverse_halfwidth_control_points is None
                or len(self.transverse_halfwidth_control_points) < 2
            ):
                raise ValueError(
                    "double-hourglass scenes require half-width control points"
                )
            points = np.asarray(
                self.transverse_halfwidth_control_points, dtype=np.float64,
            )
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not np.all(np.diff(points[:, 0]) > 0.0)
                or not np.all(points[:, 1] > 0.0)
                or not np.isclose(points[0, 0], self.longitudinal_min)
                or not np.isclose(points[-1, 0], self.longitudinal_max)
            ):
                raise ValueError("invalid double-hourglass control points")
            if any(value is not None for value in (
                self.sphere_z_mean_m, self.sphere_z_std_m,
            )):
                raise ValueError("double-hourglass scenes reject Gaussian z parameters")
            if (
                self.sphere_z_lower_m is None
                or self.sphere_z_upper_m is None
                or self.sphere_z_plane_m is None
                or self.sphere_vertical_halfwidth_scale is None
                or self.sphere_vertical_halfwidth_scale <= 0.0
                or not (
                    self.sphere_z_lower_m
                    < self.sphere_z_plane_m
                    < self.sphere_z_upper_m
                )
            ):
                raise ValueError("invalid symmetric double-hourglass z law")

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
        distribution = str(raw.get("distribution"))
        if distribution not in PATH_FOCUSED_DISTRIBUTIONS:
            raise ValueError(
                "scene distribution must be a supported path-focused law"
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
        z_range = raw.get("sphere_z_center_range_m")
        if z_range is not None and (
            not isinstance(z_range, (list, tuple)) or len(z_range) != 2
        ):
            raise ValueError("sphere_z_center_range_m must contain [lower, upper]")
        if (
            distribution in {
                PATH_FOCUSED_MIDPOINT_UNIFORM_DISTRIBUTION,
                PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION,
            }
            and family == "spheres"
            and raw.get("sphere_z_distribution") != "truncated_normal"
        ):
            raise ValueError(
                "midpoint-uniform and center-anchored spheres require "
                "sphere_z_distribution='truncated_normal'"
            )
        if (
            distribution == PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION
            and raw.get("sphere_z_distribution")
            != "symmetric_longitudinal_scaled_uniform"
        ):
            raise ValueError(
                "double-hourglass spheres require sphere_z_distribution="
                "'symmetric_longitudinal_scaled_uniform'"
            )
        start_gap = float(raw.get("minimum_start_surface_clearance_m", 0.0))
        goal_gap = float(raw.get("minimum_goal_surface_clearance_m", 0.0))
        if not np.isclose(start_gap, goal_gap, rtol=0.0, atol=0.0):
            raise ValueError(
                "path-focused scenes currently require equal start/goal gaps"
            )
        return cls(
            distribution=distribution,
            obstacle_family=family,
            count_min=count_min,
            count_max=count_max,
            physical_radius_m=float(raw["physical_radius_m"]),
            vehicle_inflation_m=float(raw["vehicle_inflation_m"]),
            modeled_radius_m=float(raw["radius_m"]),
            longitudinal_min=float(longitudinal[0]),
            longitudinal_max=float(longitudinal[1]),
            transverse_std_m=(
                float(raw["transverse_std_m"])
                if raw.get("transverse_std_m") is not None else None
            ),
            transverse_halfwidth_scale_m=(
                float(raw["transverse_halfwidth_scale_m"])
                if raw.get("transverse_halfwidth_scale_m") is not None
                else None
            ),
            transverse_halfwidth_control_points=(
                tuple(
                    (float(point[0]), float(point[1]))
                    for point in raw["transverse_halfwidth_control_points"]
                )
                if raw.get("transverse_halfwidth_control_points") is not None
                else None
            ),
            sphere_z_mean_m=(
                float(raw["sphere_z_mean_m"])
                if raw.get("sphere_z_mean_m") is not None else None
            ),
            sphere_z_std_m=(
                float(raw["sphere_z_std_m"])
                if raw.get("sphere_z_std_m") is not None else None
            ),
            sphere_z_lower_m=(
                float(z_range[0]) if z_range is not None else None
            ),
            sphere_z_upper_m=(
                float(z_range[1]) if z_range is not None else None
            ),
            sphere_z_plane_m=(
                float(raw["sphere_z_plane_m"])
                if raw.get("sphere_z_plane_m") is not None else None
            ),
            sphere_vertical_halfwidth_scale=(
                float(raw["sphere_vertical_halfwidth_scale"])
                if raw.get("sphere_vertical_halfwidth_scale") is not None
                else None
            ),
            minimum_surface_gap_m=float(
                raw["minimum_obstacle_surface_gap_m"]
            ),
            endpoint_surface_gap_m=start_gap,
            boundary_surface_gap_m=float(
                raw["minimum_taskspace_wall_surface_clearance_m"]
            ),
            domain_seed=int(raw.get("seed", 0)),
            max_layout_attempts=int(raw.get("max_layout_attempts", 100_000)),
            fixed_sphere_centers_m=(
                tuple(
                    (float(center[0]), float(center[1]), float(center[2]))
                    for center in raw["fixed_sphere_centers_m"]
                )
                if raw.get("fixed_sphere_centers_m") is not None
                else None
            ),
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
        if self.distribution == PATH_FOCUSED_DISTRIBUTION:
            return f"lab_path_focused_variable_{family}_v2"
        if self.distribution == PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION:
            return "lab_path_focused_double_hourglass_variable_spheres_v1"
        if self.distribution == PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION:
            return (
                "lab_path_focused_center_anchored_truncnorm_variable_"
                "spheres_v1"
            )
        return f"lab_path_focused_midpoint_uniform_variable_{family}_v2"

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
        if self.distribution == PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION:
            # Center-anchored contract: minimum_surface_gap_m measures the
            # PHYSICAL surface separation, floored so the inflated (modeled)
            # bodies can never overlap.  gap=0.4 with 0.1 m inflation leaves a
            # full robot-diameter (0.2 m) corridor between inflated surfaces.
            serialized_physical = float(np.float32(self.physical_radius_m))
            pair_distance = max(
                2.0 * serialized_radius,
                2.0 * serialized_physical + self.minimum_surface_gap_m,
            )
        else:
            pair_distance = (
                2.0 * serialized_radius + self.minimum_surface_gap_m
            )
        centers: list[np.ndarray] = []
        if self.fixed_sphere_centers_m is not None:
            for fixed in self.fixed_sphere_centers_m:
                anchored = np.asarray(
                    fixed[:dimensions], np.float32,
                ).astype(np.float64)
                if bool(
                    (anchored < lower).any() or (anchored > upper).any()
                ):
                    raise ValueError(
                        "fixed sphere center violates taskspace margins"
                    )
                if (
                    float(np.linalg.norm(anchored - start))
                    < endpoint_distance
                    or float(np.linalg.norm(anchored - goal))
                    < endpoint_distance
                ):
                    raise ValueError(
                        "fixed sphere center violates start/goal clearance"
                    )
                if any(
                    float(np.linalg.norm(anchored - center)) < pair_distance
                    for center in centers
                ):
                    raise ValueError(
                        "fixed sphere centers violate the pairwise gap"
                    )
                centers.append(anchored)
        if len(centers) > count:
            raise ValueError(
                "fixed sphere centers exceed the sampled obstacle count"
            )
        for _ in range(0 if len(centers) == count else self.max_layout_attempts):
            along = rng.uniform(
                self.longitudinal_min, self.longitudinal_max,
            )
            if self.distribution == PATH_FOCUSED_DISTRIBUTION:
                offset = rng.normal(
                    0.0, self.transverse_std_m, size=transverse.shape[1],
                )
                candidate = (
                    start + along * (goal - start) + transverse @ offset
                )
            else:
                start_xy = start[:2]
                goal_xy = goal[:2]
                horizontal = goal_xy - start_xy
                horizontal_length = float(np.linalg.norm(horizontal))
                if horizontal_length <= 1.0e-12:
                    raise ValueError(
                        "midpoint-uniform scenes require horizontal progress"
                    )
                horizontal /= horizontal_length
                normal = np.asarray(
                    [-horizontal[1], horizontal[0]], np.float64,
                )
                if (
                    self.distribution
                    == PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION
                ):
                    halfwidth = double_hourglass_halfwidth(
                        along, self.transverse_halfwidth_control_points,
                    )
                else:
                    halfwidth = midpoint_uniform_halfwidth(
                        along, self.transverse_halfwidth_scale_m,
                    )
                delta = rng.uniform(-halfwidth, halfwidth)
                xy = start_xy + along * (goal_xy - start_xy) + normal * delta
                if dimensions == 2:
                    candidate = xy
                else:
                    if (
                        self.distribution
                        == PATH_FOCUSED_DOUBLE_HOURGLASS_DISTRIBUTION
                    ):
                        vertical_halfwidth = min(
                            self.sphere_z_plane_m - self.sphere_z_lower_m,
                            self.sphere_z_upper_m - self.sphere_z_plane_m,
                            halfwidth * self.sphere_vertical_halfwidth_scale,
                        )
                        z = rng.uniform(
                            self.sphere_z_plane_m - vertical_halfwidth,
                            self.sphere_z_plane_m + vertical_halfwidth,
                        )
                    else:
                        z = _sample_truncated_normal(
                            rng,
                            mean=self.sphere_z_mean_m,
                            std=self.sphere_z_std_m,
                            lower=self.sphere_z_lower_m,
                            upper=self.sphere_z_upper_m,
                        )
                    candidate = np.concatenate([xy, [z]])
            candidate = candidate.astype(np.float32).astype(np.float64)
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
