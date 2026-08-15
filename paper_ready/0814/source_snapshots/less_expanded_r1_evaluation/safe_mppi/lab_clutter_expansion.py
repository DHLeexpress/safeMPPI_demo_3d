"""Dynamic three-sphere lab task for visual Safe Flow Expansion.

The learned policy receives the existing robot-centered visual context.  Exact
scene geometry is appended only for the verifier so that concurrent episodes
may use different worlds without relying on mutable task-global state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .ball_flow_task import PLAN_H, nominal_hp_chain_margins
from .ball_flow_theta import start_goal_frame
from .environment import ReferenceGovernor, TaskEnvironment
from .expansion import Verification
from .lab_flow_expansion import LabExpansionPolicyAdapter
from .lab_flow_expansion import _is_history_context_schema
from .lab_clutter import (
    ClutterScene,
    LAB_BOUNDS,
    LAB_GOAL,
    LAB_START,
    obstacle_scene_hash,
    start_goal_path_diagnostics,
)
from .lab_reference_flow_task import (
    _append_raw_history,
    _raw_history_before,
    policy_context,
)
from .lab_visual_flow import (
    LAB_HP100_EXACT_MEMORY_PACKED_DIM,
    LAB_HP100_EXACT_MEMORY_SCHEMA,
    LAB_HP100_HISTORY_PACKED_DIM,
    LAB_HP100_HISTORY_SCHEMA,
    LAB_HP100_PACKED_DIM,
    LAB_HP100_SCHEMA,
    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_SCHEMA,
    LAB_VISUAL_HISTORY_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    load_lab_reference_policy,
)
from .path_focused_clutter import (
    PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION,
    PATH_FOCUSED_DISTRIBUTIONS,
    PathFocusedClutterSpec,
)
from .verifier_polytope import certify_window


LAB_CLUTTER_SPHERE_COUNT = 3
LAB_CLUTTER_SCENE_DIM = 4 * LAB_CLUTTER_SPHERE_COUNT
LAB_CLUTTER_GOVERNOR_DIM = 6
LAB_CLUTTER_VERIFIER_SUFFIX_DIM = (
    LAB_CLUTTER_GOVERNOR_DIM + LAB_CLUTTER_SCENE_DIM
)
LAB_CLUTTER_SCENE_SCHEMA = "lab_random_three_spheres_v1"
PERPENDICULAR_TRIANGLE_DISTRIBUTION = (
    "perpendicular_triangle_anisotropic_v1"
)
PERPENDICULAR_TRIANGLE_SCENE_SCHEMA = (
    "lab_perpendicular_triangle_three_spheres_v1"
)
LAB_CLUTTER_VISUAL_CONTEXT_DIMS = {
    LAB_HP100_EXACT_MEMORY_SCHEMA: LAB_HP100_EXACT_MEMORY_PACKED_DIM,
    LAB_HP100_HISTORY_SCHEMA: LAB_HP100_HISTORY_PACKED_DIM,
    LAB_HP100_SCHEMA: LAB_HP100_PACKED_DIM,
    LAB_VISUAL_SCHEMA: LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA: LAB_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_SCHEMA: LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA: (
        LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM
    ),
}


def canonical_sphere_rows(
    values: np.ndarray,
    *,
    expected_count: int | None = None,
) -> np.ndarray:
    """Return finite spheres in deterministic lexicographic order."""
    spheres = np.asarray(values, np.float32).reshape(-1, 4)
    if expected_count is not None and len(spheres) != int(expected_count):
        raise ValueError(
            f"lab clutter scene requires exactly {expected_count} spheres"
        )
    if len(spheres) < 1:
        raise ValueError("lab clutter scene requires at least one sphere")
    if not np.isfinite(spheres).all() or bool((spheres[:, 3] <= 0.0).any()):
        raise ValueError("sphere centers/radii must be finite and radii positive")
    order = np.lexsort((
        spheres[:, 3],
        spheres[:, 2],
        spheres[:, 1],
        spheres[:, 0],
    ))
    return np.ascontiguousarray(spheres[order], dtype=np.float32)


def canonical_spheres(values: np.ndarray) -> np.ndarray:
    """Backward-compatible exact-three canonicalization."""
    return canonical_sphere_rows(
        values, expected_count=LAB_CLUTTER_SPHERE_COUNT,
    )


def scene_sha256(env: TaskEnvironment, spheres: np.ndarray) -> str:
    """Hash exact obstacle geometry using the shared clutter-archive contract."""
    del env
    return obstacle_scene_hash(spheres=canonical_sphere_rows(spheres))


@dataclass(frozen=True)
class RandomThreeSphereScene:
    """Deterministic sampler for the configured three-sphere safety geometry."""

    radius: float = 0.379
    minimum_surface_margin: float = 0.20
    endpoint_margin: float = 0.50
    boundary_surface_margin: float = 0.10
    domain_seed: int = 0
    max_attempts: int = 10_000

    @property
    def max_count(self) -> int:
        return LAB_CLUTTER_SPHERE_COUNT

    @property
    def packed_dim(self) -> int:
        return LAB_CLUTTER_SCENE_DIM

    @property
    def scene_schema(self) -> str:
        return LAB_CLUTTER_SCENE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.radius <= 0.0
            or self.minimum_surface_margin < 0.0
            or self.endpoint_margin < 0.0
            or self.boundary_surface_margin < 0.0
            or self.max_attempts < LAB_CLUTTER_SPHERE_COUNT
        ):
            raise ValueError(
                "scene radius/max_attempts must be positive and margins nonnegative"
            )

    @classmethod
    def from_config(cls, config) -> "RandomThreeSphereScene":
        raw = config.raw.get("scene_randomization")
        if raw is None:
            return cls()
        if (
            not raw.get("enabled", False)
            or raw.get("resample_frequency") != "per_episode_reset"
            or raw.get("obstacle_family") != "spheres"
            or int(raw.get("count", -1)) != LAB_CLUTTER_SPHERE_COUNT
            or tuple(raw.get("sample_center_axes", ())) != ("x", "y", "z")
            or tuple(raw.get("taskspace_wall_clearance_axes", ()))
            != ("x", "y", "z")
            or tuple(raw.get("start_goal_clearance_axes", ()))
            != ("x", "y", "z")
        ):
            raise ValueError(
                "scene_randomization must enable three x-y-z-randomized spheres"
            )
        start_margin = float(raw["minimum_start_surface_clearance_m"])
        goal_margin = float(raw["minimum_goal_surface_clearance_m"])
        if not np.isclose(start_margin, goal_margin, atol=0.0, rtol=0.0):
            raise ValueError(
                "the clutter task currently requires equal start/goal margins"
            )
        physical_radius = float(raw.get("physical_radius_m", 0.0))
        vehicle_inflation = float(raw.get("vehicle_inflation_m", 0.0))
        effective_radius = float(raw["radius_m"])
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
                "sphere radius_m must equal physical_radius_m plus "
                "vehicle_inflation_m"
            )
        return cls(
            radius=effective_radius,
            minimum_surface_margin=float(
                raw["minimum_obstacle_surface_gap_m"]
            ),
            endpoint_margin=start_margin,
            boundary_surface_margin=float(
                raw["minimum_taskspace_wall_surface_clearance_m"]
            ),
            domain_seed=int(raw.get("seed", 0)),
        )

    def sample(self, env: TaskEnvironment, seed: int) -> np.ndarray:
        """Sample bodies fully inside the geofence and clear of start/goal."""
        wall_offset = self.radius + self.boundary_surface_margin
        lower = np.asarray(env.bounds[:, 0], float) + wall_offset
        upper = np.asarray(env.bounds[:, 1], float) - wall_offset
        if bool((lower >= upper).any()):
            raise ValueError("taskspace is too small for the configured spheres")

        effective_seed = int(
            np.random.SeedSequence([
                int(self.domain_seed),
                int(seed),
            ]).generate_state(1, dtype=np.uint32)[0]
        )
        rng = np.random.default_rng(effective_seed)
        centers: list[np.ndarray] = []
        start = np.asarray(env.start[:3], float)
        goal = np.asarray(env.goal, float)
        start_distance = self.radius + self.endpoint_margin
        goal_distance = self.radius + self.endpoint_margin
        pair_distance = 2.0 * self.radius + self.minimum_surface_margin
        for _ in range(self.max_attempts):
            candidate = rng.uniform(lower, upper)
            if float(np.linalg.norm(candidate - start)) < start_distance:
                continue
            if float(np.linalg.norm(candidate - goal)) < goal_distance:
                continue
            if any(
                float(np.linalg.norm(candidate - other)) < pair_distance
                for other in centers
            ):
                continue
            centers.append(candidate)
            if len(centers) == LAB_CLUTTER_SPHERE_COUNT:
                break
        if len(centers) != LAB_CLUTTER_SPHERE_COUNT:
            raise RuntimeError(
                "failed to sample a valid three-sphere scene within max_attempts"
            )
        radii = np.full(
            (LAB_CLUTTER_SPHERE_COUNT, 1), self.radius, dtype=np.float32,
        )
        return canonical_spheres(np.concatenate([
            np.asarray(centers, np.float32), radii,
        ], axis=1))

    def validate(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        """Fail closed when a packed scene violates the sampler contract."""
        spheres = canonical_spheres(values)
        tolerance = 2.0e-6
        if not np.allclose(
            spheres[:, 3], self.radius, atol=tolerance, rtol=0.0,
        ):
            raise ValueError("packed sphere radii do not match the scene contract")
        if bool((
            spheres[:, :3] - spheres[:, 3, None]
            < env.bounds[:, 0][None] - tolerance
        ).any()) or bool((
            spheres[:, :3] + spheres[:, 3, None]
            > env.bounds[:, 1][None] + tolerance
        ).any()):
            raise ValueError("packed sphere body lies outside the taskspace")
        wall_clearance = np.minimum(
            spheres[:, :3] - spheres[:, 3, None]
            - env.bounds[:, 0][None],
            env.bounds[:, 1][None]
            - spheres[:, :3] - spheres[:, 3, None],
        )
        if bool((
            wall_clearance
            < self.boundary_surface_margin - tolerance
        ).any()):
            raise ValueError("packed sphere violates the taskspace wall margin")
        for first in range(LAB_CLUTTER_SPHERE_COUNT):
            for second in range(first + 1, LAB_CLUTTER_SPHERE_COUNT):
                required = (
                    float(spheres[first, 3])
                    + float(spheres[second, 3])
                    + self.minimum_surface_margin
                )
                if (
                    float(np.linalg.norm(
                        spheres[first, :3] - spheres[second, :3]
                    ))
                    < required - tolerance
                ):
                    raise ValueError("packed spheres violate the surface margin")
        start_clearance = np.linalg.norm(
            spheres[:, :3] - env.start[None, :3], axis=1,
        ) - spheres[:, 3]
        if bool((start_clearance < self.endpoint_margin - tolerance).any()):
            raise ValueError("packed sphere is too close to the fixed start")
        goal_clearance = np.linalg.norm(
            spheres[:, :3] - env.goal[None], axis=1,
        ) - spheres[:, 3]
        required_goal_clearance = self.endpoint_margin
        if bool((
            goal_clearance < required_goal_clearance - tolerance
        ).any()):
            raise ValueError("packed sphere intersects the fixed goal region")
        return spheres

    def pack(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        return self.validate(env, values).reshape(-1)

    def unpack(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        packed = np.asarray(values, np.float32).reshape(-1)
        if len(packed) != self.packed_dim:
            raise ValueError("three-sphere packed context has wrong length")
        return self.validate(
            env, packed.reshape(LAB_CLUTTER_SPHERE_COUNT, 4),
        )


@dataclass(frozen=True)
class PerpendicularTriangleSphereScene(RandomThreeSphereScene):
    """Tight three-sphere triangle with anisotropic local-frame jitter.

    The unperturbed centers form an equilateral triangle in the plane normal
    to the fixed start-goal axis.  Episode scenes independently perturb every
    center in the task-local frame.  Setting ``plane_std_m=0`` keeps each
    center exactly on its own start-goal-parallel line.  The inherited geometry
    validator rejects overlaps and taskspace violations; no controller-success
    filtering is used.
    """

    center_fraction: float = 0.5
    center_spacing_m: float = 0.62
    parallel_std_m: float = 0.18
    plane_std_m: float = 0.025

    @property
    def scene_schema(self) -> str:
        return PERPENDICULAR_TRIANGLE_SCENE_SCHEMA

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not 0.0 < self.center_fraction < 1.0
            or self.center_spacing_m <= 2.0 * self.radius
            or self.parallel_std_m < 0.0
            or self.plane_std_m < 0.0
            or self.parallel_std_m <= self.plane_std_m
        ):
            raise ValueError(
                "triangle scenes require an interior center fraction, a "
                "nonoverlapping base spacing, and parallel_std > plane_std >= 0"
            )

    @classmethod
    def from_config(cls, config) -> "PerpendicularTriangleSphereScene":
        raw = config.raw.get("scene_randomization")
        if not isinstance(raw, dict):
            raise ValueError("triangle scene_randomization is missing")
        if (
            raw.get("enabled") is not True
            or raw.get("resample_frequency") != "per_episode_reset"
            or raw.get("obstacle_family") != "spheres"
            or raw.get("distribution")
            != PERPENDICULAR_TRIANGLE_DISTRIBUTION
            or int(raw.get("count", -1)) != LAB_CLUTTER_SPHERE_COUNT
            or tuple(raw.get("sample_center_axes", ())) != ("x", "y", "z")
        ):
            raise ValueError(
                "triangle randomization requires three per-episode xyz spheres"
            )
        physical_radius = float(raw.get("physical_radius_m", 0.0))
        vehicle_inflation = float(raw.get("vehicle_inflation_m", 0.0))
        effective_radius = float(raw.get("radius_m", 0.0))
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
                "triangle radius_m must equal physical_radius_m plus "
                "vehicle_inflation_m"
            )
        start_margin = float(
            raw.get("minimum_start_surface_clearance_m", 0.0)
        )
        goal_margin = float(
            raw.get("minimum_goal_surface_clearance_m", 0.0)
        )
        if not np.isclose(start_margin, goal_margin, rtol=0.0, atol=0.0):
            raise ValueError("triangle scenes require equal start/goal margins")
        return cls(
            radius=effective_radius,
            minimum_surface_margin=float(
                raw.get("minimum_obstacle_surface_gap_m", 0.0)
            ),
            endpoint_margin=start_margin,
            boundary_surface_margin=float(
                raw.get("minimum_taskspace_wall_surface_clearance_m", 0.0)
            ),
            domain_seed=int(raw.get("seed", 0)),
            max_attempts=int(raw.get("max_layout_attempts", 100_000)),
            center_fraction=float(raw.get("triangle_center_fraction", 0.5)),
            center_spacing_m=float(raw["triangle_center_spacing_m"]),
            parallel_std_m=float(raw["parallel_position_std_m"]),
            plane_std_m=float(raw["perpendicular_plane_position_std_m"]),
        )

    def local_frame(self, env: TaskEnvironment) -> np.ndarray:
        """Return columns [forward, lateral, up] for this fixed task."""
        return start_goal_frame(env).astype(np.float64)

    def base_centers(self, env: TaskEnvironment) -> np.ndarray:
        """Return the exact unperturbed tight equilateral-triangle centers."""
        start = np.asarray(env.start[:3], np.float64)
        goal = np.asarray(env.goal, np.float64)
        midpoint = start + self.center_fraction * (goal - start)
        frame = self.local_frame(env)
        circumradius = self.center_spacing_m / np.sqrt(3.0)
        angles = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        local = np.column_stack([
            np.zeros(3),
            circumradius * np.sin(angles),
            circumradius * np.cos(angles),
        ])
        return midpoint[None] + local @ frame.T

    def base_scene(self, env: TaskEnvironment) -> np.ndarray:
        """Return the fixed challenge geometry without positional jitter."""
        radii = np.full((LAB_CLUTTER_SPHERE_COUNT, 1), self.radius)
        return self.validate(
            env,
            np.concatenate([self.base_centers(env), radii], axis=1),
        )

    def sample(self, env: TaskEnvironment, seed: int) -> np.ndarray:
        """Sample one geometry-only anisotropic perturbation of the triangle."""
        effective_seed = int(np.random.SeedSequence([
            int(self.domain_seed), int(seed),
        ]).generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(effective_seed)
        frame = self.local_frame(env)
        base = self.base_centers(env)
        radii = np.full((LAB_CLUTTER_SPHERE_COUNT, 1), self.radius)
        for _ in range(self.max_attempts):
            local_noise = np.column_stack([
                rng.normal(0.0, self.parallel_std_m, LAB_CLUTTER_SPHERE_COUNT),
                rng.normal(0.0, self.plane_std_m, LAB_CLUTTER_SPHERE_COUNT),
                rng.normal(0.0, self.plane_std_m, LAB_CLUTTER_SPHERE_COUNT),
            ])
            proposal = np.concatenate([
                base + local_noise @ frame.T,
                radii,
            ], axis=1)
            try:
                return self.validate(env, proposal)
            except ValueError:
                continue
        raise RuntimeError(
            "failed to sample a valid anisotropic triangle scene within "
            "max_attempts"
        )

    def sampling_contract(self) -> dict[str, float | str]:
        position_noise = (
            "independent Gaussian per sphere along its fixed "
            "start-goal-parallel line"
            if self.plane_std_m == 0.0
            else "independent Gaussian per sphere in task-local frame"
        )
        return {
            "distribution": PERPENDICULAR_TRIANGLE_DISTRIBUTION,
            "triangle_center_fraction": float(self.center_fraction),
            "triangle_center_spacing_m": float(self.center_spacing_m),
            "parallel_position_std_m": float(self.parallel_std_m),
            "perpendicular_plane_position_std_m": float(self.plane_std_m),
            "position_noise": position_noise,
            "admission": "geometry only; no controller-success conditioning",
        }


@dataclass(frozen=True)
class PathFocusedVariableSphereScene:
    """Variable-count path-focused sphere sampler with fixed-size packing."""

    spec: PathFocusedClutterSpec

    def __post_init__(self) -> None:
        if self.spec.obstacle_family != "spheres":
            raise ValueError("variable sphere task requires a sphere spec")

    @classmethod
    def from_config(cls, config) -> "PathFocusedVariableSphereScene":
        return cls(PathFocusedClutterSpec.from_config(
            config, expected_family="spheres",
        ))

    @property
    def max_count(self) -> int:
        return self.spec.count_max

    @property
    def packed_dim(self) -> int:
        return 1 + 4 * self.max_count

    @property
    def scene_schema(self) -> str:
        return self.spec.scene_schema

    @property
    def radius(self) -> float:
        return self.spec.modeled_radius_m

    @property
    def minimum_surface_margin(self) -> float:
        return self.spec.minimum_surface_gap_m

    @property
    def endpoint_margin(self) -> float:
        return self.spec.endpoint_surface_gap_m

    @property
    def boundary_surface_margin(self) -> float:
        return self.spec.boundary_surface_gap_m

    @property
    def domain_seed(self) -> int:
        return self.spec.domain_seed

    @property
    def max_attempts(self) -> int:
        return self.spec.max_layout_attempts

    def sample(self, env: TaskEnvironment, seed: int) -> np.ndarray:
        scene = self.spec.sample_scene(
            scene_index=int(seed),
            scene_seed=int(seed),
            bounds=env.bounds,
            start=env.start,
            goal=env.goal,
        )
        return self.validate(
            env, np.asarray(scene.spheres, np.float32),
        )

    def validate(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        spheres = canonical_sphere_rows(values)
        if not self.spec.count_min <= len(spheres) <= self.spec.count_max:
            raise ValueError("sphere count lies outside the configured range")
        tolerance = 2.0e-6
        if not np.allclose(
            spheres[:, 3],
            self.spec.modeled_radius_m,
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError("sphere radii do not match the path-focused contract")
        wall_clearance = np.minimum(
            spheres[:, :3] - spheres[:, 3, None]
            - env.bounds[:, 0][None],
            env.bounds[:, 1][None]
            - spheres[:, :3] - spheres[:, 3, None],
        )
        if bool((
            wall_clearance
            < self.spec.boundary_surface_gap_m - tolerance
        ).any()):
            raise ValueError("sphere violates body containment or wall margin")
        center_anchored = (
            self.spec.distribution
            == PATH_FOCUSED_CENTER_ANCHORED_DISTRIBUTION
        )
        for first in range(len(spheres)):
            for second in range(first + 1, len(spheres)):
                if center_anchored:
                    # Center-anchored contract: the configured gap measures
                    # PHYSICAL surface separation, floored by inflated-body
                    # non-overlap (matches PathFocusedClutterSpec._sample_rows).
                    required = max(
                        float(spheres[first, 3]) + float(spheres[second, 3]),
                        2.0 * float(np.float32(self.spec.physical_radius_m))
                        + self.spec.minimum_surface_gap_m,
                    )
                else:
                    required = (
                        float(spheres[first, 3])
                        + float(spheres[second, 3])
                        + self.spec.minimum_surface_gap_m
                    )
                if (
                    float(np.linalg.norm(
                        spheres[first, :3] - spheres[second, :3]
                    ))
                    < required - tolerance
                ):
                    raise ValueError("spheres violate the modeled surface gap")
        for endpoint, label in ((env.start[:3], "start"), (env.goal, "goal")):
            clearance = np.linalg.norm(
                spheres[:, :3] - endpoint[None], axis=1,
            ) - spheres[:, 3]
            if bool((
                clearance
                < self.spec.endpoint_surface_gap_m - tolerance
            ).any()):
                raise ValueError(f"sphere intersects the fixed {label}")
        return spheres

    def pack(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        spheres = self.validate(env, values)
        padded = np.zeros((self.max_count, 4), np.float32)
        padded[:len(spheres)] = spheres
        return np.concatenate([
            np.asarray([len(spheres)], np.float32),
            padded.reshape(-1),
        ])

    def unpack(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        packed = np.asarray(values, np.float32).reshape(-1)
        if len(packed) != self.packed_dim:
            raise ValueError("variable-sphere packed context has wrong length")
        count_value = float(packed[0])
        count = int(round(count_value))
        if not np.isclose(count_value, count, rtol=0.0, atol=1.0e-6):
            raise ValueError("packed sphere count is not integral")
        if not self.spec.count_min <= count <= self.spec.count_max:
            raise ValueError("packed sphere count lies outside the configured range")
        padded = packed[1:].reshape(self.max_count, 4)
        if bool(np.any(padded[count:] != 0.0)):
            raise ValueError("unused packed sphere rows must be exactly zero")
        return self.validate(env, padded[:count])


def sphere_scene_spec_from_config(config):
    randomization = config.raw.get("scene_randomization", {})
    if (
        randomization.get("distribution")
        == PERPENDICULAR_TRIANGLE_DISTRIBUTION
    ):
        return PerpendicularTriangleSphereScene.from_config(config)
    if randomization.get("distribution") in PATH_FOCUSED_DISTRIBUTIONS:
        return PathFocusedVariableSphereScene.from_config(config)
    return RandomThreeSphereScene.from_config(config)


class LabClutterExpansionPolicyAdapter(LabExpansionPolicyAdapter):
    """Strip governor memory and exact scene geometry from learned inputs."""

    def __init__(
        self,
        policy: torch.nn.Module,
        verifier_suffix_dim: int = LAB_CLUTTER_VERIFIER_SUFFIX_DIM,
        *,
        freeze_history_encoder: bool | None = None,
    ):
        if (
            getattr(policy, "context_schema", None)
            not in LAB_CLUTTER_VISUAL_CONTEXT_DIMS
        ):
            raise ValueError(
                "dynamic clutter expansion requires the visual lab policy"
            )
        super().__init__(
            policy,
            freeze_history_encoder=freeze_history_encoder,
        )
        if int(verifier_suffix_dim) < LAB_CLUTTER_GOVERNOR_DIM:
            raise ValueError("clutter verifier suffix is too short")
        self.verifier_suffix_dim = int(verifier_suffix_dim)
        self.context_dim = self.policy_context_dim + self.verifier_suffix_dim


def load_lab_clutter_expansion_policy(
    path: str | Path,
    *,
    verifier_suffix_dim: int = LAB_CLUTTER_VERIFIER_SUFFIX_DIM,
    train_history_encoder: bool = False,
) -> LabClutterExpansionPolicyAdapter:
    policy = load_lab_reference_policy(path)
    is_history = _is_history_context_schema(
        getattr(policy, "context_schema", ""),
    )
    if train_history_encoder and not is_history:
        raise ValueError(
            "train_history_encoder applies only to a GRU checkpoint"
        )
    return LabClutterExpansionPolicyAdapter(
        policy,
        verifier_suffix_dim=verifier_suffix_dim,
        freeze_history_encoder=(
            not train_history_encoder if is_history else None
        ),
    )


class LabClutterSphereExpansionTask:
    """Expansion task with a deterministic independent sphere scene per reset."""

    def __init__(
        self,
        config,
        *,
        context_schema: str,
        device: str | torch.device = "cpu",
        tight_corridor: bool = False,
        execution_z_bias_mode: str = "none",
        verifier_mode: str = "full_polytope",
        verifier_solver: str = "analytic",
        execution_clearance_exp_weight: float = 0.0,
        execution_clearance_exp_temperature: float = 0.10,
        execution_clearance_target_m: float = 0.20,
        execution_soft_clearance_weight: float | None = None,
        execution_soft_clearance_target_m: float | None = None,
        execution_taskspace_weight: float | None = None,
        execution_plane_penalty_weight: float = 0.0,
        execution_plane_penalty_sigma: float = 0.15,
        execution_plane_penalty_shape: str = "gaussian",
        execution_above_penalty_weight: float = 0.0,
        scene_spec=None,
    ):
        if context_schema not in LAB_CLUTTER_VISUAL_CONTEXT_DIMS:
            raise ValueError(
                "dynamic clutter expansion requires lab visual conditioning"
            )
        if execution_z_bias_mode != "none":
            raise ValueError("clutter expansion excludes every z-bias cost")
        if verifier_mode != "full_polytope":
            raise ValueError(
                "three-sphere clutter requires verifier_mode='full_polytope'"
            )
        if verifier_solver not in {"analytic", "cvxpy"}:
            raise ValueError(f"unknown verifier_solver: {verifier_solver}")
        if (
            not np.isfinite(execution_clearance_exp_weight)
            or execution_clearance_exp_weight < 0.0
        ):
            raise ValueError(
                "execution_clearance_exp_weight must be finite and nonnegative"
            )
        if (
            not np.isfinite(execution_clearance_exp_temperature)
            or execution_clearance_exp_temperature <= 0.0
        ):
            raise ValueError(
                "execution_clearance_exp_temperature must be finite and positive"
            )
        if (
            not np.isfinite(execution_clearance_target_m)
            or execution_clearance_target_m < 0.0
        ):
            raise ValueError(
                "execution_clearance_target_m must be finite and nonnegative"
            )
        configured_soft_weight = float(
            config.safemppi.soft_clearance_weight
        )
        configured_soft_target = float(
            config.safemppi.soft_clearance_target
        )
        effective_soft_weight = (
            configured_soft_weight
            if execution_soft_clearance_weight is None
            else float(execution_soft_clearance_weight)
        )
        effective_soft_target = (
            configured_soft_target
            if execution_soft_clearance_target_m is None
            else float(execution_soft_clearance_target_m)
        )
        effective_taskspace_weight = (
            float(config.safemppi.taskspace_exponential_weight)
            if execution_taskspace_weight is None
            else float(execution_taskspace_weight)
        )
        if (
            not np.isfinite(effective_soft_weight)
            or effective_soft_weight < 0.0
        ):
            raise ValueError(
                "execution_soft_clearance_weight must be finite and nonnegative"
            )
        if (
            not np.isfinite(effective_soft_target)
            or effective_soft_target < 0.0
        ):
            raise ValueError(
                "execution_soft_clearance_target_m must be finite and nonnegative"
            )
        if (
            not np.isfinite(effective_taskspace_weight)
            or effective_taskspace_weight < 0.0
        ):
            raise ValueError(
                "execution_taskspace_weight must be finite and nonnegative"
            )
        if (
            not np.isfinite(execution_plane_penalty_weight)
            or execution_plane_penalty_weight < 0.0
            or not np.isfinite(execution_above_penalty_weight)
            or execution_above_penalty_weight < 0.0
        ):
            raise ValueError(
                "execution plane/above penalty weights must be finite and "
                "nonnegative"
            )
        if (
            not np.isfinite(execution_plane_penalty_sigma)
            or execution_plane_penalty_sigma <= 0.0
        ):
            raise ValueError(
                "execution_plane_penalty_sigma must be finite and positive"
            )
        if execution_plane_penalty_shape not in {"gaussian", "laplacian"}:
            raise ValueError(
                "execution_plane_penalty_shape must be gaussian or laplacian"
            )
        self.config = config
        self.env = TaskEnvironment(config)
        if not np.allclose(
            self.env.bounds[:2], LAB_BOUNDS[:2], rtol=0.0, atol=1.0e-9,
        ):
            raise ValueError(
                "sphere expansion requires the fixed Minhyuk lab x/y bounds"
            )
        if not (
            np.isfinite(self.env.bounds[2]).all()
            and self.env.bounds[2, 0] < 0.9 < self.env.bounds[2, 1]
        ):
            raise ValueError(
                "sphere expansion requires a finite z range containing the "
                "start/goal plane z=0.9"
            )
        if not np.allclose(
            self.env.start, LAB_START, rtol=0.0, atol=1.0e-6,
        ):
            raise ValueError(
                "sphere expansion requires the fixed Minhyuk lab start"
            )
        if not np.allclose(
            self.env.goal, LAB_GOAL, rtol=0.0, atol=1.0e-6,
        ):
            raise ValueError(
                "sphere expansion requires the fixed Minhyuk lab goal"
            )
        if self.config.safemppi.z_bias_weight != 0.0:
            raise ValueError("sphere expansion requires zero z-bias weight")
        if (
            len(self.config.obstacles.spheres) != 0
            or len(self.config.obstacles.cylinders) != 0
        ):
            raise ValueError(
                "dynamic sphere expansion requires an empty static obstacle list"
            )
        self.context_schema = context_schema
        self.policy_context_dim = LAB_CLUTTER_VISUAL_CONTEXT_DIMS[
            context_schema
        ]
        self.device = torch.device(device)
        self.tight_corridor = bool(tight_corridor)
        self.verifier_mode = verifier_mode
        self.verifier_solver = verifier_solver
        self.execution_clearance_exp_weight = float(
            execution_clearance_exp_weight
        )
        self.execution_clearance_exp_temperature = float(
            execution_clearance_exp_temperature
        )
        self.execution_clearance_target_m = float(
            execution_clearance_target_m
        )
        self.execution_soft_clearance_weight = effective_soft_weight
        self.execution_soft_clearance_target_m = effective_soft_target
        self.execution_taskspace_weight = effective_taskspace_weight
        self.execution_plane_penalty_weight = float(
            execution_plane_penalty_weight
        )
        self.execution_plane_penalty_sigma = float(
            execution_plane_penalty_sigma
        )
        self.execution_plane_penalty_shape = str(
            execution_plane_penalty_shape
        )
        self.execution_above_penalty_weight = float(
            execution_above_penalty_weight
        )
        self.execution_plane_penalty_plane_m = float(
            config.safemppi.z_bias_plane
        )
        self.scene_spec = (
            scene_spec
            if scene_spec is not None
            else sphere_scene_spec_from_config(config)
        )
        self.verifier_suffix_dim = (
            LAB_CLUTTER_GOVERNOR_DIM + self.scene_spec.packed_dim
        )
        self.scene_schema = self.scene_spec.scene_schema
        self.scene_ledger: list[dict[str, Any]] = []
        delta = self.env.goal - self.env.start[:3]
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-12:
            raise ValueError("fixed start and goal must be distinct")
        self.forward = delta / length
        self.reference_z = 0.5 * float(
            self.env.start[2] + self.env.goal[2]
        )

    def _environment(self, spheres: np.ndarray) -> TaskEnvironment:
        spheres = self.scene_spec.validate(self.env, spheres)
        obstacles = replace(
            self.config.obstacles,
            spheres=tuple(tuple(map(float, row)) for row in spheres),
            cylinders=(),
        )
        return TaskEnvironment(replace(self.config, obstacles=obstacles))

    def reset(self, gamma: float, episode: int, seed: int) -> dict[str, Any]:
        spheres = self.scene_spec.sample(self.env, int(seed))
        scene_env = self._environment(spheres)
        state = {
            "x": scene_env.start.copy(),
            "previous_applied": np.zeros(3, np.float32),
            "previous_raw": np.zeros(3, np.float32),
            "spheres": spheres,
            "scene_seed": int(seed),
            "scene_hash": scene_sha256(scene_env, spheres),
            "steps": 0,
            "collided": False,
            "oob": False,
        }
        if _is_history_context_schema(self.context_schema):
            state["raw_history"] = _raw_history_before(
                np.empty((0, 3), np.float32), 0,
            )
        self.scene_ledger.append({
            "reset_index": len(self.scene_ledger),
            "gamma": float(gamma),
            "episode": int(episode),
            **self.scene_metadata(state),
        })
        return state

    def scene_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        spheres = self.scene_spec.validate(self.env, state["spheres"])
        scene_hash = str(state["scene_hash"])
        scene = ClutterScene(
            index=int(state["scene_seed"]),
            seed=int(state["scene_seed"]),
            spheres=tuple(
                tuple(map(float, row)) for row in spheres
            ),
            cylinders=(),
            scene_hash=scene_hash,
        )
        return {
            "schema": self.scene_schema,
            "domain_seed": int(self.scene_spec.domain_seed),
            "scene_seed": int(state["scene_seed"]),
            "scene_hash": scene_hash,
            "obstacle_count": len(spheres),
            "spheres": spheres.tolist(),
            "start_goal_path_diagnostics": start_goal_path_diagnostics(
                scene,
                start=self.env.start,
                goal=self.env.goal,
                soft_clearance_target_m=(
                    self.config.safemppi.soft_clearance_target
                ),
            ),
        }

    def context(self, state, gamma: float) -> torch.Tensor:
        spheres = self.scene_spec.validate(self.env, state["spheres"])
        scene_env = self._environment(spheres)
        learned = policy_context(
            SimpleNamespace(context_schema=self.context_schema),
            scene_env,
            state["x"],
            float(gamma),
            raw_history=state.get("raw_history"),
            previous_raw=state["previous_raw"],
            previous_applied=state["previous_applied"],
        )
        packed = np.concatenate([
            learned,
            np.asarray(state["previous_applied"], np.float32),
            np.asarray(state["previous_raw"], np.float32),
            self.scene_spec.pack(scene_env, spheres),
        ]).astype(np.float32, copy=False)
        return torch.from_numpy(packed).to(self.device)

    def _decode_context(
        self, context: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TaskEnvironment]:
        values = (
            context.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        expected = (
            self.policy_context_dim + self.verifier_suffix_dim
        )
        if values.shape != (expected,):
            raise ValueError(
                f"clutter verifier context must have shape ({expected},)"
            )
        position = self.env.goal - values[:3]
        velocity = values[3:6]
        suffix = self.policy_context_dim
        previous_applied = values[suffix:suffix + 3].copy()
        previous_raw = values[suffix + 3:suffix + 6].copy()
        spheres = self.scene_spec.unpack(
            self.env,
            values[suffix + LAB_CLUTTER_GOVERNOR_DIM:],
        )
        scene_env = self._environment(spheres)
        return (
            np.concatenate([position, velocity]).astype(np.float32),
            previous_applied,
            previous_raw,
            scene_env,
        )

    def scene_from_context(self, context: torch.Tensor) -> np.ndarray:
        values = (
            context.detach().cpu().numpy().astype(np.float32, copy=False)
        ).reshape(-1)
        expected = self.policy_context_dim + self.verifier_suffix_dim
        if len(values) != expected:
            raise ValueError("clutter context has the wrong packed length")
        start = self.policy_context_dim + LAB_CLUTTER_GOVERNOR_DIM
        return self.scene_spec.unpack(self.env, values[start:])

    def _rollout_plan(
        self,
        state6: np.ndarray,
        previous_applied: np.ndarray,
        plan: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        governor = ReferenceGovernor(self.config.safemppi)
        governor.previous_applied = np.asarray(
            previous_applied, np.float32,
        ).copy()
        states = [np.asarray(state6, np.float32).copy()]
        applied, dense_steps = [], []
        state = states[0]
        for command in np.asarray(plan, np.float32).reshape(-1, 3):
            state, applied_command, dense = governor.step(state, command)
            states.append(state.copy())
            applied.append(applied_command.copy())
            dense_steps.append(dense.copy())
        return (
            np.asarray(states, np.float32),
            np.asarray(applied, np.float32),
            np.asarray(dense_steps, np.float32),
        )

    def _native_cost(
        self,
        env: TaskEnvironment,
        states: np.ndarray,
        plan: np.ndarray,
        previous_raw: np.ndarray,
    ) -> float:
        """Configured lab cost on all spheres, with no z-bias term."""
        cfg = self.config.safemppi
        plan = np.asarray(plan, np.float32).reshape(-1, 3)
        initial_distance = float(
            np.linalg.norm(states[0, :3] - env.goal)
        )
        previous = np.asarray(previous_raw, np.float32)
        cost = 0.0
        predicted_clearances = []
        for index, command in enumerate(plan):
            point = states[index + 1, :3]
            distance = float(np.linalg.norm(point - env.goal))
            cost += cfg.running_goal_weight * distance ** 2
            cost += cfg.control_weight * float(command @ command)
            difference = command - previous
            cost += cfg.smooth_weight * float(difference @ difference)
            cost -= cfg.progress_weight * (initial_distance - distance)
            clearance = float(env.obstacle_clearance(point[None])[0])
            predicted_clearances.append(clearance)
            cost += self.execution_soft_clearance_weight * max(
                self.execution_soft_clearance_target_m - clearance, 0.0,
            ) ** 2
            violation = (
                np.maximum(env.bounds[:, 0] - point, 0.0)
                + np.maximum(point - env.bounds[:, 1], 0.0)
            )
            exponent = np.minimum(
                violation / cfg.taskspace_exponential_temperature, 20.0,
            )
            cost += self.execution_taskspace_weight * float(
                np.expm1(exponent).sum()
            )
            if (
                self.execution_plane_penalty_weight > 0.0
                or self.execution_above_penalty_weight > 0.0
            ):
                dz = float(point[2]) - self.execution_plane_penalty_plane_m
                # Gaussian decays as exp(-0.5 (dz/sigma)^2) and is already
                # negligible past ~2 sigma; the Laplacian shape keeps a
                # non-vanishing slope across the whole z box, so a broad,
                # heavy in-plane penalty can still discriminate the off-plane
                # candidates that below/above routes actually occupy.
                if self.execution_plane_penalty_shape == "laplacian":
                    kernel = np.exp(
                        -0.5 * abs(dz) / self.execution_plane_penalty_sigma
                    )
                else:
                    kernel = np.exp(
                        -0.5 * (dz / self.execution_plane_penalty_sigma) ** 2
                    )
                cost += self.execution_plane_penalty_weight * float(kernel)
                cost += self.execution_above_penalty_weight * max(dz, 0.0)
            previous = command
        terminal_distance = float(
            np.linalg.norm(states[-1, :3] - env.goal)
        )
        cost += cfg.terminal_goal_weight * terminal_distance ** 2
        if self.execution_clearance_exp_weight > 0.0:
            exponent = (
                self.execution_clearance_target_m
                - np.asarray(predicted_clearances, np.float64)
            ) / self.execution_clearance_exp_temperature
            cost += self.execution_clearance_exp_weight * float(
                np.exp(exponent).mean()
            )
        return float(cost)

    def _verify_plan(
        self,
        context: torch.Tensor,
        candidate: torch.Tensor,
        gamma: float,
    ) -> Verification:
        state6, previous_applied, previous_raw, env = self._decode_context(
            context
        )
        plan = candidate.detach().cpu().numpy().reshape(-1, 3)
        if not 1 <= len(plan) <= PLAN_H:
            raise ValueError(f"verifier plan horizon must lie in [1,{PLAN_H}]")
        states, _, dense_steps = self._rollout_plan(
            state6, previous_applied, plan,
        )
        dense = np.concatenate([
            states[:1, :3],
            dense_steps.reshape(-1, 3),
        ])
        executed_dense = np.concatenate([
            states[:1, :3],
            dense_steps[:1].reshape(-1, 3),
        ])
        inside = bool(env.inside_taskspace(executed_dense).all())
        clearance = env.obstacle_clearance(dense)
        no_collision = bool(
            np.isinf(clearance).all() or float(clearance.min()) > 0.0
        )
        certified, verifier_slack = certify_window(
            states[:, :3],
            env.spheres,
            env.cylinders,
            float(gamma),
            self.config.safemppi.sensing_range,
            face_solver=self.verifier_solver,
        )
        step_margin = nominal_hp_chain_margins(
            env,
            states[:2, :3],
            float(gamma),
            self.config.safemppi.sensing_range,
        )[0]
        longitudinal = states[:, :3] @ self.forward
        progress = float(np.diff(longitudinal).min())
        return Verification(
            valid=bool(inside and no_collision and certified),
            hp_eligible=bool(step_margin > 0.0),
            margin=float(verifier_slack),
            execution_cost=self._native_cost(
                env, states, plan, previous_raw,
            ),
            progress=progress,
            progress_eligible=bool(progress > 0.0),
            target_eligible=True,
            step_margin=float(step_margin),
        )

    def verify(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        gamma: float,
    ):
        return [
            self._verify_plan(context, candidate, gamma)
            for candidate in candidates
        ]

    def advance(self, state, candidate: torch.Tensor):
        spheres = self.scene_spec.validate(self.env, state["spheres"])
        env = self._environment(spheres)
        command = candidate.detach().cpu().numpy().reshape(-1, 3)[0]
        limited_command = np.clip(
            command,
            -float(self.config.safemppi.demo_u_max),
            float(self.config.safemppi.demo_u_max),
        ).astype(np.float32)
        governor = ReferenceGovernor(self.config.safemppi)
        governor.previous_applied = np.asarray(
            state["previous_applied"], np.float32,
        ).copy()
        after, applied, dense = governor.step(state["x"], command)
        clearance = env.obstacle_clearance(dense)
        updated = {
            "x": after,
            "previous_applied": applied,
            "previous_raw": limited_command,
            "spheres": spheres,
            "scene_seed": int(state["scene_seed"]),
            "scene_hash": str(state["scene_hash"]),
            "steps": int(state["steps"]) + 1,
            "collided": bool(
                state["collided"]
                or (
                    np.isfinite(clearance).any()
                    and float(clearance.min()) < 0.0
                )
            ),
            "oob": bool(
                state["oob"]
                or not env.inside_taskspace(dense).all()
            ),
        }
        if _is_history_context_schema(self.context_schema):
            updated["raw_history"] = _append_raw_history(
                state["raw_history"], command,
            )
        return updated

    def terminal(self, state):
        if state["collided"]:
            return "COLLISION"
        if state["oob"]:
            return "OOB"
        if self.env.reached(state["x"][:3]):
            return "SUCCESS"
        return None

    def successful_trajectory_above_fraction(self, executed_states) -> float:
        if not executed_states:
            return 0.0
        return float(np.mean([
            float(np.asarray(state["x"])[2]) >= self.reference_z
            for state in executed_states
        ]))

    def successful_trajectory_mean_z(self, executed_states) -> float:
        if not executed_states:
            return -float("inf")
        return float(np.mean([
            float(np.asarray(state["x"])[2])
            for state in executed_states
        ]))
