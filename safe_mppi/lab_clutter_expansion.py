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
from .environment import ReferenceGovernor, TaskEnvironment
from .expansion import Verification
from .lab_flow_expansion import LabExpansionPolicyAdapter
from .lab_clutter import (
    ClutterScene,
    LAB_BOUNDS,
    LAB_GOAL,
    LAB_START,
    obstacle_scene_hash,
    start_goal_path_diagnostics,
)
from .lab_reference_flow_task import policy_context
from .lab_visual_flow import (
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    load_lab_reference_policy,
)
from .verifier_polytope import certify_window


LAB_CLUTTER_SPHERE_COUNT = 3
LAB_CLUTTER_SCENE_DIM = 4 * LAB_CLUTTER_SPHERE_COUNT
LAB_CLUTTER_GOVERNOR_DIM = 6
LAB_CLUTTER_VERIFIER_SUFFIX_DIM = (
    LAB_CLUTTER_GOVERNOR_DIM + LAB_CLUTTER_SCENE_DIM
)
LAB_CLUTTER_SCENE_SCHEMA = "lab_random_three_spheres_v1"


def canonical_spheres(values: np.ndarray) -> np.ndarray:
    """Return three finite spheres in deterministic lexicographic order."""
    spheres = np.asarray(values, np.float32).reshape(-1, 4)
    if spheres.shape != (LAB_CLUTTER_SPHERE_COUNT, 4):
        raise ValueError("lab clutter scenes require exactly three spheres")
    if not np.isfinite(spheres).all() or bool((spheres[:, 3] <= 0.0).any()):
        raise ValueError("sphere centers/radii must be finite and radii positive")
    order = np.lexsort((
        spheres[:, 3],
        spheres[:, 2],
        spheres[:, 1],
        spheres[:, 0],
    ))
    return np.ascontiguousarray(spheres[order], dtype=np.float32)


def scene_sha256(env: TaskEnvironment, spheres: np.ndarray) -> str:
    """Hash exact obstacle geometry using the shared clutter-archive contract."""
    del env
    return obstacle_scene_hash(spheres=canonical_spheres(spheres))


@dataclass(frozen=True)
class RandomThreeSphereScene:
    """Deterministic sampler for the configured three-sphere safety geometry."""

    radius: float = 0.379
    minimum_surface_margin: float = 0.20
    endpoint_margin: float = 0.50
    boundary_surface_margin: float = 0.10
    domain_seed: int = 0
    max_attempts: int = 10_000

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


class LabClutterExpansionPolicyAdapter(LabExpansionPolicyAdapter):
    """Strip governor memory and exact scene geometry from learned inputs."""

    def __init__(self, policy: torch.nn.Module):
        if getattr(policy, "context_schema", None) != LAB_VISUAL_SCHEMA:
            raise ValueError(
                "dynamic clutter expansion requires the visual lab policy"
            )
        super().__init__(policy)
        self.context_dim = (
            self.policy_context_dim + LAB_CLUTTER_VERIFIER_SUFFIX_DIM
        )


def load_lab_clutter_expansion_policy(
    path: str | Path,
) -> LabClutterExpansionPolicyAdapter:
    return LabClutterExpansionPolicyAdapter(load_lab_reference_policy(path))


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
        scene_spec: RandomThreeSphereScene | None = None,
    ):
        if context_schema != LAB_VISUAL_SCHEMA:
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
        self.config = config
        self.env = TaskEnvironment(config)
        if not np.allclose(
            self.env.bounds, LAB_BOUNDS, rtol=0.0, atol=1.0e-9,
        ):
            raise ValueError(
                "sphere expansion requires the fixed Minhyuk lab bounds"
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
        self.policy_context_dim = LAB_VISUAL_PACKED_DIM
        self.device = torch.device(device)
        self.tight_corridor = bool(tight_corridor)
        self.verifier_mode = verifier_mode
        self.verifier_solver = verifier_solver
        self.scene_spec = (
            scene_spec
            if scene_spec is not None
            else RandomThreeSphereScene.from_config(config)
        )
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
            "schema": LAB_CLUTTER_SCENE_SCHEMA,
            "domain_seed": int(self.scene_spec.domain_seed),
            "scene_seed": int(state["scene_seed"]),
            "scene_hash": scene_hash,
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
        )
        packed = np.concatenate([
            learned,
            np.asarray(state["previous_applied"], np.float32),
            np.asarray(state["previous_raw"], np.float32),
            spheres.reshape(-1),
        ]).astype(np.float32, copy=False)
        return torch.from_numpy(packed).to(self.device)

    def _decode_context(
        self, context: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TaskEnvironment]:
        values = (
            context.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        expected = (
            self.policy_context_dim + LAB_CLUTTER_VERIFIER_SUFFIX_DIM
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
        spheres = values[
            suffix + LAB_CLUTTER_GOVERNOR_DIM:
        ].reshape(LAB_CLUTTER_SPHERE_COUNT, 4)
        scene_env = self._environment(spheres)
        return (
            np.concatenate([position, velocity]).astype(np.float32),
            previous_applied,
            previous_raw,
            scene_env,
        )

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
        for index, command in enumerate(plan):
            point = states[index + 1, :3]
            distance = float(np.linalg.norm(point - env.goal))
            cost += cfg.running_goal_weight * distance ** 2
            cost += cfg.control_weight * float(command @ command)
            difference = command - previous
            cost += cfg.smooth_weight * float(difference @ difference)
            cost -= cfg.progress_weight * (initial_distance - distance)
            clearance = float(env.obstacle_clearance(point[None])[0])
            cost += cfg.soft_clearance_weight * max(
                cfg.soft_clearance_target - clearance, 0.0,
            ) ** 2
            violation = (
                np.maximum(env.bounds[:, 0] - point, 0.0)
                + np.maximum(point - env.bounds[:, 1], 0.0)
            )
            exponent = np.minimum(
                violation / cfg.taskspace_exponential_temperature, 20.0,
            )
            cost += cfg.taskspace_exponential_weight * float(
                np.expm1(exponent).sum()
            )
            previous = command
        terminal_distance = float(
            np.linalg.norm(states[-1, :3] - env.goal)
        )
        cost += cfg.terminal_goal_weight * terminal_distance ** 2
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
        governor = ReferenceGovernor(self.config.safemppi)
        governor.previous_applied = np.asarray(
            state["previous_applied"], np.float32,
        ).copy()
        after, applied, dense = governor.step(state["x"], command)
        clearance = env.obstacle_clearance(dense)
        return {
            "x": after,
            "previous_applied": applied,
            "previous_raw": np.asarray(command, np.float32).copy(),
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
