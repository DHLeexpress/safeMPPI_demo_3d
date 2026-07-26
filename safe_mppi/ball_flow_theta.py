"""Local-frame, explicitly mode-conditioned adapter for the 3-D ball task.

The learned policy sees local vectors and emits local accelerations.  Safety
verification and dynamics remain unchanged in world coordinates.  A single
detour angle is fixed for an episode, resolving the otherwise unobservable
choice between symmetric routes around the ball.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .ball_flow_task import (
    PLAN_H,
    BallFlowTask,
    build_context,
    closest_approach_angular_descriptor,
    closest_boundary_point,
    inside_expansion_corridor,
)
from .config import ExperimentConfig, load_config
from .environment import TaskEnvironment

CONTEXT_DIM = 12
THETA_VALUES = (-0.5 * np.pi, 0.5 * np.pi, 0.0, np.pi)


def start_goal_frame(
    env: TaskEnvironment,
    world_up: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return columns ``[e_parallel, e_1, e_2]`` of the fixed task frame."""
    forward = np.asarray(env.goal, float) - np.asarray(env.start[:3], float)
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-12:
        raise ValueError("task start and goal must define a nonzero longitudinal axis")
    forward /= norm

    up = np.asarray(world_up, float).reshape(3).copy()
    up -= float(up @ forward) * forward
    norm = float(np.linalg.norm(up))
    if norm <= 1.0e-12:
        raise ValueError("world_up must not be parallel to the start-goal axis")
    up /= norm
    lateral = np.cross(up, forward)
    lateral /= np.linalg.norm(lateral)
    return np.column_stack([forward, lateral, up]).astype(np.float32)


def world_to_local(vectors: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Rotate row-vector world coordinates into the task-local frame."""
    values = np.asarray(vectors)
    if values.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of three")
    return (values @ np.asarray(frame).reshape(3, 3)).astype(values.dtype, copy=False)


def local_to_world(vectors: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Rotate row-vector task-local coordinates back into the world frame."""
    values = np.asarray(vectors)
    if values.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of three")
    return (values @ np.asarray(frame).reshape(3, 3).T).astype(values.dtype, copy=False)


def build_theta_context(
    env: TaskEnvironment,
    state6: np.ndarray,
    gamma: float,
    theta: float,
    frame: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``[(g-p)_L, v_L, (b_near-p)_L, gamma, cos(theta), sin(theta)]``."""
    state6 = np.asarray(state6, np.float32)
    basis = start_goal_frame(env) if frame is None else np.asarray(frame, np.float32)
    position, velocity = state6[:3], state6[3:6]
    boundary = closest_boundary_point(env, position).astype(np.float32)
    return np.concatenate([
        world_to_local(env.goal - position, basis),
        world_to_local(velocity, basis),
        world_to_local(boundary - position, basis),
        np.asarray([gamma, np.cos(theta), np.sin(theta)], np.float32),
    ]).astype(np.float32)


def theta_context_state(
    env: TaskEnvironment,
    context: np.ndarray,
    frame: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Recover world ``[p,v]`` and the requested angle from a 12-D context."""
    context = np.asarray(context, np.float32)
    if context.shape != (CONTEXT_DIM,):
        raise ValueError(f"theta context must have shape ({CONTEXT_DIM},)")
    basis = start_goal_frame(env) if frame is None else np.asarray(frame, np.float32)
    position = env.goal - local_to_world(context[:3], basis)
    velocity = local_to_world(context[3:6], basis)
    theta = float(np.arctan2(context[11], context[10]))
    return np.concatenate([position, velocity]).astype(np.float32), theta


def trajectory_crossing_theta(
    env: TaskEnvironment,
    positions: np.ndarray,
    frame: np.ndarray | None = None,
) -> float | None:
    """Angle of the first forward crossing of the ball's axial plane."""
    positions = np.asarray(positions, float).reshape(-1, 3)
    if len(positions) < 2:
        return None
    basis = start_goal_frame(env) if frame is None else np.asarray(frame, float)
    center = np.asarray(env.spheres[0, :3], float)
    axial = (positions - center) @ basis[:, 0]
    crossings = np.flatnonzero((axial[:-1] <= 0.0) & (axial[1:] >= 0.0))
    if not len(crossings):
        return None
    index = int(crossings[0])
    denominator = axial[index + 1] - axial[index]
    fraction = 0.0 if abs(denominator) <= 1.0e-12 else -axial[index] / denominator
    point = positions[index] + fraction * (positions[index + 1] - positions[index])
    transverse = world_to_local(point - center, basis)[1:]
    if float(np.linalg.norm(transverse)) <= 1.0e-12:
        return None
    return float(np.arctan2(transverse[1], transverse[0]))


def theta_name(theta: float | None) -> str:
    """Map a crossing angle to below/above/left/right."""
    if theta is None:
        return "none"
    angle = float(np.degrees(theta))
    if -135.0 <= angle < -45.0:
        return "below"
    if 45.0 <= angle < 135.0:
        return "above"
    if -45.0 <= angle < 45.0:
        return "left"
    return "right"


def requested_theta(episode: int) -> float:
    """Deterministic balanced episode schedule: below, above, left, right."""
    return float(THETA_VALUES[int(episode) % len(THETA_VALUES)])


class ThetaBallFlowTask(BallFlowTask):
    """Expansion task with local controls and one fixed detour angle per episode."""

    def __init__(
        self,
        config: ExperimentConfig,
        device: str | torch.device = "cpu",
        execution_z_bias_mode: str = "none",
        tight_corridor: bool = False,
        target_region: str = "above_wedge",
        verifier_mode: str = "full_polytope",
        verifier_solver: str = "analytic",
    ):
        super().__init__(
            config,
            device=device,
            start_diversity=False,
            execution_z_bias_mode=execution_z_bias_mode,
            tight_corridor=tight_corridor,
            target_region=target_region,
            verifier_mode=verifier_mode,
            verifier_solver=verifier_solver,
        )
        self.frame = start_goal_frame(self.env)

    def reset(self, gamma: float, episode: int, seed: int):
        theta = requested_theta(episode)
        return {
            "x": self.env.start.copy(),
            "steps": 0,
            "collided": False,
            "oob": False,
            "theta": theta,
            "requested_route": theta_name(theta),
        }

    def context(self, state, gamma: float) -> torch.Tensor:
        values = build_theta_context(
            self.env, state["x"], gamma, state["theta"], self.frame,
        )
        return torch.from_numpy(values).to(self.device)

    def _world_context(self, context: torch.Tensor) -> torch.Tensor:
        state6, _ = theta_context_state(
            self.env, context.detach().cpu().numpy(), self.frame,
        )
        values = build_context(self.env, state6, float(context[9]))
        return torch.as_tensor(values, dtype=context.dtype, device=context.device)

    def world_plan(self, local_plan: torch.Tensor) -> torch.Tensor:
        """Convert one or a batch of policy-local control windows to world axes."""
        frame = torch.as_tensor(
            self.frame, dtype=local_plan.dtype, device=local_plan.device,
        )
        return local_plan @ frame.T

    def verify(self, context: torch.Tensor, candidates: torch.Tensor, gamma: float):
        world_context = self._world_context(context)
        world_candidates = self.world_plan(candidates)
        return [
            super(ThetaBallFlowTask, self)._verify_plan(
                world_context, candidate, gamma,
            )
            for candidate in world_candidates
        ]

    def advance(self, state, candidate: torch.Tensor):
        world_candidate = self.world_plan(candidate)
        updated = super().advance(state, world_candidate)
        updated["theta"] = state["theta"]
        updated["requested_route"] = state["requested_route"]
        return updated

    def angular_descriptors(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        state6, _ = theta_context_state(
            self.env, context.detach().cpu().numpy(), self.frame,
        )
        world_candidates = self.world_plan(candidates)
        values = [
            closest_approach_angular_descriptor(
                self.env, state6, candidate.detach().cpu().numpy(),
            )
            for candidate in world_candidates
        ]
        return torch.as_tensor(
            np.asarray(values), device=candidates.device, dtype=candidates.dtype,
        )

    def target_region_eligibility(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return super().target_region_eligibility(
            self._world_context(context), self.world_plan(candidates),
        )

    def above_wedge_eligibility(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return super().above_wedge_eligibility(
            self._world_context(context), self.world_plan(candidates),
        )

    def d4_replay_batch(self, contexts: torch.Tensor, candidates: torch.Tensor):
        raise ValueError(
            "theta-conditioned expansion does not use replay augmentation"
        )


def demo_windows_theta(
    run_dir: str | Path,
    per_gamma_limit: int | None = None,
):
    """Load successful expert trajectories and express their windows locally.

    When ``per_gamma_limit`` is supplied, exactly that many successful
    trajectories are selected for every declared gamma or the loader fails.
    Each trajectory receives one angle inferred at its ball-plane crossing.
    """
    if per_gamma_limit is not None and per_gamma_limit < 1:
        raise ValueError("per_gamma_limit must be positive")
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = load_config(run_dir / "resolved_config.json")
    env = TaskEnvironment(config)
    frame = start_goal_frame(env)
    gammas = tuple(float(value) for value in manifest.get("gammas", config.data.gammas))

    rows_by_gamma = {gamma: [] for gamma in gammas}
    for row in manifest["runs"]:
        gamma = float(row["gamma"])
        if gamma in rows_by_gamma and bool(row.get("success", False)):
            rows_by_gamma[gamma].append(row)

    selected = []
    for gamma in gammas:
        rows = sorted(rows_by_gamma[gamma], key=lambda row: int(row["seed"]))
        if per_gamma_limit is not None:
            if len(rows) < per_gamma_limit:
                raise ValueError(
                    f"gamma {gamma:g} has {len(rows)} successful trajectories; "
                    f"{per_gamma_limit} required"
                )
            rows = rows[:per_gamma_limit]
        selected.extend(rows)

    contexts, plans, meta = [], [], []
    for row in selected:
        data = np.load(run_dir / row["file"], allow_pickle=True)
        states = np.asarray(data["states"], np.float32)
        controls_world = np.asarray(data["controls"], np.float32)
        theta = trajectory_crossing_theta(env, states[:, :3], frame)
        if theta is None:
            raise ValueError(
                f"successful demo {row['file']} does not cross the ball axial plane"
            )
        controls_local = world_to_local(controls_world, frame)
        gamma = float(row["gamma"])
        for start in range(len(controls_local) - PLAN_H + 1):
            contexts.append(
                build_theta_context(env, states[start], gamma, theta, frame)
            )
            plans.append(controls_local[start:start + PLAN_H])
            meta.append({
                "gamma": gamma,
                "seed": int(row["seed"]),
                "t": start,
                "theta": theta,
                "requested_route": theta_name(theta),
            })
    return (
        np.asarray(contexts, np.float32),
        np.asarray(plans, np.float32),
        meta,
        config,
    )


@torch.no_grad()
def raw_rollout_theta(
    policy,
    config: ExperimentConfig,
    gamma: float,
    seed: int,
    theta: float,
    device: str | torch.device = "cpu",
    max_steps: int | None = None,
    start: np.ndarray | None = None,
    tight_corridor: bool = False,
):
    """Bare temperature-1 rollout of a theta-conditioned local-control policy."""
    env = TaskEnvironment(config)
    frame = start_goal_frame(env)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    state = env.start.copy() if start is None else np.asarray(start, np.float32).copy()
    states, controls = [state.copy()], []
    status = "TIMEOUT"
    physical_collision = False
    corridor_violation = False
    for _ in range(max_steps or config.taskspace.max_steps):
        context = torch.from_numpy(
            build_theta_context(env, state, gamma, theta, frame)
        ).to(device)
        local_plan = policy.sample(context, 1, generator)[0].detach().cpu().numpy()
        control = local_to_world(local_plan.reshape(PLAN_H, 3)[0], frame)
        state = env.step(state, control)
        states.append(state.copy())
        controls.append(control)
        dense = env.dense_positions(
            np.asarray(states[-2:], np.float32), control[None].astype(np.float32),
        )
        clearance = env.obstacle_clearance(dense)
        physical_collision = bool(
            physical_collision
            or (np.isfinite(clearance).any() and float(clearance.min()) < 0.0)
        )
        corridor_violation = bool(
            corridor_violation
            or (tight_corridor and not inside_expansion_corridor(dense).all())
        )
        if physical_collision:
            status = "COLLISION"
            break
        if corridor_violation:
            status = "CORRIDOR_VIOLATION"
            break
        if not env.inside_taskspace(dense).all():
            status = "OOB"
            break
        if env.reached(state[:3]):
            status = "SUCCESS"
            break

    states = np.asarray(states, np.float32)
    controls = np.asarray(controls, np.float32).reshape(-1, 3)
    dense = env.dense_positions(states, controls) if len(controls) else states[:, :3]
    clearance = env.obstacle_clearance(dense)
    finite = clearance[np.isfinite(clearance)]
    realized_theta = trajectory_crossing_theta(env, dense, frame)
    return {
        "status": status,
        "states": states,
        "controls": controls,
        "physical_collision": physical_collision,
        "corridor_violation": corridor_violation,
        "requested_theta": float(theta),
        "requested_route": theta_name(float(theta)),
        "realized_theta": realized_theta,
        "mode": theta_name(realized_theta),
        "min_clearance_m": float(finite.min()) if len(finite) else None,
        "time_to_goal_s": (
            float(len(controls) * config.safemppi.dt)
            if status == "SUCCESS" else None
        ),
    }
