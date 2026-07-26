"""Ball-task adapter for the task-neutral B1 Safe Flow Expansion loop.

Context (10-D, the smallest geometrically meaningful choice):

    c_t = [g - p_t,  v_t,  b_near - p_t,  gamma]

where ``b_near`` is the closest point on the ball surface, so the boundary vector carries
distance and direction at once. Plans are H=10 acceleration rows, flattened to R^30.

This module owns everything task-specific: demo window extraction, the GREEN rebuilt-polytope
verifier chain, the native SafeMPPI execution cost (untilted: no z bias), route-mode
classification in the head-on frame (+y = left as seen from the start), and raw temperature-1
closed-loop rollouts of a bare policy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .environment import TaskEnvironment
from .expansion import Verification
from .geometry import build_nominal_polytope, hp_values
from .verifier_polytope import certify_single_sphere_affine, certify_window

PLAN_H = 10
CONTEXT_DIM = 10
ROUTE_MODES = ("below", "above", "left", "right")
CORRIDOR_X = (0.0, 3.0)
CORRIDOR_Z = (1.5, 2.5)
ABOVE_WEDGE_PLANE_Z = 2.0


def inside_expansion_corridor(points: np.ndarray, tolerance: float = 1.0e-7) -> np.ndarray:
    """Whether positions remain in the user-declared task corridor."""
    points = np.asarray(points, float).reshape(-1, 3)
    return ((points[:, 0] >= CORRIDOR_X[0] - tolerance)
            & (points[:, 0] <= CORRIDOR_X[1] + tolerance)
            & (points[:, 2] >= CORRIDOR_Z[0] - tolerance)
            & (points[:, 2] <= CORRIDOR_Z[1] + tolerance))


def above_wedge_margin(points: np.ndarray) -> np.ndarray:
    """Signed margin to the temporary upper wedge ``z - 2 >= |y|``."""
    points = np.asarray(points, float).reshape(-1, 3)
    return points[:, 2] - ABOVE_WEDGE_PLANE_Z - np.abs(points[:, 1])


def inside_above_wedge(points: np.ndarray, tolerance: float = 1.0e-7) -> np.ndarray:
    """Whether positions lie above both 45-degree planes about the x axis.

    The two planes are ``z - 2 = y`` and ``z - 2 = -y``.  Their upper
    intersection is the temporary target wedge ``z - 2 >= |y|``.
    """
    return above_wedge_margin(points) >= -tolerance


def inside_above_halfspace(points: np.ndarray,
                           tolerance: float = 1.0e-7) -> np.ndarray:
    """Whether positions lie in the upper halfspace ``z >= 2``."""
    points = np.asarray(points, float).reshape(-1, 3)
    return points[:, 2] >= ABOVE_WEDGE_PLANE_Z - tolerance


def closest_boundary_point(env: TaskEnvironment, position: np.ndarray) -> np.ndarray:
    sphere = np.asarray(env.spheres[0], float)
    center, radius = sphere[:3], float(sphere[3])
    delta = np.asarray(position, float) - center
    distance = float(np.linalg.norm(delta))
    if distance < 1.0e-9:
        return center + np.array([0.0, 0.0, radius])
    return center + radius * delta / distance


def build_context(env: TaskEnvironment, state6: np.ndarray, gamma: float) -> np.ndarray:
    state6 = np.asarray(state6, np.float32)
    position, velocity = state6[:3], state6[3:6]
    boundary = closest_boundary_point(env, position).astype(np.float32)
    return np.concatenate([env.goal - position, velocity, boundary - position,
                           np.array([gamma], np.float32)]).astype(np.float32)


def context_state(env: TaskEnvironment, context: np.ndarray) -> np.ndarray:
    """Recover [p, v] from a context row (exact for this single-ball task)."""
    context = np.asarray(context, np.float32)
    position = env.goal - context[:3]
    return np.concatenate([position, context[3:6]]).astype(np.float32)


def plan_states(env: TaskEnvironment, state6: np.ndarray, plan: np.ndarray) -> np.ndarray:
    states = [np.asarray(state6, np.float32)]
    for control in np.asarray(plan, np.float32).reshape(-1, 3):
        states.append(env.step(states[-1], control))
    return np.asarray(states, np.float32)


def closest_approach_angular_descriptor(
        env: TaskEnvironment, state6: np.ndarray, plan: np.ndarray,
        world_up: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
        ) -> np.ndarray:
    """Continuous transverse direction of a plan at the ball's axial plane.

    The forward axis is the task start-to-goal direction.  At the simulated plan
    knot closest to the plane through the ball center normal to that axis, the
    center-to-knot displacement is projected onto the transverse plane.  Its
    coordinates are returned in the ``(left, up)`` basis as a unit vector.  No
    discrete route labels or angular bins enter the descriptor.

    An exactly axial plan has no transverse direction and returns ``[0, 0]``.
    ``world_up`` is explicit so jointly rotating the task and its up direction
    leaves the descriptor unchanged.
    """
    start = np.asarray(env.start[:3], float)
    goal = np.asarray(env.goal, float)
    forward = goal - start
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-12:
        raise ValueError("task start and goal must define a nonzero forward axis")
    forward /= forward_norm

    vertical = np.asarray(world_up, float).reshape(3).copy()
    vertical -= float(vertical @ forward) * forward
    vertical_norm = float(np.linalg.norm(vertical))
    if vertical_norm <= 1.0e-12:
        raise ValueError("world_up must not be parallel to the start-to-goal axis")
    vertical /= vertical_norm
    lateral = np.cross(vertical, forward)

    center = np.asarray(env.spheres[0, :3], float)
    positions = plan_states(env, state6, plan)[:, :3].astype(float)
    relative = positions - center
    closest = int(np.argmin(np.abs(relative @ forward)))
    transverse = relative[closest] - float(relative[closest] @ forward) * forward
    descriptor = np.array([transverse @ lateral, transverse @ vertical], float)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1.0e-12:
        return np.zeros(2, np.float32)
    return (descriptor / norm).astype(np.float32)


def nominal_hp_chain_margins(env: TaskEnvironment, positions: np.ndarray, gamma: float,
                             sensing_range: float) -> np.ndarray:
    """BLUE nominal one-step ``H_P`` margins, retained for diagnostics only.

    This rebuilt-at-each-knot quantity is not the GREEN fitted full-H verifier and
    never gates expansion execution.  It is logged only to compare the generated
    plans against the nominal SafeMPPI geometry.
    """
    margins = np.empty(len(positions) - 1)
    for h in range(len(positions) - 1):
        polytope = build_nominal_polytope(
            positions[h], env.spheres, env.cylinders, env.bounds,
            sensing_range=sensing_range, obstacle_margin=0.0)
        margins[h] = float(hp_values(polytope, positions[h + 1:h + 2])[0]) - (1.0 - gamma)
    return margins


def raw_trajectory_validity(config: ExperimentConfig, states: np.ndarray,
                            controls: np.ndarray, gamma: float,
                            stride: int = 2) -> bool:
    """Post-hoc safety validity of one raw temperature-1 executed trajectory.

    The whole dense path must remain in task space and collision-free.  Every
    stride-``stride`` executed segment, including the truncated terminal segment, must pass
    the same rebuilt GREEN verifier.  Success and progress are deliberately absent.
    """
    states = np.asarray(states, np.float32)
    controls = np.asarray(controls, np.float32).reshape(-1, 3)
    if len(controls) < 1 or len(states) != len(controls) + 1 or stride < 1:
        return False
    env = TaskEnvironment(config)
    dense = env.dense_positions(states, controls)
    clearance = env.obstacle_clearance(dense)
    if (not env.inside_taskspace(dense).all()
            or (np.isfinite(clearance).any() and float(clearance.min()) <= 0.0)):
        return False
    for start in range(0, len(controls), stride):
        stop = min(start + PLAN_H, len(controls))
        certified, _ = certify_window(
            states[start:stop + 1, :3], env.spheres, env.cylinders,
            float(gamma), config.safemppi.sensing_range,
        )
        if not certified:
            return False
    return True


def raw_window_validity_fraction(config: ExperimentConfig, states: np.ndarray,
                                 controls: np.ndarray, gamma: float) -> float:
    """Fraction of executed windows that pass the GREEN safety verifier.

    A window starts at every executed control index.  Its horizon is truncated
    at the end of the trajectory, so the final starts are evaluated with
    ``H_t < PLAN_H`` rather than discarded.  Each indicator contains only the
    requested safety predicates: task-space bounds, collision avoidance, and
    the GREEN verifier.  Success, progress, and the expansion corridor are not
    part of this metric.
    """
    states = np.asarray(states, np.float32)
    controls = np.asarray(controls, np.float32).reshape(-1, 3)
    if len(controls) < 1 or len(states) != len(controls) + 1:
        return 0.0

    env = TaskEnvironment(config)
    valid = 0
    for start in range(len(controls)):
        stop = min(start + PLAN_H, len(controls))
        window_states = states[start:stop + 1]
        window_controls = controls[start:stop]
        dense = env.dense_positions(window_states, window_controls)
        clearance = env.obstacle_clearance(dense)
        in_bounds = bool(env.inside_taskspace(dense).all())
        collision_free = bool(
            np.isinf(clearance).all()
            or (np.isfinite(clearance).any() and float(clearance.min()) > 0.0)
        )
        certified, _ = certify_window(
            window_states[:, :3], env.spheres, env.cylinders,
            float(gamma), config.safemppi.sensing_range,
        )
        valid += int(in_bounds and collision_free and certified)
    return valid / len(controls)


def native_cost(env: TaskEnvironment, states: np.ndarray, plan: np.ndarray,
                execution_z_bias_mode: str = "none") -> float:
    """Expansion ranking cost: native state/control/smoothness/terminal terms only.

    The demonstration controller's ``z_bias_*`` exponential is intentionally absent.  That
    term exists only to collect below-equator demonstrations; carrying it into self-expansion
    would keep steering the learned controller toward the demonstration support.
    """
    m = env.mppi
    plan = np.asarray(plan, np.float32).reshape(-1, 3)
    goal = env.goal
    cost = 0.0
    for h, control in enumerate(plan):
        cost += m.running_goal_weight * float(((states[h + 1, :3] - goal) ** 2).sum())
        cost += m.control_weight * float((control ** 2).sum())
        if h:
            cost += m.smooth_weight * float(((control - plan[h - 1]) ** 2).sum())
    cost += m.terminal_goal_weight * float(((states[-1, :3] - goal) ** 2).sum())
    if execution_z_bias_mode == "favor_above":
        exponent = np.minimum(
            (m.z_bias_plane - np.asarray(states[1:, 2], float)) / m.z_bias_temperature,
            20.0,
        )
        cost += m.z_bias_weight * float(np.exp(exponent).sum())
    elif execution_z_bias_mode != "none":
        raise ValueError(f"unknown execution_z_bias_mode: {execution_z_bias_mode}")
    return cost


def route_mode(env: TaskEnvironment, positions: np.ndarray) -> str:
    """above/below/left/right at the ball-plane crossing (+y = left, viewed from the start)."""
    sphere = np.asarray(env.spheres[0], float)
    plane, center_y, center_z = sphere[0], sphere[1], sphere[2]
    x = np.asarray(positions, float)[:, 0]
    crossings = np.flatnonzero((x[:-1] < plane) & (x[1:] >= plane))
    if not len(crossings):
        return "none"
    i = int(crossings[0])
    fraction = (plane - x[i]) / max(x[i + 1] - x[i], 1.0e-9)
    point = positions[i] + fraction * (positions[i + 1] - positions[i])
    angle = np.degrees(np.arctan2(point[2] - center_z, point[1] - center_y))
    if -135.0 <= angle < -45.0:
        return "below"
    if 45.0 <= angle < 135.0:
        return "above"
    if -45.0 <= angle < 45.0:
        return "left"
    return "right"


class BallFlowTask:
    """ExpansionTask implementation: fail-closed GREEN verifier + native execution cost.

    With ``start_diversity`` the odd parallel replica of every gamma starts from a randomized
    collision-free pre-ball state (position spread across the equator, small forward speed).
    Replicas are how this task preserves route support: no plans or demonstrations are injected,
    and the even replica always keeps the canonical start so closed-loop success stays comparable.
    """

    def __init__(self, config: ExperimentConfig, device: str | torch.device = "cpu",
                 start_diversity: bool = False,
                 execution_z_bias_mode: str = "none",
                 tight_corridor: bool = False,
                 target_region: str = "above_wedge",
                 verifier_mode: str = "full_polytope",
                 verifier_solver: str = "analytic"):
        if execution_z_bias_mode not in {"none", "favor_above"}:
            raise ValueError(f"unknown execution_z_bias_mode: {execution_z_bias_mode}")
        if target_region not in {"above_wedge", "above_halfspace"}:
            raise ValueError(f"unknown target_region: {target_region}")
        if verifier_mode not in {"full_polytope", "single_sphere_affine"}:
            raise ValueError(f"unknown verifier_mode: {verifier_mode}")
        if verifier_solver not in {"analytic", "cvxpy"}:
            raise ValueError(f"unknown verifier_solver: {verifier_solver}")
        self.config = config
        self.env = TaskEnvironment(config)
        self.device = torch.device(device)
        self.start_diversity = bool(start_diversity)
        self.execution_z_bias_mode = execution_z_bias_mode
        self.tight_corridor = bool(tight_corridor)
        self.target_region = target_region
        self.verifier_mode = verifier_mode
        self.verifier_solver = verifier_solver

    def _diverse_start(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        for _ in range(64):
            if rng.random() < 0.5:
                position = np.array([rng.uniform(0.1, 0.9), rng.uniform(-0.3, 0.3),
                                     rng.uniform(1.75, 2.35)], np.float32)
            else:
                # Above/ahead of the ball: from here any forward crossing is an above route,
                # so policy-sampled, verifier-gated above-corridor positives can exist at all.
                position = np.array([rng.uniform(0.9, 1.35), rng.uniform(-0.2, 0.2),
                                     rng.uniform(2.30, 2.60)], np.float32)
            clearance = float(self.env.obstacle_clearance(position[None])[0])
            if self.env.inside_taskspace(position[None])[0] and clearance > 0.12:
                velocity = np.array([rng.uniform(0.2, 0.6), 0.0, 0.0], np.float32)
                return np.concatenate([position, velocity]).astype(np.float32)
        return self.env.start.copy()

    def reset(self, gamma: float, episode: int, seed: int):
        start = (self._diverse_start(seed) if self.start_diversity and episode % 2 == 1
                 else self.env.start.copy())
        return {"x": start, "steps": 0, "collided": False, "oob": False}

    def context(self, state, gamma: float) -> torch.Tensor:
        return torch.from_numpy(build_context(self.env, state["x"], gamma)).to(self.device)

    def angular_descriptors(self, context: torch.Tensor,
                            candidates: torch.Tensor) -> torch.Tensor:
        """Batch form of :func:`closest_approach_angular_descriptor`."""
        state6 = context_state(self.env, context.detach().cpu().numpy())
        values = [
            closest_approach_angular_descriptor(
                self.env, state6, candidate.detach().cpu().numpy(),
            )
            for candidate in candidates
        ]
        return torch.as_tensor(np.asarray(values), device=candidates.device,
                               dtype=candidates.dtype)

    def above_wedge_eligibility(self, context: torch.Tensor,
                                candidates: torch.Tensor) -> torch.Tensor:
        """Return the temporary target-region signal for each next executed state.

        Only ``q_1`` is checked.  The remaining unexecuted horizon is
        deliberately excluded so the target gate cannot recreate a tail-based
        NVP near the goal.
        """
        state6 = context_state(self.env, context.detach().cpu().numpy())
        eligible = []
        for candidate in candidates:
            plan = candidate.detach().cpu().numpy().reshape(PLAN_H, 3)
            next_position = plan_states(self.env, state6, plan)[:2, :3][-1:]
            eligible.append(bool(inside_above_wedge(next_position)[0]))
        return torch.as_tensor(eligible, dtype=torch.bool, device=candidates.device)

    def target_region_eligibility(self, context: torch.Tensor,
                                  candidates: torch.Tensor) -> torch.Tensor:
        """Return the configured q1-only target-region signal."""
        if self.target_region == "above_wedge":
            return self.above_wedge_eligibility(context, candidates)
        state6 = context_state(self.env, context.detach().cpu().numpy())
        eligible = []
        for candidate in candidates:
            plan = candidate.detach().cpu().numpy().reshape(PLAN_H, 3)
            next_position = plan_states(self.env, state6, plan)[1:2, :3]
            eligible.append(bool(inside_above_halfspace(next_position)[0]))
        return torch.as_tensor(eligible, dtype=torch.bool, device=candidates.device)

    def d4_replay_batch(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the task's four exact quarter-turn symmetries about the x axis.

        The goal, start, spherical obstacle, dynamics, and y-z task-space square
        are invariant under these rotations. This is an optional replay-only
        equivariance assumption; it is never used by the default expansion arm.
        """
        context_blocks, candidate_blocks = [], []
        for quarter_turn in range(4):
            angle = 0.5 * np.pi * quarter_turn
            cosine, sine = float(np.cos(angle)), float(np.sin(angle))
            context = contexts.clone()
            for start in (0, 3, 6):
                y = contexts[:, start + 1]
                z = contexts[:, start + 2]
                context[:, start + 1] = cosine * y - sine * z
                context[:, start + 2] = sine * y + cosine * z
            candidate = candidates.clone()
            y = candidates[..., 1]
            z = candidates[..., 2]
            candidate[..., 1] = cosine * y - sine * z
            candidate[..., 2] = sine * y + cosine * z
            context_blocks.append(context)
            candidate_blocks.append(candidate)
        return torch.cat(context_blocks), torch.cat(candidate_blocks)

    def _verify_plan(
        self,
        context: torch.Tensor,
        candidate: torch.Tensor,
        gamma: float,
    ) -> Verification:
        state6 = context_state(self.env, context.detach().cpu().numpy())
        plan = candidate.detach().cpu().numpy().reshape(-1, 3)
        if not 1 <= len(plan) <= PLAN_H:
            raise ValueError(f"verifier plan horizon must lie in [1,{PLAN_H}]")
        states = plan_states(self.env, state6, plan)
        dense = self.env.dense_positions(states, plan)
        executed_dense = self.env.dense_positions(states[:2], plan[:1])
        inside = bool(self.env.inside_taskspace(executed_dense).all())
        if self.tight_corridor:
            inside = bool(inside and inside_expansion_corridor(executed_dense).all())
        clearance = self.env.obstacle_clearance(dense)
        no_collision = bool(np.isinf(clearance).all() or clearance.min() > 0.0)
        verifier = (
            certify_window
            if self.verifier_mode == "full_polytope"
            else certify_single_sphere_affine
        )
        certified, verifier_slack = verifier(
            states[:, :3], self.env.spheres, self.env.cylinders, gamma,
            self.config.safemppi.sensing_range,
            face_solver=self.verifier_solver,
        )
        first_step_margin = nominal_hp_chain_margins(
            self.env, states[:2, :3], gamma,
            self.config.safemppi.sensing_range,
        )[0]
        progress = float(np.min(np.diff(states[:, 0])))
        if self.target_region == "above_wedge":
            target_eligible = bool(inside_above_wedge(states[1:2, :3])[0])
        else:
            target_eligible = bool(inside_above_halfspace(states[1:2, :3])[0])
        return Verification(
            valid=bool(inside and no_collision and certified),
            hp_eligible=bool(first_step_margin > 0.0),
            margin=float(verifier_slack),
            execution_cost=native_cost(
                self.env, states, plan,
                execution_z_bias_mode=self.execution_z_bias_mode,
            ),
            progress=progress,
            progress_eligible=bool(progress > 0.0),
            target_eligible=target_eligible,
            step_margin=float(first_step_margin),
        )

    def verify(self, context: torch.Tensor, candidates: torch.Tensor, gamma: float):
        return [
            self._verify_plan(context, candidate, gamma)
            for candidate in candidates
        ]

    def advance(self, state, candidate: torch.Tensor):
        control = candidate.detach().cpu().numpy().reshape(PLAN_H, 3)[0]
        before = state["x"]
        after = self.env.step(before, control)
        dense = self.env.dense_positions(np.stack([before, after]), control[None])
        clearance = self.env.obstacle_clearance(dense)
        return {
            "x": after, "steps": state["steps"] + 1,
            "collided": bool(state["collided"]
                             or (np.isfinite(clearance).any() and clearance.min() < 0.0)),
            "oob": bool(state["oob"] or not self.env.inside_taskspace(dense).all()),
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
        """Fraction of post-action executed states in the upper halfspace.

        This score ranks terminal-SUCCESS trajectories only. It never changes
        verification, execution eligibility, NVP, or the safety label.
        """
        if not executed_states:
            return 0.0
        z = np.asarray(
            [np.asarray(state["x"], float)[2] for state in executed_states],
            dtype=float,
        )
        return float(np.mean(z >= ABOVE_WEDGE_PLANE_Z))

    def successful_trajectory_mean_z(self, executed_states) -> float:
        """Mean post-action height used only to rank terminal successes."""
        if not executed_states:
            return -float("inf")
        return float(np.mean([
            np.asarray(state["x"], float)[2] for state in executed_states
        ]))


def load_policy(path: str | Path):
    """Rebuild a ConditionalFlowMLP from a saved {'model', 'arch'} checkpoint."""
    from .flow_model import ConditionalFlowMLP
    payload = torch.load(path, weights_only=False)
    arch = dict(payload["arch"])
    arch["plan_shape"] = tuple(arch["plan_shape"])
    policy = ConditionalFlowMLP(**arch)
    policy.load_state_dict(payload["model"])
    return policy


def demo_windows(run_dir: str | Path):
    """Sliding H-step windows over executed demo rollouts -> (contexts [N,10], plans [N,H,3])."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = load_config(run_dir / "resolved_config.json")
    env = TaskEnvironment(config)
    contexts, plans, meta = [], [], []
    for row in manifest["runs"]:
        data = np.load(run_dir / row["file"])
        states, controls = data["states"], data["controls"]
        for start in range(len(controls) - PLAN_H + 1):
            contexts.append(build_context(env, states[start], float(row["gamma"])))
            plans.append(controls[start:start + PLAN_H])
            meta.append({"gamma": float(row["gamma"]), "seed": int(row["seed"]), "t": start})
    return (np.asarray(contexts, np.float32), np.asarray(plans, np.float32), meta, config)


@torch.no_grad()
def raw_rollout(policy, config: ExperimentConfig, gamma: float, seed: int,
                device: str | torch.device = "cpu", max_steps: int | None = None,
                start: np.ndarray | None = None, tight_corridor: bool = False):
    """Closed-loop bare-policy rollout: one temperature-1 plan per step, execute first action."""
    env = TaskEnvironment(config)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    state = (env.start.copy() if start is None else np.asarray(start, np.float32).copy())
    states, controls = [state.copy()], []
    status = "TIMEOUT"
    physical_collision = False
    corridor_violation = False
    for _ in range(max_steps or config.taskspace.max_steps):
        context = torch.from_numpy(build_context(env, state, gamma)).to(device)
        plan = policy.sample(context, 1, generator)[0].detach().cpu().numpy()
        control = plan.reshape(PLAN_H, 3)[0]
        state = env.step(state, control)
        states.append(state.copy())
        controls.append(control)
        dense = env.dense_positions(np.asarray(states[-2:], np.float32),
                                    control[None].astype(np.float32))
        clearance = env.obstacle_clearance(dense)
        physical_collision = bool(
            physical_collision
            or (np.isfinite(clearance).any() and clearance.min() < 0.0)
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
    return {
        "status": status, "states": states, "controls": controls,
        "physical_collision": physical_collision,
        "corridor_violation": corridor_violation,
        "mode": route_mode(env, dense),
        "min_clearance_m": (float(finite.min()) if len(finite) else None),
        "time_to_goal_s": (float(len(controls) * config.safemppi.dt)
                           if status == "SUCCESS" else None),
    }
