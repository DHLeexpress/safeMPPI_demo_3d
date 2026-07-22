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

PLAN_H = 10
CONTEXT_DIM = 10
ROUTE_MODES = ("below", "above", "left", "right")


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


def verifier_chain_margins(env: TaskEnvironment, positions: np.ndarray, gamma: float,
                           sensing_range: float) -> np.ndarray:
    """GREEN full-horizon verifier margins: rebuilt polytope chain along the plan.

    At every plan knot the polytope is rebuilt at ``q_h`` (so ``H_P(q_h)=1``) and the next knot
    must satisfy ``H_P(q_{h+1}) >= 1-gamma``. This is exactly the certificate every *executed*
    demonstration step satisfied online (the logged one-step slack), applied along the whole
    candidate plan. The rotating tangent face certifies skirting motion — only the velocity
    component toward an obstacle consumes contraction budget — while a single start-anchored
    face could never certify a plan that passes the ball inside the horizon.
    """
    margins = np.empty(len(positions) - 1)
    for h in range(len(positions) - 1):
        polytope = build_nominal_polytope(
            positions[h], env.spheres, env.cylinders, env.bounds,
            sensing_range=sensing_range, obstacle_margin=0.0)
        margins[h] = float(hp_values(polytope, positions[h + 1:h + 2])[0]) - (1.0 - gamma)
    return margins


def native_cost(env: TaskEnvironment, states: np.ndarray, plan: np.ndarray) -> float:
    """Untilted SafeMPPI ranking cost: running goal + control + within-plan smoothness + terminal."""
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
                 start_diversity: bool = False):
        self.config = config
        self.env = TaskEnvironment(config)
        self.device = torch.device(device)
        self.start_diversity = bool(start_diversity)

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

    def verify(self, context: torch.Tensor, candidates: torch.Tensor, gamma: float):
        state6 = context_state(self.env, context.detach().cpu().numpy())
        results = []
        for candidate in candidates:
            plan = candidate.detach().cpu().numpy().reshape(PLAN_H, 3)
            states = plan_states(self.env, state6, plan)
            dense = self.env.dense_positions(states, plan)
            inside = bool(self.env.inside_taskspace(dense).all())
            clearance = self.env.obstacle_clearance(dense)
            no_collision = bool(np.isinf(clearance).all() or clearance.min() > 0.0)
            margins = verifier_chain_margins(self.env, states[:, :3], gamma,
                                             self.config.safemppi.sensing_range)
            results.append(Verification(
                valid=bool(inside and no_collision and margins.min() > 0.0),
                hp_eligible=bool(margins[0] > 0.0),
                margin=float(margins.min()),
                execution_cost=native_cost(self.env, states, plan),
            ))
        return results

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
                start: np.ndarray | None = None):
    """Closed-loop bare-policy rollout: one temperature-1 plan per step, execute first action."""
    env = TaskEnvironment(config)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    state = (env.start.copy() if start is None else np.asarray(start, np.float32).copy())
    states, controls = [state.copy()], []
    status = "TIMEOUT"
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
        if np.isfinite(clearance).any() and clearance.min() < 0.0:
            status = "COLLISION"
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
        "mode": route_mode(env, dense),
        "min_clearance_m": (float(finite.min()) if len(finite) else None),
        "time_to_goal_s": (float(len(controls) * config.safemppi.dt)
                           if status == "SUCCESS" else None),
    }
