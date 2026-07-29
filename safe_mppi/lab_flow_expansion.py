"""Lab-scene adapter for the task-agnostic B1 expansion loop.

The lab policy predicts *raw* acceleration commands.  Candidate verification
and execution therefore apply Minhyuk's stateful reference governor exactly
once.  The verifier-only governor memory is appended to the policy context and
stripped by :class:`LabExpansionPolicyAdapter`; it is never an input to the
learned flow model.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from .ball_flow_task import PLAN_H, nominal_hp_chain_margins
from .environment import ReferenceGovernor, TaskEnvironment
from .expansion import Verification
from .lab_reference_flow_task import policy_context
from .lab_visual_flow import load_lab_reference_policy
from .verifier_polytope import certify_single_sphere_affine, certify_window


LAB_VERIFIER_STATE_DIM = 6


class LabExpansionPolicyAdapter(nn.Module):
    """Hide verifier-only governor memory from the learned policy.

    Expansion archives carry ``[policy_context, previous_applied,
    previous_raw]`` so persistent verifier workers can reproduce the stateful
    lab dynamics.  Sampling, embedding, and CFM training see only the original
    pretrained policy context.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.policy_context_dim = int(policy.context_dim)
        self.context_dim = self.policy_context_dim + LAB_VERIFIER_STATE_DIM
        self.plan_shape = tuple(policy.plan_shape)
        self.control_limit = policy.control_limit
        self.nfe = int(policy.nfe)
        self.context_schema = getattr(policy, "context_schema", "lab_raw10_v1")

    def _policy_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] == self.policy_context_dim:
            return context
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                "lab expansion context must contain either the policy context "
                "or policy context plus six verifier-only governor values"
            )
        return context[..., :self.policy_context_dim]

    def sample(self, context, count, generator, base_std=1.0):
        return self.policy.sample(
            self._policy_context(context), count, generator, base_std=base_std,
        )

    def sample_with_base(self, context, count, generator, base_std=1.0):
        return self.policy.sample_with_base(
            self._policy_context(context), count, generator, base_std=base_std,
        )

    def embed(self, context, candidates, flow_time=0.9, base=None):
        return self.policy.embed(
            self._policy_context(context),
            candidates,
            flow_time=flow_time,
            base=base,
        )

    def cfm_loss(
        self,
        contexts,
        candidates,
        reduction="none",
        loss_mask=None,
    ):
        return self.policy.cfm_loss(
            self._policy_context(contexts),
            candidates,
            reduction=reduction,
            loss_mask=loss_mask,
        )

    def expansion_parameter_groups(self, base_lr, first_layer_lr_scale=1.0):
        return self.policy.expansion_parameter_groups(
            base_lr, first_layer_lr_scale,
        )

    @property
    def trunk(self):
        return self.policy.trunk

    @property
    def head(self):
        return self.policy.head

    def state_dict(self, *args, **kwargs):
        return self.policy.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.policy.load_state_dict(
            state_dict, strict=strict, assign=assign,
        )


def load_lab_expansion_policy(path: str | Path) -> LabExpansionPolicyAdapter:
    return LabExpansionPolicyAdapter(load_lab_reference_policy(path))


class LabFlowExpansionTask:
    """B1 task contract for the Minhyuk-frame single-sphere scene."""

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
    ):
        if execution_z_bias_mode != "none":
            raise ValueError(
                "lab expansion excludes the demonstration-only z bias"
            )
        if verifier_mode not in {"full_polytope", "single_sphere_affine"}:
            raise ValueError(f"unknown verifier_mode: {verifier_mode}")
        if verifier_solver not in {"analytic", "cvxpy"}:
            raise ValueError(f"unknown verifier_solver: {verifier_solver}")
        self.config = config
        self.env = TaskEnvironment(config)
        self.context_schema = str(context_schema)
        self.device = torch.device(device)
        self.tight_corridor = bool(tight_corridor)
        self.verifier_mode = verifier_mode
        self.verifier_solver = verifier_solver
        delta = self.env.goal - self.env.start[:3]
        self.forward = delta / np.linalg.norm(delta)
        self.obstacle_plane_z = float(self.env.spheres[0, 2])

    def reset(self, gamma: float, episode: int, seed: int) -> dict[str, Any]:
        del gamma, episode, seed
        return {
            "x": self.env.start.copy(),
            "previous_applied": np.zeros(3, np.float32),
            "previous_raw": np.zeros(3, np.float32),
            "steps": 0,
            "collided": False,
            "oob": False,
        }

    def context(self, state, gamma: float) -> torch.Tensor:
        learned = policy_context(
            SimpleNamespace(context_schema=self.context_schema),
            self.env,
            state["x"],
            float(gamma),
        )
        packed = np.concatenate([
            learned,
            np.asarray(state["previous_applied"], np.float32),
            np.asarray(state["previous_raw"], np.float32),
        ])
        return torch.from_numpy(packed).to(self.device)

    def _decode_context(
        self, context: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = context.detach().cpu().numpy().astype(np.float32, copy=False)
        position = self.env.goal - values[:3]
        velocity = values[3:6]
        previous_applied = values[-6:-3]
        previous_raw = values[-3:]
        return (
            np.concatenate([position, velocity]).astype(np.float32),
            previous_applied.copy(),
            previous_raw.copy(),
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
        states: np.ndarray,
        plan: np.ndarray,
        previous_raw: np.ndarray,
    ) -> float:
        """Match the lab SafeMPPI cost except its demo-only below z bias."""
        cfg = self.config.safemppi
        plan = np.asarray(plan, np.float32).reshape(-1, 3)
        initial_distance = float(
            np.linalg.norm(states[0, :3] - self.env.goal)
        )
        previous = np.asarray(previous_raw, np.float32)
        cost = 0.0
        for index, command in enumerate(plan):
            point = states[index + 1, :3]
            distance = float(np.linalg.norm(point - self.env.goal))
            cost += cfg.running_goal_weight * distance ** 2
            cost += cfg.control_weight * float(command @ command)
            difference = command - previous
            cost += cfg.smooth_weight * float(difference @ difference)
            cost -= cfg.progress_weight * (initial_distance - distance)
            clearance = float(self.env.obstacle_clearance(point[None])[0])
            cost += cfg.soft_clearance_weight * max(
                cfg.soft_clearance_target - clearance, 0.0,
            ) ** 2
            violation = (
                np.maximum(self.env.bounds[:, 0] - point, 0.0)
                + np.maximum(point - self.env.bounds[:, 1], 0.0)
            )
            exponent = np.minimum(
                violation / cfg.taskspace_exponential_temperature, 20.0,
            )
            cost += cfg.taskspace_exponential_weight * float(
                np.expm1(exponent).sum()
            )
            previous = command
        terminal_distance = float(
            np.linalg.norm(states[-1, :3] - self.env.goal)
        )
        cost += cfg.terminal_goal_weight * terminal_distance ** 2
        return float(cost)

    def _verify_plan(
        self,
        context: torch.Tensor,
        candidate: torch.Tensor,
        gamma: float,
    ) -> Verification:
        state6, previous_applied, previous_raw = self._decode_context(context)
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
        # In the lab profile ``tight corridor`` means the configured geofence;
        # the canonical [0,3] x [1.5,2.5] ball corridor is never imported.
        inside = bool(self.env.inside_taskspace(executed_dense).all())
        clearance = self.env.obstacle_clearance(dense)
        no_collision = bool(
            np.isinf(clearance).all()
            or float(clearance.min()) > 0.0
        )
        verifier = (
            certify_window
            if self.verifier_mode == "full_polytope"
            else certify_single_sphere_affine
        )
        certified, verifier_slack = verifier(
            states[:, :3],
            self.env.spheres,
            self.env.cylinders,
            float(gamma),
            self.config.safemppi.sensing_range,
            face_solver=self.verifier_solver,
        )
        step_margin = nominal_hp_chain_margins(
            self.env,
            states[:2, :3],
            float(gamma),
            self.config.safemppi.sensing_range,
        )[0]
        longitudinal = states[:, :3] @ self.forward
        knot_progress = np.diff(longitudinal)
        progress = float(knot_progress.min())
        return Verification(
            valid=bool(inside and no_collision and certified),
            hp_eligible=bool(step_margin > 0.0),
            margin=float(verifier_slack),
            execution_cost=self._native_cost(
                states, plan, previous_raw,
            ),
            progress=progress,
            progress_eligible=bool(progress > 0.0),
            target_eligible=bool(states[1, 2] >= self.obstacle_plane_z),
            step_margin=float(step_margin),
        )

    def verify(self, context: torch.Tensor, candidates: torch.Tensor, gamma: float):
        return [
            self._verify_plan(context, candidate, gamma)
            for candidate in candidates
        ]

    def advance(self, state, candidate: torch.Tensor):
        command = candidate.detach().cpu().numpy().reshape(-1, 3)[0]
        governor = ReferenceGovernor(self.config.safemppi)
        governor.previous_applied = np.asarray(
            state["previous_applied"], np.float32,
        ).copy()
        after, applied, dense = governor.step(state["x"], command)
        clearance = self.env.obstacle_clearance(dense)
        return {
            "x": after,
            "previous_applied": applied,
            "previous_raw": np.asarray(command, np.float32).copy(),
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
                or not self.env.inside_taskspace(dense).all()
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
            float(np.asarray(state["x"])[2]) >= self.obstacle_plane_z
            for state in executed_states
        ]))

    def successful_trajectory_mean_z(self, executed_states) -> float:
        if not executed_states:
            return -float("inf")
        return float(np.mean([
            float(np.asarray(state["x"])[2])
            for state in executed_states
        ]))
