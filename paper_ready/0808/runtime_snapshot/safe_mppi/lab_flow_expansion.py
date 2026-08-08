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
from .ball_flow_theta import theta_name, trajectory_crossing_theta
from .environment import ReferenceGovernor, TaskEnvironment
from .expansion import Verification
from .lab_reference_flow_task import (
    _append_raw_history,
    _raw_history_before,
    policy_context,
)
from .lab_visual_flow import (
    LAB_HP100_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_HISTORY_SCHEMA,
    load_lab_reference_policy,
)
from .verifier_polytope import certify_single_sphere_affine, certify_window


LAB_VERIFIER_STATE_DIM = 6
LAB_HISTORY_CONTEXT_SCHEMAS = frozenset({
    LAB_HP100_HISTORY_SCHEMA,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
})


def _is_history_context_schema(context_schema: str) -> bool:
    return str(context_schema) in LAB_HISTORY_CONTEXT_SCHEMAS


class LabExpansionPolicyAdapter(nn.Module):
    """Hide verifier-only governor memory from the learned policy.

    Expansion archives carry ``[policy_context, previous_applied,
    previous_raw]`` so persistent verifier workers can reproduce the stateful
    lab dynamics.  Sampling, embedding, and CFM training see only the original
    pretrained policy context.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        freeze_history_encoder: bool | None = None,
    ):
        super().__init__()
        self.policy = policy
        self.policy_context_dim = int(policy.context_dim)
        self.context_dim = self.policy_context_dim + LAB_VERIFIER_STATE_DIM
        self.plan_shape = tuple(policy.plan_shape)
        self.control_limit = policy.control_limit
        self.nfe = int(policy.nfe)
        self.context_schema = getattr(policy, "context_schema", "lab_raw10_v1")
        self.freeze_history_encoder = freeze_history_encoder
        if (
            not _is_history_context_schema(self.context_schema)
            and freeze_history_encoder is not None
        ):
            raise ValueError(
                "history freeze contract applies only to a GRU policy"
            )
        if (
            _is_history_context_schema(self.context_schema)
            and freeze_history_encoder is not None
        ):
            for parameter in self.policy.history_encoder.parameters():
                parameter.requires_grad_(not freeze_history_encoder)

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

    @torch.no_grad()
    def sample_many_with_base(
        self, contexts, count, generators, base_std=1.0,
    ):
        """Integrate independent episode bases in one batched ODE solve."""
        contexts = self._policy_context(contexts)
        if len(contexts) != len(generators):
            raise ValueError("one generator is required per rollout context")
        encoded = self.policy.encode_context(contexts)
        flow = self.policy.flow
        bases = torch.stack([
            torch.randn(
                count, flow.plan_dim,
                device=contexts.device,
                generator=generator,
            ) * float(base_std)
            for generator in generators
        ])
        repeated_context = encoded[:, None, :].expand(
            len(contexts), count, encoded.shape[-1]
        ).reshape(len(contexts) * count, encoded.shape[-1])
        integrated = flow._integrate_flow(
            bases.reshape(len(contexts) * count, flow.plan_dim),
            repeated_context,
        ).reshape(len(contexts), count, *flow.plan_shape)
        if flow.control_limit is not None:
            integrated = integrated.clamp(
                -flow.control_limit, flow.control_limit,
            )
        return integrated, bases.reshape(
            len(contexts), count, *flow.plan_shape,
        )

    def embed(self, context, candidates, flow_time=0.9, base=None):
        return self.policy.embed(
            self._policy_context(context),
            candidates,
            flow_time=flow_time,
            base=base,
        )

    @torch.no_grad()
    def embed_many(
        self, contexts, candidates, flow_time=0.9, bases=None,
    ):
        """Embed every active episode's K plans in one flow forward pass."""
        contexts = self._policy_context(contexts)
        if candidates.ndim < 3 or len(candidates) != len(contexts):
            raise ValueError(
                "batched candidates must start with episode and sample axes"
            )
        episode_count, sample_count = candidates.shape[:2]
        encoded = self.policy.encode_context(contexts)
        repeated_context = encoded[:, None, :].expand(
            episode_count, sample_count, encoded.shape[-1]
        ).reshape(episode_count * sample_count, encoded.shape[-1])
        flat_bases = (
            None
            if bases is None
            else bases.reshape(episode_count * sample_count, *bases.shape[2:])
        )
        features = self.policy.flow.embed(
            repeated_context,
            candidates.reshape(
                episode_count * sample_count, *candidates.shape[2:]
            ),
            flow_time=flow_time,
            base=flat_bases,
        )
        return features.reshape(episode_count, sample_count, -1)

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
        if _is_history_context_schema(self.context_schema):
            if self.freeze_history_encoder is None:
                raise ValueError(
                    "GRU expansion requires an explicit history freeze "
                    "contract"
                )
            groups = self.policy.expansion_parameter_groups(
                base_lr,
                first_layer_lr_scale,
                freeze_history_encoder=self.freeze_history_encoder,
            )
        else:
            groups = self.policy.expansion_parameter_groups(
                base_lr, first_layer_lr_scale,
            )
        trainable_groups = []
        for group in groups:
            trainable = [
                parameter for parameter in group["params"]
                if parameter.requires_grad
            ]
            if trainable:
                trainable_groups.append({**group, "params": trainable})
        return trainable_groups

    def freeze_visual_encoder_for_expansion(self) -> int:
        encoder = getattr(self.policy, "grid_encoder", None)
        if not isinstance(encoder, nn.Module):
            raise TypeError(
                "visual-encoder freeze requires a lab visual policy"
            )
        parameters = list(encoder.parameters())
        if not parameters:
            raise ValueError("lab visual encoder has no parameters to freeze")
        for parameter in parameters:
            parameter.requires_grad_(False)
        return sum(parameter.numel() for parameter in parameters)

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


def load_lab_expansion_policy(
    path: str | Path,
    *,
    train_history_encoder: bool = False,
) -> LabExpansionPolicyAdapter:
    policy = load_lab_reference_policy(path)
    is_history = _is_history_context_schema(
        getattr(policy, "context_schema", ""),
    )
    if train_history_encoder and not is_history:
        raise ValueError(
            "train_history_encoder applies only to a GRU checkpoint"
        )
    return LabExpansionPolicyAdapter(
        policy,
        freeze_history_encoder=(
            not train_history_encoder if is_history else None
        ),
    )


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
        execution_soft_clearance_weight: float | None = None,
        execution_soft_clearance_target_m: float | None = None,
        execution_taskspace_weight: float | None = None,
        execution_plane_escape_weight: float = 0.0,
        execution_plane_escape_sigma: float = 0.15,
        execution_plane_escape_gate_radius: float = 1.0,
        execution_plane_penalty_weight: float = 0.0,
        execution_plane_penalty_sigma: float = 0.15,
        execution_plane_penalty_shape: str = "gaussian",
        execution_above_penalty_weight: float = 0.0,
    ):
        if execution_z_bias_mode != "none":
            raise ValueError(
                "lab expansion excludes the demonstration-only z bias"
            )
        if verifier_mode not in {"full_polytope", "single_sphere_affine"}:
            raise ValueError(f"unknown verifier_mode: {verifier_mode}")
        if verifier_solver not in {"analytic", "cvxpy"}:
            raise ValueError(f"unknown verifier_solver: {verifier_solver}")
        configured_weight = float(config.safemppi.soft_clearance_weight)
        configured_target = float(config.safemppi.soft_clearance_target)
        effective_weight = (
            configured_weight
            if execution_soft_clearance_weight is None
            else float(execution_soft_clearance_weight)
        )
        effective_target = (
            configured_target
            if execution_soft_clearance_target_m is None
            else float(execution_soft_clearance_target_m)
        )
        effective_taskspace_weight = (
            float(config.safemppi.taskspace_exponential_weight)
            if execution_taskspace_weight is None
            else float(execution_taskspace_weight)
        )
        if not np.isfinite(effective_weight) or effective_weight < 0.0:
            raise ValueError(
                "execution_soft_clearance_weight must be finite and "
                "nonnegative"
            )
        if not np.isfinite(effective_target) or effective_target < 0.0:
            raise ValueError(
                "execution_soft_clearance_target_m must be finite and "
                "nonnegative"
            )
        if (
            not np.isfinite(effective_taskspace_weight)
            or effective_taskspace_weight < 0.0
        ):
            raise ValueError(
                "execution_taskspace_weight must be finite and nonnegative"
            )
        escape_weight = float(execution_plane_escape_weight)
        escape_sigma = float(execution_plane_escape_sigma)
        escape_gate_radius = float(execution_plane_escape_gate_radius)
        if not np.isfinite(escape_weight) or escape_weight < 0.0:
            raise ValueError(
                "execution_plane_escape_weight must be finite and nonnegative"
            )
        if not np.isfinite(escape_sigma) or escape_sigma <= 0.0:
            raise ValueError(
                "execution_plane_escape_sigma must be finite and positive"
            )
        if not np.isfinite(escape_gate_radius) or escape_gate_radius < 0.0:
            raise ValueError(
                "execution_plane_escape_gate_radius must be finite and "
                "nonnegative"
            )
        self.config = config
        self.env = TaskEnvironment(config)
        if escape_weight != 0.0 and len(self.env.spheres) != 1:
            raise ValueError(
                "execution_plane_escape_weight requires exactly one sphere "
                "obstacle to define the escape plane"
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
        self.execution_plane_escape_weight = escape_weight
        self.execution_plane_escape_sigma = escape_sigma
        self.execution_plane_escape_gate_radius = escape_gate_radius
        self.context_schema = str(context_schema)
        self.device = torch.device(device)
        self.tight_corridor = bool(tight_corridor)
        self.verifier_mode = verifier_mode
        self.verifier_solver = verifier_solver
        self.execution_soft_clearance_weight = effective_weight
        self.execution_soft_clearance_target_m = effective_target
        self.execution_taskspace_weight = effective_taskspace_weight
        delta = self.env.goal - self.env.start[:3]
        self.forward = delta / np.linalg.norm(delta)
        self.obstacle_plane_z = float(self.env.spheres[0, 2])
        self.obstacle_plane_center_xy = np.asarray(
            self.env.spheres[0, :2], np.float64,
        )

    def reset(self, gamma: float, episode: int, seed: int) -> dict[str, Any]:
        del gamma, episode, seed
        state = {
            "x": self.env.start.copy(),
            "previous_applied": np.zeros(3, np.float32),
            "previous_raw": np.zeros(3, np.float32),
            "steps": 0,
            "collided": False,
            "oob": False,
        }
        if _is_history_context_schema(self.context_schema):
            state["raw_history"] = _raw_history_before(
                np.empty((0, 3), np.float32), 0,
            )
        return state

    def context(self, state, gamma: float) -> torch.Tensor:
        learned = policy_context(
            SimpleNamespace(context_schema=self.context_schema),
            self.env,
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

    def _plane_escape_penalty(self, point: np.ndarray) -> float:
        """Symmetric obstacle-plane occupancy of one nominal knot.

        The penalty is a Gaussian bump in the signed height offset from the
        obstacle centre plane, gated to the horizontal pass-by disc around the
        obstacle centre so that the start and goal -- which share that plane --
        are never penalised.  Escaping the plane upward or downward relieves it
        equally; the term carries no directional preference.
        """
        horizontal = float(np.linalg.norm(
            np.asarray(point[:2], np.float64) - self.obstacle_plane_center_xy
        ))
        if horizontal > self.execution_plane_escape_gate_radius:
            return 0.0
        offset = float(point[2]) - self.obstacle_plane_z
        sigma = self.execution_plane_escape_sigma
        return float(np.exp(-(offset ** 2) / (2.0 * sigma ** 2)))

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
        plane_escape = 0.0
        for index, command in enumerate(plan):
            point = states[index + 1, :3]
            distance = float(np.linalg.norm(point - self.env.goal))
            cost += cfg.running_goal_weight * distance ** 2
            cost += cfg.control_weight * float(command @ command)
            difference = command - previous
            cost += cfg.smooth_weight * float(difference @ difference)
            cost -= cfg.progress_weight * (initial_distance - distance)
            clearance = float(self.env.obstacle_clearance(point[None])[0])
            cost += self.execution_soft_clearance_weight * max(
                self.execution_soft_clearance_target_m - clearance, 0.0,
            ) ** 2
            violation = (
                np.maximum(self.env.bounds[:, 0] - point, 0.0)
                + np.maximum(point - self.env.bounds[:, 1], 0.0)
            )
            exponent = np.minimum(
                violation / cfg.taskspace_exponential_temperature, 20.0,
            )
            cost += self.execution_taskspace_weight * float(
                np.expm1(exponent).sum()
            )
            if self.execution_plane_escape_weight != 0.0:
                plane_escape += self._plane_escape_penalty(point)
            if (
                self.execution_plane_penalty_weight > 0.0
                or self.execution_above_penalty_weight > 0.0
            ):
                # Same in-plane / above shaping as the clutter task, so a
                # single-sphere run can drive below-mode with the identical
                # calibration: the above branch is only monotonically taxed
                # when above_weight > plane_weight / (2 * sigma).
                dz = float(point[2]) - self.execution_plane_penalty_plane_m
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
            np.linalg.norm(states[-1, :3] - self.env.goal)
        )
        cost += cfg.terminal_goal_weight * terminal_distance ** 2
        if self.execution_plane_escape_weight != 0.0:
            cost += self.execution_plane_escape_weight * (
                plane_escape / len(plan)
            )
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
        clearance = self.env.obstacle_clearance(dense)
        updated = {
            "x": after,
            "last_dense": dense.copy(),
            "previous_applied": applied,
            "previous_raw": limited_command,
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

    def successful_trajectory_mode(self, executed_states) -> int | None:
        """Evaluation-identical route ID from the dense executed path."""
        if not executed_states:
            return None
        dense_path = np.concatenate([
            self.env.start[None, :3],
            *[
                np.asarray(state["last_dense"], np.float32).reshape(-1, 3)
                for state in executed_states
            ],
        ])
        name = theta_name(trajectory_crossing_theta(self.env, dense_path))
        return {
            "below": 0,
            "above": 1,
            "left": 2,
            "right": 3,
        }.get(name)
