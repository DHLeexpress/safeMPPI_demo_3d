"""Isolated PRE2 multi-sphere expansion task.

This adapter intentionally leaves the established single-sphere and clutter
tasks unchanged.  It adds the two execution-ranking costs used by the PRE2
preflight and makes a reset seed gamma-aware so each gamma receives a distinct
domain-randomized scene even when the generic expansion loop reuses a replica
seed across gamma cells.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .lab_clutter_expansion import (
    LAB_CLUTTER_GOVERNOR_DIM,
    LabClutterExpansionPolicyAdapter,
    LabClutterSphereExpansionTask,
    scene_sha256,
)
from .lab_flow_expansion import _is_history_context_schema
from .lab_reference_flow_task import _raw_history_before
from .lab_visual_flow import load_lab_reference_policy


def gamma_scene_seed(seed: int, gamma: float) -> int:
    """Mix an exact gamma cell into an episode reset seed deterministically."""
    gamma_key = int(round(float(gamma) * 1_000_000.0))
    if not np.isclose(
        gamma_key / 1_000_000.0, float(gamma), rtol=0.0, atol=1.0e-9,
    ):
        raise ValueError("gamma must be stable to six decimal places")
    return int(np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        gamma_key & 0xFFFFFFFF,
        0x50524532,
    ]).generate_state(1, dtype=np.uint32)[0])


def _gamma_key(gamma: float) -> int:
    key = int(round(float(gamma) * 1_000_000.0))
    if not np.isclose(key / 1_000_000.0, gamma, rtol=0.0, atol=5.0e-7):
        raise ValueError("gamma must be stable to six decimal places")
    return key


def rotate_points_180_about_start_goal_axis(
    points: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Rotate 3-D points by pi around the infinite start-to-goal axis."""
    values = np.asarray(points, np.float64).reshape(-1, 3)
    origin = np.asarray(start, np.float64).reshape(-1)[:3]
    direction = np.asarray(goal, np.float64).reshape(-1)[:3] - origin
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        raise ValueError("rotation axis requires distinct start and goal")
    direction /= length
    relative = values - origin[None]
    parallel = (relative @ direction)[:, None] * direction[None]
    return origin[None] + 2.0 * parallel - relative


def bowling_123_spheres(
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return the fixed 1-2-3 bowling layout in the start-goal frame."""
    start3 = np.asarray(start, np.float64).reshape(-1)[:3]
    goal3 = np.asarray(goal, np.float64).reshape(-1)[:3]
    delta = goal3 - start3
    forward = delta / np.linalg.norm(delta)
    lateral = np.asarray([-forward[1], forward[0], 0.0], np.float64)
    # 0.345/0.690 m (instead of the diagnostic's rounded 0.34/0.68)
    # leaves a 0.209 m modeled-surface gap with r=0.2405 m, clearing the
    # configured 0.20 m minimum without a float32 boundary ambiguity.
    centers = (
        start3 + 0.25 * delta,
        start3 + 0.45 * delta - 0.345 * lateral,
        start3 + 0.45 * delta + 0.345 * lateral,
        start3 + 0.65 * delta - 0.690 * lateral,
        start3 + 0.65 * delta,
        start3 + 0.65 * delta + 0.690 * lateral,
    )
    return np.asarray(
        [[*center, float(radius)] for center in centers], np.float32,
    )


class LabClutterPre2ExpansionPolicyAdapter(
    LabClutterExpansionPolicyAdapter
):
    """Salt rollout latents by gamma while preserving shared core code."""

    @staticmethod
    def _salted_generator(
        context: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Generator:
        policy_context = context.reshape(-1)
        if len(policy_context) < 7:
            raise ValueError("PRE2 policy context is missing gamma")
        device = policy_context.device
        upstream = int(torch.randint(
            0,
            2**31 - 1,
            (1,),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).detach().cpu()[0])
        seed = int(np.random.SeedSequence([
            upstream,
            _gamma_key(float(policy_context[6].detach().cpu())),
            0x4C415445,
        ]).generate_state(2, dtype=np.uint32).view(np.uint64)[0])
        return torch.Generator(device=device).manual_seed(seed)

    def sample(self, context, count, generator, base_std=1.0):
        policy_context = self._policy_context(context)
        return self.policy.sample(
            policy_context,
            count,
            self._salted_generator(policy_context, generator),
            base_std=base_std,
        )

    def sample_with_base(self, context, count, generator, base_std=1.0):
        policy_context = self._policy_context(context)
        return self.policy.sample_with_base(
            policy_context,
            count,
            self._salted_generator(policy_context, generator),
            base_std=base_std,
        )

    @torch.no_grad()
    def sample_many_with_base(
        self, contexts, count, generators, base_std=1.0,
    ):
        policy_contexts = self._policy_context(contexts)
        if len(policy_contexts) != len(generators):
            raise ValueError("one generator is required per rollout context")
        salted = [
            self._salted_generator(context, generator)
            for context, generator in zip(policy_contexts, generators)
        ]
        encoded = self.policy.encode_context(policy_contexts)
        flow = self.policy.flow
        bases = torch.stack([
            torch.randn(
                count,
                flow.plan_dim,
                device=policy_contexts.device,
                generator=generator,
            ) * float(base_std)
            for generator in salted
        ])
        repeated_context = encoded[:, None, :].expand(
            len(policy_contexts), count, encoded.shape[-1]
        ).reshape(len(policy_contexts) * count, encoded.shape[-1])
        integrated = flow._integrate_flow(
            bases.reshape(len(policy_contexts) * count, flow.plan_dim),
            repeated_context,
        ).reshape(len(policy_contexts), count, *flow.plan_shape)
        if flow.control_limit is not None:
            integrated = integrated.clamp(
                -flow.control_limit, flow.control_limit,
            )
        return integrated, bases.reshape(
            len(policy_contexts), count, *flow.plan_shape,
        )


def load_lab_clutter_pre2_expansion_policy(
    path,
    *,
    verifier_suffix_dim: int,
    train_history_encoder: bool = False,
) -> LabClutterPre2ExpansionPolicyAdapter:
    policy = load_lab_reference_policy(path)
    is_history = _is_history_context_schema(
        getattr(policy, "context_schema", ""),
    )
    if train_history_encoder and not is_history:
        raise ValueError(
            "train_history_encoder applies only to a GRU checkpoint"
        )
    if verifier_suffix_dim < LAB_CLUTTER_GOVERNOR_DIM:
        raise ValueError("clutter verifier suffix is too short")
    return LabClutterPre2ExpansionPolicyAdapter(
        policy,
        verifier_suffix_dim=verifier_suffix_dim,
        freeze_history_encoder=(
            not train_history_encoder if is_history else None
        ),
    )


class LabClutterPre2ExpansionTask(LabClutterSphereExpansionTask):
    """Dynamic clutter task with gamma-distinct scenes and interior costs."""

    def __init__(
        self,
        *args,
        execution_taskspace_quadratic_weight: float = 0.0,
        execution_taskspace_quadratic_target_m: float = 0.10,
        execution_axis_cylinder_quadratic_weight: float = 0.0,
        execution_axis_cylinder_radius_m: float = 1.10,
        execution_control_weight: float | None = None,
        execution_obstacle_speed_weight: float = 0.0,
        paired_scene_rotation: str = "none",
        paired_scene_seed: int = 0,
        paired_scene_max_proposals: int = 10_000,
        fixed_scene_layout: str = "none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        values = (
            execution_taskspace_quadratic_weight,
            execution_taskspace_quadratic_target_m,
            execution_axis_cylinder_quadratic_weight,
            execution_axis_cylinder_radius_m,
            execution_obstacle_speed_weight,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("PRE2 execution-cost parameters must be finite")
        if (
            execution_taskspace_quadratic_weight < 0.0
            or execution_taskspace_quadratic_target_m < 0.0
            or execution_axis_cylinder_quadratic_weight < 0.0
            or execution_axis_cylinder_radius_m <= 0.0
            or execution_obstacle_speed_weight < 0.0
        ):
            raise ValueError(
                "PRE2 execution weights/targets must be nonnegative and the "
                "axis-cylinder radius positive"
            )
        self.execution_taskspace_quadratic_weight = float(
            execution_taskspace_quadratic_weight
        )
        self.execution_taskspace_quadratic_target_m = float(
            execution_taskspace_quadratic_target_m
        )
        self.execution_axis_cylinder_quadratic_weight = float(
            execution_axis_cylinder_quadratic_weight
        )
        self.execution_axis_cylinder_radius_m = float(
            execution_axis_cylinder_radius_m
        )
        self.execution_obstacle_speed_weight = float(
            execution_obstacle_speed_weight
        )
        self.execution_control_weight = (
            float(self.config.safemppi.control_weight)
            if execution_control_weight is None
            else float(execution_control_weight)
        )
        if (
            not np.isfinite(self.execution_control_weight)
            or self.execution_control_weight < 0.0
        ):
            raise ValueError(
                "execution_control_weight must be finite and nonnegative"
            )
        axis = np.asarray(self.env.goal - self.env.start[:3], np.float64)
        self._axis_unit = axis / np.linalg.norm(axis)

        if paired_scene_rotation not in {"none", "start_goal_axis_180"}:
            raise ValueError(
                "paired_scene_rotation must be none or start_goal_axis_180"
            )
        if fixed_scene_layout not in {"none", "bowling_123"}:
            raise ValueError(
                "fixed_scene_layout must be none or bowling_123"
            )
        if paired_scene_rotation != "none" and fixed_scene_layout != "none":
            raise ValueError(
                "paired randomized scenes and a fixed scene are mutually exclusive"
            )
        if int(paired_scene_max_proposals) < 1:
            raise ValueError("paired_scene_max_proposals must be positive")
        self.paired_scene_rotation = str(paired_scene_rotation)
        self.paired_scene_seed = int(paired_scene_seed)
        self.paired_scene_max_proposals = int(paired_scene_max_proposals)
        self.fixed_scene_layout = str(fixed_scene_layout)
        self.expansion_round = 1
        self._paired_scene_cache: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]
        ] = {}
        self._fixed_spheres = None
        if self.fixed_scene_layout == "bowling_123":
            self._fixed_spheres = self.scene_spec.validate(
                self.env,
                bowling_123_spheres(
                    self.env.start,
                    self.env.goal,
                    self.scene_spec.radius,
                ),
            )

    def begin_expansion_round(
        self,
        round_index: int,
        *,
        clear_scene_ledger: bool = False,
    ) -> None:
        """Select the deterministic scene-pair namespace for a real round."""
        if int(round_index) < 1:
            raise ValueError("expansion round must be positive")
        self.expansion_round = int(round_index)
        if clear_scene_ledger:
            self.scene_ledger.clear()

    def _new_state(
        self,
        *,
        gamma: float,
        episode: int,
        spheres: np.ndarray,
        scene_seed: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spheres = self.scene_spec.validate(self.env, spheres)
        scene_env = self._environment(spheres)
        state: dict[str, Any] = {
            "x": scene_env.start.copy(),
            "previous_applied": np.zeros(3, np.float32),
            "previous_raw": np.zeros(3, np.float32),
            "spheres": spheres,
            "scene_seed": int(scene_seed),
            "scene_hash": scene_sha256(scene_env, spheres),
            "steps": 0,
            "collided": False,
            "oob": False,
        }
        if metadata:
            state.update(metadata)
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

    def _paired_spheres(
        self,
        gamma: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        key = (self.expansion_round, _gamma_key(gamma))
        cached = self._paired_scene_cache.get(key)
        if cached is not None:
            return cached
        pair_seed = int(np.random.SeedSequence([
            self.paired_scene_seed & 0xFFFFFFFF,
            self.expansion_round & 0xFFFFFFFF,
            key[1] & 0xFFFFFFFF,
            0x50414952,
        ]).generate_state(1, dtype=np.uint32)[0])
        for proposal_index in range(self.paired_scene_max_proposals):
            scene_seed = int(np.random.SeedSequence([
                pair_seed,
                proposal_index,
            ]).generate_state(1, dtype=np.uint32)[0])
            source = self.scene_spec.sample(self.env, scene_seed)
            rotated = source.copy()
            rotated[:, :3] = rotate_points_180_about_start_goal_axis(
                source[:, :3], self.env.start, self.env.goal,
            ).astype(np.float32)
            try:
                source = self.scene_spec.validate(self.env, source)
                rotated = self.scene_spec.validate(self.env, rotated)
            except ValueError:
                continue
            source_env = self._environment(source)
            rotated_env = self._environment(rotated)
            source_hash = scene_sha256(source_env, source)
            rotated_hash = scene_sha256(rotated_env, rotated)
            if source_hash == rotated_hash:
                continue
            recovered = rotate_points_180_about_start_goal_axis(
                rotated[:, :3], self.env.start, self.env.goal,
            )
            recovered_rows = rotated.copy()
            recovered_rows[:, :3] = recovered.astype(np.float32)
            recovered_rows = self.scene_spec.validate(
                self.env, recovered_rows,
            )
            if not np.allclose(
                recovered_rows,
                source,
                rtol=0.0,
                atol=2.0e-6,
            ):
                raise RuntimeError("paired sphere rotation failed involution")
            metadata = {
                "paired_scene_id": (
                    f"r{self.expansion_round:03d}_g{float(gamma):.9g}"
                ),
                "paired_scene_seed": scene_seed,
                "paired_scene_proposal_index": proposal_index,
                "paired_source_scene_hash": source_hash,
                "paired_rotated_scene_hash": rotated_hash,
                "paired_rotation": "start_goal_axis_180",
            }
            cached = (source, rotated, metadata)
            self._paired_scene_cache[key] = cached
            return cached
        raise RuntimeError(
            "could not sample a start-goal-axis-rotatable sphere scene within "
            f"{self.paired_scene_max_proposals} proposals"
        )

    def reset(self, gamma: float, episode: int, seed: int) -> dict[str, Any]:
        if self.paired_scene_rotation != "none":
            source, rotated, pair_metadata = self._paired_spheres(gamma)
            member = int(episode) % 2
            spheres = source if member == 0 else rotated
            metadata = {
                **pair_metadata,
                "paired_scene_member": member,
                "paired_scene_member_name": (
                    "original" if member == 0 else "axis_180"
                ),
                "base_scene_seed": int(seed),
            }
            return self._new_state(
                gamma=gamma,
                episode=episode,
                spheres=spheres,
                scene_seed=int(pair_metadata["paired_scene_seed"]),
                metadata=metadata,
            )
        if self._fixed_spheres is not None:
            return self._new_state(
                gamma=gamma,
                episode=episode,
                spheres=self._fixed_spheres.copy(),
                scene_seed=self.paired_scene_seed,
                metadata={
                    "fixed_scene_layout": self.fixed_scene_layout,
                    "base_scene_seed": int(seed),
                },
            )
        effective_seed = gamma_scene_seed(seed, gamma)
        state = super().reset(gamma, episode, effective_seed)
        state["base_scene_seed"] = int(seed)
        return state

    def scene_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        metadata = super().scene_metadata(state)
        if "base_scene_seed" in state:
            metadata["base_scene_seed"] = int(state["base_scene_seed"])
            if "paired_scene_id" in state:
                metadata["scene_seed_derivation"] = (
                    "paired_round_gamma_scene_seed_v1"
                )
            elif "fixed_scene_layout" in state:
                metadata["scene_seed_derivation"] = "fixed_scene_v1"
            else:
                metadata["scene_seed_derivation"] = "gamma_scene_seed_v1"
        for name in (
            "paired_scene_id",
            "paired_scene_seed",
            "paired_scene_proposal_index",
            "paired_source_scene_hash",
            "paired_rotated_scene_hash",
            "paired_rotation",
            "paired_scene_member",
            "paired_scene_member_name",
            "fixed_scene_layout",
        ):
            if name in state:
                metadata[name] = state[name]
        return metadata

    def advance(self, state, candidate: torch.Tensor):
        updated = super().advance(state, candidate)
        for name in (
            "base_scene_seed",
            "paired_scene_id",
            "paired_scene_seed",
            "paired_scene_proposal_index",
            "paired_source_scene_hash",
            "paired_rotated_scene_hash",
            "paired_rotation",
            "paired_scene_member",
            "paired_scene_member_name",
            "fixed_scene_layout",
        ):
            if name in state:
                updated[name] = state[name]
        return updated

    def successful_trajectory_mode(self, executed_states) -> int | None:
        """Expose original/rotated membership to the isolated pair quota."""
        if self.paired_scene_rotation == "none":
            return None
        if not executed_states:
            return None
        members = {
            int(state["paired_scene_member"])
            for state in executed_states
            if "paired_scene_member" in state
        }
        if len(members) != 1:
            raise RuntimeError(
                "successful paired trajectory lost a stable member label"
            )
        return members.pop()

    def execution_cost_breakdown(
        self, env, states, plan, previous_raw,
    ) -> dict[str, float]:
        base = super()._native_cost(env, states, plan, previous_raw)
        configured_control = float(self.config.safemppi.control_weight)
        control_override = 0.0
        if self.execution_control_weight != configured_control:
            commands = np.asarray(plan, np.float64).reshape(-1, 3)
            control_override = (
                self.execution_control_weight - configured_control
            ) * float(np.square(commands).sum(axis=1).sum())
        positions = np.asarray(states, np.float64)[1:, :3]
        wall = 0.0
        if self.execution_taskspace_quadratic_weight > 0.0:
            face_clearance = np.concatenate([
                positions - env.bounds[:, 0][None],
                env.bounds[:, 1][None] - positions,
            ], axis=1)
            shortfall = np.maximum(
                self.execution_taskspace_quadratic_target_m - face_clearance,
                0.0,
            )
            wall = self.execution_taskspace_quadratic_weight * float(
                np.square(shortfall).sum(axis=1).mean()
            )
        axis = 0.0
        if self.execution_axis_cylinder_quadratic_weight > 0.0:
            displacement = positions[-1] - self.env.start[:3]
            axial = float(displacement @ self._axis_unit) * self._axis_unit
            radial_distance = float(np.linalg.norm(displacement - axial))
            axis = self.execution_axis_cylinder_quadratic_weight * (
                radial_distance / self.execution_axis_cylinder_radius_m
            ) ** 2
        obstacle_speed = 0.0
        if self.execution_obstacle_speed_weight > 0.0:
            clearances = np.asarray(
                env.obstacle_clearance(positions), np.float64,
            )
            scaled = np.clip(
                (
                    clearances - self.execution_clearance_target_m
                ) / self.execution_clearance_exp_temperature,
                -60.0,
                60.0,
            )
            proximity = 1.0 / (1.0 + np.exp(scaled))
            speed_squared = np.square(
                np.asarray(states, np.float64)[1:, 3:6]
            ).sum(axis=1)
            obstacle_speed = self.execution_obstacle_speed_weight * float(
                np.mean(proximity * speed_squared)
            )
        return {
            "base_native": float(base),
            "control_override_delta": float(control_override),
            "interior_wall": float(wall),
            "axis_cylinder": float(axis),
            "obstacle_conditioned_speed": float(obstacle_speed),
            "total": float(
                base + control_override + wall + axis + obstacle_speed
            ),
        }

    def _native_cost(self, env, states, plan, previous_raw) -> float:
        return self.execution_cost_breakdown(
            env, states, plan, previous_raw,
        )["total"]
