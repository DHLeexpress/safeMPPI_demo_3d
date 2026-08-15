"""CFM--MPPI deployment for the fixed 3-D multi-sphere task.

This is an isolated port of the paper baseline.  It deliberately does not use
the H_P/NVP verifier for proposal selection: guidance acts inside the flow ODE,
then the configured native SafeMPPI cost ranks and refines raw action windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class CfmMppiConfig:
    proposal_count: int = 32
    elite_count: int = 8
    copies_per_elite: int = 32
    mppi_sigma: float = 0.20
    mppi_lambda: float = 0.10
    alpha_cbf: float = 1.0
    cbf_margin_m: float = 0.0
    worst_constraint_count: int = 5
    markup: float = 1.01
    goal_coefficient_max: float = 0.25
    safety_coefficient_max: float = 0.50
    normalized_goal: float = 0.5
    normalized_safety: float = 0.5
    warm_start_tau: float = 0.75
    base_std: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.proposal_count,
            self.elite_count,
            self.copies_per_elite,
            self.mppi_sigma,
            self.mppi_lambda,
            self.alpha_cbf,
            self.worst_constraint_count,
            self.markup,
            self.goal_coefficient_max,
            self.safety_coefficient_max,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in positive):
            raise ValueError("CFM--MPPI positive parameters must be finite and positive")
        if self.elite_count > self.proposal_count:
            raise ValueError("elite_count cannot exceed proposal_count")
        if not 0.0 <= self.cbf_margin_m or not 0.0 <= self.base_std:
            raise ValueError("CBF margin and base_std must be nonnegative")
        if not 0.0 <= self.normalized_goal <= 1.0:
            raise ValueError("normalized_goal must lie in [0,1]")
        if not 0.0 <= self.normalized_safety <= 1.0:
            raise ValueError("normalized_safety must lie in [0,1]")
        if not 0.0 < self.warm_start_tau < 1.0:
            raise ValueError("warm_start_tau must lie in (0,1)")

    @property
    def goal_coefficient(self) -> float:
        return self.normalized_goal * self.goal_coefficient_max

    @property
    def safety_coefficient(self) -> float:
        return self.normalized_safety * self.safety_coefficient_max


def reference_rollout_torch(
    state6: torch.Tensor,
    raw_plans: torch.Tensor,
    previous_applied: torch.Tensor,
    mppi_config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact batched differentiable form of :class:`ReferenceGovernor`."""
    count, horizon, width = raw_plans.shape
    if width != 3:
        raise ValueError("3-D raw action plans are required")
    raw = raw_plans.clamp(-mppi_config.demo_u_max, mppi_config.demo_u_max)
    state = state6.reshape(1, 6).expand(count, -1)
    applied_previous = previous_applied.reshape(1, 3).expand(count, -1)
    states = [state]
    applied_rows = []
    dt_sub = float(mppi_config.dt) / int(mppi_config.integration_substeps)
    for step in range(horizon):
        applied = (
            float(mppi_config.deployment_accel_smooth) * raw[:, step]
            + (1.0 - float(mppi_config.deployment_accel_smooth)) * applied_previous
        )
        position = state[:, :3]
        velocity = state[:, 3:6]
        for _ in range(int(mppi_config.integration_substeps)):
            velocity = velocity + dt_sub * applied
            speed = torch.linalg.norm(velocity, dim=1, keepdim=True)
            velocity = velocity * torch.clamp(
                float(mppi_config.max_speed) / speed.clamp_min(1.0e-12),
                max=1.0,
            )
            vertical = velocity[:, 2].clamp(
                -float(mppi_config.max_vertical_speed),
                float(mppi_config.max_vertical_speed),
            )
            velocity = torch.cat((velocity[:, :2], vertical[:, None]), dim=1)
            position = position + dt_sub * velocity
        state = torch.cat((position, velocity), dim=1)
        states.append(state)
        applied_rows.append(applied)
        applied_previous = applied
    return torch.stack(states, dim=1), torch.stack(applied_rows, dim=1)


def native_safemppi_costs(
    state6: torch.Tensor,
    raw_plans: torch.Tensor,
    previous_raw: torch.Tensor,
    previous_applied: torch.Tensor,
    goal: torch.Tensor,
    spheres: torch.Tensor,
    bounds: torch.Tensor,
    mppi_config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native bowling SafeMPPI soft cost, without its H_P hard gate."""
    plans = raw_plans.clamp(-mppi_config.demo_u_max, mppi_config.demo_u_max)
    states, _ = reference_rollout_torch(
        state6, plans, previous_applied, mppi_config,
    )
    points = states[:, 1:, :3]
    initial_distance = torch.linalg.norm(state6[:3] - goal)
    distance = torch.linalg.norm(points - goal[None, None], dim=2)
    costs = float(mppi_config.running_goal_weight) * distance.square().sum(dim=1)
    costs += float(mppi_config.control_weight) * plans.square().sum(dim=(1, 2))
    previous = torch.cat((
        previous_raw.reshape(1, 1, 3).expand(len(plans), -1, -1),
        plans[:, :-1],
    ), dim=1)
    costs += float(mppi_config.smooth_weight) * (plans - previous).square().sum(dim=(1, 2))
    costs -= float(mppi_config.progress_weight) * (initial_distance - distance).sum(dim=1)
    if float(mppi_config.soft_clearance_weight) > 0.0 and spheres.numel():
        clearance = (
            torch.linalg.norm(points[:, :, None] - spheres[None, None, :, :3], dim=3)
            - spheres[None, None, :, 3]
        ).amin(dim=2)
        costs += float(mppi_config.soft_clearance_weight) * torch.relu(
            float(mppi_config.soft_clearance_target) - clearance
        ).square().sum(dim=1)
    if float(mppi_config.taskspace_exponential_weight) > 0.0:
        violation = (
            torch.relu(bounds[None, None, :, 0] - points)
            + torch.relu(points - bounds[None, None, :, 1])
        )
        exponent = (
            violation / float(mppi_config.taskspace_exponential_temperature)
        ).clamp(max=20.0)
        costs += float(mppi_config.taskspace_exponential_weight) * torch.expm1(
            exponent
        ).sum(dim=(1, 2))
    costs += float(mppi_config.terminal_goal_weight) * distance[:, -1].square()
    return costs, states


def cbf_reward(
    states: torch.Tensor,
    spheres: torch.Tensor,
    bounds: torch.Tensor,
    *,
    alpha: float,
    margin_m: float,
    worst_count: int,
) -> torch.Tensor:
    """Signed-clearance CBF reward for spheres and all six task-space faces."""
    position = states[:, 1:, :3]
    velocity = states[:, 1:, 3:6]
    residuals = []
    if spheres.numel():
        relative = position[:, :, None] - spheres[None, None, :, :3]
        distance = torch.linalg.norm(relative, dim=3).clamp_min(1.0e-8)
        h = distance - spheres[None, None, :, 3] - float(margin_m)
        h_dot = (relative * velocity[:, :, None]).sum(dim=3) / distance
        residuals.append(h_dot + float(alpha) * h)
    lower_h = position - bounds[None, None, :, 0] - float(margin_m)
    upper_h = bounds[None, None, :, 1] - position - float(margin_m)
    residuals.extend((
        velocity + float(alpha) * lower_h,
        -velocity + float(alpha) * upper_h,
    ))
    all_residuals = torch.cat(residuals, dim=2)
    count = min(int(worst_count), all_residuals.shape[2])
    worst = torch.topk(all_residuals, k=count, dim=2, largest=False).values
    weights = torch.arange(
        count, 0, -1, device=states.device, dtype=states.dtype,
    )
    return (torch.minimum(worst, torch.zeros_like(worst)) * weights).sum(dim=(1, 2))


def _global_normalize(gradient: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    return gradient * (torch.linalg.norm(field) / torch.linalg.norm(gradient).clamp_min(1.0e-8))


def guided_flow_proposals(
    wrapped,
    context: torch.Tensor,
    state6: torch.Tensor,
    previous_applied: torch.Tensor,
    goal: torch.Tensor,
    spheres: torch.Tensor,
    bounds: torch.Tensor,
    generator: torch.Generator,
    config: CfmMppiConfig,
    previous_plan: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Create 32 guided PRE2 plans with the reference batch-global gradients."""
    policy_context = wrapped._policy_context(context)
    salted = wrapped._salted_generator(policy_context, generator)
    encoded = wrapped.policy.encode_context(policy_context)
    flow = wrapped.policy.flow
    count = int(config.proposal_count)
    encoded = encoded.reshape(1, -1).expand(count, -1)
    if previous_plan is None:
        z = torch.randn(
            count, flow.plan_dim, device=context.device, generator=salted,
        ) * float(config.base_std)
        first_index = 0
    else:
        shifted = torch.cat((previous_plan[1:], previous_plan[-1:]), dim=0)
        z = (
            float(config.warm_start_tau) * shifted.reshape(1, -1).expand(count, -1)
            + (1.0 - float(config.warm_start_tau))
            * torch.randn(count, flow.plan_dim, device=context.device, generator=salted)
        )
        first_index = int(math.ceil(float(config.warm_start_tau) * int(flow.nfe)))
    markup = float(config.markup) ** torch.arange(
        flow.plan_shape[0] - 1, -1, -1,
        device=context.device, dtype=z.dtype,
    )[None, :, None]
    goal_norms, safety_norms, field_norms = [], [], []
    for index in range(first_index, int(flow.nfe)):
        tau = index / int(flow.nfe)
        t = torch.full((count,), tau, device=context.device, dtype=z.dtype)
        with torch.no_grad():
            field = flow(z, t, encoded)
        endpoint = (z + (1.0 - tau) * field).detach().requires_grad_(True)
        plans = endpoint.reshape(count, *flow.plan_shape)
        states, _ = reference_rollout_torch(
            state6, plans, previous_applied, wrapped.policy.flow_config,
        ) if hasattr(wrapped.policy, "flow_config") else reference_rollout_torch(
            state6, plans, previous_applied, wrapped._cfm_mppi_config,
        )
        safe_value = cbf_reward(
            states, spheres, bounds,
            alpha=config.alpha_cbf,
            margin_m=config.cbf_margin_m,
            worst_count=config.worst_constraint_count,
        ).sum()
        goal_value = -torch.linalg.norm(states[:, -1, :3] - goal[None], dim=1).sum()
        safe_gradient, = torch.autograd.grad(safe_value, endpoint, retain_graph=True)
        goal_gradient, = torch.autograd.grad(goal_value, endpoint)
        safe_gradient = _global_normalize(safe_gradient, field).reshape(
            count, *flow.plan_shape,
        )
        goal_gradient = _global_normalize(goal_gradient, field).reshape(
            count, *flow.plan_shape,
        )
        guidance = (
            config.goal_coefficient * goal_gradient
            + config.safety_coefficient * safe_gradient * markup
        )
        field_norms.append(float(torch.linalg.norm(field).detach().cpu()))
        goal_norms.append(float(torch.linalg.norm(config.goal_coefficient * goal_gradient).detach().cpu()))
        safety_norms.append(float(torch.linalg.norm(config.safety_coefficient * safe_gradient * markup).detach().cpu()))
        z = z + (field + guidance.reshape(count, -1)) / int(flow.nfe)
    plans = z.reshape(count, *flow.plan_shape)
    if flow.control_limit is not None:
        plans = plans.clamp(-flow.control_limit, flow.control_limit)
    return plans.detach(), {
        "mean_field_norm": float(np.mean(field_norms)),
        "mean_goal_guidance_norm": float(np.mean(goal_norms)),
        "mean_safety_guidance_norm": float(np.mean(safety_norms)),
    }


def refine_plans(
    generated: torch.Tensor,
    state6: torch.Tensor,
    previous_raw: torch.Tensor,
    previous_applied: torch.Tensor,
    goal: torch.Tensor,
    spheres: torch.Tensor,
    bounds: torch.Tensor,
    mppi_config,
    generator: torch.Generator,
    config: CfmMppiConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Top-E selection, per-elite MPPI averaging, and final native-cost rank."""
    with torch.no_grad():
        generated_cost, generated_states = native_safemppi_costs(
            state6, generated, previous_raw, previous_applied,
            goal, spheres, bounds, mppi_config,
        )
        top = torch.topk(
            generated_cost, k=min(config.elite_count, len(generated)), largest=False,
        ).indices
        elites = generated[top]
        elite_count = len(elites)
        perturbation = torch.randn(
            elite_count, config.copies_per_elite, *elites.shape[1:],
            generator=generator, device=generated.device,
        ) * float(config.mppi_sigma)
        perturbed = (
            elites[:, None] + perturbation
        ).clamp(-mppi_config.demo_u_max, mppi_config.demo_u_max)
        flat = perturbed.reshape(-1, *generated.shape[1:])
        perturbed_cost, _ = native_safemppi_costs(
            state6, flat, previous_raw, previous_applied,
            goal, spheres, bounds, mppi_config,
        )
        perturbed_cost = perturbed_cost.reshape(elite_count, config.copies_per_elite)
        baseline = perturbed_cost.amin(dim=1, keepdim=True)
        weights = torch.softmax(
            -(perturbed_cost - baseline) / float(config.mppi_lambda), dim=1,
        )
        refined = (weights[:, :, None, None] * perturbed).sum(dim=1)
        refined_cost, refined_states = native_safemppi_costs(
            state6, refined, previous_raw, previous_applied,
            goal, spheres, bounds, mppi_config,
        )
        best = int(torch.argmin(refined_cost))
    return refined[best], {
        "generated_costs": generated_cost.detach(),
        "generated_states": generated_states.detach(),
        "refined_costs": refined_cost.detach(),
        "refined_states": refined_states.detach(),
        "refined_plans": refined.detach(),
        "best_refined_index": torch.tensor(best),
    }


def attach_reference_config(wrapped, mppi_config) -> None:
    """Bind the deployment recurrence without modifying the PRE2 checkpoint."""
    wrapped._cfm_mppi_config = mppi_config
