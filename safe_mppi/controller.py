"""Current mode-1-only SafeMPPI controller; alternate sampling modes are intentionally absent."""
from __future__ import annotations

import time

import numpy as np
import torch

from .config import MPPIConfig
from .environment import TaskEnvironment
from .geometry import build_nominal_polytope, hp_values, polytope_centroid


def _clearance(points, spheres, cylinders):
    clearance = torch.full((len(points),), float("inf"), device=points.device, dtype=points.dtype)
    if spheres.numel():
        distance = torch.linalg.norm(points[:, None] - spheres[None, :, :3], dim=2)
        clearance = torch.minimum(clearance, (distance - spheres[None, :, 3]).min(dim=1).values)
    if cylinders.numel():
        distance = torch.linalg.norm(points[:, None, :2] - cylinders[None, :, :2], dim=2)
        clearance = torch.minimum(clearance, (distance - cylinders[None, :, 2]).min(dim=1).values)
    return clearance


class Mode1SafeMPPI:
    """Gaussian warm-start plus the current centroid-anisotropic mode-1 mixture, nothing else."""

    def __init__(self, config: MPPIConfig, environment: TaskEnvironment, device="cpu"):
        self.cfg = config
        self.env = environment
        self.device = torch.device(device)
        self.u_min = torch.full((3,), -config.demo_u_max, device=self.device)
        self.u_max = torch.full((3,), config.demo_u_max, device=self.device)
        self.sigma = torch.tensor(config.noise_sigma, device=self.device)
        self.spheres, self.cylinders = environment.torch_obstacles(self.device)
        self._previous_sequence = None
        self._last_action = None
        self._previous_mix = None

    def reset(self):
        self._previous_sequence = None
        self._last_action = None
        self._previous_mix = None

    def _step(self, states, controls):
        dt = self.cfg.dt
        position = states[:, :3] + dt * states[:, 3:6] + 0.5 * dt * dt * controls
        velocity = states[:, 3:6] + dt * controls
        return torch.cat([position, velocity], dim=1)

    @staticmethod
    def _hp(points, A, b, margins):
        return ((b[None] - points @ A.T) / margins[None]).min(dim=1).values

    def plan(self, state, goal, gamma, seed):
        started = time.perf_counter()
        cfg = self.cfg
        state_np = np.asarray(state, np.float32).reshape(6)
        goal_np = np.asarray(goal, np.float32).reshape(3)
        poly = build_nominal_polytope(
            state_np[:3], self.env.spheres, self.env.cylinders, self.env.bounds,
            sensing_range=cfg.sensing_range, obstacle_margin=cfg.obstacle_margin)
        A = torch.tensor(poly.A, device=self.device, dtype=torch.float32)
        b = torch.tensor(poly.b, device=self.device, dtype=torch.float32)
        center = torch.tensor(poly.center, device=self.device, dtype=torch.float32)
        margins = (b - A @ center).clamp_min(1e-3)
        centroid_np = polytope_centroid(poly)
        centroid_delta = torch.tensor(centroid_np - poly.center, device=self.device,
                                      dtype=torch.float32)
        centroid_norm = torch.linalg.norm(centroid_delta)
        centroid_direction = (centroid_delta / centroid_norm if float(centroid_norm) > 1e-6
                              else torch.zeros(3, device=self.device))
        size = float(margins.min().detach().cpu())
        trapped = max(0.0, (cfg.sensing_range - size) / (size + cfg.centroid_eps))

        H, N = cfg.horizon, cfg.num_samples
        if cfg.warm_start and self._previous_sequence is not None:
            nominal = torch.cat([self._previous_sequence[1:], self._previous_sequence[-1:]], dim=0)
        else:
            nominal = torch.tensor(
                cfg.initial_control, dtype=torch.float32, device=self.device
            ).repeat(H, 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(N, H, 3, generator=generator, device=self.device) * self.sigma
        centers = nominal[None].expand(N, -1, -1).clone()

        mix = 0.0
        if float(torch.linalg.norm(centroid_direction)) > 1e-6:
            mix = min(max(cfg.centroid_gain * trapped, cfg.urgency_floor), 1.0)
            if self._previous_mix is not None:
                mix = (1.0 - cfg.centroid_smooth) * mix + cfg.centroid_smooth * self._previous_mix
            self._previous_mix = mix
            n_centroid = min(int(round(mix * N)), N)
            if n_centroid:
                start = N - n_centroid
                target = cfg.demo_u_max * centroid_direction
                centers[start:] += target[None, None]
                direction = centroid_direction[None, None]
                component = (noise[start:] * direction).sum(dim=-1, keepdim=True)
                noise[start:] += (cfg.sigma_aniso - 1.0) * component * direction
        controls = torch.clamp(centers + noise, self.u_min, self.u_max)

        state0 = torch.tensor(state_np, device=self.device).view(1, 6).expand(N, -1)
        goal_t = torch.tensor(goal_np, device=self.device)
        states = state0.clone()
        costs = torch.zeros(N, device=self.device)
        infeasible = torch.zeros(N, dtype=torch.bool, device=self.device)
        minimum_hp = torch.full((N,), float("inf"), device=self.device)
        initial_distance = torch.linalg.norm(states[:, :3] - goal_t, dim=1)
        previous_action = (torch.zeros(N, 3, device=self.device) if self._last_action is None
                           else self._last_action[None].expand(N, -1))
        for step in range(H):
            next_states = self._step(states, controls[:, step])
            old_hp = self._hp(states[:, :3], A, b, margins)
            new_hp = self._hp(next_states[:, :3], A, b, margins)
            minimum_hp = torch.minimum(minimum_hp, new_hp)
            infeasible |= new_hp < (1.0 - float(gamma)) * old_hp
            distance = torch.linalg.norm(next_states[:, :3] - goal_t, dim=1)
            costs += cfg.running_goal_weight * distance.square()
            costs += cfg.control_weight * controls[:, step].square().sum(dim=1)
            costs += cfg.smooth_weight * (controls[:, step] - previous_action).square().sum(dim=1)
            costs -= cfg.progress_weight * (initial_distance - distance)
            clearance = _clearance(next_states[:, :3], self.spheres, self.cylinders)
            costs += cfg.soft_clearance_weight * torch.relu(
                cfg.soft_clearance_target - clearance).square()
            if cfg.z_bias_weight:
                exponent = (
                    (next_states[:, 2] - cfg.z_bias_plane) / cfg.z_bias_temperature
                ).clamp(max=20.0)
                costs += cfg.z_bias_weight * torch.exp(exponent)
            states = next_states
            previous_action = controls[:, step]
        costs += cfg.terminal_goal_weight * torch.linalg.norm(states[:, :3] - goal_t, dim=1).square()
        raw_costs = costs.clone()
        costs = torch.where(infeasible, torch.full_like(costs, float("inf")), costs)
        all_infeasible = bool(torch.isinf(costs).all())
        if all_infeasible:
            costs = -minimum_hp + 1e-3 * raw_costs
        weights = torch.softmax(-(costs - costs.min()) / max(cfg.temperature, 1e-6), dim=0)
        weights = torch.nan_to_num(weights, nan=0.0)
        if float(weights.sum()) < 1e-8:
            chosen = int(torch.argmin(costs))
            averaged_sequence = controls[chosen]
        else:
            weights /= weights.sum()
            averaged_sequence = torch.sum(weights[:, None, None] * controls, dim=0)
        action = torch.clamp(averaged_sequence[0], self.u_min, self.u_max)
        self._previous_sequence = averaged_sequence.detach()
        self._last_action = action.detach()

        next_state = self._step(torch.tensor(state_np, device=self.device).view(1, 6),
                                action.view(1, 3))[0, :3]
        center_hp = self._hp(center.view(1, 3), A, b, margins)[0]
        next_hp = self._hp(next_state.view(1, 3), A, b, margins)[0]
        info = {
            "A": poly.A.astype(np.float32),
            "b": poly.b.astype(np.float32),
            "center": poly.center.astype(np.float32),
            "feasible_fraction": float((~infeasible).float().mean().detach().cpu()),
            "all_infeasible": all_infeasible,
            "mixture_fraction": float(mix),
            "online_one_step_slack": float((next_hp - (1.0 - gamma) * center_hp).detach().cpu()),
            "plan_time_s": time.perf_counter() - started,
            "averaged_control_sequence": averaged_sequence.detach().cpu().numpy(),
        }
        return action.detach().cpu().numpy().astype(np.float32), info
