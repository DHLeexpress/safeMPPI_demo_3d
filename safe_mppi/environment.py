"""User-defined taskspace, static obstacles, and 3D double-integrator dynamics."""
from __future__ import annotations

import numpy as np
import torch

from .config import ExperimentConfig


class TaskEnvironment:
    def __init__(self, cfg: ExperimentConfig):
        self.task = cfg.taskspace
        self.mppi = cfg.safemppi
        self.bounds = self.task.bounds
        self.start = np.asarray(self.task.start, np.float32)
        self.goal = np.asarray(self.task.goal, np.float32)
        self.spheres = np.asarray(cfg.obstacles.spheres, np.float32).reshape(-1, 4)
        self.cylinders = np.asarray(cfg.obstacles.cylinders, np.float32).reshape(-1, 3)

    def step(self, state, control):
        state = np.asarray(state, np.float32)
        control = np.asarray(control, np.float32)
        dt = self.mppi.dt
        p = state[:3] + dt * state[3:6] + 0.5 * dt * dt * control
        v = state[3:6] + dt * control
        return np.concatenate([p, v]).astype(np.float32)

    def dense_positions(self, states, controls, substeps=10):
        states = np.asarray(states, float)
        controls = np.asarray(controls, float)
        out = [states[0, :3]]
        tau = np.linspace(self.mppi.dt / substeps, self.mppi.dt, substeps)
        for state, control in zip(states[:-1], controls):
            out.extend(state[:3][None] + tau[:, None] * state[3:6][None]
                       + 0.5 * tau[:, None] ** 2 * control[None])
        return np.asarray(out, np.float32)

    def obstacle_clearance(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        clearance = np.full(len(points), np.inf)
        if len(self.spheres):
            dist = np.linalg.norm(points[:, None] - self.spheres[None, :, :3], axis=2)
            clearance = np.minimum(clearance, (dist - self.spheres[None, :, 3]).min(axis=1))
        if len(self.cylinders):
            dist = np.linalg.norm(points[:, None, :2] - self.cylinders[None, :, :2], axis=2)
            clearance = np.minimum(clearance, (dist - self.cylinders[None, :, 2]).min(axis=1))
        return clearance

    def inside_taskspace(self, points, tolerance=1e-7):
        points = np.asarray(points, float).reshape(-1, 3)
        return np.all((points >= self.bounds[:, 0] - tolerance)
                      & (points <= self.bounds[:, 1] + tolerance), axis=1)

    def reached(self, point):
        return bool(np.linalg.norm(np.asarray(point, float)[:3] - self.goal) < self.task.reach_radius)

    def torch_obstacles(self, device, dtype=torch.float32):
        return (torch.as_tensor(self.spheres, device=device, dtype=dtype),
                torch.as_tensor(self.cylinders, device=device, dtype=dtype))
