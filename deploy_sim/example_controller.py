"""Minimal example controller — a template for plugging in your own.

This is deliberately the simplest thing that satisfies the protocol: a PD law to
the goal with a repulsive term around obstacles. It is NOT a good controller; it
exists so you can see the required interface and confirm the harness picks up an
external controller.

    python deploy_sim/run_offline.py --controller example_controller:PDController

Copy this file, keep the two methods, and put your own algorithm in `plan`.
"""
from __future__ import annotations

import numpy as np


class PDController:
    """Constructed exactly like the bundled controller: (mppi_cfg, env, device=...)."""

    def __init__(self, cfg, env, device="cpu"):
        self.cfg = cfg          # safemppi block: dt, demo_u_max, horizon, ...
        self.env = env          # TaskEnvironment: .spheres, .cylinders, .bounds, .goal
        self.kp = 1.2
        self.kd = 2.2
        self.repulse = 1.2      # strength of the obstacle push-off
        self.reset()

    def reset(self):
        """Called once before a run; clear any internal state here."""
        self._last_u = np.zeros(3)

    def plan(self, state, goal, gamma, seed=0):
        """state=[x,y,z,vx,vy,vz] -> (acceleration (3,), info dict).

        `gamma` is the barrier-contraction parameter used by the bundled
        SafeMPPI; ignore it if your method has no equivalent.
        """
        p, v = np.asarray(state, float)[:3], np.asarray(state, float)[3:6]
        u = self.kp * (np.asarray(goal, float) - p) - self.kd * v

        # Push away from each sphere when inside its influence radius.
        for sx, sy, sz, sr in np.asarray(self.env.spheres, float).reshape(-1, 4):
            d = p - np.array([sx, sy, sz])
            dist = float(np.linalg.norm(d))
            reach = sr + 0.35
            if 1e-6 < dist < reach:
                u = u + self.repulse * (reach - dist) / reach * (d / dist)

        u_max = float(self.cfg.demo_u_max)
        n = float(np.linalg.norm(u))
        if n > u_max:
            u = u * (u_max / n)
        self._last_u = u
        return u, {}
