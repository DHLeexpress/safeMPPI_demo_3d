"""Typed JSON configuration and strict validation for the minimal package."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    origin: tuple[float, float, float]
    size: tuple[float, float, float]
    start: tuple[float, float, float, float, float, float]
    goal: tuple[float, float, float]
    reach_radius: float
    max_steps: int

    @property
    def bounds(self) -> np.ndarray:
        origin = np.asarray(self.origin, float)
        return np.stack([origin, origin + np.asarray(self.size, float)], axis=1)


@dataclass(frozen=True)
class ObstacleConfig:
    spheres: tuple[tuple[float, float, float, float], ...]
    cylinders: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class MPPIConfig:
    sampling_mode: str
    horizon: int
    dt: float
    num_samples: int
    temperature: float
    noise_sigma: tuple[float, float, float]
    demo_u_max: float
    platform_u_max: float
    sensing_range: float
    polytope_faces: int
    centroid_gain: float
    centroid_smooth: float
    centroid_eps: float
    sigma_aniso: float
    urgency_floor: float
    warm_start: bool
    running_goal_weight: float
    terminal_goal_weight: float
    control_weight: float
    smooth_weight: float
    soft_clearance_weight: float
    soft_clearance_target: float
    progress_weight: float
    robot_radius: float
    obstacle_margin: float
    safety_margin: float
    predict_gain: float
    zoh_guard: float
    z_bias_weight: float = 0.0
    z_bias_temperature: float = 0.05
    z_bias_ref: float = 0.0
    warm_start_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DataConfig:
    gammas: tuple[float, ...]
    episodes_per_gamma: int
    seed_start: int


@dataclass(frozen=True)
class ExperimentConfig:
    taskspace: TaskConfig
    obstacles: ObstacleConfig
    safemppi: MPPIConfig
    data: DataConfig
    raw: dict[str, Any]


def _tuples(rows, width, name):
    out = tuple(tuple(float(x) for x in row) for row in rows)
    if any(len(row) != width for row in out):
        raise ValueError(f"{name} rows must have {width} numbers")
    return out


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)
    t, o, m, d = (raw[key] for key in ("taskspace", "obstacles", "safemppi", "data"))
    task = TaskConfig(
        origin=tuple(map(float, t["origin"])), size=tuple(map(float, t["size"])),
        start=tuple(map(float, t["start"])), goal=tuple(map(float, t["goal"])),
        reach_radius=float(t["reach_radius"]), max_steps=int(t["max_steps"]))
    obstacles = ObstacleConfig(
        spheres=_tuples(o.get("spheres", []), 4, "spheres"),
        cylinders=_tuples(o.get("cylinders", []), 3, "cylinders"))
    mppi_raw = {**m, "noise_sigma": tuple(map(float, m["noise_sigma"]))}
    if "warm_start_bias" in mppi_raw:
        mppi_raw["warm_start_bias"] = tuple(map(float, mppi_raw["warm_start_bias"]))
    mppi = MPPIConfig(**mppi_raw)
    data = DataConfig(gammas=tuple(map(float, d["gammas"])),
                      episodes_per_gamma=int(d["episodes_per_gamma"]),
                      seed_start=int(d["seed_start"]))
    cfg = ExperimentConfig(task, obstacles, mppi, data, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    t, m, d = cfg.taskspace, cfg.safemppi, cfg.data
    if len(t.origin) != 3 or len(t.size) != 3 or len(t.start) != 6 or len(t.goal) != 3:
        raise ValueError("origin/size/goal must be 3D and start must be [position, velocity] in R6")
    if min(t.size) <= 0 or t.reach_radius <= 0 or t.max_steps < 1:
        raise ValueError("task size, reach radius, and max steps must be positive")
    bounds = t.bounds
    for name, point in (("start", t.start[:3]), ("goal", t.goal)):
        p = np.asarray(point, float)
        if np.any(p < bounds[:, 0]) or np.any(p > bounds[:, 1]):
            raise ValueError(f"{name} position {p.tolist()} lies outside taskspace {bounds.tolist()}")
    if m.sampling_mode != "mode1_centroid_anisotropic":
        raise ValueError("this package intentionally contains only mode1_centroid_anisotropic")
    if m.horizon != 10 or m.polytope_faces != 80:
        raise ValueError("the current standalone spec fixes H=10 and 80 triangular nominal faces")
    if len(m.noise_sigma) != 3 or min(m.noise_sigma) <= 0:
        raise ValueError("noise_sigma must contain three positive values")
    if min(m.robot_radius, m.obstacle_margin, m.safety_margin, m.predict_gain, m.zoh_guard) != 0.0 \
            or max(m.robot_radius, m.obstacle_margin, m.safety_margin, m.predict_gain, m.zoh_guard) != 0.0:
        raise ValueError("robot radius and every hard obstacle/gain/guard margin must remain exactly 0")
    if not (0.0 < m.demo_u_max <= m.platform_u_max) or m.num_samples < 1 or m.dt <= 0:
        raise ValueError("require 0 < demo_u_max <= platform_u_max, samples >=1, and dt >0")
    if m.z_bias_weight < 0.0 or (m.z_bias_weight > 0.0 and m.z_bias_temperature <= 0.0):
        raise ValueError("z_bias_weight must be >=0 and use a positive z_bias_temperature when active")
    if len(m.warm_start_bias) != 3 or max(abs(v) for v in m.warm_start_bias) > m.demo_u_max:
        raise ValueError("warm_start_bias must be a 3-vector within the demonstration control cap")
    if not d.gammas or any(not 0.0 < g <= 1.0 for g in d.gammas):
        raise ValueError("all gamma values must be in (0,1]")
    if d.episodes_per_gamma < 1:
        raise ValueError("episodes_per_gamma must be positive")
