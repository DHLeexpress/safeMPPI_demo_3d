"""As-built bowling geometry and hard flight-reference constraints.

This module is deliberately separate from the randomized clutter scene law.
The measured balls have different radii and two inflated shells overlap, so
the randomized-law separation validator is not applicable to this fixed scene.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_clutter_expansion import scene_sha256
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)


DEFAULT_EFFECTIVE_MARGIN_M = 0.16
DEFAULT_STRING_RADIUS_M = 0.10


def load_as_built_geometry(
    scene_json: str | Path,
    *,
    effective_margin_m: float = DEFAULT_EFFECTIVE_MARGIN_M,
    string_radius_m: float = DEFAULT_STRING_RADIUS_M,
) -> dict:
    """Load measured centers/radii and derive the requested hard geometry."""
    path = Path(scene_json)
    payload = json.loads(path.read_text())
    balls = payload["as_built_measured"]["balls"]
    physical = np.asarray([
        [row["x"], row["y"], row["z"], row["measured_radius_m"]]
        for row in balls
    ], np.float32)
    effective = physical.copy()
    effective[:, 3] += np.float32(effective_margin_m)
    return {
        "scene_json": str(path),
        "physical_spheres": physical,
        "effective_spheres": effective,
        "effective_margin_m": float(effective_margin_m),
        "string_radius_m": float(string_radius_m),
        "string_start_z_m": physical[:, 2] + physical[:, 3],
        "measurement": payload["as_built_measured"],
    }


def string_clearance_m(
    points: np.ndarray,
    physical_spheres: np.ndarray,
    string_radius_m: float = DEFAULT_STRING_RADIUS_M,
) -> np.ndarray:
    """Clearance to each ceiling-reaching string above its physical ball."""
    points = np.asarray(points, float).reshape(-1, 3)
    physical = np.asarray(physical_spheres, float).reshape(-1, 4)
    radial = np.linalg.norm(
        points[:, None, :2] - physical[None, :, :2], axis=2,
    ) - float(string_radius_m)
    starts = physical[:, 2] + physical[:, 3]
    radial[points[:, None, 2] < starts[None, :]] = np.inf
    return radial.min(axis=1)


def hard_path_diagnostics(
    points: np.ndarray,
    effective_spheres: np.ndarray,
    physical_spheres: np.ndarray,
    string_radius_m: float = DEFAULT_STRING_RADIUS_M,
) -> dict:
    """Return exact hard-gate diagnostics for one dense flight path."""
    points = np.asarray(points, float).reshape(-1, 3)
    effective = np.asarray(effective_spheres, float).reshape(-1, 4)
    sphere_clearance = (
        np.linalg.norm(
            points[:, None, :3] - effective[None, :, :3], axis=2,
        ) - effective[None, :, 3]
    ).min(axis=1)
    string_clearance = string_clearance_m(
        points, physical_spheres, string_radius_m,
    )
    return {
        "effective_sphere_min_clearance_m": float(sphere_clearance.min()),
        "string_min_clearance_m": float(string_clearance.min()),
        "effective_sphere_valid": bool(np.all(sphere_clearance > 0.0)),
        "string_valid": bool(np.all(string_clearance > 0.0)),
        "hard_valid": bool(
            np.all(sphere_clearance > 0.0)
            and np.all(string_clearance > 0.0)
        ),
    }


class FixedMeasuredSphereScene:
    """Fixed six-sphere packer that permits measured per-sphere radii."""

    def __init__(self, effective_spheres: np.ndarray):
        values = np.asarray(effective_spheres, np.float32).reshape(-1, 4)
        if values.shape != (6, 4):
            raise ValueError("as-built bowling scene must contain six spheres")
        if not np.isfinite(values).all() or bool((values[:, 3] <= 0.0).any()):
            raise ValueError("as-built sphere rows must be finite and positive")
        self._spheres = values.copy()

    @property
    def max_count(self) -> int:
        return 6

    @property
    def packed_dim(self) -> int:
        return 25

    @property
    def scene_schema(self) -> str:
        return "bowling_as_built_variable_radius_v1"

    @property
    def radius(self) -> float:
        return float(self._spheres[:, 3].max())

    @property
    def domain_seed(self) -> int:
        return 0

    def validate(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        spheres = np.asarray(values, np.float32).reshape(-1, 4)
        if spheres.shape != (6, 4):
            raise ValueError("as-built scene requires exactly six sphere rows")
        if not np.allclose(spheres, self._spheres, rtol=0.0, atol=2.0e-6):
            raise ValueError("sphere rows differ from the frozen as-built scene")
        if not np.isfinite(spheres).all() or bool((spheres[:, 3] <= 0.0).any()):
            raise ValueError("invalid as-built sphere rows")
        return spheres.copy()

    def sample(self, env: TaskEnvironment, seed: int) -> np.ndarray:
        del seed
        return self.validate(env, self._spheres)

    def pack(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        spheres = self.validate(env, values)
        return np.concatenate([
            np.asarray([6.0], np.float32), spheres.reshape(-1),
        ])

    def unpack(self, env: TaskEnvironment, values: np.ndarray) -> np.ndarray:
        packed = np.asarray(values, np.float32).reshape(-1)
        if packed.shape != (25,) or not np.isclose(packed[0], 6.0):
            raise ValueError("invalid packed as-built sphere context")
        return self.validate(env, packed[1:].reshape(6, 4))


class RealBowlingTask(LabClutterPre2MultiPairExpansionTask):
    """Fixed measured bowling task with a hard vertical-string collision gate."""

    def __init__(
        self,
        config,
        *,
        physical_spheres: np.ndarray,
        effective_spheres: np.ndarray,
        string_radius_m: float = DEFAULT_STRING_RADIUS_M,
        **kwargs,
    ):
        self.physical_spheres = np.asarray(
            physical_spheres, np.float32,
        ).reshape(6, 4)
        self.effective_spheres = np.asarray(
            effective_spheres, np.float32,
        ).reshape(6, 4)
        self.string_radius_m = float(string_radius_m)
        kwargs["scene_spec"] = FixedMeasuredSphereScene(self.effective_spheres)
        super().__init__(config, **kwargs)

    def _environment(self, spheres: np.ndarray) -> TaskEnvironment:
        spheres = self.scene_spec.validate(self.env, spheres)
        obstacles = replace(
            self.config.obstacles,
            spheres=tuple(tuple(map(float, row)) for row in spheres),
            cylinders=(),
        )
        return TaskEnvironment(replace(self.config, obstacles=obstacles))

    def new_state(self, seed: int) -> dict:
        env = self._environment(self.effective_spheres)
        return {
            "x": env.start.copy(),
            "previous_applied": np.zeros(3, np.float32),
            "previous_raw": np.zeros(3, np.float32),
            "spheres": self.effective_spheres.copy(),
            "scene_seed": int(seed),
            "scene_hash": scene_sha256(env, self.effective_spheres),
            "steps": 0,
            "collided": False,
            "string_collision": False,
            "oob": False,
        }

    def advance(self, state, candidate):
        _, _, dense = self._rollout_plan(
            np.asarray(state["x"], np.float32),
            np.asarray(state["previous_applied"], np.float32),
            candidate.detach().cpu().numpy().reshape(-1, 3)[:1],
        )
        updated = super().advance(state, candidate)
        string_clearance = string_clearance_m(
            dense.reshape(-1, 3),
            self.physical_spheres,
            self.string_radius_m,
        )
        string_collision = bool(np.any(string_clearance <= 0.0))
        updated["string_collision"] = bool(
            state.get("string_collision", False) or string_collision
        )
        updated["collided"] = bool(updated["collided"] or string_collision)
        return updated

