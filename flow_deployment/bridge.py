"""Temporary, explicit frame bridge for offline deployment diagnostics.

This module intentionally lives outside ``deploy_sim``.  It maps the lab start
and goal exactly into the canonical ball-policy frame, conditions the policy on
the actual lab sphere expressed in that frame, and maps the first generated
acceleration back to the lab frame.

The bridge is not a sim-to-real certificate.  In particular, action magnitude
is matched by authority fraction rather than by similarity dynamics because
the research checkpoint was trained at 1 m/s^2 while the flight configuration
is capped at 0.3 m/s^2.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safe_mppi.ball_flow_task import load_policy


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_deploy_sim_lock(
    repo_root: str | Path,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail if any pinned Minhyuk deployment file differs from its lock."""
    repo_root = Path(repo_root).resolve()
    lock_path = (
        Path(lock_path)
        if lock_path is not None
        else Path(__file__).with_name("deploy_sim_lock.json")
    )
    lock = json.loads(lock_path.read_text())
    mismatches = []
    for relative, expected in lock["files"].items():
        path = repo_root / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({
                "path": relative,
                "expected": expected,
                "actual": actual,
            })
    if mismatches:
        raise RuntimeError(
            "deploy_sim lock mismatch; refusing to run external bridge: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "DEPLOY_SIM_LOCK_VERIFIED",
        "source_commit": lock["source_commit"],
        "files": lock["files"],
    }


@dataclass(frozen=True)
class EndpointSimilarity:
    """Right-handed, world-up-preserving similarity between two task frames."""

    source_start: np.ndarray
    source_goal: np.ndarray
    target_start: np.ndarray
    target_goal: np.ndarray
    scale: float
    rotation: np.ndarray

    @classmethod
    def from_endpoints(
        cls,
        source_start: np.ndarray,
        source_goal: np.ndarray,
        target_start: np.ndarray,
        target_goal: np.ndarray,
        world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> "EndpointSimilarity":
        source_start = np.asarray(source_start, float).reshape(3)
        source_goal = np.asarray(source_goal, float).reshape(3)
        target_start = np.asarray(target_start, float).reshape(3)
        target_goal = np.asarray(target_goal, float).reshape(3)
        source_delta = source_goal - source_start
        target_delta = target_goal - target_start
        source_length = float(np.linalg.norm(source_delta))
        target_length = float(np.linalg.norm(target_delta))
        if source_length <= 0.0 or target_length <= 0.0:
            raise ValueError("source and target endpoints must be distinct")

        world_up = np.asarray(world_up, float).reshape(3)

        def basis(delta):
            forward = delta / np.linalg.norm(delta)
            up = world_up - float(world_up @ forward) * forward
            up_norm = float(np.linalg.norm(up))
            if up_norm <= 1.0e-12:
                raise ValueError("world_up must not be parallel to either task path")
            up /= up_norm
            left = np.cross(up, forward)
            left /= np.linalg.norm(left)
            return np.column_stack([forward, left, up])

        rotation = basis(target_delta) @ basis(source_delta).T
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10):
            raise ValueError("failed to construct an orthonormal frame")
        if float(np.linalg.det(rotation)) <= 0.0:
            raise ValueError("frame mapping must remain right handed")
        return cls(
            source_start=source_start,
            source_goal=source_goal,
            target_start=target_start,
            target_goal=target_goal,
            scale=target_length / source_length,
            rotation=rotation,
        )

    def source_to_target_position(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, float)
        return self.target_start + self.scale * (
            (point - self.source_start) @ self.rotation.T
        )

    def target_to_source_position(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, float)
        return self.source_start + (
            (point - self.target_start) @ self.rotation
        ) / self.scale

    def source_to_target_direction(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, float) @ self.rotation.T

    def target_to_source_velocity(self, velocity: np.ndarray) -> np.ndarray:
        return (np.asarray(velocity, float) @ self.rotation) / self.scale

    def target_sphere_to_source(self, sphere: np.ndarray) -> np.ndarray:
        sphere = np.asarray(sphere, float).reshape(4)
        return np.concatenate([
            self.target_to_source_position(sphere[:3]),
            np.array([sphere[3] / self.scale]),
        ])

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "endpoint_exact_world_up_similarity",
            "source_start": self.source_start.tolist(),
            "source_goal": self.source_goal.tolist(),
            "target_start": self.target_start.tolist(),
            "target_goal": self.target_goal.tolist(),
            "scale_target_per_source": self.scale,
            "rotation_source_axes_in_target": self.rotation.tolist(),
            "det_rotation": float(np.linalg.det(self.rotation)),
        }


def _closest_sphere_boundary(position: np.ndarray, sphere: np.ndarray) -> np.ndarray:
    center, radius = sphere[:3], float(sphere[3])
    delta = np.asarray(position, float) - center
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        return center + np.array([0.0, 0.0, radius])
    return center + radius * delta / distance


def load_flow_policy(
    pretrain_dir: str | Path,
    expansion: str | Path | None = None,
    round_index: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the pretrained policy and optionally one expanded checkpoint."""
    pretrain_dir = Path(pretrain_dir)
    pretrained = pretrain_dir / "pretrained.pt"
    policy = load_policy(pretrained)
    provenance: dict[str, Any] = {
        "pretrained": str(pretrained.resolve()),
        "pretrained_sha256": sha256_file(pretrained),
        "expansion": None,
        "round": 0,
    }
    if expansion is not None:
        if round_index is None:
            raise ValueError("round_index is required with an expansion directory")
        checkpoint = Path(expansion) / f"checkpoint_{round_index:03d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("round", -1)) != int(round_index):
            raise ValueError("expanded checkpoint round does not match the request")
        policy.load_state_dict(payload["model"], strict=True)
        provenance.update({
            "expansion": str(checkpoint.resolve()),
            "expansion_sha256": sha256_file(checkpoint),
            "round": int(round_index),
        })
    policy.eval()
    return policy, provenance


class FlowDeploymentController:
    """Generate one raw plan and return its first authority-matched acceleration."""

    def __init__(
        self,
        policy: torch.nn.Module,
        frame: EndpointSimilarity,
        target_sphere: np.ndarray,
        target_action_limit: float,
        device: str | torch.device = "cpu",
    ):
        if tuple(policy.plan_shape) != (10, 3) or int(policy.context_dim) != 10:
            raise ValueError("deployment bridge requires the legacy10 H=10, 3-D policy")
        source_limit = getattr(policy, "control_limit", None)
        if source_limit is None or float(source_limit) <= 0.0:
            raise ValueError("policy must declare a positive control_limit")
        if target_action_limit <= 0.0:
            raise ValueError("target_action_limit must be positive")
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.frame = frame
        self.target_sphere = np.asarray(target_sphere, float).reshape(4)
        self.source_sphere = frame.target_sphere_to_source(self.target_sphere)
        self.target_action_limit = float(target_action_limit)
        self.source_action_limit = float(source_limit)
        self.authority_ratio = self.target_action_limit / self.source_action_limit
        self.trace: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.trace = []

    def plan(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        gamma: float,
        seed: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = np.asarray(state, float).reshape(6)
        goal = np.asarray(goal, float).reshape(3)
        source_position = self.frame.target_to_source_position(state[:3])
        source_goal = self.frame.target_to_source_position(goal)
        source_velocity = self.frame.target_to_source_velocity(state[3:])
        source_boundary = _closest_sphere_boundary(
            source_position, self.source_sphere,
        )
        context = np.concatenate([
            source_goal - source_position,
            source_velocity,
            source_boundary - source_position,
            np.array([float(gamma)]),
        ]).astype(np.float32)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        context_tensor = torch.from_numpy(context).to(self.device)
        with torch.no_grad():
            source_plan = self.policy.sample(
                context_tensor, 1, generator,
            )[0].detach().cpu().numpy()
        if not np.isfinite(source_plan).all():
            raise ValueError("flow policy generated a non-finite plan")

        # Temporary diagnostic mapping: preserve acceleration direction and the
        # fraction of available authority.  Geometric similarity acceleration
        # would exceed the Crazyflie configuration by more than fourfold.
        target_plan = self.authority_ratio * self.frame.source_to_target_direction(
            source_plan,
        )
        peak = np.max(np.abs(target_plan), axis=1, keepdims=True)
        saturation_scale = np.minimum(
            1.0,
            self.target_action_limit / np.maximum(peak, 1.0e-12),
        )
        target_plan = (target_plan * saturation_scale).astype(np.float32)
        action = target_plan[0].copy()
        record = {
            "seed": int(seed),
            "gamma": float(gamma),
            "target_state": state.astype(np.float32),
            "source_state": np.concatenate(
                [source_position, source_velocity],
            ).astype(np.float32),
            "context": context,
            "source_plan": source_plan.astype(np.float32),
            "target_plan": target_plan,
            "saturation_scale": saturation_scale.astype(np.float32),
            "action": action,
        }
        self.trace.append(record)
        info = {
            "bridge": "temporary_endpoint_similarity",
            "source_state": record["source_state"],
            "source_plan": record["source_plan"],
            "target_plan": record["target_plan"],
        }
        return action, info

    def contract(self) -> dict[str, Any]:
        return {
            "policy_contract": "legacy10_raw_temperature1",
            "plan_shape": list(self.policy.plan_shape),
            "source_action_limit": self.source_action_limit,
            "target_action_limit": self.target_action_limit,
            "action_mapping": "authority_matched_rotation_then_uniform_axis_limit",
            "authority_ratio": self.authority_ratio,
            "target_sphere": self.target_sphere.tolist(),
            "target_sphere_in_source_frame": self.source_sphere.tolist(),
            "velocity_semantics": "uses the second half of the state supplied by deploy_sim",
            "dynamic_similarity": (
                "not exact: endpoint spatial scaling, unchanged 0.1 s replan "
                "period, and authority-ratio acceleration scaling cannot all "
                "describe one dynamically similar double integrator"
            ),
            "warning": (
                "Temporary OOD offline diagnostic only; not a flight safety "
                "certificate or a dynamically exact similarity."
            ),
        }
