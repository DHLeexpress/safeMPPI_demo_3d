"""Load the frozen lab policy for online or frozen-reference deployment."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from safe_mppi.environment import TaskEnvironment
from safe_mppi.flow_model import ConditionalFlowMLP
from safe_mppi.lab_reference_flow_task import LabReferenceFlowController


DEFAULT_CHECKPOINT_SHA256 = (
    "cdc27062cdf60e3faf54ed5a7e6dd23e0c45e8208343ff00b9455a799034d059"
)
DEFAULT_CONFIG_SHA256 = (
    "aaa686ff5edadd3a6c29a08ced66625d3e9a1c24afbde548c44a721fe0cb519a"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lab_reference_policy(path: str | Path):
    """Load a completed raw10 or qualified visual lab checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = dict(payload["arch"])
    kind = arch.pop("kind", "conditional_flow_mlp")
    arch["plan_shape"] = tuple(arch["plan_shape"])
    if kind == "conditional_flow_mlp":
        policy = ConditionalFlowMLP(**arch)
    else:
        from safe_mppi.lab_visual_flow import (
            LAB_VISUAL_SCHEMA,
            LabVisualFlowPolicy,
        )
        if kind != LAB_VISUAL_SCHEMA:
            raise ValueError(f"unknown lab policy architecture {kind!r}")
        policy = LabVisualFlowPolicy(**arch)
    policy.load_state_dict(payload["model"], strict=True)
    return policy


def load_lab_deployment_controller(
    checkpoint: str | Path,
    env: TaskEnvironment,
    *,
    sampling_temperature: float = 1.0,
    device: str = "cpu",
    expected_sha256: str | None = None,
) -> tuple[LabReferenceFlowController, dict]:
    """Return the state-feedback controller and an authenticated contract."""
    checkpoint = Path(checkpoint).resolve()
    actual_sha256 = sha256_file(checkpoint)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    policy = load_lab_reference_policy(checkpoint)
    if not float(policy.control_limit) == float(env.mppi.demo_u_max):
        raise ValueError(
            "policy/config action-limit mismatch: "
            f"{policy.control_limit} versus {env.mppi.demo_u_max}"
        )
    controller = LabReferenceFlowController(
        policy,
        env,
        sampling_temperature=sampling_temperature,
        device=device,
    )
    return controller, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha256,
        "context_schema": getattr(policy, "context_schema", "lab_raw10_v1"),
        "context_dim": int(policy.context_dim),
        "plan_shape": list(policy.plan_shape),
        "control_limit": float(policy.control_limit),
        "nfe": int(policy.nfe),
        "sampling_temperature": float(sampling_temperature),
        "sampling_temperature_definition": "x0 ~ N(0, tau^2 I)",
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "tracking_and_governor": "external",
    }
