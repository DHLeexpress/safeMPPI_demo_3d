"""Lab reference flow with a compact robot-centered 3-D safety encoder."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .environment import TaskEnvironment
from .flow_model import ConditionalFlowMLP
from .geometry import build_nominal_polytope


LAB_VISUAL_GRID_SHAPE = (3, 16, 12, 12)
LAB_VISUAL_LOW_DIM = 7
LAB_VISUAL_PACKED_DIM = LAB_VISUAL_LOW_DIM + math.prod(LAB_VISUAL_GRID_SHAPE)
LAB_VISUAL_SCHEMA = "lab_spherical_hp3d_v1"
LAB_VISUAL_HISTORY_LENGTH = 10
LAB_VISUAL_HISTORY_STEP_DIM = 4
LAB_VISUAL_HISTORY_PACKED_DIM = (
    LAB_VISUAL_PACKED_DIM
    + LAB_VISUAL_HISTORY_LENGTH * LAB_VISUAL_HISTORY_STEP_DIM
)
LAB_VISUAL_HISTORY_SCHEMA = "lab_spherical_hp3d_gru_v1"
LAB_VISUAL_CHANNELS = (
    "occupancy",
    "nominal_polytope_mask",
    "clipped_hp",
)
LAB_VISUAL_FRAME = "robot_centered_world_spherical_equal_area"


class _AzimuthCircularPad(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.cat([
            values[:, :, -1:],
            values,
            values[:, :, :1],
        ], dim=2)
        return F.pad(values, (1, 1, 1, 1, 0, 0))


def spherical_grid_points(
    position: np.ndarray,
    sensing_range: float,
    *,
    n_azimuth: int = 16,
    n_elevation: int = 12,
    n_radius: int = 12,
) -> np.ndarray:
    """Return the world-frame center of every spherical grid cell."""
    position = np.asarray(position, float).reshape(3)
    azimuth = (
        -np.pi
        + (np.arange(n_azimuth, dtype=float) + 0.5)
        * (2.0 * np.pi / n_azimuth)
    )
    elevation = np.arcsin(
        -1.0
        + (np.arange(n_elevation, dtype=float) + 0.5)
        * (2.0 / n_elevation)
    )
    radius = (
        (np.arange(n_radius, dtype=float) + 0.5)
        * (float(sensing_range) / n_radius)
    )
    cos_elevation = np.cos(elevation)[None, :, None]
    directions = np.concatenate([
        cos_elevation * np.cos(azimuth)[:, None, None],
        cos_elevation * np.sin(azimuth)[:, None, None],
        np.broadcast_to(
            np.sin(elevation)[None, :, None],
            (n_azimuth, n_elevation, 1),
        ),
    ], axis=2)
    return (
        position[None, None, None, :]
        + directions[:, :, None, :] * radius[None, None, :, None]
    )


def spherical_safety_grid(
    env: TaskEnvironment,
    position: np.ndarray,
    *,
    n_azimuth: int = 16,
    n_elevation: int = 12,
    n_radius: int = 12,
) -> np.ndarray:
    """Return occupancy, nominal-polytope mask, and clipped H_P.

    The lattice is robot-centered but world-axis aligned. This preserves the
    existing global control convention: azimuth is measured about world z,
    elevation from the world x-y plane, and radius spans the nominal sensing
    range.
    """
    position = np.asarray(position, float).reshape(3)
    points = spherical_grid_points(
        position,
        env.mppi.sensing_range,
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        n_radius=n_radius,
    )
    flat_points = points.reshape(-1, 3)

    clearance = env.obstacle_clearance(flat_points)
    occupancy = (clearance < 0.0).astype(np.float32)

    polytope = build_nominal_polytope(
        position,
        env.spheres,
        env.cylinders,
        env.bounds,
        sensing_range=env.mppi.sensing_range,
        obstacle_margin=0.0,
    )
    margins = np.maximum(polytope.margins, 1.0e-3)
    hp = (
        polytope.b[None] - flat_points @ polytope.A.T
    ) / margins[None]
    hp = hp.min(axis=1)
    mask = (hp >= 0.0).astype(np.float32)
    clipped_hp = np.clip(hp, -1.0, 1.0).astype(np.float32)
    return np.stack([
        occupancy,
        mask,
        clipped_hp,
    ]).reshape(3, n_azimuth, n_elevation, n_radius)


def build_visual_context(
    env: TaskEnvironment,
    state6: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Pack [goal-position, velocity, gamma] and the 3-D safety volume."""
    state6 = np.asarray(state6, np.float32).reshape(6)
    low = np.concatenate([
        env.goal - state6[:3],
        state6[3:6],
        np.asarray([gamma], np.float32),
    ]).astype(np.float32)
    grid = spherical_safety_grid(env, state6[:3]).astype(np.float32)
    return np.concatenate([low, grid.reshape(-1)]).astype(np.float32)


def build_visual_history_context(
    env: TaskEnvironment,
    state6: np.ndarray,
    gamma: float,
    raw_history: np.ndarray,
) -> np.ndarray:
    """Append ten past raw commands and their left-padding validity bits."""
    history = np.asarray(raw_history, np.float32)
    if history.shape != (
        LAB_VISUAL_HISTORY_LENGTH,
        LAB_VISUAL_HISTORY_STEP_DIM,
    ):
        raise ValueError(
            "raw_history must have shape "
            f"({LAB_VISUAL_HISTORY_LENGTH},{LAB_VISUAL_HISTORY_STEP_DIM})"
        )
    if not np.isfinite(history).all():
        raise ValueError("raw_history must be finite")
    if not np.isin(history[:, 3], (0.0, 1.0)).all():
        raise ValueError("raw_history validity bits must be binary")
    if bool((history[history[:, 3] == 0.0, :3] != 0.0).any()):
        raise ValueError("padded raw-history actions must be exactly zero")
    return np.concatenate([
        build_visual_context(env, state6, gamma),
        history.reshape(-1),
    ]).astype(np.float32)


class LabVisualFlowPolicy(nn.Module):
    """CFM policy with an isolated 3-D visual conditioning encoder."""

    context_schema = LAB_VISUAL_SCHEMA
    context_dim = LAB_VISUAL_PACKED_DIM

    def __init__(
        self,
        plan_shape: tuple[int, ...] = (10, 3),
        hidden: int = 48,
        representation_dim: int = 32,
        grid_token_dim: int = 32,
        control_limit: float | None = None,
        nfe: int = 16,
        trunk_depth: int = 2,
        time_features: str = "raw1",
        grid_shape: tuple[int, ...] = LAB_VISUAL_GRID_SHAPE,
        grid_channels: tuple[str, ...] = LAB_VISUAL_CHANNELS,
        grid_frame: str = LAB_VISUAL_FRAME,
    ):
        super().__init__()
        if tuple(plan_shape) != (10, 3):
            raise ValueError("lab visual flow requires plan_shape=(10,3)")
        if trunk_depth != 2 or time_features != "raw1":
            raise ValueError(
                "lab visual flow fixes trunk_depth=2 and time_features='raw1'"
            )
        if tuple(grid_shape) != LAB_VISUAL_GRID_SHAPE:
            raise ValueError("lab visual flow grid shape contract changed")
        if tuple(grid_channels) != LAB_VISUAL_CHANNELS:
            raise ValueError("lab visual flow channel contract changed")
        if grid_frame != LAB_VISUAL_FRAME:
            raise ValueError("lab visual flow frame contract changed")
        self.plan_shape = tuple(plan_shape)
        self.plan_dim = math.prod(self.plan_shape)
        self.control_limit = control_limit
        self.nfe = int(nfe)
        self.grid_token_dim = int(grid_token_dim)
        self.grid_encoder = nn.Sequential(
            _AzimuthCircularPad(),
            nn.Conv3d(3, 8, kernel_size=3),
            nn.SiLU(),
            _AzimuthCircularPad(),
            nn.Conv3d(8, 16, kernel_size=3),
            nn.SiLU(),
            nn.AvgPool3d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(16 * 4 * 3 * 3, self.grid_token_dim),
            nn.SiLU(),
        )
        self.flow = ConditionalFlowMLP(
            context_dim=LAB_VISUAL_LOW_DIM + self.grid_token_dim,
            plan_shape=self.plan_shape,
            hidden=hidden,
            representation_dim=representation_dim,
            control_limit=control_limit,
            nfe=nfe,
            trunk_depth=trunk_depth,
            time_features=time_features,
        )

    def expansion_parameter_groups(
        self,
        base_lr: float,
        first_layer_lr_scale: float = 1.0,
    ) -> list[dict[str, object]]:
        if base_lr <= 0.0 or not 0.0 < first_layer_lr_scale <= 1.0:
            raise ValueError(
                "require base_lr>0 and first_layer_lr_scale in (0,1]"
            )
        slow = (
            list(self.grid_encoder.parameters())
            + list(self.flow.trunk[0].parameters())
        )
        slow_ids = {id(parameter) for parameter in slow}
        remaining = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in slow_ids
        ]
        return [
            {"params": slow, "lr": base_lr * first_layer_lr_scale},
            {"params": remaining, "lr": base_lr},
        ]

    @property
    def trunk(self):
        return self.flow.trunk

    @property
    def head(self):
        return self.flow.head

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        single = context.ndim == 1
        context = context.reshape(-1, self.context_dim).float()
        low = context[:, :LAB_VISUAL_LOW_DIM]
        grid = context[:, LAB_VISUAL_LOW_DIM:].reshape(
            -1, *LAB_VISUAL_GRID_SHAPE
        )
        encoded = torch.cat([low, self.grid_encoder(grid)], dim=1)
        return encoded[0] if single else encoded

    def cfm_loss(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
        reduction: str = "none",
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.flow.cfm_loss(
            self.encode_context(contexts),
            candidates,
            reduction=reduction,
            loss_mask=loss_mask,
        )

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        count: int,
        generator: torch.Generator,
        base_std: float = 1.0,
    ) -> torch.Tensor:
        return self.flow.sample(
            self.encode_context(context),
            count,
            generator,
            base_std=base_std,
        )

    @torch.no_grad()
    def sample_with_base(
        self,
        context: torch.Tensor,
        count: int,
        generator: torch.Generator,
        base_std: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.flow.sample_with_base(
            self.encode_context(context),
            count,
            generator,
            base_std=base_std,
        )

    @torch.no_grad()
    def embed(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        flow_time: float = 0.9,
        base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.flow.embed(
            self.encode_context(context),
            candidates,
            flow_time=flow_time,
            base=base,
        )


class LabVisualHistoryFlowPolicy(LabVisualFlowPolicy):
    """Visual CFM policy with a compact GRU over past raw commands."""

    context_schema = LAB_VISUAL_HISTORY_SCHEMA
    context_dim = LAB_VISUAL_HISTORY_PACKED_DIM

    def __init__(
        self,
        plan_shape: tuple[int, ...] = (10, 3),
        hidden: int = 48,
        representation_dim: int = 32,
        grid_token_dim: int = 32,
        history_token_dim: int = 16,
        history_length: int = LAB_VISUAL_HISTORY_LENGTH,
        control_limit: float | None = None,
        nfe: int = 16,
        trunk_depth: int = 2,
        time_features: str = "raw1",
        grid_shape: tuple[int, ...] = LAB_VISUAL_GRID_SHAPE,
        grid_channels: tuple[str, ...] = LAB_VISUAL_CHANNELS,
        grid_frame: str = LAB_VISUAL_FRAME,
    ):
        if int(history_length) != LAB_VISUAL_HISTORY_LENGTH:
            raise ValueError(
                f"lab visual history length must be {LAB_VISUAL_HISTORY_LENGTH}"
            )
        super().__init__(
            plan_shape=plan_shape,
            hidden=hidden,
            representation_dim=representation_dim,
            grid_token_dim=grid_token_dim,
            control_limit=control_limit,
            nfe=nfe,
            trunk_depth=trunk_depth,
            time_features=time_features,
            grid_shape=grid_shape,
            grid_channels=grid_channels,
            grid_frame=grid_frame,
        )
        self.history_length = int(history_length)
        self.history_token_dim = int(history_token_dim)
        self.history_encoder = nn.GRU(
            input_size=LAB_VISUAL_HISTORY_STEP_DIM,
            hidden_size=self.history_token_dim,
            batch_first=True,
        )
        self.flow = ConditionalFlowMLP(
            context_dim=(
                LAB_VISUAL_LOW_DIM
                + self.grid_token_dim
                + self.history_token_dim
            ),
            plan_shape=self.plan_shape,
            hidden=hidden,
            representation_dim=representation_dim,
            control_limit=control_limit,
            nfe=nfe,
            trunk_depth=trunk_depth,
            time_features=time_features,
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        single = context.ndim == 1
        context = context.reshape(-1, self.context_dim).float()
        low = context[:, :LAB_VISUAL_LOW_DIM]
        grid_stop = LAB_VISUAL_PACKED_DIM
        grid = context[:, LAB_VISUAL_LOW_DIM:grid_stop].reshape(
            -1, *LAB_VISUAL_GRID_SHAPE
        )
        history = context[:, grid_stop:].reshape(
            -1,
            LAB_VISUAL_HISTORY_LENGTH,
            LAB_VISUAL_HISTORY_STEP_DIM,
        )
        _, hidden = self.history_encoder(history)
        encoded = torch.cat([
            low,
            self.grid_encoder(grid),
            hidden[-1],
        ], dim=1)
        return encoded[0] if single else encoded

    def expansion_parameter_groups(
        self,
        base_lr: float,
        first_layer_lr_scale: float = 1.0,
    ) -> list[dict[str, object]]:
        """Freeze the action-history encoder during online expansion."""
        if base_lr <= 0.0 or not 0.0 < first_layer_lr_scale <= 1.0:
            raise ValueError(
                "require base_lr>0 and first_layer_lr_scale in (0,1]"
            )
        for parameter in self.history_encoder.parameters():
            parameter.requires_grad_(False)
        slow = (
            list(self.grid_encoder.parameters())
            + list(self.flow.trunk[0].parameters())
        )
        slow_ids = {id(parameter) for parameter in slow}
        remaining = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in slow_ids
        ]
        return [
            {"params": slow, "lr": base_lr * first_layer_lr_scale},
            {"params": remaining, "lr": base_lr},
        ]


def load_lab_reference_policy(path: str | Path):
    """Load either the legacy raw10 lab policy or the visual lab policy."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = dict(payload["arch"])
    kind = arch.pop("kind", "conditional_flow_mlp")
    arch["plan_shape"] = tuple(arch["plan_shape"])
    if kind == LAB_VISUAL_SCHEMA:
        required = {"grid_shape", "grid_channels", "grid_frame"}
        missing = required.difference(arch)
        if missing:
            raise ValueError(
                f"visual checkpoint is missing semantic fields {sorted(missing)}"
            )
        policy = LabVisualFlowPolicy(**arch)
    elif kind == LAB_VISUAL_HISTORY_SCHEMA:
        required = {
            "grid_shape",
            "grid_channels",
            "grid_frame",
            "history_length",
            "history_token_dim",
        }
        missing = required.difference(arch)
        if missing:
            raise ValueError(
                "visual-history checkpoint is missing semantic fields "
                f"{sorted(missing)}"
            )
        policy = LabVisualHistoryFlowPolicy(**arch)
    elif kind == "conditional_flow_mlp":
        policy = ConditionalFlowMLP(**arch)
    else:
        raise ValueError(f"unknown lab policy architecture {kind!r}")
    policy.load_state_dict(payload["model"], strict=True)
    return policy
