"""Focused physical and architecture QA for the nonuniform radial encoder."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_visual_flow import (
    LAB_RADIAL_VISUAL_ENCODER_CHANNELS,
    LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE,
    LAB_RADIAL_VISUAL_GRID_SHAPE,
    LAB_RADIAL_VISUAL_RADIAL_EDGES,
    LabNonuniformRadialEncoder,
    LabNonuniformRadialFlowPolicy,
    LabNonuniformRadialHistoryFlowPolicy,
    _SphericalTopologyPad3d,
    nonuniform_radial_grid_points,
    nonuniform_radial_safety_grid,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
    / "resolved_config.json"
)
EXPECTED_EDGES = (
    0.0,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
    1.50,
    2.00,
)


def _symmetric_config(*, spheres=(), cylinders=()):
    config = load_config(CONFIG)
    return replace(
        config,
        taskspace=replace(
            config.taskspace,
            origin=(-3.0, -3.0, -3.0),
            size=(6.0, 6.0, 6.0),
            start=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal=(2.0, 0.0, 0.0),
        ),
        obstacles=replace(
            config.obstacles,
            spheres=tuple(spheres),
            cylinders=tuple(cylinders),
        ),
    )


def test_radial_edges_and_coordinate_channels_are_exact():
    assert LAB_RADIAL_VISUAL_RADIAL_EDGES == EXPECTED_EDGES
    assert LAB_RADIAL_VISUAL_GRID_SHAPE == (3, 32, 32, 13)
    assert LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE == (5, 32, 32, 13)
    assert LAB_RADIAL_VISUAL_ENCODER_CHANNELS == (
        "occupancy",
        "nominal_polytope_mask",
        "clipped_hp",
        "radius_center_m",
        "radius_bin_width_m",
    )

    encoder = LabNonuniformRadialEncoder(64)
    dynamic = torch.zeros(2, *LAB_RADIAL_VISUAL_GRID_SHAPE)
    assembled = encoder.encoder_grid(dynamic)
    edges = np.asarray(EXPECTED_EDGES, np.float32)
    centers = torch.from_numpy(0.5 * (edges[:-1] + edges[1:]))
    widths = torch.from_numpy(np.diff(edges))

    assert torch.equal(assembled[:, :3], dynamic)
    assert torch.allclose(assembled[:, 3], centers.view(1, 1, 1, 13))
    assert torch.allclose(assembled[:, 4], widths.view(1, 1, 1, 13))
    assert not any(
        isinstance(module, torch.nn.AvgPool3d)
        for module in encoder.modules()
    )


@pytest.mark.parametrize("obstacle_kind", ["sphere", "cylinder"])
def test_near_obstacle_changes_occupancy_hp_grid_and_encoder_token(
    obstacle_kind,
):
    observer = np.zeros(3)
    points = nonuniform_radial_grid_points(observer)
    obstacle_center = points[16, 16, 7]
    base_env = TaskEnvironment(_symmetric_config())
    if obstacle_kind == "sphere":
        obstacle_config = _symmetric_config(
            spheres=((*obstacle_center, 0.035),),
        )
    else:
        obstacle_config = _symmetric_config(
            cylinders=((obstacle_center[0], obstacle_center[1], 0.035),),
        )
    obstacle_env = TaskEnvironment(obstacle_config)

    base_grid = nonuniform_radial_safety_grid(base_env, observer)
    obstacle_grid = nonuniform_radial_safety_grid(obstacle_env, observer)
    assert base_grid.shape == LAB_RADIAL_VISUAL_GRID_SHAPE
    assert obstacle_grid[0].sum() > base_grid[0].sum()
    assert not np.array_equal(obstacle_grid[1], base_grid[1])
    assert not np.array_equal(obstacle_grid[2], base_grid[2])
    assert not np.array_equal(obstacle_grid, base_grid)

    torch.manual_seed(731)
    encoder = LabNonuniformRadialEncoder(64).eval()
    grids = torch.from_numpy(np.stack([base_grid, obstacle_grid])).float()
    with torch.no_grad():
        tokens = encoder(grids)
    assert tokens.shape == (2, 64)
    assert float(torch.linalg.vector_norm(tokens[0] - tokens[1])) > 1.0e-6


def test_spherical_padding_wraps_azimuth_and_rotates_across_poles():
    values = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(
        1,
        1,
        4,
        3,
        2,
    )
    padded = _SphericalTopologyPad3d()(values)

    assert padded.shape == (1, 1, 6, 5, 4)
    assert torch.equal(
        padded[:, :, 0, 1:-1, 1:-1],
        values[:, :, -1],
    )
    assert torch.equal(
        padded[:, :, -1, 1:-1, 1:-1],
        values[:, :, 0],
    )
    assert torch.equal(
        padded[:, :, 1:-1, 0, 1:-1],
        torch.roll(values[:, :, :, 0], shifts=2, dims=2),
    )
    assert torch.equal(
        padded[:, :, 1:-1, -1, 1:-1],
        torch.roll(values[:, :, :, -1], shifts=2, dims=2),
    )
    assert torch.equal(
        padded[:, :, 1:-1, 1:-1, 0],
        values[:, :, :, :, 0],
    )
    assert torch.equal(
        padded[:, :, 1:-1, 1:-1, -1],
        values[:, :, :, :, -1],
    )


@pytest.mark.parametrize(
    ("name", "token_dim", "trunk_depth", "gru_hidden", "parameter_count"),
    [
        ("radial64_d2", 64, 2, None, 207_582),
        ("radial128_d2", 128, 2, None, 276_254),
        ("radial256_d2", 256, 2, None, 413_598),
        ("radial64_d3", 64, 3, None, 209_934),
        ("radial64_d3_gru32", 64, 3, 32, 215_118),
    ],
)
def test_five_arm_parameter_and_encoded_context_contracts(
    name,
    token_dim,
    trunk_depth,
    gru_hidden,
    parameter_count,
):
    del name
    if gru_hidden is None:
        policy = LabNonuniformRadialFlowPolicy(
            grid_token_dim=token_dim,
            trunk_depth=trunk_depth,
        )
        expected_context_dim = 7 + token_dim
    else:
        policy = LabNonuniformRadialHistoryFlowPolicy(
            grid_token_dim=token_dim,
            trunk_depth=trunk_depth,
            history_token_dim=gru_hidden,
        )
        assert policy.history_encoder.hidden_size == 32
        expected_context_dim = 7 + token_dim + 32

    assert policy.flow.context_dim == expected_context_dim
    assert sum(parameter.numel() for parameter in policy.parameters()) == (
        parameter_count
    )
    assert policy.grid_encoder.conv3d[1].weight.shape == (16, 5, 3, 3, 3)
    assert policy.grid_encoder.conv3d[4].weight.shape == (32, 16, 3, 3, 3)
    assert policy.grid_encoder.conv3d[7].weight.shape == (48, 32, 3, 3, 3)
    assert policy.grid_encoder.radial_mixer[0].weight.shape == (
        64,
        48 * 13,
        1,
        1,
    )
    assert policy.grid_encoder.radial_mixer[5].weight.shape == (
        token_dim,
        64 * 4 * 4,
    )


@pytest.mark.parametrize(
    ("token_dim", "expected_macs"),
    [
        (64, 112_476_160),
        (128, 112_541_696),
        (256, 112_672_768),
    ],
)
def test_radial_encoder_dense_mac_count_contract(token_dim, expected_macs):
    """Count multiply-accumulates for one dense encoder grid."""
    macs = (
        16 * 32 * 32 * 13 * (5 * 3 * 3 * 3)
        + 32 * 16 * 16 * 13 * (16 * 3 * 3 * 3)
        + 48 * 8 * 8 * 13 * (32 * 3 * 3 * 3)
        + 64 * 8 * 8 * (48 * 13)
        + 64 * 4 * 4 * (64 * 3 * 3)
        + token_dim * (64 * 4 * 4)
    )
    assert macs == expected_macs
