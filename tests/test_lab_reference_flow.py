from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.flow_model import ConditionalFlowMLP
from safe_mppi.lab_reference_flow_task import (
    LAB_REFERENCE_CONTEXT_DIM,
    LabReferenceFlowController,
    lab_reference_demo_windows,
    policy_context,
    raw_reference_rollout,
    reference_window_validity_fraction,
)
from safe_mppi.lab_visual_flow import (
    LAB_VISUAL_CHANNELS,
    LAB_VISUAL_FRAME,
    LAB_VISUAL_GRID_SHAPE,
    LAB_VISUAL_HISTORY_LENGTH,
    LAB_VISUAL_HISTORY_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    LabVisualHistoryFlowPolicy,
    LabVisualFlowPolicy,
    load_lab_reference_policy,
    spherical_grid_points,
    spherical_safety_grid,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
CONFIG = ARCHIVE / "resolved_config.json"


def test_lab_reference_loader_is_ten_dimensional_and_uses_raw_targets():
    contexts, plans, metadata, _ = lab_reference_demo_windows(
        ARCHIVE,
        validate_archive=False,
    )
    assert contexts.shape[1] == LAB_REFERENCE_CONTEXT_DIM
    assert plans.shape[1:] == (10, 3)
    first = np.load(ARCHIVE / "run_g0.1_s3.npz")
    assert np.allclose(plans[0], first["controls"][:10])
    assert metadata[0]["t"] == 0


def test_raw_reference_rollout_has_no_stateful_governor():
    config = load_config(CONFIG)
    torch.manual_seed(3)
    policy = ConditionalFlowMLP(
        context_dim=LAB_REFERENCE_CONTEXT_DIM,
        plan_shape=(10, 3),
        hidden=8,
        representation_dim=4,
        control_limit=config.safemppi.demo_u_max,
        nfe=2,
        trunk_depth=2,
        time_features="raw1",
    )
    first = raw_reference_rollout(policy, config, 0.3, 11, max_steps=2)
    second = raw_reference_rollout(policy, config, 0.3, 11, max_steps=2)
    assert np.array_equal(first["controls"], second["controls"])
    assert first["dense_steps"].shape == (2, 10, 3)


def test_sampling_temperature_default_is_one_and_changes_the_base_scale():
    config = load_config(CONFIG)
    torch.manual_seed(8)
    policy = ConditionalFlowMLP(
        context_dim=LAB_REFERENCE_CONTEXT_DIM,
        plan_shape=(10, 3),
        hidden=8,
        representation_dim=4,
        control_limit=config.safemppi.demo_u_max,
        nfe=2,
        trunk_depth=2,
        time_features="raw1",
    )
    default = raw_reference_rollout(policy, config, 0.3, 19, max_steps=2)
    explicit = raw_reference_rollout(
        policy,
        config,
        0.3,
        19,
        max_steps=2,
        sampling_temperature=1.0,
    )
    cooler = raw_reference_rollout(
        policy,
        config,
        0.3,
        19,
        max_steps=2,
        sampling_temperature=0.3,
    )
    assert np.array_equal(default["controls"], explicit["controls"])
    assert not np.array_equal(default["controls"], cooler["controls"])


def test_deploy_controller_exposes_temperature_without_internal_smoothing():
    config = load_config(CONFIG)
    env = TaskEnvironment(config)
    torch.manual_seed(12)
    policy = ConditionalFlowMLP(
        context_dim=LAB_REFERENCE_CONTEXT_DIM,
        plan_shape=(10, 3),
        hidden=8,
        representation_dim=4,
        control_limit=config.safemppi.demo_u_max,
        nfe=2,
        trunk_depth=2,
        time_features="raw1",
    )
    controller = LabReferenceFlowController(
        policy,
        env,
        sampling_temperature=0.5,
    )
    action, info = controller.plan(env.start, env.goal, 0.3, seed=7)
    assert action.shape == (3,)
    assert np.max(np.abs(action)) <= config.safemppi.demo_u_max + 1.0e-6
    assert info["sampling_temperature"] == 0.5


def test_reference_validity_checks_truncated_tail(monkeypatch):
    config = load_config(CONFIG)
    states = np.repeat(np.asarray(config.taskspace.start, np.float32)[None], 4, axis=0)
    dense = np.repeat(states[1:, None, :3], 10, axis=1)
    horizons = []

    def fake_certify(positions, *args, **kwargs):
        horizons.append(len(positions) - 1)
        return True, 0.1

    monkeypatch.setattr(
        "safe_mppi.lab_reference_flow_task.certify_window",
        fake_certify,
    )
    assert reference_window_validity_fraction(config, states, dense, 0.3) == 1.0
    assert horizons == [3, 2, 1]


def test_spherical_safety_grid_is_bounded_and_multi_obstacle_sensitive():
    config = load_config(CONFIG)
    env = TaskEnvironment(config)
    grid = spherical_safety_grid(env, env.start[:3])
    assert grid.shape == LAB_VISUAL_GRID_SHAPE
    assert np.isin(grid[0], [0.0, 1.0]).all()
    assert np.isin(grid[1], [0.0, 1.0]).all()
    assert float(grid[2].min()) >= -1.0
    assert float(grid[2].max()) <= 1.0

    extra = tuple(config.obstacles.spheres) + ((-1.7, 1.1, 0.9, 0.2),)
    multi = replace(
        config,
        obstacles=replace(config.obstacles, spheres=extra),
    )
    multi_grid = spherical_safety_grid(
        TaskEnvironment(multi),
        env.start[:3],
    )
    assert not np.array_equal(grid, multi_grid)


def test_visual_grid_reconstructs_sphere_and_cylinder_occupancy():
    config = load_config(CONFIG)
    combined = replace(
        config,
        obstacles=replace(
            config.obstacles,
            cylinders=((-1.15, 0.85, 0.22),),
        ),
    )
    env = TaskEnvironment(combined)
    observer = np.array([-1.55, 0.25, 0.9])
    points = spherical_grid_points(
        observer,
        env.mppi.sensing_range,
    ).reshape(-1, 3)
    grid = spherical_safety_grid(env, observer)
    sphere = env.spheres[0]
    cylinder = env.cylinders[0]
    sphere_inside = (
        np.linalg.norm(points - sphere[:3], axis=1) < sphere[3]
    )
    cylinder_inside = (
        np.linalg.norm(points[:, :2] - cylinder[:2], axis=1)
        < cylinder[2]
    )
    assert sphere_inside.any()
    assert cylinder_inside.any()
    assert np.array_equal(
        grid[0].reshape(-1).astype(bool),
        sphere_inside | cylinder_inside,
    )
    assert np.array_equal(
        grid[1].astype(bool),
        grid[2] >= 0.0,
    )


def test_visual_policy_uses_packed_grid_and_preserves_raw_action_contract(tmp_path):
    config = load_config(CONFIG)
    env = TaskEnvironment(config)
    torch.manual_seed(21)
    policy = LabVisualFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=4,
        control_limit=config.safemppi.demo_u_max,
        nfe=2,
    )
    context = torch.from_numpy(
        policy_context(policy, env, env.start, 0.3)
    )
    assert context.shape == (LAB_VISUAL_PACKED_DIM,)
    plan = policy.sample(
        context,
        2,
        torch.Generator().manual_seed(5),
    )
    assert plan.shape == (2, 10, 3)
    assert float(plan.abs().max()) <= config.safemppi.demo_u_max + 1.0e-6
    assert policy.cfm_loss(
        context[None].expand(2, -1),
        plan,
        reduction="none",
    ).shape == (2,)
    assert policy.embed(
        context[None].expand(2, -1),
        plan,
    ).shape == (2, 4)

    checkpoint = tmp_path / "visual.pt"
    torch.save({
        "model": policy.state_dict(),
        "arch": {
            "kind": LAB_VISUAL_SCHEMA,
            "plan_shape": [10, 3],
            "hidden": 8,
            "representation_dim": 4,
            "grid_token_dim": 4,
            "grid_shape": list(LAB_VISUAL_GRID_SHAPE),
            "grid_channels": list(LAB_VISUAL_CHANNELS),
            "grid_frame": LAB_VISUAL_FRAME,
            "control_limit": config.safemppi.demo_u_max,
            "nfe": 2,
            "trunk_depth": 2,
            "time_features": "raw1",
        },
    }, checkpoint)
    loaded = load_lab_reference_policy(checkpoint)
    assert isinstance(loaded, LabVisualFlowPolicy)
    assert loaded.context_schema == LAB_VISUAL_SCHEMA


def test_visual_history_windows_use_only_preceding_raw_commands():
    contexts, _, metadata, _ = lab_reference_demo_windows(
        ARCHIVE,
        context_schema=LAB_VISUAL_HISTORY_SCHEMA,
        validate_archive=False,
    )
    first_run = np.load(ARCHIVE / "run_g0.1_s3.npz")
    first_rows = [
        index for index, row in enumerate(metadata)
        if row["gamma"] == 0.1 and row["seed"] == 3
    ]
    assert first_rows

    first_history = contexts[first_rows[0], LAB_VISUAL_PACKED_DIM:].reshape(
        LAB_VISUAL_HISTORY_LENGTH, 4
    )
    assert not first_history[:, 3].any()

    second_history = contexts[first_rows[1], LAB_VISUAL_PACKED_DIM:].reshape(
        LAB_VISUAL_HISTORY_LENGTH, 4
    )
    assert second_history[-1, 3] == 1.0
    assert np.array_equal(second_history[-1, :3], first_run["controls"][0])
    assert not second_history[:-1, 3].any()


def test_visual_history_policy_round_trip_and_expansion_freeze(tmp_path):
    config = load_config(CONFIG)
    env = TaskEnvironment(config)
    torch.manual_seed(27)
    policy = LabVisualHistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=4,
        history_token_dim=3,
        control_limit=config.safemppi.demo_u_max,
        nfe=2,
    )
    history = np.zeros((LAB_VISUAL_HISTORY_LENGTH, 4), np.float32)
    context = torch.from_numpy(
        policy_context(
            policy,
            env,
            env.start,
            0.3,
            raw_history=history,
        )
    )
    assert context.shape == (LAB_VISUAL_HISTORY_PACKED_DIM,)
    assert policy.sample(
        context,
        2,
        torch.Generator().manual_seed(11),
    ).shape == (2, 10, 3)

    checkpoint = tmp_path / "visual_history.pt"
    torch.save({
        "model": policy.state_dict(),
        "arch": {
            "kind": LAB_VISUAL_HISTORY_SCHEMA,
            "plan_shape": [10, 3],
            "hidden": 8,
            "representation_dim": 4,
            "grid_token_dim": 4,
            "history_token_dim": 3,
            "history_length": LAB_VISUAL_HISTORY_LENGTH,
            "grid_shape": list(LAB_VISUAL_GRID_SHAPE),
            "grid_channels": list(LAB_VISUAL_CHANNELS),
            "grid_frame": LAB_VISUAL_FRAME,
            "control_limit": config.safemppi.demo_u_max,
            "nfe": 2,
            "trunk_depth": 2,
            "time_features": "raw1",
        },
    }, checkpoint)
    loaded = load_lab_reference_policy(checkpoint)
    assert isinstance(loaded, LabVisualHistoryFlowPolicy)
    assert loaded.context_schema == LAB_VISUAL_HISTORY_SCHEMA
    with pytest.raises(ValueError, match="explicit"):
        loaded.expansion_parameter_groups(1.0e-4, 0.1)
    loaded.expansion_parameter_groups(
        1.0e-4, 0.1, freeze_history_encoder=True,
    )
    assert not any(
        parameter.requires_grad
        for parameter in loaded.history_encoder.parameters()
    )
