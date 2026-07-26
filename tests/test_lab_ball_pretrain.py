import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.acquire import run_episode
from safe_mppi.config import ExperimentConfig, load_config
from safe_mppi.controller import Mode1SafeMPPI, _taskspace_exponential_penalty
from safe_mppi.environment import ReferenceGovernor, TaskEnvironment
from safe_mppi.lab_flow_task import (
    LAB_CONTEXT_DIM,
    build_lab_context,
    governed_plan_states,
    lab_demo_windows,
)


ROOT = Path(__file__).resolve().parents[1]
LAB_CONFIG = ROOT / "configs/lab_ball_pretrain.json"
LAB_ARCHIVE = (
    ROOT
    / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
)


def test_lab_config_fixes_minhyuk_reference_contract():
    raw = json.loads(LAB_CONFIG.read_text())
    config = load_config(LAB_CONFIG)
    assert config.taskspace.start == (-2.1, 1.5, 0.9, 0.0, 0.0, 0.0)
    assert config.taskspace.goal == (0.7, -1.5, 0.9)
    assert config.obstacles.spheres == ((-0.7, 0.0, 0.9, 0.379),)
    assert np.allclose(
        config.taskspace.bounds,
        [[-2.5, 1.3], [-1.7, 1.8], [0.4, 2.0]],
    )
    assert raw["safemppi"]["deployment_accel_smooth"] == 0.4
    assert config.safemppi.demo_u_max == 0.3
    assert config.safemppi.max_speed == 0.7
    assert config.safemppi.max_vertical_speed == 0.3
    assert config.safemppi.integration_substeps == 10
    assert config.data.rollout_dynamics == "minhyuk_reference_governor"
    assert config.data.acceptance == "nominal_safe_success"


def test_lab_config_missing_governor_constant_fails_closed(tmp_path):
    raw = json.loads(LAB_CONFIG.read_text())
    del raw["safemppi"]["deployment_accel_smooth"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exact deployment constants"):
        load_config(path)


def test_reference_governor_matches_declared_recurrence():
    config = load_config(LAB_CONFIG)
    governor = ReferenceGovernor(config.safemppi)
    state = np.zeros(6, np.float32)
    command = np.array([0.8, -0.8, 0.8], np.float32)
    next_state, applied, dense = governor.step(state, command)
    assert np.allclose(applied, [0.12, -0.12, 0.12])

    position = np.zeros(3)
    velocity = np.zeros(3)
    for _ in range(10):
        velocity += 0.01 * applied
        speed = np.linalg.norm(velocity)
        if speed > 0.7:
            velocity *= 0.7 / speed
        velocity[2] = np.clip(velocity[2], -0.3, 0.3)
        position += 0.01 * velocity
    assert np.allclose(next_state, np.r_[position, velocity], atol=1.0e-7)
    assert np.allclose(dense[-1], position, atol=1.0e-7)


def test_lab_context_contains_previous_applied_acceleration():
    config = load_config(LAB_CONFIG)
    env = TaskEnvironment(config)
    previous = np.array([0.02, -0.03, 0.04], np.float32)
    context = build_lab_context(env, env.start, 0.3, previous)
    assert context.shape == (LAB_CONTEXT_DIM,)
    assert np.allclose(context[-3:], previous)


def test_lab_window_loader_uses_raw_targets_and_governor_state():
    contexts, plans, meta, config = lab_demo_windows(
        LAB_ARCHIVE, validate_archive=False,
    )
    assert contexts.shape[1] == LAB_CONTEXT_DIM
    assert plans.shape[1:] == (10, 3)
    assert np.allclose(contexts[0, -3:], 0.0)

    manifest = json.loads((LAB_ARCHIVE / "manifest.json").read_text())
    first = manifest["runs"][0]
    data = np.load(LAB_ARCHIVE / first["file"])
    assert np.allclose(plans[0], data["controls"][:10])
    assert meta[0]["t"] == 0
    assert meta[1]["t"] == 1
    assert np.allclose(contexts[1, -3:], data["executed_controls"][0])

    env = TaskEnvironment(config)
    states, applied, _ = governed_plan_states(
        env,
        data["states"][0],
        data["controls"][:10],
        np.zeros(3, np.float32),
    )
    assert np.allclose(states, data["states"][:11], atol=1.0e-6, rtol=0.0)
    assert np.allclose(
        applied, data["executed_controls"][:10], atol=1.0e-6, rtol=0.0,
    )


def test_lab_archive_keeps_raw_and_applied_controls_synchronized():
    config = load_config(LAB_CONFIG)
    short_task = replace(config.taskspace, max_steps=1)
    small_mppi = replace(config.safemppi, num_samples=16)
    short = ExperimentConfig(
        short_task, config.obstacles, small_mppi, config.data, config.raw,
    )
    env = TaskEnvironment(short)
    controller = Mode1SafeMPPI(small_mppi, env)
    _, arrays = run_episode(
        env, controller, 0.3, 7, short.data.rollout_dynamics,
    )
    assert arrays["controls"].shape == (1, 3)
    assert arrays["executed_controls"].shape == (1, 3)
    assert np.allclose(
        arrays["executed_controls"][0],
        0.4 * arrays["controls"][0],
        atol=1.0e-6,
    )
    assert np.max(np.abs(arrays["controls"])) <= 0.3 + 1.0e-6
    assert np.max(np.abs(arrays["executed_controls"])) <= 0.3 + 1.0e-6


def test_legacy_environment_step_remains_analytic():
    config = load_config(ROOT / "configs/ball_fan_demo.json")
    env = TaskEnvironment(config)
    state = np.array([0.0, 0.0, 2.0, 0.2, -0.1, 0.0], np.float32)
    control = np.array([0.3, 0.2, -0.1], np.float32)
    expected = np.r_[
        state[:3] + 0.1 * state[3:] + 0.5 * 0.1**2 * control,
        state[3:] + 0.1 * control,
    ]
    assert np.allclose(env.step(state, control), expected)


def test_taskspace_exponential_penalty_is_zero_inside_positive_outside():
    bounds = torch.tensor([[-2.5, 1.3], [-1.7, 1.8], [0.4, 2.0]])
    points = torch.tensor([[0.0, 0.0, 1.0], [1.4, 0.0, 1.0]])
    penalty = _taskspace_exponential_penalty(points, bounds, 5.0, 0.05)
    assert penalty[0].item() == 0.0
    assert penalty[1].item() > 0.0


def test_checked_in_lab_archive_has_50_accepted_below_plane_demos_per_gamma():
    manifest = json.loads((LAB_ARCHIVE / "manifest.json").read_text())
    qualification = json.loads(
        (LAB_ARCHIVE / "qualification/lab_ball_qualification.json").read_text()
    )
    for gamma in (0.1, 0.3, 0.5, 1.0):
        rows = [
            row for row in manifest["runs"]
            if np.isclose(float(row["gamma"]), gamma)
        ]
        assert len(rows) == 50
        assert all(
            row["accepted"] and row["success"]
            and not row["collision"] and not row["taskspace_violation"]
            for row in rows
        )
        summary = next(
            row for row in qualification["per_gamma"]
            if np.isclose(float(row["gamma"]), gamma)
        )
        assert summary["crossing_below_plane_fraction"] == 1.0
        assert summary["max_abs_raw_accel_mps2"] <= 0.300001
        assert summary["max_speed_mps"] <= 0.700001
        assert summary["max_vertical_speed_mps"] <= 0.300001
