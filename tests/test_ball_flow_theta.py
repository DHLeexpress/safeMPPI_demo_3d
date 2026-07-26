from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.ball_flow_task import BallFlowTask, build_context
from safe_mppi.ball_flow_theta import (
    CONTEXT_DIM,
    THETA_VALUES,
    ThetaBallFlowTask,
    build_theta_context,
    demo_windows_theta,
    local_to_world,
    raw_rollout_theta,
    requested_theta,
    start_goal_frame,
    theta_context_state,
    theta_name,
    trajectory_crossing_theta,
    world_to_local,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


ROOT = Path(__file__).resolve().parents[1]


def test_frame_context_and_vector_round_trip():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    env = TaskEnvironment(config)
    frame = start_goal_frame(env)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1.0e-7)
    np.testing.assert_allclose(frame[:, 0], [1.0, 0.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(frame[:, 1], [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(frame[:, 2], [0.0, 0.0, 1.0], atol=1.0e-7)

    vectors = np.array([[0.2, -0.4, 0.8], [-1.0, 2.0, 0.5]], np.float32)
    np.testing.assert_allclose(
        local_to_world(world_to_local(vectors, frame), frame), vectors, atol=1.0e-7,
    )

    state = np.array([0.3, 0.1, 1.9, 0.2, -0.1, 0.4], np.float32)
    context = build_theta_context(env, state, gamma=0.3, theta=np.pi / 2, frame=frame)
    assert context.shape == (CONTEXT_DIM,)
    np.testing.assert_allclose(context[-3:], [0.3, 0.0, 1.0], atol=1.0e-7)
    recovered, theta = theta_context_state(env, context, frame)
    np.testing.assert_allclose(recovered, state, atol=1.0e-6)
    assert theta == pytest.approx(np.pi / 2)


def test_crossing_angle_and_balanced_episode_cycle():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    env = TaskEnvironment(config)
    paths = {
        "below": np.array([[1.4, 0.0, 1.8], [1.6, 0.0, 1.8]]),
        "above": np.array([[1.4, 0.0, 2.2], [1.6, 0.0, 2.2]]),
        "left": np.array([[1.4, 0.2, 2.0], [1.6, 0.2, 2.0]]),
        "right": np.array([[1.4, -0.2, 2.0], [1.6, -0.2, 2.0]]),
    }
    for expected, positions in paths.items():
        assert theta_name(trajectory_crossing_theta(env, positions)) == expected
    assert trajectory_crossing_theta(
        env, np.array([[0.1, 0.0, 2.0], [0.5, 0.0, 2.0]])
    ) is None
    assert tuple(requested_theta(index) for index in range(8)) == THETA_VALUES * 2


def test_theta_task_converts_local_plans_for_verification_and_advance():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    theta_task = ThetaBallFlowTask(config)
    legacy_task = BallFlowTask(config)
    state = theta_task.reset(gamma=0.3, episode=1, seed=123)
    assert state["theta"] == pytest.approx(np.pi / 2)
    context = theta_task.context(state, 0.3)
    assert context.shape == (CONTEXT_DIM,)

    local_plan = torch.zeros(1, 10, 3)
    local_plan[:, :, 0] = 0.1
    world_plan = theta_task.world_plan(local_plan)
    legacy_context = torch.from_numpy(build_context(legacy_task.env, state["x"], 0.3))
    actual = theta_task.verify(context, local_plan, 0.3)[0]
    expected = legacy_task.verify(legacy_context, world_plan, 0.3)[0]
    assert actual == expected

    advanced = theta_task.advance(state, local_plan[0])
    legacy_advanced = legacy_task.advance(
        {"x": state["x"], "steps": 0, "collided": False, "oob": False},
        world_plan[0],
    )
    np.testing.assert_allclose(advanced["x"], legacy_advanced["x"], atol=1.0e-7)
    assert advanced["theta"] == state["theta"]


def _write_demo(path: Path, gamma: float, seed: int, route_z: float):
    controls = np.zeros((12, 3), np.float32)
    states = np.zeros((13, 6), np.float32)
    states[:, 0] = np.linspace(0.0, 3.0, 13)
    states[:, 2] = route_z
    np.savez(path, states=states, controls=controls, gamma=gamma, seed=seed)


def test_demo_loader_filters_failures_and_enforces_per_gamma_limit(tmp_path):
    source_config = ROOT / "configs" / "ball_fan_demo.json"
    (tmp_path / "resolved_config.json").write_text(source_config.read_text())
    rows = []
    for gamma, seed, success, z in [
        (0.1, 1, True, 1.8),
        (0.1, 2, True, 2.2),
        (0.1, 3, False, 2.2),
        (0.3, 4, True, 2.2),
    ]:
        filename = f"g{gamma}_s{seed}.npz"
        _write_demo(tmp_path / filename, gamma, seed, z)
        rows.append({
            "gamma": gamma,
            "seed": seed,
            "success": success,
            "file": filename,
        })
    (tmp_path / "manifest.json").write_text(json.dumps({
        "gammas": [0.1, 0.3],
        "runs": rows,
    }))

    contexts, plans, meta, _ = demo_windows_theta(tmp_path, per_gamma_limit=1)
    assert contexts.shape == (6, 12)
    assert plans.shape == (6, 10, 3)
    assert {(row["gamma"], row["seed"]) for row in meta} == {(0.1, 1), (0.3, 4)}
    assert {row["requested_route"] for row in meta} == {"below", "above"}
    with pytest.raises(ValueError, match="2 required"):
        demo_windows_theta(tmp_path, per_gamma_limit=2)


def test_raw_rollout_emits_world_controls_and_requested_theta():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")

    class LocalPolicy:
        def sample(self, context, count, generator):
            assert context.shape == (CONTEXT_DIM,)
            plan = torch.zeros(count, 10, 3)
            plan[:, :, 0] = 0.1
            return plan

    result = raw_rollout_theta(
        LocalPolicy(), config, gamma=0.5, seed=9, theta=np.pi / 2,
        max_steps=2,
    )
    assert result["requested_theta"] == pytest.approx(np.pi / 2)
    assert result["requested_route"] == "above"
    assert result["controls"].shape == (2, 3)
    np.testing.assert_allclose(result["controls"][:, 1:], 0.0, atol=1.0e-7)
