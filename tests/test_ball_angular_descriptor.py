from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from safe_mppi.ball_flow_task import (
    BallFlowTask,
    build_context,
    closest_approach_angular_descriptor,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


ROOT = Path(__file__).resolve().parents[1]


def _crossing_state(y: float, z: float) -> np.ndarray:
    return np.array([1.4, y, z, 1.0, 0.0, 0.0], np.float32)


def test_descriptor_maps_four_continuous_transverse_directions():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    env = TaskEnvironment(config)
    plan = np.zeros((10, 3), np.float32)

    expected = {
        "left": ((0.4, 2.0), (1.0, 0.0)),
        "right": ((-0.4, 2.0), (-1.0, 0.0)),
        "above": ((0.0, 2.4), (0.0, 1.0)),
        "below": ((0.0, 1.6), (0.0, -1.0)),
    }
    for (y, z), direction in expected.values():
        actual = closest_approach_angular_descriptor(
            env, _crossing_state(y, z), plan,
        )
        np.testing.assert_allclose(actual, direction, atol=1.0e-7)
        np.testing.assert_allclose(np.linalg.norm(actual), 1.0, atol=1.0e-7)


def test_descriptor_is_invariant_to_joint_rigid_rotation():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    env = TaskEnvironment(config)
    state = _crossing_state(0.3, 2.2)
    plan = np.zeros((10, 3), np.float32)
    reference = closest_approach_angular_descriptor(env, state, plan)

    angle = 0.61
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated_env = TaskEnvironment(config)
    rotated_env.start[:3] = rotation @ env.start[:3]
    rotated_env.goal = (rotation @ env.goal).astype(np.float32)
    rotated_env.spheres[:, :3] = env.spheres[:, :3] @ rotation.T
    rotated_state = state.copy()
    rotated_state[:3] = rotation @ state[:3]
    rotated_state[3:6] = rotation @ state[3:6]
    rotated_plan = plan @ rotation.T

    actual = closest_approach_angular_descriptor(
        rotated_env, rotated_state, rotated_plan,
        world_up=rotation @ np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(actual, reference, atol=1.0e-6)


def test_task_batch_descriptor_uses_context_and_candidate_plans():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    task = BallFlowTask(config)
    state = _crossing_state(0.4, 2.0)
    context = torch.from_numpy(build_context(task.env, state, gamma=0.3))
    candidates = torch.zeros((2, 10, 3), dtype=torch.float32)

    descriptors = task.angular_descriptors(context, candidates)

    assert descriptors.shape == (2, 2)
    assert descriptors.dtype == candidates.dtype
    np.testing.assert_allclose(
        descriptors.cpu().numpy(),
        np.array([[1.0, 0.0], [1.0, 0.0]], np.float32),
        atol=1.0e-7,
    )


def test_d4_replay_batch_rotates_context_vectors_and_controls():
    config = load_config(ROOT / "configs" / "ball_fan_demo.json")
    task = BallFlowTask(config)
    contexts = torch.tensor([[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        0.3,
    ]])
    candidates = torch.zeros(1, 10, 3)
    candidates[0, :, 1] = 2.0
    candidates[0, :, 2] = 3.0

    rotated_contexts, rotated_candidates = task.d4_replay_batch(
        contexts, candidates,
    )

    assert rotated_contexts.shape == (4, 10)
    assert rotated_candidates.shape == (4, 10, 3)
    assert torch.allclose(rotated_contexts[0], contexts[0])
    assert torch.allclose(
        rotated_contexts[1, [1, 2, 4, 5, 7, 8]],
        torch.tensor([-3.0, 2.0, -6.0, 5.0, -9.0, 8.0]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        rotated_candidates[1, 0],
        torch.tensor([0.0, -3.0, 2.0]),
        atol=1.0e-6,
    )
