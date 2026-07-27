from pathlib import Path

import numpy as np
import pytest

from flow_deployment.lab_pretrained import (
    DEFAULT_CHECKPOINT_SHA256,
    load_lab_deployment_controller,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "results/lab_ball_pretrain/pretrain_raw10_h48p32_s0/pretrained.pt"
)
CONFIG = ROOT / "configs/lab_ball_pretrain.json"


def test_completed_checkpoint_loads_with_runtime_temperature():
    env = TaskEnvironment(load_config(CONFIG))
    controller, contract = load_lab_deployment_controller(
        CHECKPOINT,
        env,
        sampling_temperature=0.7,
        expected_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    action, info = controller.plan(
        env.start,
        env.goal,
        0.3,
        seed=17,
    )
    assert action.shape == (3,)
    assert np.max(np.abs(action)) <= 0.3 + 1.0e-6
    assert info["sampling_temperature"] == 0.7
    assert contract["sampling_temperature"] == 0.7
    assert contract["context_schema"] == "lab_raw10_v1"


def test_checkpoint_contract_fails_closed_on_wrong_hash():
    env = TaskEnvironment(load_config(CONFIG))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_lab_deployment_controller(
            CHECKPOINT,
            env,
            expected_sha256="0" * 64,
        )


def test_online_controller_rebuilds_context_from_current_state():
    env = TaskEnvironment(load_config(CONFIG))
    controller, _ = load_lab_deployment_controller(
        CHECKPOINT,
        env,
        expected_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    controller.plan(env.start, env.goal, 0.3, seed=23)
    moved = env.start.copy()
    moved[:3] += np.array([0.1, -0.05, 0.02], np.float32)
    controller.plan(moved, env.goal, 0.3, seed=23)
    assert np.array_equal(controller.trace[0]["state"], env.start)
    assert np.array_equal(controller.trace[1]["state"], moved)
    assert not np.array_equal(
        controller.trace[0]["action"],
        controller.trace[1]["action"],
    )
