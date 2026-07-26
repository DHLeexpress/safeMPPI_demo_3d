from __future__ import annotations

import numpy as np

from safe_mppi.ball_flow_task import raw_window_validity_fraction
from safe_mppi.config import load_config


def test_window_validity_uses_every_start_and_truncated_tail(monkeypatch):
    config = load_config("configs/ball_biased_demo.json")
    controls = np.zeros((12, 3), np.float32)
    states = np.zeros((13, 6), np.float32)
    states[:, 2] = 1.4  # Outside the expansion corridor, but inside the task space.
    horizons = []

    def fake_certify(positions, spheres, cylinders, gamma, sensing_range):
        horizon = len(positions) - 1
        horizons.append(horizon)
        return horizon != 4, 1.0

    monkeypatch.setattr("safe_mppi.ball_flow_task.certify_window", fake_certify)
    fraction = raw_window_validity_fraction(config, states, controls, 0.3)

    assert horizons == [10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert fraction == 11 / 12


def test_window_validity_rejects_only_windows_with_failed_safety_predicate(monkeypatch):
    config = load_config("configs/ball_biased_demo.json")
    controls = np.zeros((3, 3), np.float32)
    states = np.zeros((4, 6), np.float32)
    states[:, :3] = np.array([0.0, 0.0, 2.0], np.float32)
    call = 0

    def fake_certify(positions, spheres, cylinders, gamma, sensing_range):
        nonlocal call
        call += 1
        return call != 2, 1.0

    monkeypatch.setattr("safe_mppi.ball_flow_task.certify_window", fake_certify)
    assert raw_window_validity_fraction(config, states, controls, 1.0) == 2 / 3


def test_window_validity_empty_or_malformed_trajectory_is_zero():
    config = load_config("configs/ball_biased_demo.json")
    assert raw_window_validity_fraction(
        config, np.zeros((1, 6), np.float32), np.zeros((0, 3), np.float32), 0.5,
    ) == 0.0
    assert raw_window_validity_fraction(
        config, np.zeros((2, 6), np.float32), np.zeros((2, 3), np.float32), 0.5,
    ) == 0.0
