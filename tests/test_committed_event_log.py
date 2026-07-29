from __future__ import annotations

import numpy as np

from scripts.run_ball_expansion import _retain_committed_round_events


def _event(gamma, episode, step, *, status=None):
    return {
        "round": 1,
        "gamma": gamma,
        "episode": episode,
        "step": step,
        "status": status,
        "candidates": np.full((4, 3, 2), episode, np.float32),
        "sigma_K": np.arange(4, dtype=np.float32),
        "selected": [1, 3],
        "verification": [{"valid": True}, {"valid": False}],
        "chosen_local": 0,
    }


def _round_row():
    return {
        "round": 1,
        "successful_executed_commit_by_gamma": {
            "0.1": {
                "success_episode_count": 2,
                "success_episode_above_fractions": {"0": None, "1": None},
                "committed_trajectory_count": 1,
                "committed_episode_ids": [1],
            },
            "0.3": {
                "success_episode_count": 1,
                "success_episode_above_fractions": {"3": None},
                "committed_trajectory_count": 1,
                "committed_episode_ids": [3],
            },
        },
    }


def test_committed_event_log_keeps_all_successes_and_exact_event_records():
    pending = [
        _event(0.1, 0, 0),
        _event(0.1, 0, 1, status="SUCCESS"),
        _event(0.1, 1, 0, status="SUCCESS"),
        _event(0.1, 2, 0, status="NVP"),
        _event(0.3, 3, 0, status="SUCCESS"),
    ]

    retained = _retain_committed_round_events(pending, _round_row())

    assert [id(event) for event in retained] == [
        id(pending[index]) for index in (0, 1, 2, 4)
    ]
    assert retained[0]["candidates"] is pending[0]["candidates"]
    assert retained[0]["sigma_K"] is pending[0]["sigma_K"]
    assert retained[0]["selected"] is pending[0]["selected"]
    assert retained[0]["verification"] is pending[0]["verification"]


def test_committed_event_log_rejects_missing_resolved_success_trace():
    pending = [
        _event(0.1, 0, 0, status="SUCCESS"),
        _event(0.3, 3, 0, status="SUCCESS"),
    ]

    try:
        _retain_committed_round_events(pending, _round_row())
    except RuntimeError as error:
        assert "have no event trace" in str(error)
    else:
        raise AssertionError("missing resolved SUCCESS trace was accepted")
