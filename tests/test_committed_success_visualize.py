from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from safe_mppi.committed_success_visualize import (
    COMMITTED_COLOR,
    FAILED_COLOR,
    OTHER_SUCCESS_COLOR,
    draw_gathering_cell,
    plot_gathering_commit_gallery,
    resolve_committed_success,
)


class _Scene:
    spheres = np.asarray([[1.5, 0.0, 2.0, 0.25]], float)
    start = np.asarray([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], float)
    goal = np.asarray([3.0, 0.0, 2.0], float)


def _event(round_i, gamma, episode, step, context_id, *, terminal=None,
           chosen=True, reason=None, y=0.0, z=2.0):
    before = np.asarray(
        [0.3 * step, y, z, 0.0, 0.0, 0.0], np.float32,
    )
    after = before.copy()
    if chosen:
        after[0] += 0.3
    return {
        "round": round_i,
        "gamma": gamma,
        "episode": episode,
        "step": step,
        "context_id": context_id,
        "robot": before,
        "robot_after": after,
        "chosen_local": (0 if chosen else None),
        "selected": [0],
        "sigma_K": np.asarray([0.5], np.float32),
        "status": terminal,
        "nvp_reason": reason,
    }


def _fixture():
    events = [
        _event(1, 0.1, 0, 0, 10, z=2.05),
        _event(1, 0.1, 0, 1, 11, z=2.10),
        _event(1, 0.1, 0, 2, 12, terminal="SUCCESS", z=2.15),
        _event(1, 0.1, 1, 0, 20, y=0.2, z=1.95),
        _event(1, 0.1, 1, 1, 21, terminal="SUCCESS", y=0.2, z=1.90),
        _event(1, 0.1, 2, 0, 30, z=1.85),
        _event(
            1, 0.1, 2, 1, 31, terminal="NVP", chosen=False,
            reason="VERIFIER", z=1.80,
        ),
        _event(
            1, 0.3, 3, 0, 40, terminal="NVP", chosen=False,
            reason="TARGET", z=1.95,
        ),
    ]
    committed_id = "r0001:g0.1:e000000"
    windows = [f"{committed_id}:w000000", f"{committed_id}:w000001"]
    empty = {
        "selector": "lowest_episode_id",
        "success_episode_count": 0,
        "success_episode_above_fractions": {},
        "committed_trajectory_id": None,
        "committed_episode_id": None,
        "above_fraction": None,
        "executed_steps": 0,
        "candidate_window_count": 0,
        "full_h_valid_window_count": 0,
        "committed_window_count": 0,
        "committed_window_ids": [],
    }
    manifest = {
        "config": {"gammas": [0.1, 0.3], "rounds": 1},
        "rounds": [{
            "round": 1,
            "successful_executed_commit_by_gamma": {
                "0.1": {
                    "selector": "lowest_episode_id",
                    "success_episode_count": 2,
                    "success_episode_above_fractions": {
                        "0": 0.8,
                        "1": 0.2,
                    },
                    "committed_trajectory_id": committed_id,
                    "committed_episode_id": 0,
                    "above_fraction": 0.8,
                    "executed_steps": 3,
                    "candidate_window_count": 2,
                    "full_h_valid_window_count": 2,
                    "committed_window_count": 2,
                    "committed_window_ids": windows,
                },
                "0.3": empty,
            },
            "committed_trajectory_ids": [committed_id],
            "committed_window_ids": windows,
        }],
    }
    return manifest, events


def test_resolve_committed_success_maps_authoritative_window_starts():
    manifest, events = _fixture()
    cells = resolve_committed_success(manifest, events)
    committed = cells[(1, 0.1)]
    empty = cells[(1, 0.3)]

    assert committed.committed_episode_id == 0
    assert committed.committed_window_context_ids == (10, 11)
    assert np.allclose(
        committed.committed_window_positions[:, 0],
        [0.0, 0.3],
    )
    assert committed.committed_episode.status == "SUCCESS"
    assert empty.committed_episode is None
    assert empty.committed_window_context_ids == ()


def test_resolve_committed_success_accepts_exact_success_only_event_log():
    manifest, events = _fixture()
    manifest["event_log"] = "committed_success"
    success_events = [
        event for event in events if int(event["episode"]) in {0, 1}
    ]

    cells = resolve_committed_success(manifest, success_events)

    assert cells[(1, 0.1)].committed_episode_id == 0
    assert {episode.episode for episode in cells[(1, 0.1)].episodes} == {0, 1}
    assert cells[(1, 0.3)].episodes == ()


def test_resolve_committed_success_rejects_missing_or_failed_trace():
    manifest, events = _fixture()
    manifest["event_log"] = "committed_success"
    success_events = [
        event for event in events if int(event["episode"]) in {0, 1}
    ]
    missing = [
        event for event in success_events if int(event["episode"]) != 1
    ]
    with pytest.raises(ValueError, match="success count mismatch"):
        resolve_committed_success(manifest, missing)

    with_failure = success_events + [
        event for event in events if int(event["episode"]) == 2
    ]
    with pytest.raises(ValueError, match="non-SUCCESS trace"):
        resolve_committed_success(manifest, with_failure)


def test_resolve_committed_success_maps_multiple_committed_trajectories():
    manifest, events = _fixture()
    detail = manifest["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"]
    first_id = detail["committed_trajectory_id"]
    second_id = "r0001:g0.1:e000001"
    first_windows = list(detail["committed_window_ids"])
    second_windows = [
        f"{second_id}:w000000",
        f"{second_id}:w000001",
    ]
    detail.update({
        "committed_trajectory_count": 2,
        "committed_trajectory_ids": [first_id, second_id],
        "committed_episode_ids": [0, 1],
        "committed_trajectories": [
            {
                "trajectory_id": first_id,
                "episode_id": 0,
                "executed_steps": 3,
                "committed_window_count": 2,
                "committed_window_ids": first_windows,
            },
            {
                "trajectory_id": second_id,
                "episode_id": 1,
                "executed_steps": 2,
                "committed_window_count": 2,
                "committed_window_ids": second_windows,
            },
        ],
        "all_committed_window_count": 4,
        "all_committed_window_ids": first_windows + second_windows,
    })
    manifest["rounds"][0]["committed_trajectory_ids"] = [
        first_id, second_id,
    ]
    manifest["rounds"][0]["committed_window_ids"] = (
        first_windows + second_windows
    )

    cell = resolve_committed_success(manifest, events)[(1, 0.1)]
    assert cell.committed_episode_ids == (0, 1)
    assert tuple(row.episode for row in cell.committed_episodes) == (0, 1)
    assert cell.committed_window_context_ids == (10, 11, 20, 21)


def test_resolve_committed_success_fails_on_unresolved_trajectory_id():
    manifest, events = _fixture()
    broken = copy.deepcopy(manifest)
    broken["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"][
        "committed_trajectory_id"
    ] = "r0001:g0.1:e000099"
    with pytest.raises(ValueError, match="does not match"):
        resolve_committed_success(broken, events)


def test_resolve_committed_success_fails_on_unresolved_window_start():
    manifest, events = _fixture()
    broken = copy.deepcopy(manifest)
    detail = broken["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"]
    invalid = "r0001:g0.1:e000000:w000099"
    detail["committed_window_ids"][-1] = invalid
    broken["rounds"][0]["committed_window_ids"][-1] = invalid
    with pytest.raises(ValueError, match="starts after the executed trace"):
        resolve_committed_success(broken, events)


def test_gathering_cell_styles_commit_success_and_failure_separately():
    manifest, events = _fixture()
    cell = resolve_committed_success(manifest, events)[(1, 0.1)]
    fig, (ax_side, ax_head) = plt.subplots(1, 2)
    draw_gathering_cell(
        ax_side, ax_head, _Scene(), cell,
        target_region="above_halfspace", gate_active=True,
    )
    colors = [line.get_color() for line in ax_side.lines]
    plt.close(fig)

    assert COMMITTED_COLOR in colors
    assert OTHER_SUCCESS_COLOR in colors
    assert FAILED_COLOR in colors


def test_gathering_commit_gallery_writes_png_and_pdf(tmp_path):
    manifest, events = _fixture()
    output = tmp_path / "gathering_committed_success_gallery.png"
    result = plot_gathering_commit_gallery(
        _Scene(),
        manifest,
        events,
        output,
        target_region="above_halfspace",
        target_gate_start_round=1,
    )
    assert result == output
    assert output.stat().st_size > 0
    assert output.with_suffix(".pdf").stat().st_size > 0
