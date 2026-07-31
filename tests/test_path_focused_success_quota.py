import json
from pathlib import Path

import numpy as np
import pytest

from safe_mppi.config import load_config
from safe_mppi.lab_visual_flow import LAB_RADIAL_VISUAL_SCHEMA
from safe_mppi.lab_reference_flow_task import lab_reference_demo_windows
from safe_mppi.path_focused_collection import (
    collect_path_focused_success_quota,
    path_focused_collection_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_cylinders_path_v2.json"


def _fake_controller(*args, **kwargs):
    del args, kwargs
    return object()


def _episode_row(env, gamma, seed, success):
    controls = np.zeros((10, 3), np.float32)
    states = np.zeros((11, 6), np.float32)
    states[0] = env.start
    endpoint = env.goal if success else (env.start[:3] + env.goal) / 2.0
    states[:, :3] = np.linspace(env.start[:3], endpoint, 11)
    dense = states[:, :3].copy()
    row = {
        "gamma": float(gamma),
        "seed": int(seed),
        "success": bool(success),
        "collision": False,
        "taskspace_violation": False,
        "steps": 10,
        "time_to_goal_s": 1.0 if success else None,
        "min_clearance_m": 0.2,
        "mean_feasible_fraction": 1.0,
        "minimum_feasible_fraction": 1.0,
        "all_infeasible": False,
        "all_infeasible_steps": 0,
        "minimum_controller_one_step_slack": 0.1,
        "minimum_online_one_step_slack": 0.1,
        "mean_plan_time_ms": 1.0,
        "mean_control_variation_mps2": 0.0,
        "mean_applied_control_variation_mps2": 0.0,
        "max_abs_control_mps2": 0.0,
        "max_abs_applied_control_mps2": 0.0,
        "peak_speed_mps": 0.0,
        "peak_vertical_speed_mps": 0.0,
        "deployment_speed_compatible": True,
        "fraction_internal_states_below_z_bias_plane": 0.0,
        "mean_internal_z_m": float(env.start[2]),
    }
    arrays = {
        "states": states,
        "controls": controls,
        "executed_controls": controls.copy(),
        "dense_positions": dense,
        "gamma": np.asarray(gamma, np.float32),
        "seed": np.asarray(seed, np.int64),
    }
    return row, arrays


def _pattern_episode(env, controller, gamma, seed, rollout_dynamics):
    del controller, rollout_dynamics
    scene_index = int(seed) // 1009
    if np.isclose(gamma, 0.1):
        success = scene_index in {0, 2}
    else:
        success = scene_index in {1, 2}
    return _episode_row(env, gamma, seed, success)


def _always_success(env, controller, gamma, seed, rollout_dynamics):
    del controller, rollout_dynamics
    return _episode_row(env, gamma, seed, True)


def test_exact_quota_keeps_failed_attempts_and_loads_as_training_archive(
    tmp_path,
):
    output = tmp_path / "quota"
    manifest = collect_path_focused_success_quota(
        load_config(CONFIG),
        output,
        target_successes_per_gamma=2,
        max_scene_count=4,
        gammas=(0.1, 0.3),
        episode_runner=_pattern_episode,
        controller_factory=_fake_controller,
    )

    assert manifest["status"] == "COMPLETE_EXACT_SUCCESS_QUOTA"
    assert manifest["accepted_counts_by_gamma"] == {"0.1": 2, "0.3": 2}
    assert len(manifest["runs"]) == 4
    assert len(manifest["attempts"]) == 6
    assert sum(not row["accepted"] for row in manifest["attempts"]) == 2
    assert len(list((output / "attempt_shards").glob("*.json"))) == 6
    assert manifest["sampling_distribution"][
        "expert_success_used_for_scene_admission"
    ] is False

    contexts, plans, metadata, config = lab_reference_demo_windows(
        output,
        validate_archive=False,
        context_schema=LAB_RADIAL_VISUAL_SCHEMA,
    )
    assert contexts.shape[0] == plans.shape[0] == len(metadata) == 4
    assert set(config.data.gammas) == {0.1, 0.3}


def test_interrupted_collection_resumes_without_replaying_committed_cells(
    tmp_path,
):
    output = tmp_path / "resumed"
    calls = {"count": 0}

    def interrupting(env, controller, gamma, seed, rollout_dynamics):
        calls["count"] += 1
        if calls["count"] == 3:
            raise KeyboardInterrupt("synthetic interruption")
        return _always_success(
            env, controller, gamma, seed, rollout_dynamics,
        )

    with pytest.raises(KeyboardInterrupt):
        collect_path_focused_success_quota(
            load_config(CONFIG),
            output,
            target_successes_per_gamma=2,
            max_scene_count=3,
            gammas=(0.1, 0.3),
            episode_runner=interrupting,
            controller_factory=_fake_controller,
        )
    assert len(list((output / "attempt_shards").glob("*.json"))) == 2

    resumed_calls = {"count": 0}

    def resumed(env, controller, gamma, seed, rollout_dynamics):
        resumed_calls["count"] += 1
        return _always_success(
            env, controller, gamma, seed, rollout_dynamics,
        )

    manifest = collect_path_focused_success_quota(
        load_config(CONFIG),
        output,
        target_successes_per_gamma=2,
        max_scene_count=3,
        gammas=(0.1, 0.3),
        episode_runner=resumed,
        controller_factory=_fake_controller,
    )
    assert resumed_calls["count"] == 2
    assert len(manifest["attempts"]) == 4
    assert len(manifest["runs"]) == 4
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_quota_fails_closed_after_finite_geometry_budget(tmp_path):
    output = tmp_path / "failed"

    def first_only(env, controller, gamma, seed, rollout_dynamics):
        del controller, rollout_dynamics
        return _episode_row(env, gamma, seed, int(seed) // 1009 == 0)

    with pytest.raises(RuntimeError, match="success quota was not reached"):
        collect_path_focused_success_quota(
            load_config(CONFIG),
            output,
            target_successes_per_gamma=2,
            max_scene_count=2,
            gammas=(1.0,),
            episode_runner=first_only,
            controller_factory=_fake_controller,
        )
    failure = json.loads((output / "FAILED_collection.json").read_text())
    assert failure["status"] == "FAILED_MAX_SCENE_BUDGET"
    assert failure["accepted_counts_by_gamma"] == {"1": 1}
    assert len(failure["attempts"]) == 2
    assert not (output / "manifest.json").exists()


def test_collection_override_is_explicit_and_validated():
    config = path_focused_collection_config(
        load_config(CONFIG),
        gammas=(1.0,),
        episodes_per_gamma=50,
        max_attempts_per_gamma=50,
        centroid_gain=0.2,
        centroid_smooth=0.25,
        sigma_aniso=2.5,
    )
    assert config.data.gammas == (1.0,)
    assert config.data.episodes_per_gamma == 50
    assert config.safemppi.centroid_gain == 0.2
    assert config.safemppi.centroid_smooth == 0.25
    assert config.safemppi.sigma_aniso == 2.5
    assert config.raw["data"]["gammas"] == [1.0]

    with pytest.raises(ValueError, match="centroid_smooth"):
        path_focused_collection_config(
            load_config(CONFIG), centroid_smooth=1.1,
        )
