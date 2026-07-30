import json
from pathlib import Path

import numpy as np

from safe_mppi.config import load_config
from safe_mppi.lab_clutter import LAB_BOUNDS, LAB_GOAL, LAB_START
from safe_mppi.path_focused_clutter import path_focused_scene_bank
from safe_mppi.path_focused_collection import (
    collect_path_focused_clutter_demos,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_cylinders_path_v2.json"
SPHERE_CONFIG = ROOT / "configs/lab_clutter_spheres_path_v2.json"
SIGMA_PERP_VALUES = (0.20, 0.35, 0.50, 0.65)


def _config_with_sigma_perp(value: float):
    config = load_config(CONFIG)
    raw = dict(config.raw)
    raw["scene_randomization"] = dict(raw["scene_randomization"])
    raw["scene_randomization"]["transverse_std_m"] = float(value)
    return type(config)(
        config.taskspace,
        config.obstacles,
        config.safemppi,
        config.data,
        raw,
    )


def _transverse_offsets(scenes) -> np.ndarray:
    start = LAB_START[:2]
    direction = LAB_GOAL[:2] - start
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    return np.concatenate([
        (np.asarray(scene.cylinders)[:, :2] - start[None]) @ normal
        for scene in scenes
    ])


def test_path_focused_bank_is_deterministic_and_obeys_geometry_contract():
    config = _config_with_sigma_perp(0.35)
    first = path_focused_scene_bank(config, 64, seed=17)
    repeated = path_focused_scene_bank(config, 64, seed=17)
    different = path_focused_scene_bank(config, 64, seed=18)

    assert first == repeated
    assert [scene.scene_hash for scene in first] != [
        scene.scene_hash for scene in different
    ]

    spec = config.raw["scene_randomization"]
    radius = float(spec["radius_m"])
    minimum_gap = float(spec["minimum_obstacle_surface_gap_m"])
    wall_gap = float(spec["minimum_taskspace_wall_surface_clearance_m"])
    endpoint_gap = float(spec["minimum_start_surface_clearance_m"])
    for scene in first:
        cylinders = np.asarray(scene.cylinders)
        assert spec["count_min"] <= len(cylinders) <= spec["count_max"]
        np.testing.assert_allclose(cylinders[:, 2], radius)
        assert np.all(
            cylinders[:, :2] - radius
            >= LAB_BOUNDS[:2, 0] + wall_gap - 2.0e-6
        )
        assert np.all(
            cylinders[:, :2] + radius
            <= LAB_BOUNDS[:2, 1] - wall_gap + 2.0e-6
        )
        for index, cylinder in enumerate(cylinders):
            for other in cylinders[index + 1:]:
                assert (
                    np.linalg.norm(cylinder[:2] - other[:2])
                    - 2.0 * radius
                    >= minimum_gap - 2.0e-6
                )
            for endpoint in (LAB_START, LAB_GOAL):
                assert (
                    np.linalg.norm(cylinder[:2] - endpoint[:2])
                    - radius
                    >= endpoint_gap - 2.0e-6
                )


def test_sigma_perp_monotonically_increases_same_seed_transverse_spread():
    rms_values = []
    hashes = []
    for sigma_perp in SIGMA_PERP_VALUES:
        scenes = path_focused_scene_bank(
            _config_with_sigma_perp(sigma_perp),
            256,
            seed=0,
        )
        offsets = _transverse_offsets(scenes)
        rms_values.append(float(np.sqrt(np.mean(offsets * offsets))))
        hashes.append([scene.scene_hash for scene in scenes])

    assert np.all(np.diff(rms_values) > 0.10)
    assert rms_values[0] < 0.25
    assert rms_values[-1] > 0.60
    assert all(hashes[0] != other for other in hashes[1:])


def test_path_focused_spheres_cover_declared_counts_and_zero_extra_gaps():
    config = load_config(SPHERE_CONFIG)
    scenes = path_focused_scene_bank(config, 128, seed=23)
    randomization = config.raw["scene_randomization"]
    radius = float(randomization["radius_m"])
    counts = set()
    for scene in scenes:
        spheres = np.asarray(scene.spheres, np.float64)
        counts.add(len(spheres))
        np.testing.assert_allclose(spheres[:, 3], radius)
        assert np.all(
            spheres[:, :3] - radius >= LAB_BOUNDS[:, 0] - 2.0e-6
        )
        assert np.all(
            spheres[:, :3] + radius <= LAB_BOUNDS[:, 1] + 2.0e-6
        )
        for index, sphere in enumerate(spheres):
            for other in spheres[index + 1:]:
                assert (
                    np.linalg.norm(sphere[:3] - other[:3])
                    >= sphere[3] + other[3] - 2.0e-6
                )
    assert counts == {3, 4, 5, 6}


def _fake_controller(*args, **kwargs):
    del args, kwargs
    return object()


def _fake_episode(env, controller, gamma, seed, rollout_dynamics):
    del controller, rollout_dynamics
    success = not (seed == 0 and np.isclose(gamma, 1.0))
    path = np.linspace(env.start[:3], env.goal, 11, dtype=np.float32)
    states = np.zeros((2, 6), np.float32)
    states[0] = env.start
    states[1, :3] = env.goal if success else path[5]
    row = {
        "gamma": float(gamma),
        "seed": int(seed),
        "success": bool(success),
        "collision": False,
        "taskspace_violation": False,
        "steps": 1,
        "time_to_goal_s": 0.1 if success else None,
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
        "controls": np.zeros((1, 3), np.float32),
        "executed_controls": np.zeros((1, 3), np.float32),
        "dense_positions": path,
        "gamma": np.asarray(gamma, np.float32),
        "seed": np.asarray(seed, np.int64),
    }
    return row, arrays


def test_collector_keeps_fixed_bank_and_failed_scene_gamma_cells(tmp_path):
    output = tmp_path / "archive"
    manifest = collect_path_focused_clutter_demos(
        load_config(CONFIG),
        output,
        scene_count=3,
        domain_seed=0,
        rollout_seed_start=0,
        transverse_std_m=0.50,
        episode_runner=_fake_episode,
        controller_factory=_fake_controller,
    )

    gammas = tuple(map(float, load_config(CONFIG).data.gammas))
    assert manifest["status"] == "COMPLETE_GEOMETRY_BANK_EVALUATED"
    assert manifest["evaluated_scene_count"] == 3
    assert manifest["admitted_scene_count"] == 3
    assert manifest["rejected_scene_count"] == 0
    assert manifest["sampling_distribution"]["unconditioned_geometry"] is True
    assert (
        manifest["sampling_distribution"][
            "expert_success_used_for_scene_admission"
        ]
        is False
    )
    assert len(manifest["scene_bank"]["scenes"]) == 3
    assert len(manifest["attempts"]) == 3 * len(gammas)
    assert len(manifest["runs"]) == 3 * len(gammas) - 1

    failed = [
        row for row in manifest["attempts"]
        if row["seed"] == 0 and np.isclose(row["gamma"], 1.0)
    ]
    assert len(failed) == 1
    assert failed[0]["success"] is False
    assert failed[0]["accepted"] is False
    assert failed[0]["file"] is None
    assert failed[0]["scene_hash"] in {
        row["scene_hash"] for row in manifest["scene_bank"]["scenes"]
    }

    for scene_index in range(3):
        scene_rows = [
            row for row in manifest["attempts"]
            if row["scene_index"] == scene_index
        ]
        assert len(scene_rows) == len(gammas)
        assert len({row["seed"] for row in scene_rows}) == 1
        assert {
            round(float(row["gamma"]), 6) for row in scene_rows
        } == {round(gamma, 6) for gamma in gammas}

    resolved = json.loads((output / "resolved_config.json").read_text())
    assert resolved["data"]["episodes_per_gamma"] == 3
    assert resolved["data"]["max_attempts_per_gamma"] >= 3
    assert resolved["scene_randomization"]["transverse_std_m"] == 0.50
    assert (
        resolved["scene_randomization"]["expert_rollouts_per_scene_gamma"]
        == 1
    )
    assert len(list(output.glob("*.npz"))) == len(manifest["runs"])
