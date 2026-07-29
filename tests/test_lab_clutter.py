import json
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi.lab_clutter import (
    ClutterScene,
    DEFAULT_CYLINDER_COUNT,
    DEFAULT_CYLINDER_RADIUS_M,
    DEFAULT_MIN_SURFACE_GAP_M,
    LAB_BOUNDS,
    LAB_GOAL,
    LAB_START,
    collect_clutter_demos,
    config_for_scene,
    cylinder_scene_bank,
    fixed_lab_clutter_config,
    obstacle_scene_hash,
    start_goal_path_diagnostics,
    summarize_start_goal_path_diagnostics,
)
from safe_mppi.environment import ReferenceGovernor, TaskEnvironment
from safe_mppi.lab_reference_flow_task import lab_reference_demo_windows
from safe_mppi.lab_visual_flow import (
    LAB_VISUAL_SCHEMA,
    build_visual_context,
)
from scripts.pretrain_lab_reference_flow import (
    pretrained_sample_calibration,
    source_archive_digest,
    split_provenance,
    trajectory_split,
)


ROOT = Path(__file__).resolve().parents[1]
LAB_CONFIG = ROOT / "configs/lab_ball_pretrain.json"
CLUTTER_CONFIG = ROOT / "configs/lab_clutter_cylinders_pretrain.json"


def test_fixed_clutter_contract_uses_lab_geometry_and_no_z_bias():
    config = fixed_lab_clutter_config(load_config(LAB_CONFIG))
    assert np.allclose(config.taskspace.bounds, LAB_BOUNDS)
    assert np.allclose(config.taskspace.start, LAB_START)
    assert np.allclose(config.taskspace.goal, LAB_GOAL)
    assert config.safemppi.z_bias_weight == 0.0
    assert config.obstacles.spheres == ()
    assert config.obstacles.cylinders == ()
    assert config.raw["safety"] == {
        "safe_min": LAB_BOUNDS[:, 0].tolist(),
        "safe_max": LAB_BOUNDS[:, 1].tolist(),
    }


def test_cylinder_scene_bank_is_deterministic_and_respects_margins():
    first = cylinder_scene_bank(8, seed=17)
    second = cylinder_scene_bank(8, seed=17)
    different = cylinder_scene_bank(8, seed=18)
    assert first == second
    assert [scene.scene_hash for scene in first] != [
        scene.scene_hash for scene in different
    ]

    for scene in first:
        cylinders = np.asarray(scene.cylinders)
        assert cylinders.shape == (DEFAULT_CYLINDER_COUNT, 3)
        assert np.allclose(cylinders[:, 2], DEFAULT_CYLINDER_RADIUS_M)
        assert np.all(
            cylinders[:, :2] - cylinders[:, 2, None]
            >= LAB_BOUNDS[:2, 0] - 1.0e-12
        )
        assert np.all(
            cylinders[:, :2] + cylinders[:, 2, None]
            <= LAB_BOUNDS[:2, 1] + 1.0e-12
        )
        for index, cylinder in enumerate(cylinders):
            for other in cylinders[index + 1:]:
                surface_gap = (
                    np.linalg.norm(cylinder[:2] - other[:2])
                    - cylinder[2] - other[2]
                )
                assert surface_gap >= DEFAULT_MIN_SURFACE_GAP_M - 1.0e-12
            for endpoint in (LAB_START, LAB_GOAL):
                surface_gap = (
                    np.linalg.norm(cylinder[:2] - endpoint[:2])
                    - cylinder[2]
                )
                assert surface_gap >= DEFAULT_MIN_SURFACE_GAP_M - 1.0e-12


def test_scene_hash_covers_geometry_not_scene_number():
    cylinders = ((-1.0, 0.0, 0.1), (0.0, 0.0, 0.1))
    value = obstacle_scene_hash(cylinders=cylinders)
    assert len(value) == 64
    assert value == obstacle_scene_hash(cylinders=cylinders)
    assert value != obstacle_scene_hash(
        cylinders=((-1.0, 0.0, 0.1), (0.1, 0.0, 0.1)),
    )


def test_start_goal_path_diagnostics_are_measurements_not_filters():
    def scene_at(index, y):
        cylinders = ((1.0, y, 0.1),)
        return ClutterScene(
            index=index,
            seed=index,
            spheres=(),
            cylinders=cylinders,
            scene_hash=obstacle_scene_hash(cylinders=cylinders),
        )

    scenes = (
        scene_at(0, 0.05),
        scene_at(1, 0.35),
        scene_at(2, 0.50),
    )
    rows = [
        start_goal_path_diagnostics(
            scene,
            start=np.asarray([0.0, 0.0, 0.0]),
            goal=np.asarray([2.0, 0.0, 0.0]),
            soft_clearance_target_m=0.3,
        )
        for scene in scenes
    ]
    assert np.isclose(rows[0]["minimum_obstacle_surface_distance_m"], -0.05)
    assert rows[0]["modeled_hard_path_intersection"] is True
    assert rows[0]["within_soft_clearance_tube"] is True
    assert rows[1]["modeled_hard_path_intersection"] is False
    assert rows[1]["within_soft_clearance_tube"] is True
    assert rows[2]["modeled_hard_path_intersection"] is False
    assert rows[2]["within_soft_clearance_tube"] is False
    assert all(row["used_for_scene_filtering"] is False for row in rows)

    summary = summarize_start_goal_path_diagnostics(rows)
    assert summary["scene_count"] == len(scenes)
    assert summary["modeled_hard_path_intersection_scene_count"] == 1
    assert summary["within_soft_clearance_tube_scene_count"] == 2
    assert summary["used_for_scene_filtering"] is False


def _successful_fake_episode(env, controller, gamma, seed, rollout_dynamics):
    del controller, rollout_dynamics
    state = env.start.copy()
    next_state = state.copy()
    next_state[:3] += np.array([0.01, -0.01, 0.0], np.float32)
    row = {
        "gamma": float(gamma),
        "seed": int(seed),
        "success": True,
        "collision": False,
        "taskspace_violation": False,
        "steps": 1,
        "time_to_goal_s": 0.1,
        "min_clearance_m": 0.25,
        "mean_feasible_fraction": 1.0,
        "minimum_feasible_fraction": 1.0,
        "all_infeasible": False,
        "all_infeasible_steps": 0,
        "minimum_controller_one_step_slack": 0.1,
        "minimum_online_one_step_slack": 0.1,
        "mean_plan_time_ms": 1.0,
        "mean_control_variation_mps2": 0.0,
        "mean_applied_control_variation_mps2": 0.0,
        "max_abs_control_mps2": 0.1,
        "max_abs_applied_control_mps2": 0.04,
        "peak_speed_mps": 0.1,
        "peak_vertical_speed_mps": 0.0,
        "deployment_speed_compatible": True,
        "fraction_internal_states_below_z_bias_plane": 0.0,
        "mean_internal_z_m": float(state[2]),
    }
    arrays = {
        "states": np.stack([state, next_state]).astype(np.float32),
        "controls": np.zeros((1, 3), np.float32),
        "executed_controls": np.zeros((1, 3), np.float32),
        "dense_positions": np.stack([state[:3], next_state[:3]]),
        "gamma": np.asarray(gamma, np.float32),
        "seed": np.asarray(seed, np.int64),
    }
    return row, arrays


def _fake_controller(*args, **kwargs):
    del args, kwargs
    return object()


def test_collection_pairs_scene_bank_across_gamma_and_saves_obstacles(tmp_path):
    manifest = collect_clutter_demos(
        load_config(LAB_CONFIG),
        tmp_path / "archive",
        scene_count=2,
        domain_seed=23,
        max_rollout_attempts_per_scene=1,
        episode_runner=_successful_fake_episode,
        controller_factory=_fake_controller,
    )
    assert manifest["z_bias_weight"] == 0.0
    assert manifest["status"] == "COMPLETE"
    assert manifest["max_candidate_scenes"] == 8
    assert manifest["candidate_scenes_evaluated"] == 2
    assert manifest["rejected_scene_count"] == 0
    assert manifest["sampling_distribution"]["unconditioned_uniform"] is False
    assert manifest["scene_bank"]["shared_across_gamma"] is True
    assert len(manifest["runs"]) == 8
    path_summary = manifest["scene_bank"]["start_goal_path_summary"]
    scene_rows = manifest["scene_bank"]["scenes"]
    assert path_summary["scene_count"] == len(scene_rows)
    assert path_summary["soft_clearance_target_m"] == 0.3
    assert path_summary["modeled_hard_path_intersection_scene_count"] == sum(
        row["start_goal_path_diagnostics"]["modeled_hard_path_intersection"]
        for row in scene_rows
    )
    assert path_summary["within_soft_clearance_tube_scene_count"] == sum(
        row["start_goal_path_diagnostics"]["within_soft_clearance_tube"]
        for row in scene_rows
    )
    assert path_summary["used_for_scene_filtering"] is False
    expected_ids = {
        scene["scene_id"] for scene in scene_rows
    }
    for gamma in (0.1, 0.3, 0.5, 1.0):
        assert {
            row["scene_id"] for row in manifest["runs"]
            if np.isclose(row["gamma"], gamma)
        } == expected_ids

    archive = tmp_path / "archive"
    resolved = json.loads((archive / "resolved_config.json").read_text())
    assert resolved["safemppi"]["z_bias_weight"] == 0.0
    assert resolved["obstacles"] == {"spheres": [], "cylinders": []}
    for row in manifest["runs"]:
        data = np.load(archive / row["file"])
        assert data["spheres"].shape == (0, 4)
        assert data["cylinders"].shape == (3, 3)
        assert data["scene_id"].item() == row["scene_id"]
        assert data["scene_hash"].item() == row["scene_hash"]
        assert obstacle_scene_hash(
            data["spheres"], data["cylinders"],
        ) == row["scene_hash"]


def test_collection_rejects_partial_gamma_candidate_without_writing_npz(
    tmp_path,
):
    candidates = cylinder_scene_bank(3, seed=23)
    rejected = candidates[0]

    def episode_runner(env, controller, gamma, seed, rollout_dynamics):
        row, arrays = _successful_fake_episode(
            env, controller, gamma, seed, rollout_dynamics,
        )
        scene_hash = obstacle_scene_hash(env.spheres, env.cylinders)
        if scene_hash == rejected.scene_hash and np.isclose(gamma, 0.3):
            row["success"] = False
            row["time_to_goal_s"] = None
        return row, arrays

    output = tmp_path / "conditioned_archive"
    manifest = collect_clutter_demos(
        load_config(LAB_CONFIG),
        output,
        scene_count=2,
        domain_seed=23,
        max_candidate_scenes=3,
        max_rollout_attempts_per_scene=1,
        episode_runner=episode_runner,
        controller_factory=_fake_controller,
    )
    accepted_ids = {
        row["scene_id"] for row in manifest["scene_bank"]["scenes"]
    }
    assert accepted_ids == {candidates[1].scene_id, candidates[2].scene_id}
    assert len(manifest["runs"]) == 8
    assert manifest["candidate_scenes_evaluated"] == 3
    assert manifest["rejected_scene_count"] == 1
    rejection = manifest["scene_bank"]["rejected_scenes"][0]
    assert rejection["scene_hash"] == rejected.scene_hash
    assert rejection["rejection_reason"] == (
        "no_accepted_trajectory_for_configured_gamma_within_attempt_limit"
    )
    assert np.isclose(rejection["failed_gamma"], 0.3)
    assert rejection["successful_gammas_before_rejection"] == [0.1]
    assert rejection["unevaluated_gammas"] == [0.5, 1.0]

    rejected_attempts = [
        row for row in manifest["attempts"]
        if row["scene_hash"] == rejected.scene_hash
    ]
    assert len(rejected_attempts) == 2
    assert any(row["trajectory_accepted"] for row in rejected_attempts)
    assert all(row["candidate_admitted"] is False for row in rejected_attempts)
    assert all(row["accepted"] is False for row in rejected_attempts)
    assert all(row["file"] is None for row in rejected_attempts)
    assert not any(
        rejected.scene_id in path.name for path in output.glob("*.npz")
    )
    assert len(list(output.glob("*.npz"))) == 8
    assert manifest["sampling_distribution"]["effective"] == (
        "uniform proposals conditioned on observed expert success within "
        "the finite retry budget for every configured gamma"
    )


def test_collection_failure_keeps_rejected_attempt_provenance(tmp_path):
    def failed_episode(env, controller, gamma, seed, rollout_dynamics):
        row, arrays = _successful_fake_episode(
            env, controller, gamma, seed, rollout_dynamics,
        )
        row["success"] = False
        row["time_to_goal_s"] = None
        return row, arrays

    output = tmp_path / "failed_archive"
    with pytest.raises(RuntimeError, match="admitted 0/1 scenes"):
        collect_clutter_demos(
            load_config(LAB_CONFIG),
            output,
            scene_count=1,
            max_candidate_scenes=1,
            max_rollout_attempts_per_scene=1,
            episode_runner=failed_episode,
            controller_factory=_fake_controller,
        )
    failure = json.loads((output / "FAILED_collection.json").read_text())
    assert failure["status"] == "FAILED_MAX_CANDIDATE_SCENES"
    assert failure["requested_scene_count"] == 1
    assert failure["candidate_scenes_evaluated"] == 1
    assert failure["admitted_scene_count"] == 0
    assert failure["rejected_scene_count"] == 1
    assert len(failure["attempts"]) == 1
    assert failure["scene_bank"]["rejected_scenes"][0][
        "rejection_reason"
    ] == "no_accepted_trajectory_for_configured_gamma_within_attempt_limit"
    assert not list(output.glob("*.npz"))
    assert not (output / "manifest.json").exists()


def test_visual_loader_reconstructs_each_npz_scene(tmp_path):
    template = fixed_lab_clutter_config(load_config(LAB_CONFIG))
    scene = cylinder_scene_bank(1, seed=47)[0]
    config = config_for_scene(template, scene)
    env = TaskEnvironment(config)
    governor = ReferenceGovernor(config.safemppi)
    controls = np.zeros((10, 3), np.float32)
    states = [env.start.copy()]
    for control in controls:
        state, _, _ = governor.step(states[-1], control)
        states.append(state)

    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "resolved_config.json").write_text(
        json.dumps(template.raw) + "\n"
    )
    filename = "run.npz"
    np.savez_compressed(
        archive / filename,
        states=np.asarray(states, np.float32),
        controls=controls,
        spheres=np.asarray(scene.spheres, np.float32).reshape(-1, 4),
        cylinders=np.asarray(scene.cylinders, np.float32).reshape(-1, 3),
        scene_hash=np.asarray(scene.scene_hash),
    )
    (archive / "manifest.json").write_text(json.dumps({
        "runs": [{
            "accepted": True,
            "file": filename,
            "gamma": 0.3,
            "seed": 1,
            "scene_id": scene.scene_id,
            "scene_hash": scene.scene_hash,
        }],
    }) + "\n")

    contexts, plans, metadata, _ = lab_reference_demo_windows(
        archive,
        context_schema=LAB_VISUAL_SCHEMA,
    )
    assert contexts.shape == (1, 6919)
    assert plans.shape == (1, 10, 3)
    assert metadata[0]["scene_hash"] == scene.scene_hash
    np.testing.assert_array_equal(
        contexts[0],
        build_visual_context(env, env.start, 0.3),
    )


def test_collection_defaults_come_from_scene_randomization(tmp_path):
    manifest = collect_clutter_demos(
        load_config(CLUTTER_CONFIG),
        tmp_path / "configured_archive",
        scene_count=2,
        max_rollout_attempts_per_scene=1,
        episode_runner=_successful_fake_episode,
        controller_factory=_fake_controller,
    )
    contract = json.loads(
        (tmp_path / "configured_archive" / "resolved_config.json").read_text()
    )["domain_randomization"]
    assert contract["domain_seed"] == 0
    assert contract["cylinder_count"] == 3
    assert contract["cylinder_radius_m"] == DEFAULT_CYLINDER_RADIUS_M
    assert contract["min_surface_gap_m"] == 0.2
    assert contract["minimum_start_surface_clearance_m"] == 0.5
    assert contract["minimum_goal_surface_clearance_m"] == 0.5
    assert contract["boundary_surface_gap_m"] == 0.1
    for scene in manifest["scene_bank"]["scenes"]:
        cylinders = np.asarray(scene["cylinders"])
        for cylinder in cylinders:
            assert (
                np.linalg.norm(cylinder[:2] - LAB_START[:2]) - cylinder[2]
                >= 0.5 - 1.0e-12
            )
            assert (
                np.linalg.norm(cylinder[:2] - LAB_GOAL[:2]) - cylinder[2]
                >= 0.5 - 1.0e-12
            )
            assert np.all(
                cylinder[:2] - cylinder[2]
                >= LAB_BOUNDS[:2, 0] + 0.1 - 1.0e-12
            )
            assert np.all(
                cylinder[:2] + cylinder[2]
                <= LAB_BOUNDS[:2, 1] - 0.1 + 1.0e-12
            )


def test_pretrain_provenance_records_exact_split_and_archive_digest(tmp_path):
    metadata = [
        {
            "gamma": gamma,
            "seed": scene_index,
            "scene_id": f"scene-{scene_index}",
            "scene_hash": scene_hash,
            "t": 0,
        }
        for scene_index, scene_hash in enumerate(("a", "b", "c"))
        for gamma in (0.1, 0.3, 0.5, 1.0)
    ]
    training_ids, validation_ids = trajectory_split(metadata, seed=13)
    provenance = split_provenance(
        metadata, training_ids, validation_ids, seed=13,
    )
    training_hashes = set(provenance["training_scene_hashes"])
    validation_hashes = set(provenance["validation_scene_hashes"])
    assert provenance["split_seed"] == 13
    assert provenance["split_unit"] == "scene_sha256_across_all_gamma"
    assert provenance["randomized_scene_count"] == 3
    assert training_hashes.isdisjoint(validation_hashes)
    assert training_hashes | validation_hashes == {"a", "b", "c"}

    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "resolved_config.json").write_text("{}\n")
    (archive / "run.npz").write_bytes(b"first")
    (archive / "manifest.json").write_text(json.dumps({
        "runs": [{"file": "run.npz"}],
    }) + "\n")
    first = source_archive_digest(archive)
    assert first["algorithm"] == "sha256"
    assert first["file_count"] == 3
    (archive / "unrelated.txt").write_text("not part of the archive contract")
    assert source_archive_digest(archive) == first
    (archive / "run.npz").write_bytes(b"second")
    assert source_archive_digest(archive)["sha256"] != first["sha256"]


def test_rbf_calibration_uses_balanced_pretrained_policy_samples():
    class FakePolicy:
        def sample_with_base(self, context, count, generator, base_std=1.0):
            del context
            base = torch.randn(count, 2, generator=generator) * base_std
            return base + 3.0, base

        def embed(self, context, candidates, flow_time=0.9, base=None):
            del context, flow_time
            assert base is not None
            return candidates + 0.1 * base

    contexts = torch.zeros(80, 3)
    metadata = [
        {"gamma": gamma, "scene_hash": f"scene-{index // 4}"}
        for index in range(20)
        for gamma in (0.1, 0.3, 0.5, 1.0)
    ]
    features, selected = pretrained_sample_calibration(
        FakePolicy(),
        contexts,
        metadata,
        torch.arange(80),
        seed=7,
        count=20,
    )
    assert features.shape == (20, 2)
    assert not torch.allclose(features, torch.zeros_like(features))
    selected_gammas = [metadata[index]["gamma"] for index in selected]
    assert {
        gamma: selected_gammas.count(gamma)
        for gamma in (0.1, 0.3, 0.5, 1.0)
    } == {0.1: 5, 0.3: 5, 0.5: 5, 1.0: 5}
