import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_ball_expansion as expansion_runner
from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_SCENE_DIM,
    LAB_CLUTTER_VERIFIER_SUFFIX_DIM,
    LabClutterExpansionPolicyAdapter,
    LabClutterSphereExpansionTask,
    PathFocusedVariableSphereScene,
    RandomThreeSphereScene,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_visual_flow import (
    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_VISUAL_HISTORY_LENGTH,
    LAB_VISUAL_HISTORY_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    LabNonuniformRadialHistoryFlowPolicy,
    LabVisualHistoryFlowPolicy,
    LabVisualFlowPolicy,
    build_visual_context,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_spheres_ood.json"
CYLINDER_CONFIG = ROOT / "configs/lab_clutter_cylinders_pretrain.json"
MIDPOINT_SPHERE_CONFIG = (
    ROOT / "configs/lab_clutter_spheres_path_midpoint_uniform_v2.json"
)


def _task_and_policy(scene_spec=None):
    config = load_config(CONFIG)
    policy = LabClutterExpansionPolicyAdapter(
        LabVisualFlowPolicy(
            hidden=8,
            representation_dim=4,
            grid_token_dim=4,
            control_limit=config.safemppi.demo_u_max,
            nfe=2,
        )
    )
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_VISUAL_SCHEMA,
        tight_corridor=True,
        scene_spec=scene_spec,
    )
    return config, task, policy


def test_lab_default_rbf_calibration_samples_from_pretrained_policy(monkeypatch):
    sentinel = torch.tensor([[1.0, 2.0]])
    calls = []

    def sampled(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        expansion_runner, "_lab_pretrained_phi_calibration", sampled,
    )
    monkeypatch.setattr(
        expansion_runner,
        "_pretrained_phi_calibration",
        lambda *args, **kwargs: pytest.fail("generic calibration was used"),
    )
    result = expansion_runner._learned_phi_calibration(
        object(),
        Path("/unused"),
        {"kind": "lab raw-command reference-flow pretraining"},
        object(),
        19,
        lab_profile=True,
        flow_base_std=1.0,
        paired_noised_representation=False,
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][1]["flow_base_std"] == 1.0
    assert calls[0][1]["paired_noised_representation"] is False


def test_lab_rbf_calibration_uses_fifty_sampled_plans(monkeypatch, tmp_path):
    config = load_config(CONFIG)
    gammas = list(map(float, config.data.gammas))
    contexts = np.zeros((80, 5), np.float32)
    expert_targets = np.full((80, 10, 3), 999.0, np.float32)
    metadata = [
        {"gamma": gamma}
        for gamma in gammas
        for _ in range(20)
    ]

    class FakePolicy(torch.nn.Module):
        context_schema = "fake_lab"

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.sample_calls = 0
            self.embedded = None

        def sample_with_base(self, context, count, generator, base_std):
            del context, generator
            assert count == 1
            assert base_std == 1.0
            self.sample_calls += 1
            plan = torch.full((1, 10, 3), float(self.sample_calls))
            return plan, torch.zeros_like(plan)

        def embed(self, contexts, candidates, base=None):
            del contexts
            assert base is None
            self.embedded = candidates.detach().clone()
            return candidates.reshape(len(candidates), -1)[:, :4]

    monkeypatch.setattr(
        expansion_runner,
        "_lab_source_demo_dir",
        lambda *args: tmp_path,
    )
    monkeypatch.setattr(
        expansion_runner,
        "lab_reference_demo_windows",
        lambda *args, **kwargs: (
            contexts, expert_targets, metadata, config,
        ),
    )
    policy = FakePolicy()
    features = expansion_runner._lab_pretrained_phi_calibration(
        policy,
        tmp_path,
        {},
        config,
        7,
        flow_base_std=1.0,
        paired_noised_representation=False,
    )

    assert policy.sample_calls == 50
    assert features.shape == (50, 4)
    assert policy.embedded is not None
    assert not bool((policy.embedded == 999.0).any())


def test_lab_randomization_fails_explicitly_unless_it_is_three_spheres():
    assert expansion_runner._lab_clutter_profile(load_config(CONFIG))
    with pytest.raises(ValueError, match="exactly three spheres"):
        expansion_runner._lab_clutter_profile(load_config(CYLINDER_CONFIG))


def test_midpoint_uniform_spheres_route_to_path_focused_expansion():
    config = load_config(MIDPOINT_SPHERE_CONFIG)
    assert expansion_runner._lab_clutter_profile(config)
    spec = sphere_scene_spec_from_config(config)
    assert isinstance(spec, PathFocusedVariableSphereScene)
    assert spec.scene_schema == (
        "lab_path_focused_midpoint_uniform_variable_spheres_v2"
    )


def test_lab_setup_failure_writes_provenance_without_overwriting(tmp_path):
    output = tmp_path / "failed"
    expansion_runner._record_lab_setup_failure(
        output,
        output_was_unsafe=False,
        stage="lab_rbf_calibration",
        error=ValueError("bad calibration"),
    )
    payload = json.loads((output / "FAILED.json").read_text())
    assert payload == {
        "status": "EXPANSION_FAILED_CLOSED",
        "stage": "lab_rbf_calibration",
        "error_type": "ValueError",
        "error": "bad calibration",
    }

    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "keep.txt").write_text("keep")
    expansion_runner._record_lab_setup_failure(
        protected,
        output_was_unsafe=True,
        stage="lab_task_setup",
        error=RuntimeError("do not write"),
    )
    assert not (protected / "FAILED.json").exists()


def test_reset_deterministically_samples_three_valid_spheres():
    config, task, _ = _task_and_policy()
    np.testing.assert_allclose(
        config.taskspace.bounds,
        np.asarray([
            [-2.5, 1.3],
            [-1.7, 1.8],
            [0.4, 2.0],
        ]),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        config.taskspace.start,
        np.asarray([-2.1, 1.5, 0.9, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        config.taskspace.goal,
        np.asarray([0.7, -1.5, 0.9]),
    )
    first = task.reset(0.3, 0, 123)
    repeated = task.reset(1.0, 99, 123)
    other = task.reset(0.3, 0, 124)

    np.testing.assert_array_equal(first["spheres"], repeated["spheres"])
    assert first["scene_hash"] == repeated["scene_hash"]
    assert not np.array_equal(first["spheres"], other["spheres"])
    assert first["scene_hash"] != other["scene_hash"]
    assert task.scene_ledger[0] == {
        "reset_index": 0,
        "gamma": 0.3,
        "episode": 0,
        **task.scene_metadata(first),
    }
    np.testing.assert_allclose(
        first["x"], config.taskspace.start, atol=1.0e-7, rtol=0.0,
    )

    spheres = first["spheres"]
    spec = task.scene_spec
    randomization = config.raw["scene_randomization"]
    assert spec.radius == randomization["radius_m"]
    assert (
        spec.minimum_surface_margin
        == randomization["minimum_obstacle_surface_gap_m"]
    )
    assert (
        spec.boundary_surface_margin
        == randomization["minimum_taskspace_wall_surface_clearance_m"]
    )
    assert (
        spec.endpoint_margin
        == randomization["minimum_start_surface_clearance_m"]
        == randomization["minimum_goal_surface_clearance_m"]
    )
    assert spheres.shape == (3, 4)
    assert np.allclose(spheres[:, 3], spec.radius)
    assert np.all(
        spheres[:, :3] - spheres[:, 3, None]
        >= (
            config.taskspace.bounds[:, 0][None]
            + spec.boundary_surface_margin
            - 1.0e-6
        )
    )
    assert np.all(
        spheres[:, :3] + spheres[:, 3, None]
        <= (
            config.taskspace.bounds[:, 1][None]
            - spec.boundary_surface_margin
            + 1.0e-6
        )
    )
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            distance = np.linalg.norm(
                spheres[first_index, :3] - spheres[second_index, :3]
            )
            assert (
                distance
                >= 2.0 * spec.radius + spec.minimum_surface_margin - 2.0e-6
            )


def test_context_packs_exact_scene_after_governor_and_adapter_strips_it():
    _, task, policy = _task_and_policy()
    state = task.reset(0.3, 0, 17)
    state["previous_applied"][:] = (0.1, -0.2, 0.05)
    state["previous_raw"][:] = (-0.3, 0.2, 0.1)
    context = task.context(state, 0.3)

    assert context.shape == (
        LAB_VISUAL_PACKED_DIM + LAB_CLUTTER_VERIFIER_SUFFIX_DIM,
    )
    learned = policy._policy_context(context)
    assert learned.shape == (LAB_VISUAL_PACKED_DIM,)
    scene_env = task._environment(state["spheres"])
    np.testing.assert_array_equal(
        learned.numpy(),
        build_visual_context(scene_env, state["x"], 0.3),
    )
    suffix = context.numpy()[LAB_VISUAL_PACKED_DIM:]
    np.testing.assert_array_equal(suffix[:3], state["previous_applied"])
    np.testing.assert_array_equal(suffix[3:6], state["previous_raw"])
    np.testing.assert_array_equal(
        suffix[6:],
        state["spheres"].reshape(LAB_CLUTTER_SCENE_DIM),
    )
    assert policy.context_dim == len(context)


def test_gru_clutter_context_preserves_history_and_dynamic_suffix():
    config = load_config(CONFIG)
    base = LabVisualHistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=4,
        history_token_dim=3,
        control_limit=config.safemppi.demo_u_max,
        nfe=2,
    )
    policy = LabClutterExpansionPolicyAdapter(
        base,
        freeze_history_encoder=True,
    )
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_VISUAL_HISTORY_SCHEMA,
        tight_corridor=True,
    )
    state = task.reset(0.3, 0, 17)
    context = task.context(state, 0.3)
    assert context.shape == (
        LAB_VISUAL_HISTORY_PACKED_DIM
        + LAB_CLUTTER_VERIFIER_SUFFIX_DIM,
    )
    assert policy._policy_context(context).shape == (
        LAB_VISUAL_HISTORY_PACKED_DIM,
    )
    history = context.numpy()[
        LAB_VISUAL_PACKED_DIM:LAB_VISUAL_HISTORY_PACKED_DIM
    ].reshape(LAB_VISUAL_HISTORY_LENGTH, 4)
    assert not bool(history[:, 3].any())
    np.testing.assert_array_equal(
        task.scene_from_context(context), state["spheres"],
    )

    candidate = torch.zeros(10, 3)
    candidate[0] = torch.tensor([0.1, 0.05, -0.02])
    updated = task.advance(state, candidate)
    np.testing.assert_array_equal(
        updated["raw_history"][-1],
        np.asarray([0.1, 0.05, -0.02, 1.0], np.float32),
    )
    updated_context = task.context(updated, 0.3)
    np.testing.assert_array_equal(
        updated_context.numpy()[
            LAB_VISUAL_PACKED_DIM:LAB_VISUAL_HISTORY_PACKED_DIM
        ].reshape(LAB_VISUAL_HISTORY_LENGTH, 4),
        updated["raw_history"],
    )


def test_radial_gru_clutter_context_preserves_history_and_scene_suffix():
    config = load_config(CONFIG)
    base = LabNonuniformRadialHistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=32,
        control_limit=config.safemppi.demo_u_max,
        nfe=1,
        trunk_depth=3,
    )
    policy = LabClutterExpansionPolicyAdapter(
        base,
        freeze_history_encoder=True,
    )
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        tight_corridor=True,
    )
    state = task.reset(0.3, 0, 23)
    context = task.context(state, 0.3)
    assert context.shape == (
        LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM
        + LAB_CLUTTER_VERIFIER_SUFFIX_DIM,
    )
    assert policy._policy_context(context).shape == (
        LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    )
    history = context.numpy()[
        LAB_RADIAL_VISUAL_PACKED_DIM:
        LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM
    ].reshape(LAB_VISUAL_HISTORY_LENGTH, 4)
    assert not bool(history[:, 3].any())
    np.testing.assert_array_equal(
        task.scene_from_context(context),
        state["spheres"],
    )
    assert task.context_schema == LAB_RADIAL_VISUAL_HISTORY_SCHEMA
    assert base.context_schema == LAB_RADIAL_VISUAL_HISTORY_SCHEMA


def test_verifier_receives_all_three_spheres_from_packed_context(monkeypatch):
    _, task, _ = _task_and_policy()
    state = task.reset(0.3, 0, 29)
    context = task.context(state, 0.3)
    observed = {}

    def fake_certify(positions, spheres, cylinders, gamma, sensing_range, **kwargs):
        observed["positions"] = np.asarray(positions)
        observed["spheres"] = np.asarray(spheres)
        observed["cylinders"] = np.asarray(cylinders)
        return True, 0.25

    monkeypatch.setattr(
        "safe_mppi.lab_clutter_expansion.certify_window",
        fake_certify,
    )
    monkeypatch.setattr(
        "safe_mppi.lab_clutter_expansion.nominal_hp_chain_margins",
        lambda *args, **kwargs: np.asarray([0.1]),
    )
    result = task.verify(context, torch.zeros(1, 10, 3), 0.3)[0]

    np.testing.assert_array_equal(observed["spheres"], state["spheres"])
    assert observed["spheres"].shape == (3, 4)
    assert observed["cylinders"].shape == (0, 3)
    assert observed["positions"].shape == (11, 3)
    assert result.valid
    assert result.target_eligible
    assert result.margin == 0.25


def test_actual_full_polytope_verifies_all_three_spheres():
    _, task, _ = _task_and_policy()
    state = task.reset(0.3, 0, 31)
    result = task.verify(
        task.context(state, 0.3),
        torch.zeros(1, 10, 3),
        0.3,
    )[0]
    assert result.valid
    assert result.hp_eligible
    assert np.isfinite(result.margin)


def test_advance_and_terminal_use_episode_scene():
    _, task, _ = _task_and_policy()
    state = task.reset(0.3, 0, 41)
    state["x"][:3] = state["spheres"][0, :3]
    updated = task.advance(state, torch.zeros(10, 3))
    assert updated["collided"]
    assert task.terminal(updated) == "COLLISION"
    assert updated["scene_hash"] == state["scene_hash"]
    np.testing.assert_array_equal(updated["spheres"], state["spheres"])


def test_native_cost_is_symmetric_about_fixed_z_without_z_bias():
    config = load_config(CONFIG)
    symmetric_spheres = (
        (-1.4, -0.6, 0.9, 0.10),
        (-0.4, 1.2, 0.9, 0.10),
        (0.8, 0.6, 0.9, 0.10),
    )
    spec = RandomThreeSphereScene(
        radius=0.10,
        minimum_surface_margin=0.0,
        endpoint_margin=0.0,
        boundary_surface_margin=0.0,
    )
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_VISUAL_SCHEMA,
        scene_spec=spec,
    )
    env = task._environment(np.asarray(symmetric_spheres, np.float32))
    plan = np.zeros((10, 3), np.float32)
    above = np.zeros((11, 6), np.float32)
    below = np.zeros((11, 6), np.float32)
    above[:, :3] = (-2.0, 1.4, 1.0)
    below[:, :3] = (-2.0, 1.4, 0.8)
    above_cost = task._native_cost(env, above, plan, np.zeros(3))
    below_cost = task._native_cost(env, below, plan, np.zeros(3))
    assert above_cost == pytest.approx(below_cost, abs=1.0e-6)


def test_clutter_task_rejects_single_sphere_verifier_and_raw_context():
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="full_polytope"):
        LabClutterSphereExpansionTask(
            config,
            context_schema=LAB_VISUAL_SCHEMA,
            verifier_mode="single_sphere_affine",
        )
    with pytest.raises(ValueError, match="visual"):
        LabClutterSphereExpansionTask(
            config,
            context_schema="lab_raw10_v1",
        )
