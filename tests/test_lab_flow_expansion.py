from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import ReferenceGovernor
from safe_mppi.lab_flow_expansion import (
    LabExpansionPolicyAdapter,
    LabFlowExpansionTask,
    load_lab_expansion_policy,
)
from safe_mppi.lab_visual_flow import LabVisualFlowPolicy
from safe_mppi.lab_visual_flow import (
    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_VISUAL_HISTORY_LENGTH,
    LAB_VISUAL_PACKED_DIM,
    LabNonuniformRadialHistoryFlowPolicy,
    LabNonuniformRadialFlowPolicy,
    LabVisualHistoryFlowPolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def _task_and_policy():
    config = load_config(ROOT / "configs" / "lab_ball_pretrain.json")
    policy = LabExpansionPolicyAdapter(LabVisualFlowPolicy(control_limit=0.3))
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        tight_corridor=True,
    )
    return config, task, policy


def test_lab_context_hides_governor_memory_from_visual_policy():
    _, task, policy = _task_and_policy()
    state = task.reset(0.3, 0, 1)
    state["previous_applied"][:] = (0.1, -0.2, 0.05)
    state["previous_raw"][:] = (-0.3, 0.2, 0.1)
    context = task.context(state, 0.3)
    assert context.shape == (policy.policy_context_dim + 6,)
    assert policy._policy_context(context).shape == (policy.policy_context_dim,)
    generator = torch.Generator().manual_seed(4)
    assert policy.sample(context, 2, generator).shape == (2, 10, 3)


def test_lab_plan_rollout_matches_stateful_reference_governor():
    config, task, _ = _task_and_policy()
    rng = np.random.default_rng(7)
    state = task.env.start.copy()
    previous_applied = np.asarray([0.07, -0.03, 0.02], np.float32)
    plan = rng.uniform(-0.3, 0.3, size=(10, 3)).astype(np.float32)
    states, applied, dense = task._rollout_plan(
        state, previous_applied, plan,
    )

    governor = ReferenceGovernor(config.safemppi)
    governor.previous_applied = previous_applied.copy()
    expected_states = [state.copy()]
    expected_applied, expected_dense = [], []
    for command in plan:
        state, command_applied, step_dense = governor.step(state, command)
        expected_states.append(state.copy())
        expected_applied.append(command_applied)
        expected_dense.append(step_dense)
    np.testing.assert_allclose(states, expected_states, atol=1.0e-7)
    np.testing.assert_allclose(applied, expected_applied, atol=1.0e-7)
    np.testing.assert_allclose(dense, expected_dense, atol=1.0e-7)


def test_visual_lr_group_includes_encoder_and_first_flow_layer_only():
    policy = LabVisualFlowPolicy(control_limit=0.3)
    groups = policy.expansion_parameter_groups(1.0e-3, 0.1)
    slow_ids = {id(parameter) for parameter in groups[0]["params"]}
    fast_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert slow_ids.isdisjoint(fast_ids)
    assert slow_ids | fast_ids == {
        id(parameter) for parameter in policy.parameters()
    }
    assert all(
        id(parameter) in slow_ids
        for parameter in policy.grid_encoder.parameters()
    )
    assert all(
        id(parameter) in slow_ids
        for parameter in policy.flow.trunk[0].parameters()
    )
    assert groups[0]["lr"] == 1.0e-4
    assert groups[1]["lr"] == 1.0e-3


def test_lab_tight_corridor_never_imports_canonical_ball_bounds():
    _, task, _ = _task_and_policy()
    state = task.reset(0.3, 0, 1)
    context = task.context(state, 0.3)
    zero_plan = torch.zeros(1, 10, 3)
    result = task.verify(context, zero_plan, 0.3)[0]
    assert result.valid


def test_gru_expansion_context_tracks_only_executed_raw_commands():
    config = load_config(ROOT / "configs" / "lab_ball_pretrain.json")
    base = LabVisualHistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=4,
        history_token_dim=3,
        control_limit=0.3,
        nfe=2,
    )
    policy = LabExpansionPolicyAdapter(
        base, freeze_history_encoder=True,
    )
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        tight_corridor=True,
    )
    state = task.reset(0.3, 0, 1)
    before = task.context(state, 0.3)
    history_before = before[
        LAB_VISUAL_PACKED_DIM:policy.policy_context_dim
    ].reshape(LAB_VISUAL_HISTORY_LENGTH, 4)
    assert not bool(history_before[:, 3].any())

    # Candidate verification must not mutate history because nothing executed.
    task.verify(before, torch.zeros(1, 10, 3), 0.3)
    assert not bool(state["raw_history"][:, 3].any())

    candidate = torch.zeros(10, 3)
    candidate[0] = torch.tensor([0.2, -0.1, 0.05])
    updated = task.advance(state, candidate)
    np.testing.assert_array_equal(
        updated["raw_history"][-1],
        np.asarray([0.2, -0.1, 0.05, 1.0], np.float32),
    )
    after = task.context(updated, 0.3)
    history_after = after[
        LAB_VISUAL_PACKED_DIM:policy.policy_context_dim
    ].reshape(LAB_VISUAL_HISTORY_LENGTH, 4)
    np.testing.assert_array_equal(
        history_after, updated["raw_history"],
    )


def test_gru_expansion_freeze_requires_explicit_contract():
    base = LabVisualHistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=4,
        history_token_dim=3,
        control_limit=0.3,
        nfe=2,
    )
    unspecified = LabExpansionPolicyAdapter(base)
    with pytest.raises(ValueError, match="explicit history freeze"):
        unspecified.expansion_parameter_groups(1.0e-4, 0.1)

    frozen = LabExpansionPolicyAdapter(
        base, freeze_history_encoder=True,
    )
    assert not any(
        parameter.requires_grad
        for parameter in base.history_encoder.parameters()
    )
    frozen.expansion_parameter_groups(1.0e-4, 0.1)
    assert not any(
        parameter.requires_grad
        for parameter in base.history_encoder.parameters()
    )

    trainable = LabExpansionPolicyAdapter(
        base, freeze_history_encoder=False,
    )
    trainable.expansion_parameter_groups(1.0e-4, 0.1)
    assert all(
        parameter.requires_grad
        for parameter in base.history_encoder.parameters()
    )


def test_gru_loader_freezes_by_default_and_unfreezes_only_explicitly(
    monkeypatch,
):
    def policy():
        return LabVisualHistoryFlowPolicy(
            hidden=8,
            representation_dim=4,
            grid_token_dim=4,
            history_token_dim=3,
            control_limit=0.3,
            nfe=2,
        )

    frozen_base = policy()
    monkeypatch.setattr(
        "safe_mppi.lab_flow_expansion.load_lab_reference_policy",
        lambda path: frozen_base,
    )
    frozen = load_lab_expansion_policy("/unused")
    assert frozen.freeze_history_encoder is True
    assert not any(
        parameter.requires_grad
        for parameter in frozen_base.history_encoder.parameters()
    )

    trainable_base = policy()
    monkeypatch.setattr(
        "safe_mppi.lab_flow_expansion.load_lab_reference_policy",
        lambda path: trainable_base,
    )
    trainable = load_lab_expansion_policy(
        "/unused", train_history_encoder=True,
    )
    assert trainable.freeze_history_encoder is False
    assert all(
        parameter.requires_grad
        for parameter in trainable_base.history_encoder.parameters()
    )

    monkeypatch.setattr(
        "safe_mppi.lab_flow_expansion.load_lab_reference_policy",
        lambda path: LabVisualFlowPolicy(control_limit=0.3),
    )
    with pytest.raises(ValueError, match="only to a GRU"):
        load_lab_expansion_policy(
            "/unused", train_history_encoder=True,
        )


def test_radial_gru_expansion_samples_embeds_and_freezes_history():
    config = load_config(ROOT / "configs" / "lab_ball_pretrain.json")
    base = LabNonuniformRadialHistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=32,
        control_limit=0.3,
        nfe=1,
        trunk_depth=3,
    )
    policy = LabExpansionPolicyAdapter(
        base,
        freeze_history_encoder=True,
    )
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        tight_corridor=True,
    )
    state = task.reset(0.3, 0, 11)
    context = task.context(state, 0.3)
    assert context.shape == (
        LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM + 6,
    )
    assert not any(
        parameter.requires_grad
        for parameter in base.history_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in base.grid_encoder.parameters()
    )

    generator = torch.Generator().manual_seed(13)
    candidates = policy.sample(context, 1, generator)
    assert candidates.shape == (1, 10, 3)
    features = policy.embed(
        context[None],
        candidates,
    )
    assert features.shape == (1, 4)

    groups = policy.expansion_parameter_groups(1.0e-4, 0.1)
    slow_ids = {id(parameter) for parameter in groups[0]["params"]}
    assert groups[0]["lr"] == 1.0e-5
    assert all(
        id(parameter) in slow_ids
        for parameter in base.grid_encoder.parameters()
    )
    assert all(
        id(parameter) in slow_ids
        for parameter in base.flow.trunk[0].parameters()
    )
    assert not any(
        id(parameter) in {
            id(group_parameter)
            for group in groups
            for group_parameter in group["params"]
        }
        for parameter in base.history_encoder.parameters()
    )


def test_radial_visual_adapter_preserves_full_policy_context():
    config = load_config(ROOT / "configs" / "lab_ball_pretrain.json")
    base = LabNonuniformRadialFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        control_limit=0.3,
        nfe=1,
    )
    policy = LabExpansionPolicyAdapter(base)
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
    )
    context = task.context(task.reset(0.3, 0, 17), 0.3)
    assert context.shape == (LAB_RADIAL_VISUAL_PACKED_DIM + 6,)
    assert policy._policy_context(context).shape == (
        LAB_RADIAL_VISUAL_PACKED_DIM,
    )
