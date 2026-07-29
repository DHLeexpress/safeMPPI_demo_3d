from pathlib import Path

import numpy as np
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import ReferenceGovernor
from safe_mppi.lab_flow_expansion import (
    LabExpansionPolicyAdapter,
    LabFlowExpansionTask,
)
from safe_mppi.lab_visual_flow import LabVisualFlowPolicy


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
