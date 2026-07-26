from __future__ import annotations

import numpy as np
import pytest
import torch

import safe_mppi.expansion as expansion
from safe_mppi.expansion import (
    ExpansionConfig,
    QueryRecord,
    Verification,
    _sliding_success_gp_rows,
    run_safe_expansion,
)
from safe_mppi.expansion_visualize import (
    round_sigma_statistics,
    within_round_normalized_sigma,
)
from safe_mppi.flow_model import ConditionalFlowMLP


def _successful_window(
    gamma: float,
    trajectory: int,
    window_start: int,
    *,
    round_i: int = 1,
) -> QueryRecord:
    trajectory_id = f"r{round_i}:g{gamma}:trajectory{trajectory}"
    return QueryRecord(
        round=round_i,
        gamma=gamma,
        episode=trajectory,
        context_id=100 * trajectory + window_start,
        context=torch.tensor([gamma, float(window_start)]),
        candidate=torch.tensor([[float(trajectory), float(window_start)]]),
        verification=Verification(
            valid=True,
            hp_eligible=True,
            margin=1.0,
            execution_cost=0.0,
            progress=1.0,
            progress_eligible=True,
        ),
        trajectory_id=trajectory_id,
        window_id=f"{trajectory_id}:w{window_start}",
        window_start=window_start,
        valid_horizon=1,
        loss_mask=torch.ones(1, 2),
    )


def test_sliding_gp_trajectory_uniform_covers_each_trajectory_and_departure():
    records = [
        _successful_window(0.1, trajectory, start)
        for trajectory in (0, 1)
        for start in range(5)
    ]
    # This later record must not leak into a through-round-1 posterior.
    records.append(_successful_window(0.1, 2, 0, round_i=2))

    selected = _sliding_success_gp_rows(
        records,
        (0.1,),
        total_cap=6,
        through_round=1,
        selector="trajectory_uniform",
    )
    reversed_selected = _sliding_success_gp_rows(
        list(reversed(records)),
        (0.1,),
        total_cap=6,
        through_round=1,
        selector="trajectory_uniform",
    )

    assert [row.window_id for row in selected] == [
        row.window_id for row in reversed_selected
    ]
    assert {row.round for row in selected} == {1}
    assert {
        trajectory: [row.window_start for row in selected
                     if row.episode == trajectory]
        for trajectory in (0, 1)
    } == {
        0: [0, 2, 4],
        1: [0, 2, 4],
    }


def test_sliding_gp_fifo_tail_preserves_legacy_high_start_bias():
    records = [
        _successful_window(0.1, trajectory, start)
        for trajectory in (0, 1)
        for start in range(5)
    ]

    selected = _sliding_success_gp_rows(
        records,
        (0.1,),
        total_cap=6,
        through_round=1,
        selector="fifo_tail",
    )

    assert [row.window_start for row in selected] == [2, 2, 3, 3, 4, 4]
    assert not any(row.window_start == 0 for row in selected)


def test_sliding_gp_trajectory_uniform_stays_start_stratified_at_saturation():
    records = [
        _successful_window(0.1, trajectory, start)
        for trajectory in range(8)
        for start in range(10)
    ]

    selected = _sliding_success_gp_rows(
        records,
        (0.1,),
        total_cap=8,
        through_round=1,
        selector="trajectory_uniform",
    )

    starts = [row.window_start for row in selected]
    assert len(selected) == 8
    assert len({row.trajectory_id for row in selected}) == 8
    assert min(starts) == 0
    assert max(starts) == 9
    assert len(set(starts)) >= 6


def test_phi_s_endpoint_formula_is_current_trunk_at_s_point_nine():
    torch.manual_seed(7)
    policy = ConditionalFlowMLP(
        context_dim=3,
        plan_shape=(2, 2),
        hidden=11,
        representation_dim=5,
        trunk_depth=2,
        time_features="raw1",
    )
    contexts = torch.randn(4, 3)
    candidates = torch.randn(4, 2, 2)

    actual = policy.embed(contexts, candidates, flow_time=0.9)
    flat = candidates.reshape(4, -1)
    expected = policy._features(
        0.9 * flat,
        torch.full((4,), 0.9),
        contexts,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class _TwoStepSuccessTask:
    def reset(self, gamma, episode, seed):
        return {"episode": int(episode), "step": 0}

    def context(self, state, gamma):
        return torch.tensor(
            [float(gamma), float(state["episode"]), float(state["step"])],
            dtype=torch.float32,
        )

    def verify(self, context, candidates, gamma):
        return [
            Verification(
                valid=True,
                hp_eligible=True,
                margin=1.0,
                execution_cost=float(candidate.square().sum()),
                progress=1.0,
                progress_eligible=True,
            )
            for candidate in candidates
        ]

    def advance(self, state, candidate):
        return {"episode": state["episode"], "step": state["step"] + 1}

    def terminal(self, state):
        return "SUCCESS" if state["step"] >= 2 else None


def _sliding_config(mode: str) -> ExpansionConfig:
    return ExpansionConfig(
        rounds=3,
        gammas=(0.1,),
        parallel_episodes=1,
        verifier_workers=1,
        max_retry_batches=1,
        max_steps=2,
        K=2,
        B=1,
        batch_size=2,
        inner_steps=1,
        learning_rate=1.0e-3,
        replay_rounds=2,
        gp_buffer_cap=4,
        rbf_lengthscale=1.0,
        replay_selector="uniform",
        archive_rule="successful_executed_windows",
        successful_trajectory_selector="lowest_episode_id",
        successful_trajectories_per_gamma=1,
        negative_alpha=0.0,
        gp_reference_mode=mode,
        gp_sliding_row_selector="trajectory_uniform",
        seed=3,
    )


@pytest.mark.parametrize(
    ("mode", "expected_unique_hashes", "expected_frozen"),
    [
        ("sliding_success_per_gamma_current_phi", 3, False),
        ("sliding_success_per_gamma_frozen_phi", 1, True),
    ],
)
def test_sliding_gp_representation_tracks_current_or_legacy_frozen_model(
    tmp_path,
    monkeypatch,
    mode,
    expected_unique_hashes,
    expected_frozen,
):
    def deterministic_update(policy, task, optimizer, positive, negative, cfg, rng):
        # Isolate representation chronology from optimizer randomness: the
        # current policy changes once after each gather, while a legacy frozen
        # acquisition copy must remain unchanged.
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(0.01)
        return {"steps": 1, "positive_loss": 0.0, "negative_loss": None}

    monkeypatch.setattr(expansion, "_update", deterministic_update)
    torch.manual_seed(11)
    policy = ConditionalFlowMLP(
        context_dim=3,
        plan_shape=(2, 1),
        hidden=8,
        representation_dim=4,
        trunk_depth=2,
        time_features="raw1",
    )
    result = run_safe_expansion(
        policy,
        _TwoStepSuccessTask(),
        tmp_path,
        config=_sliding_config(mode),
    )

    hashes = [
        row["acquisition_representation_hash"]
        for row in result["rounds"]
    ]
    assert len(set(hashes)) == expected_unique_hashes
    assert result["gp_reference"]["frozen_phi"] is expected_frozen
    assert result["gp_reference"]["representation"] == (
        "pretrained_phi0"
        if expected_frozen
        else "current_round_phi_reembedded_before_gather"
    )
    assert result["rounds"][1]["gp_buffer_by_gamma"] == {"0.1": 2}
    assert result["rounds"][1]["gp_feature_hash_by_gamma"]["0.1"] is not None


def test_sigma_normalization_is_robust_and_independent_per_round():
    events = [
        {"round": 1, "sigma_K": np.arange(100, dtype=float)},
        {"round": 2, "sigma_K": 100.0 + 10.0 * np.arange(100, dtype=float)},
        {"round": 3, "sigma_K": np.full(8, 7.0)},
        # An empty GP produces sigma=1 up to floating-point noise. It must
        # render as neutral, not as a fictitious full-range uncertainty map.
        {"round": 4, "sigma_K": 1.0 + np.linspace(-5.0e-7, 5.0e-7, 16)},
    ]
    statistics = round_sigma_statistics(events)

    assert statistics[1] == pytest.approx({
        "q02": 1.98,
        "median": 49.5,
        "q98": 97.02,
    })
    assert statistics[2] == pytest.approx({
        "q02": 119.8,
        "median": 595.0,
        "q98": 1070.2,
    })
    relative_rank_round_1 = within_round_normalized_sigma(49.5, statistics[1])
    relative_rank_round_2 = within_round_normalized_sigma(595.0, statistics[2])
    assert relative_rank_round_1 == pytest.approx(0.5)
    assert relative_rank_round_2 == pytest.approx(0.5)
    np.testing.assert_allclose(
        within_round_normalized_sigma(
            np.asarray([-1.0, 49.5, 101.0]), statistics[1]
        ),
        np.asarray([0.0, 0.5, 1.0]),
    )
    assert within_round_normalized_sigma(7.0, statistics[3]) == 0.5
    assert within_round_normalized_sigma(1.0, statistics[4]) == 0.5
