"""Contracts for optional large-archive pretraining efficiency controls."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi import lab_reference_flow_task as task
from scripts.pretrain_lab_reference_flow import (
    train,
    validate_training_efficiency_contract,
    window_sampling_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_cylinders_path_midpoint_uniform_v2.json"


def _tiny_archive(path: Path, control_count: int = 20) -> None:
    config = load_config(CONFIG)
    path.mkdir()
    (path / "resolved_config.json").write_text(
        json.dumps(config.raw) + "\n"
    )
    states = np.repeat(
        np.asarray(config.taskspace.start, np.float32)[None],
        control_count + 1,
        axis=0,
    )
    np.savez_compressed(
        path / "run.npz",
        states=states,
        controls=np.zeros((control_count, 3), np.float32),
    )
    (path / "manifest.json").write_text(json.dumps({
        "runs": [{
            "accepted": True,
            "file": "run.npz",
            "gamma": 0.3,
            "seed": 7,
        }],
    }) + "\n")


def test_window_cap_is_deterministic_endpoint_inclusive_and_preconstruction(
    tmp_path, monkeypatch,
):
    archive = tmp_path / "archive"
    _tiny_archive(archive)
    calls = []

    def counted_context(env, state, gamma):
        del env, state
        calls.append(float(gamma))
        return np.zeros(10, np.float32)

    monkeypatch.setattr(task, "build_context", counted_context)
    contexts, plans, metadata, _ = task.lab_reference_demo_windows(
        archive,
        validate_archive=False,
        max_windows_per_trajectory=5,
    )
    assert contexts.shape == (5, 10)
    assert plans.shape == (5, 10, 3)
    assert len(calls) == 5
    assert [row["t"] for row in metadata] == [0, 2, 5, 8, 10]
    assert {
        row["trajectory_available_windows"] for row in metadata
    } == {11}
    assert {row["trajectory_selected_windows"] for row in metadata} == {5}
    provenance = window_sampling_provenance(metadata, 5)
    assert provenance == {
        "schema": "endpoint_stratified_per_trajectory_v1",
        "enabled": True,
        "max_windows_per_trajectory": 5,
        "endpoint_inclusive": True,
        "trajectory_count": 1,
        "available_windows": 11,
        "selected_windows": 5,
        "all_trajectories_represented": True,
    }


def test_window_cap_none_preserves_all_windows(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    _tiny_archive(archive)
    monkeypatch.setattr(
        task,
        "build_context",
        lambda env, state, gamma: np.zeros(10, np.float32),
    )
    contexts, _, metadata, _ = task.lab_reference_demo_windows(
        archive,
        validate_archive=False,
    )
    assert len(contexts) == 11
    assert [row["t"] for row in metadata] == list(range(11))

    legacy_metadata = [
        {"file": "a.npz"},
        {"file": "a.npz"},
        {"file": "b.npz"},
    ]
    provenance = window_sampling_provenance(legacy_metadata, None)
    assert provenance["available_windows"] == 3
    assert provenance["selected_windows"] == 3
    assert provenance["all_trajectories_represented"] is True


def test_cuda_amp_is_opt_in_and_default_cpu_training_still_runs():
    validate_training_efficiency_contract(
        max_windows_per_trajectory=None,
        cuda_amp=False,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="CUDA AMP requires"):
        validate_training_efficiency_contract(
            max_windows_per_trajectory=4,
            cuda_amp=True,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="at least 2"):
        validate_training_efficiency_contract(
            max_windows_per_trajectory=1,
            cuda_amp=False,
            device=torch.device("cpu"),
        )

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def cfm_loss(self, contexts, plans, reduction="mean"):
            values = (self.weight * contexts[:, 0] - plans[:, 0, 0]).square()
            return values.mean() if reduction == "mean" else values

    contexts = torch.arange(20, dtype=torch.float32).reshape(20, 1) / 20
    plans = torch.zeros(20, 10, 3)
    metadata = [
        {
            "gamma": 0.3,
            "seed": index // 5,
            "scene_hash": f"scene-{index // 5}",
            "file": f"trajectory-{index // 5}.npz",
            "t": index % 5,
        }
        for index in range(20)
    ]
    history, training_ids, validation_ids, *_ = train(
        TinyPolicy(),
        contexts,
        plans,
        metadata,
        epochs=1,
        batch_size=4,
        learning_rate=1.0e-3,
        seed=3,
        device=torch.device("cpu"),
    )
    assert len(history) == 1
    assert len(training_ids) + len(validation_ids) == 20
    assert np.isfinite(history[0]["train"])
    assert np.isfinite(history[0]["valid"])
