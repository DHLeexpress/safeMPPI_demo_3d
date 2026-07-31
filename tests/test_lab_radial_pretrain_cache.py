import json
from pathlib import Path

import numpy as np
import pytest

from safe_mppi.config import load_config
from safe_mppi.lab_visual_flow import (
    LAB_RADIAL_VISUAL_ENCODER_CHANNELS,
    LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE,
    LAB_RADIAL_VISUAL_FRAME,
    LAB_RADIAL_VISUAL_GRID_SHAPE,
    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_RADIAL_EDGES,
    LAB_RADIAL_VISUAL_SCHEMA,
    LabNonuniformRadialFlowPolicy,
    LabNonuniformRadialHistoryFlowPolicy,
)
from scripts import pretrain_lab_reference_flow as pretrain
from scripts.run_lab_radial_architecture_sweep import ARMS


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = (
    ROOT
    / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
    / "resolved_config.json"
)


def _fake_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "resolved_config.json").write_bytes(
        REFERENCE_CONFIG.read_bytes()
    )
    (archive / "run.npz").write_bytes(b"provenance")
    (archive / "manifest.json").write_text(json.dumps({
        "runs": [{"file": "run.npz", "accepted": True}],
    }) + "\n")
    return archive


def _fake_dataset():
    metadata = [
        {
            "gamma": gamma,
            "seed": scene,
            "scene_id": f"scene-{scene}",
            "scene_hash": f"hash-{scene}",
            "t": 0,
            "file": f"scene-{scene}-g{gamma:g}.npz",
        }
        for scene in range(2)
        for gamma in (0.1, 0.3, 0.5, 1.0)
    ]
    contexts = np.arange(
        len(metadata) * LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
        dtype=np.float32,
    ).reshape(len(metadata), LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM)
    plans = np.zeros((len(metadata), 10, 3), np.float32)
    return contexts, plans, metadata


def test_radial_cache_reuses_one_history_superset_for_both_models(
    tmp_path,
    monkeypatch,
):
    archive = _fake_archive(tmp_path)
    contexts, plans, metadata = _fake_dataset()
    config = load_config(archive / "resolved_config.json")
    monkeypatch.setattr(
        pretrain,
        "lab_reference_demo_windows",
        lambda *args, **kwargs: (contexts, plans, metadata, config),
    )
    cache = tmp_path / "cache"
    manifest = pretrain.build_radial_context_cache(
        archive,
        cache,
        split_seed=7,
    )
    assert manifest["stored_context_schema"] == (
        LAB_RADIAL_VISUAL_HISTORY_SCHEMA
    )
    assert manifest["grid_shape"] == list(LAB_RADIAL_VISUAL_GRID_SHAPE)
    assert manifest["encoder_grid_shape"] == list(
        LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE
    )

    radial = pretrain.load_radial_context_cache(
        cache,
        archive,
        context_model="radial_hp3d",
        split_seed=7,
    )
    history = pretrain.load_radial_context_cache(
        cache,
        archive,
        context_model="radial_hp3d_gru",
        split_seed=7,
    )
    assert radial[0].shape == (len(metadata), LAB_RADIAL_VISUAL_PACKED_DIM)
    assert history[0].shape == (
        len(metadata),
        LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    )
    assert np.array_equal(radial[0], contexts[:, :radial[0].shape[1]])
    assert np.array_equal(history[0], contexts)
    assert np.array_equal(radial[1], plans)
    assert radial[4].tolist() == history[4].tolist()
    assert radial[5].tolist() == history[5].tolist()
    original_digest = pretrain.context_builder_implementation_digest()
    monkeypatch.setattr(
        pretrain,
        "context_builder_implementation_digest",
        lambda: {**original_digest, "sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="implementation mismatch"):
        pretrain.load_radial_context_cache(
            cache,
            archive,
            context_model="radial_hp3d",
            split_seed=7,
        )


def test_radial_cache_fails_closed_on_file_source_and_split_mismatch(
    tmp_path,
    monkeypatch,
):
    archive = _fake_archive(tmp_path)
    contexts, plans, metadata = _fake_dataset()
    config = load_config(archive / "resolved_config.json")
    monkeypatch.setattr(
        pretrain,
        "lab_reference_demo_windows",
        lambda *args, **kwargs: (contexts, plans, metadata, config),
    )
    cache = tmp_path / "cache"
    pretrain.build_radial_context_cache(archive, cache, split_seed=3)
    with pytest.raises(ValueError, match="split seed mismatch"):
        pretrain.load_radial_context_cache(
            cache,
            archive,
            context_model="radial_hp3d",
            split_seed=4,
        )

    with (cache / "plans.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        pretrain.load_radial_context_cache(
            cache,
            archive,
            context_model="radial_hp3d",
            split_seed=3,
        )

    cache2 = tmp_path / "cache2"
    pretrain.build_radial_context_cache(archive, cache2, split_seed=3)
    (archive / "run.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source archive mismatch"):
        pretrain.load_radial_context_cache(
            cache2,
            archive,
            context_model="radial_hp3d",
            split_seed=3,
        )


def test_radial_cache_is_not_published_after_builder_failure(
    tmp_path,
    monkeypatch,
):
    archive = _fake_archive(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic builder failure")

    monkeypatch.setattr(pretrain, "lab_reference_demo_windows", fail)
    output = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="synthetic builder failure"):
        pretrain.build_radial_context_cache(
            archive,
            output,
            split_seed=0,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".cache.building-*"))


def test_radial_pretraining_architecture_contract_and_gru32():
    config = load_config(REFERENCE_CONFIG)
    radial = pretrain.build_pretraining_policy(
        "radial_hp3d",
        config,
        hidden=48,
        representation_dim=32,
        grid_token_dim=128,
        history_token_dim=16,
        nfe=16,
        trunk_depth=2,
    )
    assert isinstance(radial, LabNonuniformRadialFlowPolicy)
    assert radial.grid_token_dim == 128
    with pytest.raises(ValueError, match="history-token-dim=32"):
        pretrain.build_pretraining_policy(
            "radial_hp3d_gru",
            config,
            hidden=48,
            representation_dim=32,
            grid_token_dim=64,
            history_token_dim=16,
            nfe=16,
            trunk_depth=3,
        )
    history = pretrain.build_pretraining_policy(
        "radial_hp3d_gru",
        config,
        hidden=48,
        representation_dim=32,
        grid_token_dim=64,
        history_token_dim=32,
        nfe=16,
        trunk_depth=3,
    )
    assert isinstance(history, LabNonuniformRadialHistoryFlowPolicy)
    assert history.history_encoder.hidden_size == 32

    arch = pretrain.pretraining_arch(
        "radial_hp3d_gru",
        config,
        hidden=48,
        representation_dim=32,
        grid_token_dim=64,
        history_token_dim=32,
        nfe=16,
        trunk_depth=3,
    )
    assert arch == {
        "kind": LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        "plan_shape": [10, 3],
        "hidden": 48,
        "representation_dim": 32,
        "grid_token_dim": 64,
        "history_token_dim": 32,
        "history_length": 10,
        "grid_shape": list(LAB_RADIAL_VISUAL_GRID_SHAPE),
        "grid_channels": [
            "occupancy",
            "nominal_polytope_mask",
            "clipped_hp",
        ],
        "encoder_grid_shape": list(LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE),
        "encoder_grid_channels": list(
            LAB_RADIAL_VISUAL_ENCODER_CHANNELS
        ),
        "grid_frame": LAB_RADIAL_VISUAL_FRAME,
        "radial_edges": list(LAB_RADIAL_VISUAL_RADIAL_EDGES),
        "control_limit": config.safemppi.demo_u_max,
        "nfe": 16,
        "trunk_depth": 3,
        "time_features": "raw1",
    }
    radial_arch = pretrain.pretraining_arch(
        "radial_hp3d",
        config,
        hidden=48,
        representation_dim=32,
        grid_token_dim=64,
        history_token_dim=32,
        nfe=16,
        trunk_depth=2,
    )
    assert radial_arch["kind"] == LAB_RADIAL_VISUAL_SCHEMA
    assert "history_token_dim" not in radial_arch
    assert tuple(radial_arch["radial_edges"]) == (
        LAB_RADIAL_VISUAL_RADIAL_EDGES
    )


def test_cfm_training_rng_is_reset_independently_of_model_size():
    import torch

    torch.manual_seed(11)
    _ = torch.randn(3)
    torch.manual_seed(pretrain.cfm_training_rng_seed(5))
    first = torch.randn(8)

    torch.manual_seed(11)
    _ = torch.randn(3000)
    torch.manual_seed(pretrain.cfm_training_rng_seed(5))
    second = torch.randn(8)
    assert torch.equal(first, second)


def test_fixed_sweep_contains_only_the_five_requested_arms():
    assert [
        (
            arm["context_model"],
            arm["grid_token_dim"],
            arm["trunk_depth"],
        )
        for arm in ARMS
    ] == [
        ("radial_hp3d", 64, 2),
        ("radial_hp3d", 128, 2),
        ("radial_hp3d", 256, 2),
        ("radial_hp3d", 64, 3),
        ("radial_hp3d_gru", 64, 3),
    ]
