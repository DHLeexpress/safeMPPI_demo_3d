import argparse
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_lab_hp100_token_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("hp100_token_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sources(tmp_path):
    demo_dir = tmp_path / "success2000pg"
    demo_dir.mkdir()
    (demo_dir / "manifest.json").write_text("{}")
    ood_config = tmp_path / "spheres.json"
    ood_config.write_text("{}")
    return demo_dir, ood_config


def _args(tmp_path):
    return argparse.Namespace(
        output_root=tmp_path / "out",
        epochs=500,
        batch_size=32,
        max_windows_per_trajectory=32,
        learning_rate=3.0e-4,
        patience=50,
        min_delta=1.0e-4,
        min_epochs=50,
        audit_seed=91000,
        ood_audit_seed=191000,
        seed=0,
        cpu_threads_per_arm=8,
    )


def test_arm_matrix_changes_only_hp100_token_width(tmp_path):
    demo_dir, ood_config = _sources(tmp_path)
    arms = MODULE.build_arms(
        demo_dir, ood_config, ["cuda:1", "cuda:3"]
    )

    assert [arm["name"] for arm in arms] == [
        "hp100_t64_d3",
        "hp100_t128_d3",
        "hp100_t256_d3",
    ]
    assert [arm["grid_token_dim"] for arm in arms] == [64, 128, 256]
    assert [arm["trunk_depth"] for arm in arms] == [3, 3, 3]
    assert [arm["device"] for arm in arms] == [
        "cuda:1", "cuda:3", "cuda:1",
    ]
    assert {arm["demo_dir"] for arm in arms} == {demo_dir}


def test_command_uses_direct_archive_amp_early_stop_and_m100(tmp_path):
    demo_dir, ood_config = _sources(tmp_path)
    arm = MODULE.build_arms(demo_dir, ood_config, ["cuda:3"])[1]
    command = MODULE.trainer_command(_args(tmp_path), arm)
    joined = " ".join(command)

    assert "--context-model uniform_hp100" in joined
    assert "--grid-token-dim 128" in joined
    assert "--trunk-depth 3" in joined
    assert "--max-windows-per-trajectory 32" in joined
    assert "--cuda-amp" in command
    assert "--patience 50" in joined
    assert "--min-delta 0.0001" in joined
    assert "--min-epochs 50" in joined
    assert "--audit-episodes 100" in joined
    assert "--ood-audit-episodes 100" in joined
    assert "--context-cache" not in command


def test_summary_row_pools_id_and_ood_m100_metrics(tmp_path):
    demo_dir, ood_config = _sources(tmp_path)
    arm = MODULE.build_arms(demo_dir, ood_config, ["cuda:1"])[0]
    output = tmp_path / "arm"
    output.mkdir()
    summaries = [
        {"SR": 0.5, "CR": 0.3, "OOB": 0.2, "window_validity": 0.8},
        {"SR": 0.7, "CR": 0.2, "OOB": 0.1, "window_validity": 0.9},
    ]
    (output / "pretrain_manifest.json").write_text(json.dumps({
        "trainable_parameter_count": 123,
        "actual_epochs": 91,
        "selected_epoch": 40,
        "selected_valid_loss": 0.25,
        "raw_audit_summary": summaries,
        "ood_raw_audit_summary": summaries,
    }))

    row = MODULE.summary_row(arm, output)
    assert row["actual_epochs"] == 91
    assert row["id_SR"] == pytest.approx(0.6)
    assert row["ood_window_validity"] == pytest.approx(0.85)


def test_source_contract_requires_demo_manifest(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    ood_config = tmp_path / "ood.json"
    ood_config.write_text("{}")

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        MODULE.validate_sources(demo_dir, ood_config)
