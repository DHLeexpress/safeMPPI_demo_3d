import argparse
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_lab_radial_dual_dataset_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("dual_radial_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _datasets(tmp_path):
    rows = []
    for label in ("sigma02_n300", "sigma04_n500"):
        paths = []
        for suffix in ("demo", "cache", "ood.json"):
            path = tmp_path / f"{label}_{suffix}"
            if path.suffix:
                path.write_text("{}")
            else:
                path.mkdir()
            paths.append(str(path))
        rows.append([label, *paths])
    return MODULE.parse_datasets(rows)


def test_arm_matrix_and_device_assignment_are_deterministic(tmp_path):
    arms = MODULE.build_arms(_datasets(tmp_path), ["cuda:0", "cuda:1"])

    assert [arm["grid_token_dim"] for arm in arms] == [64, 128, 256] * 2
    assert [arm["trunk_depth"] for arm in arms] == [3] * 6
    assert [arm["device"] for arm in arms] == ["cuda:0", "cuda:1"] * 3
    assert arms[-1]["name"] == "sigma04_n500_radial_t256_d3"


def test_command_passes_early_stop_contract(tmp_path):
    arm = MODULE.build_arms(_datasets(tmp_path), ["cuda:3"])[0]
    args = argparse.Namespace(
        output_root=tmp_path / "out",
        epochs=500,
        batch_size=32,
        learning_rate=3e-4,
        patience=40,
        min_delta=1e-4,
        min_epochs=150,
        audit_episodes=100,
        audit_seed=91000,
        ood_audit_seed=191000,
        seed=0,
        cpu_threads_per_arm=8,
    )
    command = MODULE.trainer_command(args, arm)
    joined = " ".join(command)

    assert "--context-model radial_hp3d" in joined
    assert "--trunk-depth 3" in joined
    assert "--grid-token-dim 64" in joined
    assert "--patience 40" in joined
    assert "--min-delta 0.0001" in joined
    assert "--min-epochs 150" in joined
    assert "--device cuda:3" in joined


def test_run_arm_bounds_cpu_threads(tmp_path, monkeypatch):
    arm = MODULE.build_arms(_datasets(tmp_path), ["cuda:1"])[0]
    output_root = tmp_path / "out"
    output_root.mkdir()
    args = argparse.Namespace(
        output_root=output_root,
        epochs=500,
        batch_size=32,
        learning_rate=3e-4,
        patience=50,
        min_delta=1e-4,
        min_epochs=50,
        audit_episodes=100,
        audit_seed=91000,
        ood_audit_seed=191000,
        seed=0,
        cpu_threads_per_arm=8,
    )
    captured = {}

    def fake_run(command, *, check, env, stdout, stderr):
        del command, check, stdout, stderr
        captured.update(env)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        MODULE,
        "summary_row",
        lambda supplied_arm, output: {
            "arm": supplied_arm["name"],
            "output": str(output),
        },
    )

    MODULE.run_arm(args, arm)

    assert captured["OMP_NUM_THREADS"] == "8"
    assert captured["MKL_NUM_THREADS"] == "8"
    assert captured["OPENBLAS_NUM_THREADS"] == "8"


def test_exclusive_device_wait_uses_physical_gpu_uuid(monkeypatch):
    outputs = iter([
        "1, GPU-one\n3, GPU-three\n",
        "GPU-three, 123\n",
        "",
    ])
    sleeps = []

    def fake_run(command, **kwargs):
        del command, kwargs
        return argparse.Namespace(stdout=next(outputs))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)

    MODULE.wait_for_exclusive_device("cuda:3", 2.5)

    assert sleeps == [2.5]


def test_summary_row_aggregates_id_and_ood(tmp_path):
    arm = MODULE.build_arms(_datasets(tmp_path), ["cuda:0"])[0]
    output = tmp_path / "arm"
    output.mkdir()
    summaries = [
        {"SR": 0.5, "CR": 0.3, "OOB": 0.2, "window_validity": 0.8},
        {"SR": 0.7, "CR": 0.2, "OOB": 0.1, "window_validity": 0.9},
    ]
    (output / "pretrain_manifest.json").write_text(json.dumps({
        "trainable_parameter_count": 123,
        "epochs": 500,
        "requested_epochs": 500,
        "actual_epochs": 211,
        "selected_epoch": 170,
        "selected_valid_loss": 0.25,
        "raw_audit_summary": summaries,
        "ood_raw_audit_summary": summaries,
    }))

    row = MODULE.summary_row(arm, output)
    assert row["actual_epochs"] == 211
    assert row["id_SR"] == pytest.approx(0.6)
    assert row["ood_window_validity"] == pytest.approx(0.85)


def test_dataset_contract_rejects_duplicate_labels(tmp_path):
    values = []
    for index in range(2):
        fields = []
        for suffix in ("demo", "cache", "ood.json"):
            path = tmp_path / f"{index}_{suffix}"
            path.write_text("{}") if path.suffix else path.mkdir()
            fields.append(str(path))
        values.append(["same", *fields])

    with pytest.raises(ValueError, match="duplicate dataset"):
        MODULE.parse_datasets(values)
