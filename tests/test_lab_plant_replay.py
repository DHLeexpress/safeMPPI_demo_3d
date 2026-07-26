import json
from pathlib import Path

import numpy as np

from safe_mppi.config import load_config
from safe_mppi.lab_plant_replay import replay_demo_on_plant


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"


def test_accepted_reference_is_replayed_without_replanning_or_double_smoothing():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    config = load_config(ARCHIVE / "resolved_config.json")
    row = manifest["runs"][0]
    data = np.load(ARCHIVE / row["file"])
    result = replay_demo_on_plant(
        config,
        data["dense_positions"],
        data["executed_controls"],
        seed=int(row["seed"]),
        clip_commanded_position=False,
    )

    substeps = config.safemppi.integration_substeps
    assert len(result["reference_positions"]) == len(data["controls"]) * substeps
    assert np.allclose(
        result["reference_positions"],
        data["dense_positions"][1:],
        atol=1.0e-6,
        rtol=0.0,
    )
    assert np.allclose(
        result["applied_controls"],
        np.repeat(data["executed_controls"], substeps, axis=0),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert result["command_clip_fraction"] == 0.0
    assert result["reference_command_rmse_m"] == 0.0
    assert result["reference_success"]
    assert result["max_abs_applied_acceleration_mps2"] <= 0.300001
    assert result["peak_reference_speed_mps"] <= 0.70001


def test_command_position_clipping_is_explicit_not_implicit():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    config = load_config(ARCHIVE / "resolved_config.json")
    row = manifest["runs"][0]
    data = np.load(ARCHIVE / row["file"])
    result = replay_demo_on_plant(
        config,
        data["dense_positions"],
        data["executed_controls"],
        seed=int(row["seed"]),
        clip_commanded_position=True,
    )
    assert result["command_clip_fraction"] > 0.0
