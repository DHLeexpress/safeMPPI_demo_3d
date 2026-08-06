from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "paper_ready" / "0806"


def test_locked_shared_manifest() -> None:
    for line in (CAMPAIGN / "LOCKED_SHARED.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (CAMPAIGN / relative).resolve()
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_0806_campaign_contract() -> None:
    manifest = json.loads((CAMPAIGN / "campaign_manifest.json").read_text())
    assert manifest["campaign_id"] == "2026-08-06"
    assert manifest["status"] == "FROZEN_INPUTS_READY"
    assert manifest["source_snapshot_git_sha"] == (
        "9cafc00551e4964b9dbe559b1a4ba95104e9c88a"
    )
    assert manifest["task"]["scenario_count"] == 2
    assert manifest["task"]["scenario_ids"] == [
        "symmetric_scene_outer",
        "symmetric_scene_inner",
    ]
    assert manifest["task"]["obstacle_count_per_scenario"] == 5
    assert manifest["task"]["policies"] == ["safemppi", "pretrained"]
    assert manifest["task"]["gammas"] == [0.1, 0.3, 0.5, 1.0]
    assert manifest["task"]["planned_flight_cells"] == 16
    assert "lab_pillars_asbuilt.json" in manifest["excluded_inputs"]

    with (CAMPAIGN / "FLIGHT_INDEX_TEMPLATE.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest["task"]["planned_flight_cells"]
    assert {row["scene_id"] for row in rows} == {
        "symmetric_scene_outer",
        "symmetric_scene_inner",
    }
    assert {row["policy"] for row in rows} == {"safemppi", "pretrained"}
    assert {row["simulated_outcome"] for row in rows} == {
        "SUCCESS",
        "COLLISION",
        "OOB",
    }
    for row in rows:
        assert (CAMPAIGN / row["planned_trajectory_relpath"]).is_file()


def test_frozen_0806_suite() -> None:
    suite = CAMPAIGN / "inputs" / "0806_flight_demonstration_suite"
    lines = (suite / "FROZEN.sha256").read_text().splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = suite / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    reproduction = json.loads(
        (suite / "renderer" / "BYTE_IDENTICAL_REPRODUCTION.json").read_text()
    )
    assert reproduction["status"] == "PASS"
    assert reproduction["count"] == 16
    assert all(row["byte_identical"] for row in reproduction["videos"])

    references = sorted(suite.glob("scenes/*/flight_references/*_100hz.npz"))
    assert len(references) == 16
    for path in references:
        with np.load(path, allow_pickle=False) as data:
            count = len(data["time_s"])
            assert data["position_ref"].shape == (count, 3)
            assert data["velocity_ref"].shape == (count, 3)
            assert data["acceleration_ref"].shape == (count, 3)
            np.testing.assert_allclose(np.diff(data["time_s"]), 0.01, atol=1e-9)


def test_cylinder_snapshots_match_pinned_contract() -> None:
    common = ROOT / "paper_ready" / "common" / "configs"
    id_config = json.loads((common / "cylinder_id_4to8.json").read_text())
    five_config = json.loads((common / "cylinder_five_generator.json").read_text())

    assert id_config["scene_randomization"]["count_min"] == 4
    assert id_config["scene_randomization"]["count_max"] == 8
    assert five_config["scene_randomization"]["count_min"] == 5
    assert five_config["scene_randomization"]["count_max"] == 5
    for config in (id_config, five_config):
        assert config["taskspace"]["start"] == [-2.1, 1.5, 0.9, 0.0, 0.0, 0.0]
        assert config["taskspace"]["goal"] == [0.7, -1.5, 0.9]
        assert config["safemppi"]["horizon"] == 10
        assert config["safemppi"]["num_samples"] == 512
        assert config["safemppi"]["demo_u_max"] == 0.3
        assert config["safemppi"]["centroid_gain"] == 0.0
        assert config["safemppi"]["sigma_aniso"] == 1.0
        assert config["safemppi"]["z_bias_weight"] == 0.0
