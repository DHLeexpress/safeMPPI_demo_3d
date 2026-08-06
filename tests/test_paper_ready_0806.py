from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


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
    assert manifest["status"] == "PREPARING_INPUTS"
    assert manifest["source_snapshot_git_sha"] == (
        "dabb5011dfc674864e1de275a1e1c2adab58f4af"
    )
    assert manifest["task"]["scenario_count"] == 3
    assert manifest["task"]["obstacle_count_per_scenario"] == 5
    assert manifest["task"]["policies"] == ["SafeMPPI", "pretrained_flow"]
    assert manifest["task"]["gammas"] == [0.1, 0.3, 0.5, 1.0]
    assert manifest["task"]["planned_flight_cells"] == 24
    assert "lab_pillars_asbuilt.json" in manifest["excluded_inputs"]

    with (CAMPAIGN / "FLIGHT_INDEX_TEMPLATE.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest["task"]["planned_flight_cells"]
    assert {row["scenario_id"] for row in rows} == {
        "scenario_01",
        "scenario_02",
        "scenario_03",
    }
    assert {row["policy"] for row in rows} == {"SafeMPPI", "pretrained_flow"}


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
