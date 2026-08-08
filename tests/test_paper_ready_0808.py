from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "paper_ready" / "0808"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_0808_manifest_and_frozen_trajectory_matrix() -> None:
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    assert manifest["counts"] == {
        "expanded_quality_v2_success": 16,
        "pretrained_success": 4,
        "pretrained_collision": 2,
        "total_trajectories": 22,
        "frozen_100hz_references": 22,
    }
    assert manifest["safemppi_status"] == "ON_HOLD_NOT_INCLUDED"
    assert manifest["expanded_policy"]["packaged_nfe"] == 12
    assert manifest["pretrained_policy"]["packaged_nfe"] == 16
    assert manifest["operator_boundaries"]["deploy_sim_must_not_be_modified"]

    groups = {
        "expanded_quality_v2": (16, {"SUCCESS"}),
        "pretrained_success": (4, {"SUCCESS"}),
        "pretrained_collisions": (2, {"COLLISION"}),
    }
    for group, (count, statuses) in groups.items():
        paths = sorted((BUNDLE / "trajectories" / group).glob("*.npz"))
        assert len(paths) == count
        observed = set()
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                observed.add(str(archive["status"].item()))
        assert observed == statuses


def test_0808_flight_reference_contract() -> None:
    manifest = json.loads(
        (BUNDLE / "flight_references" / "manifest.json").read_text()
    )
    assert manifest["count"] == 22
    assert not manifest["contract"]["policy_or_planner_called_by_exporter"]
    assert manifest["contract"]["governor_application_count"] == 1
    assert manifest["contract"]["player_must_not_reapply_governor"]

    with (BUNDLE / "flight_references" / "FLIGHT_INDEX.csv").open(newline="") as handle:
        index = list(csv.DictReader(handle))
    assert len(index) == 22
    assert len({row["flight_id"] for row in index}) == 22
    assert sum(
        row["hardware_eligibility"] == "SIMULATION_ONLY_KNOWN_COLLISION"
        for row in index
    ) == 2

    for row in manifest["runs"]:
        reference = BUNDLE / str(row["flight_reference"])
        source = Path(str(row["source_archive"]))
        if not source.is_absolute():
            source = BUNDLE / source
        assert sha256_file(reference) == row["flight_reference_sha256"]
        assert sha256_file(source) == row["source_archive_sha256"]
        with np.load(reference, allow_pickle=False) as archive:
            count = len(archive["time_s"])
            assert archive["position_ref"].shape == (count, 3)
            assert archive["velocity_ref"].shape == (count, 3)
            assert archive["acceleration_ref"].shape == (count, 3)
            np.testing.assert_allclose(
                np.diff(archive["time_s"]), 0.01, atol=1.0e-9, rtol=0.0
            )
            assert np.linalg.norm(archive["velocity_ref"], axis=1).max() <= 0.700001
            assert np.abs(archive["velocity_ref"][:, 2]).max() <= 0.300001
            assert np.abs(archive["acceleration_ref"]).max() <= 0.300001


def test_0808_player_rejects_known_collision() -> None:
    player_path = BUNDLE / "minhyuk" / "templates" / "frozen_reference_player.py"
    player = runpy.run_path(str(player_path), run_name="p0808_player")
    collision_path = next(
        (BUNDLE / "flight_references" / "pretrained_collisions").glob("*.npz")
    )
    reference = player["load_reference"](collision_path)
    with pytest.raises(ValueError, match="hardware playback is forbidden"):
        player["stream_reference"](reference, lambda *_: None)


def test_0808_sha_inventory() -> None:
    expected = {}
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual = {
        path.relative_to(BUNDLE).as_posix(): sha256_file(path)
        for path in sorted(BUNDLE.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert actual == expected
