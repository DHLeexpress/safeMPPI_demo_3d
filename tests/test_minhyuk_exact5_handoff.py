from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "flow_deployment" / "minhyuk_stage1_handoff"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_five_candidates_are_provenance_bound() -> None:
    config = json.loads(
        (ROOT / "configs/lab_clutter_cylinders_lab_five_v2.json").read_text()
    )
    assert config["scene_randomization"]["count_min"] == 5
    assert config["scene_randomization"]["count_max"] == 5

    manifest = json.loads((HANDOFF / "asset_manifest.json").read_text())
    evidence = manifest["id_cylinder_safemppi"]
    assert evidence["evidence_contract"]["physical_lab_obstacle_count"] == 5
    candidates = evidence["candidates"]
    assert [candidate["candidate"] for candidate in candidates] == ["E", "F"]
    assert all(candidate["all_success"] for candidate in candidates)
    assert all(candidate["route_signature_count"] == 3 for candidate in candidates)

    for candidate in candidates:
        assert (HANDOFF / candidate["overlay_png"]).is_file()
        assert (HANDOFF / candidate["overlay_pdf"]).is_file()
        assert len(candidate["rows"]) == 4
        for row in candidate["rows"]:
            archive = ROOT / row["archive"]
            assert _sha256(archive) == row["archive_sha256"]
            with np.load(archive, allow_pickle=True) as data:
                assert data["cylinders"].shape == (5, 3)
                assert str(data["scene_id"].item()) == candidate["scene_id"]
                assert int(data["seed"]) == candidate["rollout_seed"]


def test_handoff_checksums_cover_current_files() -> None:
    entries = {}
    for line in (HANDOFF / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    for path in HANDOFF.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(ROOT).as_posix()
        assert entries[relative] == _sha256(path)
