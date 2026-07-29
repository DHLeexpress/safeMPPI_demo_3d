from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from flow_deployment.lab_pretrained import load_lab_reference_policy


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "flow_deployment" / "minhyuk_handoff"
RESULT = ROOT / "results" / "lab_ball_expansion" / "minhyuk_nozB5_r20"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pretrained_handoff_is_expansion_base() -> None:
    deployment = HANDOFF / "pretrained_visual_hp3d.pt"
    expansion = HANDOFF / "expansion_pretrain" / "pretrained.pt"
    expected = "fc4d215817b56d74730a0a90f6abc57d17dbeb7626302add535760399cdeeeb4"
    assert _sha256(deployment) == expected
    assert _sha256(expansion) == expected
    manifest = json.loads(
        (HANDOFF / "expansion_pretrain" / "pretrain_manifest.json").read_text()
    )
    source = (
        HANDOFF / "expansion_pretrain" / manifest["source_demo_dir"]
    ).resolve()
    assert (source / "manifest.json").is_file()
    assert (source / "resolved_config.json").is_file()


def test_packaged_expansion_matches_raw_r20_state() -> None:
    packaged_path = HANDOFF / "expanded_visual_nozB5_r20.pt"
    raw_path = RESULT / "checkpoint_020.pt"
    assert _sha256(raw_path) == (
        "d10ce121cc490c9056c1165b5197126be2edbc62d22fff07a44f649ee004db67"
    )
    packaged = torch.load(
        packaged_path, map_location="cpu", weights_only=False,
    )
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    assert packaged["provenance"]["expansion_round"] == 20
    assert packaged["model"].keys() == raw["model"].keys()
    for key in raw["model"]:
        assert torch.equal(packaged["model"][key], raw["model"][key])
    policy = load_lab_reference_policy(packaged_path)
    assert tuple(policy.plan_shape) == (10, 3)
    assert policy.context_schema == "lab_spherical_hp3d_v1"
    assert policy.control_limit == 0.3


def test_compact_checkpoint_endpoints_are_present() -> None:
    assert (RESULT / "checkpoint_000.pt").is_file()
    assert (RESULT / "checkpoint_020.pt").is_file()
    manifest = json.loads((RESULT / "manifest.json").read_text())
    assert manifest["config"]["rounds"] == 20
    assert manifest["claim"].startswith("Coverage-promising experimental")
