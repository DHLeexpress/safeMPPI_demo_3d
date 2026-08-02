from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from flow_deployment.lab_reference_contract import load_governed_reference
from flow_deployment.lab_pretrained import load_lab_reference_policy


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "flow_deployment/minhyuk_stage1_handoff"
RUN = HANDOFF / "default_v0_mppicost_lamd7e4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lamd7e4_terminal_checkpoint_is_deployment_ready() -> None:
    raw = RUN / "checkpoint_005.pt"
    packaged = HANDOFF / "checkpoints/stage1_lamd7e4_terminal_r5.pt"
    assert _sha256(raw) == (
        "81d73de001b2080387459a72436069f8b39dc44788ea43c50f732d637f6630cc"
    )
    assert _sha256(packaged) == (
        "115a50e38b9f1c52819649663853d0699568bd07a00df3f4c6ef899262243d99"
    )
    assert sorted(path.name for path in RUN.glob("checkpoint_*.pt")) == [
        "checkpoint_005.pt"
    ]

    payload = torch.load(packaged, map_location="cpu", weights_only=False)
    assert payload["provenance"]["expansion_round"] == 5
    assert payload["provenance"]["expansion_checkpoint_sha256"] == _sha256(raw)
    assert payload["contract"]["deployment_safety_qualified"] is False
    policy = load_lab_reference_policy(packaged)
    assert tuple(policy.plan_shape) == (10, 3)
    assert policy.context_schema == (
        "lab_spherical_hp3d_uniform_radial100_planepack_v1"
    )


def test_lamd7e4_raw_evaluation_and_gallery_are_exact() -> None:
    raw = json.loads((RUN / "eval/raw_eval.json").read_text())
    r0 = raw["summary"]["0"]["pooled"]
    r5 = raw["summary"]["5"]["pooled"]
    assert (r0["SR"], r0["CR"], r0["OOB"]) == (0.1375, 0.8, 0.0625)
    assert (r5["SR"], r5["CR"], r5["OOB"]) == (0.9875, 0.0, 0.0125)
    assert r5["window_validity"] == 0.9977416180263337
    assert _sha256(RUN / "eval/raw_gallery.png") == (
        "4327dc65f1d538020a961b7eb397cc06428651776a29e519f4869fbac411400d"
    )


def test_lamd7e4_paired_references_are_governed_and_honest() -> None:
    reference_root = RUN / "references"
    expanded = json.loads(
        (reference_root / "expanded_r5_seed91074/manifest.json").read_text()
    )
    pretrained = json.loads(
        (reference_root / "pretrained_r0_seed91074/manifest.json").read_text()
    )
    assert expanded["all_successful"] is True
    assert {row["mode"] for row in expanded["runs"]} == {"left"}
    assert {row["status"] for row in expanded["runs"]} == {"SUCCESS"}
    assert pretrained["all_successful"] is False
    assert {row["mode"] for row in pretrained["runs"]} == {"none"}
    assert {row["status"] for row in pretrained["runs"]} == {"COLLISION"}

    for manifest, folder in (
        (expanded, reference_root / "expanded_r5_seed91074"),
        (pretrained, reference_root / "pretrained_r0_seed91074"),
    ):
        for row in manifest["runs"]:
            reference = load_governed_reference(
                folder / row["file"],
                gamma=float(row["gamma"]),
                seed=int(row["seed"]),
                integration_substeps=10,
                action_limit=0.3,
            )
            assert reference.seed == 91074
            assert reference.raw_controls is not None

    audit = json.loads(
        (reference_root / "reference_search_audit.json").read_text()
    )
    assert audit["temperature_one_search"]["successful_route_counts"] == {
        "below": 0,
        "above": 0,
        "left": 395,
        "right": 0,
    }
    assert audit["gamma_0p3_temperature_diagnostic"]["non_left_successes"] == 0
