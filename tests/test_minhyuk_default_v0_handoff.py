from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from flow_deployment.lab_pretrained import load_lab_reference_policy


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "flow_deployment/minhyuk_stage1_handoff"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_v0_packaged_checkpoints_load() -> None:
    expected = {
        "stage1_default_v0_best_r3.pt": (
            "dfa4b72a04e3b892bbbe7ea152b1f24a1c4a086a4ccd653f0bf67183027fd69c",
            3,
        ),
        "stage1_default_v0_terminal_r5.pt": (
            "e58b81e03f9276d0e6118ff0f0451bea5bec9b5e5c13da3d4ff957d74ce84ed4",
            5,
        ),
    }
    for name, (digest, round_i) in expected.items():
        path = HANDOFF / "checkpoints" / name
        assert _sha256(path) == digest
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["provenance"]["expansion_round"] == round_i
        assert payload["contract"]["deployment_safety_qualified"] is False
        policy = load_lab_reference_policy(path)
        assert tuple(policy.plan_shape) == (10, 3)
        assert policy.context_schema == (
            "lab_spherical_hp3d_uniform_radial100_planepack_v1"
        )


def test_default_v0_raw_contract_and_metrics_are_exact() -> None:
    contract = json.loads(
        (HANDOFF / "expanded_default_v0_contract.json").read_text()
    )
    assert contract["status"] == "EXPERIMENTAL_DEFAULT_V0_HANDOFF"
    assert contract["selected_checkpoint"]["round"] == 3
    assert contract["selected_checkpoint"]["independent_selection_bank"] is False
    assert contract["raw_evaluation_contract"] == {
        "episodes_per_gamma": 20,
        "gammas": [0.1, 0.3, 0.5, 1.0],
        "seed_formula": "91000 + 37 * episode",
        "sampling_temperature": 1.0,
        "uncertainty_tilting": False,
        "verifier_controller": False,
        "fallback": False,
        "device_used_by_evaluator": "cpu",
        "metrics": "default_v0/raw_eval_m20.json",
    }

    raw = json.loads((HANDOFF / "default_v0/raw_eval_m20.json").read_text())
    r0 = raw["summary"]["0"]["pooled"]
    r3 = raw["summary"]["3"]["pooled"]
    r5 = raw["summary"]["5"]["pooled"]
    assert (r0["SR"], r0["CR"], r0["OOB"]) == (0.1375, 0.8, 0.0625)
    assert (r3["SR"], r3["CR"], r3["OOB"]) == (0.925, 0.025, 0.05)
    assert r3["route_counts"] == {
        "below": 0, "above": 0, "left": 68, "right": 6,
    }
    assert (r5["SR"], r5["CR"], r5["OOB"]) == (0.875, 0.0, 0.075)
    assert r5["route_counts"] == {
        "below": 0, "above": 0, "left": 70, "right": 0,
    }


def test_default_v0_paired_overlay_is_provenance_bound() -> None:
    record = json.loads(
        (HANDOFF / "default_v0/paired_overlay_seed91074.json").read_text()
    )
    assert record["contract"] == {
        "seed": 91074,
        "episode": 2,
        "sampling_temperature": 1.0,
        "raw_sampling": True,
        "uncertainty_tilting": False,
        "verifier_controller": False,
        "fallback": False,
        "device": "cpu",
    }
    r0 = [row for row in record["rows"] if row["checkpoint"] == "r0"]
    r3 = [row for row in record["rows"] if row["checkpoint"] == "r3"]
    assert len(r0) == len(r3) == 4
    assert {row["status"] for row in r0} == {"COLLISION"}
    assert {row["status"] for row in r3} == {"SUCCESS"}
    assert {row["route_mode"] for row in r3} == {"left", "right"}
    for row in record["rows"]:
        path = HANDOFF / row["archive"]
        assert _sha256(path) == row["archive_sha256"]
    for kind in ("png", "pdf"):
        path = HANDOFF / record["figure"][kind]
        assert _sha256(path) == record["figure"][f"{kind}_sha256"]
