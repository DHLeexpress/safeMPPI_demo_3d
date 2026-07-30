import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "compare_lab_pretrain_manifests.py"
)
SPEC = importlib.util.spec_from_file_location("gru_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


GAMMAS = (0.1, 0.3, 0.5, 1.0)


def _manifest(
    *,
    id_candidate: bool,
    ood_candidate: bool,
) -> dict:
    def rows(split: str, candidate: bool) -> list[dict]:
        result = []
        for episode in range(20):
            for gamma_index, gamma in enumerate(GAMMAS):
                baseline_success = episode < 8
                if split == "id" and candidate:
                    success = episode < 12
                elif split == "ood" and candidate:
                    success = episode < 7
                else:
                    success = baseline_success
                if success:
                    status = "SUCCESS"
                elif (episode + gamma_index) % 2:
                    status = "COLLISION"
                else:
                    status = "OOB"
                result.append({
                    "gamma": gamma,
                    "episode": episode,
                    "scene_id": f"{split}_{episode:03d}",
                    "scene_hash": f"{split}_hash_{episode:03d}",
                    "status": status,
                    "window_validity": (
                        0.90 if not candidate else 0.89
                    ),
                })
        return result

    return {
        "raw_audit": rows("id", id_candidate),
        "ood_raw_audit": rows("ood", ood_candidate),
    }


def test_gate_passes_and_bootstrap_is_deterministic():
    baseline = _manifest(id_candidate=False, ood_candidate=False)
    candidate = _manifest(id_candidate=True, ood_candidate=True)
    first = MODULE.compare_manifests(
        baseline, candidate, bootstrap_replicates=500, bootstrap_seed=7
    )
    second = MODULE.compare_manifests(
        baseline, candidate, bootstrap_replicates=500, bootstrap_seed=7
    )

    assert first == second
    assert first["passed"]
    assert first["splits"]["id"]["pooled"]["SR"]["delta"] == pytest.approx(
        0.20
    )
    assert first["splits"]["ood"]["pooled"]["SR"]["delta"] == pytest.approx(
        -0.05
    )
    assert (
        first["splits"]["id"]["cluster_bootstrap"]["cluster_count"]
        == 20
    )
    assert (
        first["splits"]["id"]["cluster_bootstrap"]["metrics"]["SR"]["p10"]
        > 0
    )


def test_gate_fails_when_id_gain_is_too_small():
    baseline = _manifest(id_candidate=False, ood_candidate=False)
    candidate = copy.deepcopy(baseline)
    result = MODULE.compare_manifests(
        baseline, candidate, bootstrap_replicates=100, bootstrap_seed=3
    )

    assert not result["passed"]
    assert result["status"] == "GRU_PRETRAIN_GATE_FAIL"
    assert not result["gate"]["criteria"]["id_pooled_sr_delta"]["passed"]
    assert not result["gate"]["criteria"][
        "id_cluster_bootstrap_sr_delta_p10"
    ]["passed"]


def test_pairing_rejects_a_different_scene():
    baseline = _manifest(id_candidate=False, ood_candidate=False)
    candidate = _manifest(id_candidate=True, ood_candidate=True)
    candidate["raw_audit"][0]["scene_hash"] = "different"

    with pytest.raises(ValueError, match="paired row mismatch"):
        MODULE.compare_manifests(
            baseline, candidate, bootstrap_replicates=10
        )


def test_cluster_bootstrap_requires_complete_gamma_sweeps():
    baseline = _manifest(id_candidate=False, ood_candidate=False)
    candidate = _manifest(id_candidate=True, ood_candidate=True)
    baseline["raw_audit"].pop()
    candidate["raw_audit"].pop()

    with pytest.raises(ValueError, match="complete gamma sweep"):
        MODULE.compare_manifests(
            baseline, candidate, bootstrap_replicates=10
        )
