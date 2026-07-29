import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_mppi.config import load_config
from safe_mppi.lab_clutter_evaluation import (
    EVALUATION_SCENE_SEED_STRIDE,
    LAB_CLUTTER_TASK_PROFILE,
    START_PROBE_SCENE_SEED_OFFSET,
    _evaluation_scene_provenance,
    _fixed_evaluation_scene_bank,
    evaluate_lab_clutter_expansion,
    is_lab_clutter_evaluation_manifest,
)
from safe_mppi.lab_clutter_expansion import LAB_CLUTTER_SCENE_SCHEMA
from safe_mppi.lab_visual_flow import LAB_VISUAL_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_spheres_ood.json"


def _manifest(scene_hash: str = "expansion-scene-hash") -> dict:
    return {
        "task_profile": LAB_CLUTTER_TASK_PROFILE,
        "lab_conditioning": {"context_schema": LAB_VISUAL_SCHEMA},
        "lab_scene_ledger": [{
            "schema": LAB_CLUTTER_SCENE_SCHEMA,
            "sha256": scene_hash,
        }],
        "config": {"rounds": 0},
    }


def test_clutter_dispatch_requires_task_profile_and_schemas():
    manifest = _manifest()
    assert is_lab_clutter_evaluation_manifest(manifest)

    legacy = copy.deepcopy(manifest)
    legacy["task_profile"] = "minhyuk_lab_ball_visual_expansion"
    assert not is_lab_clutter_evaluation_manifest(legacy)

    wrong_context = copy.deepcopy(manifest)
    wrong_context["lab_conditioning"]["context_schema"] = "lab_raw10_v1"
    with pytest.raises(ValueError, match="visual lab context schema"):
        is_lab_clutter_evaluation_manifest(wrong_context)

    wrong_scene = copy.deepcopy(manifest)
    wrong_scene["lab_scene_ledger"][0]["schema"] = "other_scene_v1"
    with pytest.raises(ValueError, match="scene ledger schema mismatch"):
        is_lab_clutter_evaluation_manifest(wrong_scene)


def test_fixed_evaluation_scene_provenance_is_serializable_and_disjoint():
    config = load_config(CONFIG)
    domain_seed = 91_000
    provenance = _evaluation_scene_provenance(
        config,
        episodes=3,
        domain_seed=domain_seed,
        manifest=_manifest(),
    )
    repeated = _evaluation_scene_provenance(
        config,
        episodes=3,
        domain_seed=domain_seed,
        manifest=_manifest(),
    )

    assert provenance == repeated
    assert provenance["evaluation_seed"] == domain_seed
    assert provenance["configured_sampler_domain_seed"] == 0
    assert provenance["schema"] == LAB_CLUTTER_SCENE_SCHEMA
    assert provenance["overlap_count"] == 0
    assert provenance["shared_across_rounds"] is True
    assert provenance["shared_across_gamma"] is True
    assert [
        row["scene_seed"] for row in provenance["scenes"]
    ] == [
        domain_seed + EVALUATION_SCENE_SEED_STRIDE * episode
        for episode in range(3)
    ]
    assert (
        provenance["start_probe_scene"]["scene_seed"]
        == domain_seed + START_PROBE_SCENE_SEED_OFFSET
    )
    assert provenance["sampler"]["implementation"] == (
        "RandomThreeSphereScene.sample"
    )
    assert provenance["sampler"]["count"] == 3
    json.dumps(provenance, allow_nan=False)


@pytest.mark.parametrize("hash_key", ["sha256", "scene_hash"])
def test_evaluation_scene_provenance_rejects_expansion_overlap(hash_key):
    config = load_config(CONFIG)
    bank = _fixed_evaluation_scene_bank(config, episodes=2, domain_seed=41)
    manifest = _manifest()
    manifest["lab_scene_ledger"] = [{
        "schema": LAB_CLUTTER_SCENE_SCHEMA,
        hash_key: bank["scenes"][1]["scene_hash"],
    }]

    with pytest.raises(ValueError, match="overlaps expansion"):
        _evaluation_scene_provenance(
            config,
            episodes=2,
            domain_seed=41,
            manifest=manifest,
        )


def test_metrics_output_serializes_evaluation_scene_bank(tmp_path, monkeypatch):
    import safe_mppi.lab_clutter_evaluation as evaluation

    config = load_config(CONFIG)
    manifest = _manifest()
    args = SimpleNamespace(
        pretrain_dir=tmp_path,
        expansion=tmp_path,
        episodes=2,
        probe_samples=1,
        stride=1,
        seed=91_000,
        metrics_only=True,
    )

    monkeypatch.setattr(
        evaluation,
        "_checkpoint_policy",
        lambda *args: SimpleNamespace(context_schema=LAB_VISUAL_SCHEMA),
    )

    def fake_raw_rows(policy, task_config, gammas, scene_bank, domain_seed):
        del policy, task_config, domain_seed
        return [{
            "gamma": float(gamma),
            "episode": int(scene["episode"]),
            "scene_seed": int(scene["scene_seed"]),
            "scene_hash": scene["scene_hash"],
            "spheres": scene["spheres"],
            "status": "TIMEOUT",
            "window_validity": 0.0,
            "min_clearance_m": None,
            "time_to_goal_s": None,
        } for gamma in gammas for scene in scene_bank]

    monkeypatch.setattr(evaluation, "_raw_rows", fake_raw_rows)
    monkeypatch.setattr(
        evaluation,
        "_start_probe",
        lambda *args: [{
            "gamma": 0.1,
            "sample": 0,
            "valid": True,
            "margin": 0.1,
            "scene_hash": args[-1]["scene_hash"],
        }],
    )

    evaluate_lab_clutter_expansion(args, config, {}, manifest)

    payload = json.loads((tmp_path / "eval/raw_eval.json").read_text())
    assert payload["scene_bank"]["evaluation_seed"] == 91_000
    assert payload["scene_bank"]["overlap_count"] == 0
    assert len(payload["scene_bank"]["scenes"]) == 2
    assert payload["scene_bank"]["start_probe_scene"]["scene_hash"]
