import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi.lab_clutter_evaluation import (
    EVALUATION_SCENE_SEED_STRIDE,
    FIXED_SCENE_ROLLOUT_SEED_OFFSET,
    LAB_CLUTTER_TASK_PROFILE,
    PATH_FOCUSED_MIDPOINT_UNIFORM_SPHERE_SCENE_SCHEMA,
    PATH_FOCUSED_SPHERE_TASK_PROFILE,
    START_PROBE_SCENE_SEED_OFFSET,
    _event_scene_index,
    _evaluation_scene_provenance,
    _fixed_evaluation_scene_bank,
    _fixed_scene_provenance,
    _fixed_scene_rows,
    evaluate_lab_clutter_expansion,
    is_lab_clutter_evaluation_manifest,
    successful_path_spread,
    _validate_round_zero_equivalence,
)
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_SCENE_SCHEMA,
    LabClutterSphereExpansionTask,
)
from safe_mppi.lab_visual_flow import (
    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_SCHEMA,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_SCHEMA,
)


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
    history = copy.deepcopy(manifest)
    history["lab_conditioning"][
        "context_schema"
    ] = LAB_VISUAL_HISTORY_SCHEMA
    history["lab_conditioning"]["history_encoder"] = {
        "present": True,
        "frozen_during_expansion": True,
        "explicit_unfreeze_flag": False,
    }
    assert is_lab_clutter_evaluation_manifest(history)

    radial = copy.deepcopy(manifest)
    radial["lab_conditioning"]["context_schema"] = (
        LAB_RADIAL_VISUAL_SCHEMA
    )
    assert is_lab_clutter_evaluation_manifest(radial)

    radial_history = copy.deepcopy(manifest)
    radial_history["lab_conditioning"]["context_schema"] = (
        LAB_RADIAL_VISUAL_HISTORY_SCHEMA
    )
    radial_history["lab_conditioning"]["history_encoder"] = {
        "present": True,
        "frozen_during_expansion": True,
        "explicit_unfreeze_flag": False,
    }
    assert is_lab_clutter_evaluation_manifest(radial_history)

    del history["lab_conditioning"]["history_encoder"]
    with pytest.raises(ValueError, match="freeze contract"):
        is_lab_clutter_evaluation_manifest(history)

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


def test_midpoint_uniform_path_focused_scene_schema_is_supported():
    manifest = _manifest()
    manifest["task_profile"] = PATH_FOCUSED_SPHERE_TASK_PROFILE
    manifest["lab_scene_ledger"][0]["schema"] = (
        PATH_FOCUSED_MIDPOINT_UNIFORM_SPHERE_SCENE_SCHEMA
    )

    assert is_lab_clutter_evaluation_manifest(manifest)


def test_round_zero_must_equal_pretrained_model_bitwise(tmp_path):
    pretrained = tmp_path / "pretrained.pt"
    checkpoint_zero = tmp_path / "checkpoint_000.pt"
    torch.save(
        {"model": {"weight": torch.tensor([1.0], dtype=torch.float32)}},
        pretrained,
    )
    torch.save(
        {
            "round": 0,
            "pretrained": True,
            "model": {"weight": torch.tensor([1.0], dtype=torch.float32)},
        },
        checkpoint_zero,
    )
    _validate_round_zero_equivalence(pretrained, checkpoint_zero)

    torch.save(
        {
            "round": 0,
            "pretrained": True,
            "model": {"weight": torch.tensor([2.0], dtype=torch.float32)},
        },
        checkpoint_zero,
    )
    with pytest.raises(ValueError, match="tensor differs"):
        _validate_round_zero_equivalence(pretrained, checkpoint_zero)


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


def test_fixed_scene_provenance_rejects_randomized_bank_overlap():
    config = load_config(CONFIG)
    bank = _fixed_evaluation_scene_bank(config, episodes=2, domain_seed=41)
    with pytest.raises(ValueError, match="randomized evaluation bank"):
        _fixed_scene_provenance(
            config,
            scene_seed=bank["scenes"][0]["scene_seed"],
            manifest=_manifest(),
            randomized_scene_bank=bank,
        )


def test_fixed_scene_rows_reuse_exact_independent_seeds(monkeypatch):
    import safe_mppi.lab_clutter_evaluation as evaluation

    config = load_config(CONFIG)
    bank = _fixed_evaluation_scene_bank(config, episodes=1, domain_seed=91_000)
    calls = []

    def fake_rollout(policy, task_config, gamma, seed, sampling_temperature):
        del policy, task_config
        calls.append((float(gamma), int(seed), float(sampling_temperature)))
        return {
            "status": "TIMEOUT",
            "states": np.zeros((2, 6), np.float32),
            "window_validity": 0.0,
            "min_clearance_m": None,
            "time_to_goal_s": None,
        }

    monkeypatch.setattr(evaluation, "raw_reference_rollout", fake_rollout)
    first = _fixed_scene_rows(
        object(), config, [0.1, 0.3], bank["scenes"][0], 3, 17,
    )
    second = _fixed_scene_rows(
        object(), config, [0.1, 0.3], bank["scenes"][0], 3, 17,
    )

    first_seeds = [row["rollout_seed"] for row in first]
    assert first_seeds == [row["rollout_seed"] for row in second]
    assert len(first_seeds) == len(set(first_seeds))
    assert first_seeds[0] == 17 + FIXED_SCENE_ROLLOUT_SEED_OFFSET
    assert all(row["scene_hash"] == first[0]["scene_hash"] for row in first)
    assert all(temperature == 1.0 for _, _, temperature in calls)


def test_successful_path_spread_uses_arc_length_not_time_index():
    coarse = np.column_stack([
        [0.0, 0.5, 1.0],
        np.zeros(3),
        np.zeros(3),
    ])
    dense = np.column_stack([
        np.linspace(0.0, 1.0, 9),
        np.zeros(9),
        np.zeros(9),
    ])
    identical = successful_path_spread([
        {"status": "SUCCESS", "states": coarse},
        {"status": "SUCCESS", "states": dense},
    ])
    assert identical == pytest.approx(0.0, abs=1.0e-12)

    bowed = dense.copy()
    bowed[:, 1] = 0.3 * np.sin(np.pi * bowed[:, 0])
    spread = successful_path_spread([
        {"status": "SUCCESS", "states": dense},
        {"status": "SUCCESS", "states": bowed},
    ])
    assert spread is not None and spread > 0.1


def _compact_event(task, state, gamma, *, step=0):
    context = task.context(state, gamma).detach().cpu().numpy()
    return {
        "round": 1,
        "gamma": float(gamma),
        "episode": 0,
        "step": int(step),
        "robot": np.asarray(state["x"], np.float32),
        "context": np.concatenate([context[:7], context[-18:]]),
    }


def test_event_scene_index_decodes_compact_context_and_rejects_mixing():
    config = load_config(CONFIG)
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_VISUAL_SCHEMA,
        tight_corridor=True,
    )
    first = task.reset(0.3, 0, 123)
    first["previous_applied"][:] = [0.1, -0.2, 0.3]
    second = task.reset(0.3, 0, 124)
    event = _compact_event(task, first, 0.3)
    manifest = _manifest()
    manifest["lab_scene_ledger"] = copy.deepcopy(task.scene_ledger)

    index = _event_scene_index([event], task, manifest)
    decoded = index[(1, 0.3, 0)]
    assert decoded["scene_hash"] == first["scene_hash"]
    assert np.allclose(decoded["previous_applied"], [0.1, -0.2, 0.3])

    mixed = _compact_event(task, second, 0.3, step=1)
    with pytest.raises(ValueError, match="changed within one expansion episode"):
        _event_scene_index([event, mixed], task, manifest)


def test_event_scene_index_decodes_radial_gru_compact_context():
    config = load_config(CONFIG)
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        tight_corridor=True,
    )
    state = task.reset(0.3, 0, 321)
    state["previous_applied"][:] = [0.03, -0.02, 0.01]
    context = task.context(state, 0.3).detach().cpu().numpy()
    event = {
        "round": 1,
        "gamma": 0.3,
        "episode": 0,
        "step": 0,
        "robot": np.asarray(state["x"], np.float32),
        "context": np.concatenate([
            context[:7],
            context[
                LAB_RADIAL_VISUAL_PACKED_DIM:
                LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM
            ],
            context[-18:],
        ]),
    }
    manifest = _manifest()
    manifest["lab_conditioning"]["context_schema"] = (
        LAB_RADIAL_VISUAL_HISTORY_SCHEMA
    )
    manifest["lab_conditioning"]["history_encoder"] = {
        "present": True,
        "frozen_during_expansion": True,
        "explicit_unfreeze_flag": False,
    }
    manifest["lab_scene_ledger"] = copy.deepcopy(task.scene_ledger)

    index = _event_scene_index([event], task, manifest)
    decoded = index[(1, 0.3, 0)]
    assert decoded["scene_hash"] == state["scene_hash"]
    assert np.allclose(
        decoded["previous_applied"],
        [0.03, -0.02, 0.01],
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
    exact_model = {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    torch.save(
        {"model": exact_model},
        tmp_path / "pretrained.pt",
    )
    (tmp_path / "pretrain_manifest.json").write_bytes(
        b'{"kind":"exact pretrain manifest"}\n'
    )
    torch.save(
        {
            "round": 0,
            "pretrained": True,
            "model": exact_model,
        },
        tmp_path / "checkpoint_000.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
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
        "raw_reference_rollout",
        lambda *args, **kwargs: {
            "status": "TIMEOUT",
            "states": np.zeros((2, 6), np.float32),
            "window_validity": 0.0,
            "min_clearance_m": None,
            "time_to_goal_s": None,
        },
    )
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
    binding = payload["artifact_binding"]
    assert binding["pretrained_checkpoint_sha256"] == hashlib.sha256(
        (tmp_path / "pretrained.pt").read_bytes()
    ).hexdigest()
    assert binding["pretrain_manifest_sha256"] == hashlib.sha256(
        b'{"kind":"exact pretrain manifest"}\n'
    ).hexdigest()
    assert binding["checkpoint_sha256_by_round"]["0"] == hashlib.sha256(
        (tmp_path / "checkpoint_000.pt").read_bytes()
    ).hexdigest()
    assert binding["round_zero_model_bitwise_equal_to_pretrained"] is True
    assert (
        payload["summary"]["0"]["pooled"]["successful_path_spread_m"]
        is None
    )
    assert payload["summary"]["0"]["pooled"][
        "successful_path_spread_domain"
    ] == "not_applicable_cross_scene_or_gamma"

    fixed = json.loads(
        (tmp_path / "eval/fixed_scene_raw_eval.json").read_text()
    )
    assert fixed["rollouts_per_gamma"] == 10
    assert fixed["path_spread_resample_points"] == 64
    assert fixed["common_random_numbers_across_checkpoints"] is True
    assert fixed["artifact_binding"] == binding
    concrete_path = tmp_path / "eval/fixed_scene_config.json"
    concrete = load_config(concrete_path)
    assert len(concrete.obstacles.spheres) == 3
    assert concrete.raw["scene_randomization"]["enabled"] is False
    assert fixed["concrete_config"]["sha256"] == hashlib.sha256(
        concrete_path.read_bytes()
    ).hexdigest()
    assert (
        fixed["concrete_config"]["scene_hash"]
        == fixed["scene_provenance"]["scene"]["scene_hash"]
    )
    assert set(fixed["rollout_seeds_by_gamma"]) == {
        "0.1", "0.3", "0.5", "1"
    }
    assert fixed["summary"]["0"]["per_gamma"]["0.1"][
        "successful_path_spread_domain"
    ] == "fixed_scene_single_gamma"
    assert all(
        row["arc_length_resampled_path_xyz"] is None
        for row in fixed["rows"]["0"]
    )
