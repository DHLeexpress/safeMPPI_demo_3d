from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flow_deployment.lab_pretrained import load_lab_reference_policy
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_SCENE_SCHEMA,
    RandomThreeSphereScene,
    scene_sha256,
)
from safe_mppi.lab_visual_flow import (
    LAB_VISUAL_CHANNELS,
    LAB_VISUAL_FRAME,
    LAB_VISUAL_GRID_SHAPE,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    LabVisualFlowPolicy,
)
from scripts.package_lab_clutter_handoff import (
    LEGACY_HANDOFF,
    package_clutter_handoff,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_TEMPLATE = ROOT / "configs" / "lab_clutter_spheres_ood.json"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rollout_summary(rows: list[dict], *, spread=None) -> dict:
    successful = [row for row in rows if row["status"] == "SUCCESS"]
    clearances = [
        row["min_clearance_m"] for row in successful
        if row["min_clearance_m"] is not None
    ]
    times = [
        row["time_to_goal_s"] for row in successful
        if row["time_to_goal_s"] is not None
    ]
    count = len(rows)
    return {
        "episodes": count,
        "SR": sum(row["status"] == "SUCCESS" for row in rows) / count,
        "CR": sum(row["status"] == "COLLISION" for row in rows) / count,
        "OOB": sum(row["status"] == "OOB" for row in rows) / count,
        "timeout": sum(row["status"] == "TIMEOUT" for row in rows) / count,
        "window_validity": float(np.mean([
            row["window_validity"] for row in rows
        ])),
        "successful_min_clearance_m": (
            float(np.mean(clearances)) if clearances else None
        ),
        "successful_time_to_goal_s": (
            float(np.mean(times)) if times else None
        ),
        "successful_path_spread_m": spread,
    }


def _valid_sources(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    template_raw = json.loads(TASK_TEMPLATE.read_text())
    template_path = tmp_path / "task_template.json"
    _write_json(template_path, template_raw)
    template = load_config(template_path)
    env = TaskEnvironment(template)
    scene_spec = RandomThreeSphereScene.from_config(template)
    spheres = scene_spec.sample(env, 41)

    concrete_raw = deepcopy(template_raw)
    concrete_raw["obstacles"] = {
        "spheres": spheres.tolist(),
        "cylinders": [],
    }
    concrete_raw["scene_randomization"] = {
        "enabled": False,
        "contract": "fixed_known_map_for_deployment",
    }
    concrete_path = tmp_path / "concrete.json"
    _write_json(concrete_path, concrete_raw)

    policy = LabVisualFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=4,
        control_limit=template.safemppi.demo_u_max,
        nfe=2,
    )
    arch = {
        "kind": LAB_VISUAL_SCHEMA,
        "plan_shape": [10, 3],
        "hidden": 8,
        "representation_dim": 4,
        "grid_token_dim": 4,
        "grid_shape": list(LAB_VISUAL_GRID_SHAPE),
        "grid_channels": list(LAB_VISUAL_CHANNELS),
        "grid_frame": LAB_VISUAL_FRAME,
        "control_limit": template.safemppi.demo_u_max,
        "nfe": 2,
        "trunk_depth": 2,
        "time_features": "raw1",
    }
    contract = {
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "deployment_smoothing_and_tracking": "external",
    }
    pretrained_path = tmp_path / "pretrained.pt"
    torch.save({
        "model": policy.state_dict(),
        "arch": arch,
        "contract": contract,
    }, pretrained_path)

    pretrain_manifest_path = tmp_path / "pretrain_manifest.json"
    _write_json(pretrain_manifest_path, {
        "kind": "lab raw-command reference-flow pretraining",
        "context_model": "visual_hp3d",
        "context_schema": LAB_VISUAL_SCHEMA,
        "external_context_dim": LAB_VISUAL_PACKED_DIM,
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "deployment_smoothing_and_tracking": "external",
    })

    expanded_state = {
        key: value.detach().clone() for key, value in policy.state_dict().items()
    }
    first = next(iter(expanded_state.values()))
    first.reshape(-1)[0] += 0.01
    expansion_config = {
        "rounds": 2,
        "gammas": [0.1, 0.3, 0.5, 1.0],
        "execution_rule": "min_cost",
    }
    expansion_checkpoint_path = tmp_path / "checkpoint_002.pt"
    torch.save({
        "round": 2,
        "model": expanded_state,
        "config": expansion_config,
        "pretrained": False,
    }, expansion_checkpoint_path)

    expansion_scene = scene_spec.sample(env, 83)
    expansion_manifest_path = tmp_path / "expansion_manifest.json"
    _write_json(expansion_manifest_path, {
        "status": "SAFE_FLOW_EXPANSION_COMPLETE",
        "task_profile": "minhyuk_lab_random_three_sphere_visual_expansion",
        "config": expansion_config,
        "lab_conditioning": {
            "context_schema": LAB_VISUAL_SCHEMA,
            "policy_context_dim": LAB_VISUAL_PACKED_DIM,
        },
        "lab_verifier": {
            "variant": "full_polytope",
            "full_h_collision_and_green": True,
        },
        "lab_execution_cost": {
            "excluded_term": "demonstration-only below-plane z bias",
        },
        "lab_scene_randomization": template_raw["scene_randomization"],
        "lab_scene_ledger": [{
            "schema": LAB_CLUTTER_SCENE_SCHEMA,
            "scene_hash": scene_sha256(env, expansion_scene),
        }],
    })
    artifact_binding = {
        "pretrained_checkpoint_sha256": _sha256(pretrained_path),
        "pretrain_manifest_sha256": _sha256(pretrain_manifest_path),
        "expansion_manifest_sha256": _sha256(expansion_manifest_path),
        "round_zero_model_bitwise_equal_to_pretrained": True,
        "checkpoint_sha256_by_round": {
            "0": "a" * 64,
            "2": _sha256(expansion_checkpoint_path),
        },
    }

    gamma_names = ["0.1", "0.3", "0.5", "1"]
    randomized_spheres = [
        scene_spec.sample(env, seed) for seed in (101, 102)
    ]
    randomized_scenes = [{
        "episode": index,
        "scene_seed": 101 + index,
        "scene_hash": scene_sha256(env, values),
        "spheres": values.tolist(),
    } for index, values in enumerate(randomized_spheres)]
    probe_spheres = scene_spec.sample(env, 103)
    randomized_rows = []
    for gamma in gamma_names:
        for index, scene in enumerate(randomized_scenes):
            success = index == 0
            randomized_rows.append({
                "gamma": float(gamma),
                "episode": index,
                "scene_hash": scene["scene_hash"],
                "spheres": scene["spheres"],
                "status": "SUCCESS" if success else "COLLISION",
                "window_validity": 1.0 if success else 0.5,
                "min_clearance_m": 0.2 if success else -0.01,
                "time_to_goal_s": 1.5 if success else None,
            })
    randomized_per_gamma = {
        gamma: _rollout_summary([
            row for row in randomized_rows
            if f"{row['gamma']:g}" == gamma
        ])
        for gamma in gamma_names
    }
    evaluation_path = tmp_path / "raw_eval.json"
    _write_json(evaluation_path, {
        "status": "LAB_CLUTTER_RAW_TEMPERATURE1_EVALUATION_COMPLETE",
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "artifact_binding": artifact_binding,
        "scene_bank": {
            "schema": LAB_CLUTTER_SCENE_SCHEMA,
            "overlap_count": 0,
            "evaluation_unique_scene_count": 3,
            "scenes": randomized_scenes,
            "start_probe_scene": {
                "scene_seed": 103,
                "scene_hash": scene_sha256(env, probe_spheres),
                "spheres": probe_spheres.tolist(),
            },
            "sampler": {
                "obstacle_family": "spheres",
                "count": 3,
                "radius_m": scene_spec.radius,
                "minimum_obstacle_surface_gap_m":
                    scene_spec.minimum_surface_margin,
                "minimum_start_goal_surface_clearance_m":
                    scene_spec.endpoint_margin,
                "minimum_taskspace_wall_surface_clearance_m":
                    scene_spec.boundary_surface_margin,
            },
        },
        "summary": {
            "2": {
                "pooled": _rollout_summary(randomized_rows),
                "per_gamma": randomized_per_gamma,
            },
        },
        "rows": {"2": randomized_rows},
    })
    fixed_seeds = {
        gamma: [200_000 + 100 * index, 200_001 + 100 * index]
        for index, gamma in enumerate(gamma_names)
    }
    fixed_rows_by_round = {"0": [], "2": []}
    for gamma in gamma_names:
        for rollout, seed in enumerate(fixed_seeds[gamma]):
            pretrained_success = gamma != "1" and rollout == 1
            expanded_success = gamma != "1"
            path = np.column_stack([
                np.linspace(0.0, 1.0, 64),
                np.full(64, 0.12 * rollout),
                np.zeros(64),
            ]).tolist()
            for round_name, success, failure in (
                ("0", pretrained_success, "TIMEOUT"),
                ("2", expanded_success, "COLLISION"),
            ):
                fixed_rows_by_round[round_name].append({
                    "gamma": float(gamma),
                    "rollout_seed": seed,
                    "scene_hash": scene_sha256(env, spheres),
                    "spheres": spheres.tolist(),
                    "status": "SUCCESS" if success else failure,
                    "window_validity": 1.0 if success else 0.5,
                    "min_clearance_m": 0.2 if success else None,
                    "time_to_goal_s": 1.5 if success else None,
                    "arc_length_resampled_path_xyz": path if success else None,
                })
    fixed_summaries = {}
    for round_name, rows in fixed_rows_by_round.items():
        per_gamma = {}
        for gamma in gamma_names:
            gamma_rows = [
                row for row in rows if f"{row['gamma']:g}" == gamma
            ]
            success_count = sum(
                row["status"] == "SUCCESS" for row in gamma_rows
            )
            per_gamma[gamma] = _rollout_summary(
                gamma_rows,
                spread=(0.12 if success_count >= 2 else None),
            )
            per_gamma[gamma]["successful_path_spread_domain"] = (
                "fixed_scene_single_gamma"
            )
        fixed_summaries[round_name] = {
            "pooled": _rollout_summary(rows),
            "per_gamma": per_gamma,
        }
    fixed_scene_evaluation_path = tmp_path / "fixed_scene_raw_eval.json"
    _write_json(fixed_scene_evaluation_path, {
        "status":
            "LAB_CLUTTER_FIXED_SCENE_RAW_TEMPERATURE1_EVALUATION_COMPLETE",
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "artifact_binding": artifact_binding,
        "rollouts_per_gamma": 2,
        "path_spread_resample_points": 64,
        "common_random_numbers_across_checkpoints": True,
        "concrete_config": {
            "file": concrete_path.name,
            "sha256": _sha256(concrete_path),
            "scene_hash": scene_sha256(env, spheres),
        },
        "rollout_seeds_by_gamma": fixed_seeds,
        "scene_provenance": {
            "schema": LAB_CLUTTER_SCENE_SCHEMA,
            "preregistered_before_checkpoint_evaluation": True,
            "shared_across_rounds": True,
            "shared_across_gamma": True,
            "scene": {
                "scene_seed": 41,
                "scene_hash": scene_sha256(env, spheres),
                "spheres": spheres.tolist(),
            },
            "expansion_overlap_count": 0,
            "randomized_evaluation_overlap_count": 0,
        },
        "summary": fixed_summaries,
        "rows": fixed_rows_by_round,
    })
    return {
        "pretrained_path": pretrained_path,
        "pretrain_manifest_path": pretrain_manifest_path,
        "expansion_checkpoint_path": expansion_checkpoint_path,
        "expansion_manifest_path": expansion_manifest_path,
        "evaluation_path": evaluation_path,
        "fixed_scene_evaluation_path": fixed_scene_evaluation_path,
        "task_template_path": template_path,
        "concrete_config_path": concrete_path,
    }


def _package(
    sources: dict[str, Path],
    output: Path,
    **kwargs,
) -> Path:
    return package_clutter_handoff(
        **sources,
        output_dir=output,
        selection_criterion=kwargs.pop(
            "selection_criterion",
            "lowest collision rate, then highest success rate",
        ),
        source_revision="test-revision",
        **kwargs,
    )


def test_packages_authenticated_deployment_only_handoff(tmp_path: Path) -> None:
    sources = _valid_sources(tmp_path)
    output = _package(sources, tmp_path / "handoff")

    pretrained = output / "pretrained_visual_clutter_hp3d.pt"
    expanded = output / "expanded_visual_clutter_r002.pt"
    config = output / "concrete_three_sphere_config.json"
    assert pretrained.read_bytes() == sources["pretrained_path"].read_bytes()
    assert (
        (output / "fixed_scene_raw_eval.json").read_bytes()
        == sources["fixed_scene_evaluation_path"].read_bytes()
    )
    assert load_config(config).obstacles.spheres
    assert len(load_config(config).obstacles.spheres) == 3
    assert not load_config(config).raw["scene_randomization"]["enabled"]

    raw = torch.load(
        sources["expansion_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    packaged = torch.load(expanded, map_location="cpu", weights_only=False)
    assert packaged["provenance"]["expansion_round"] == 2
    assert (
        packaged["contract"]["scope"]
        == "experimental_safe_flow_expansion_checkpoint"
    )
    assert packaged["contract"]["deployment_safety_qualified"] is False
    for key in raw["model"]:
        assert torch.equal(packaged["model"][key], raw["model"][key])
    policy = load_lab_reference_policy(expanded)
    assert policy.context_schema == LAB_VISUAL_SCHEMA
    assert tuple(policy.plan_shape) == (10, 3)
    assert policy.control_limit == 0.3

    manifest = json.loads((output / "handoff_manifest.json").read_text())
    assert manifest["selected_round"] == 2
    assert len(manifest["packager_sha256"]) == 64
    assert manifest["evaluation_artifact_binding"][
        "checkpoint_sha256_by_round"
    ]["2"] == _sha256(sources["expansion_checkpoint_path"])
    assert manifest["deployment_map"]["sphere_count"] == 3
    assert manifest["deployment_map"]["randomization_enabled"] is False
    assert (
        manifest["selected_randomized_evaluation_summary"]["pooled"]["SR"]
        == 0.5
    )
    fixed = manifest["selected_fixed_scene_evaluation"]
    assert fixed["pretrained_round_000"]["pooled"]["SR"] == 0.375
    assert fixed["expanded_round_002"]["pooled"]["SR"] == 0.75
    assert (
        fixed["expanded_round_002"]["per_gamma"]["0.3"][
            "successful_path_spread_m"
        ]
        == 0.12
    )
    assert fixed["rollout_seeds_by_gamma"]["0.3"] == [200_100, 200_101]
    assert fixed["concrete_config"]["sha256"] == _sha256(
        sources["concrete_config_path"]
    )
    assert (
        manifest["source_sha256"]["expansion_checkpoint"]
        == _sha256(sources["expansion_checkpoint_path"])
    )
    assert manifest["successful_seed_by_gamma"]["pretrained"]["0.3"] == 200_101
    assert manifest["successful_seed_by_gamma"]["expanded"]["0.3"] == 200_100
    assert manifest["successful_seed_by_gamma"]["pretrained"]["1"] is None
    assert manifest["successful_seed_by_gamma"]["expanded"]["1"] is None
    assert (
        manifest["pretrain_binding"]["pretrain_manifest_sha256"]
        == _sha256(sources["pretrain_manifest_path"])
    )
    assert manifest["source_paths"] == {
        "pretrained": sources["pretrained_path"].name,
        "pretrain_manifest": sources["pretrain_manifest_path"].name,
        "expansion_checkpoint": sources["expansion_checkpoint_path"].name,
        "expansion_manifest": sources["expansion_manifest_path"].name,
        "evaluation": sources["evaluation_path"].name,
        "fixed_scene_evaluation":
            sources["fixed_scene_evaluation_path"].name,
        "task_template": sources["task_template_path"].name,
        "concrete_config": sources["concrete_config_path"].name,
    }
    assert not any(
        Path(path).is_absolute()
        for path in manifest["source_paths"].values()
    )

    readme = (output / "README.md").read_text()
    assert str(output.resolve()) in readme
    assert _sha256(expanded) in readme
    assert _sha256(config) in readme
    assert _sha256(sources["fixed_scene_evaluation_path"]) in readme
    assert "--expected-config-sha256" in readme
    assert "export_lab_flow_frozen_references.py" in readme
    assert "run_lab_flow_deployment.py" in readme
    assert "repository has no live-flight CLI" in readme
    assert "Pretrained round 0" in readme
    assert "--seeds 200001 200101 200201" in readme
    assert "--seeds 200000 200100 200200" in readme
    assert "| 1 | `null` | `null` |" in readme
    assert "seed `91000` below is syntax-only" in readme

    checksums = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
        assert _sha256(output / name) == digest
    assert set(checksums) == {
        path.name for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }


@pytest.mark.parametrize(
    "case,match",
    [
        ("empty", "exactly three spheres"),
        ("wrong_radius", "radii do not match"),
        ("randomized", "must disable scene_randomization"),
        ("taskspace", "taskspace.*does not match"),
    ],
)
def test_rejects_missing_or_mismatched_concrete_map(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    sources = _valid_sources(tmp_path)
    path = sources["concrete_config_path"]
    config = json.loads(path.read_text())
    if case == "empty":
        config["obstacles"]["spheres"] = []
    elif case == "wrong_radius":
        config["obstacles"]["spheres"][0][3] -= 0.01
    elif case == "randomized":
        config["scene_randomization"]["enabled"] = True
    else:
        config["taskspace"]["goal"][0] -= 0.1
    _write_json(path, config)

    output = tmp_path / "handoff"
    with pytest.raises(ValueError, match=match):
        _package(sources, output)
    assert not output.exists()


def test_rejects_incompatible_visual_schema_and_unevaluated_round(
    tmp_path: Path,
) -> None:
    sources = _valid_sources(tmp_path)
    checkpoint = torch.load(
        sources["pretrained_path"], map_location="cpu", weights_only=False,
    )
    checkpoint["arch"]["kind"] = "conditional_flow_mlp"
    torch.save(checkpoint, sources["pretrained_path"])
    with pytest.raises(ValueError, match="visual schema mismatch"):
        _package(sources, tmp_path / "wrong_schema")

    sources = _valid_sources(tmp_path / "second")
    evaluation = json.loads(sources["evaluation_path"].read_text())
    del evaluation["summary"]["2"]
    _write_json(sources["evaluation_path"], evaluation)
    with pytest.raises(ValueError, match="does not contain selected round 2"):
        _package(sources, tmp_path / "unevaluated")


@pytest.mark.parametrize(
    "case,match",
    [
        ("scene_hash", "scene_sha does not match"),
        ("sphere_rows", "sphere rows do not exactly match"),
        ("selected_round", "does not contain selected round 2"),
        ("config_binding", "concrete config binding does not match"),
    ],
)
def test_rejects_fixed_scene_evaluation_for_a_different_or_missing_map(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    sources = _valid_sources(tmp_path)
    path = sources["fixed_scene_evaluation_path"]
    evaluation = json.loads(path.read_text())
    if case == "scene_hash":
        evaluation["scene_provenance"]["scene"]["scene_hash"] = "0" * 64
    elif case == "sphere_rows":
        evaluation["scene_provenance"]["scene"]["spheres"].reverse()
    elif case == "selected_round":
        del evaluation["summary"]["2"]
    else:
        evaluation["concrete_config"]["sha256"] = "0" * 64
    _write_json(path, evaluation)

    output = tmp_path / "fixed_scene_mismatch"
    with pytest.raises(ValueError, match=match):
        _package(sources, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "case,match",
    [
        ("pretrained", "different pretrained checkpoint"),
        ("pretrain_manifest", "different pretrain manifest"),
        ("manifest", "different expansion manifest"),
        ("checkpoint", "different selected checkpoint"),
        ("round_zero", "round-zero/pretrained model equivalence"),
        ("mapping", "artifact_binding mappings differ"),
    ],
)
def test_rejects_evaluation_bound_to_different_artifacts(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    sources = _valid_sources(tmp_path)
    raw_path = sources["evaluation_path"]
    fixed_path = sources["fixed_scene_evaluation_path"]
    raw = json.loads(raw_path.read_text())
    fixed = json.loads(fixed_path.read_text())
    if case == "pretrained":
        raw["artifact_binding"]["pretrained_checkpoint_sha256"] = "0" * 64
    elif case == "pretrain_manifest":
        raw["artifact_binding"]["pretrain_manifest_sha256"] = "0" * 64
    elif case == "manifest":
        fixed["artifact_binding"]["expansion_manifest_sha256"] = "0" * 64
    elif case == "checkpoint":
        raw["artifact_binding"]["checkpoint_sha256_by_round"]["2"] = "0" * 64
    elif case == "round_zero":
        raw["artifact_binding"][
            "round_zero_model_bitwise_equal_to_pretrained"
        ] = False
    else:
        fixed["artifact_binding"]["checkpoint_sha256_by_round"]["0"] = "1" * 64
    _write_json(raw_path, raw)
    _write_json(fixed_path, fixed)

    output = tmp_path / "artifact_mismatch"
    with pytest.raises(ValueError, match=match):
        _package(sources, output)
    assert not output.exists()


@pytest.mark.parametrize("domain", ["randomized", "fixed"])
def test_rejects_metrics_that_disagree_with_bound_rollout_rows(
    tmp_path: Path,
    domain: str,
) -> None:
    sources = _valid_sources(tmp_path)
    path = (
        sources["evaluation_path"]
        if domain == "randomized"
        else sources["fixed_scene_evaluation_path"]
    )
    evaluation = json.loads(path.read_text())
    evaluation["summary"]["2"]["pooled"]["SR"] = 0.123
    _write_json(path, evaluation)

    output = tmp_path / f"{domain}_summary_mismatch"
    with pytest.raises(ValueError, match="summary 'SR' disagrees"):
        _package(sources, output)
    assert not output.exists()


def test_rejects_fixed_scene_path_spread_tampering(tmp_path: Path) -> None:
    sources = _valid_sources(tmp_path)
    path = sources["fixed_scene_evaluation_path"]
    evaluation = json.loads(path.read_text())
    row = next(
        row for row in evaluation["rows"]["2"]
        if row["status"] == "SUCCESS" and row["gamma"] == 0.1
    )
    row["arc_length_resampled_path_xyz"][0][1] += 0.5
    _write_json(path, evaluation)

    output = tmp_path / "path_spread_mismatch"
    with pytest.raises(
        ValueError,
        match="successful path spread disagrees with serialized paths",
    ):
        _package(sources, output)
    assert not output.exists()


def test_recomputes_fixed_scene_overlap_from_bound_expansion_manifest(
    tmp_path: Path,
) -> None:
    sources = _valid_sources(tmp_path)
    fixed = json.loads(sources["fixed_scene_evaluation_path"].read_text())
    fixed_hash = fixed["scene_provenance"]["scene"]["scene_hash"]
    manifest = json.loads(sources["expansion_manifest_path"].read_text())
    manifest["lab_scene_ledger"][0]["scene_hash"] = fixed_hash
    _write_json(sources["expansion_manifest_path"], manifest)
    manifest_sha = _sha256(sources["expansion_manifest_path"])
    for label in ("evaluation_path", "fixed_scene_evaluation_path"):
        evaluation = json.loads(sources[label].read_text())
        evaluation["artifact_binding"]["expansion_manifest_sha256"] = (
            manifest_sha
        )
        _write_json(sources[label], evaluation)

    output = tmp_path / "overlapping_fixed_scene"
    with pytest.raises(
        ValueError, match="fixed-scene map overlaps expansion",
    ):
        _package(sources, output)
    assert not output.exists()


def test_rejects_hash_mismatch_and_legacy_handoff_target(
    tmp_path: Path,
) -> None:
    sources = _valid_sources(tmp_path)
    with pytest.raises(ValueError, match="pretrained SHA-256 mismatch"):
        _package(
            sources,
            tmp_path / "hash_mismatch",
            expected_hashes={"pretrained": "0" * 64},
        )

    sentinel = _sha256(LEGACY_HANDOFF / "README.md")
    with pytest.raises(ValueError, match="minhyuk_handoff"):
        _package(sources, LEGACY_HANDOFF)
    assert _sha256(LEGACY_HANDOFF / "README.md") == sentinel
