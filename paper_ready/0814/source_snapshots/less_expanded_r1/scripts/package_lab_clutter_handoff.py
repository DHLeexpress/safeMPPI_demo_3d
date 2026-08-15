#!/usr/bin/env python3
"""Build an authenticated deployment-only handoff for lab clutter policies."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_deployment.lab_pretrained import load_lab_reference_policy  # noqa: E402
from safe_mppi.config import ObstacleConfig, load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_clutter import LAB_BOUNDS, LAB_GOAL, LAB_START  # noqa: E402
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    LAB_CLUTTER_SCENE_SCHEMA,
    RandomThreeSphereScene,
    scene_sha256,
)
from safe_mppi.lab_visual_flow import (  # noqa: E402
    LAB_VISUAL_CHANNELS,
    LAB_VISUAL_FRAME,
    LAB_VISUAL_GRID_SHAPE,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    spherical_safety_grid,
)


DEFAULT_OUTPUT = ROOT / "flow_deployment" / "minhyuk_clutter_handoff"
LEGACY_HANDOFF = ROOT / "flow_deployment" / "minhyuk_handoff"
CLUTTER_TASK_PROFILE = "minhyuk_lab_random_three_sphere_visual_expansion"
EVALUATION_STATUS = "LAB_CLUTTER_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
FIXED_EVALUATION_STATUS = (
    "LAB_CLUTTER_FIXED_SCENE_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
)
PATH_SPREAD_RESAMPLE_POINTS = 64


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_nonempty_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _portable_source_path(path: Path) -> str:
    """Return a repo-relative source path, or only its basename if external."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _readme_handoff_path(output: Path) -> str:
    """Name the actual handoff location for copy-pasteable README commands."""
    resolved = output.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    path = _require_nonempty_file(path, label)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _check_expected_hash(
    path: Path,
    label: str,
    expected: str | None,
) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _require_keys(value: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise KeyError(f"{label} is missing keys {missing}")


def _validate_output_dir(output: str | Path) -> Path:
    output = Path(output)
    resolved = output.resolve()
    legacy = LEGACY_HANDOFF.resolve()
    if resolved == legacy or legacy in resolved.parents:
        raise ValueError(
            "clutter packaging may not write to or below "
            "flow_deployment/minhyuk_handoff"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    return output


def _validate_template(path: Path):
    template = load_config(path)
    if template.obstacles.spheres or template.obstacles.cylinders:
        raise ValueError(
            "randomized task template must have empty static obstacle lists"
        )
    if not np.allclose(
        template.taskspace.bounds, LAB_BOUNDS, atol=1.0e-9, rtol=0.0,
    ):
        raise ValueError("task template does not use the fixed Minhyuk bounds")
    if not np.allclose(
        template.taskspace.start, LAB_START, atol=1.0e-6, rtol=0.0,
    ):
        raise ValueError("task template does not use the fixed Minhyuk start")
    if not np.allclose(
        template.taskspace.goal, LAB_GOAL, atol=1.0e-6, rtol=0.0,
    ):
        raise ValueError("task template does not use the fixed Minhyuk goal")
    if template.safemppi.z_bias_weight != 0.0:
        raise ValueError("clutter deployment requires zero z-bias weight")
    scene_spec = RandomThreeSphereScene.from_config(template)
    return template, scene_spec


def _validate_concrete_config(
    path: Path,
    template,
    scene_spec: RandomThreeSphereScene,
) -> tuple[Any, np.ndarray, str]:
    concrete = load_config(path)
    randomization = concrete.raw.get("scene_randomization")
    if isinstance(randomization, dict) and bool(randomization.get("enabled")):
        raise ValueError(
            "concrete deployment config must disable scene_randomization"
        )
    if len(concrete.obstacles.spheres) != 3:
        raise ValueError(
            "concrete deployment config must contain exactly three spheres"
        )
    if concrete.obstacles.cylinders:
        raise ValueError(
            "concrete three-sphere deployment config may not contain cylinders"
        )
    for section in ("taskspace", "safemppi", "safety", "data"):
        if _canonical_json(concrete.raw.get(section)) != _canonical_json(
            template.raw.get(section)
        ):
            raise ValueError(
                f"concrete deployment config {section!r} does not match "
                "the randomized task template"
            )

    template_env = TaskEnvironment(template)
    concrete_rows = np.asarray(
        concrete.obstacles.spheres, np.float32,
    ).reshape(3, 4)
    spheres = scene_spec.validate(
        template_env,
        concrete_rows,
    )
    canonical_rows = tuple(tuple(map(float, row)) for row in spheres)
    concrete = replace(
        concrete,
        obstacles=ObstacleConfig(spheres=canonical_rows, cylinders=()),
    )
    concrete_env = TaskEnvironment(concrete)
    for index in range(3):
        reduced = replace(
            concrete,
            obstacles=ObstacleConfig(
                spheres=tuple(
                    row for row_index, row in enumerate(canonical_rows)
                    if row_index != index
                ),
                cylinders=(),
            ),
        )
        full_grid = spherical_safety_grid(
            concrete_env, spheres[index, :3],
        )
        reduced_grid = spherical_safety_grid(
            TaskEnvironment(reduced), spheres[index, :3],
        )
        if np.array_equal(full_grid, reduced_grid):
            raise ValueError(
                f"sphere {index} has no effect on the deployment visual context"
            )
    return concrete, concrete_rows, scene_sha256(concrete_env, concrete_rows)


def _load_pretrained(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("pretrained checkpoint must contain a mapping")
    _require_keys(payload, ("model", "arch", "contract"), "pretrained checkpoint")
    arch = payload["arch"]
    contract = payload["contract"]
    if not isinstance(arch, dict) or not isinstance(contract, dict):
        raise ValueError("pretrained arch and contract must be mappings")
    expected_arch = {
        "kind": LAB_VISUAL_SCHEMA,
        "plan_shape": [10, 3],
        "grid_shape": list(LAB_VISUAL_GRID_SHAPE),
        "grid_channels": list(LAB_VISUAL_CHANNELS),
        "grid_frame": LAB_VISUAL_FRAME,
    }
    for key, expected in expected_arch.items():
        if arch.get(key) != expected:
            raise ValueError(
                f"pretrained visual schema mismatch for {key!r}: "
                f"expected {expected!r}, got {arch.get(key)!r}"
            )
    expected_contract = {
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "deployment_smoothing_and_tracking": "external",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"pretrained deployment contract mismatch for {key!r}"
            )
    policy = load_lab_reference_policy(path).eval()
    if (
        policy.context_schema != LAB_VISUAL_SCHEMA
        or int(policy.context_dim) != LAB_VISUAL_PACKED_DIM
        or tuple(policy.plan_shape) != (10, 3)
    ):
        raise ValueError("pretrained policy does not satisfy the visual lab contract")
    return payload, policy


def _validate_pretrain_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "kind": "lab raw-command reference-flow pretraining",
        "context_model": "visual_hp3d",
        "context_schema": LAB_VISUAL_SCHEMA,
        "external_context_dim": LAB_VISUAL_PACKED_DIM,
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "deployment_smoothing_and_tracking": "external",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"pretrain manifest mismatch for {key!r}: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )


def _validate_expansion(
    pretrained: dict[str, Any],
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    template,
) -> int:
    _require_keys(
        checkpoint,
        ("round", "model", "config", "pretrained"),
        "expansion checkpoint",
    )
    selected_round = int(checkpoint["round"])
    if selected_round < 1 or checkpoint["pretrained"] is not False:
        raise ValueError("selected expansion checkpoint must be a trained round >= 1")
    if manifest.get("status") != "SAFE_FLOW_EXPANSION_COMPLETE":
        raise ValueError("expansion manifest is not complete")
    if manifest.get("task_profile") != CLUTTER_TASK_PROFILE:
        raise ValueError("expansion manifest is not the three-sphere clutter profile")
    conditioning = manifest.get("lab_conditioning")
    if (
        not isinstance(conditioning, dict)
        or conditioning.get("context_schema") != LAB_VISUAL_SCHEMA
        or int(conditioning.get("policy_context_dim", -1))
        != LAB_VISUAL_PACKED_DIM
    ):
        raise ValueError("expansion manifest has an incompatible visual schema")
    verifier = manifest.get("lab_verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("variant") != "full_polytope"
        or verifier.get("full_h_collision_and_green") is not True
    ):
        raise ValueError("clutter expansion must use the full-polytope verifier")
    execution_cost = manifest.get("lab_execution_cost")
    if (
        not isinstance(execution_cost, dict)
        or execution_cost.get("excluded_term")
        != "demonstration-only below-plane z bias"
    ):
        raise ValueError("clutter expansion manifest does not exclude the z-bias cost")
    if _canonical_json(manifest.get("lab_scene_randomization")) != _canonical_json(
        template.raw.get("scene_randomization")
    ):
        raise ValueError(
            "expansion scene randomization does not match the task template"
        )
    ledger = manifest.get("lab_scene_ledger")
    if (
        not isinstance(ledger, list)
        or not ledger
        or any(
            not isinstance(row, dict)
            or row.get("schema") != LAB_CLUTTER_SCENE_SCHEMA
            for row in ledger
        )
    ):
        raise ValueError("expansion manifest lacks a valid clutter scene ledger")
    if _canonical_json(checkpoint["config"]) != _canonical_json(
        manifest.get("config")
    ):
        raise ValueError("expansion checkpoint config and manifest config differ")
    if selected_round > int(manifest["config"].get("rounds", -1)):
        raise ValueError("selected checkpoint round exceeds the completed expansion")

    base_state = pretrained["model"]
    expanded_state = checkpoint["model"]
    if not isinstance(base_state, dict) or not isinstance(expanded_state, dict):
        raise ValueError("checkpoint model states must be mappings")
    if base_state.keys() != expanded_state.keys():
        raise ValueError("pretrained and expanded state keys differ")
    for key in base_state:
        base = base_state[key]
        expanded = expanded_state[key]
        if (
            not isinstance(base, torch.Tensor)
            or not isinstance(expanded, torch.Tensor)
            or base.shape != expanded.shape
            or base.dtype != expanded.dtype
        ):
            raise ValueError(f"incompatible expanded tensor {key!r}")
        if not bool(torch.isfinite(expanded).all()):
            raise ValueError(f"non-finite expanded tensor {key!r}")
    return selected_round


def _expansion_scene_hashes(manifest: dict[str, Any]) -> set[str]:
    hashes = {
        str(row.get("scene_hash", row.get("sha256", "")))
        for row in manifest["lab_scene_ledger"]
    }
    if "" in hashes:
        raise ValueError("expansion scene ledger contains an empty scene hash")
    return hashes


def _recomputed_rollout_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation cell contains no rollout rows")
    successful = [row for row in rows if row.get("status") == "SUCCESS"]
    clearances = [
        float(row["min_clearance_m"])
        for row in successful if row.get("min_clearance_m") is not None
    ]
    times = [
        float(row["time_to_goal_s"])
        for row in successful if row.get("time_to_goal_s") is not None
    ]
    count = len(rows)
    return {
        "episodes": count,
        "SR": sum(row.get("status") == "SUCCESS" for row in rows) / count,
        "CR": sum(row.get("status") == "COLLISION" for row in rows) / count,
        "OOB": sum(row.get("status") == "OOB" for row in rows) / count,
        "timeout": sum(row.get("status") == "TIMEOUT" for row in rows) / count,
        "window_validity": float(np.mean([
            float(row["window_validity"]) for row in rows
        ])),
        "successful_min_clearance_m": (
            float(np.mean(clearances)) if clearances else None
        ),
        "successful_time_to_goal_s": (
            float(np.mean(times)) if times else None
        ),
    }


def _recomputed_fixed_path_spread(
    rows: list[dict[str, Any]],
    label: str,
) -> float | None:
    """Recompute fixed-scene spread from serialized successful path evidence."""
    paths = []
    for row in rows:
        values = row.get("arc_length_resampled_path_xyz")
        if row.get("status") != "SUCCESS":
            if values is not None:
                raise ValueError(
                    f"{label} non-success row carries successful path evidence"
                )
            continue
        path = np.asarray(values, float)
        if (
            path.shape != (PATH_SPREAD_RESAMPLE_POINTS, 3)
            or not bool(np.isfinite(path).all())
        ):
            raise ValueError(
                f"{label} successful path evidence must have shape "
                f"({PATH_SPREAD_RESAMPLE_POINTS},3) and be finite"
            )
        paths.append(path)
    if len(paths) < 2:
        return None
    pairwise = []
    for first in range(len(paths)):
        for second in range(first + 1, len(paths)):
            difference = paths[first] - paths[second]
            pairwise.append(float(np.sqrt(np.mean(np.sum(
                difference * difference, axis=1,
            )))))
    return float(np.mean(pairwise))


def _validate_fixed_path_spread(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    actual = _recomputed_fixed_path_spread(rows, label)
    reported = summary.get("successful_path_spread_m")
    if actual is None:
        matches = reported is None
    else:
        matches = (
            reported is not None
            and bool(np.isclose(
                float(reported), actual, atol=1.0e-12, rtol=0.0,
            ))
        )
    if not matches:
        raise ValueError(
            f"{label} successful path spread disagrees with serialized paths"
        )
    if summary.get("successful_path_spread_domain") != (
        "fixed_scene_single_gamma"
    ):
        raise ValueError(f"{label} has an invalid path-spread domain")


def _validate_rollout_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    recomputed = _recomputed_rollout_metrics(rows)
    for key, actual in recomputed.items():
        if key not in summary:
            raise ValueError(f"{label} summary is missing {key!r}")
        reported = summary[key]
        if actual is None:
            matches = reported is None
        elif key == "episodes":
            matches = int(reported) == actual
        else:
            matches = bool(np.isclose(
                float(reported), float(actual), atol=1.0e-12, rtol=0.0,
            ))
        if not matches:
            raise ValueError(
                f"{label} summary {key!r} disagrees with rollout rows"
            )


def _validate_selected_rollout_summaries(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    pooled = summary.get("pooled")
    per_gamma = summary.get("per_gamma")
    if not isinstance(pooled, dict) or not isinstance(per_gamma, dict):
        raise ValueError(f"{label} lacks pooled/per-gamma summaries")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gamma = f"{float(row.get('gamma')):g}"
        grouped.setdefault(gamma, []).append(row)
    if set(grouped) != set(per_gamma):
        raise ValueError(f"{label} per-gamma summary keys disagree with rows")
    _validate_rollout_summary(pooled, rows, f"{label} pooled")
    for gamma, gamma_rows in grouped.items():
        if not isinstance(per_gamma[gamma], dict):
            raise ValueError(f"{label} gamma={gamma} summary is not a mapping")
        _validate_rollout_summary(
            per_gamma[gamma], gamma_rows, f"{label} gamma={gamma}",
        )


def _validate_randomized_scene_bank(
    scene_bank: dict[str, Any],
    env: TaskEnvironment,
    scene_spec: RandomThreeSphereScene,
    expansion_hashes: set[str],
) -> tuple[set[str], dict[str, np.ndarray]]:
    scenes = scene_bank.get("scenes")
    start_probe = scene_bank.get("start_probe_scene")
    if (
        not isinstance(scenes, list)
        or not scenes
        or not isinstance(start_probe, dict)
    ):
        raise ValueError(
            "randomized evaluation scene bank lacks scenes/start probe"
        )
    by_hash: dict[str, np.ndarray] = {}
    for row in [*scenes, start_probe]:
        if not isinstance(row, dict):
            raise ValueError("randomized evaluation scene rows must be mappings")
        spheres = scene_spec.validate(
            env, np.asarray(row.get("spheres"), np.float32),
        )
        actual_hash = scene_sha256(env, spheres)
        if row.get("scene_hash") != actual_hash:
            raise ValueError(
                "randomized evaluation scene hash disagrees with sphere rows"
            )
        if actual_hash in by_hash:
            raise ValueError("randomized evaluation scene hashes are not unique")
        by_hash[actual_hash] = spheres
    hashes = set(by_hash)
    if hashes & expansion_hashes:
        raise ValueError(
            "randomized evaluation scenes overlap the expansion scene ledger"
        )
    declared_count = scene_bank.get("evaluation_unique_scene_count")
    if declared_count is not None and int(declared_count) != len(hashes):
        raise ValueError("randomized evaluation unique-scene count is incorrect")
    return hashes, by_hash


def _validate_evaluation(
    evaluation: dict[str, Any],
    selected_round: int,
    scene_spec: RandomThreeSphereScene,
    expected_artifacts: dict[str, str],
    template_env: TaskEnvironment,
    expansion_hashes: set[str],
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    if evaluation.get("status") != EVALUATION_STATUS:
        raise ValueError("clutter evaluation is missing or incomplete")
    if (
        float(evaluation.get("sampling_temperature", float("nan"))) != 1.0
        or evaluation.get("sigma_tilt_used") is not False
    ):
        raise ValueError("selected checkpoint requires raw temperature-1 evaluation")
    scene_bank = evaluation.get("scene_bank")
    if (
        not isinstance(scene_bank, dict)
        or scene_bank.get("schema") != LAB_CLUTTER_SCENE_SCHEMA
        or int(scene_bank.get("overlap_count", -1)) != 0
    ):
        raise ValueError("evaluation scene bank is not disjoint clutter data")
    sampler = scene_bank.get("sampler")
    expected_sampler = {
        "obstacle_family": "spheres",
        "count": 3,
        "radius_m": float(scene_spec.radius),
        "minimum_obstacle_surface_gap_m": float(
            scene_spec.minimum_surface_margin
        ),
        "minimum_start_goal_surface_clearance_m": float(
            scene_spec.endpoint_margin
        ),
        "minimum_taskspace_wall_surface_clearance_m": float(
            scene_spec.boundary_surface_margin
        ),
    }
    if not isinstance(sampler, dict):
        raise ValueError("evaluation scene bank lacks its sampler contract")
    for key, expected in expected_sampler.items():
        actual = sampler.get(key)
        if isinstance(expected, float):
            matches = bool(np.isclose(actual, expected, atol=1.0e-9, rtol=0.0))
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"evaluation sampler mismatch for {key!r}")
    randomized_hashes, scene_rows_by_hash = _validate_randomized_scene_bank(
        scene_bank, template_env, scene_spec, expansion_hashes,
    )
    summary = evaluation.get("summary")
    selected = (
        summary.get(str(selected_round))
        if isinstance(summary, dict) else None
    )
    if not isinstance(selected, dict):
        raise ValueError(
            f"evaluation does not contain selected round {selected_round}"
        )
    rows_by_round = evaluation.get("rows")
    selected_rows = (
        rows_by_round.get(str(selected_round))
        if isinstance(rows_by_round, dict) else None
    )
    if not isinstance(selected_rows, list) or not selected_rows:
        raise ValueError("randomized evaluation lacks selected-round rollout rows")
    rollout_hashes = {
        str(row["scene_hash"]) for row in scene_bank["scenes"]
    }
    rows_by_gamma: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        if not isinstance(row, dict):
            raise ValueError("randomized rollout rows must be mappings")
        scene_hash = str(row.get("scene_hash", ""))
        if scene_hash not in rollout_hashes:
            raise ValueError(
                "randomized rollout row uses an undeclared evaluation scene"
            )
        row_spheres = np.asarray(row.get("spheres"), np.float32)
        if (
            row_spheres.shape != (3, 4)
            or not np.array_equal(
                row_spheres, scene_rows_by_hash[scene_hash],
            )
        ):
            raise ValueError(
                "randomized rollout sphere rows disagree with its scene record"
            )
        gamma = f"{float(row.get('gamma')):g}"
        rows_by_gamma.setdefault(gamma, []).append(row)
    for gamma, gamma_rows in rows_by_gamma.items():
        if (
            len(gamma_rows) != len(rollout_hashes)
            or {str(row["scene_hash"]) for row in gamma_rows}
            != rollout_hashes
        ):
            raise ValueError(
                f"randomized gamma={gamma} does not cover the fixed scene bank"
            )
    _validate_selected_rollout_summaries(
        selected, selected_rows, "randomized selected round",
    )
    binding = _validate_evaluation_artifact_binding(
        evaluation,
        selected_round,
        expected_artifacts,
        "randomized evaluation",
    )
    return selected, binding, randomized_hashes


def _validate_evaluation_artifact_binding(
    evaluation: dict[str, Any],
    selected_round: int,
    expected: dict[str, str],
    label: str,
) -> dict[str, Any]:
    binding = evaluation.get("artifact_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"{label} lacks artifact_binding")
    if (
        binding.get("pretrained_checkpoint_sha256")
        != expected["pretrained"]
    ):
        raise ValueError(
            f"{label} is bound to a different pretrained checkpoint"
        )
    if (
        binding.get("pretrain_manifest_sha256")
        != expected["pretrain_manifest"]
    ):
        raise ValueError(
            f"{label} is bound to a different pretrain manifest"
        )
    if (
        binding.get("expansion_manifest_sha256")
        != expected["expansion_manifest"]
    ):
        raise ValueError(
            f"{label} is bound to a different expansion manifest"
        )
    if binding.get("round_zero_model_bitwise_equal_to_pretrained") is not True:
        raise ValueError(
            f"{label} does not prove round-zero/pretrained model equivalence"
        )
    checkpoints = binding.get("checkpoint_sha256_by_round")
    if (
        not isinstance(checkpoints, dict)
        or not isinstance(checkpoints.get("0"), str)
        or len(checkpoints["0"]) != 64
        or checkpoints.get(str(selected_round))
        != expected["expansion_checkpoint"]
    ):
        raise ValueError(
            f"{label} is bound to a different selected checkpoint"
        )
    return binding


def _validate_fixed_scene_evaluation(
    evaluation: dict[str, Any],
    selected_round: int,
    concrete_spheres: np.ndarray,
    concrete_scene_hash: str,
    expected_artifacts: dict[str, str],
    expansion_hashes: set[str],
    randomized_hashes: set[str],
) -> dict[str, Any]:
    if evaluation.get("status") != FIXED_EVALUATION_STATUS:
        raise ValueError("fixed-scene evaluation is missing or incomplete")
    if (
        float(evaluation.get("sampling_temperature", float("nan"))) != 1.0
        or evaluation.get("sigma_tilt_used") is not False
    ):
        raise ValueError(
            "fixed-scene checkpoint evaluation must be raw temperature-1"
        )
    if evaluation.get("common_random_numbers_across_checkpoints") is not True:
        raise ValueError(
            "fixed-scene evaluation must use common random numbers"
        )
    rollouts_per_gamma = int(evaluation.get("rollouts_per_gamma", 0))
    if rollouts_per_gamma < 1:
        raise ValueError("fixed-scene evaluation has no rollouts")
    if int(evaluation.get("path_spread_resample_points", 0)) != (
        PATH_SPREAD_RESAMPLE_POINTS
    ):
        raise ValueError(
            "fixed-scene evaluation has an incompatible path-spread "
            "resampling contract"
        )

    provenance = evaluation.get("scene_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != LAB_CLUTTER_SCENE_SCHEMA
        or provenance.get("preregistered_before_checkpoint_evaluation")
        is not True
        or provenance.get("shared_across_rounds") is not True
        or provenance.get("shared_across_gamma") is not True
        or int(provenance.get("expansion_overlap_count", -1)) != 0
        or int(provenance.get("randomized_evaluation_overlap_count", -1))
        != 0
    ):
        raise ValueError("fixed-scene provenance is incomplete or overlapping")
    scene = provenance.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("fixed-scene provenance lacks its concrete scene")
    if scene.get("scene_hash") != concrete_scene_hash:
        raise ValueError(
            "fixed-scene scene_sha does not match the concrete obstacle map"
        )
    if (
        concrete_scene_hash in expansion_hashes
        or concrete_scene_hash in randomized_hashes
    ):
        raise ValueError(
            "fixed-scene map overlaps expansion or randomized evaluation data"
        )
    try:
        evaluated_spheres = np.asarray(scene["spheres"], np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "fixed-scene provenance lacks valid sphere rows"
        ) from error
    if (
        evaluated_spheres.shape != (3, 4)
        or not np.array_equal(evaluated_spheres, concrete_spheres)
    ):
        raise ValueError(
            "fixed-scene sphere rows do not exactly match the concrete config"
        )
    concrete_binding = evaluation.get("concrete_config")
    if (
        not isinstance(concrete_binding, dict)
        or concrete_binding.get("sha256")
        != expected_artifacts["concrete_config"]
        or concrete_binding.get("scene_hash") != concrete_scene_hash
    ):
        raise ValueError(
            "fixed-scene concrete config binding does not match the supplied map"
        )

    summary = evaluation.get("summary")
    rows_by_round = evaluation.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows_by_round, dict):
        raise ValueError("fixed-scene evaluation lacks summary/rollout mappings")
    selected = summary.get(str(selected_round))
    if not isinstance(selected, dict):
        raise ValueError(
            "fixed-scene evaluation does not contain selected round "
            f"{selected_round}"
        )
    if (
        not isinstance(selected.get("pooled"), dict)
        or not isinstance(selected.get("per_gamma"), dict)
        or not selected["per_gamma"]
    ):
        raise ValueError("fixed-scene selected summary is incomplete")
    seeds = evaluation.get("rollout_seeds_by_gamma")
    if not isinstance(seeds, dict) or set(seeds) != set(selected["per_gamma"]):
        raise ValueError(
            "fixed-scene rollout seeds do not match the per-gamma metrics"
        )
    normalized_seeds: dict[str, list[int]] = {}
    for gamma, values in seeds.items():
        if not isinstance(values, list):
            raise ValueError("fixed-scene rollout seed rows must be lists")
        normalized = [int(value) for value in values]
        if (
            len(normalized) != rollouts_per_gamma
            or len(set(normalized)) != rollouts_per_gamma
        ):
            raise ValueError(
                f"fixed-scene rollout seeds for gamma={gamma} are incomplete"
            )
        normalized_seeds[str(gamma)] = normalized

    def validate_round(round_i: int, label: str):
        cell = summary.get(str(round_i))
        rows = rows_by_round.get(str(round_i))
        if not isinstance(cell, dict) or not isinstance(rows, list) or not rows:
            raise ValueError(
                f"fixed-scene evaluation lacks {label} round {round_i}"
            )
        observed = {gamma: [] for gamma in normalized_seeds}
        successful: dict[str, int | None] = {
            gamma: None for gamma in normalized_seeds
        }
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("fixed-scene rollout rows must be mappings")
            gamma = f"{float(row.get('gamma')):g}"
            if gamma not in observed:
                raise ValueError("fixed-scene rollout has an undeclared gamma")
            if row.get("scene_hash") != concrete_scene_hash:
                raise ValueError(
                    "fixed-scene rollout scene_sha differs from the concrete map"
                )
            row_spheres = np.asarray(row.get("spheres"), np.float32)
            if (
                row_spheres.shape != (3, 4)
                or not np.array_equal(row_spheres, concrete_spheres)
            ):
                raise ValueError(
                    "fixed-scene rollout sphere rows differ from the concrete map"
                )
            seed = int(row["rollout_seed"])
            observed[gamma].append(seed)
            if row.get("status") == "SUCCESS" and successful[gamma] is None:
                successful[gamma] = seed
        if observed != normalized_seeds:
            raise ValueError(
                f"fixed-scene {label} rollout rows and seed ledger differ"
            )
        _validate_selected_rollout_summaries(
            cell, rows, f"fixed-scene {label}",
        )
        rows_by_gamma = {
            gamma: [
                row for row in rows
                if f"{float(row['gamma']):g}" == gamma
            ]
            for gamma in normalized_seeds
        }
        for gamma, gamma_rows in rows_by_gamma.items():
            _validate_fixed_path_spread(
                cell["per_gamma"][gamma],
                gamma_rows,
                f"fixed-scene {label} gamma={gamma}",
            )
        return cell, successful

    pretrained_summary, pretrained_successes = validate_round(
        0, "pretrained",
    )
    selected_summary, selected_successes = validate_round(
        selected_round, "selected expanded",
    )
    binding = _validate_evaluation_artifact_binding(
        evaluation,
        selected_round,
        expected_artifacts,
        "fixed-scene evaluation",
    )
    return {
        "pretrained_summary": pretrained_summary,
        "selected_summary": selected_summary,
        "rollout_seeds_by_gamma": normalized_seeds,
        "successful_seed_by_gamma": {
            "pretrained": pretrained_successes,
            "expanded": selected_successes,
        },
        "scene_provenance": provenance,
        "artifact_binding": binding,
        "concrete_config": concrete_binding,
    }


def _write_packaged_expansion(
    output: Path,
    pretrained: dict[str, Any],
    expansion: dict[str, Any],
    pretrained_sha: str,
    expansion_sha: str,
) -> dict[str, Any]:
    contract = dict(pretrained["contract"])
    contract.update({
        "scope": "experimental_safe_flow_expansion_checkpoint",
        "deployment_safety_qualified": False,
    })
    payload = {
        "model": expansion["model"],
        "arch": pretrained["arch"],
        "contract": contract,
        "provenance": {
            "pretrained_checkpoint_sha256": pretrained_sha,
            "expansion_checkpoint_sha256": expansion_sha,
            "expansion_round": int(expansion["round"]),
            "expansion_config": expansion["config"],
        },
    }
    torch.save(payload, output)
    return payload


def _readme(
    *,
    selected_round: int,
    pretrained_name: str,
    pretrained_sha: str,
    expanded_name: str,
    expanded_sha: str,
    config_name: str,
    config_sha: str,
    fixed_evaluation_sha: str,
    selection_criterion: str,
    successful_seed_by_gamma: dict[str, dict[str, int | None]],
    handoff_path: str,
) -> str:
    gamma_keys = list(successful_seed_by_gamma["pretrained"])
    seed_table = "\n".join(
        "| "
        + gamma
        + " | "
        + (
            "`null`"
            if successful_seed_by_gamma["pretrained"][gamma] is None
            else f"`{successful_seed_by_gamma['pretrained'][gamma]}`"
        )
        + " | "
        + (
            "`null`"
            if successful_seed_by_gamma["expanded"][gamma] is None
            else f"`{successful_seed_by_gamma['expanded'][gamma]}`"
        )
        + " |"
        for gamma in gamma_keys
    )

    def successful_export(
        label: str,
        model_key: str,
        checkpoint_name: str,
        checkpoint_sha: str,
    ) -> str:
        pairs = [
            (gamma, seed)
            for gamma, seed in successful_seed_by_gamma[model_key].items()
            if seed is not None
        ]
        if not pairs:
            return f"{label}: no validated successful fixed-scene seed.\n"
        gammas = " ".join(gamma for gamma, _ in pairs)
        seeds = " ".join(str(seed) for _, seed in pairs)
        return f"""### {label}

```bash
python scripts/export_lab_flow_frozen_references.py \\
  --config "{handoff_path}/{config_name}" \\
  --expected-config-sha256 {config_sha} \\
  --checkpoint "{handoff_path}/{checkpoint_name}" \\
  --expected-checkpoint-sha256 {checkpoint_sha} \\
  --sampling-temperature 1.0 \\
  --gammas {gammas} \\
  --seeds {seeds} \\
  --output outputs/minhyuk_clutter_{model_key}_validated_successes
```
"""

    successful_commands = "\n".join((
        successful_export(
            "Pretrained round 0", "pretrained",
            pretrained_name, pretrained_sha,
        ),
        successful_export(
            f"Expanded round {selected_round}", "expanded",
            expanded_name, expanded_sha,
        ),
    ))
    return f"""# Minhyuk three-sphere clutter deployment handoff

Deployment-only package for one known three-sphere map. The model has no
onboard perception: rebuild this handoff with a new concrete config whenever
the obstacle map changes. The randomized template is provenance only and must
never be passed to a deployment runner.

- selected expansion round: `{selected_round}`
- selection criterion: {selection_criterion}
- pretrained SHA-256: `{pretrained_sha}`
- expanded SHA-256: `{expanded_sha}`
- concrete config SHA-256: `{config_sha}`
- exact-map fixed-scene evaluation SHA-256: `{fixed_evaluation_sha}`
- checkpoint output: raw pre-smoothing acceleration; governor/tracking external
- status: experimental, not flight-safety-qualified

## Deterministic successful fixed-scene examples

These are the first `SUCCESS` rows in evaluator seed order, without manual
trajectory curation. `null` means that the unbiased fixed-scene M-rollout bank
contained no successful example for that model/gamma.

| gamma | pretrained r0 seed | expanded r{selected_round:03d} seed |
|---:|---:|---:|
{seed_table}

{successful_commands}

## Frozen/open-loop reference export

The generator replans against its simulated reference state; the resulting NPZ
is the frozen/open-loop artifact. The seed `91000` below is syntax-only and is
not claimed to be successful; use the validated commands above for successful
examples.

```bash
python scripts/export_lab_flow_frozen_references.py \\
  --config "{handoff_path}/{config_name}" \\
  --expected-config-sha256 {config_sha} \\
  --checkpoint "{handoff_path}/{expanded_name}" \\
  --expected-checkpoint-sha256 {expanded_sha} \\
  --sampling-temperature 1.0 \\
  --gammas 0.1 0.3 0.5 1.0 \\
  --seeds 91000 \\
  --output outputs/minhyuk_clutter_frozen_r{selected_round:03d}
```

Inspect `manifest.json`; process exit alone does not establish success.

## Native closed-loop state-feedback smoke

```bash
python scripts/run_lab_flow_deployment.py \\
  --config "{handoff_path}/{config_name}" \\
  --expected-config-sha256 {config_sha} \\
  --checkpoint "{handoff_path}/{expanded_name}" \\
  --expected-checkpoint-sha256 {expanded_sha} \\
  --sampling-temperature 1.0 \\
  --gamma 0.3 \\
  --seed 91000 \\
  --output outputs/minhyuk_clutter_closed_loop_r{selected_round:03d}
```

This exercises the unchanged native harness offline; it is not a hardware
flight command or safety certificate. Verify `deployment_contract.json`.
Aggregate clearance covers all three spheres, but the unchanged `deploy_sim`
log/GIF displays only the first sphere and has no online obstacle-collision
abort.

## Live hardware controller contract

The repository has no live-flight CLI. Hardware integration must construct the
same authenticated controller and pass `[p_meas, v_ref]` at every 10 Hz replan:

```python
from flow_deployment.lab_pretrained import (
    load_lab_deployment_controller,
    sha256_file,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment

config_path = "{handoff_path}/{config_name}"
if sha256_file(config_path) != "{config_sha}":
    raise ValueError("concrete deployment config SHA-256 mismatch")
cfg = load_config(config_path)
env = TaskEnvironment(cfg)
controller, contract = load_lab_deployment_controller(
    "{handoff_path}/{expanded_name}",
    env,
    sampling_temperature=1.0,
    expected_sha256="{expanded_sha}",
)
action, info = controller.plan(
    state=[px, py, pz, vx_ref, vy_ref, vz_ref],
    goal=env.goal,
    gamma=0.3,
    seed=episode_seed * 100_000 + step,
)
```

To use the byte-identical pretrained baseline instead, replace the checkpoint
with `{pretrained_name}` and its SHA-256 `{pretrained_sha}`. Apply smoothing and
the reference governor exactly once outside the policy.
"""


def _write_sha256s(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in paths
    )
    (output / "SHA256SUMS").write_text(text)


def package_clutter_handoff(
    *,
    pretrained_path: str | Path,
    pretrain_manifest_path: str | Path,
    expansion_checkpoint_path: str | Path,
    expansion_manifest_path: str | Path,
    evaluation_path: str | Path,
    fixed_scene_evaluation_path: str | Path,
    task_template_path: str | Path,
    concrete_config_path: str | Path,
    output_dir: str | Path,
    selection_criterion: str,
    source_revision: str,
    expected_hashes: dict[str, str | None] | None = None,
) -> Path:
    """Validate and write one new, self-contained clutter deployment handoff."""
    output = _validate_output_dir(output_dir)
    if not selection_criterion.strip():
        raise ValueError("selection criterion must be nonempty")
    if not source_revision.strip():
        raise ValueError("source revision must be nonempty")
    inputs = {
        "pretrained": _require_nonempty_file(
            pretrained_path, "pretrained checkpoint",
        ),
        "pretrain_manifest": _require_nonempty_file(
            pretrain_manifest_path, "pretrain manifest",
        ),
        "expansion_checkpoint": _require_nonempty_file(
            expansion_checkpoint_path, "expansion checkpoint",
        ),
        "expansion_manifest": _require_nonempty_file(
            expansion_manifest_path, "expansion manifest",
        ),
        "evaluation": _require_nonempty_file(
            evaluation_path, "raw evaluation",
        ),
        "fixed_scene_evaluation": _require_nonempty_file(
            fixed_scene_evaluation_path, "fixed-scene raw evaluation",
        ),
        "task_template": _require_nonempty_file(
            task_template_path, "randomized task template",
        ),
        "concrete_config": _require_nonempty_file(
            concrete_config_path, "concrete deployment config",
        ),
    }
    expected_hashes = expected_hashes or {}
    input_hashes = {
        label: _check_expected_hash(
            path, label, expected_hashes.get(label),
        )
        for label, path in inputs.items()
    }

    template, scene_spec = _validate_template(inputs["task_template"])
    concrete, spheres, scene_hash = _validate_concrete_config(
        inputs["concrete_config"], template, scene_spec,
    )
    del concrete
    pretrained, policy = _load_pretrained(inputs["pretrained"])
    if not np.isclose(
        float(policy.control_limit),
        float(template.safemppi.demo_u_max),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("pretrained action limit does not match the task template")
    pretrain_manifest = _load_json(
        inputs["pretrain_manifest"], "pretrain manifest",
    )
    _validate_pretrain_manifest(pretrain_manifest)
    expansion = torch.load(
        inputs["expansion_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(expansion, dict):
        raise ValueError("expansion checkpoint must contain a mapping")
    expansion_manifest = _load_json(
        inputs["expansion_manifest"], "expansion manifest",
    )
    selected_round = _validate_expansion(
        pretrained, expansion, expansion_manifest, template,
    )
    expansion_hashes = _expansion_scene_hashes(expansion_manifest)
    template_env = TaskEnvironment(template)
    expected_evaluation_artifacts = {
        "pretrained": input_hashes["pretrained"],
        "pretrain_manifest": input_hashes["pretrain_manifest"],
        "expansion_manifest": input_hashes["expansion_manifest"],
        "expansion_checkpoint": input_hashes["expansion_checkpoint"],
        "concrete_config": input_hashes["concrete_config"],
    }
    evaluation = _load_json(inputs["evaluation"], "raw evaluation")
    (
        selected_summary,
        randomized_binding,
        randomized_hashes,
    ) = _validate_evaluation(
        evaluation,
        selected_round,
        scene_spec,
        expected_evaluation_artifacts,
        template_env,
        expansion_hashes,
    )
    fixed_scene_evaluation = _load_json(
        inputs["fixed_scene_evaluation"], "fixed-scene raw evaluation",
    )
    fixed_evidence = _validate_fixed_scene_evaluation(
        fixed_scene_evaluation,
        selected_round,
        spheres,
        scene_hash,
        expected_evaluation_artifacts,
        expansion_hashes,
        randomized_hashes,
    )
    if _canonical_json(randomized_binding) != _canonical_json(
        fixed_evidence["artifact_binding"]
    ):
        raise ValueError(
            "randomized and fixed-scene artifact_binding mappings differ"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        pretrained_name = "pretrained_visual_clutter_hp3d.pt"
        expanded_name = f"expanded_visual_clutter_r{selected_round:03d}.pt"
        config_name = "concrete_three_sphere_config.json"
        copied_names = {
            "pretrain_manifest": "pretrain_manifest.json",
            "expansion_manifest": "expansion_manifest.json",
            "evaluation": "raw_eval.json",
            "fixed_scene_evaluation": "fixed_scene_raw_eval.json",
            "task_template": "randomized_sphere_template.json",
            "concrete_config": config_name,
        }
        shutil.copyfile(inputs["pretrained"], output / pretrained_name)
        if sha256_file(output / pretrained_name) != input_hashes["pretrained"]:
            raise RuntimeError("pretrained checkpoint copy is not byte-identical")
        for label, name in copied_names.items():
            shutil.copyfile(inputs[label], output / name)
            if sha256_file(output / name) != input_hashes[label]:
                raise RuntimeError(f"{label} copy is not byte-identical")

        packaged = _write_packaged_expansion(
            output / expanded_name,
            pretrained,
            expansion,
            input_hashes["pretrained"],
            input_hashes["expansion_checkpoint"],
        )
        loaded = load_lab_reference_policy(output / expanded_name).eval()
        if (
            loaded.context_schema != LAB_VISUAL_SCHEMA
            or tuple(loaded.plan_shape) != (10, 3)
            or float(loaded.control_limit) != float(template.safemppi.demo_u_max)
        ):
            raise RuntimeError("packaged expansion is not deployment-loadable")
        for key, tensor in expansion["model"].items():
            if not torch.equal(packaged["model"][key], tensor):
                raise RuntimeError(
                    f"packaged expansion changed model tensor {key!r}"
                )
        expanded_sha = sha256_file(output / expanded_name)
        pretrained_sha = sha256_file(output / pretrained_name)
        config_sha = sha256_file(output / config_name)
        model_contracts = {
            "schema": LAB_VISUAL_SCHEMA,
            "external_context_dim": LAB_VISUAL_PACKED_DIM,
            "architecture": pretrained["arch"],
            "pretrained": pretrained["contract"],
            "expanded": packaged["contract"],
        }
        (output / "model_contracts.json").write_text(
            json.dumps(model_contracts, indent=2, allow_nan=False) + "\n"
        )
        handoff_manifest = {
            "status": "MINHYUK_CLUTTER_DEPLOYMENT_HANDOFF_COMPLETE",
            "scope": "deployment_only_known_map; not flight safety qualified",
            "source_revision": source_revision.strip(),
            "packager_sha256": sha256_file(Path(__file__)),
            "selected_round": selected_round,
            "selection_criterion": selection_criterion.strip(),
            "pretrain_binding": {
                "association": (
                    "packaging_time_only; the legacy pretrain manifest did "
                    "not intrinsically hash the checkpoint"
                ),
                "pretrained_checkpoint_sha256": input_hashes["pretrained"],
                "pretrain_manifest_sha256":
                    input_hashes["pretrain_manifest"],
            },
            "evaluation_artifact_binding": randomized_binding,
            "selected_randomized_evaluation_summary": selected_summary,
            "selected_fixed_scene_evaluation": {
                "pretrained_round_000":
                    fixed_evidence["pretrained_summary"],
                f"expanded_round_{selected_round:03d}":
                    fixed_evidence["selected_summary"],
                "rollout_seeds_by_gamma":
                    fixed_evidence["rollout_seeds_by_gamma"],
                "scene_provenance": fixed_evidence["scene_provenance"],
                "concrete_config": fixed_evidence["concrete_config"],
            },
            "successful_seed_by_gamma":
                fixed_evidence["successful_seed_by_gamma"],
            "deployment_map": {
                "scene_schema": LAB_CLUTTER_SCENE_SCHEMA,
                "scene_sha256": scene_hash,
                "sphere_count": 3,
                "spheres": spheres.tolist(),
                "modeled_radius_m": float(scene_spec.radius),
                "vehicle_inflation_already_included": True,
                "randomization_enabled": False,
            },
            "source_paths": {
                label: _portable_source_path(path)
                for label, path in inputs.items()
            },
            "source_sha256": input_hashes,
            "artifacts": {
                pretrained_name: pretrained_sha,
                expanded_name: expanded_sha,
                config_name: config_sha,
                **{
                    name: sha256_file(output / name)
                    for name in copied_names.values()
                    if name != config_name
                },
            },
            "checkpoint_contract": {
                "context_schema": LAB_VISUAL_SCHEMA,
                "context_dim": LAB_VISUAL_PACKED_DIM,
                "plan_shape": [10, 3],
                "control_limit": float(policy.control_limit),
                "nfe": int(policy.nfe),
                "sampling_temperature": 1.0,
                "sampling_temperature_definition": "x0 ~ N(0, tau^2 I)",
                "policy_output": "pre_smoothing_raw_acceleration_command",
                "tracking_and_governor": "external_exactly_once",
            },
        }
        (output / "handoff_manifest.json").write_text(
            json.dumps(handoff_manifest, indent=2, allow_nan=False) + "\n"
        )
        (output / "README.md").write_text(_readme(
            selected_round=selected_round,
            pretrained_name=pretrained_name,
            pretrained_sha=pretrained_sha,
            expanded_name=expanded_name,
            expanded_sha=expanded_sha,
            config_name=config_name,
            config_sha=config_sha,
            fixed_evaluation_sha=input_hashes["fixed_scene_evaluation"],
            selection_criterion=selection_criterion.strip(),
            successful_seed_by_gamma=(
                fixed_evidence["successful_seed_by_gamma"]
            ),
            handoff_path=_readme_handoff_path(output),
        ))
        _write_sha256s(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--pretrain-manifest", type=Path, required=True)
    parser.add_argument("--expansion-checkpoint", type=Path, required=True)
    parser.add_argument("--expansion-manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--fixed-scene-evaluation", type=Path, required=True,
    )
    parser.add_argument("--task-template", type=Path, required=True)
    parser.add_argument("--concrete-config", type=Path, required=True)
    parser.add_argument("--selection-criterion", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    for label in (
        "pretrained",
        "pretrain-manifest",
        "expansion-manifest",
        "evaluation",
        "fixed-scene-evaluation",
        "task-template",
        "concrete-config",
    ):
        parser.add_argument(f"--expected-{label}-sha256")
    parser.add_argument(
        "--expected-expansion-sha256",
        "--expected-expansion-checkpoint-sha256",
        dest="expected_expansion_checkpoint_sha256",
    )
    args = parser.parse_args()
    expected_hashes = {
        "pretrained": args.expected_pretrained_sha256,
        "pretrain_manifest": args.expected_pretrain_manifest_sha256,
        "expansion_checkpoint": args.expected_expansion_checkpoint_sha256,
        "expansion_manifest": args.expected_expansion_manifest_sha256,
        "evaluation": args.expected_evaluation_sha256,
        "fixed_scene_evaluation":
            args.expected_fixed_scene_evaluation_sha256,
        "task_template": args.expected_task_template_sha256,
        "concrete_config": args.expected_concrete_config_sha256,
    }
    output = package_clutter_handoff(
        pretrained_path=args.pretrained,
        pretrain_manifest_path=args.pretrain_manifest,
        expansion_checkpoint_path=args.expansion_checkpoint,
        expansion_manifest_path=args.expansion_manifest,
        evaluation_path=args.evaluation,
        fixed_scene_evaluation_path=args.fixed_scene_evaluation,
        task_template_path=args.task_template,
        concrete_config_path=args.concrete_config,
        output_dir=args.output_dir,
        selection_criterion=args.selection_criterion,
        source_revision=args.source_revision,
        expected_hashes=expected_hashes,
    )
    print(f"[handoff] {output}")
    print((output / "SHA256SUMS").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
