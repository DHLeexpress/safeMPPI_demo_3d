"""Assemble disjoint-gamma success-quota archives without rewriting data."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .acquire import aggregate_metrics
from .config import load_config
from .lab_clutter import summarize_start_goal_path_diagnostics
from .path_focused_collection import SUCCESS_QUOTA_SCHEMA


ASSEMBLY_SCHEMA = "path_focused_quota_archive_assembly_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_recipe(config: dict) -> dict:
    result = copy.deepcopy(config)
    data = result.setdefault("data", {})
    data.pop("gammas", None)
    data.pop("max_attempts_per_gamma", None)
    return result


def _publish_file(source: Path, destination: Path) -> tuple[str, str]:
    source_hash = _sha256(source)
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy"
        if _sha256(destination) != source_hash:
            raise RuntimeError(f"copied trajectory hash mismatch: {source}")
    return method, source_hash


def _load_source(path: Path) -> dict:
    required = (
        "manifest.json", "resolved_config.json", "quota_contract.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"quota archive {path} lacks {missing}")
    manifest = json.loads((path / "manifest.json").read_text())
    resolved = json.loads((path / "resolved_config.json").read_text())
    contract = json.loads((path / "quota_contract.json").read_text())
    if manifest.get("status") != "COMPLETE_EXACT_SUCCESS_QUOTA":
        raise ValueError(f"quota archive is not complete: {path}")
    if contract.get("schema") != SUCCESS_QUOTA_SCHEMA:
        raise ValueError(f"quota archive contract schema mismatch: {path}")
    if contract.get("resolved_config") != resolved:
        raise ValueError(f"quota archive config/contract mismatch: {path}")
    gammas = tuple(map(float, manifest.get("gammas", ())))
    if not gammas or gammas != tuple(map(float, resolved["data"]["gammas"])):
        raise ValueError(f"quota archive gamma declaration mismatch: {path}")
    if gammas != tuple(map(float, contract.get("gammas", ()))):
        raise ValueError(f"quota archive contract gamma mismatch: {path}")
    target = int(manifest.get("target_successes_per_gamma", -1))
    if target < 1 or target != int(
        contract.get("target_successes_per_gamma", -2)
    ):
        raise ValueError(f"quota archive target mismatch: {path}")
    if manifest.get("sampling_distribution", {}).get(
        "unconditioned_geometry"
    ) is not True:
        raise ValueError(f"quota archive is not geometry-unconditioned: {path}")
    if manifest.get("sampling_distribution", {}).get(
        "expert_success_used_for_scene_admission"
    ) is not False:
        raise ValueError(f"quota archive used expert scene admission: {path}")

    runs = list(manifest.get("runs", ()))
    attempts = list(manifest.get("attempts", ()))
    run_files = []
    counts = {gamma: 0 for gamma in gammas}
    for row in runs:
        gamma = float(row["gamma"])
        if gamma not in counts or row.get("accepted") is not True:
            raise ValueError(f"quota archive contains an invalid run: {path}")
        file_name = str(row.get("file", ""))
        if Path(file_name).name != file_name or not file_name:
            raise ValueError(f"quota archive run path is not a basename: {path}")
        if not (path / file_name).is_file():
            raise FileNotFoundError(path / file_name)
        run_files.append(file_name)
        counts[gamma] += 1
    if len(run_files) != len(set(run_files)):
        raise ValueError(f"quota archive has duplicate run filenames: {path}")
    if any(count != target for count in counts.values()):
        raise ValueError(
            f"quota archive does not contain exactly {target} runs/gamma: {path}"
        )
    accepted_attempt_files = sorted(
        str(row["file"]) for row in attempts if row.get("accepted") is True
    )
    if accepted_attempt_files != sorted(run_files):
        raise ValueError(f"quota archive runs/attempts disagree: {path}")
    if any(float(row["gamma"]) not in counts for row in attempts):
        raise ValueError(f"quota archive attempt gamma mismatch: {path}")

    scene_rows = list(manifest.get("scene_bank", {}).get("scenes", ()))
    scene_by_index = {}
    for row in scene_rows:
        index = int(row["scene_index"])
        if index in scene_by_index:
            raise ValueError(f"quota archive has duplicate scene index: {path}")
        scene_by_index[index] = row
    for row in attempts:
        index = int(row["scene_index"])
        if index not in scene_by_index:
            raise ValueError(f"quota archive attempt lacks scene row: {path}")
        if str(row["scene_hash"]) != str(scene_by_index[index]["scene_hash"]):
            raise ValueError(f"quota archive scene hash mismatch: {path}")

    return {
        "path": path,
        "manifest": manifest,
        "resolved": resolved,
        "contract": contract,
        "gammas": gammas,
        "target": target,
        "runs": runs,
        "attempts": attempts,
        "scenes": scene_by_index,
    }


def assemble_path_focused_quota_archives(
    source_dirs: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
) -> dict:
    """Atomically publish one union of complete disjoint-gamma archives."""
    sources = [_load_source(Path(path).resolve()) for path in source_dirs]
    if not sources:
        raise ValueError("at least one source quota archive is required")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    first = sources[0]
    target = first["target"]
    recipe = _canonical_recipe(first["resolved"])
    contract_reference = {
        key: value for key, value in first["contract"].items()
        if key not in {"gammas", "resolved_config", "max_scene_count"}
    }
    gamma_owner = {}
    for source_index, source in enumerate(sources):
        if source["target"] != target:
            raise ValueError("source quota targets differ")
        if _canonical_recipe(source["resolved"]) != recipe:
            raise ValueError("source resolved recipes differ beyond gamma/budget")
        source_contract = {
            key: value for key, value in source["contract"].items()
            if key not in {"gammas", "resolved_config", "max_scene_count"}
        }
        if source_contract != contract_reference:
            raise ValueError("source quota provenance differs")
        for gamma in source["gammas"]:
            if gamma in gamma_owner:
                raise ValueError(f"gamma {gamma:g} appears in multiple sources")
            gamma_owner[gamma] = source_index

    merged_scenes = {}
    for source in sources:
        for index, row in source["scenes"].items():
            if index in merged_scenes and merged_scenes[index] != row:
                raise ValueError(
                    f"scene index {index} differs across source archives"
                )
            merged_scenes[index] = row

    gamma_values = tuple(sorted(gamma_owner))
    resolved = copy.deepcopy(first["resolved"])
    resolved["data"] = dict(resolved["data"])
    resolved["data"]["gammas"] = list(gamma_values)
    resolved["data"]["episodes_per_gamma"] = target
    resolved["data"]["max_attempts_per_gamma"] = max(
        int(source["resolved"]["data"].get("max_attempts_per_gamma") or target)
        for source in sources
    )

    staging = output.parent / f".{output.name}.assemble-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"stale assembly staging directory: {staging}")
    staging.mkdir()
    try:
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n"
        )
        load_config(staging / "resolved_config.json")

        attempts = []
        runs = []
        file_declarations = []
        used_names = set()
        hardlinks = copies = 0
        for source_index, source in enumerate(sources):
            attempts.extend(copy.deepcopy(source["attempts"]))
            for row in source["runs"]:
                row = copy.deepcopy(row)
                name = str(row["file"])
                if name in used_names:
                    raise ValueError(f"duplicate output run filename: {name}")
                used_names.add(name)
                method, digest = _publish_file(
                    source["path"] / name, staging / name,
                )
                hardlinks += method == "hardlink"
                copies += method == "copy"
                file_declarations.append({
                    "file": name,
                    "sha256": digest,
                    "source_index": source_index,
                    "publication": method,
                })
                runs.append(row)

        attempts.sort(key=lambda row: (
            int(row["scene_index"]), float(row["gamma"]), int(row["seed"]),
        ))
        runs.sort(key=lambda row: (
            float(row["gamma"]), int(row["scene_index"]), int(row["seed"]),
        ))
        scenes = [merged_scenes[index] for index in sorted(merged_scenes)]
        metrics = aggregate_metrics(attempts, gamma_values)
        behavior_by_gamma = {}
        for source in sources:
            for row in source["manifest"].get("behavior_metrics", ()):
                gamma = float(row["gamma"])
                if gamma in behavior_by_gamma:
                    raise ValueError(f"duplicate behavior metric gamma {gamma:g}")
                behavior_by_gamma[gamma] = row

        source_provenance = []
        for source_index, source in enumerate(sources):
            source_provenance.append({
                "source_index": source_index,
                "path": str(source["path"]),
                "gammas": list(source["gammas"]),
                "manifest_sha256": _sha256(source["path"] / "manifest.json"),
                "resolved_config_sha256": _sha256(
                    source["path"] / "resolved_config.json"
                ),
                "quota_contract_sha256": _sha256(
                    source["path"] / "quota_contract.json"
                ),
                "attempt_count": len(source["attempts"]),
                "run_count": len(source["runs"]),
            })
        assembly = {
            "schema": ASSEMBLY_SCHEMA,
            "target_successes_per_gamma": target,
            "gammas": list(gamma_values),
            "sources": source_provenance,
            "trajectory_files": file_declarations,
            "publication": {"hardlinks": hardlinks, "copies": copies},
        }
        (staging / "assembly_contract.json").write_text(
            json.dumps(assembly, indent=2) + "\n"
        )
        manifest = {
            "kind": (
                "assembled Minhyuk lab path-focused randomized-cylinder "
                "SafeMPPI success-quota demonstrations"
            ),
            "schema_version": 5,
            "status": "COMPLETE_EXACT_SUCCESS_QUOTA",
            "config": "resolved_config.json",
            "assembly_contract": "assembly_contract.json",
            "rollout_dynamics": first["manifest"]["rollout_dynamics"],
            "acceptance": first["manifest"]["acceptance"],
            "target_successes_per_gamma": target,
            "evaluated_scene_count": len(scenes),
            "accepted_counts_by_gamma": {
                f"{gamma:.9g}": int(sum(
                    np.isclose(float(row["gamma"]), gamma) for row in runs
                ))
                for gamma in gamma_values
            },
            "gammas": list(gamma_values),
            "sampling_distribution": {
                **copy.deepcopy(first["manifest"]["sampling_distribution"]),
                "assembled_from_disjoint_gamma_archives": True,
            },
            "scene_bank": {
                "shared_across_gamma": False,
                "shared_deterministic_scene_prefix": True,
                "domain_seed": first["manifest"]["scene_bank"]["domain_seed"],
                "start_goal_path_summary": (
                    summarize_start_goal_path_diagnostics([
                        row["start_goal_path_diagnostics"] for row in scenes
                    ])
                    if scenes else None
                ),
                "scenes": scenes,
                "rejected_scenes": [],
            },
            "runs": runs,
            "attempts": attempts,
            "metrics": metrics,
            "behavior_metrics": [
                behavior_by_gamma[gamma]
                for gamma in gamma_values if gamma in behavior_by_gamma
            ],
        }
        (staging / "metrics.json").write_text(json.dumps({
            "expert_outcomes": metrics,
            "behavior": manifest["behavior_metrics"],
        }, indent=2) + "\n")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
