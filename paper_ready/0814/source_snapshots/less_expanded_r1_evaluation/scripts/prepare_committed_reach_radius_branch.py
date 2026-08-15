#!/usr/bin/env python3
"""Clone a committed expansion boundary for a reach-radius-only continuation.

The source is never modified.  Every resume-critical artifact is copied to a
temporary sibling directory, verified byte-for-byte (and as a distinct inode),
then atomically published as the requested output directory.  The resolved task
configuration may differ only at ``taskspace.reach_radius``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch


RESUME_REQUIRED_KEYS = frozenset({
    "version",
    "status",
    "completed_round",
    "config",
    "model",
    "optimizer",
    "optimizer_metadata",
    "numpy_rng_state",
    "torch_rng_state",
    "torch_cpu_rng_state",
    "torch_device_rng_state",
    "archive",
    "frozen_gp_rows",
    "frozen_gp_hash",
    "round1_gp_candidates",
    "gp_evidence",
    "cumulative_anchors",
    "cumulative_adaptive",
    "round_rows",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_model(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _json_differences(
    left: Any, right: Any, path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left:
                differences.append((path + (key,), None, right[key]))
            elif key not in right:
                differences.append((path + (key,), left[key], None))
            else:
                differences.extend(
                    _json_differences(left[key], right[key], path + (key,))
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            return [(path + ("<length>",), len(left), len(right))]
        for index, (old_value, new_value) in enumerate(zip(left, right)):
            differences.extend(_json_differences(
                old_value, new_value, path + (str(index),),
            ))
        return differences
    return [] if left == right else [(path, left, right)]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def _validate_task_delta(
    source_config: Path,
    target_config: Path,
    source_radius: float,
    target_radius: float,
) -> dict[str, Any]:
    source = _load_json(source_config)
    target = _load_json(target_config)
    try:
        observed_source = float(source["taskspace"]["reach_radius"])
        observed_target = float(target["taskspace"]["reach_radius"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("both task configs must define taskspace.reach_radius") from error
    if not math.isclose(observed_source, source_radius, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"source reach_radius is {observed_source}, expected {source_radius}"
        )
    if not math.isclose(observed_target, target_radius, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"target reach_radius is {observed_target}, expected {target_radius}"
        )
    differences = _json_differences(source, target)
    expected = [("taskspace", "reach_radius")]
    observed_paths = [difference[0] for difference in differences]
    if observed_paths != expected:
        rendered = [".".join(path) for path in observed_paths]
        raise ValueError(
            "target task config must differ only at taskspace.reach_radius; "
            f"observed differences: {rendered}"
        )
    return {
        "path": "taskspace.reach_radius",
        "source_value": observed_source,
        "target_value": observed_target,
    }


def _validate_source(source: Path, completed_round: int) -> list[str]:
    if not source.is_dir():
        raise FileNotFoundError(f"source expansion directory is missing: {source}")
    in_progress = source / "RESUME_IN_PROGRESS.json"
    if in_progress.exists():
        raise RuntimeError(
            f"source has an in-progress continuation marker: {in_progress}"
        )

    resume_path = source / "resume_state_latest.pt"
    metadata_path = source / "resume_state.json"
    manifest_path = source / "manifest.json"
    for path in (resume_path, metadata_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = _load_json(metadata_path)
    if metadata.get("status") != "COMMITTED_ROUND_RESUME":
        raise ValueError("resume_state.json is not a committed-round snapshot")
    if int(metadata.get("version", -1)) != 1:
        raise ValueError("unsupported resume_state.json version")
    if int(metadata.get("completed_round", -1)) != completed_round:
        raise ValueError("resume_state.json does not end at the requested boundary")

    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if not isinstance(resume, dict):
        raise ValueError("resume_state_latest.pt must contain a mapping")
    missing_resume = sorted(RESUME_REQUIRED_KEYS - resume.keys())
    if missing_resume:
        raise ValueError(
            "resume state omits optimizer/RNG/GP/replay fields: "
            + ", ".join(missing_resume)
        )
    if resume["status"] != "COMMITTED_ROUND_RESUME" or int(resume["version"]) != 1:
        raise ValueError("resume_state_latest.pt is not a supported committed snapshot")
    if int(resume["completed_round"]) != completed_round:
        raise ValueError("resume_state_latest.pt does not end at the requested boundary")
    if len(resume["round_rows"]) != completed_round:
        raise ValueError("resume round history does not match the requested boundary")

    checkpoint_path = source / f"checkpoint_{completed_round:03d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("committed checkpoint does not contain a model state")
    if not _same_model(resume["model"], checkpoint["model"]):
        raise ValueError("resume model does not equal the committed checkpoint model")

    manifest = _load_json(manifest_path)
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != completed_round:
        raise ValueError("manifest does not contain exactly the committed rounds")
    if completed_round and int(rounds[-1].get("round", -1)) != completed_round:
        raise ValueError("manifest does not end at the requested boundary")

    artifact_names = [
        "manifest.json",
        "metrics.jsonl",
        "resume_state.json",
        "resume_state_latest.pt",
        "query_archive.pt",
        "gp_evidence.pt",
        "first_action_stats.json",
        "fa_alloc_log.json",
    ]
    artifact_names.extend(
        f"checkpoint_{round_index:03d}.pt"
        for round_index in range(completed_round + 1)
    )
    artifact_names.extend(
        f"query_archive_round_{round_index:03d}.pt"
        for round_index in range(1, completed_round + 1)
    )
    if manifest.get("event_log") in {"full", "committed_success"}:
        artifact_names.extend(
            f"events_round_{round_index:03d}.pt"
            for round_index in range(1, completed_round + 1)
        )
        if (source / "events.pt").is_file():
            artifact_names.append("events.pt")
    missing = [name for name in artifact_names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "source is missing resume-critical artifacts: " + ", ".join(missing)
        )
    return artifact_names


def prepare_branch(
    *,
    source: Path,
    output: Path,
    task_config: Path,
    completed_round: int = 1,
    source_radius: float = 0.2,
    target_radius: float = 0.3,
    validate_only: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    task_config = task_config.resolve()
    if source == output:
        raise ValueError("source and output must be distinct directories")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if completed_round < 1:
        raise ValueError("completed_round must be positive")

    artifact_names = _validate_source(source, completed_round)
    semantic_delta = _validate_task_delta(
        source / "task_config_resolved.json",
        task_config,
        source_radius,
        target_radius,
    )
    source_task_path = source / "task_config_resolved.json"
    source_task_snapshot = {
        "sha256": _sha256(source_task_path),
        "inode": int(source_task_path.stat().st_ino),
        "size": int(source_task_path.stat().st_size),
        "mtime_ns": int(source_task_path.stat().st_mtime_ns),
    }
    source_snapshot = {
        name: {
            "sha256": _sha256(source / name),
            "inode": int((source / name).stat().st_ino),
            "size": int((source / name).stat().st_size),
            "mtime_ns": int((source / name).stat().st_mtime_ns),
        }
        for name in artifact_names
    }
    report = {
        "status": (
            "VALIDATED_COMMITTED_REACH_RADIUS_BRANCH"
            if validate_only else "COMMITTED_REACH_RADIUS_BRANCH_READY"
        ),
        "source": str(source),
        "output": str(output),
        "completed_round": completed_round,
        "target_task_config": str(task_config),
        "semantic_delta": semantic_delta,
        "copied_artifact_sha256": {
            name: source_snapshot[name]["sha256"] for name in artifact_names
        },
        "state_contract": (
            "checkpoint/model, Adam optimizer and schedule, NumPy/torch/device "
            "RNG, GP evidence/reference, cumulative replay, query archives, and "
            "committed event history are byte-identical copies; only the resolved "
            "task reach radius changes for future terminal labeling"
        ),
        "launch_contract": {
            "resume_from_must_equal_output": True,
            "resume_in_place": str(output),
            "lab_task_config": str(task_config),
            "minimum_target_round": completed_round + 1,
        },
        "source_mutated": False,
    }
    if validate_only:
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.prepare-", dir=output.parent,
    ))
    try:
        for name in artifact_names:
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        shutil.copy2(task_config, temporary / "task_config_resolved.json")

        for name in artifact_names:
            source_stat = (source / name).stat()
            target_stat = (temporary / name).stat()
            before = source_snapshot[name]
            if _sha256(temporary / name) != before["sha256"]:
                raise RuntimeError(f"copied artifact hash mismatch: {name}")
            if (source_stat.st_dev, source_stat.st_ino) == (
                target_stat.st_dev, target_stat.st_ino,
            ):
                raise RuntimeError(f"copied artifact aliases the source inode: {name}")
            if (
                int(source_stat.st_ino) != before["inode"]
                or int(source_stat.st_size) != before["size"]
                or int(source_stat.st_mtime_ns) != before["mtime_ns"]
                or _sha256(source / name) != before["sha256"]
            ):
                raise RuntimeError(f"source changed while preparing branch: {name}")

        target_config_hash = _sha256(task_config)
        if _sha256(temporary / "task_config_resolved.json") != target_config_hash:
            raise RuntimeError("target task config copy mismatch")
        source_task_before = source_task_snapshot
        source_task_after = source / "task_config_resolved.json"
        if (
            _sha256(source_task_after) != source_task_before["sha256"]
            or int(source_task_after.stat().st_ino) != source_task_before["inode"]
            or int(source_task_after.stat().st_size) != source_task_before["size"]
            or int(source_task_after.stat().st_mtime_ns) != source_task_before["mtime_ns"]
        ):
            raise RuntimeError("source task config changed while preparing branch")
        report["source_task_config_sha256"] = source_task_before["sha256"]
        report["target_task_config_sha256"] = target_config_hash
        report["copied_files_use_distinct_inodes"] = True
        (temporary / "BRANCH_PROVENANCE.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if output.exists():
            raise FileExistsError(
                f"output appeared while preparing branch; refusing to overwrite: {output}"
            )
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--source-reach-radius", type=float, default=0.2)
    parser.add_argument("--target-reach-radius", type=float, default=0.3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = prepare_branch(
        source=args.source,
        output=args.output,
        task_config=args.task_config,
        completed_round=args.round,
        source_radius=args.source_reach_radius,
        target_radius=args.target_reach_radius,
        validate_only=args.validate_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
