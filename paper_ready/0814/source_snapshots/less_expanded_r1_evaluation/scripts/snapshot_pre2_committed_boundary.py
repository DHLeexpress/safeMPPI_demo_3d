#!/usr/bin/env python3
"""Snapshot a committed PRE2 boundary while the next round is collecting."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(
            left, right, equal_nan=True,
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right)
        )
    if is_dataclass(left) and type(left) is type(right):
        return all(
            _same(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    return left == right


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def snapshot(
    source: Path,
    output: Path,
    round_index: int,
    semantic_manifest: Path,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    semantic_manifest = semantic_manifest.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    metadata = _json(source / "resume_state.json")
    if (
        metadata.get("status") != "COMMITTED_ROUND_RESUME"
        or int(metadata.get("completed_round", -1)) != round_index
    ):
        raise RuntimeError("source metadata is not the requested boundary")
    state = torch.load(
        source / "resume_state_latest.pt", map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(state, dict)
        or state.get("status") != "COMMITTED_ROUND_RESUME"
        or int(state.get("completed_round", -1)) != round_index
    ):
        raise RuntimeError("source resume payload is not the requested boundary")
    optimizer_step = int(state["optimizer_metadata"]["_safe_mppi_schedule_step"])
    if int(metadata.get("optimizer_step", -1)) != optimizer_step:
        raise RuntimeError("optimizer step differs between resume artifacts")
    rows = _metrics(source / "metrics.jsonl")
    if not (
        len(rows) == round_index
        and rows == state["round_rows"]
        and int(rows[-1]["round"]) == round_index
    ):
        raise RuntimeError("metrics and resume round histories differ")
    checkpoint = torch.load(
        source / f"checkpoint_{round_index:03d}.pt", map_location="cpu",
        weights_only=False,
    )
    if not _same(state["model"], checkpoint.get("model")):
        raise RuntimeError("checkpoint model differs from resume model")

    mutable = [
        "resume_state.json", "resume_state_latest.pt", "metrics.jsonl",
        "first_action_stats.json", "fa_alloc_log.json",
        "task_config_resolved.json",
    ]
    mutable += [
        name for name in (
            "BRANCH_PROVENANCE.json", "BRANCH_SCREENING_MANIFEST.json",
        )
        if (source / name).is_file()
    ]
    immutable = [
        f"checkpoint_{index:03d}.pt" for index in range(round_index + 1)
    ] + [
        f"query_archive_round_{index:03d}.pt"
        for index in range(1, round_index + 1)
    ]
    if (source / "events_round_001.pt").is_file():
        immutable += [
            f"events_round_{index:03d}.pt"
            for index in range(1, round_index + 1)
        ]
    missing = [
        name for name in mutable + immutable
        if not (source / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing boundary artifacts: " + ", ".join(missing))
    before = {name: _snapshot(source / name) for name in mutable + immutable}

    base_manifest = _json(semantic_manifest)
    manifest = dict(base_manifest)
    manifest["status"] = "COMMITTED_ROUND_CALIBRATION_MANIFEST"
    manifest["config"] = state["config"]
    manifest["rounds"] = rows
    manifest["resume"] = {
        "latest_committed_round": round_index,
        "state_file": "resume_state_latest.pt",
        "boundary": "after_optimizer_update_and_committed_archive",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.snapshot-", dir=output.parent,
    ))
    try:
        for name in mutable:
            shutil.copy2(source / name, temporary / name)
        for name in immutable:
            os.link(source / name, temporary / name)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        shutil.copy2(semantic_manifest, temporary / "MANIFEST_SEMANTIC_SOURCE.json")

        cloned = torch.load(
            temporary / "resume_state_latest.pt", map_location="cpu",
            weights_only=False,
        )
        if not _same(state, cloned):
            raise RuntimeError("cloned resume payload is not exact")
        for name, record in before.items():
            if _sha256(temporary / name) != record["sha256"]:
                raise RuntimeError(f"cloned artifact hash mismatch: {name}")
        for name in mutable:
            if os.path.samestat((source / name).stat(), (temporary / name).stat()):
                raise RuntimeError(f"mutable copy aliases source: {name}")
        for name in immutable:
            if not os.path.samestat(
                (source / name).stat(), (temporary / name).stat(),
            ):
                raise RuntimeError(f"immutable hard link differs: {name}")
        for name, record in before.items():
            if _snapshot(source / name) != record:
                raise RuntimeError(f"source changed while snapshotting: {name}")
        current_metadata = _json(source / "resume_state.json")
        if int(current_metadata.get("completed_round", -1)) != round_index:
            raise RuntimeError("source crossed a boundary while snapshotting")

        provenance = {
            "status": f"EXACT_R{round_index}_ADAPTIVE_SNAPSHOT_READY",
            "source": str(source),
            "snapshot": str(output),
            "completed_round": round_index,
            "optimizer_step": optimizer_step,
            "files": {
                f"checkpoint_{round_index:03d}.pt": before[
                    f"checkpoint_{round_index:03d}.pt"
                ]["sha256"],
                "resume_state_latest.pt": before[
                    "resume_state_latest.pt"
                ]["sha256"],
                "resume_state.json": before["resume_state.json"]["sha256"],
                "manifest.json": _sha256(temporary / "manifest.json"),
            },
            "semantic_manifest_source": str(semantic_manifest),
            "semantic_manifest_source_sha256": _sha256(semantic_manifest),
            "storage": (
                "immutable per-round artifacts hard-linked; mutable resume/log "
                "state independently copied"
            ),
        }
        (temporary / "SNAPSHOT_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"output appeared during snapshot: {output}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = snapshot(
        args.source, args.output, args.round, args.semantic_manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
