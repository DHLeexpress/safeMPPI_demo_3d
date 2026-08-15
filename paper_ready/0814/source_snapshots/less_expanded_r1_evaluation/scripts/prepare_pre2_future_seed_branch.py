#!/usr/bin/env python3
"""Clone an exact PRE2 boundary for a future-only sampling stream."""
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
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _validate_source(source: Path) -> tuple[
    int, dict[str, Any], list[str], list[str], dict[str, dict[str, Any]]
]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if (source / "RESUME_IN_PROGRESS.json").exists():
        raise RuntimeError("refusing an in-progress resume source")
    metadata = _json(source / "resume_state.json")
    if metadata.get("status") != "COMMITTED_ROUND_RESUME":
        raise ValueError("source is not a committed resume boundary")
    if int(metadata.get("version", -1)) != 1:
        raise ValueError("unsupported resume metadata version")
    completed = int(metadata["completed_round"])
    if int(metadata.get("next_round", -1)) != completed + 1:
        raise ValueError("resume next_round is inconsistent")
    state = torch.load(
        source / "resume_state_latest.pt", map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict) or state.get("status") != "COMMITTED_ROUND_RESUME":
        raise ValueError("resume payload is not committed")
    if int(state.get("version", -1)) != 1 or int(
        state.get("completed_round", -1)
    ) != completed:
        raise ValueError("resume payload boundary is inconsistent")
    schedule_step = int(state["optimizer_metadata"][
        "_safe_mppi_schedule_step"
    ])
    if int(metadata.get("optimizer_step", -1)) != schedule_step:
        raise ValueError("optimizer step metadata is inconsistent")

    manifest = _json(source / "manifest.json")
    metrics = _read_metrics(source / "metrics.jsonl")
    if not (
        state["round_rows"] == manifest.get("rounds") == metrics
        and len(metrics) == completed
        and int(metrics[-1]["round"]) == completed
    ):
        raise ValueError("resume, manifest, and metrics round histories differ")
    checkpoint = torch.load(
        source / f"checkpoint_{completed:03d}.pt", map_location="cpu",
        weights_only=False,
    )
    if not _same(state["model"], checkpoint.get("model")):
        raise ValueError("resume model differs from committed checkpoint")

    per_round_archive = []
    for index in range(1, completed + 1):
        per_round_archive.extend(torch.load(
            source / f"query_archive_round_{index:03d}.pt",
            map_location="cpu", weights_only=False,
        ))
    if not _same(state["archive"], per_round_archive):
        raise ValueError("embedded replay differs from per-round archives")
    if not _same(state["gp_evidence"], per_round_archive):
        raise ValueError("embedded GP evidence differs from committed archives")

    mutable = [
        "resume_state.json", "resume_state_latest.pt", "metrics.jsonl",
        "manifest.json", "first_action_stats.json", "fa_alloc_log.json",
        "task_config_resolved.json",
    ]
    immutable = [
        f"checkpoint_{index:03d}.pt" for index in range(completed + 1)
    ] + [
        f"query_archive_round_{index:03d}.pt"
        for index in range(1, completed + 1)
    ]
    event_mode = manifest.get("event_log", manifest.get("config", {}).get(
        "event_log",
    ))
    if event_mode in {"full", "committed_success"} or (
        source / "events_round_001.pt"
    ).is_file():
        immutable += [
            f"events_round_{index:03d}.pt"
            for index in range(1, completed + 1)
        ]
    mutable += [
        name for name in ("query_archive.pt", "gp_evidence.pt", "events.pt")
        if (source / name).is_file()
    ]
    mutable += [
        name for name in (
            "BRANCH_PROVENANCE.json", "BRANCH_SCREENING_MANIFEST.json",
            "SNAPSHOT_PROVENANCE.json",
        )
        if (source / name).is_file()
    ]
    names = mutable + immutable
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("missing exact-state artifacts: " + ", ".join(missing))
    snapshots = {name: _snapshot(source / name) for name in names}
    return completed, state, mutable, immutable, snapshots


def prepare(source: Path, output: Path, future_sampling_seed: int) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if future_sampling_seed < 0:
        raise ValueError("future sampling seed must be nonnegative")
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    completed, state, mutable, immutable, snapshots = _validate_source(source)
    historical_seed = int(state["config"]["seed"])
    if future_sampling_seed == historical_seed:
        raise ValueError("future sampling seed must differ from historical seed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.prepare-", dir=output.parent,
    ))
    try:
        for name in mutable:
            shutil.copy2(source / name, temporary / name)
        for name in immutable:
            os.link(source / name, temporary / name)
        copied_state = torch.load(
            temporary / "resume_state_latest.pt", map_location="cpu",
            weights_only=False,
        )
        if not _same(state, copied_state):
            raise RuntimeError("cloned resume payload is not exact")
        for name in mutable + immutable:
            if _sha256(temporary / name) != snapshots[name]["sha256"]:
                raise RuntimeError(f"cloned artifact hash mismatch: {name}")
        for name in mutable:
            if os.path.samestat(
                (source / name).stat(), (temporary / name).stat(),
            ):
                raise RuntimeError(f"mutable copy aliases source inode: {name}")
        for name in immutable:
            if not os.path.samestat(
                (source / name).stat(), (temporary / name).stat(),
            ):
                raise RuntimeError(f"committed hard link inode differs: {name}")
        # Detect a source mutation or boundary transition during cloning.
        if (source / "RESUME_IN_PROGRESS.json").exists():
            raise RuntimeError("source entered resume while cloning")
        for name, before in snapshots.items():
            if _snapshot(source / name) != before:
                raise RuntimeError(f"source changed while cloning: {name}")

        provenance = {
            "status": "FUTURE_SAMPLING_STREAM_BRANCH_READY",
            "source": str(source),
            "output": str(output),
            "completed_round": completed,
            "historical_seed": historical_seed,
            "future_sampling_seed": future_sampling_seed,
            "scientific_delta": (
                "future counter-based scene reset, candidate gathering, and "
                "random-success selection stream only"
            ),
            "preserved_exact_state": "entire resume_state_latest.pt",
            "source_checkpoint_sha256": snapshots[
                f"checkpoint_{completed:03d}.pt"
            ]["sha256"],
            "source_resume_sha256": snapshots[
                "resume_state_latest.pt"
            ]["sha256"],
            "artifact_storage": {
                "mutable_state": "independent copies",
                "committed_round_artifacts": "hard-linked immutable-by-contract",
            },
            "launch_contract": {
                "resume_from_must_equal_output": True,
                "seed": historical_seed,
                "future_sampling_seed": future_sampling_seed,
                "minimum_target_round": completed + 1,
            },
            "selection_scope": (
                "calibration-bank resampling; final holdout is not used to "
                "choose this branch"
            ),
        }
        (temporary / "FUTURE_SAMPLING_STREAM.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"output appeared during preparation: {output}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-sampling-seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.source, args.output, args.future_sampling_seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
