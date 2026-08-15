#!/usr/bin/env python3
"""Fork an exact committed w50 state for the approved task-space probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from safe_mppi.config import load_config


OLD_BOUNDS = np.asarray([
    [-2.5, 1.3],
    [-1.7, 1.8],
    [0.1, 1.7],
], np.float64)
NEW_BOUNDS = np.asarray([
    [-2.5, 1.3],
    [-2.1, 1.8],
    [0.1, 1.7],
], np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_model(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def _validate_recipe(source: Path, round_i: int) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    resume = torch.load(
        source / "resume_state_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = torch.load(
        source / f"checkpoint_{round_i:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(resume["completed_round"]) != round_i:
        raise ValueError("source latest resume is not the requested boundary")
    if len(manifest["rounds"]) != round_i:
        raise ValueError("source manifest does not end at the requested boundary")
    if not _same_model(resume["model"], checkpoint["model"]):
        raise ValueError("resume model does not equal the committed checkpoint")

    config = resume["config"]
    expected = {
        "K": 16,
        "B": 8,
        "retry_B": 8,
        "retry_verify_all_fast_path": False,
        "successful_trajectories_per_gamma": 12,
        "sample_update_mode": (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3),
        "beta": 0.1,
        "inner_steps": 2500,
        "seed": 82410,
    }
    for key, wanted in expected.items():
        observed = config[key]
        if key == "sample_update_mode":
            observed = tuple(observed)
        if observed != wanted:
            raise ValueError(f"source is not the approved w50 recipe: {key}")

    execution = manifest["lab_execution_cost"]
    goal_box = execution["execution_goal_box_exponential"]
    cylinder = execution["execution_axis_cylinder_quadratic"]
    braking = execution["execution_goal_braking"]
    verifier = manifest["lab_verifier"]
    contract = {
        "goal_box_weight": float(goal_box["weight"]),
        "goal_box_half_extent_m": float(goal_box["half_extent_m"]),
        "goal_box_temperature_m": float(goal_box["temperature_m"]),
        "axis_weight": float(cylinder["weight"]),
        "axis_radius_m": float(cylinder["radius_m"]),
        "axis_finite_segment": bool(cylinder["finite_segment"]),
        "control_weight": float(execution["execution_control"]["effective_weight"]),
        "terminal_weight": float(
            execution["execution_terminal_goal"]["effective_weight"]
        ),
        "braking_weight": float(braking["weight"]),
        "full_h_taskspace": bool(
            verifier["unexecuted_tail_taskspace_gate"]
        ),
        "stopping_margin_m": verifier[
            "taskspace_stopping_backup"
        ]["face_margin_m"],
    }
    required = {
        "goal_box_weight": 50.0,
        "goal_box_half_extent_m": 0.2,
        "goal_box_temperature_m": 1.0,
        "axis_weight": 5.0,
        "axis_radius_m": 1.1,
        "axis_finite_segment": True,
        "control_weight": 1.0,
        "terminal_weight": 80.0,
        "braking_weight": 0.0,
        "full_h_taskspace": True,
        "stopping_margin_m": None,
    }
    if contract != required:
        raise ValueError(f"source execution/verifier contract changed: {contract}")
    return {
        "manifest": manifest,
        "resume": resume,
        "checkpoint": checkpoint,
        "contract": contract,
    }


def _validate_taskspace(source: Path, target_config: Path) -> dict:
    old = load_config(source / "task_config_resolved.json")
    new = load_config(target_config)
    if not np.allclose(old.taskspace.bounds, OLD_BOUNDS, rtol=0.0, atol=1e-9):
        raise ValueError("source task-space bounds are not the legacy contract")
    if not np.allclose(new.taskspace.bounds, NEW_BOUNDS, rtol=0.0, atol=1e-9):
        raise ValueError("target task-space bounds are not the approved y_min -0.4 m contract")
    for key in ("start", "goal", "reach_radius", "max_steps"):
        if getattr(old.taskspace, key) != getattr(new.taskspace, key):
            raise ValueError(f"target task-space unexpectedly changes {key}")
    if old.obstacles != new.obstacles:
        raise ValueError("target config changes obstacle geometry")
    if old.safemppi != new.safemppi:
        raise ValueError("target config changes SafeMPPI dynamics or costs")
    if old.data != new.data:
        raise ValueError("target config changes gamma/data settings")
    return {
        "old_bounds": OLD_BOUNDS.tolist(),
        "new_bounds": NEW_BOUNDS.tolist(),
        "goal_clearance_old_m": {"x_max": 0.6, "y_min": 0.2},
        "goal_clearance_new_m": {"x_max": 0.6, "y_min": 0.6},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    task_config = args.task_config.resolve()
    recipe = _validate_recipe(source, args.round)
    geometry = _validate_taskspace(source, task_config)
    if args.validate_only:
        print(json.dumps({
            "status": "VALIDATED_ONLY",
            "source": str(source),
            "output": str(output),
            "task_config": str(task_config),
            "completed_round": args.round,
            "contract": recipe["contract"],
            **geometry,
        }, indent=2))
        return

    copied_names = [
        "manifest.json",
        "metrics.jsonl",
        "resume_state.json",
        "resume_state_latest.pt",
        "first_action_stats.json",
        "fa_alloc_log.json",
        "query_archive.pt",
        "gp_evidence.pt",
    ]
    immutable_names = [
        f"checkpoint_{index:03d}.pt" for index in range(args.round + 1)
    ]
    immutable_names.extend(
        f"query_archive_round_{index:03d}.pt"
        for index in range(1, args.round + 1)
    )
    immutable_names.extend(
        f"events_round_{index:03d}.pt"
        for index in range(1, args.round + 1)
    )
    for name in copied_names + immutable_names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != output.stat().st_dev:
        raise ValueError(
            "source and output must share a filesystem for immutable hard links"
        )

    source_immutable_metadata = {
        name: {
            "inode": int((source / name).stat().st_ino),
            "size": int((source / name).stat().st_size),
            "mtime_ns": int((source / name).stat().st_mtime_ns),
        }
        for name in immutable_names
    }
    for name in immutable_names:
        os.link(source / name, output / name)
    for name in copied_names:
        shutil.copy2(source / name, output / name)
    shutil.copy2(task_config, output / "task_config_resolved.json")

    for name in immutable_names:
        source_stat = (source / name).stat()
        output_stat = (output / name).stat()
        before = source_immutable_metadata[name]
        if (source_stat.st_dev, source_stat.st_ino) != (
            output_stat.st_dev,
            output_stat.st_ino,
        ):
            raise RuntimeError(f"immutable artifact was not hard-linked: {name}")
        if (
            int(source_stat.st_ino) != before["inode"]
            or int(source_stat.st_size) != before["size"]
            or int(source_stat.st_mtime_ns) != before["mtime_ns"]
        ):
            raise RuntimeError(f"source immutable artifact changed: {name}")
    for name in copied_names:
        source_stat = (source / name).stat()
        output_stat = (output / name).stat()
        if (source_stat.st_dev, source_stat.st_ino) == (
            output_stat.st_dev,
            output_stat.st_ino,
        ):
            raise RuntimeError(f"mutable artifact was hard-linked: {name}")
        if _sha256(source / name) != _sha256(output / name):
            raise RuntimeError(f"mutable artifact copy mismatch: {name}")

    provenance = {
        "status": "EXACT_COMMITTED_TASKSPACE_BRANCH_READY",
        "source": str(source),
        "source_round": args.round,
        "source_checkpoint_sha256": _sha256(
            source / f"checkpoint_{args.round:03d}.pt"
        ),
        "source_resume_sha256": _sha256(source / "resume_state_latest.pt"),
        "source_task_config_sha256": _sha256(
            source / "task_config_resolved.json"
        ),
        "target_task_config": str(task_config),
        "target_task_config_sha256": _sha256(task_config),
        "state_contract": (
            "exact model, Adam, optimizer schedule, NumPy RNG, torch RNG, "
            "GP evidence, and cumulative replay preserved from committed r5"
        ),
        "storage_contract": {
            "hardlinked_immutable_artifacts": immutable_names,
            "copied_mutable_artifacts": copied_names,
            "omitted_redundant_artifacts": ["events.pt"],
            "hardlink_same_device_inode_verified": True,
            "source_inode_size_and_mtime_unchanged_verified": True,
            "mutable_copy_hash_and_distinct_inode_verified": True,
            "downstream_write_contract": (
                "committed round files are append-only and must never be "
                "modified in place; continuation writes only later rounds"
            ),
        },
        "prepared_geometry_intervention": (
            "task-space y_min -1.7->-2.1; x_max remains 1.3"
        ),
        "execution_delta_contract": (
            "the launch command must explicitly record any terminal "
            "goal-side-wall or braking weight added to this base fork"
        ),
        "source_immutable": True,
        "execution_and_verifier_contract": recipe["contract"],
        **geometry,
    }
    (output / "BRANCH_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
