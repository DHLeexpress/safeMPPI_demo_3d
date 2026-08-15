#!/usr/bin/env python3
"""Prepare and launch the paired r5-conditioned obstacle-tail sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import torch


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / (
    "results/stage1_single_ball_t128/0811_pre2_r1_inward_oob_probe/"
    "resume_backups/reach03_e15_r5_exact"
)
STAGE = ROOT / (
    "results/stage1_single_ball_t128/0811_pre2_e15_obstacle_tail_sweep"
)
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
TASK_CONFIG = ROOT / (
    "configs/lab_ball_stage1_goalspace_yminus04_z01_17_r15in_reach03_v1.json"
)
SOURCE_RESUME_SHA256 = (
    "1ca1462c877a419d49bc3dec1fa75c12e50f0911d320f1e9d4f1fc7933dff82d"
)
SOURCE_CHECKPOINT_SHA256 = (
    "3291d6a90f0f94ccbcd0b8a69314e66337156243c2a94e888b5a9692a6a8f805"
)
ARMS = (
    {
        "name": "r5_tail_w40p7742_t020", "gpu": 1,
        "weight": 40.7742274269, "temperature": 0.20,
    },
    {
        "name": "r5_tail_w74p2955_t025", "gpu": 3,
        "weight": 74.2954863659, "temperature": 0.25,
    },
    {
        "name": "r5_tail_w110p8358_t030", "gpu": 1,
        "weight": 110.835841484, "temperature": 0.30,
    },
)
MUTABLE = (
    "manifest.json", "metrics.jsonl", "resume_state.json",
    "resume_state_latest.pt", "query_archive.pt", "gp_evidence.pt",
    "first_action_stats.json", "fa_alloc_log.json",
    "task_config_resolved.json",
)
IMMUTABLE = tuple(
    [f"checkpoint_{index:03d}.pt" for index in range(6)]
    + [f"query_archive_round_{index:03d}.pt" for index in range(1, 6)]
    + [f"events_round_{index:03d}.pt" for index in range(1, 6)]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _validate_source() -> None:
    for name in MUTABLE + IMMUTABLE:
        if not (SOURCE / name).is_file():
            raise FileNotFoundError(SOURCE / name)
    metadata = json.loads((SOURCE / "resume_state.json").read_text())
    if metadata.get("status") != "COMMITTED_ROUND_RESUME":
        raise ValueError("source is not a committed resume boundary")
    if int(metadata.get("completed_round", -1)) != 5:
        raise ValueError("source must be the exact committed r5 boundary")
    if int(metadata.get("optimizer_step", -1)) != 12500:
        raise ValueError("source optimizer schedule is not at step 12500")
    if _sha256(SOURCE / "resume_state_latest.pt") != SOURCE_RESUME_SHA256:
        raise ValueError("source r5 resume hash changed")
    if _sha256(SOURCE / "checkpoint_005.pt") != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("source r5 checkpoint hash changed")
    resume = torch.load(
        SOURCE / "resume_state_latest.pt", map_location="cpu",
        weights_only=False,
    )
    if int(resume.get("completed_round", -1)) != 5:
        raise ValueError("serialized resume state does not end at r5")
    config = resume["config"]
    expected = {
        "K": 16,
        "B": 8,
        "retry_B": 8,
        "retry_verify_all_fast_path": False,
        "successful_trajectories_per_gamma": 12,
        "beta": 0.1,
        "seed": 82410,
        "execution_rule": "exponential_cost",
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(f"source recipe mismatch for {key}: {config.get(key)!r}")


def _prepare_arm(arm: dict) -> Path:
    output = STAGE / "arms" / arm["name"]
    if output.exists():
        provenance = output / "BRANCH_PROVENANCE.json"
        if provenance.is_file():
            payload = json.loads(provenance.read_text())
            if payload.get("source_resume_sha256") == SOURCE_RESUME_SHA256:
                return output
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{arm['name']}.prepare-", dir=output.parent,
    ))
    try:
        for name in IMMUTABLE:
            os.link(SOURCE / name, temporary / name)
        for name in MUTABLE:
            shutil.copy2(SOURCE / name, temporary / name)
        for name in IMMUTABLE:
            if (SOURCE / name).stat().st_ino != (temporary / name).stat().st_ino:
                raise RuntimeError(f"immutable artifact was not hard-linked: {name}")
        for name in MUTABLE:
            if (SOURCE / name).stat().st_ino == (temporary / name).stat().st_ino:
                raise RuntimeError(f"mutable artifact aliases source: {name}")
            if _sha256(SOURCE / name) != _sha256(temporary / name):
                raise RuntimeError(f"mutable artifact hash mismatch: {name}")
        provenance = {
            "status": "EXACT_R5_EXECUTION_COST_BRANCH_READY",
            "source": str(SOURCE),
            "source_round": 5,
            "source_optimizer_step": 12500,
            "source_resume_sha256": SOURCE_RESUME_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "execution_cost_delta": {
                "legacy": {"weight": 15.0, "temperature_m": 0.15},
                "branch": {
                    "weight": arm["weight"],
                    "temperature_m": arm["temperature"],
                },
                "target_m": 0.6,
                "contact_penalty": 15.0 * math.exp(4.0),
                "contract": (
                    "contact penalty is held equal to legacy E15; only the "
                    "long-range exponential tail changes for rounds 6-7"
                ),
            },
            "paired_state_contract": (
                "r5 model, Adam/schedule, NumPy/torch/device RNG, GP, replay, "
                "query archives and committed events are identical"
            ),
            "gpu": arm["gpu"],
            "target_round": 7,
        }
        (temporary / "BRANCH_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _active(output: Path) -> bool:
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    return any(str(output) in line for line in commands.splitlines())


def _launch(arm: dict, output: Path, dry_run: bool) -> dict:
    launch_marker = output / "LAUNCHED.json"
    if launch_marker.is_file() or _active(output):
        return {
            "name": arm["name"], "output": str(output),
            "status": "ALREADY_LAUNCHED_OR_ACTIVE",
        }
    environment = os.environ.copy()
    environment.update({
        "PRE": str(PRETRAIN),
        "OUT": str(output),
        "HELIOS_GPU": str(arm["gpu"]),
        "ROUNDS": "7",
        "LAB_TASK_CONFIG": str(TASK_CONFIG),
        "SEED": "82410",
        "RETRY_RESAMPLE_BATCH_CAP": "192",
        "CONTROL_WEIGHT": "1",
        "TERMINAL_WEIGHT": "80",
        "BRAKING_WEIGHT": "0",
        "FINITE_SEGMENT": "1",
        "TASKSPACE_WEIGHT": "0",
        "TASKSPACE_TARGET": "0.15",
        "GOAL_SIDE_WALL_WEIGHT": "0",
        "GOAL_SIDE_WALL_TARGET": "0.6",
        "AXIS_WEIGHT": "5",
        "AXIS_RADIUS": "1.1",
        "EXECUTION_RULE": "exponential_cost",
        "CLEARANCE_EXP_WEIGHT": f"{arm['weight']:.12g}",
        "CLEARANCE_EXP_TEMPERATURE": f"{arm['temperature']:.12g}",
        "CLEARANCE_EXP_TARGET": "0.6",
        "SUCCESS_QUOTA": "12",
        "SAMPLE_MODES": "0,0,0,1,1,1,2,2,2,3,3,3",
        "FAITHFUL_RETRY": "1",
        "GOAL_BOX_WEIGHT": "100",
        "GOAL_BOX_HALF_EXTENT": "0.2",
        "GOAL_BOX_TEMPERATURE": "1.5",
        "FULL_H_TASKSPACE": "1",
        "STOPPING_MARGIN": "",
    })
    log = STAGE / "logs" / f"{arm['name']}.screen.log"
    command = ["bash", "scripts/RECIPE_control_braking_run_and_eval.sh"]
    if dry_run:
        return {
            "name": arm["name"], "output": str(output), "gpu": arm["gpu"],
            "status": "DRY_RUN", "command": command, "log": str(log),
        }
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a")
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    finally:
        stream.close()
    record = {
        "name": arm["name"], "output": str(output), "gpu": arm["gpu"],
        "weight": arm["weight"], "temperature_m": arm["temperature"],
        "target_round": 7, "pid": process.pid, "log": str(log),
        "launched_unix": time.time(), "status": "LAUNCHED",
    }
    _atomic_json(launch_marker, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _validate_source()
    outputs = [(arm, _prepare_arm(arm)) for arm in ARMS]
    records = [] if args.prepare_only else [
        _launch(arm, output, args.dry_run) for arm, output in outputs
    ]
    payload = {
        "status": (
            "PAIRED_R5_OBSTACLE_TAIL_SWEEP_PREPARED"
            if args.prepare_only else
            "PAIRED_R5_OBSTACLE_TAIL_SWEEP_LAUNCHED"
        ),
        "updated_unix": time.time(),
        "source": str(SOURCE),
        "source_resume_sha256": SOURCE_RESUME_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_round": 5,
        "target_round": 7,
        "fixed": {
            "target_m": 0.6,
            "contact_penalty": 15.0 * math.exp(4.0),
            "task_config": str(TASK_CONFIG),
            "pretrain": str(PRETRAIN),
            "seed": 82410,
            "K": 16,
            "B": 8,
            "retry_B": 8,
            "retry_verify_all_fast_path": False,
            "successful_trajectories_per_gamma": 12,
            "sample_update_mode": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "fixed_eval": {
                "rounds": [0, 1, 2, 3, 4, 5, 6, 7],
                "seed": 91000,
                "episodes_per_gamma": 40,
                "flow_nfe": 12
            },
            "gpu_policy": "GPU0/GPU2 prohibited; GPU1/GPU3 shared",
        },
        "calibration": {
            "method": "saved GP-selected B8 counterfactual at r5",
            "source_events": str(SOURCE / "events_round_005.pt"),
            "source_events_sha256": "0477c1cc20acd7ebc368074f03a582bea53d7d1c5daf064302c63e1c19a574aa",
            "contexts_with_robot_clearance_le_1p2m": 5920,
            "choice_change_rates": [0.048, 0.091, 0.133],
            "scope": "ranking-only calibration; not closed-loop SR evidence"
        },
        "legacy_comparator": {
            "weight": 15.0, "temperature_m": 0.15,
            "fixed_eval": str(ROOT / (
                "results/stage1_single_ball_t128/0811_pre2_r1_inward_oob_probe/"
                "reach03_continuations/fresh_r1_exp15_goalbox100_t150_s82410_"
                "reach03_r1_to_r5/fixed_eval_r000_r010/raw_eval.json"
            )),
        },
        "arms": records,
    }
    _atomic_json(STAGE / "SWEEP.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
