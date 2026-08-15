#!/usr/bin/env python3
"""Prepare, dry-run, or explicitly launch the paired E15 update sweep.

The default action is preparation only.  Launching requires both ``--launch``
and the exact confirmation phrase so an inspection command can never start GPU
work accidentally.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / (
    "results/stage1_single_ball_t128/0811_pre2_r1_inward_oob_probe/"
    "resume_backups/reach03_e15_r5_exact"
)
STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0811_pre2_e15_update_aggregation_sweep"
)
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
TASK_CONFIG = ROOT / (
    "configs/lab_ball_stage1_goalspace_yminus04_z01_17_"
    "r15in_reach03_v1.json"
)
SOURCE_RESUME_SHA256 = (
    "1ca1462c877a419d49bc3dec1fa75c12e50f0911d320f1e9d4f1fc7933dff82d"
)
SOURCE_CHECKPOINT_SHA256 = (
    "3291d6a90f0f94ccbcd0b8a69314e66337156243c2a94e888b5a9692a6a8f805"
)
CONFIRMATION = "I_UNDERSTAND_THIS_STARTS_FIVE_HELIOS_JOBS"

R5_ARMS = (
    {
        "name": "r5_opt0010_expmax_w5p0584", "gpu": 1,
        "optimizer_steps": 10, "aggregation": "max",
        "weight": 5.0583765087,
    },
    {
        "name": "r5_opt0050_expmax_w5p0584", "gpu": 3,
        "optimizer_steps": 50, "aggregation": "max",
        "weight": 5.0583765087,
    },
    {
        "name": "r5_opt0010_exptop3_w6p7799", "gpu": 1,
        "optimizer_steps": 10, "aggregation": "top3_mean",
        "weight": 6.7799170530,
    },
    {
        "name": "r5_opt0050_exptop3_w6p7799", "gpu": 3,
        "optimizer_steps": 50, "aggregation": "top3_mean",
        "weight": 6.7799170530,
    },
)
FRESH_ARM = {
    "name": "fresh_pre2_opt0100_expmean_q8angular",
    "gpu": 1,
    "optimizer_steps": 100,
    "aggregation": "mean",
    "weight": 15.0,
    "fresh_pre2": True,
    "angular8": True,
}

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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _same_value(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left) and torch.is_tensor(right)
            and left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict) and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(_same_value(a, b) for a, b in zip(left, right))
        )
    if is_dataclass(left) or is_dataclass(right):
        return (
            type(left) is type(right) and is_dataclass(left)
            and all(
                _same_value(getattr(left, field.name), getattr(right, field.name))
                for field in fields(left)
            )
        )
    return left == right


def _validate_code_contract() -> None:
    entrypoint = (
        ROOT / "scripts/research_ball_expansion_optimization.py"
    ).read_text()
    wrapper = (ROOT / "scripts/RECIPE_control_braking_trunk3.sh").read_text()
    required = {
        "aggregation CLI": "--execution-clearance-exp-aggregation",
        "angular8 CLI": "--sample-update-submodes",
    }
    for label, token in required.items():
        if token not in entrypoint:
            raise RuntimeError(f"missing {label}: {token}")
    for token in (
        "CLEARANCE_EXP_AGGREGATION", "OPTIMIZER_STEPS_PER_ROUND",
        "SAMPLE_UPDATE_SUBMODES",
    ):
        if token not in wrapper:
            raise RuntimeError(f"wrapper does not expose {token}")


def _validate_source() -> dict[str, Any]:
    _validate_code_contract()
    for name in MUTABLE + IMMUTABLE:
        if not (SOURCE / name).is_file():
            raise FileNotFoundError(SOURCE / name)
    metadata = json.loads((SOURCE / "resume_state.json").read_text())
    if (
        metadata.get("status") != "COMMITTED_ROUND_RESUME"
        or int(metadata.get("completed_round", -1)) != 5
        or int(metadata.get("optimizer_step", -1)) != 12500
    ):
        raise ValueError("source is not exact committed E15 r5/step12500")
    if _sha256(SOURCE / "resume_state_latest.pt") != SOURCE_RESUME_SHA256:
        raise ValueError("source r5 serialized resume hash changed")
    if _sha256(SOURCE / "checkpoint_005.pt") != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("source r5 checkpoint hash changed")
    state = torch.load(
        SOURCE / "resume_state_latest.pt", map_location="cpu",
        weights_only=False,
    )
    if int(state.get("completed_round", -1)) != 5:
        raise ValueError("serialized state does not end at r5")
    config = state["config"]
    expected = {
        "inner_steps": 2500,
        "microbatch_repeats": 1,
        "successful_trajectories_per_gamma": 12,
        "sample_update_mode": (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3),
        "execution_rule": "exponential_cost",
        "K": 16, "B": 8, "retry_B": 8,
        "retry_verify_all_fast_path": False,
        "seed": 82410,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(
                f"source recipe mismatch for {key}: {config.get(key)!r}"
            )
    return state


def _patch_resume_exposure(
    path: Path, source_state: dict[str, Any], optimizer_steps: int,
) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not _same_value(state, source_state):
        raise RuntimeError("copied serialized resume differs before patch")
    state["config"]["inner_steps"] = optimizer_steps
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    patched = torch.load(path, map_location="cpu", weights_only=False)
    if patched["config"]["inner_steps"] != optimizer_steps:
        raise RuntimeError("optimizer exposure patch was not serialized")
    for key in source_state:
        if key == "config":
            continue
        if not _same_value(source_state[key], patched[key]):
            raise RuntimeError(f"resume payload changed outside config: {key}")
    expected_config = dict(source_state["config"])
    expected_config["inner_steps"] = optimizer_steps
    if not _same_value(expected_config, patched["config"]):
        raise RuntimeError("resume config changed beyond inner_steps")
    return _sha256(path)


def _prepare_r5_arm(
    arm: dict[str, Any], source_state: dict[str, Any],
) -> Path:
    output = STAGE / "arms" / arm["name"]
    provenance = output / "BRANCH_PROVENANCE.json"
    if output.exists():
        if provenance.is_file():
            payload = json.loads(provenance.read_text())
            configured_steps = payload.get(
                "optimizer_steps_per_round",
                payload.get("authorized_resume_config_delta", {}).get("branch"),
            )
            if (
                payload.get("source_resume_sha256") == SOURCE_RESUME_SHA256
                and configured_steps == arm["optimizer_steps"]
                and payload.get("aggregation") == arm["aggregation"]
            ):
                return output
        raise FileExistsError(f"refusing to overwrite {output}")
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
                raise RuntimeError(f"immutable artifact not hard-linked: {name}")
        for name in MUTABLE:
            if (SOURCE / name).stat().st_ino == (temporary / name).stat().st_ino:
                raise RuntimeError(f"mutable artifact aliases source: {name}")
        patched_sha = _patch_resume_exposure(
            temporary / "resume_state_latest.pt", source_state,
            arm["optimizer_steps"],
        )
        record = {
            "status": "EXACT_R5_OPTIMIZER_EXPOSURE_BRANCH_READY",
            "source": str(SOURCE),
            "source_round": 5,
            "source_optimizer_step": 12500,
            "source_resume_sha256": SOURCE_RESUME_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "patched_resume_sha256": patched_sha,
            "authorized_resume_config_delta": {
                "field": "inner_steps",
                "meaning": "optimizer_steps_per_round",
                "source": 2500,
                "branch": arm["optimizer_steps"],
                "microbatch_repeats_unchanged": 1,
            },
            "preserved_state_contract": (
                "model, Adam and schedule state, NumPy/torch/device RNG, GP, "
                "replay, query archives and committed events are value-equal; "
                "only copied resume config.inner_steps changes"
            ),
            "aggregation": arm["aggregation"],
            "optimizer_steps_per_round": arm["optimizer_steps"],
            "clearance_weight": arm["weight"],
            "clearance_target_m": 0.6,
            "clearance_temperature_m": 0.15,
            "gpu": arm["gpu"],
            "target_round": 7,
            "source_mutated": False,
        }
        (temporary / "BRANCH_PROVENANCE.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _environment(arm: dict[str, Any], output: Path) -> dict[str, str]:
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
        "CLEARANCE_EXP_TEMPERATURE": "0.15",
        "CLEARANCE_EXP_TARGET": "0.6",
        "CLEARANCE_EXP_AGGREGATION": arm["aggregation"],
        "OPTIMIZER_STEPS_PER_ROUND": str(arm["optimizer_steps"]),
        "SUCCESS_QUOTA": "8" if arm.get("angular8") else "12",
        "SAMPLE_MODES": "0,0,0,1,1,1,2,2,2,3,3,3",
        "SAMPLE_UPDATE_SUBMODES": "angular8" if arm.get("angular8") else "none",
        "FAITHFUL_RETRY": "1",
        "GOAL_BOX_WEIGHT": "100",
        "GOAL_BOX_HALF_EXTENT": "0.2",
        "GOAL_BOX_TEMPERATURE": "1.5",
        "FULL_H_TASKSPACE": "1",
        "STOPPING_MARGIN": "",
    })
    return environment


def _active(output: Path) -> bool:
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    return any(str(output) in line for line in commands.splitlines())


def _launch_arm(
    arm: dict[str, Any], output: Path, *, dry_run: bool,
) -> dict[str, Any]:
    if arm["gpu"] not in {1, 3}:
        raise ValueError("GPU0/GPU2 are prohibited")
    marker = STAGE / "launch_records" / f"{arm['name']}.json"
    if marker.is_file() or _active(output):
        return {"name": arm["name"], "status": "ALREADY_LAUNCHED_OR_ACTIVE"}
    if arm.get("fresh_pre2") and output.exists():
        raise FileExistsError(f"fresh output must not pre-exist: {output}")
    command = ["bash", "scripts/RECIPE_control_braking_run_and_eval.sh"]
    log = STAGE / "logs" / f"{arm['name']}.screen.log"
    record = {
        "name": arm["name"], "output": str(output), "gpu": arm["gpu"],
        "target_round": 7, "command": command, "log": str(log),
        "environment_delta": {
            key: value for key, value in _environment(arm, output).items()
            if key in {
                "HELIOS_GPU", "ROUNDS", "CLEARANCE_EXP_WEIGHT",
                "CLEARANCE_EXP_AGGREGATION", "OPTIMIZER_STEPS_PER_ROUND",
                "SUCCESS_QUOTA", "SAMPLE_UPDATE_SUBMODES",
            }
        },
        "status": "DRY_RUN" if dry_run else "LAUNCHED",
    }
    if dry_run:
        return record
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=_environment(arm, output),
            stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record.update({"pid": process.pid, "launched_unix": time.time()})
    _atomic_json(marker, record)
    return record


def _stage_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "PREPARED_NOT_LAUNCHED" if not records else records[0]["status"],
        "updated_unix": time.time(),
        "source": str(SOURCE),
        "source_resume_sha256": SOURCE_RESUME_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "fresh_pre2_contract": (
            "fresh checkpoint0 from PRE2 with no r5 replay, GP, Adam, or RNG inheritance"
        ),
        "target_round": 7,
        "fixed_evaluation": {
            "rounds": list(range(8)), "seed": 91000,
            "episodes_per_gamma": 40, "flow_nfe": 12,
            "one_shot": True,
        },
        "success_contract": {
            "collision_rate": "trend toward 0",
            "validity": "trend toward 1",
            "mean_min_clearance_m": "rise then plateau",
            "time_to_goal_s": "rise then plateau; no hard cutoff",
            "gamma_trend": (
                "report measured gamma-wise TtG/clearance; expected higher "
                "gamma has shorter TtG and lower clearance, never fabricate monotonicity"
            ),
            "coverage": "report four-mode and angular8 submode counts",
        },
        "continuation_policy": "NO_AUTO_CONTINUATION_BEYOND_R7",
        "gpu_policy": (
            "GPU0/GPU2 prohibited; share GPU1/GPU3; never stop existing jobs"
        ),
        "arms": [*R5_ARMS, FRESH_ARM],
        "launch_records": records,
    }


def prepare() -> dict[str, Any]:
    source_state = _validate_source()
    STAGE.mkdir(parents=True, exist_ok=True)
    for arm in R5_ARMS:
        _prepare_r5_arm(arm, source_state)
    fresh_output = STAGE / "arms" / FRESH_ARM["name"]
    if fresh_output.exists():
        raise FileExistsError(
            f"fresh PRE2 output must remain absent before launch: {fresh_output}"
        )
    payload = _stage_payload([])
    _atomic_json(STAGE / "SWEEP.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirm-launch", default="")
    args = parser.parse_args()
    if args.launch and args.dry_run:
        parser.error("--launch and --dry-run are mutually exclusive")
    prepare()
    arms = [*R5_ARMS, FRESH_ARM]
    if not (args.launch or args.dry_run):
        print((STAGE / "SWEEP.json").read_text(), end="")
        return
    if args.launch and args.confirm_launch != CONFIRMATION:
        parser.error(f"--launch requires --confirm-launch {CONFIRMATION}")
    records = []
    for arm in arms:
        output = STAGE / "arms" / arm["name"]
        records.append(_launch_arm(arm, output, dry_run=args.dry_run))
    payload = _stage_payload(records)
    _atomic_json(STAGE / "SWEEP.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
