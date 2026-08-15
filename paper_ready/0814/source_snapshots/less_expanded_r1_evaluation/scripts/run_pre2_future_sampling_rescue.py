#!/usr/bin/env python3
"""Run one fail-closed future-sampling PRE2 round, then calibration-bank eval."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import torch


APPROVED_RECIPE_SHA256 = (
    "707b298e093416315cc8908eb35ebf26cb4b891598acac97da9f8a3303c2b639"
)
HISTORICAL_SEED = 82410
RESERVED_SEEDS = {91000, 93211, 95000, 193211}
HASHED_FIELDS = (
    "kind", "name", "physical_gpu", "source_id", "source_config_hash",
    "approved_recipe_sha256", "remote_source", "remote_pretrain",
    "remote_task_config", "parent_output", "branch_output",
    "remote_control", "parent_round", "target_round", "historical_seed",
    "future_sampling_seed", "expansion_args", "parent_contract",
    "evaluation",
)
PARENT_CONTRACT_FIELDS = {
    "completed_round", "optimizer_step", "historical_seed",
    "future_sampling_seed", "checkpoint_sha256", "resume_state_sha256",
    "resume_json_sha256", "manifest_sha256", "metrics_sha256",
}
NATIVE_EVALUATION = {
    "episodes_per_gamma": 40,
    "fixed_scene_rollouts": 10,
    "fixed_scene_seed": 193211,
    "flow_nfe": 12,
    "probe_samples": 16,
    "seed": 93211,
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"metrics rows must be objects: {path}")
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _arg_value(args: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(args) if value == name]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one {name}")
    try:
        return args[positions[0] + 1]
    except IndexError as error:
        raise ValueError(f"missing value for {name}") from error


def _validate_spec(spec: dict[str, Any], *, check_cuda: bool = True) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported future-sampling rescue schema")
    if spec.get("kind") != "pre2_future_sampling_one_round_rescue":
        raise ValueError("unexpected future-sampling rescue kind")
    missing = [field for field in HASHED_FIELDS if field not in spec]
    if missing:
        raise ValueError(f"missing hashed spec fields: {missing}")
    expected_payload = {field: spec[field] for field in HASHED_FIELDS}
    if spec.get("hash_payload") != expected_payload:
        raise ValueError("hash_payload is not the exact execution contract")
    if _canonical_sha256(expected_payload) != spec.get("config_hash"):
        raise ValueError("future-sampling rescue config hash mismatch")
    if spec["approved_recipe_sha256"] != APPROVED_RECIPE_SHA256:
        raise ValueError("approved W300/Adam5000 recipe hash changed")

    physical_gpu = int(spec["physical_gpu"])
    if physical_gpu not in {1, 3}:
        raise ValueError("future-sampling rescue is restricted to GPU1/GPU3")
    if check_cuda and os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {physical_gpu}"
        )
    parent_round = int(spec["parent_round"])
    target_round = int(spec["target_round"])
    if parent_round < 1 or target_round != parent_round + 1:
        raise ValueError("rescue must advance exactly one committed round")

    path_fields = (
        "remote_source", "remote_pretrain", "remote_task_config",
        "parent_output", "branch_output", "remote_control",
    )
    for key in path_fields:
        if not isinstance(spec.get(key), str) or not Path(spec[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    parent = Path(spec["parent_output"])
    branch = Path(spec["branch_output"])
    if parent == branch or parent in branch.parents or branch in parent.parents:
        raise ValueError("parent and branch outputs must be disjoint directories")

    historical_seed = int(spec["historical_seed"])
    future_seed = int(spec["future_sampling_seed"])
    if historical_seed != HISTORICAL_SEED:
        raise ValueError("historical training seed changed")
    if future_seed < 0 or future_seed == historical_seed:
        raise ValueError("future sampling seed must be a distinct nonnegative seed")
    if future_seed in RESERVED_SEEDS:
        raise ValueError("future sampling seed collides with a reserved eval seed")

    args = spec.get("expansion_args")
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise ValueError("expansion_args must be a string list")
    if _canonical_sha256(args) != APPROVED_RECIPE_SHA256:
        raise ValueError("expansion_args are not the approved W300/Adam5000 recipe")
    forbidden = {
        "--helios", "--helios-gpu", "--output", "--resume-from",
        "--pretrain-dir", "--lab-task-config", "--device", "--rounds",
        "--future-sampling-seed",
    }
    overlap = forbidden.intersection(args)
    if overlap:
        raise ValueError(f"worker-owned expansion arguments: {sorted(overlap)}")
    guarded = {
        "--optimizer-steps-per-round": "5000",
        "--execution-obstacle-speed-weight": "300.0",
        "--successful-trajectories-per-gamma": "12",
        "--sample-update-mode": "0,0,0,1,1,1,2,2,2,3,3,3",
        "--K": "16", "--B": "8", "--retry-B": "8",
        "--execution-cost-band-fraction": "0.05",
        "--replay-scope": "cumulative", "--replay-rounds": "100",
        "--seed": str(HISTORICAL_SEED),
    }
    for name, expected in guarded.items():
        actual = _arg_value(args, name)
        if actual != expected:
            raise ValueError(f"{name} changed: {actual!r} != {expected!r}")
    required = {
        "--no-optimizer-steps-total", "--verifier-full-h-taskspace",
        "--paired-noised-representation",
        "--execution-axis-cylinder-finite-segment",
    }
    absent = sorted(required.difference(args))
    if absent:
        raise ValueError(f"required W300 recipe flags missing: {absent}")
    if "--retry-verify-all-fast-path" in args:
        raise ValueError("faithful retry must not use verify-all fast path")

    contract = spec.get("parent_contract")
    if not isinstance(contract, dict) or set(contract) != PARENT_CONTRACT_FIELDS:
        raise ValueError("parent_contract fields changed")
    if int(contract["completed_round"]) != parent_round:
        raise ValueError("parent contract round changed")
    if int(contract["optimizer_step"]) != parent_round * 5000:
        raise ValueError("parent optimizer step is not Adam5000 x round")
    if int(contract["historical_seed"]) != historical_seed:
        raise ValueError("parent historical seed changed")
    previous_future = contract["future_sampling_seed"]
    if previous_future is not None and int(previous_future) == future_seed:
        raise ValueError("future sampling seed must differ from the parent stream")
    for name in (
        "checkpoint_sha256", "resume_state_sha256", "resume_json_sha256",
        "manifest_sha256", "metrics_sha256",
    ):
        value = contract[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid parent contract digest: {name}")
    if spec.get("evaluation") != NATIVE_EVALUATION:
        raise ValueError("calibration-bank native evaluation contract changed")


def _boundary_contract(output: Path, expected_round: int) -> dict[str, Any]:
    if (output / "RESUME_IN_PROGRESS.json").exists():
        raise RuntimeError(f"refusing in-progress boundary: {output}")
    checkpoint = output / f"checkpoint_{expected_round:03d}.pt"
    required = {
        "checkpoint": checkpoint,
        "query_archive": output / f"query_archive_round_{expected_round:03d}.pt",
        "events": output / f"events_round_{expected_round:03d}.pt",
        "resume_json": output / "resume_state.json",
        "resume_state": output / "resume_state_latest.pt",
        "manifest": output / "manifest.json",
        "metrics": output / "metrics.jsonl",
    }
    missing = [
        str(path) for path in required.values()
        if not path.is_file() or not path.stat().st_size
    ]
    if missing:
        raise FileNotFoundError("incomplete committed boundary: " + ", ".join(missing))
    metadata = _read_json(required["resume_json"])
    optimizer_step = expected_round * 5000
    if (
        metadata.get("status") != "COMMITTED_ROUND_RESUME"
        or int(metadata.get("version", -1)) != 1
        or int(metadata.get("completed_round", -1)) != expected_round
        or int(metadata.get("next_round", -1)) != expected_round + 1
        or int(metadata.get("optimizer_step", -1)) != optimizer_step
    ):
        raise RuntimeError("resume metadata is not the expected exact boundary")
    state = torch.load(
        required["resume_state"], map_location="cpu", weights_only=False,
    )
    if (
        not isinstance(state, dict)
        or state.get("status") != "COMMITTED_ROUND_RESUME"
        or int(state.get("version", -1)) != 1
        or int(state.get("completed_round", -1)) != expected_round
    ):
        raise RuntimeError("resume payload is not the expected exact boundary")
    schedule_step = int(
        state["optimizer_metadata"]["_safe_mppi_schedule_step"]
    )
    if schedule_step != optimizer_step:
        raise RuntimeError("optimizer schedule step does not match Adam5000 boundary")
    config = state.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("resume payload config is absent")
    guarded_config = {
        "seed": HISTORICAL_SEED, "inner_steps": 5000,
        "successful_trajectories_per_gamma": 12,
        "K": 16, "B": 8, "retry_B": 8,
        "execution_cost_band_fraction": 0.05,
        "replay_scope": "cumulative", "replay_rounds": 100,
        "retry_verify_all_fast_path": False,
        "paired_noised_representation": True,
    }
    for name, expected in guarded_config.items():
        if config.get(name) != expected:
            raise RuntimeError(
                f"resume config {name} changed: {config.get(name)!r} != {expected!r}"
            )
    manifest = _read_json(required["manifest"])
    metrics = _read_metrics(required["metrics"])
    expected_numbers = list(range(1, expected_round + 1))
    if [int(row.get("round", -1)) for row in metrics] != expected_numbers:
        raise RuntimeError("metrics do not contain one exact row per committed round")
    if not (
        state.get("round_rows") == manifest.get("rounds") == metrics
        and len(metrics) == expected_round
    ):
        raise RuntimeError("resume, manifest, and metrics histories differ")
    return {
        "completed_round": expected_round,
        "optimizer_step": optimizer_step,
        "historical_seed": int(config["seed"]),
        "future_sampling_seed": config.get("future_sampling_seed"),
        "checkpoint_sha256": _sha256(checkpoint),
        "resume_state_sha256": _sha256(required["resume_state"]),
        "resume_json_sha256": _sha256(required["resume_json"]),
        "manifest_sha256": _sha256(required["manifest"]),
        "metrics_sha256": _sha256(required["metrics"]),
    }


def _assert_parent_contract(spec: dict[str, Any]) -> dict[str, Any]:
    actual = _boundary_contract(
        Path(spec["parent_output"]), int(spec["parent_round"]),
    )
    if actual != spec["parent_contract"]:
        differences = {
            key: {"expected": spec["parent_contract"].get(key), "actual": value}
            for key, value in actual.items()
            if spec["parent_contract"].get(key) != value
        }
        raise RuntimeError(f"parent exact-state contract changed: {differences}")
    return actual


def _validate_branch_provenance(spec: dict[str, Any]) -> None:
    branch = Path(spec["branch_output"])
    provenance = _read_json(branch / "FUTURE_SAMPLING_STREAM.json")
    parent = spec["parent_contract"]
    expected = {
        "status": "FUTURE_SAMPLING_STREAM_BRANCH_READY",
        "source": str(Path(spec["parent_output"]).resolve()),
        "output": str(branch.resolve()),
        "completed_round": int(spec["parent_round"]),
        "historical_seed": HISTORICAL_SEED,
        "future_sampling_seed": int(spec["future_sampling_seed"]),
        "source_checkpoint_sha256": parent["checkpoint_sha256"],
        "source_resume_sha256": parent["resume_state_sha256"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"future branch provenance changed at {key}")
    launch = provenance.get("launch_contract", {})
    if (
        launch.get("resume_from_must_equal_output") is not True
        or int(launch.get("seed", -1)) != HISTORICAL_SEED
        or int(launch.get("future_sampling_seed", -1))
        != int(spec["future_sampling_seed"])
        or int(launch.get("minimum_target_round", -1))
        != int(spec["target_round"])
    ):
        raise RuntimeError("future branch launch contract changed")
    if "final holdout is not used" not in provenance.get("selection_scope", ""):
        raise RuntimeError("future branch selection scope is not calibration-only")


def _expansion_command(spec: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(Path(spec["remote_source"]) / "scripts/research_ball_expansion_optimization.py"),
        "--pretrain-dir", spec["remote_pretrain"],
        "--lab-task-config", spec["remote_task_config"],
        "--output", spec["branch_output"],
        "--resume-from", spec["branch_output"],
        "--device", "cuda:0",
        "--rounds", str(spec["target_round"]),
        *spec["expansion_args"],
        "--future-sampling-seed", str(spec["future_sampling_seed"]),
    ]


def _evaluation_output(spec: dict[str, Any]) -> Path:
    parent = int(spec["parent_round"])
    target = int(spec["target_round"])
    evaluation = spec["evaluation"]
    return Path(spec["branch_output"]) / (
        f"fixed_eval_r{parent:03d}_r{target:03d}_native_calibration_"
        f"seed{evaluation['seed']}_n{evaluation['episodes_per_gamma']}"
    )


def _evaluation_command(spec: dict[str, Any], output: Path) -> list[str]:
    config = spec["evaluation"]
    parent = str(spec["parent_round"])
    target = str(spec["target_round"])
    return [
        sys.executable,
        str(Path(spec["remote_source"]) / "scripts/research_evaluate_ball_expansion.py"),
        "--pretrain-dir", spec["remote_pretrain"],
        "--expansion", spec["branch_output"],
        "--expansion-manifest", str(Path(spec["branch_output"]) / "manifest.json"),
        "--lab-task-config", spec["remote_task_config"],
        "--evaluation-output", str(output),
        "--device", "cuda:0",
        "--episodes", str(config["episodes_per_gamma"]),
        "--probe-samples", str(config["probe_samples"]),
        "--flow-nfe", str(config["flow_nfe"]),
        "--evaluation-rounds", parent, target,
        "--seed", str(config["seed"]),
        "--fixed-scene-rollouts", str(config["fixed_scene_rollouts"]),
        "--fixed-scene-seed", str(config["fixed_scene_seed"]),
        "--gallery-rounds", parent, target,
        "--gallery-view", "head_on",
        "--save-raw-trajectories", "--screening-only",
    ]


def _native_eval_complete(spec: dict[str, Any], output: Path) -> bool:
    raw_eval = output / "raw_eval.json"
    trajectories = output / "raw_trajectories.pt"
    if (
        not raw_eval.is_file() or not raw_eval.stat().st_size
        or not trajectories.is_file() or not trajectories.stat().st_size
    ):
        return False
    payload = _read_json(raw_eval)
    expected_rounds = {
        str(spec["parent_round"]), str(spec["target_round"]),
    }
    rows = payload.get("rows", {})
    probes = payload.get("probe_rows", {})
    expected_rows = 4 * int(spec["evaluation"]["episodes_per_gamma"])
    return (
        payload.get("status") == "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
        and payload.get("sampling_temperature") == 1.0
        and payload.get("sigma_tilt_used") is False
        and int(payload.get("flow_nfe_override", -1))
        == int(spec["evaluation"]["flow_nfe"])
        and set(payload.get("summary", {})) == expected_rounds
        and set(rows) == expected_rounds
        and set(probes) == expected_rounds
        and all(len(rows[key]) == expected_rows for key in expected_rounds)
    )


def _run(command: list[str], cwd: Path) -> None:
    print("[future-sampling-rescue]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run(spec_path: Path) -> None:
    spec = _read_json(spec_path)
    _validate_spec(spec)
    parent = Path(spec["parent_output"])
    branch = Path(spec["branch_output"])
    control = Path(spec["remote_control"])
    key = f"{spec['name']}--{spec['config_hash'][:12]}"
    status_dir = control / "status"
    lock_path = control / "locks" / f"{key}.lock"
    running = status_dir / f"{key}.RUNNING.json"
    complete = status_dir / f"{key}.COMPLETE.json"
    failed = status_dir / f"{key}.FAILED.json"
    status_dir.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"duplicate future-sampling worker: {key}") from error

        evaluation_output = _evaluation_output(spec)
        if complete.is_file():
            marker = _read_json(complete)
            if (
                marker.get("status") != "CALIBRATION_BRANCH_COMPLETE"
                or marker.get("config_hash") != spec["config_hash"]
                or marker.get("branch_output") != spec["branch_output"]
            ):
                raise RuntimeError("COMPLETE marker does not match this exact branch")
            _assert_parent_contract(spec)
            _validate_branch_provenance(spec)
            branch_contract = _boundary_contract(branch, int(spec["target_round"]))
            if branch_contract["future_sampling_seed"] != spec["future_sampling_seed"]:
                raise RuntimeError("complete branch has the wrong future sampling seed")
            if not _native_eval_complete(spec, evaluation_output):
                raise RuntimeError("COMPLETE marker exists without native calibration eval")
            return

        started = time.time()
        _atomic_json(running, {
            "status": "RUNNING", "pid": os.getpid(), "name": spec["name"],
            "physical_gpu": spec["physical_gpu"], "started_unix": started,
            "config_hash": spec["config_hash"], "parent_output": str(parent),
            "branch_output": str(branch), "parent_round": spec["parent_round"],
            "target_round": spec["target_round"],
            "future_sampling_seed": spec["future_sampling_seed"],
        })
        try:
            _assert_parent_contract(spec)
            if branch.exists():
                raise FileExistsError(
                    f"partial/existing branch preserved fail-closed: {branch}"
                )
            prepare_command = [
                sys.executable,
                str(Path(spec["remote_source"]) / "scripts/prepare_pre2_future_seed_branch.py"),
                "--source", str(parent), "--output", str(branch),
                "--future-sampling-seed", str(spec["future_sampling_seed"]),
            ]
            _run(prepare_command, Path(spec["remote_source"]))
            _validate_branch_provenance(spec)
            prepared = _boundary_contract(branch, int(spec["parent_round"]))
            if prepared != spec["parent_contract"]:
                raise RuntimeError("prepared branch is not an exact parent-state clone")
            _assert_parent_contract(spec)

            expansion_command = _expansion_command(spec)
            if expansion_command.count("--future-sampling-seed") != 1:
                raise RuntimeError("future sampling seed must be injected exactly once")
            _run(expansion_command, Path(spec["remote_source"]))
            branch_contract = _boundary_contract(branch, int(spec["target_round"]))
            if branch_contract["historical_seed"] != HISTORICAL_SEED:
                raise RuntimeError("branch historical seed changed")
            if branch_contract["future_sampling_seed"] != spec["future_sampling_seed"]:
                raise RuntimeError("dedicated future sampling seed was not committed")
            _assert_parent_contract(spec)

            if evaluation_output.exists():
                raise FileExistsError(
                    f"partial calibration evaluation preserved fail-closed: {evaluation_output}"
                )
            evaluation_command = _evaluation_command(spec, evaluation_output)
            forbidden_eval = {
                "--future-sampling-seed", "--execution-obstacle-speed-weight",
                "--execution-cost-band-fraction", "--verifier-full-h-taskspace",
            }
            if forbidden_eval.intersection(evaluation_command):
                raise RuntimeError("native evaluation contains expansion-time shaping")
            _run(evaluation_command, Path(spec["remote_source"]))
            if not _native_eval_complete(spec, evaluation_output):
                raise RuntimeError("native calibration evaluation did not complete")
            _atomic_json(evaluation_output / "EVALUATION_PROVENANCE.json", {
                "schema_version": 1,
                "status": "CALIBRATION_BANK_NATIVE_RAW_COMPLETE",
                "config_hash": spec["config_hash"],
                "selection_bank_seed": spec["evaluation"]["seed"],
                "fixed_scene_seed": spec["evaluation"]["fixed_scene_seed"],
                "episodes_per_gamma": spec["evaluation"]["episodes_per_gamma"],
                "evaluated_rounds": [spec["parent_round"], spec["target_round"]],
                "native_deployment": (
                    "temperature-1 raw; no verifier, expansion cost shaping, "
                    "speed penalty, or bounded-cost margin selector"
                ),
                "final_seed91000_used_for_selection": False,
                "command": evaluation_command,
            })
            _assert_parent_contract(spec)
            finished = time.time()
            _atomic_json(complete, {
                "schema_version": 1,
                "status": "CALIBRATION_BRANCH_COMPLETE",
                "name": spec["name"], "config_hash": spec["config_hash"],
                "physical_gpu": spec["physical_gpu"],
                "started_unix": started, "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "parent_output": str(parent), "branch_output": str(branch),
                "parent_round": spec["parent_round"],
                "target_round": spec["target_round"],
                "historical_seed": HISTORICAL_SEED,
                "future_sampling_seed": spec["future_sampling_seed"],
                "parent_contract": spec["parent_contract"],
                "branch_contract": branch_contract,
                "raw_eval": str(evaluation_output / "raw_eval.json"),
                "raw_eval_sha256": _sha256(evaluation_output / "raw_eval.json"),
                "raw_trajectories_sha256": _sha256(
                    evaluation_output / "raw_trajectories.pt"
                ),
                "evaluation": spec["evaluation"],
                "selection_scope": "calibration bank only; final seed91000 unused",
            })
            failed.unlink(missing_ok=True)
        except BaseException as error:
            _atomic_json(failed, {
                "schema_version": 1, "status": "FAILED_CLOSED",
                "name": spec.get("name"), "config_hash": spec.get("config_hash"),
                "physical_gpu": spec.get("physical_gpu"),
                "failed_unix": time.time(), "error": repr(error),
                "traceback": traceback.format_exc(),
                "parent_output": str(parent), "branch_output": str(branch),
                "preservation_policy": (
                    "parent untouched; partial branch/evaluation retained; no auto-retry"
                ),
            })
            raise
        finally:
            running.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(args.spec.resolve())


if __name__ == "__main__":
    main()
