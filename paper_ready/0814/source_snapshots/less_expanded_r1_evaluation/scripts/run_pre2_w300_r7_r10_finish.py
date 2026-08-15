#!/usr/bin/env python3
"""Continue the exact W300/5000 winner from r7 to r10 and evaluate it."""
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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


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


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported continuation spec schema")
    if spec.get("kind") != "w300_steps5000_exact_r7_r10":
        raise ValueError("unexpected continuation kind")
    payload = spec.get("hash_payload")
    if not isinstance(payload, dict):
        raise ValueError("hash_payload must be an object")
    if _canonical_sha256(payload) != spec.get("config_hash"):
        raise ValueError("continuation config hash mismatch")
    for key, value in payload.items():
        if spec.get(key) != value:
            raise ValueError(f"{key} differs from the hashed payload")
    if int(spec["physical_gpu"]) != 3:
        raise ValueError("this continuation is pinned to physical GPU3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 3")
    if int(spec["start_round"]) != 7 or int(spec["target_round"]) != 10:
        raise ValueError("the only permitted boundary is r7 -> r10")
    for key in (
        "remote_source", "remote_pretrain", "remote_task_config",
        "remote_output", "remote_control",
    ):
        if not isinstance(spec.get(key), str) or not spec[key].startswith("/"):
            raise ValueError(f"{key} must be an absolute path")
    args = spec.get("expansion_args")
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise ValueError("expansion_args must be a string list")
    forbidden = {
        "--helios", "--helios-gpu", "--output", "--resume-from",
        "--pretrain-dir", "--lab-task-config", "--device", "--rounds",
    }
    overlap = forbidden.intersection(args)
    if overlap:
        raise ValueError(f"transport-owned expansion arguments: {sorted(overlap)}")
    guarded_values = {
        "--optimizer-steps-per-round": "5000",
        "--execution-obstacle-speed-weight": "300.0",
        "--successful-trajectories-per-gamma": "12",
        "--K": "16",
        "--B": "8",
        "--retry-B": "8",
        "--execution-cost-band-fraction": "0.05",
        "--replay-scope": "cumulative",
        "--replay-rounds": "100",
        "--seed": "82410",
    }
    for name, expected in guarded_values.items():
        actual = _arg_value(args, name)
        if actual != expected:
            raise ValueError(f"{name} changed: {actual!r} != {expected!r}")
    for required in (
        "--verifier-full-h-taskspace", "--paired-noised-representation",
        "--no-optimizer-steps-total", "--axis-cylinder-finite-segment",
    ):
        # Historical specs use the longer execution-prefixed cylinder flag.
        if required == "--axis-cylinder-finite-segment":
            required = "--execution-axis-cylinder-finite-segment"
        if required not in args:
            raise ValueError(f"required recipe flag missing: {required}")
    if "--retry-verify-all-fast-path" in args:
        raise ValueError("faithful retry must not use the verify-all fast path")
    evaluation = spec.get("evaluation")
    if evaluation != {
        "episodes_per_gamma": 160,
        "fixed_scene_rollouts": 10,
        "fixed_scene_seed": 191000,
        "flow_nfe": 12,
        "probe_samples": 16,
        "seed": 91000,
    }:
        raise ValueError("native evaluation contract changed")


def _resume_boundary(output: Path, expected_round: int) -> dict[str, Any]:
    optimizer_step = expected_round * 5000
    required = [
        output / f"checkpoint_{expected_round:03d}.pt",
        output / f"query_archive_round_{expected_round:03d}.pt",
        output / f"events_round_{expected_round:03d}.pt",
        output / "resume_state.json",
        output / "resume_state_latest.pt",
        output / "metrics.jsonl",
        output / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("incomplete committed boundary: " + ", ".join(missing))
    metadata = _read_json(output / "resume_state.json")
    if (
        metadata.get("status") != "COMMITTED_ROUND_RESUME"
        or int(metadata.get("completed_round", -1)) != expected_round
        or int(metadata.get("optimizer_step", -1)) != optimizer_step
    ):
        raise RuntimeError("resume metadata does not match the committed boundary")
    return metadata


def _restore_committed_manifest(output: Path, current_round: int) -> None:
    """Repair only the known pre-commit resume interruption state."""
    manifest = output / "manifest.json"
    if manifest.is_file():
        return
    in_progress = output / "RESUME_IN_PROGRESS.json"
    archived = output / f"manifest_before_resume_round_{current_round:03d}.json"
    if not in_progress.is_file() or not archived.is_file():
        raise FileNotFoundError(
            "manifest is absent without the known resume-interruption evidence"
        )
    state = _read_json(in_progress)
    if int(state.get("resumed_from_round", -1)) != current_round:
        raise RuntimeError("resume-interruption marker disagrees with latest commit")
    # Copy, not rename: preserve interruption provenance for audit.
    temporary = manifest.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_bytes(archived.read_bytes())
    temporary.replace(manifest)


def _native_eval_complete(path: Path) -> bool:
    raw_eval = path / "raw_eval.json"
    trajectories = path / "raw_trajectories.pt"
    if not raw_eval.is_file() or not trajectories.is_file():
        return False
    payload = _read_json(raw_eval)
    return (
        payload.get("status") == "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
        and payload.get("sampling_temperature") == 1.0
        and payload.get("sigma_tilt_used") is False
        and set(payload.get("summary", {})) == {str(value) for value in range(11)}
    )


def _run(command: list[str], cwd: Path) -> None:
    print("[w300-r10]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _evaluation_command(spec: dict[str, Any], output: Path) -> list[str]:
    config = spec["evaluation"]
    return [
        sys.executable,
        str(Path(spec["remote_source"]) / "scripts/research_evaluate_ball_expansion.py"),
        "--pretrain-dir", spec["remote_pretrain"],
        "--expansion", spec["remote_output"],
        "--expansion-manifest", str(Path(spec["remote_output"]) / "manifest.json"),
        "--lab-task-config", spec["remote_task_config"],
        "--evaluation-output", str(output),
        "--device", "cuda:0",
        "--episodes", str(config["episodes_per_gamma"]),
        "--probe-samples", str(config["probe_samples"]),
        "--flow-nfe", str(config["flow_nfe"]),
        "--evaluation-rounds", *(str(value) for value in range(11)),
        "--seed", str(config["seed"]),
        "--fixed-scene-rollouts", str(config["fixed_scene_rollouts"]),
        "--fixed-scene-seed", str(config["fixed_scene_seed"]),
        "--gallery-rounds", "7", "8", "9", "10",
        "--gallery-view", "head_on",
        "--save-raw-trajectories", "--screening-only",
    ]


def run(spec_path: Path) -> None:
    spec = _read_json(spec_path)
    _validate_spec(spec)
    output = Path(spec["remote_output"])
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
            raise RuntimeError(f"duplicate continuation worker: {key}") from error
        evaluation_output = output / "fixed_eval_r000_r010_native_seed91000_n160"
        if complete.is_file():
            if not _native_eval_complete(evaluation_output):
                raise RuntimeError("COMPLETE marker exists without complete evaluation")
            return
        started = time.time()
        _atomic_json(running, {
            "status": "RUNNING", "pid": os.getpid(), "name": spec["name"],
            "physical_gpu": 3, "started_unix": started,
            "config_hash": spec["config_hash"],
        })
        try:
            metadata = _read_json(output / "resume_state.json")
            current_round = int(metadata.get("completed_round", -1))
            if current_round not in {7, 8, 9, 10}:
                raise RuntimeError(f"unexpected resume round {current_round}")
            _restore_committed_manifest(output, current_round)
            _resume_boundary(output, current_round)
            if current_round < 10:
                command = [
                    sys.executable,
                    str(Path(spec["remote_source"]) / "scripts/research_ball_expansion_optimization.py"),
                    "--pretrain-dir", spec["remote_pretrain"],
                    "--lab-task-config", spec["remote_task_config"],
                    "--output", spec["remote_output"],
                    "--resume-from", spec["remote_output"],
                    "--device", "cuda:0", "--rounds", "10",
                    *spec["expansion_args"],
                ]
                _run(command, Path(spec["remote_source"]))
            _resume_boundary(output, 10)
            if not _native_eval_complete(evaluation_output):
                if evaluation_output.exists() and any(evaluation_output.iterdir()):
                    raise RuntimeError("partial r0-r10 evaluation preserved fail-closed")
                _run(
                    _evaluation_command(spec, evaluation_output),
                    Path(spec["remote_source"]),
                )
            if not _native_eval_complete(evaluation_output):
                raise RuntimeError("r0-r10 native evaluation did not complete")
            finished = time.time()
            _atomic_json(complete, {
                "status": "COMPLETE", "name": spec["name"],
                "physical_gpu": 3, "rounds": 10,
                "config_hash": spec["config_hash"],
                "started_unix": started, "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "remote_output": spec["remote_output"],
                "raw_eval": str(evaluation_output / "raw_eval.json"),
                "evaluation_episodes_per_gamma": 160,
                "evaluation_selection": "preregistered full seed91000 bank; no cherry-picking",
            })
            failed.unlink(missing_ok=True)
        except BaseException as error:
            _atomic_json(failed, {
                "status": "FAILED_CLOSED", "name": spec.get("name"),
                "config_hash": spec.get("config_hash"),
                "physical_gpu": 3, "failed_unix": time.time(),
                "error": repr(error), "traceback": traceback.format_exc(),
                "remote_output": str(output),
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
