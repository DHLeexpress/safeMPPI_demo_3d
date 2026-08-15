#!/usr/bin/env python3
"""Run one already-staged PRE2 expansion arm and its fixed-bank evaluation.

This is a remote worker, not a Helios transport wrapper.  Its spec contains
fully resolved remote paths and the exact scientific CLI.  It deliberately
does not accept arbitrary command-line overrides.
"""
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported spool-arm schema")
    payload = spec.get("hash_payload")
    if not isinstance(payload, dict):
        raise ValueError("spec.hash_payload must be an object")
    if _canonical_sha256(payload) != spec.get("config_hash"):
        raise ValueError("spool-arm config hash mismatch")
    if int(spec.get("physical_gpu", -1)) not in {1, 3}:
        raise ValueError("only physical GPU1/GPU3 are permitted")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["physical_gpu"]):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not match the assigned physical GPU"
        )
    if int(spec.get("rounds", -1)) < 0:
        raise ValueError("rounds must be nonnegative")
    if int(spec["rounds"]) != int(payload.get("rounds", -1)):
        raise ValueError("rounds differ from hashed payload")
    for key in (
        "remote_source", "remote_pretrain", "remote_task_config",
        "remote_output", "remote_control",
    ):
        if not isinstance(spec.get(key), str) or not spec[key].startswith("/"):
            raise ValueError(f"{key} must be an absolute remote path")
    args = spec.get("expansion_args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("expansion_args must be a string list")
    if args != payload.get("expansion_args"):
        raise ValueError("expansion_args differ from hashed payload")
    if spec.get("evaluation") != payload.get("evaluation"):
        raise ValueError("evaluation differs from hashed payload")
    for key in ("source_id", "pretrained_sha256", "task_config_sha256"):
        if spec.get(key) != payload.get(key):
            raise ValueError(f"{key} differs from hashed payload")
    forbidden = {
        "--helios", "--helios-gpu", "--output", "--resume-from",
        "--pretrain-dir", "--lab-task-config", "--device", "--rounds",
    }
    overlap = forbidden.intersection(args)
    if overlap:
        raise ValueError(f"transport-owned CLI present in expansion_args: {sorted(overlap)}")


def _expansion_complete(output: Path, rounds: int) -> bool:
    required = (
        output / f"checkpoint_{rounds:03d}.pt",
        output / "resume_state.json",
        output / "resume_state_latest.pt",
        output / "manifest.json",
        output / "metrics.jsonl",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    metadata = _read_json(output / "resume_state.json")
    return (
        metadata.get("status") == "COMMITTED_ROUND_RESUME"
        and int(metadata.get("completed_round", -1)) == rounds
    )


def _evaluation_complete(output: Path, rounds: int) -> bool:
    raw_eval = output / "raw_eval.json"
    if not raw_eval.is_file() or raw_eval.stat().st_size == 0:
        return False
    payload = _read_json(raw_eval)
    summary = payload.get("summary", {})
    return (
        payload.get("status") == "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
        and isinstance(summary, dict)
        and all(str(index) in summary for index in range(rounds + 1))
    )


def _run(command: list[str], cwd: Path) -> None:
    print("[spooled-arm]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run(spec_path: Path) -> None:
    spec = _read_json(spec_path)
    _validate_spec(spec)
    rounds = int(spec["rounds"])
    remote_source = Path(spec["remote_source"])
    output = Path(spec["remote_output"])
    evaluation = output / f"fixed_eval_r000_r{rounds:03d}"
    control = Path(spec["remote_control"])
    status_dir = control / "status"
    key = f"{spec['name']}--{spec['config_hash'][:12]}"
    lock_path = control / "locks" / f"{key}.lock"
    running = status_dir / f"{key}.RUNNING.json"
    complete = status_dir / f"{key}.COMPLETE.json"
    failed = status_dir / f"{key}.FAILED.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"duplicate live worker for {key}") from error

        if complete.is_file():
            if not (
                _expansion_complete(output, rounds)
                and _evaluation_complete(evaluation, rounds)
            ):
                raise RuntimeError("COMPLETE marker exists but artifacts are incomplete")
            print(f"[spooled-arm] already complete: {key}", flush=True)
            return

        started = time.time()
        _atomic_json(running, {
            "status": "RUNNING", "name": spec["name"],
            "config_hash": spec["config_hash"], "physical_gpu": spec["physical_gpu"],
            "pid": os.getpid(), "started_unix": started,
        })
        try:
            if not _expansion_complete(output, rounds):
                if output.exists() and any(output.iterdir()):
                    raise RuntimeError(
                        "partial/nonempty expansion output is preserved fail-closed; "
                        "use a distinct attempt or an explicitly prepared resume"
                    )
                expansion_command = [
                    sys.executable,
                    str(remote_source / "scripts/research_ball_expansion_optimization.py"),
                    "--pretrain-dir", spec["remote_pretrain"],
                    "--lab-task-config", spec["remote_task_config"],
                    "--output", str(output),
                    "--device", "cuda:0",
                    "--rounds", str(rounds),
                    *spec["expansion_args"],
                ]
                _run(expansion_command, remote_source)
            if not _expansion_complete(output, rounds):
                raise RuntimeError("expansion returned without a committed final round")

            _atomic_json(output / "SPOOL_PROVENANCE.json", {
                "name": spec["name"], "config_hash": spec["config_hash"],
                "physical_gpu": spec["physical_gpu"],
                "remote_spec": str(spec_path), "rounds": rounds,
                "source_id": spec["source_id"],
                "pretrained_sha256": spec["pretrained_sha256"],
                "task_config_sha256": spec["task_config_sha256"],
            })

            if not _evaluation_complete(evaluation, rounds):
                if evaluation.exists() and any(evaluation.iterdir()):
                    raise RuntimeError(
                        "partial/nonempty evaluation output is preserved fail-closed"
                    )
                evaluation_config = spec["evaluation"]
                evaluation_command = [
                    sys.executable,
                    str(remote_source / "scripts/research_evaluate_ball_expansion.py"),
                    "--pretrain-dir", spec["remote_pretrain"],
                    "--expansion", str(output),
                    "--lab-task-config", spec["remote_task_config"],
                    "--evaluation-output", str(evaluation),
                    "--device", "cuda:0",
                    "--episodes", str(evaluation_config["episodes_per_gamma"]),
                    "--probe-samples", str(evaluation_config["probe_samples"]),
                    "--flow-nfe", str(evaluation_config["flow_nfe"]),
                    "--evaluation-rounds",
                    *(str(index) for index in range(rounds + 1)),
                    "--seed", str(evaluation_config["seed"]),
                    "--fixed-scene-rollouts",
                    str(evaluation_config["fixed_scene_rollouts"]),
                    "--fixed-scene-seed", str(evaluation_config["fixed_scene_seed"]),
                    "--gallery-rounds", str(rounds),
                    "--gallery-view", "head_on",
                    "--save-raw-trajectories",
                    "--screening-only",
                ]
                _run(evaluation_command, remote_source)
            if not _evaluation_complete(evaluation, rounds):
                raise RuntimeError("evaluation returned without complete r0..rN summary")

            finished = time.time()
            _atomic_json(complete, {
                "status": "COMPLETE", "name": spec["name"],
                "config_hash": spec["config_hash"], "physical_gpu": spec["physical_gpu"],
                "rounds": rounds, "started_unix": started,
                "finished_unix": finished, "elapsed_seconds": finished - started,
                "remote_output": str(output), "raw_eval": str(evaluation / "raw_eval.json"),
            })
            failed.unlink(missing_ok=True)
        except BaseException as error:
            _atomic_json(failed, {
                "status": "FAILED_CLOSED", "name": spec.get("name"),
                "config_hash": spec.get("config_hash"),
                "physical_gpu": spec.get("physical_gpu"),
                "failed_unix": time.time(), "error": repr(error),
                "traceback": traceback.format_exc(),
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
