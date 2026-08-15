#!/usr/bin/env python3
"""CPU-only readiness gate for a live r1 source and its saved-r1 branch.

The waiter never exports a CUDA device and never runs model code.  After an
atomic, recipe-matched r1 boundary appears, it enqueues exactly one GPU
bootstrap and one dependent exact-resume worker on the recorded task-spooler
socket.  A nonblocking file lock and durable state prevent duplicate enqueue.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

import torch


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _arg_value(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported waiter schema")
    payload = spec.get("hash_payload")
    if not isinstance(payload, dict) or _canonical_sha256(payload) != spec.get("config_hash"):
        raise ValueError("waiter config hash mismatch")
    for key in (
        "source_output", "source_spec", "source_config_hash", "branch_output",
        "branch_spec", "remote_source", "remote_pretrain", "physical_gpu",
        "tsp_socket", "target_optimizer_steps", "target_rounds",
    ):
        if spec.get(key) != payload.get(key):
            raise ValueError(f"{key} differs from hashed payload")
    if int(spec["physical_gpu"]) not in {1, 3}:
        raise ValueError("only GPU1/GPU3 may receive dependent work")
    if int(spec["target_optimizer_steps"]) != 5000 or int(spec["target_rounds"]) != 7:
        raise ValueError("waiter is restricted to the approved 5000/r7 branch")


def _source_ready(spec: dict[str, Any]) -> bool:
    source = Path(spec["source_output"])
    required = (
        source / "checkpoint_000.pt", source / "checkpoint_001.pt",
        source / "query_archive_round_001.pt", source / "events_round_001.pt",
        source / "metrics.jsonl", source / "first_action_stats.json",
        source / "fa_alloc_log.json", source / "resume_state.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    source_spec = _read_json(Path(spec["source_spec"]))
    if source_spec.get("config_hash") != spec["source_config_hash"]:
        raise RuntimeError("live source spec hash changed")
    source_args = source_spec.get("expansion_args", [])
    if (
        int(_arg_value(source_args, "--optimizer-steps-per-round")) != 2500
        or abs(float(_arg_value(
            source_args, "--execution-obstacle-speed-weight",
        )) - 300.0) > 1e-12
    ):
        raise RuntimeError("live source spec is not the paired W300/2500 arm")
    metadata = _read_json(source / "resume_state.json")
    if (
        metadata.get("status") != "COMMITTED_ROUND_RESUME"
        or int(metadata.get("completed_round", -1)) < 1
        or int(metadata.get("optimizer_step", -1)) < 2500
    ):
        return False
    checkpoint = torch.load(
        source / "checkpoint_001.pt", map_location="cpu", weights_only=False,
    )
    config = checkpoint.get("config", {})
    if int(checkpoint.get("round", -1)) != 1:
        raise RuntimeError("published source checkpoint is not r1")
    if int(config.get("inner_steps", -1)) != 2500:
        raise RuntimeError("source r1 does not use 2500 optimizer steps")
    return True


def _enqueue(socket: str, label: str, shell: str, dependency: int | None = None) -> int:
    parts = ["tsp", "-B", "-L", label]
    if dependency is not None:
        parts.extend(["-D", str(dependency)])
    parts.extend(["bash", "-lc", shell])
    environment = dict(os.environ)
    environment["TS_SOCKET"] = socket
    output = subprocess.check_output(parts, env=environment, text=True).strip()
    if not output.isdigit():
        raise RuntimeError(f"unexpected task-spooler job ID: {output!r}")
    return int(output)


def wait_and_enqueue(spec_path: Path, *, poll_seconds: float, timeout_seconds: float) -> dict[str, Any]:
    spec = _read_json(spec_path)
    _validate_spec(spec)
    control = Path(spec["remote_control"])
    state_path = control / "W300_SAVED_R1_WAITER_STATE.json"
    lock_path = control / "locks/W300_SAVED_R1_WAITER.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("duplicate W300 saved-r1 waiter") from error
        if state_path.is_file():
            state = _read_json(state_path)
            if state.get("status") == "ENQUEUED":
                return state
            raise RuntimeError("stale/nonterminal waiter state preserved fail-closed")
        started = time.time()
        while not _source_ready(spec):
            if timeout_seconds > 0 and time.time() - started >= timeout_seconds:
                raise TimeoutError("timed out waiting for atomic W300 r1 boundary")
            time.sleep(poll_seconds)
        branch_output = Path(spec["branch_output"])
        if branch_output.exists():
            raise FileExistsError(f"branch output already exists: {branch_output}")
        log_dir = control / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_log = log_dir / "speed300_steps5000_r1_bootstrap.log"
        bootstrap = shlex.join([
            spec["remote_python"],
            f"{spec['remote_source']}/scripts/prepare_pre2_saved_r1_optimizer_branch.py",
            "--source", spec["source_output"],
            "--pretrain-dir", spec["remote_pretrain"],
            "--output", spec["branch_output"],
            "--device", "cuda:0", "--reference-optimizer-steps", "2500",
            "--target-optimizer-steps", "5000", "--target-rounds", "7",
            "--trainable-trunk-layers", "3",
        ])
        gpu = int(spec["physical_gpu"])
        bootstrap_shell = (
            "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
            f"export CUDA_VISIBLE_DEVICES={gpu}; "
            "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
            f"exec {bootstrap} >> {shlex.quote(str(bootstrap_log))} 2>&1"
        )
        bootstrap_id = _enqueue(
            spec["tsp_socket"], "saved-r1:speed300:bootstrap", bootstrap_shell,
        )
        worker_log = log_dir / "speed300_steps5000_r7.log"
        worker = shlex.join([
            spec["remote_python"],
            f"{spec['remote_source']}/scripts/run_pre2_exact_optimizer_resume.py",
            "--spec", spec["branch_spec"],
        ])
        worker_shell = (
            "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
            f"export CUDA_VISIBLE_DEVICES={gpu}; "
            "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
            f"exec {worker} >> {shlex.quote(str(worker_log))} 2>&1"
        )
        worker_id = _enqueue(
            spec["tsp_socket"], "saved-r1:speed300:r7", worker_shell,
            dependency=bootstrap_id,
        )
        state = {
            "status": "ENQUEUED", "config_hash": spec["config_hash"],
            "source_ready_unix": time.time(), "source_output": spec["source_output"],
            "source_config_hash": spec["source_config_hash"],
            "physical_gpu": gpu, "tsp_socket": spec["tsp_socket"],
            "bootstrap_job_id": bootstrap_id, "worker_job_id": worker_id,
            "branch_output": spec["branch_output"], "branch_spec": spec["branch_spec"],
        }
        _atomic_json(state_path, state)
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()
    result = wait_and_enqueue(
        args.spec.resolve(), poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
