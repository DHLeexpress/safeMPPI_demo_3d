#!/usr/bin/env python3
"""Run one staged multi-sphere hybrid expansion arm and synchronized M8 eval."""
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
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported arm spec")
    if int(spec.get("physical_gpu", -1)) not in {0, 1, 3}:
        raise ValueError("only physical GPU0/GPU1/GPU3 are allowed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["physical_gpu"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES disagrees with arm spec")
    if _canonical_sha256(spec["hash_payload"]) != spec.get("config_hash"):
        raise ValueError("arm config hash mismatch")
    for key in (
        "remote_source", "remote_pretrain", "remote_task_config",
        "remote_scene_bank", "remote_output", "remote_control",
    ):
        if not str(spec.get(key, "")).startswith("/"):
            raise ValueError(f"{key} must be an absolute path")


def _expansion_complete(output: Path, rounds: int) -> bool:
    checkpoint = output / f"checkpoint_{rounds:03d}.pt"
    resume = output / "resume_state.json"
    if not checkpoint.is_file() or not resume.is_file():
        return False
    value = _read(resume)
    return (
        value.get("status") == "COMMITTED_ROUND_RESUME"
        and int(value.get("completed_round", -1)) == rounds
    )


def _evaluation_complete(output: Path, rounds: int) -> bool:
    path = output / "raw_eval.json"
    if not path.is_file():
        return False
    value = _read(path)
    return (
        value.get("status") == "RAW_COST_RANKED_BEST_OF_M_DEPLOYMENT_COMPLETE"
        and all(str(index) in value.get("summary", {}) for index in range(rounds + 1))
    )


def _run(command: list[str], cwd: Path) -> None:
    print("[multisphere-spool]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run(spec_path: Path) -> None:
    spec = _read(spec_path)
    _validate(spec)
    source = Path(spec["remote_source"])
    output = Path(spec["remote_output"])
    rounds = int(spec["rounds"])
    evaluation = output / "eval_m50_m8_hybrid_r0_r5"
    control = Path(spec["remote_control"])
    key = f"{spec['name']}--{spec['config_hash'][:12]}"
    lock_path = control / "locks" / f"{key}.lock"
    running = control / "status" / f"{key}.RUNNING.json"
    complete = control / "status" / f"{key}.COMPLETE.json"
    failed = control / "status" / f"{key}.FAILED.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    running.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if complete.is_file():
            if not (
                _expansion_complete(output, rounds)
                and _evaluation_complete(evaluation, rounds)
            ):
                raise RuntimeError("COMPLETE marker has incomplete artifacts")
            return
        started = time.time()
        _write(running, {
            "status": "RUNNING",
            "name": spec["name"],
            "physical_gpu": spec["physical_gpu"],
            "started_unix": started,
            "pid": os.getpid(),
        })
        try:
            if not _expansion_complete(output, rounds):
                if output.exists() and any(output.iterdir()):
                    raise RuntimeError("preserving partial expansion fail-closed")
                _run([
                    sys.executable,
                    str(source / "scripts/research_multisphere_expansion_multipair.py"),
                    "--pretrain-dir", spec["remote_pretrain"],
                    "--lab-task-config", spec["remote_task_config"],
                    "--output", str(output),
                    "--device", "cuda:0",
                    "--rounds", str(rounds),
                    *spec["expansion_args"],
                ], source)
            if not _expansion_complete(output, rounds):
                raise RuntimeError("expansion returned without committed R5")
            _write(output / "SPOOL_PROVENANCE.json", {
                "name": spec["name"],
                "config_hash": spec["config_hash"],
                "physical_gpu": spec["physical_gpu"],
                "source_id": spec["source_id"],
                "task_config_sha256": spec["task_config_sha256"],
                "pretrained_sha256": spec["pretrained_sha256"],
            })
            if not _evaluation_complete(evaluation, rounds):
                if evaluation.exists() and any(evaluation.iterdir()):
                    raise RuntimeError("preserving partial evaluation fail-closed")
                _run([
                    sys.executable,
                    str(source / "scripts/evaluate_multisphere_min_cost_deployment.py"),
                    "--pretrain-dir", spec["remote_pretrain"],
                    "--expansion", str(output),
                    "--checkpoint-rounds", f"0-{rounds}",
                    "--scene-bank-json", spec["remote_scene_bank"],
                    "--evaluation-output", str(evaluation),
                    "--device", "cuda:0",
                    "--episodes", "50",
                    "--samples-per-step", "8",
                    "--sampling-temperature", "1.0",
                    "--execution-clearance-exp-weight", "15",
                    "--execution-clearance-target-m", "0.6",
                    "--execution-clearance-exp-temperature", "0.15",
                    "--execution-taskspace-quadratic-weight", "250",
                    "--execution-taskspace-quadratic-target-m", "0.15",
                    "--execution-axis-cylinder-quadratic-weight", "5",
                    "--execution-axis-cylinder-radius-m", "1.1",
                    "--execution-control-weight", "0.05",
                    "--execution-obstacle-speed-weight",
                    str(spec["obstacle_speed_weight"]),
                    "--execution-cost-band-fraction",
                    str(spec["cost_band_fraction"]),
                    "--seed", "91000",
                ], source)
            if not _evaluation_complete(evaluation, rounds):
                raise RuntimeError("evaluation returned without R0..R5")
            finished = time.time()
            _write(complete, {
                "status": "COMPLETE",
                "name": spec["name"],
                "config_hash": spec["config_hash"],
                "physical_gpu": spec["physical_gpu"],
                "started_unix": started,
                "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "remote_output": str(output),
                "raw_eval": str(evaluation / "raw_eval.json"),
            })
            failed.unlink(missing_ok=True)
        except BaseException as error:
            _write(failed, {
                "status": "FAILED_CLOSED",
                "name": spec.get("name"),
                "physical_gpu": spec.get("physical_gpu"),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_unix": time.time(),
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
