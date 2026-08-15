#!/usr/bin/env python3
"""Fail-closed exact-state PRE2 resume followed by native raw evaluation.

This worker is intentionally separate from ``run_pre2_spooled_arm.py``.  Its
input is an already committed resume boundary, not an empty fresh-PRE2 output.
For a saved-r1/5000 branch it first evaluates checkpoint 1, then resumes r2-r7.
For the live W300/2500 arm it requires the original r0-r2 native evaluation
before resuming the exact committed r2 state to r7.
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


def _arg_value(args: list[str], name: str) -> str:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing required expansion option {name}") from error


def _optional_arg_value(
    args: list[str], name: str, default: str | None = None,
) -> str | None:
    positions = [index for index, value in enumerate(args) if value == name]
    if not positions:
        return default
    if len(positions) != 1:
        raise ValueError(f"duplicate guarded expansion option {name}")
    try:
        return args[positions[0] + 1]
    except IndexError as error:
        raise ValueError(f"missing value for expansion option {name}") from error


def _ensure_saved_r1_resume_manifest(
    spec: dict[str, Any], output: Path,
) -> Path:
    """Materialize the semantic contract needed to resume saved r1.

    A saved-r1 branch is produced before the expansion driver completes a
    run, so it does not naturally have ``manifest.json``.  Build the guarded
    full-H/goal-box contract from the hashed CLI arguments.  The expansion
    driver archives this file and publishes its normal full manifest at r7.
    """
    manifest = output / "manifest.json"
    args = spec["expansion_args"]
    stopping = _optional_arg_value(
        args, "--verifier-taskspace-stopping-margin-m",
    )
    guarded_values = {
        name: _optional_arg_value(args, name)
        for name in (
            "--execution-goal-box-exp-weight",
            "--execution-goal-box-half-extent-m",
            "--execution-goal-box-exp-temperature-m",
            "--execution-goal-side-wall-quadratic-weight",
            "--execution-goal-side-wall-target-m",
        )
    }
    missing = [name for name, value in guarded_values.items() if value is None]
    if missing:
        raise ValueError(f"missing guarded expansion options: {missing}")
    contract = {
        "status": "SAVED_R1_RESUME_CONTRACT",
        "config_hash": spec["config_hash"],
        "optimizer_branch_provenance": "BRANCH_PROVENANCE.json",
        "lab_verifier": {
            "unexecuted_tail_taskspace_gate": (
                "--verifier-full-h-taskspace" in args
            ),
            "taskspace_stopping_backup": {
                "face_margin_m": (
                    None if stopping is None else float(stopping)
                ),
            },
        },
        "lab_execution_cost": {
            "execution_goal_box_exponential": {
                "weight": float(_arg_value(
                    args, "--execution-goal-box-exp-weight",
                )),
                "half_extent_m": float(_arg_value(
                    args, "--execution-goal-box-half-extent-m",
                )),
                "temperature_m": float(_arg_value(
                    args, "--execution-goal-box-exp-temperature-m",
                )),
            },
            "execution_goal_side_wall_quadratic": {
                "weight": float(_arg_value(
                    args, "--execution-goal-side-wall-quadratic-weight",
                )),
                "target_m": float(_arg_value(
                    args, "--execution-goal-side-wall-target-m",
                )),
            },
        },
    }
    archived = output / "manifest_before_resume_round_001.json"
    if archived.exists():
        raise RuntimeError(
            "saved-r1 resume already archived a manifest; preserve the "
            "interrupted continuation fail-closed"
        )
    if manifest.is_file():
        if _read_json(manifest) != contract:
            raise RuntimeError("saved-r1 resume manifest contract mismatch")
        return manifest
    _atomic_json(manifest, contract)
    return manifest


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported exact-resume spec schema")
    if spec.get("kind") not in {"saved_r1_optimizer_branch", "exact_r2_continuation"}:
        raise ValueError("unsupported exact-resume kind")
    payload = spec.get("hash_payload")
    if not isinstance(payload, dict):
        raise ValueError("hash_payload must be an object")
    if _canonical_sha256(payload) != spec.get("config_hash"):
        raise ValueError("exact-resume config hash mismatch")
    for key in (
        "kind", "source_id", "source_config_hash", "source_output",
        "physical_gpu", "rounds", "start_round",
        "expansion_args", "evaluation", "remote_output",
    ):
        if spec.get(key) != payload.get(key):
            raise ValueError(f"{key} differs from hashed payload")
    gpu = int(spec.get("physical_gpu", -1))
    if gpu not in {1, 3}:
        raise ValueError("only physical GPU1/GPU3 are permitted")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match physical_gpu")
    for key in (
        "remote_source", "remote_pretrain", "remote_task_config",
        "remote_output", "remote_control",
    ):
        if not isinstance(spec.get(key), str) or not spec[key].startswith("/"):
            raise ValueError(f"{key} must be an absolute remote path")
    args = spec.get("expansion_args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("expansion_args must be a string list")
    forbidden = {
        "--helios", "--helios-gpu", "--output", "--resume-from",
        "--pretrain-dir", "--lab-task-config", "--device", "--rounds",
    }
    overlap = forbidden.intersection(args)
    if overlap:
        raise ValueError(f"transport-owned CLI in expansion_args: {sorted(overlap)}")
    expected_steps = 5000 if spec["kind"] == "saved_r1_optimizer_branch" else 2500
    if int(_arg_value(args, "--optimizer-steps-per-round")) != expected_steps:
        raise ValueError("optimizer exposure does not match exact-resume kind")
    if int(spec["rounds"]) != 7:
        raise ValueError("this experiment must stop at r7")
    expected_start = 1 if spec["kind"] == "saved_r1_optimizer_branch" else 2
    if int(spec["start_round"]) != expected_start:
        raise ValueError("unexpected committed start boundary")
    evaluation = spec.get("evaluation")
    if evaluation != {
        "episodes_per_gamma": 40,
        "fixed_scene_rollouts": 10,
        "fixed_scene_seed": 191000,
        "flow_nfe": 12,
        "probe_samples": 16,
        "seed": 91000,
    }:
        raise ValueError("native fixed-bank evaluation contract changed")


def _native_eval_complete(path: Path, rounds: tuple[int, ...]) -> bool:
    raw = path / "raw_eval.json"
    if not raw.is_file() or raw.stat().st_size <= 0:
        return False
    payload = _read_json(raw)
    summary = payload.get("summary")
    return (
        payload.get("status") == "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
        and payload.get("sigma_tilt_used") is False
        and isinstance(summary, dict)
        and set(summary) == {str(value) for value in rounds}
    )


def _resume_boundary(output: Path, round_i: int, optimizer_step: int) -> None:
    required = (
        output / f"checkpoint_{round_i:03d}.pt",
        output / f"query_archive_round_{round_i:03d}.pt",
        output / f"events_round_{round_i:03d}.pt",
        output / "resume_state.json",
        output / "resume_state_latest.pt",
        output / "metrics.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError("committed resume boundary is incomplete: " + ", ".join(missing))
    metadata = _read_json(output / "resume_state.json")
    if (
        metadata.get("status") != "COMMITTED_ROUND_RESUME"
        or int(metadata.get("completed_round", -1)) != round_i
        or int(metadata.get("optimizer_step", -1)) != optimizer_step
    ):
        raise RuntimeError("resume metadata does not match the exact expected boundary")


def _run(command: list[str], cwd: Path) -> None:
    print("[exact-resume]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _evaluation_command(
    spec: dict[str, Any], *, rounds: tuple[int, ...], output: Path,
    manifest: Path | None = None,
) -> list[str]:
    config = spec["evaluation"]
    command = [
        sys.executable,
        str(Path(spec["remote_source"]) / "scripts/research_evaluate_ball_expansion.py"),
        "--pretrain-dir", spec["remote_pretrain"],
        "--expansion", spec["remote_output"],
        "--lab-task-config", spec["remote_task_config"],
        "--evaluation-output", str(output),
        "--device", "cuda:0",
        "--episodes", str(config["episodes_per_gamma"]),
        "--probe-samples", str(config["probe_samples"]),
        "--flow-nfe", str(config["flow_nfe"]),
        "--evaluation-rounds", *(str(value) for value in rounds),
        "--seed", str(config["seed"]),
        "--fixed-scene-rollouts", str(config["fixed_scene_rollouts"]),
        "--fixed-scene-seed", str(config["fixed_scene_seed"]),
        "--gallery-rounds", str(rounds[-1]),
        "--gallery-view", "head_on",
        "--save-raw-trajectories", "--screening-only",
    ]
    if manifest is not None:
        command.extend(["--expansion-manifest", str(manifest)])
    return command


def run(spec_path: Path) -> None:
    spec = _read_json(spec_path)
    _validate_spec(spec)
    output = Path(spec["remote_output"])
    control = Path(spec["remote_control"])
    key = f"{spec['name']}--{spec['config_hash'][:12]}"
    status = control / "status"
    lock_path = control / "locks" / f"{key}.lock"
    running = status / f"{key}.RUNNING.json"
    complete = status / f"{key}.COMPLETE.json"
    failed = status / f"{key}.FAILED.json"
    status.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"duplicate live exact-resume worker for {key}") from error
        if complete.is_file():
            final_eval = output / "fixed_eval_r000_r007_native_seed91000"
            if not _native_eval_complete(final_eval, tuple(range(8))):
                raise RuntimeError("COMPLETE marker exists without exact final evaluation")
            return
        started = time.time()
        _atomic_json(running, {
            "status": "RUNNING", "name": spec["name"],
            "config_hash": spec["config_hash"], "pid": os.getpid(),
            "physical_gpu": spec["physical_gpu"], "started_unix": started,
        })
        try:
            start_round = int(spec["start_round"])
            start_steps = start_round * int(_arg_value(
                spec["expansion_args"], "--optimizer-steps-per-round",
            ))
            _resume_boundary(output, start_round, start_steps)
            if spec["kind"] == "saved_r1_optimizer_branch":
                provenance = _read_json(output / "BRANCH_PROVENANCE.json")
                if not (
                    provenance.get("historical_2500_model_bitwise_reproduced") is True
                    and int(provenance.get("target_optimizer_steps", -1)) == 5000
                    and int(provenance.get("target_rounds", -1)) == 7
                    and provenance.get("source") == spec["source_output"]
                ):
                    raise RuntimeError("saved-r1 bootstrap provenance mismatch")
                initial_eval = output / "fixed_eval_r001_steps5000_native_seed91000"
                screening_manifest = output / "BRANCH_SCREENING_MANIFEST.json"
                if not screening_manifest.is_file():
                    raise FileNotFoundError(screening_manifest)
                if not _native_eval_complete(initial_eval, (1,)):
                    if initial_eval.exists() and any(initial_eval.iterdir()):
                        raise RuntimeError("partial r1 native evaluation preserved fail-closed")
                    _run(_evaluation_command(
                        spec, rounds=(1,), output=initial_eval,
                        manifest=screening_manifest,
                    ), Path(spec["remote_source"]))
                if not _native_eval_complete(initial_eval, (1,)):
                    raise RuntimeError("r1 native evaluation did not complete")
                _ensure_saved_r1_resume_manifest(spec, output)
            else:
                source_eval = output / "fixed_eval_r000_r002"
                if not _native_eval_complete(source_eval, (0, 1, 2)):
                    raise RuntimeError("source W300/2500 r0-r2 native evaluation is incomplete")

            expansion_command = [
                sys.executable,
                str(Path(spec["remote_source"]) / "scripts/research_ball_expansion_optimization.py"),
                "--pretrain-dir", spec["remote_pretrain"],
                "--lab-task-config", spec["remote_task_config"],
                "--output", str(output), "--resume-from", str(output),
                "--device", "cuda:0", "--rounds", str(spec["rounds"]),
                *spec["expansion_args"],
            ]
            _run(expansion_command, Path(spec["remote_source"]))
            _resume_boundary(
                output, 7,
                7 * int(_arg_value(spec["expansion_args"], "--optimizer-steps-per-round")),
            )
            manifest = output / "manifest.json"
            if not manifest.is_file():
                raise RuntimeError("r7 expansion returned without manifest")
            final_eval = output / "fixed_eval_r000_r007_native_seed91000"
            if not _native_eval_complete(final_eval, tuple(range(8))):
                if final_eval.exists() and any(final_eval.iterdir()):
                    raise RuntimeError("partial r0-r7 native evaluation preserved fail-closed")
                _run(_evaluation_command(
                    spec, rounds=tuple(range(8)), output=final_eval,
                    manifest=manifest,
                ), Path(spec["remote_source"]))
            if not _native_eval_complete(final_eval, tuple(range(8))):
                raise RuntimeError("r0-r7 native evaluation did not complete")
            finished = time.time()
            _atomic_json(complete, {
                "status": "COMPLETE", "name": spec["name"],
                "config_hash": spec["config_hash"], "rounds": 7,
                "physical_gpu": spec["physical_gpu"],
                "started_unix": started, "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "remote_output": str(output),
                "raw_eval": str(final_eval / "raw_eval.json"),
                "initial_r1_raw_eval": (
                    str(output / "fixed_eval_r001_steps5000_native_seed91000/raw_eval.json")
                    if spec["kind"] == "saved_r1_optimizer_branch" else None
                ),
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
