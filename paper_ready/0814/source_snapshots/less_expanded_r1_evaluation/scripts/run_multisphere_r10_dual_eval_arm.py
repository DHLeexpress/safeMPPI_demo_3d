#!/usr/bin/env python3
"""Resume one verified R5 arm to R10 and run faithful-M1 plus hack-M8 curves."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    if int(spec.get("physical_gpu", -1)) not in {1, 3}:
        raise ValueError("this continuation permits only physical GPU1/GPU3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["physical_gpu"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES disagrees with arm spec")
    if _canonical_sha256(spec["hash_payload"]) != spec.get("config_hash"):
        raise ValueError("arm config hash mismatch")
    for key in (
        "expansion_source", "evaluation_source", "remote_pretrain",
        "remote_task_config", "remote_scene_bank", "remote_output",
        "remote_control", "remote_plotter",
    ):
        if not str(spec.get(key, "")).startswith("/"):
            raise ValueError(f"{key} must be an absolute path")


def _completed_round(output: Path) -> int:
    metadata = output / "resume_state.json"
    if not metadata.is_file():
        return -1
    value = _read(metadata)
    if value.get("status") != "COMMITTED_ROUND_RESUME":
        return -1
    return int(value.get("completed_round", -1))


def _evaluation_complete(output: Path, rounds: int, samples: int) -> bool:
    path = output / "raw_eval.json"
    if not path.is_file():
        return False
    value = _read(path)
    return (
        value.get("status") == "RAW_COST_RANKED_BEST_OF_M_DEPLOYMENT_COMPLETE"
        and int(value.get("samples_per_step", -1)) == int(samples)
        and all(str(index) in value.get("summary", {}) for index in range(rounds + 1))
    )


def _run(command: list[str], cwd: Path) -> None:
    print("[multisphere-r10]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _freeze_source_provenance(output: Path, source_round: int) -> None:
    frozen = output / f"r{source_round}_frozen_provenance"
    frozen.mkdir(exist_ok=True)
    for name in (
        "manifest.json", "resume_state.json", "SPOOL_PROVENANCE.json",
        "paired_scene_replacements.json", "first_action_stats.json",
    ):
        source = output / name
        target = frozen / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)


def _prepare_exact_resume_manifest(
    output: Path, source_round: int,
) -> Path | None:
    """Route a core-only manifest through the verified reconstruction path."""
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return None
    payload = _read(manifest)
    if isinstance(payload.get("lab_scene_ledger"), list):
        return None
    frozen = output / f"r{source_round}_frozen_provenance"
    frozen.mkdir(exist_ok=True)
    backup = frozen / "manifest_core_only_before_r10.json"
    if backup.exists():
        if backup.read_bytes() != manifest.read_bytes():
            raise RuntimeError(f"resume-manifest backup disagrees with root: {backup}")
        manifest.unlink()
        return backup
    # The expansion entry point has a strict deterministic reconstruction path
    # for a missing final manifest.  Preserve this core-only manifest first,
    # then let that path rebuild every scene and verify its hash against the
    # committed event logs before accepting R6.
    manifest.replace(backup)
    return backup


def _evaluate(spec: dict[str, Any], samples: int, label: str) -> Path:
    output = Path(spec["remote_output"])
    rounds = int(spec["target_rounds"])
    evaluation = output / f"eval_m50_{label}_r0_r{rounds}"
    if _evaluation_complete(evaluation, rounds, samples):
        return evaluation
    if evaluation.exists() and any(evaluation.iterdir()):
        raise RuntimeError(f"preserving partial evaluation fail-closed: {evaluation}")
    source = Path(spec["evaluation_source"])
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
        "--samples-per-step", str(samples),
        "--sampling-temperature", "1.0",
        "--execution-clearance-exp-weight", "15",
        "--execution-clearance-target-m", "0.6",
        "--execution-clearance-exp-temperature", "0.15",
        "--execution-taskspace-quadratic-weight", "250",
        "--execution-taskspace-quadratic-target-m", "0.15",
        "--execution-axis-cylinder-quadratic-weight", "5",
        "--execution-axis-cylinder-radius-m", "1.1",
        "--execution-control-weight", "0.05",
        "--execution-obstacle-speed-weight", str(spec["obstacle_speed_weight"]),
        "--execution-cost-band-fraction", "0.05",
        "--seed", "91000",
    ], source)
    if not _evaluation_complete(evaluation, rounds, samples):
        raise RuntimeError(f"evaluation returned without R0..R{rounds}: {label}")

    jsonl = evaluation / f"random_domain_m50_{label}_r0_r{rounds}.jsonl"
    provenance = jsonl.with_suffix(".provenance.json")
    _run([
        sys.executable,
        str(source / "scripts/convert_multisphere_raw_curve.py"),
        "--raw-eval", str(evaluation / "raw_eval.json"),
        "--output", str(jsonl),
        "--provenance-output", str(provenance),
        "--expected-rounds", f"0-{rounds}",
        "--expected-m", "50",
        "--expected-temperature", "1.0",
        "--task-config", str(output / "task_config_resolved.json"),
        "--expansion-manifest", str(output / "manifest.json"),
    ], source)
    paper = evaluation / "paper"
    arm_label = "Faithful raw M1" if samples == 1 else "Hack raw M8"
    _run([
        sys.executable, spec["remote_plotter"],
        "--arm", f"{arm_label}={jsonl}",
        "--panel-heading", "D. 3D multi spheres — random-domain M=50",
        "--outdir", str(paper),
        "--stem", f"{spec['scene_id']}_{spec['variant_id']}_{label}_r0_r{rounds}",
    ], source)
    return evaluation


def run(spec_path: Path) -> None:
    spec = _read(spec_path)
    _validate(spec)
    output = Path(spec["remote_output"])
    target_rounds = int(spec["target_rounds"])
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
        started = time.time()
        _write(running, {
            "status": "RUNNING", "name": spec["name"],
            "physical_gpu": spec["physical_gpu"], "started_unix": started,
            "pid": os.getpid(),
        })
        try:
            completed = _completed_round(output)
            if completed < int(spec["source_round"]):
                raise RuntimeError(
                    f"expected committed R{spec['source_round']}, found R{completed}"
                )
            if completed < target_rounds:
                # A previous continuation may have committed one or more rounds
                # before a later sampling deadlock.  Freeze and reconstruct from
                # that latest committed boundary, not only from the arm's first
                # source round.
                _freeze_source_provenance(output, completed)
                moved_manifest = _prepare_exact_resume_manifest(
                    output, completed,
                )
                expansion_source = Path(spec["expansion_source"])
                try:
                    _run([
                        sys.executable,
                        str(expansion_source / "scripts/research_multisphere_expansion_multipair.py"),
                        "--pretrain-dir", spec["remote_pretrain"],
                        "--lab-task-config", spec["remote_task_config"],
                        "--output", str(output),
                        "--resume-from", str(output),
                        "--device", "cuda:0",
                        "--rounds", str(target_rounds),
                        *spec["expansion_args"],
                    ], expansion_source)
                except BaseException:
                    if moved_manifest is not None and not (output / "manifest.json").exists():
                        shutil.copy2(moved_manifest, output / "manifest.json")
                    raise
            if _completed_round(output) != target_rounds:
                raise RuntimeError(f"expansion returned without committed R{target_rounds}")

            faithful = _evaluate(spec, 1, "faithful_m1")
            hack = _evaluate(spec, 8, "hack_m8")
            comparison = output / f"eval_comparison_faithful_vs_hack_r0_r{target_rounds}"
            faithful_jsonl = faithful / f"random_domain_m50_faithful_m1_r0_r{target_rounds}.jsonl"
            hack_jsonl = hack / f"random_domain_m50_hack_m8_r0_r{target_rounds}.jsonl"
            source = Path(spec["evaluation_source"])
            _run([
                sys.executable, spec["remote_plotter"],
                "--arm", f"Faithful raw M1={faithful_jsonl}",
                "--arm", f"Hack raw M8={hack_jsonl}",
                "--panel-heading", "D. 3D multi spheres — random-domain M=50",
                "--outdir", str(comparison),
                "--stem", f"{spec['scene_id']}_{spec['variant_id']}_faithful_vs_hack_r0_r{target_rounds}",
            ], source)
            _write(output / "R10_DUAL_EVAL_PROVENANCE.json", {
                "status": "R10_DUAL_EVAL_COMPLETE",
                "config_hash": spec["config_hash"],
                "source_round": spec["source_round"],
                "target_rounds": target_rounds,
                "faithful": {
                    "NFE": 16, "raw_plans_per_step": 1,
                    "verifier_or_progress_used_for_selection": False,
                    "raw_eval": str(faithful / "raw_eval.json"),
                },
                "hack": {
                    "NFE": 16, "raw_plans_per_step": 8,
                    "verifier_or_progress_used_for_selection": False,
                    "selection": "5% cost-band then nominal first-step margin",
                    "raw_eval": str(hack / "raw_eval.json"),
                },
                "common": {
                    "episodes_per_gamma": 50, "attempts_per_checkpoint": 200,
                    "fixed_seed_bank": str(spec["remote_scene_bank"]),
                    "seed": 91000,
                },
            })
            finished = time.time()
            _write(complete, {
                "status": "COMPLETE", "name": spec["name"],
                "config_hash": spec["config_hash"],
                "physical_gpu": spec["physical_gpu"],
                "started_unix": started, "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "remote_output": str(output),
                "faithful_raw_eval": str(faithful / "raw_eval.json"),
                "hack_raw_eval": str(hack / "raw_eval.json"),
            })
            failed.unlink(missing_ok=True)
        except BaseException as error:
            _write(failed, {
                "status": "FAILED_CLOSED", "name": spec.get("name"),
                "physical_gpu": spec.get("physical_gpu"), "error": repr(error),
                "traceback": traceback.format_exc(), "failed_unix": time.time(),
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
