#!/usr/bin/env python3
"""Run the two preregistered faithful-M1 speed400 experiments on GPU1.

Experiment A clones the authoritative dense-z speed400 R5 transaction,
revalidates its faithful M1 curve, resumes the exact optimizer/RNG state to
R10, and evaluates R0..R10.  Experiment B packages that exact R5 model as a
new initialization and performs a conservative cumulative-replay stability
continuation on fresh paired scenes before another faithful-M1 curve.

This worker deliberately never runs an M8 deployment evaluation.
"""
from __future__ import annotations

import argparse
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

import torch


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path) -> None:
    print("[speed400-faithful]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _replace_value(arguments: list[str], option: str, value: str) -> list[str]:
    updated = list(arguments)
    try:
        index = updated.index(option)
    except ValueError as error:
        raise RuntimeError(f"missing required option {option}") from error
    if index + 1 >= len(updated):
        raise RuntimeError(f"missing value after {option}")
    updated[index + 1] = value
    return updated


def _append_flag_value(
    arguments: list[str], option: str, value: str,
) -> list[str]:
    if option in arguments:
        return _replace_value(arguments, option, value)
    return [*arguments, option, value]


def _completed_round(output: Path) -> int:
    path = output / "resume_state.json"
    if not path.is_file():
        return -1
    value = _read(path)
    if value.get("status") != "COMMITTED_ROUND_RESUME":
        return -1
    return int(value.get("completed_round", -1))


def _prepare_exact_resume_manifest(output: Path) -> Path | None:
    """Route a compact core-only manifest through deterministic reconstruction."""
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return None
    payload = _read(manifest)
    if isinstance(payload.get("lab_scene_ledger"), list):
        return None
    frozen = output / "r5_frozen_provenance"
    frozen.mkdir(exist_ok=True)
    backup = frozen / "manifest_core_only_before_extension.json"
    if backup.exists():
        if backup.read_bytes() != manifest.read_bytes():
            raise RuntimeError("existing compact-manifest backup disagrees with root")
        manifest.unlink()
        return backup
    manifest.replace(backup)
    return backup


def _evaluation_complete(output: Path, rounds: int) -> bool:
    path = output / "raw_eval.json"
    if not path.is_file():
        return False
    value = _read(path)
    return (
        value.get("status")
        == "RAW_COST_RANKED_BEST_OF_M_DEPLOYMENT_COMPLETE"
        and int(value.get("samples_per_step", -1)) == 1
        and all(str(index) in value.get("summary", {}) for index in range(rounds + 1))
    )


def _evaluate(
    spec: dict[str, Any], expansion: Path, pretrain: Path,
    rounds: int, label: str,
) -> Path:
    evaluation = expansion / label
    if _evaluation_complete(evaluation, rounds):
        return evaluation
    if evaluation.exists() and any(evaluation.iterdir()):
        raise RuntimeError(f"preserving partial evaluation fail-closed: {evaluation}")
    source = Path(spec["scientific_source"])
    _run([
        sys.executable,
        str(source / "scripts/evaluate_multisphere_min_cost_deployment.py"),
        "--pretrain-dir", str(pretrain),
        "--expansion", str(expansion),
        "--checkpoint-rounds", f"0-{rounds}",
        "--scene-bank-json", spec["scene_bank"],
        "--evaluation-output", str(evaluation),
        "--device", "cuda:0",
        "--episodes", "50",
        "--samples-per-step", "1",
        "--sampling-temperature", "1.0",
        "--execution-clearance-exp-weight", "15",
        "--execution-clearance-target-m", "0.6",
        "--execution-clearance-exp-temperature", "0.15",
        "--execution-taskspace-quadratic-weight", "250",
        "--execution-taskspace-quadratic-target-m", "0.15",
        "--execution-axis-cylinder-quadratic-weight", "5",
        "--execution-axis-cylinder-radius-m", "1.1",
        "--execution-control-weight", "0.05",
        "--execution-obstacle-speed-weight", "400",
        "--execution-cost-band-fraction", "0.05",
        "--seed", "91000",
    ], source)
    if not _evaluation_complete(evaluation, rounds):
        raise RuntimeError(f"faithful evaluation incomplete: {evaluation}")
    return evaluation


def _scientific_summary(raw_eval: Path) -> dict[str, Any]:
    value = _read(raw_eval)
    return {
        "samples_per_step": value["samples_per_step"],
        "sampling_temperature": value["sampling_temperature"],
        "summary": value["summary"],
    }


def _revalidation_equivalence(
    reference_path: Path, reproduced_path: Path,
) -> dict[str, Any]:
    """Check identical inputs and a bounded fixed-bank stochastic envelope.

    Closed-loop M1 outcomes are not bitwise reproducible across physical CUDA
    devices even with identical checkpoints and rollout seeds: tiny flow ODE
    floating-point differences can switch a terminal outcome.  The immutable
    checkpoint/scene/seed identities must still match exactly; pooled metrics
    are admitted only inside a preregistered, per-round statistical envelope.
    """
    reference = _read(reference_path)
    reproduced = _read(reproduced_path)
    reference_rows = reference.get("rows", {})
    reproduced_rows = reproduced.get("rows", {})
    identity_fields = (
        "round", "gamma", "episode", "rollout_seed", "scene_seed", "scene_hash",
    )
    reference_identities = {
        str(round_index): [
            tuple(row[field] for field in identity_fields)
            for row in reference_rows[str(round_index)]
        ]
        for round_index in range(6)
    }
    reproduced_identities = {
        str(round_index): [
            tuple(row[field] for field in identity_fields)
            for row in reproduced_rows[str(round_index)]
        ]
        for round_index in range(6)
    }
    checkpoint_binding_match = (
        reference["artifact_binding"]["checkpoint_sha256_by_round"]
        == reproduced["artifact_binding"]["checkpoint_sha256_by_round"]
    )
    scene_seed_identity_match = reference_identities == reproduced_identities
    thresholds = {
        "SR": 0.07,
        "CR": 0.07,
        "OOB": 0.07,
        "timeout": 0.07,
        "window_validity": 0.01,
        "successful_min_clearance_m": 0.015,
        "successful_time_to_goal_s": 1.0,
    }
    differences: dict[str, dict[str, float]] = {}
    within_envelope = True
    for round_index in range(6):
        reference_pooled = reference["summary"][str(round_index)]["pooled"]
        reproduced_pooled = reproduced["summary"][str(round_index)]["pooled"]
        differences[str(round_index)] = {}
        for metric, threshold in thresholds.items():
            difference = abs(
                float(reference_pooled[metric]) - float(reproduced_pooled[metric])
            )
            differences[str(round_index)][metric] = difference
            within_envelope &= difference <= threshold
    return {
        "checkpoint_binding_exact_match": checkpoint_binding_match,
        "scene_and_seed_identities_exact_match": scene_seed_identity_match,
        "pooled_metric_thresholds": thresholds,
        "absolute_pooled_metric_differences": differences,
        "pooled_metrics_within_preregistered_envelope": bool(within_envelope),
        "accepted": bool(
            checkpoint_binding_match and scene_seed_identity_match and within_envelope
        ),
    }


def _prepare_r5_pretrain(
    source_pretrain: Path, checkpoint: Path, output: Path,
) -> None:
    if (output / "pretrained.pt").is_file():
        return
    output.mkdir(parents=True, exist_ok=False)
    pretrained = torch.load(
        source_pretrain / "pretrained.pt", map_location="cpu", weights_only=False,
    )
    r5 = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(pretrained, dict) or not {"arch", "contract", "model"} <= set(pretrained):
        raise ValueError("unsupported source pretrained payload")
    if not isinstance(r5, dict) or int(r5.get("round", -1)) != 5 or "model" not in r5:
        raise ValueError("source checkpoint is not a round-5 model")
    temporary = output / ".pretrained.pt.tmp"
    torch.save({
        "model": r5["model"],
        "arch": pretrained["arch"],
        "contract": pretrained["contract"],
    }, temporary)
    temporary.replace(output / "pretrained.pt")
    shutil.copy2(
        source_pretrain / "pretrain_manifest.json",
        output / "pretrain_manifest.json",
    )
    _write(output / "R5_PARENT.json", {
        "status": "R5_MODEL_PACKAGED_AS_FRESH_INITIALIZATION",
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": _sha256(checkpoint),
        "optimizer_and_rng_inherited": False,
        "model_weights_inherited": True,
    })


def _stability_arguments(original: list[str]) -> list[str]:
    values = list(original)
    for option, value in (
        ("--replay-scope", "cumulative"),
        ("--replay-rounds", "5"),
        ("--replay-passes-per-round", "2"),
        ("--inner-steps", "10"),
        ("--learning-rate", "5e-6"),
        ("--max-retry-batches", "2"),
        ("--seed", "81625"),
    ):
        values = _replace_value(values, option, value)
    values = _append_flag_value(values, "--first-layer-lr-scale", "0.25")
    values = _append_flag_value(
        values, "--paired-scene-replace-after-retry-batches", "1",
    )
    values = _append_flag_value(
        values, "--paired-scene-max-replacements-per-slot", "1",
    )
    return values


def _pooled(raw_eval: Path, round_index: int) -> dict[str, Any]:
    return _read(raw_eval)["summary"][str(round_index)]["pooled"]


def run(spec_path: Path) -> None:
    spec = _read(spec_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("this worker is restricted to physical GPU1")
    control = Path(spec["control"])
    running = control / "RUNNING.json"
    complete = control / "COMPLETE.json"
    failed = control / "FAILED.json"
    started = time.time()
    _write(running, {"status": "RUNNING", "pid": os.getpid(), "started_unix": started})
    try:
        source_r5 = Path(spec["source_r5"])
        extension = Path(spec["extension_output"])
        stability = Path(spec["stability_output"])
        source_pretrain = Path(spec["source_pretrain"])
        r5_pretrain = Path(spec["r5_pretrain"])
        source_checkpoint = source_r5 / "checkpoint_005.pt"
        if _completed_round(source_r5) != 5 or _completed_round(extension) != 5:
            raise RuntimeError("authoritative and cloned extension inputs must both be R5")
        if _sha256(source_checkpoint) != _sha256(extension / "checkpoint_005.pt"):
            raise RuntimeError("cloned R5 checkpoint is not bit-identical")

        revalidation = _evaluate(
            spec, extension, source_pretrain, 5,
            "eval_m50_faithful_m1_r5_revalidation",
        )
        reference = Path(spec["reference_faithful_r5"])
        reproduced = revalidation / "raw_eval.json"
        exact_summary_match = _scientific_summary(reference) == _scientific_summary(reproduced)
        equivalence = _revalidation_equivalence(reference, reproduced)
        _write(control / "R5_REVALIDATION.json", {
            "status": (
                "BIT_IDENTICAL_SCIENTIFIC_SUMMARY"
                if exact_summary_match else
                "IDENTICAL_INPUTS_STATISTICALLY_EQUIVALENT_CUDA_OUTCOMES"
                if equivalence["accepted"] else
                "MISMATCH"
            ),
            "checkpoint_sha256": _sha256(source_checkpoint),
            "reference_raw_eval_sha256": _sha256(reference),
            "reproduced_raw_eval_sha256": _sha256(reproduced),
            "scientific_summary_exact_match": exact_summary_match,
            "equivalence": equivalence,
        })
        if not exact_summary_match and not equivalence["accepted"]:
            raise RuntimeError("faithful R5 revalidation disagrees with the reference")

        if _completed_round(extension) < 10:
            source = Path(spec["resume_source"])
            moved_manifest = _prepare_exact_resume_manifest(extension)
            try:
                _run([
                    sys.executable,
                    str(source / "scripts/research_multisphere_expansion_multipair.py"),
                    "--pretrain-dir", str(source_pretrain),
                    "--lab-task-config", spec["task_config"],
                    "--output", str(extension),
                    "--resume-from", str(extension),
                    "--device", "cuda:0",
                    "--rounds", "10",
                    *spec["original_expansion_args"],
                ], source)
            except BaseException:
                if moved_manifest is not None and not (extension / "manifest.json").exists():
                    shutil.copy2(moved_manifest, extension / "manifest.json")
                raise
        if _completed_round(extension) != 10:
            raise RuntimeError("exact extension returned without committed R10")
        extension_eval = _evaluate(
            spec, extension, source_pretrain, 10,
            "eval_m50_faithful_m1_r0_r10",
        )

        _prepare_r5_pretrain(source_pretrain, source_checkpoint, r5_pretrain)
        if not stability.exists():
            source = Path(spec["scientific_source"])
            _run([
                sys.executable,
                str(source / "scripts/research_multisphere_expansion_multipair.py"),
                "--pretrain-dir", str(r5_pretrain),
                "--lab-task-config", spec["task_config"],
                "--output", str(stability),
                "--device", "cuda:0",
                "--rounds", "5",
                *_stability_arguments(spec["original_expansion_args"]),
            ], source)
        if _completed_round(stability) != 5:
            raise RuntimeError("stability branch returned without committed stage 5")
        stability_eval = _evaluate(
            spec, stability, r5_pretrain, 5,
            "eval_m50_faithful_m1_r5base_to_s5",
        )

        finished = time.time()
        _write(complete, {
            "status": "COMPLETE",
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "r5_revalidation": str(control / "R5_REVALIDATION.json"),
            "exact_extension": {
                "output": str(extension),
                "raw_eval": str(extension_eval / "raw_eval.json"),
                "r10": _pooled(extension_eval / "raw_eval.json", 10),
            },
            "stability_branch": {
                "output": str(stability),
                "raw_eval": str(stability_eval / "raw_eval.json"),
                "stage0_r5_parent": _pooled(stability_eval / "raw_eval.json", 0),
                "stage5": _pooled(stability_eval / "raw_eval.json", 5),
            },
            "evaluation_contract": {
                "NFE": 16,
                "raw_plans_per_step": 1,
                "M8_used": False,
                "episodes_per_gamma": 50,
                "attempts_per_checkpoint": 200,
                "fixed_seed": 91000,
            },
        })
        failed.unlink(missing_ok=True)
    except BaseException as error:
        _write(failed, {
            "status": "FAILED_CLOSED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "failed_unix": time.time(),
        })
        raise
    finally:
        running.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(args.spec)


if __name__ == "__main__":
    main()
