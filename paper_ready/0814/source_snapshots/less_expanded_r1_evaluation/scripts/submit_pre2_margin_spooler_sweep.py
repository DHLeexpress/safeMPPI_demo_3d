#!/usr/bin/env python3
"""Validate, stage, or enqueue the 0812 PRE2 margin-hybrid sweep.

The default action is local validation only.  GPU work requires either
``--enqueue`` or ``--smoke`` plus its exact confirmation phrase.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))

from safe_mppi.helios_remote import (  # noqa: E402
    HELIOS_HOST,
    REMOTE_ARTIFACT_BASE,
    REMOTE_PYTHON,
    REMOTE_SOURCE_BASE,
    _rsync,
    _sha256,
    _source_id,
    _source_stage_lock,
    _ssh_master,
    _stage_pretrain,
    _stage_source,
)


STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_margin_hybrid_spooler_sweep"
)
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
TASK_CONFIG = ROOT / (
    "configs/lab_ball_stage1_goalspace_yminus04_z01_17_"
    "r15in_reach03_v1.json"
)
CONFIRM_ENQUEUE = "I_UNDERSTAND_THIS_ENQUEUES_THE_64_ARM_SWEEP"
CONFIRM_SMOKE = "I_UNDERSTAND_THIS_RUNS_ONE_GPU_SMOKE_JOB"
ALLOWED_DELTAS = {
    "execution_rule",
    "execution_clearance_exp_aggregation",
    "execution_clearance_exp_weight",
    "execution_clearance_exp_temperature",
    "execution_clearance_exp_target",
    "execution_step_margin_mode",
    "execution_step_margin_weight",
    "execution_cost_band_fraction",
    "execution_cost_band_margin",
}
EXPECTED_COMMON = {
    "pretrain_dir": str(PRETRAIN.relative_to(ROOT)),
    "lab_task_config": str(TASK_CONFIG.relative_to(ROOT)),
    "fresh_pre2": True, "target_round": 8, "flow_nfe": 12,
    "batched_rollout_sampling": True, "fa_alloc": "none",
    "flow_base_std": 1.0, "flow_base_std_schedule": "none",
    "flow_base_std_final": 1.0, "candidate_perturb_std": 0.0,
    "candidate_perturb_scope": "coherent_horizon", "learning_rate": 5e-5,
    "learning_rate_final": 5e-5, "learning_rate_decay_steps": 1,
    "gradient_clip_norm": 5.0, "round_learning_rate_warmup_power": 0.0,
    "trainable_trunk_layers": 3,
    "freeze_visual_encoder_during_expansion": True, "beta": 0.1,
    "adaptive_beta": False, "parallel_episodes": 16, "verifier_workers": 8,
    "max_retry_batches": 32, "retry_exhaustion_policy": "resample_scene",
    "retry_resample_batch_cap": 384, "sample_update_submodes": "none",
    "sample_update_cohorts": "unguided_only", "K": 16, "B": 8,
    "retry_B": 8, "retry_verify_all_fast_path": False, "inner_steps": 1,
    "optimizer_steps_total": None, "batch_size": 256,
    "replay_scope": "cumulative",
    "replay_batch_sampler": "mode_gamma_stratified",
    "replay_top_fraction": 1.0, "replay_selector": "uniform",
    "replay_rounds": 100, "gp_buffer_cap": 1536,
    "gp_reference_mode": "sliding_success_per_gamma_current_phi",
    "gp_sliding_row_selector": "trajectory_uniform", "negative_alpha": 0.0,
    "archive_rule": "successful_executed_windows",
    "successful_trajectory_selector": "random_success",
    "replay_acceptance": "execution_eligible",
    "execution_rule": "exponential_cost", "execution_step_margin_weight": 0.0,
    "execution_step_margin_mode": "linear",
    "execution_clearance_exp_weight": 15.0,
    "execution_clearance_exp_temperature": 0.15,
    "execution_clearance_exp_aggregation": "mean",
    "execution_clearance_target_m": 0.6,
    "execution_taskspace_quadratic_weight": 0.0,
    "execution_taskspace_quadratic_target_m": 0.15,
    "execution_goal_side_wall_quadratic_weight": 0.0,
    "execution_goal_side_wall_target_m": 0.6,
    "execution_goal_box_exp_weight": 100.0,
    "execution_goal_box_half_extent_m": 0.2,
    "execution_goal_box_exp_temperature_m": 1.5,
    "execution_axis_cylinder_quadratic_weight": 5.0,
    "execution_axis_cylinder_radius_m": 1.1,
    "execution_axis_cylinder_finite_segment": True,
    "execution_control_weight": 1.0, "execution_terminal_goal_weight": 80.0,
    "execution_goal_braking_weight": 0.0,
    "execution_goal_braking_distance_m": 0.6,
    "execution_goal_braking_temperature_m": 0.15,
    "acquisition_feature": "learned_phi", "coverage_replay": "none",
    "replay_augmentation": "none", "execution_z_bias_mode": "none",
    "tight_corridor": True, "verifier_mode": "full_polytope",
    "verifier_solver": "analytic", "verifier_full_h_taskspace": True,
    "event_log": "committed_success", "paired_noised_representation": True,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ARM_MATRIX schema")
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("ARM_MATRIX.arms must be a nonempty list")
    common = payload.get("common_parameters")
    if common is not None:
        for key, expected in EXPECTED_COMMON.items():
            if common.get(key) != expected:
                raise ValueError(
                    f"ARM_MATRIX common parameter mismatch for {key}: "
                    f"{common.get(key)!r} != {expected!r}"
                )
    names: set[str] = set()
    indices: set[int] = set()
    for arm in arms:
        name = arm.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"unsafe arm name: {name!r}")
        index = int(arm.get("index", -1))
        if name in names or index in indices:
            raise ValueError(f"duplicate arm name/index: {name}/{index}")
        names.add(name)
        indices.add(index)
        if int(arm.get("target_round", -1)) != 8:
            raise ValueError(f"{name}: target_round must equal 8")
        quota = arm.get("quota", {})
        count = int(quota.get("successful_trajectories_per_gamma", -1))
        modes = quota.get("sample_update_mode")
        expected = [0, 1, 2, 3] if count == 4 else [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        if count not in {4, 12} or modes != expected:
            raise ValueError(f"{name}: quota/mode contract is not q4 or q12")
        if int(arm.get("optimizer_steps_per_round", -1)) not in {10, 50}:
            raise ValueError(f"{name}: optimizer exposure must be 10 or 50")
        unknown = set(arm.get("parameter_delta", {})) - ALLOWED_DELTAS
        if unknown:
            raise ValueError(f"{name}: unsupported parameter deltas {sorted(unknown)}")
    return payload


def _scientific_args(arm: dict[str, Any], *, smoke: bool = False) -> list[str]:
    delta = {
        "execution_rule": "exponential_cost",
        "execution_clearance_exp_aggregation": "mean",
        "execution_clearance_exp_weight": 15,
        "execution_clearance_exp_temperature": 0.15,
        "execution_clearance_exp_target": 0.6,
        "execution_step_margin_mode": "linear",
        "execution_step_margin_weight": 0,
        "execution_cost_band_fraction": 0,
        "execution_cost_band_margin": "step",
        **arm.get("parameter_delta", {}),
    }
    if delta["execution_clearance_exp_aggregation"] not in {
        "mean", "max", "top3_mean",
    }:
        raise ValueError(f"{arm['name']}: invalid exponential aggregation")
    if delta["execution_rule"] not in {
        "min_cost", "exponential_cost", "quadratic_cost",
    }:
        raise ValueError(f"{arm['name']}: invalid cost-ranked execution rule")
    if delta["execution_step_margin_mode"] not in {"linear", "normalized"}:
        raise ValueError(f"{arm['name']}: invalid margin mode")
    if delta["execution_cost_band_margin"] not in {"step", "full"}:
        raise ValueError(f"{arm['name']}: invalid cost-band margin")
    if (
        float(delta["execution_cost_band_fraction"]) > 0
        and float(delta["execution_step_margin_weight"]) > 0
    ):
        raise ValueError(f"{arm['name']}: cost band and weighted margin conflict")
    quota = (
        {"successful_trajectories_per_gamma": 1, "sample_update_mode": [0]}
        if smoke else arm["quota"]
    )
    modes = ",".join(str(value) for value in quota["sample_update_mode"])
    return [
        "--flow-nfe", "12", "--batched-rollout-sampling",
        "--fa-alloc", "none",
        "--flow-base-std", "1", "--flow-base-std-schedule", "none",
        "--flow-base-std-final", "1", "--candidate-perturb-std", "0",
        "--candidate-perturb-scope", "coherent_horizon",
        "--learning-rate", "5e-5", "--learning-rate-final", "5e-5",
        "--learning-rate-decay-steps", "1", "--gradient-clip-norm", "5",
        "--round-learning-rate-warmup-power", "0",
        "--trainable-trunk-layers", "3",
        "--freeze-visual-encoder-during-expansion",
        "--beta", "0.1", "--parallel-episodes", "16",
        "--verifier-workers", "8", "--max-retry-batches", "32",
        "--retry-exhaustion-policy", "resample_scene",
        "--retry-resample-batch-cap", "32" if smoke else "384",
        "--successful-trajectories-per-gamma",
        str(quota["successful_trajectories_per_gamma"]),
        "--sample-update-mode", modes,
        "--sample-update-cohorts", "unguided_only",
        "--K", "16", "--B", "8", "--retry-B", "8",
        "--inner-steps", "1", "--optimizer-steps-per-round",
        "1" if smoke else str(arm["optimizer_steps_per_round"]),
        "--no-optimizer-steps-total",
        "--batch-size", "256", "--replay-scope", "cumulative",
        "--replay-batch-sampler", "mode_gamma_stratified",
        "--replay-top-fraction", "1", "--replay-selector", "uniform",
        "--replay-rounds", "100", "--gp-buffer-cap", "1536",
        "--gp-reference-mode", "sliding_success_per_gamma_current_phi",
        "--gp-sliding-row-selector", "trajectory_uniform",
        "--negative-alpha", "0",
        "--archive-rule", "successful_executed_windows",
        "--successful-trajectory-selector", "random_success",
        "--replay-acceptance", "execution_eligible",
        "--execution-rule", str(delta["execution_rule"]),
        "--execution-clearance-exp-weight",
        str(delta["execution_clearance_exp_weight"]),
        "--execution-clearance-exp-temperature",
        str(delta["execution_clearance_exp_temperature"]),
        "--execution-clearance-exp-aggregation",
        str(delta["execution_clearance_exp_aggregation"]),
        "--execution-clearance-target-m",
        str(delta["execution_clearance_exp_target"]),
        "--execution-taskspace-quadratic-weight", "0",
        "--execution-taskspace-quadratic-target-m", "0.15",
        "--execution-goal-side-wall-quadratic-weight", "0",
        "--execution-goal-side-wall-target-m", "0.6",
        "--execution-goal-box-exp-weight", "100",
        "--execution-goal-box-half-extent-m", "0.2",
        "--execution-goal-box-exp-temperature-m", "1.5",
        "--execution-axis-cylinder-quadratic-weight", "5",
        "--execution-axis-cylinder-radius-m", "1.1",
        "--execution-axis-cylinder-finite-segment",
        "--execution-control-weight", "1",
        "--execution-terminal-goal-weight", "80",
        "--execution-goal-braking-weight", "0",
        "--execution-goal-braking-distance-m", "0.6",
        "--execution-goal-braking-temperature-m", "0.15",
        "--execution-step-margin-weight",
        str(delta["execution_step_margin_weight"]),
        "--execution-step-margin-mode", str(delta["execution_step_margin_mode"]),
        "--execution-cost-band-fraction",
        str(delta["execution_cost_band_fraction"]),
        "--execution-cost-band-margin", str(delta["execution_cost_band_margin"]),
        "--acquisition-feature", "learned_phi", "--coverage-replay", "none",
        "--replay-augmentation", "none", "--execution-z-bias-mode", "none",
        "--tight-corridor", "--verifier-mode", "full_polytope",
        "--verifier-solver", "analytic", "--verifier-full-h-taskspace",
        "--event-log", "committed_success",
        "--paired-noised-representation", "--seed", str(arm["seed"]),
    ]


def _build_specs(
    matrix: dict[str, Any], *, source_id: str, remote_source: str,
    remote_pretrain: str, pretrained_sha256: str,
    task_config_sha256: str, smoke: bool,
) -> list[dict[str, Any]]:
    remote_stage = f"{REMOTE_ARTIFACT_BASE}/spooled_sweeps/{STAGE.name}"
    candidates = sorted(
        (arm for arm in matrix["arms"] if arm.get("launch_eligible")),
        key=lambda arm: (int(arm.get("priority", 10**9)), int(arm["index"])),
    )
    if smoke:
        candidates = candidates[:1]
    specs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for ordinal, arm in enumerate(candidates):
        gpu = 1 if ordinal % 2 == 0 else 3
        rounds = 1 if smoke else 8
        evaluation = {
            "episodes_per_gamma": 1 if smoke else 40,
            "probe_samples": 2 if smoke else 16,
            "flow_nfe": 12,
            "seed": 991234 if smoke else 91000,
            "fixed_scene_rollouts": 1 if smoke else 10,
            "fixed_scene_seed": 1991234 if smoke else 191000,
        }
        hash_payload = {
            "source_id": source_id,
            "pretrained_sha256": pretrained_sha256,
            "task_config_sha256": task_config_sha256,
            "rounds": rounds,
            "expansion_args": _scientific_args(arm, smoke=smoke),
            "evaluation": evaluation,
        }
        config_hash = _canonical_sha256(hash_payload)
        if config_hash in seen_hashes:
            raise ValueError(f"duplicate scientific configuration at {arm['name']}")
        seen_hashes.add(config_hash)
        suffix = "smoke" if smoke else f"{int(arm['index']):02d}"
        output = f"{remote_stage}/arms/{suffix}_{arm['name']}_{config_hash[:12]}"
        specs.append({
            "schema_version": 1, "name": arm["name"],
            "matrix_index": int(arm["index"]), "priority": arm.get("priority"),
            "wave": arm.get("wave"), "selector_id": arm.get("selector_id"),
            "physical_gpu": gpu, "rounds": rounds,
            "source_id": source_id, "remote_source": remote_source,
            "remote_pretrain": remote_pretrain,
            "pretrained_sha256": pretrained_sha256,
            "remote_task_config": f"{remote_source}/{TASK_CONFIG.relative_to(ROOT)}",
            "task_config_sha256": task_config_sha256,
            "remote_output": output, "remote_control": f"{remote_stage}/control",
            "expansion_args": hash_payload["expansion_args"],
            "evaluation": evaluation, "hash_payload": hash_payload,
            "config_hash": config_hash, "smoke": smoke,
        })
    if not specs:
        raise ValueError("ARM_MATRIX contains no launch-eligible arms")
    return specs


@contextmanager
def _submitter_lock() -> Iterator[None]:
    STAGE.mkdir(parents=True, exist_ok=True)
    path = STAGE / ".submitter.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another sweep submitter is active") from error
        yield


def _remote_queue_command(socket: str, control: str, gpu: int) -> str:
    save = f"{control}/gpu{gpu}.tsp.savelist"
    return (
        f"export TS_SOCKET={shlex.quote(socket)} "
        f"TS_SAVELIST={shlex.quote(save)} TS_MAXFINISHED=200; "
        "tsp -S 2 >/dev/null; test \"$(tsp -S)\" = 2"
    )


def _enqueue_spec(spec: dict[str, Any], remote_spec: str, socket: str) -> int:
    control = spec["remote_control"]
    log = f"{control}/logs/{spec['name']}--{spec['config_hash'][:12]}.log"
    label = f"{spec['matrix_index']:02d}:{spec['name']}:{spec['config_hash'][:8]}"
    worker = shlex.join([
        REMOTE_PYTHON,
        f"{spec['remote_source']}/scripts/run_pre2_spooled_arm.py",
        "--spec", remote_spec,
    ])
    shell = (
        "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
        f"export CUDA_VISIBLE_DEVICES={spec['physical_gpu']}; "
        "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
        f"exec {worker} >> {shlex.quote(log)} 2>&1"
    )
    command = (
        f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=200; "
        f"tsp -B -L {shlex.quote(label)} bash -lc {shlex.quote(shell)}"
    )
    output = subprocess.check_output(
        ["ssh", HELIOS_HOST, command], text=True,
    ).strip()
    if not output.isdigit():
        raise RuntimeError(f"unexpected tsp job id: {output!r}")
    return int(output)


def _stage_and_optionally_enqueue(
    matrix_path: Path, *, enqueue: bool, smoke: bool,
) -> dict[str, Any]:
    matrix = _read_matrix(matrix_path)
    if enqueue and not smoke:
        eligible = sum(bool(arm.get("launch_eligible")) for arm in matrix["arms"])
        if len(matrix["arms"]) != 64 or eligible != 64:
            raise ValueError(
                "full enqueue requires exactly 64/64 launch-eligible arms; "
                f"found {eligible}/{len(matrix['arms'])}"
            )
    source_id = _source_id(ROOT)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    task_sha = _sha256(TASK_CONFIG)
    with _ssh_master(HELIOS_HOST) as ssh:
        with _source_stage_lock(source_id):
            _stage_source(ssh, ROOT, remote_source)
            remote_pretrain, pretrain_sha = _stage_pretrain(ssh, PRETRAIN)
    specs = _build_specs(
        matrix, source_id=source_id, remote_source=remote_source,
        remote_pretrain=remote_pretrain, pretrained_sha256=pretrain_sha,
        task_config_sha256=task_sha, smoke=smoke,
    )
    remote_stage = f"{REMOTE_ARTIFACT_BASE}/spooled_sweeps/{STAGE.name}"
    remote_control = f"{remote_stage}/control"
    socket_tag = hashlib.sha256(STAGE.name.encode()).hexdigest()[:10]
    sockets = {gpu: f"/tmp/smppi-{socket_tag}-g{gpu}.sock" for gpu in (1, 3)}
    local_specs = STAGE / ("smoke_specs" if smoke else "specs")
    local_specs.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        _atomic_json(local_specs / f"{spec['matrix_index']:02d}_{spec['config_hash'][:12]}.json", spec)
    remote_specs = f"{remote_control}/{'smoke_specs' if smoke else 'specs'}"
    subprocess.run([
        "ssh", HELIOS_HOST,
        f"mkdir -p {shlex.quote(remote_specs)} {shlex.quote(remote_control)}/logs "
        f"{shlex.quote(remote_control)}/locks {shlex.quote(remote_control)}/status",
    ], check=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh", f"{local_specs}/",
        f"{HELIOS_HOST}:{remote_specs}/",
    ], check=True)
    records: list[dict[str, Any]] = []
    queue_path = STAGE / ("SMOKE_QUEUE.json" if smoke else "QUEUE.json")
    base_payload = {
        "schema_version": 1,
        "status": "ENQUEUE_IN_PROGRESS" if enqueue else "REMOTE_PREPARED_NOT_ENQUEUED",
        "created_unix": time.time(), "smoke": smoke,
        "matrix": str(matrix_path), "source_id": source_id,
        "remote_source": remote_source, "remote_pretrain": remote_pretrain,
        "pretrained_sha256": pretrain_sha, "task_config_sha256": task_sha,
        "remote_stage": remote_stage, "remote_control": remote_control,
        "gpu_policy": "GPU1/GPU3 only; independent task-spooler queues; 2 slots/GPU",
        "tsp_sockets": {str(key): value for key, value in sockets.items()},
        "arm_count": len(specs), "records": records,
    }
    _atomic_json(queue_path, base_payload)
    if enqueue:
        for gpu in (1, 3):
            subprocess.run([
                "ssh", HELIOS_HOST,
                _remote_queue_command(sockets[gpu], remote_control, gpu),
            ], check=True)
        for spec in specs:
            remote_spec = (
                f"{remote_control}/{'smoke_specs' if smoke else 'specs'}/"
                f"{spec['matrix_index']:02d}_{spec['config_hash'][:12]}.json"
            )
            job_id = _enqueue_spec(spec, remote_spec, sockets[spec["physical_gpu"]])
            records.append({
                "name": spec["name"], "config_hash": spec["config_hash"],
                "physical_gpu": spec["physical_gpu"], "tsp_job_id": job_id,
                "tsp_socket": sockets[spec["physical_gpu"]],
                "remote_spec": remote_spec, "remote_output": spec["remote_output"],
            })
            _atomic_json(queue_path, base_payload)
    payload = dict(base_payload)
    payload["status"] = "ENQUEUED" if enqueue else "REMOTE_PREPARED_NOT_ENQUEUED"
    _atomic_json(queue_path, payload)
    return payload


def local_validate(matrix_path: Path) -> dict[str, Any]:
    matrix = _read_matrix(matrix_path)
    source_id = _source_id(ROOT)
    specs = _build_specs(
        matrix, source_id=source_id,
        remote_source=f"{REMOTE_SOURCE_BASE}/{source_id}",
        remote_pretrain=f"{REMOTE_ARTIFACT_BASE}/pretrains/VALIDATION_ONLY",
        pretrained_sha256=_sha256(PRETRAIN / "pretrained.pt"),
        task_config_sha256=_sha256(TASK_CONFIG), smoke=False,
    )
    payload = {
        "status": "LOCAL_VALIDATION_PASSED", "arm_count": len(specs),
        "gpu_counts": {
            str(gpu): sum(spec["physical_gpu"] == gpu for spec in specs)
            for gpu in (1, 3)
        },
        "unique_config_hashes": len({spec["config_hash"] for spec in specs}),
        "source_id": source_id,
    }
    _atomic_json(STAGE / "SPOOLER_PREFLIGHT.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=STAGE / "ARM_MATRIX.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-remote", action="store_true")
    action.add_argument("--enqueue", action="store_true")
    action.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    with _submitter_lock():
        if args.enqueue:
            if args.confirm != CONFIRM_ENQUEUE:
                parser.error(f"--enqueue requires --confirm {CONFIRM_ENQUEUE}")
            existing = STAGE / "QUEUE.json"
            if existing.is_file():
                prior = json.loads(existing.read_text())
                if prior.get("status") != "REMOTE_PREPARED_NOT_ENQUEUED":
                    raise RuntimeError(f"refusing duplicate enqueue: {existing}")
            payload = _stage_and_optionally_enqueue(
                matrix_path, enqueue=True, smoke=False,
            )
        elif args.smoke:
            if args.confirm != CONFIRM_SMOKE:
                parser.error(f"--smoke requires --confirm {CONFIRM_SMOKE}")
            existing = STAGE / "SMOKE_QUEUE.json"
            if existing.is_file():
                raise RuntimeError(f"refusing duplicate smoke enqueue: {existing}")
            payload = _stage_and_optionally_enqueue(
                matrix_path, enqueue=True, smoke=True,
            )
        elif args.prepare_remote:
            payload = _stage_and_optionally_enqueue(
                matrix_path, enqueue=False, smoke=False,
            )
        else:
            payload = local_validate(matrix_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
