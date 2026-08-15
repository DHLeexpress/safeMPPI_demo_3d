#!/usr/bin/env python3
"""Validate, stage, or enqueue paired PRE2 r0->r2 speed-band pilots.

The pilots preserve the registered reach03/E15 legacy recipe and change only
the expansion-time obstacle-conditioned speed weight plus the approved 5%
bounded-cost/margin selector.  Evaluation is delegated to the generic spooled
worker, whose fixed-bank evaluator remains raw deployment.

The default action is local validation.  GPU work requires ``--enqueue`` and
the exact confirmation phrase printed by ``--help``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterator


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))

from safe_mppi.helios_remote import (  # noqa: E402
    HELIOS_HOST,
    REMOTE_ARTIFACT_BASE,
    REMOTE_SOURCE_BASE,
    _sha256,
    _source_id,
    _source_stage_lock,
    _ssh_master,
    _stage_pretrain,
    _stage_source,
)
from scripts.submit_pre2_margin_spooler_sweep import (  # noqa: E402
    _atomic_json,
    _canonical_sha256,
    _enqueue_spec,
    _remote_queue_command,
)


STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_obstacle_speed_band_pilot"
)
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
TASK_CONFIG = ROOT / (
    "configs/lab_ball_stage1_goalspace_yminus04_z01_17_"
    "r15in_reach03_v1.json"
)
MATRIX = STAGE / "PILOT_MATRIX.json"
CONFIRM_ENQUEUE = "I_UNDERSTAND_THIS_ENQUEUES_THE_R2_SPEED_BAND_PILOTS"


def _read_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported PILOT_MATRIX schema")
    if payload.get("paired_seed") != 82410:
        raise ValueError("paired_seed must equal the legacy seed 82410")
    arms = payload.get("arms")
    if not isinstance(arms, list) or not 2 <= len(arms) <= 8:
        raise ValueError("PILOT_MATRIX.arms must contain 2..8 paired arms")
    names: set[str] = set()
    weights: set[float] = set()
    for index, arm in enumerate(arms):
        name = arm.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name,
        ):
            raise ValueError(f"unsafe arm name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate arm name: {name}")
        names.add(name)
        if int(arm.get("index", -1)) != index:
            raise ValueError(f"{name}: indices must be dense and ordered")
        weight = float(arm.get("execution_obstacle_speed_weight", -1.0))
        if not 0.0 <= weight < 400.0:
            raise ValueError(f"{name}: speed weight must be in [0, 400)")
        if weight in weights:
            raise ValueError(f"duplicate speed weight: {weight}")
        weights.add(weight)
        if float(arm.get("execution_cost_band_fraction", -1.0)) != 0.05:
            raise ValueError(f"{name}: cost-band fraction must equal 0.05")
        if arm.get("execution_cost_band_margin") != "step":
            raise ValueError(f"{name}: cost-band margin must equal step")
    return payload


def _scientific_args(arm: dict[str, Any]) -> list[str]:
    """Return the exact legacy E15 q12 recipe plus two execution deltas."""
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
        "--retry-resample-batch-cap", "192",
        "--successful-trajectories-per-gamma", "12",
        "--sample-update-mode", "0,0,0,1,1,1,2,2,2,3,3,3",
        "--sample-update-cohorts", "unguided_only",
        "--K", "16", "--B", "8", "--retry-B", "8",
        "--inner-steps", "1", "--optimizer-steps-per-round", "2500",
        "--no-optimizer-steps-total", "--batch-size", "256",
        "--replay-scope", "cumulative",
        "--replay-batch-sampler", "mode_gamma_stratified",
        "--replay-top-fraction", "1", "--replay-selector", "uniform",
        "--replay-rounds", "100", "--gp-buffer-cap", "1536",
        "--gp-reference-mode", "sliding_success_per_gamma_current_phi",
        "--gp-sliding-row-selector", "trajectory_uniform",
        "--negative-alpha", "0",
        "--archive-rule", "successful_executed_windows",
        "--successful-trajectory-selector", "random_success",
        "--replay-acceptance", "execution_eligible",
        "--execution-rule", "exponential_cost",
        "--execution-clearance-exp-weight", "15",
        "--execution-clearance-exp-temperature", "0.15",
        "--execution-clearance-exp-aggregation", "mean",
        "--execution-clearance-target-m", "0.6",
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
        "--execution-step-margin-weight", "0",
        "--execution-step-margin-mode", "linear",
        "--execution-cost-band-fraction",
        str(arm["execution_cost_band_fraction"]),
        "--execution-cost-band-margin", arm["execution_cost_band_margin"],
        "--execution-obstacle-speed-weight",
        str(arm["execution_obstacle_speed_weight"]),
        "--acquisition-feature", "learned_phi", "--coverage-replay", "none",
        "--replay-augmentation", "none", "--execution-z-bias-mode", "none",
        "--tight-corridor", "--verifier-mode", "full_polytope",
        "--verifier-solver", "analytic", "--verifier-full-h-taskspace",
        "--event-log", "committed_success",
        "--paired-noised-representation", "--seed", "82410",
    ]


def _build_specs(
    matrix: dict[str, Any], *, source_id: str, remote_source: str,
    remote_pretrain: str, pretrained_sha256: str, task_config_sha256: str,
) -> list[dict[str, Any]]:
    remote_stage = f"{REMOTE_ARTIFACT_BASE}/spooled_sweeps/{STAGE.name}"
    specs: list[dict[str, Any]] = []
    hashes: set[str] = set()
    evaluation = {
        "episodes_per_gamma": 40,
        "probe_samples": 16,
        "flow_nfe": 12,
        "seed": 91000,
        "fixed_scene_rollouts": 10,
        "fixed_scene_seed": 191000,
    }
    for arm in matrix["arms"]:
        expansion_args = _scientific_args(arm)
        hash_payload = {
            "source_id": source_id,
            "pretrained_sha256": pretrained_sha256,
            "task_config_sha256": task_config_sha256,
            "rounds": 2,
            "expansion_args": expansion_args,
            "evaluation": evaluation,
        }
        config_hash = _canonical_sha256(hash_payload)
        if config_hash in hashes:
            raise ValueError(f"duplicate scientific config: {arm['name']}")
        hashes.add(config_hash)
        index = int(arm["index"])
        gpu = 1 if index % 2 == 0 else 3
        remote_output = (
            f"{remote_stage}/arms/{index:02d}_{arm['name']}_{config_hash[:12]}"
        )
        specs.append({
            "schema_version": 1,
            "name": arm["name"],
            "matrix_index": index,
            "physical_gpu": gpu,
            "rounds": 2,
            "source_id": source_id,
            "remote_source": remote_source,
            "remote_pretrain": remote_pretrain,
            "pretrained_sha256": pretrained_sha256,
            "remote_task_config": f"{remote_source}/{TASK_CONFIG.relative_to(ROOT)}",
            "task_config_sha256": task_config_sha256,
            "remote_output": remote_output,
            "remote_control": f"{remote_stage}/control",
            "expansion_args": expansion_args,
            "evaluation": evaluation,
            "hash_payload": hash_payload,
            "config_hash": config_hash,
            "scientific_delta": {
                "execution_obstacle_speed_weight": arm[
                    "execution_obstacle_speed_weight"
                ],
                "execution_cost_band_fraction": 0.05,
                "execution_cost_band_margin": "step",
            },
        })
    return specs


@contextmanager
def _submitter_lock() -> Iterator[None]:
    STAGE.mkdir(parents=True, exist_ok=True)
    with (STAGE / ".submitter.lock").open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another speed-band submitter is active") from error
        yield


def _specs_for_local_validation(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = _source_id(ROOT)
    return _build_specs(
        matrix,
        source_id=source_id,
        remote_source=f"{REMOTE_SOURCE_BASE}/{source_id}",
        remote_pretrain=f"{REMOTE_ARTIFACT_BASE}/pretrains/VALIDATION_ONLY",
        pretrained_sha256=_sha256(PRETRAIN / "pretrained.pt"),
        task_config_sha256=_sha256(TASK_CONFIG),
    )


def _assert_execution_only_plumbing() -> None:
    expansion_script = ROOT / "scripts/research_ball_expansion_optimization.py"
    help_text = subprocess.check_output(
        [sys.executable, str(expansion_script), "--help"],
        cwd=ROOT,
        text=True,
    )
    for option in (
        "--execution-obstacle-speed-weight",
        "--execution-cost-band-fraction",
        "--execution-cost-band-margin",
    ):
        if option not in help_text:
            raise RuntimeError(f"expansion CLI is missing required option {option}")

    worker_text = (ROOT / "scripts/run_pre2_spooled_arm.py").read_text()
    try:
        evaluation_block = worker_text.split("evaluation_command = [", 1)[1].split(
            "_run(evaluation_command", 1,
        )[0]
    except IndexError as error:
        raise RuntimeError("could not audit the spooled evaluation command") from error
    forbidden = (
        "execution-obstacle-speed-weight",
        "execution-cost-band-fraction",
        "execution-cost-band-margin",
    )
    leaked = [token for token in forbidden if token in evaluation_block]
    if leaked:
        raise RuntimeError(f"expansion selector leaked into raw evaluation: {leaked}")


def local_validate(matrix_path: Path) -> dict[str, Any]:
    _assert_execution_only_plumbing()
    matrix = _read_matrix(matrix_path)
    specs = _specs_for_local_validation(matrix)
    forbidden_eval_tokens = {
        "--execution-obstacle-speed-weight",
        "--execution-cost-band-fraction",
        "--execution-cost-band-margin",
    }
    if any(forbidden_eval_tokens.intersection(spec["evaluation"]) for spec in specs):
        raise AssertionError("expansion-only selector leaked into evaluation")
    payload = {
        "status": "LOCAL_VALIDATION_PASSED",
        "arm_count": len(specs),
        "paired_seed": 82410,
        "rounds": 2,
        "gpu_counts": {
            str(gpu): sum(spec["physical_gpu"] == gpu for spec in specs)
            for gpu in (1, 3)
        },
        "unique_config_hashes": len({spec["config_hash"] for spec in specs}),
        "raw_evaluation_has_execution_selector": False,
        "source_id": specs[0]["source_id"],
    }
    _atomic_json(STAGE / "PILOT_PREFLIGHT.json", payload)
    return payload


def _stage_and_enqueue(matrix_path: Path) -> dict[str, Any]:
    _assert_execution_only_plumbing()
    matrix = _read_matrix(matrix_path)
    source_id = _source_id(ROOT)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    task_sha = _sha256(TASK_CONFIG)
    with _ssh_master(HELIOS_HOST) as ssh:
        with _source_stage_lock(source_id):
            _stage_source(ssh, ROOT, remote_source)
            remote_pretrain, pretrain_sha = _stage_pretrain(ssh, PRETRAIN)
    specs = _build_specs(
        matrix,
        source_id=source_id,
        remote_source=remote_source,
        remote_pretrain=remote_pretrain,
        pretrained_sha256=pretrain_sha,
        task_config_sha256=task_sha,
    )
    remote_stage = f"{REMOTE_ARTIFACT_BASE}/spooled_sweeps/{STAGE.name}"
    remote_control = f"{remote_stage}/control"
    sockets = {
        1: "/tmp/smppi-speedband-r2-g1.sock",
        3: "/tmp/smppi-speedband-r2-g3.sock",
    }
    for socket in sockets.values():
        subprocess.run([
            "ssh", HELIOS_HOST, f"test ! -e {shlex.quote(socket)}",
        ], check=True)
    local_specs = STAGE / "specs"
    local_specs.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        _atomic_json(
            local_specs / f"{spec['matrix_index']:02d}_{spec['config_hash'][:12]}.json",
            spec,
        )
    remote_specs = f"{remote_control}/specs"
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
    payload = {
        "schema_version": 1,
        "status": "ENQUEUE_IN_PROGRESS",
        "created_unix": time.time(),
        "matrix": str(matrix_path),
        "source_id": source_id,
        "remote_source": remote_source,
        "remote_pretrain": remote_pretrain,
        "pretrained_sha256": pretrain_sha,
        "task_config_sha256": task_sha,
        "remote_stage": remote_stage,
        "remote_control": remote_control,
        "gpu_policy": "GPU1/GPU3 only; 2 task-spooler slots per GPU",
        "tsp_sockets": {str(gpu): socket for gpu, socket in sockets.items()},
        "arm_count": len(specs),
        "records": records,
        "raw_evaluation_policy": (
            "fixed seed91000 raw deployment; no execution selector/cost transplant"
        ),
    }
    queue_path = STAGE / "QUEUE.json"
    _atomic_json(queue_path, payload)
    for gpu in (1, 3):
        subprocess.run([
            "ssh", HELIOS_HOST,
            _remote_queue_command(sockets[gpu], remote_control, gpu),
        ], check=True)
    for spec in specs:
        remote_spec = (
            f"{remote_specs}/{spec['matrix_index']:02d}_"
            f"{spec['config_hash'][:12]}.json"
        )
        job_id = _enqueue_spec(spec, remote_spec, sockets[spec["physical_gpu"]])
        records.append({
            "name": spec["name"],
            "config_hash": spec["config_hash"],
            "physical_gpu": spec["physical_gpu"],
            "tsp_job_id": job_id,
            "tsp_socket": sockets[spec["physical_gpu"]],
            "remote_spec": remote_spec,
            "remote_output": spec["remote_output"],
        })
        _atomic_json(queue_path, payload)
    payload["status"] = "ENQUEUED"
    _atomic_json(queue_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=f"required with --enqueue: {CONFIRM_ENQUEUE}",
    )
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    with _submitter_lock():
        if args.enqueue:
            if args.confirm != CONFIRM_ENQUEUE:
                parser.error(f"--enqueue requires --confirm {CONFIRM_ENQUEUE}")
            queue = STAGE / "QUEUE.json"
            if queue.is_file():
                raise RuntimeError(f"refusing duplicate enqueue: {queue}")
            payload = _stage_and_enqueue(matrix_path)
        else:
            payload = local_validate(matrix_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
