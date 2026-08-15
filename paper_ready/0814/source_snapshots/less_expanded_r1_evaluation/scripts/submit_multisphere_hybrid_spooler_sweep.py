#!/usr/bin/env python3
"""Stage and enqueue the GPU0-only multi-sphere hybrid R5 sweep."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
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
    REMOTE_PYTHON,
    REMOTE_SOURCE_BASE,
    _sha256,
    _source_id,
    _source_stage_lock,
    _ssh_master,
    _stage_pretrain,
    _stage_source,
)


STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_hybrid_spooler_sweep"
MATRIX = STAGE / "ARM_MATRIX.json"
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
CONFIRM = "I_UNDERSTAND_THIS_ENQUEUES_THE_MULTISPHERE_HYBRID_SWEEP"
GPUS = (0,)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def _expansion_args(variant: dict[str, Any]) -> list[str]:
    return [
        "--paired-scene-rotation", "start_goal_axis_180",
        "--flow-nfe", "16",
        "--batched-rollout-sampling",
        "--flow-base-std", "1.0",
        "--flow-base-std-schedule", "none",
        "--candidate-perturb-std", "0",
        "--fa-alloc", "none",
        "--parallel-episodes", "24",
        "--verifier-workers", "8",
        "--max-retry-batches", "1",
        "--retry-exhaustion-policy", "commit_available",
        "--successful-trajectories-per-gamma", "12",
        "--successful-trajectories-per-round", "48",
        "--sample-update-cohorts", "unguided_only",
        "--K", "16", "--B", "8", "--retry-B", "8",
        "--beta", "0.1",
        "--archive-rule", "successful_executed_windows",
        "--successful-trajectory-selector", "random_success",
        "--replay-acceptance", "execution_eligible",
        "--replay-scope", "sliding",
        "--replay-rounds", "3",
        "--replay-batch-sampler", "row_permutation",
        "--replay-passes-per-round", "1",
        "--inner-steps", str(variant["inner_steps"]),
        "--batch-size", "128",
        "--no-optimizer-steps-total",
        "--replay-top-fraction", "1",
        "--replay-selector", "uniform",
        "--learning-rate", "2e-5",
        "--gradient-clip-norm", "1",
        "--trainable-trunk-layers", "3",
        "--freeze-visual-encoder-during-expansion",
        "--negative-alpha", "0",
        "--execution-rule", "min_cost",
        "--execution-clearance-exp-weight", "15",
        "--execution-clearance-target-m", "0.6",
        "--execution-clearance-exp-temperature", "0.15",
        "--execution-taskspace-quadratic-weight", "250",
        "--execution-taskspace-quadratic-target-m", "0.15",
        "--execution-axis-cylinder-quadratic-weight", "5",
        "--execution-axis-cylinder-radius-m", "1.1",
        "--execution-control-weight", "0.05",
        "--execution-obstacle-speed-weight",
        str(variant["obstacle_speed_weight"]),
        "--execution-cost-band-fraction",
        str(variant["cost_band_fraction"]),
        "--execution-z-bias-mode", "none",
        "--acquisition-feature", "learned_phi",
        "--gp-buffer-cap", "1536",
        "--gp-reference-mode", "sliding_success_per_gamma_current_phi",
        "--gp-sliding-row-selector", "trajectory_uniform",
        "--coverage-replay", "none",
        "--replay-augmentation", "none",
        "--paired-noised-representation",
        "--event-log", "committed_success",
        "--seed", "81620",
    ]


def _matrix() -> dict[str, Any]:
    value = json.loads(MATRIX.read_text())
    if value.get("schema_version") != 1:
        raise ValueError("unsupported matrix schema")
    if len(value.get("scenes", [])) != 2 or len(value.get("variants", [])) < 1:
        raise ValueError("matrix requires two scenes and variants")
    return value


def _specs(
    matrix: dict[str, Any], source_id: str, remote_source: str,
    remote_pretrain: str, pretrained_sha256: str, remote_control: str,
) -> list[dict[str, Any]]:
    rows = []
    ordinal = 0
    for variant in sorted(matrix["variants"], key=lambda row: row["priority"]):
        for scene in matrix["scenes"]:
            task_config = ROOT / scene["task_config"]
            scene_bank = ROOT / scene["scene_bank"]
            if not task_config.is_file() or not scene_bank.is_file():
                raise FileNotFoundError(task_config if not task_config.is_file() else scene_bank)
            name = f"{scene['id']}__{variant['id']}"
            gpu = GPUS[ordinal % len(GPUS)]
            expansion_args = _expansion_args(variant)
            payload = {
                "source_id": source_id,
                "pretrained_sha256": pretrained_sha256,
                "task_config_sha256": _sha256(task_config),
                "scene_bank_sha256": _sha256(scene_bank),
                "rounds": 5,
                "expansion_args": expansion_args,
                "obstacle_speed_weight": variant["obstacle_speed_weight"],
                "cost_band_fraction": variant["cost_band_fraction"],
            }
            config_hash = _canonical_sha256(payload)
            rows.append({
                "schema_version": 1,
                "name": name,
                "scene_id": scene["id"],
                "variant_id": variant["id"],
                "priority": variant["priority"],
                "role": variant["role"],
                "physical_gpu": gpu,
                "rounds": 5,
                "source_id": source_id,
                "remote_source": remote_source,
                "remote_pretrain": remote_pretrain,
                "pretrained_sha256": pretrained_sha256,
                "remote_task_config": (
                    f"{remote_source}/{task_config.relative_to(ROOT)}"
                ),
                "task_config_sha256": payload["task_config_sha256"],
                "remote_scene_bank": (
                    f"{remote_control}/inputs/{scene_bank.name}"
                ),
                "scene_bank_sha256": payload["scene_bank_sha256"],
                "remote_output": (
                    f"{remote_control.rsplit('/control', 1)[0]}/arms/"
                    f"{ordinal:02d}_{name}_{config_hash[:12]}"
                ),
                "remote_control": remote_control,
                "expansion_args": expansion_args,
                "obstacle_speed_weight": variant["obstacle_speed_weight"],
                "cost_band_fraction": variant["cost_band_fraction"],
                "inner_steps": variant["inner_steps"],
                "hash_payload": payload,
                "config_hash": config_hash,
            })
            ordinal += 1
    return rows


@contextmanager
def _lock() -> Iterator[None]:
    STAGE.mkdir(parents=True, exist_ok=True)
    with (STAGE / ".submitter.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.enqueue and args.confirm != CONFIRM:
        parser.error(f"--enqueue requires --confirm {CONFIRM}")
    with _lock():
        if args.enqueue and (STAGE / "QUEUE.json").exists():
            raise RuntimeError("refusing duplicate enqueue")
        matrix = _matrix()
        source_id = _source_id(ROOT)
        remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
        remote_stage = f"{REMOTE_ARTIFACT_BASE}/spooled_sweeps/{STAGE.name}"
        remote_control = f"{remote_stage}/control"
        with _ssh_master(HELIOS_HOST) as ssh:
            with _source_stage_lock(source_id):
                _stage_source(ssh, ROOT, remote_source)
                remote_pretrain, pretrain_sha = _stage_pretrain(ssh, PRETRAIN)
        specs = _specs(
            matrix, source_id, remote_source, remote_pretrain,
            pretrain_sha, remote_control,
        )
        local_specs = STAGE / "specs"
        local_specs.mkdir(parents=True, exist_ok=True)
        for index, spec in enumerate(specs):
            _write(local_specs / f"{index:02d}_{spec['config_hash'][:12]}.json", spec)
        subprocess.run([
            "ssh", HELIOS_HOST,
            f"mkdir -p {shlex.quote(remote_control)}/{{specs,inputs,logs,locks,status}}",
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", f"{local_specs}/",
            f"{HELIOS_HOST}:{remote_control}/specs/",
        ], check=True)
        for scene in matrix["scenes"]:
            scene_bank = ROOT / scene["scene_bank"]
            subprocess.run([
                "rsync", "-az", "-e", "ssh", str(scene_bank),
                f"{HELIOS_HOST}:{remote_control}/inputs/{scene_bank.name}",
            ], check=True)
        sockets = {
            gpu: f"/tmp/smppi-ms-hybrid-{hashlib.sha256(STAGE.name.encode()).hexdigest()[:8]}-g{gpu}.sock"
            for gpu in GPUS
        }
        records = []
        if args.enqueue:
            for gpu, socket in sockets.items():
                command = (
                    f"export TS_SOCKET={shlex.quote(socket)} "
                    f"TS_SAVELIST={shlex.quote(remote_control + f'/gpu{gpu}.tsp.savelist')} "
                    "TS_MAXFINISHED=100; tsp -S 2 >/dev/null; test \"$(tsp -S)\" = 2"
                )
                subprocess.run(["ssh", HELIOS_HOST, command], check=True)
            for index, spec in enumerate(specs):
                remote_spec = (
                    f"{remote_control}/specs/"
                    f"{index:02d}_{spec['config_hash'][:12]}.json"
                )
                log = f"{remote_control}/logs/{spec['name']}--{spec['config_hash'][:12]}.log"
                worker = shlex.join([
                    REMOTE_PYTHON,
                    f"{remote_source}/scripts/run_multisphere_hybrid_spooled_arm.py",
                    "--spec", remote_spec,
                ])
                shell = (
                    "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
                    f"export CUDA_VISIBLE_DEVICES={spec['physical_gpu']}; "
                    "export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16; "
                    f"exec {worker} >> {shlex.quote(log)} 2>&1"
                )
                socket = sockets[spec["physical_gpu"]]
                enqueue = (
                    f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=100; "
                    f"tsp -B -L {shlex.quote(spec['name'])} bash -lc {shlex.quote(shell)}"
                )
                job_id = subprocess.check_output(
                    ["ssh", HELIOS_HOST, enqueue], text=True,
                ).strip()
                records.append({
                    "name": spec["name"],
                    "physical_gpu": spec["physical_gpu"],
                    "tsp_job_id": int(job_id),
                    "tsp_socket": socket,
                    "remote_output": spec["remote_output"],
                    "config_hash": spec["config_hash"],
                })
        payload = {
            "schema_version": 1,
            "status": "ENQUEUED" if args.enqueue else "REMOTE_PREPARED_NOT_ENQUEUED",
            "created_unix": time.time(),
            "source_id": source_id,
            "remote_source": remote_source,
            "remote_control": remote_control,
            "pretrained_sha256": pretrain_sha,
            "gpu_policy": (
                "physical GPU0 only; existing GPU1/GPU3 work untouched; "
                "GPU2 forbidden"
            ),
            "slots_per_gpu": 2,
            "tsp_sockets": {str(gpu): socket for gpu, socket in sockets.items()},
            "arm_count": len(specs),
            "records": records,
        }
        _write(STAGE / ("QUEUE.json" if args.enqueue else "PREFLIGHT.json"), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
