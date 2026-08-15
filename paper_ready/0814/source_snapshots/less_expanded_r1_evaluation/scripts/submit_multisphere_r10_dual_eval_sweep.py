#!/usr/bin/env python3
"""Queue four exact R5→R10 continuations and faithful/hack raw curves."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
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
    _source_id,
    _source_stage_lock,
    _ssh_master,
    _stage_source,
)


SOURCE_STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_hybrid_spooler_sweep"
STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_r10_dual_eval_gpu13"
PLOTTER = Path(
    "/Users/dhl/Documents/safe_flow_expansion/scripts/paper_b1_margin50_trends.py"
)
CONFIRM = "I_UNDERSTAND_THIS_QUEUES_FOUR_R10_MULTISPHERE_ARMS_ON_GPU1_GPU3"
SOCKETS = {
    1: "/tmp/smppi-speedband-r2-g1.sock",
    3: "/tmp/smppi-speedband-r2-g3.sock",
}
ARM_ASSIGNMENT = (
    ("dense_z0711__speed200_band05_inner50", 1),
    ("uniform_z0612__speed200_band05_inner50", 3),
    ("dense_z0711__speed400_band05_inner50", 1),
    ("uniform_z0612__speed400_band05_inner50", 3),
)
SPEED100_ASSIGNMENT = (
    ("dense_z0711__speed100_band05_inner50", 1),
    ("uniform_z0612__speed100_band05_inner50", 3),
)
SOURCE_ROUNDS = {
    "dense_z0711__speed100_band05_inner50": 5,
    "uniform_z0612__speed100_band05_inner50": 3,
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def _replace_cli_value(arguments: list[str], option: str, value: str) -> list[str]:
    updated = list(arguments)
    try:
        index = updated.index(option)
    except ValueError as error:
        raise RuntimeError(f"missing required expansion option: {option}") from error
    if index + 1 >= len(updated):
        raise RuntimeError(f"missing value for expansion option: {option}")
    updated[index + 1] = value
    return updated


def _load_original_specs() -> dict[str, dict[str, Any]]:
    queue = json.loads((SOURCE_STAGE / "QUEUE.json").read_text())
    hashes = {row["name"]: row["config_hash"] for row in queue["records"]}
    output: dict[str, dict[str, Any]] = {}
    for name, config_hash in hashes.items():
        candidates = list((SOURCE_STAGE / "specs").glob(f"*_{config_hash[:12]}.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"cannot resolve authoritative R5 spec for {name}")
        output[name] = json.loads(candidates[0].read_text())
    return output


@contextmanager
def _lock() -> Iterator[None]:
    STAGE.mkdir(parents=True, exist_ok=True)
    with (STAGE / ".submitter.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _specs(
    originals: dict[str, dict[str, Any]], evaluation_source: str,
    remote_control: str, remote_plotter: str,
    assignment: tuple[tuple[str, int], ...] = ARM_ASSIGNMENT,
) -> list[dict[str, Any]]:
    specs = []
    for ordinal, (name, gpu) in enumerate(assignment):
        original = originals[name]
        source_round = int(SOURCE_ROUNDS.get(name, 5))
        payload = {
            "original_config_hash": original["config_hash"],
            "expansion_source": evaluation_source,
            "evaluation_source": evaluation_source,
            "source_round": source_round,
            "target_rounds": 10,
            "physical_gpu": gpu,
            "expansion_args": original["expansion_args"],
            "faithful": {"NFE": 16, "raw_plans_per_step": 1},
            "hack": {"NFE": 16, "raw_plans_per_step": 8},
            "episodes_per_gamma": 50,
            "fixed_seed": 91000,
        }
        config_hash = _canonical_sha256(payload)
        specs.append({
            "schema_version": 1,
            "name": name,
            "scene_id": original["scene_id"],
            "variant_id": original["variant_id"],
            "physical_gpu": gpu,
            "source_round": source_round,
            "target_rounds": 10,
            "expansion_source": evaluation_source,
            "evaluation_source": evaluation_source,
            "remote_pretrain": original["remote_pretrain"],
            "remote_task_config": original["remote_task_config"],
            "remote_scene_bank": original["remote_scene_bank"],
            "remote_output": original["remote_output"],
            "remote_control": remote_control,
            "remote_plotter": remote_plotter,
            "obstacle_speed_weight": original["obstacle_speed_weight"],
            "cost_band_fraction": original["cost_band_fraction"],
            "inner_steps": original["inner_steps"],
            "expansion_args": original["expansion_args"],
            "original_config_hash": original["config_hash"],
            "hash_payload": payload,
            "config_hash": config_hash,
            "ordinal": ordinal,
        })
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument(
        "--recover-failed", action="store_true",
        help="re-enqueue the four failed plumbing jobs after staging a patched worker",
    )
    parser.add_argument(
        "--recover-deadlock", choices=tuple(name for name, _ in ARM_ASSIGNMENT),
        help=(
            "re-enqueue only one fail-closed scientific sampling deadlock with "
            "one extra whole parallel batch; healthy/running arms are untouched"
        ),
    )
    parser.add_argument(
        "--replace-speed400-with100", action="store_true",
        help=(
            "queue only dense/uniform speed100 continuations after preserving "
            "and superseding the two speed400 jobs"
        ),
    )
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.recover_failed or args.recover_deadlock or args.replace_speed400_with100:
        args.enqueue = True
    if args.enqueue and args.confirm != CONFIRM:
        parser.error(f"--enqueue requires --confirm {CONFIRM}")
    if not PLOTTER.is_file():
        raise FileNotFoundError(PLOTTER)

    with _lock():
        if (
            args.enqueue and (STAGE / "QUEUE.json").exists()
            and not args.recover_failed and not args.recover_deadlock
            and not args.replace_speed400_with100
        ):
            raise RuntimeError("refusing duplicate enqueue")
        if (
            (args.recover_failed or args.recover_deadlock or args.replace_speed400_with100)
            and not (STAGE / "QUEUE.json").is_file()
        ):
            raise RuntimeError("recovery requires the original QUEUE.json")
        originals = _load_original_specs()
        source_id = _source_id(ROOT)
        remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
        remote_stage = f"{REMOTE_ARTIFACT_BASE}/spooled_sweeps/{STAGE.name}"
        remote_control = f"{remote_stage}/control"
        remote_plotter = f"{remote_control}/inputs/{PLOTTER.name}"
        with _ssh_master(HELIOS_HOST) as ssh:
            with _source_stage_lock(source_id):
                _stage_source(ssh, ROOT, remote_source)
        assignment = (
            SPEED100_ASSIGNMENT
            if args.replace_speed400_with100 else ARM_ASSIGNMENT
        )
        specs = _specs(
            originals, remote_source, remote_control, remote_plotter,
            assignment=assignment,
        )
        if args.replace_speed400_with100:
            for spec in specs:
                spec["expansion_args"] = _replace_cli_value(
                    spec["expansion_args"], "--max-retry-batches", "2",
                )
                spec["hash_payload"]["expansion_args"] = spec["expansion_args"]
                spec["hash_payload"]["speed_sweep_replacement"] = {
                    "supersedes_speed": 400.0,
                    "replacement_speed": 100.0,
                    "bounded_max_retry_batches": 2,
                }
                spec["config_hash"] = _canonical_sha256(spec["hash_payload"])
        if args.recover_deadlock:
            specs = [spec for spec in specs if spec["name"] == args.recover_deadlock]
            if len(specs) != 1:
                raise RuntimeError("deadlock recovery did not resolve exactly one arm")
            spec = specs[0]
            spec["expansion_args"] = _replace_cli_value(
                spec["expansion_args"], "--max-retry-batches", "2",
            )
            spec["hash_payload"]["expansion_args"] = spec["expansion_args"]
            spec["hash_payload"]["deadlock_rescue"] = {
                "trigger": "no_commit_capable_success_for_one_gamma",
                "base_max_retry_batches": 1,
                "rescue_max_retry_batches": 2,
                "semantics": (
                    "replay deterministic batch 0, then permit exactly one "
                    "additional whole 24-episode parallel batch"
                ),
            }
            spec["config_hash"] = _canonical_sha256(spec["hash_payload"])
        local_specs = STAGE / "specs"
        local_specs.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            _write(
                local_specs / f"{spec['ordinal']:02d}_{spec['config_hash'][:12]}.json",
                spec,
            )
        subprocess.run([
            "ssh", HELIOS_HOST,
            f"mkdir -p {shlex.quote(remote_control)}/{{specs,inputs,logs,locks,status}}",
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", f"{local_specs}/",
            f"{HELIOS_HOST}:{remote_control}/specs/",
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", str(PLOTTER),
            f"{HELIOS_HOST}:{remote_plotter}",
        ], check=True)

        # Reuse the already-running GPU1/GPU3 task-spooler servers.  Do not
        # alter their slot counts or any active/queued job.
        for gpu, socket in SOCKETS.items():
            subprocess.run([
                "ssh", HELIOS_HOST,
                f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=200; tsp >/dev/null",
            ], check=True)

        records = []
        if args.enqueue:
            for spec in specs:
                remote_spec = (
                    f"{remote_control}/specs/{spec['ordinal']:02d}_"
                    f"{spec['config_hash'][:12]}.json"
                )
                log = (
                    f"{remote_control}/logs/{spec['name']}--"
                    f"{spec['config_hash'][:12]}.log"
                )
                worker = shlex.join([
                    REMOTE_PYTHON,
                    f"{remote_source}/scripts/run_multisphere_r10_dual_eval_arm.py",
                    "--spec", remote_spec,
                ])
                shell = (
                    "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
                    f"export CUDA_VISIBLE_DEVICES={spec['physical_gpu']}; "
                    "export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16; "
                    f"exec {worker} >> {shlex.quote(log)} 2>&1"
                )
                socket = SOCKETS[spec["physical_gpu"]]
                if args.replace_speed400_with100:
                    label_prefix = "r10-speed100-replacement:"
                elif args.recover_deadlock:
                    label_prefix = "r10-deadlock-rescue:"
                elif args.recover_failed:
                    label_prefix = "r10-recovery:"
                else:
                    label_prefix = "r10:"
                enqueue = (
                    f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=200; "
                    f"tsp -B -L {shlex.quote(label_prefix + spec['name'])} "
                    f"bash -lc {shlex.quote(shell)}"
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
            "evaluation_source": remote_source,
            "remote_control": remote_control,
            "gpu_policy": (
                "physical GPU1/GPU3 only; reuse existing task-spooler servers "
                "without changing their slots or active jobs; GPU0/GPU2 unused"
            ),
            "scientific_contract": {
                "resume": "authoritative completed R5 checkpoint to R10 in place",
                "scenes": ["dense_z0711", "uniform_z0612"],
                "obstacle_speed_weights": [200.0, 400.0],
                "faithful": "NFE16 raw M1; no verifier/progress selection",
                "hack": "NFE16 raw M8; 5% cost-band then nominal margin",
                "attempts_per_checkpoint": 200,
                "fixed_seed_bank_per_scene": True,
            },
            "records": records,
        }
        if args.replace_speed400_with100:
            output_name = "SPEED100_REPLACEMENT_QUEUE.json"
            payload["scientific_contract"].update({
                "scenes": ["dense_z0711", "uniform_z0612"],
                "obstacle_speed_weights": [100.0, 200.0],
                "speed100_source_rounds": SOURCE_ROUNDS,
                "bounded_max_retry_batches": 2,
                "superseded_speed": 400.0,
            })
        elif args.recover_deadlock:
            recovery_index = 1
            while (STAGE / f"DEADLOCK_RECOVERY_QUEUE_{recovery_index:02d}.json").exists():
                recovery_index += 1
            output_name = f"DEADLOCK_RECOVERY_QUEUE_{recovery_index:02d}.json"
            payload["scientific_contract"]["deadlock_rescue"] = {
                "arm": args.recover_deadlock,
                "max_retry_batches": 2,
                "extra_parallel_batches": 1,
            }
        elif args.recover_failed:
            recovery_index = 1
            while (STAGE / f"RECOVERY_QUEUE_{recovery_index:02d}.json").exists():
                recovery_index += 1
            output_name = f"RECOVERY_QUEUE_{recovery_index:02d}.json"
        else:
            output_name = "QUEUE.json" if args.enqueue else "PREFLIGHT.json"
        _write(STAGE / output_name, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
