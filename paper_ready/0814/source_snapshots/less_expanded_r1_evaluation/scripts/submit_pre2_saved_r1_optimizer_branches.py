#!/usr/bin/env python3
"""Stage and enqueue the approved W150/W300 optimizer-exposure branches.

Default execution is a local, read-only validation.  ``--enqueue`` attaches
the W150/5000 and W300/5000 workers to their already-enqueued bootstrap job
IDs, and attaches the W300/2500 r2-r7 continuation to the original r2 worker.
The separate CPU-only waiter remains available for a future branch whose
bootstrap has not already been enqueued.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))

from safe_mppi.helios_remote import (  # noqa: E402
    HELIOS_HOST, REMOTE_PYTHON, REMOTE_SOURCE_BASE,
    _source_id, _source_stage_lock, _ssh_master, _stage_source,
)


STAGE = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_saved_r1_steps5000_r7"
)
W150_SPEC = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed_band_steps_r10/"
    "specs/03_41d1c90fd1b1.json"
)
W300_SPEC = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed300_pilot/"
    "specs/00_a6047e8a106b.json"
)
W300_QUEUE = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed300_pilot/QUEUE.json"
)
REMOTE_STAGE = (
    "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
    "0812_pre2_saved_r1_steps5000_r7"
)
GPU1_SOCKET = "/tmp/smppi-speedband-r2-g1.sock"
GPU3_SOCKET = "/tmp/smppi-speedband-r2-g3.sock"
CONFIRM = "I_UNDERSTAND_THIS_ENQUEUES_THE_EXACT_SAVED_R1_BRANCHES"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _replace_arg(args: list[str], name: str, value: str) -> None:
    args[args.index(name) + 1] = value


def _resume_spec(
    base: dict[str, Any], *, kind: str, name: str, source_output: str,
    remote_output: str, physical_gpu: int, optimizer_steps: int,
    remote_source: str, source_id: str,
) -> dict[str, Any]:
    from scripts.run_pre2_exact_optimizer_resume import _canonical_sha256, _validate_spec

    args = deepcopy(base["expansion_args"])
    _replace_arg(args, "--optimizer-steps-per-round", str(optimizer_steps))
    spec = {
        "schema_version": 1, "kind": kind, "name": name,
        "physical_gpu": physical_gpu, "source_id": source_id,
        "source_config_hash": base["config_hash"],
        "source_output": source_output,
        "remote_source": remote_source,
        "remote_pretrain": base["remote_pretrain"],
        "remote_task_config": base["remote_task_config"],
        "remote_output": remote_output,
        "remote_control": f"{REMOTE_STAGE}/control",
        "rounds": 7,
        "start_round": 1 if kind == "saved_r1_optimizer_branch" else 2,
        "expansion_args": args,
        "evaluation": deepcopy(base["evaluation"]),
    }
    spec["hash_payload"] = {
        key: spec[key] for key in (
            "kind", "source_id", "source_config_hash", "source_output",
            "physical_gpu", "rounds", "start_round", "expansion_args",
            "evaluation", "remote_output",
        )
    }
    spec["config_hash"] = _canonical_sha256(spec["hash_payload"])
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        _validate_spec(spec)
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    return spec


def build_specs(
    *, source_id: str | None = None, remote_source: str | None = None,
) -> dict[str, dict[str, Any]]:
    source_id = _source_id(ROOT) if source_id is None else source_id
    remote_source = (
        f"{REMOTE_SOURCE_BASE}/{source_id}"
        if remote_source is None else remote_source
    )
    w150 = _read(W150_SPEC)
    w300 = _read(W300_SPEC)
    w150_output = f"{REMOTE_STAGE}/arms/speed150_steps5000_r7_s82410"
    w300_output = f"{REMOTE_STAGE}/arms/speed300_steps5000_r7_s82410"
    saved150 = _resume_spec(
        w150, kind="saved_r1_optimizer_branch",
        name="speed150_steps5000_r7_s82410",
        source_output=w150["remote_output"], remote_output=w150_output,
        physical_gpu=3, optimizer_steps=5000,
        remote_source=remote_source, source_id=source_id,
    )
    saved300 = _resume_spec(
        w300, kind="saved_r1_optimizer_branch",
        name="speed300_steps5000_r7_s82410",
        source_output=w300["remote_output"], remote_output=w300_output,
        physical_gpu=3, optimizer_steps=5000,
        remote_source=remote_source, source_id=source_id,
    )
    continue300 = _resume_spec(
        w300, kind="exact_r2_continuation",
        name="speed300_steps2500_r7_s82410",
        source_output=w300["remote_output"], remote_output=w300["remote_output"],
        physical_gpu=1, optimizer_steps=2500,
        remote_source=remote_source, source_id=source_id,
    )
    return {"w150_5000": saved150, "w300_5000": saved300, "w300_2500": continue300}


def _waiter_spec(
    saved300: dict[str, Any], *, source_spec: str, branch_spec: str,
) -> dict[str, Any]:
    from scripts.wait_enqueue_pre2_saved_r1_branch import (
        _canonical_sha256, _validate_spec,
    )

    spec = {
        "schema_version": 1,
        "source_output": saved300["source_output"],
        "source_spec": source_spec,
        "source_config_hash": saved300["source_config_hash"],
        "branch_output": saved300["remote_output"],
        "branch_spec": branch_spec,
        "remote_source": saved300["remote_source"],
        "remote_pretrain": saved300["remote_pretrain"],
        "remote_control": saved300["remote_control"],
        "remote_python": REMOTE_PYTHON,
        "physical_gpu": 3, "tsp_socket": GPU3_SOCKET,
        "target_optimizer_steps": 5000, "target_rounds": 7,
    }
    spec["hash_payload"] = {
        key: spec[key] for key in (
            "source_output", "source_spec", "source_config_hash",
            "branch_output", "branch_spec", "remote_source",
            "remote_pretrain", "physical_gpu", "tsp_socket",
            "target_optimizer_steps", "target_rounds",
        )
    }
    spec["config_hash"] = _canonical_sha256(spec["hash_payload"])
    _validate_spec(spec)
    return spec


def _enqueue_worker(
    spec: dict[str, Any], remote_spec: str, *, socket: str,
    dependency: int | None,
) -> int:
    label = f"exact:{spec['name']}:{spec['config_hash'][:8]}"
    log = f"{spec['remote_control']}/logs/{spec['name']}--{spec['config_hash'][:12]}.log"
    worker = shlex.join([
        REMOTE_PYTHON,
        f"{spec['remote_source']}/scripts/run_pre2_exact_optimizer_resume.py",
        "--spec", remote_spec,
    ])
    shell = (
        "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
        f"export CUDA_VISIBLE_DEVICES={spec['physical_gpu']}; "
        "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
        f"exec {worker} >> {shlex.quote(log)} 2>&1"
    )
    dependency_arg = "" if dependency is None else f" -D {dependency}"
    command = (
        f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=200; "
        f"tsp -B -L {shlex.quote(label)}{dependency_arg} "
        f"bash -lc {shlex.quote(shell)}"
    )
    output = subprocess.check_output(["ssh", HELIOS_HOST, command], text=True).strip()
    if not output.isdigit():
        raise RuntimeError(f"unexpected task-spooler job ID: {output!r}")
    return int(output)


def local_validate() -> dict[str, Any]:
    specs = build_specs(source_id="VALIDATION_SOURCE", remote_source="/validation/source")
    return {
        "status": "LOCAL_VALIDATION_PASSED",
        "spec_hashes": {key: spec["config_hash"] for key, spec in specs.items()},
        "contracts": {
            "w150_5000": "prepared r1 native eval, exact r2-r7, native r0-r7",
            "w300_5000": "bootstrap dependency -> prepared r1 native eval -> r2-r7",
            "w300_2500": "depends on original r2+native-eval completion -> exact r3-r7",
        },
        "evaluation": next(iter(specs.values()))["evaluation"],
    }


def enqueue(*, w150_bootstrap_job: int, w300_bootstrap_job: int) -> dict[str, Any]:
    queue_path = STAGE / "QUEUE.json"
    STAGE.mkdir(parents=True, exist_ok=True)
    lock_path = STAGE / ".submitter.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another exact-branch submitter is active") from error
        if queue_path.exists():
            raise RuntimeError(f"refusing duplicate enqueue: {queue_path}")
        source_id = _source_id(ROOT)
        remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
        with _ssh_master(HELIOS_HOST) as ssh:
            with _source_stage_lock(source_id):
                _stage_source(ssh, ROOT, remote_source)
        specs = build_specs(source_id=source_id, remote_source=remote_source)
        w300_queue = _read(W300_QUEUE)
        if (
            int(w300_queue["tsp_job_id"]) != 13
            or w300_queue["tsp_socket"] != GPU1_SOCKET
            or w300_queue["config_hash"] != specs["w300_2500"]["source_config_hash"]
        ):
            raise RuntimeError("recorded live W300/2500 source job changed")
        local_specs = STAGE / "specs"
        local_specs.mkdir(parents=True, exist_ok=True)
        for key, spec in specs.items():
            _atomic_json(local_specs / f"{key}--{spec['config_hash'][:12]}.json", spec)
        remote_specs = f"{REMOTE_STAGE}/control/specs"
        subprocess.run([
            "ssh", HELIOS_HOST,
            f"mkdir -p {shlex.quote(remote_specs)} "
            f"{shlex.quote(REMOTE_STAGE + '/control/logs')} "
            f"{shlex.quote(REMOTE_STAGE + '/control/locks')} "
            f"{shlex.quote(REMOTE_STAGE + '/control/status')}",
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", f"{local_specs}/",
            f"{HELIOS_HOST}:{remote_specs}/",
        ], check=True)
        remote_spec_paths = {
            key: f"{remote_specs}/{key}--{spec['config_hash'][:12]}.json"
            for key, spec in specs.items()
        }
        w150_worker = _enqueue_worker(
            specs["w150_5000"], remote_spec_paths["w150_5000"],
            socket=GPU3_SOCKET, dependency=w150_bootstrap_job,
        )
        w300_continuation = _enqueue_worker(
            specs["w300_2500"], remote_spec_paths["w300_2500"],
            socket=GPU1_SOCKET, dependency=int(w300_queue["tsp_job_id"]),
        )
        w300_worker = _enqueue_worker(
            specs["w300_5000"], remote_spec_paths["w300_5000"],
            socket=GPU3_SOCKET, dependency=w300_bootstrap_job,
        )
        payload = {
            "schema_version": 1, "status": "ENQUEUED",
            "created_unix": time.time(), "source_id": source_id,
            "remote_source": remote_source, "remote_control": f"{REMOTE_STAGE}/control",
            "jobs": {
                "w150_5000": {
                    "bootstrap_dependency": w150_bootstrap_job,
                    "worker_job_id": w150_worker, "socket": GPU3_SOCKET,
                    "spec": remote_spec_paths["w150_5000"],
                },
                "w300_2500": {
                    "source_dependency": int(w300_queue["tsp_job_id"]),
                    "worker_job_id": w300_continuation, "socket": GPU1_SOCKET,
                    "spec": remote_spec_paths["w300_2500"],
                },
                "w300_5000": {
                    "bootstrap_dependency": w300_bootstrap_job,
                    "worker_job_id": w300_worker, "socket": GPU3_SOCKET,
                    "spec": remote_spec_paths["w300_5000"],
                },
            },
            "raw_evaluation_policy": (
                "r1 5000-exposure screening before continuation; final r0-r7; "
                "native seed91000 40 episodes/gamma without execution selector"
            ),
        }
        _atomic_json(queue_path, payload)
        return payload


def recover_manifest_failures() -> dict[str, Any]:
    """Resume the two saved-r1 branches after the audited manifest-only fault."""
    queue = _read(STAGE / "QUEUE.json")
    recovery_path = STAGE / "MANIFEST_RECOVERY.json"
    lock_path = STAGE / ".manifest-recovery.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another manifest recovery is active") from error
        if recovery_path.exists():
            raise RuntimeError(f"refusing duplicate recovery: {recovery_path}")
        old_jobs = {
            key: int(queue["jobs"][key]["worker_job_id"])
            for key in ("w150_5000", "w300_5000")
        }
        for key, job_id in old_jobs.items():
            state = subprocess.check_output([
                "ssh", HELIOS_HOST,
                f"export TS_SOCKET={shlex.quote(GPU3_SOCKET)}; tsp -s {job_id}",
            ], text=True).strip()
            detail = subprocess.check_output([
                "ssh", HELIOS_HOST,
                f"export TS_SOCKET={shlex.quote(GPU3_SOCKET)}; tsp -i {job_id}",
            ], text=True)
            if state != "finished" or "died with exit code 1" not in detail:
                raise RuntimeError(f"{key} old worker is not the audited failure")
        remote_check = (
            f"set -e; base={shlex.quote(REMOTE_STAGE)}; "
            "for arm in speed150_steps5000_r7_s82410 "
            "speed300_steps5000_r7_s82410; do "
            "test -s \"$base/arms/$arm/BRANCH_PROVENANCE.json\"; "
            "test -s \"$base/arms/$arm/checkpoint_001.pt\"; "
            "test ! -e \"$base/arms/$arm/checkpoint_002.pt\"; "
            "test -s \"$base/arms/$arm/fixed_eval_r001_steps5000_native_seed91000/raw_eval.json\"; "
            "test ! -e \"$base/arms/$arm/manifest.json\"; done"
        )
        subprocess.run(["ssh", HELIOS_HOST, remote_check], check=True)

        source_id = _source_id(ROOT)
        remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
        with _ssh_master(HELIOS_HOST) as ssh:
            with _source_stage_lock(source_id):
                _stage_source(ssh, ROOT, remote_source)
        specs = build_specs(source_id=source_id, remote_source=remote_source)
        local_specs = STAGE / "recovery_specs"
        local_specs.mkdir(parents=True, exist_ok=True)
        selected = {key: specs[key] for key in ("w150_5000", "w300_5000")}
        for key, spec in selected.items():
            _atomic_json(
                local_specs / f"{key}--{spec['config_hash'][:12]}.json", spec,
            )
        remote_specs = f"{REMOTE_STAGE}/control/recovery_specs"
        subprocess.run([
            "ssh", HELIOS_HOST,
            f"mkdir -p {shlex.quote(remote_specs)}",
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", f"{local_specs}/",
            f"{HELIOS_HOST}:{remote_specs}/",
        ], check=True)
        new_jobs: dict[str, int] = {}
        remote_spec_paths: dict[str, str] = {}
        for key, spec in selected.items():
            remote_spec = (
                f"{remote_specs}/{key}--{spec['config_hash'][:12]}.json"
            )
            remote_spec_paths[key] = remote_spec
            new_jobs[key] = _enqueue_worker(
                spec, remote_spec, socket=GPU3_SOCKET, dependency=None,
            )
        payload = {
            "schema_version": 1,
            "status": "MANIFEST_ONLY_FAILURE_RECOVERED",
            "created_unix": time.time(),
            "cause": (
                "saved-r1 branch lacked the canonical semantic manifest; r1 "
                "native evaluations completed and no r2 artifact was written"
            ),
            "scientific_delta": "none",
            "evaluation_reused": True,
            "source_id": source_id,
            "remote_source": remote_source,
            "old_failed_jobs": old_jobs,
            "new_jobs": new_jobs,
            "remote_specs": remote_spec_paths,
        }
        _atomic_json(recovery_path, payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--w150-bootstrap-job", type=int, default=15)
    parser.add_argument("--w300-bootstrap-job", type=int, default=16)
    parser.add_argument("--recover-manifest-failures", action="store_true")
    args = parser.parse_args()
    if args.recover_manifest_failures:
        if args.confirm != CONFIRM:
            parser.error(
                f"--recover-manifest-failures requires --confirm {CONFIRM}"
            )
        payload = recover_manifest_failures()
    elif args.enqueue:
        if args.confirm != CONFIRM:
            parser.error(f"--enqueue requires --confirm {CONFIRM}")
        payload = enqueue(
            w150_bootstrap_job=args.w150_bootstrap_job,
            w300_bootstrap_job=args.w300_bootstrap_job,
        )
    else:
        payload = local_validate()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
