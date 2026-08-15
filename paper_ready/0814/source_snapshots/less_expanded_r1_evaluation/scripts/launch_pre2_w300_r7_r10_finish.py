#!/usr/bin/env python3
"""Validate or enqueue the exact W300/5000 r7-to-r10 finish on GPU3."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))
STAGE = ROOT / "results/stage1_single_ball_t128/0812_pre2_saved_r1_steps5000_r7"
SOURCE_SPEC = next((STAGE / "recovery_specs").glob("w300_5000--*.json"))
REMOTE_STAGE = (
    "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
    "0812_pre2_saved_r1_steps5000_r7/r7_r10_finish"
)
REMOTE_OUTPUT = (
    "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
    "0812_pre2_saved_r1_steps5000_r7/arms/speed300_steps5000_r7_s82410"
)
HELIOS = "dohyun@helios.robotics.caltech.edu"
REMOTE_PYTHON = "/home/dohyun/miniforge3/envs/cfm_mppi/bin/python"
GPU3_SOCKET = "/tmp/smppi-speedband-r2-g3.sock"
CONFIRM = "I_UNDERSTAND_THIS_CONTINUES_EXACT_W300_R7_TO_R10_ON_GPU3"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_spec() -> dict[str, Any]:
    from scripts.run_pre2_w300_r7_r10_finish import (
        _canonical_sha256, _validate_spec,
    )

    source = _read(SOURCE_SPEC)
    spec = {
        "schema_version": 1,
        "kind": "w300_steps5000_exact_r7_r10",
        "name": "speed300_steps5000_exact_r10_s82410",
        "physical_gpu": 3,
        "source_config_hash": source["config_hash"],
        "remote_source": source["remote_source"],
        "remote_pretrain": source["remote_pretrain"],
        "remote_task_config": source["remote_task_config"],
        "remote_output": REMOTE_OUTPUT,
        "remote_control": f"{REMOTE_STAGE}/control",
        "start_round": 7,
        "target_round": 10,
        "expansion_args": deepcopy(source["expansion_args"]),
        "evaluation": {
            "episodes_per_gamma": 160,
            "fixed_scene_rollouts": 10,
            "fixed_scene_seed": 191000,
            "flow_nfe": 12,
            "probe_samples": 16,
            "seed": 91000,
        },
    }
    spec["hash_payload"] = {
        key: spec[key] for key in (
            "kind", "name", "physical_gpu", "source_config_hash",
            "remote_source", "remote_pretrain", "remote_task_config",
            "remote_output", "remote_control", "start_round", "target_round",
            "expansion_args", "evaluation",
        )
    }
    spec["config_hash"] = _canonical_sha256(spec["hash_payload"])
    previous = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    try:
        __import__("os").environ["CUDA_VISIBLE_DEVICES"] = "3"
        _validate_spec(spec)
    finally:
        if previous is None:
            __import__("os").environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            __import__("os").environ["CUDA_VISIBLE_DEVICES"] = previous
    return spec


def remote_validate(spec: dict[str, Any]) -> None:
    command = (
        "set -euo pipefail; "
        f"out={shlex.quote(spec['remote_output'])}; "
        "test -s \"$out/checkpoint_007.pt\"; "
        "test -s \"$out/query_archive_round_007.pt\"; "
        "test -s \"$out/events_round_007.pt\"; "
        "test -s \"$out/resume_state_latest.pt\"; "
        "test -s \"$out/manifest.json\"; "
        "test ! -e \"$out/checkpoint_008.pt\"; "
        "test -d " + shlex.quote(spec["remote_source"]) + "; "
        "test -s " + shlex.quote(
            f"{spec['remote_source']}/scripts/research_ball_expansion_optimization.py"
        )
    )
    subprocess.run(["ssh", HELIOS, command], check=True)


def launch() -> dict[str, Any]:
    marker = STAGE / "R10_FINISH_LAUNCHED.json"
    lock_path = STAGE / ".r10-finish-launch.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another r10 finish launcher is active") from error
        if marker.exists():
            raise RuntimeError(f"refusing duplicate launch: {marker}")
        spec = build_spec()
        remote_validate(spec)
        local_spec = STAGE / "r10_finish" / f"spec--{spec['config_hash'][:12]}.json"
        _atomic_json(local_spec, spec)
        remote_spec_dir = f"{REMOTE_STAGE}/control/specs"
        remote_worker_dir = f"{REMOTE_STAGE}/control/worker"
        remote_spec = f"{remote_spec_dir}/{local_spec.name}"
        remote_worker = f"{remote_worker_dir}/run_pre2_w300_r7_r10_finish.py"
        log = f"{REMOTE_STAGE}/control/logs/{spec['name']}--{spec['config_hash'][:12]}.log"
        subprocess.run([
            "ssh", HELIOS,
            "mkdir -p " + " ".join(map(shlex.quote, (
                remote_spec_dir, remote_worker_dir,
                f"{REMOTE_STAGE}/control/logs", f"{REMOTE_STAGE}/control/locks",
                f"{REMOTE_STAGE}/control/status",
            ))),
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", str(local_spec),
            f"{HELIOS}:{remote_spec}",
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh",
            str(ROOT / "scripts/run_pre2_w300_r7_r10_finish.py"),
            f"{HELIOS}:{remote_worker}",
        ], check=True)
        worker = shlex.join([REMOTE_PYTHON, remote_worker, "--spec", remote_spec])
        shell = (
            "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
            "export CUDA_VISIBLE_DEVICES=3; "
            "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
            f"exec {worker} >> {shlex.quote(log)} 2>&1"
        )
        label = f"w300-r10:{spec['config_hash'][:8]}"
        submit = (
            f"export TS_SOCKET={shlex.quote(GPU3_SOCKET)} TS_MAXFINISHED=200; "
            f"tsp -B -L {shlex.quote(label)} bash -lc {shlex.quote(shell)}"
        )
        output = subprocess.check_output(["ssh", HELIOS, submit], text=True).strip()
        if not output.isdigit():
            raise RuntimeError(f"unexpected task-spooler job id: {output!r}")
        payload = {
            "schema_version": 1,
            "status": "LAUNCHED",
            "created_unix": time.time(),
            "physical_gpu": 3,
            "tsp_socket": GPU3_SOCKET,
            "tsp_job_id": int(output),
            "config_hash": spec["config_hash"],
            "local_spec": str(local_spec),
            "remote_spec": remote_spec,
            "remote_worker": remote_worker,
            "remote_log": log,
            "remote_output": spec["remote_output"],
            "start_round": 7,
            "target_round": 10,
            "evaluation": spec["evaluation"],
            "evaluation_policy": (
                "all preregistered seed91000 rollouts; no cherry-picking; "
                "95% Wilson and mean +/- 1.96 SE downstream"
            ),
        }
        _atomic_json(marker, payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.launch:
        if args.confirm != CONFIRM:
            parser.error(f"--launch requires --confirm {CONFIRM}")
        payload = launch()
    else:
        payload = {
            "status": "LOCAL_VALIDATION_PASSED",
            "spec": build_spec(),
            "source_spec": str(SOURCE_SPEC),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
