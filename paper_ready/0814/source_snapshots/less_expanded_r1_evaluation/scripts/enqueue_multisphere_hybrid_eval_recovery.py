#!/usr/bin/env python3
"""Queue evaluation-only recovery for committed R5 hybrid arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_hybrid_spooler_sweep"
QUEUE = STAGE / "QUEUE.json"
RECOVERY = STAGE / "EVAL_RECOVERY_QUEUE.json"
HOST = "dohyun@helios.robotics.caltech.edu"
REMOTE_PYTHON = "/home/dohyun/miniforge3/envs/cfm_mppi/bin/python"


def _ssh(command: str) -> str:
    return subprocess.check_output(["ssh", HOST, command], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-ids", default="0,1,2,3,5,6",
        help="comma-separated original task-spooler job IDs",
    )
    args = parser.parse_args()
    queue = json.loads(QUEUE.read_text())
    if queue.get("gpu_policy", "").find("GPU0 only") < 0:
        raise RuntimeError("recovery is restricted to the GPU0 sweep")
    recovery = (
        json.loads(RECOVERY.read_text()) if RECOVERY.is_file() else {
            "status": "EVALUATION_ONLY_RECOVERY_ENQUEUED",
            "created_unix": time.time(),
            "scientific_configuration_changed": False,
            "expansion_rerun": False,
            "records": [],
            "append_events": [],
        }
    )
    existing = {
        int(record["original_job_id"])
        for record in recovery.get("records", [])
    }
    requested_job_ids = tuple(
        int(value.strip()) for value in args.job_ids.split(",") if value.strip()
    )
    selected_job_ids = tuple(
        job_id for job_id in requested_job_ids if job_id not in existing
    )
    if not selected_job_ids:
        raise RuntimeError("all requested evaluation recoveries are already queued")
    records = []
    # Primary performance, primary safety, then controls/ablation. These arms
    # all reached an authoritative committed R5 before the legacy exit check.
    by_id = {int(row["tsp_job_id"]): row for row in queue["records"]}
    for original_job_id in selected_job_ids:
        record = by_id[original_job_id]
        state = _ssh(
            f"export TS_SOCKET={shlex.quote(record['tsp_socket'])}; "
            f"tsp -s {original_job_id}"
        ).lower()
        if state != "finished":
            raise RuntimeError(
                f"original job {original_job_id} is not finished: {state}"
            )
        running_glob = (
            f"{queue['remote_control']}/status/"
            f"{record['name']}--{record['config_hash'][:12]}.RUNNING.json"
        )
        check = (
            f"test -f {shlex.quote(record['remote_output'] + '/checkpoint_005.pt')} && "
            f"test -f {shlex.quote(record['remote_output'] + '/resume_state.json')} && "
            f"test ! -f {shlex.quote(running_glob)}"
        )
        subprocess.run(["ssh", HOST, check], check=True)
        remote_spec = (
            f"{queue['remote_control']}/specs/"
            f"{original_job_id:02d}_{record['config_hash'][:12]}.json"
        )
        log = (
            f"{queue['remote_control']}/logs/"
            f"RECOVER_EVAL__{record['name']}--{record['config_hash'][:12]}.log"
        )
        worker = shlex.join([
            REMOTE_PYTHON,
            f"{queue['remote_source']}/scripts/run_multisphere_hybrid_spooled_arm.py",
            "--spec", remote_spec,
        ])
        shell = (
            "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
            "export CUDA_VISIBLE_DEVICES=0; "
            "export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16; "
            f"exec {worker} >> {shlex.quote(log)} 2>&1"
        )
        enqueue = (
            f"export TS_SOCKET={shlex.quote(record['tsp_socket'])} TS_MAXFINISHED=100; "
            f"tsp -B -L {shlex.quote('recover_eval__' + record['name'])} "
            f"bash -lc {shlex.quote(shell)}"
        )
        recovery_job_id = int(_ssh(enqueue))
        records.append({
            "name": record["name"],
            "original_job_id": original_job_id,
            "recovery_job_id": recovery_job_id,
            "config_hash": record["config_hash"],
            "purpose": "evaluation only from existing committed checkpoint_005.pt",
        })
    # tsp -u moves one job to the front; reverse traversal preserves the
    # desired primary->safety->control order after all insertions.
    socket = queue["tsp_sockets"]["0"]
    for row in reversed(records):
        _ssh(
            f"export TS_SOCKET={shlex.quote(socket)}; "
            f"tsp -u {int(row['recovery_job_id'])}"
        )
    recovery["records"].extend(records)
    recovery.setdefault("append_events", []).append({
        "created_unix": time.time(),
        "original_job_ids": list(selected_job_ids),
        "recovery_job_ids": [row["recovery_job_id"] for row in records],
    })
    RECOVERY.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n")
    print(json.dumps(recovery, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
