#!/usr/bin/env python3
"""Launch three fail-closed multi-sphere continuations directly on GPU1/GPU3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


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


R10_STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_r10_dual_eval_gpu13"
DIRECT_STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_r10_direct_gpu13"
PLOTTER = Path("/Users/dhl/Documents/safe_flow_expansion/scripts/paper_b1_margin50_trends.py")
CONFIRM = "I_UNDERSTAND_THIS_LAUNCHES_THREE_DIRECT_GPU_PROCESSES"
BASE_SPECS = (
    ("00_bf399c51d609.json", 1),
    ("01_c36b4ebc2bd7.json", 3),
    ("01_80bf3c5fe763.json", 3),
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def _replace_cli_value(arguments: list[str], option: str, value: str) -> None:
    index = arguments.index(option)
    arguments[index + 1] = value


def _set_or_append(arguments: list[str], option: str, value: str) -> None:
    if option in arguments:
        _replace_cli_value(arguments, option, value)
    else:
        arguments.extend([option, value])


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument(
        "--recover",
        action="store_true",
        help="preserve a failed direct launch and write a numbered relaunch record",
    )
    args = parser.parse_args()
    if args.confirm != CONFIRM:
        parser.error(f"--confirm must equal {CONFIRM}")
    launch_record = DIRECT_STAGE / "DIRECT_RUNS.json"
    if launch_record.exists():
        if not args.recover:
            raise RuntimeError("refusing to duplicate the direct launch")
        index = 1
        while (DIRECT_STAGE / f"DIRECT_RUNS_{index:02d}.json").exists():
            index += 1
        launch_record = DIRECT_STAGE / f"DIRECT_RUNS_{index:02d}.json"

    source_id = _source_id(ROOT)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    remote_stage = f"{REMOTE_ARTIFACT_BASE}/direct_runs/{DIRECT_STAGE.name}"
    remote_control = f"{remote_stage}/control"
    remote_plotter = f"{remote_control}/inputs/{PLOTTER.name}"
    with _ssh_master(HELIOS_HOST) as ssh:
        with _source_stage_lock(source_id):
            _stage_source(ssh, ROOT, remote_source)

    specs: list[dict[str, Any]] = []
    for ordinal, (filename, gpu) in enumerate(BASE_SPECS):
        base = json.loads((R10_STAGE / "specs" / filename).read_text())
        expansion_args = list(base["expansion_args"])
        _set_or_append(expansion_args, "--max-retry-batches", "8")
        _set_or_append(
            expansion_args,
            "--paired-scene-replace-after-retry-batches", "2",
        )
        _set_or_append(
            expansion_args,
            "--paired-scene-max-replacements-per-slot", "1",
        )
        _set_or_append(
            expansion_args,
            "--paired-scene-replace-interval-batches", "1",
        )
        base.update({
            "schema_version": 1,
            "ordinal": ordinal,
            "physical_gpu": gpu,
            "expansion_source": remote_source,
            "evaluation_source": remote_source,
            "remote_control": remote_control,
            "remote_plotter": remote_plotter,
            "expansion_args": expansion_args,
        })
        base["hash_payload"] = {
            **base["hash_payload"],
            "expansion_source": remote_source,
            "evaluation_source": remote_source,
            "expansion_args": expansion_args,
            "direct_recovery": {
                "launcher": "nohup_direct_no_task_spooler",
                "max_retry_batches": 8,
                "paired_scene_replace_after_retry_batches": 2,
                "paired_scene_max_replacements_per_slot": 1,
                "paired_scene_replace_interval_batches": 1,
            },
        }
        base["config_hash"] = _canonical_sha256(base["hash_payload"])
        specs.append(base)

    local_specs = DIRECT_STAGE / "specs"
    for spec in specs:
        _write(
            local_specs / f"{spec['ordinal']:02d}_{spec['config_hash'][:12]}.json",
            spec,
        )
    subprocess.run([
        "ssh", HELIOS_HOST,
        f"mkdir -p {shlex.quote(remote_control)}/{{specs,inputs,logs,status}}",
    ], check=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh", f"{local_specs}/",
        f"{HELIOS_HOST}:{remote_control}/specs/",
    ], check=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh", str(PLOTTER),
        f"{HELIOS_HOST}:{remote_plotter}",
    ], check=True)

    records = []
    for spec in specs:
        remote_spec = (
            f"{remote_control}/specs/{spec['ordinal']:02d}_"
            f"{spec['config_hash'][:12]}.json"
        )
        log = f"{remote_control}/logs/{spec['name']}--{spec['config_hash'][:12]}.log"
        worker = shlex.join([
            REMOTE_PYTHON,
            f"{remote_source}/scripts/run_multisphere_r10_dual_eval_arm.py",
            "--spec", remote_spec,
        ])
        shell = (
            "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
            f"export CUDA_VISIBLE_DEVICES={spec['physical_gpu']}; "
            "export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12; "
            f"exec {worker}"
        )
        launch = (
            f"nohup bash -lc {shlex.quote(shell)} > {shlex.quote(log)} 2>&1 "
            f"< /dev/null & pid=$!; sleep 1; kill -0 $pid; echo $pid"
        )
        pid = int(subprocess.check_output(
            ["ssh", HELIOS_HOST, launch], text=True,
        ).strip())
        records.append({
            "name": spec["name"],
            "physical_gpu": spec["physical_gpu"],
            "pid": pid,
            "config_hash": spec["config_hash"],
            "remote_spec": remote_spec,
            "remote_log": log,
            "remote_output": spec["remote_output"],
        })

    payload = {
        "schema_version": 1,
        "status": "DIRECT_RUNNING",
        "created_unix": time.time(),
        "source_id": source_id,
        "remote_source": remote_source,
        "remote_control": remote_control,
        "task_spooler_used": False,
        "gpu_policy": "direct nohup on physical GPU1/GPU3; GPU0/GPU2 unused",
        "recovery_contract": {
            "max_retry_batches": 8,
            "paired_scene_replace_after_retry_batches": 2,
            "paired_scene_max_replacements_per_slot": 1,
            "paired_scene_replace_interval_batches": 1,
        },
        "records": records,
    }
    payload["launch_record"] = str(launch_record)
    _write(launch_record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
