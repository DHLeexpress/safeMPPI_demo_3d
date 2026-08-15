#!/usr/bin/env python3
"""Enqueue one fresh-PRE2 W300/2500 r0->r2 calibration arm."""
from __future__ import annotations

import argparse
from copy import deepcopy
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
SOURCE_STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_obstacle_speed_band_steps_r10"
)
STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_obstacle_speed300_pilot"
)
BASE_SPEC = SOURCE_STAGE / "specs/03_41d1c90fd1b1.json"
SOCKET = "/tmp/smppi-speedband-r2-g1.sock"
CONFIRM = "I_UNDERSTAND_THIS_ENQUEUES_THE_W300_R2_PILOT"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build_spec(base: dict[str, Any] | None = None) -> dict[str, Any]:
    from scripts.run_pre2_spooled_arm import _canonical_sha256

    source = deepcopy(_read(BASE_SPEC) if base is None else base)
    args = source["expansion_args"]
    weight_index = args.index("--execution-obstacle-speed-weight") + 1
    steps_index = args.index("--optimizer-steps-per-round") + 1
    if float(args[weight_index]) != 150.0 or int(args[steps_index]) != 2500:
        raise ValueError("base spec is not the paired W150/2500 arm")
    args[weight_index] = "300.0"

    source["name"] = "speed300_steps2500_r2_s82410"
    source["matrix_index"] = 0
    source["physical_gpu"] = 1
    source["rounds"] = 2
    source["remote_control"] = (
        "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
        f"{STAGE.name}/control"
    )
    source["scientific_delta"] = {
        "execution_obstacle_speed_weight": 300.0,
        "execution_cost_band_fraction": 0.05,
        "execution_cost_band_margin": "step",
        "optimizer_steps_per_round": 2500,
    }
    source["hash_payload"] = {
        "source_id": source["source_id"],
        "pretrained_sha256": source["pretrained_sha256"],
        "task_config_sha256": source["task_config_sha256"],
        "rounds": source["rounds"],
        "expansion_args": source["expansion_args"],
        "evaluation": source["evaluation"],
    }
    source["config_hash"] = _canonical_sha256(source["hash_payload"])
    source["remote_output"] = (
        "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
        f"{STAGE.name}/arms/00_{source['name']}_{source['config_hash'][:12]}"
    )
    return source


def validate(spec: dict[str, Any]) -> None:
    from scripts.run_pre2_spooled_arm import _validate_spec

    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        _validate_spec(spec)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
    forbidden = {
        "execution_obstacle_speed_weight",
        "execution_cost_band_fraction",
        "execution_cost_band_margin",
    }
    if forbidden.intersection(spec["evaluation"]):
        raise ValueError("expansion-only selector leaked into raw evaluation")


def enqueue(spec: dict[str, Any]) -> dict[str, Any]:
    from scripts.submit_pre2_margin_spooler_sweep import (
        HELIOS_HOST,
        _atomic_json,
        _enqueue_spec,
    )

    queue = STAGE / "QUEUE.json"
    if queue.exists():
        raise RuntimeError(f"refusing duplicate enqueue: {queue}")
    subprocess.run([
        "ssh", HELIOS_HOST,
        f"test -S {shlex.quote(SOCKET)} && "
        f"test ! -e {shlex.quote(spec['remote_output'])}",
    ], check=True)
    local_specs = STAGE / "specs"
    local_specs.mkdir(parents=True, exist_ok=True)
    local_spec = local_specs / f"00_{spec['config_hash'][:12]}.json"
    _atomic_json(local_spec, spec)
    remote_specs = f"{spec['remote_control']}/specs"
    subprocess.run([
        "ssh", HELIOS_HOST,
        f"mkdir -p {shlex.quote(remote_specs)} "
        f"{shlex.quote(spec['remote_control'])}/logs "
        f"{shlex.quote(spec['remote_control'])}/locks "
        f"{shlex.quote(spec['remote_control'])}/status",
    ], check=True)
    remote_spec = f"{remote_specs}/00_{spec['config_hash'][:12]}.json"
    subprocess.run([
        "rsync", "-az", "-e", "ssh", str(local_spec),
        f"{HELIOS_HOST}:{remote_spec}",
    ], check=True)
    job_id = _enqueue_spec(spec, remote_spec, SOCKET)
    payload = {
        "schema_version": 1,
        "status": "ENQUEUED",
        "created_unix": time.time(),
        "design": "fresh PRE2 W300/2500 r0-r2",
        "rationale": (
            "Same optimizer exposure as W150/2500 isolates speed weight; "
            "r1 acquisition occurs before the first optimizer update."
        ),
        "physical_gpu": 1,
        "tsp_socket": SOCKET,
        "tsp_job_id": job_id,
        "config_hash": spec["config_hash"],
        "remote_spec": remote_spec,
        "remote_output": spec["remote_output"],
        "raw_evaluation_policy": (
            "fixed seed91000 native raw deployment with no speed cost, "
            "verifier, or margin selector"
        ),
    }
    _atomic_json(queue, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    spec = build_spec()
    validate(spec)
    if args.enqueue:
        if args.confirm != CONFIRM:
            parser.error(f"--enqueue requires --confirm {CONFIRM}")
        payload = enqueue(spec)
    else:
        payload = {
            "status": "LOCAL_VALIDATION_PASSED",
            "name": spec["name"],
            "rounds": spec["rounds"],
            "physical_gpu": spec["physical_gpu"],
            "scientific_delta": spec["scientific_delta"],
            "config_hash": spec["config_hash"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
