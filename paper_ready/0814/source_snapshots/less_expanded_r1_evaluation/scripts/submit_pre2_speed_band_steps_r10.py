#!/usr/bin/env python3
"""Enqueue the paired speed-weight x optimizer-exposure r0->r10 matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))
PILOT_STAGE = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed_band_pilot"
)
STAGE = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed_band_steps_r10"
)
MATRIX = STAGE / "STEP_MATRIX.json"
CONFIRM = "I_UNDERSTAND_THIS_ENQUEUES_THE_R10_WEIGHT_BY_STEPS_FACTORIAL"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build_specs() -> list[dict[str, Any]]:
    from scripts.promote_pre2_speed_band_r10 import _build_specs, _validate_gate

    _validate_gate(_read(PILOT_STAGE / "PILOT_RESULT.json"))
    return _build_specs(
        _read(MATRIX), _read(PILOT_STAGE / "QUEUE.json"), stage=STAGE,
    )


def enqueue() -> dict[str, Any]:
    from scripts.submit_pre2_margin_spooler_sweep import (
        HELIOS_HOST,
        _atomic_json,
        _enqueue_spec,
    )

    specs = build_specs()
    queue_path = STAGE / "QUEUE.json"
    if queue_path.exists():
        raise RuntimeError(f"refusing duplicate enqueue: {queue_path}")
    pilot_queue = _read(PILOT_STAGE / "QUEUE.json")
    sockets = {int(gpu): value for gpu, value in pilot_queue["tsp_sockets"].items()}
    for gpu, socket in sockets.items():
        slots = subprocess.check_output([
            "ssh", HELIOS_HOST, f"export TS_SOCKET={socket}; tsp -S",
        ], text=True).strip()
        if slots != "2":
            raise RuntimeError(f"GPU{gpu} shared spooler unavailable: {slots!r}")

    local_specs = STAGE / "specs"
    local_specs.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        _atomic_json(
            local_specs / f"{spec['matrix_index']:02d}_{spec['config_hash'][:12]}.json",
            spec,
        )
    remote_control = specs[0]["remote_control"]
    remote_specs = f"{remote_control}/specs"
    subprocess.run([
        "ssh", HELIOS_HOST,
        f"mkdir -p {remote_specs} {remote_control}/logs "
        f"{remote_control}/locks {remote_control}/status",
    ], check=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh", f"{local_specs}/",
        f"{HELIOS_HOST}:{remote_specs}/",
    ], check=True)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "ENQUEUE_IN_PROGRESS",
        "created_unix": time.time(),
        "matrix": str(MATRIX),
        "source_pilot_result": str(PILOT_STAGE / "PILOT_RESULT.json"),
        "source_id": pilot_queue["source_id"],
        "remote_source": pilot_queue["remote_source"],
        "remote_pretrain": pilot_queue["remote_pretrain"],
        "pretrained_sha256": pilot_queue["pretrained_sha256"],
        "task_config_sha256": pilot_queue["task_config_sha256"],
        "remote_control": remote_control,
        "tsp_sockets": {str(gpu): value for gpu, value in sockets.items()},
        "gpu_policy": "GPU1/GPU3 shared 2-slot queues; cross-over arm ordering",
        "raw_evaluation_policy": (
            "fixed seed91000 native raw deployment; no expansion selector or cost"
        ),
        "records": [],
    }
    _atomic_json(queue_path, payload)
    matrix = _read(MATRIX)
    for spec in specs:
        remote_spec = (
            f"{remote_specs}/{spec['matrix_index']:02d}_"
            f"{spec['config_hash'][:12]}.json"
        )
        job_id = _enqueue_spec(spec, remote_spec, sockets[spec["physical_gpu"]])
        payload["records"].append({
            "name": spec["name"],
            "role": matrix["arms"][spec["matrix_index"]]["role"],
            "physical_gpu": spec["physical_gpu"],
            "optimizer_steps_per_round": spec["scientific_delta"][
                "optimizer_steps_per_round"
            ],
            "execution_obstacle_speed_weight": spec["scientific_delta"][
                "execution_obstacle_speed_weight"
            ],
            "config_hash": spec["config_hash"],
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
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    specs = build_specs()
    if args.enqueue:
        if args.confirm != CONFIRM:
            parser.error(f"--enqueue requires --confirm {CONFIRM}")
        payload = enqueue()
    else:
        payload = {
            "status": "LOCAL_VALIDATION_PASSED",
            "arm_count": len(specs),
            "unique_config_hashes": len({spec["config_hash"] for spec in specs}),
            "order": [
                {
                    "name": spec["name"],
                    "gpu": spec["physical_gpu"],
                    **spec["scientific_delta"],
                }
                for spec in specs
            ],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
