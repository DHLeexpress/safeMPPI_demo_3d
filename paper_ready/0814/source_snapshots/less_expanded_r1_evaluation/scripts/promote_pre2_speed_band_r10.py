#!/usr/bin/env python3
"""Enqueue the evidence-gated paired fresh-PRE2 speed-band r0->r10 run."""
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
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed_band_r10"
)
MATRIX = STAGE / "R10_MATRIX.json"
CONFIRM = "I_UNDERSTAND_THIS_ENQUEUES_THE_PAIRED_R10_SPEED_BAND_RUN"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_gate(result: dict[str, Any]) -> None:
    if result.get("decision") != "PROMOTE_SPEED150_WITH_PAIRED_SPEED0_CONTROL":
        raise ValueError("pilot result does not authorize this promotion")
    control = result["r2"]["speed000"]
    treatment = result["r2"]["speed150"]
    required = (
        treatment["SR"] > control["SR"]
        and treatment["CR"] < control["CR"]
        and treatment["OOB"] < control["OOB"]
        and treatment["validity"] > control["validity"]
        and treatment["timeout"] == 0.0
        and treatment["time_to_goal_s"] - control["time_to_goal_s"] < 1.0
        and treatment["coverage"] >= control["coverage"]
    )
    if not required:
        raise ValueError("speed150 no longer satisfies the recorded promotion gate")


def _build_specs(
    matrix: dict[str, Any], pilot_queue: dict[str, Any], *, stage: Path = STAGE,
) -> list[dict[str, Any]]:
    from scripts.run_pre2_spooled_arm import _canonical_sha256
    from scripts.submit_pre2_speed_band_pilot import (
        TASK_CONFIG,
        _scientific_args,
    )

    arms = matrix.get("arms", [])
    if matrix.get("paired_seed") != 82410 or not 2 <= len(arms) <= 4:
        raise ValueError("R10 matrix must contain 2..4 paired arms at seed82410")
    weights = [float(arm["execution_obstacle_speed_weight"]) for arm in matrix["arms"]]
    if not set(weights).issubset({0.0, 150.0}) or 150.0 not in weights:
        raise ValueError("R10 matrix may use only speed0/speed150 and needs speed150")

    rounds = 10
    evaluation = {
        "episodes_per_gamma": 40,
        "probe_samples": 16,
        "flow_nfe": 12,
        "seed": 91000,
        "fixed_scene_rollouts": 10,
        "fixed_scene_seed": 191000,
    }
    remote_stage = (
        "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
        f"{stage.name}"
    )
    remote_source = pilot_queue["remote_source"]
    specs = []
    for arm in matrix["arms"]:
        index = int(arm["index"])
        args = _scientific_args(arm)
        optimizer_steps = int(arm.get("optimizer_steps_per_round", 2500))
        if optimizer_steps not in {250, 2500}:
            raise ValueError("optimizer steps must be 250 or 2500")
        args[args.index("--optimizer-steps-per-round") + 1] = str(
            optimizer_steps
        )
        hash_payload = {
            "source_id": pilot_queue["source_id"],
            "pretrained_sha256": pilot_queue["pretrained_sha256"],
            "task_config_sha256": pilot_queue["task_config_sha256"],
            "rounds": rounds,
            "expansion_args": args,
            "evaluation": evaluation,
        }
        config_hash = _canonical_sha256(hash_payload)
        specs.append({
            "schema_version": 1,
            "name": arm["name"],
            "matrix_index": index,
            "physical_gpu": int(arm.get("physical_gpu", 1 if index % 2 == 0 else 3)),
            "rounds": rounds,
            "source_id": pilot_queue["source_id"],
            "remote_source": remote_source,
            "remote_pretrain": pilot_queue["remote_pretrain"],
            "pretrained_sha256": pilot_queue["pretrained_sha256"],
            "remote_task_config": f"{remote_source}/{TASK_CONFIG.relative_to(ROOT)}",
            "task_config_sha256": pilot_queue["task_config_sha256"],
            "remote_output": (
                f"{remote_stage}/arms/{index:02d}_{arm['name']}_{config_hash[:12]}"
            ),
            "remote_control": f"{remote_stage}/control",
            "expansion_args": args,
            "evaluation": evaluation,
            "hash_payload": hash_payload,
            "config_hash": config_hash,
            "scientific_delta": {
                "execution_obstacle_speed_weight": arm[
                    "execution_obstacle_speed_weight"
                ],
                "execution_cost_band_fraction": 0.05,
                "execution_cost_band_margin": "step",
                "optimizer_steps_per_round": optimizer_steps,
            },
        })
    return specs


def enqueue() -> dict[str, Any]:
    from scripts.submit_pre2_margin_spooler_sweep import (
        HELIOS_HOST,
        _atomic_json,
        _enqueue_spec,
    )

    result = _load_json(PILOT_STAGE / "PILOT_RESULT.json")
    _validate_gate(result)
    pilot_queue = _load_json(PILOT_STAGE / "QUEUE.json")
    matrix = _load_json(MATRIX)
    specs = _build_specs(matrix, pilot_queue)
    queue_path = STAGE / "QUEUE.json"
    if queue_path.exists():
        raise RuntimeError(f"refusing duplicate promotion: {queue_path}")

    sockets = {int(gpu): socket for gpu, socket in pilot_queue["tsp_sockets"].items()}
    for gpu, socket in sockets.items():
        slots = subprocess.check_output([
            "ssh", HELIOS_HOST,
            f"export TS_SOCKET={socket}; tsp -S",
        ], text=True).strip()
        if slots != "2":
            raise RuntimeError(f"GPU{gpu} shared spooler is unavailable: slots={slots!r}")

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
        "tsp_sockets": {str(gpu): socket for gpu, socket in sockets.items()},
        "gpu_policy": "GPU1/GPU3 shared spoolers; one paired arm per GPU",
        "raw_evaluation_policy": (
            "fixed seed91000 native raw deployment; no expansion selector or cost"
        ),
        "records": [],
    }
    _atomic_json(queue_path, payload)
    for spec in specs:
        remote_spec = (
            f"{remote_specs}/{spec['matrix_index']:02d}_"
            f"{spec['config_hash'][:12]}.json"
        )
        job_id = _enqueue_spec(spec, remote_spec, sockets[spec["physical_gpu"]])
        payload["records"].append({
            "name": spec["name"],
            "role": matrix["arms"][spec["matrix_index"]]["role"],
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
    _atomic_json(PILOT_STAGE / "R10_PROMOTION.json", {
        "status": "R10_PROMOTED",
        "promoted_unix": time.time(),
        "treatment": "speed150",
        "paired_control": "speed0",
        "target_round": 10,
        "fresh_pre2": True,
        "queue": str(queue_path),
        "raw_evaluation_execution_selector": False,
        "records": payload["records"],
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    result = _load_json(PILOT_STAGE / "PILOT_RESULT.json")
    _validate_gate(result)
    matrix = _load_json(MATRIX)
    specs = _build_specs(matrix, _load_json(PILOT_STAGE / "QUEUE.json"))
    if args.enqueue:
        if args.confirm != CONFIRM:
            parser.error(f"--enqueue requires --confirm {CONFIRM}")
        payload = enqueue()
    else:
        payload = {
            "status": "LOCAL_VALIDATION_PASSED",
            "arm_count": len(specs),
            "rounds": 10,
            "weights": [
                spec["scientific_delta"]["execution_obstacle_speed_weight"]
                for spec in specs
            ],
            "unique_config_hashes": len({spec["config_hash"] for spec in specs}),
            "source_id": specs[0]["source_id"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
