#!/usr/bin/env python3
"""Launch the paired fresh-PRE2 q2 no-brake execution-cost sweep."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / (
    "results/stage1_single_ball_t128/0810_pre2_no_brake_cost_sweep"
)
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
TASK_CONFIG = ROOT / "configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json"
SEED = 82610
ARMS = (
    {
        "name": "q2_nb_base_c5_t120_w750_a5",
        "gpu": 1,
        "clearance_target": 0.55,
        "wall_weight": 750.0,
        "axis_weight": 5.0,
        "control_weight": 5.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "q2_nb_terminal160_c5_w750_a5",
        "gpu": 3,
        "clearance_target": 0.55,
        "wall_weight": 750.0,
        "axis_weight": 5.0,
        "control_weight": 5.0,
        "terminal_weight": 160.0,
    },
    {
        "name": "q2_nb_control10_t120_w750_a5",
        "gpu": 1,
        "clearance_target": 0.55,
        "wall_weight": 750.0,
        "axis_weight": 5.0,
        "control_weight": 10.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "q2_nb_wall5k_axis25_c5_t120",
        "gpu": 3,
        "clearance_target": 0.55,
        "wall_weight": 5000.0,
        "axis_weight": 25.0,
        "control_weight": 5.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "q2_nb_wall20k_axis100_c10_t160",
        "gpu": 1,
        "clearance_target": 0.55,
        "wall_weight": 20000.0,
        "axis_weight": 100.0,
        "control_weight": 10.0,
        "terminal_weight": 160.0,
    },
    {
        "name": "q2_nb_wall50k_axis250_c20_t240",
        "gpu": 3,
        "clearance_target": 0.50,
        "wall_weight": 50000.0,
        "axis_weight": 250.0,
        "control_weight": 20.0,
        "terminal_weight": 240.0,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _active(commands: str, output: Path) -> bool:
    return any(
        str(output) in command
        for command in commands.splitlines()
        if "RECIPE_control_braking_run_and_eval.sh" in command
        or "research_ball_expansion_optimization.py" in command
        or "research_evaluate_ball_expansion.py" in command
    )


def main() -> None:
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    records = []
    for arm in ARMS:
        output = STAGE / "arms" / arm["name"]
        raw_eval = output / "fixed_eval_r000_r005/raw_eval.json"
        if raw_eval.is_file() or _active(commands, output):
            records.append({
                **arm,
                "output": str(output),
                "status": "ALREADY_ACTIVE_OR_COMPLETE",
            })
            continue
        if output.exists() and any(output.iterdir()):
            records.append({
                **arm,
                "output": str(output),
                "status": "BLOCKED_NONEMPTY_INACTIVE_OUTPUT",
            })
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        log = STAGE / "logs" / f"{arm['name']}.screen.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({
            "PRE": str(PRETRAIN),
            "OUT": str(output),
            "HELIOS_GPU": str(arm["gpu"]),
            "ROUNDS": "5",
            "SEED": str(SEED),
            "CONTROL_WEIGHT": f"{arm['control_weight']:g}",
            "TERMINAL_WEIGHT": f"{arm['terminal_weight']:g}",
            "BRAKING_WEIGHT": "0",
            "FINITE_SEGMENT": "1",
            "TASKSPACE_WEIGHT": f"{arm['wall_weight']:g}",
            "TASKSPACE_TARGET": "0.19",
            "AXIS_WEIGHT": f"{arm['axis_weight']:g}",
            "AXIS_RADIUS": "0.8",
            "CLEARANCE_WEIGHT": "2500",
            "CLEARANCE_TARGET": f"{arm['clearance_target']:g}",
            "SUCCESS_QUOTA": "8",
            "SAMPLE_MODES": "0,0,1,1,2,2,3,3",
            "FAITHFUL_RETRY": "1",
        })
        stream = log.open("a")
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_control_braking_run_and_eval.sh"],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        records.append({
            **arm,
            "output": str(output),
            "log": str(log),
            "pid": process.pid,
            "status": "LAUNCHED",
            "launched_unix": time.time(),
        })
    _atomic_json(STAGE / "NO_BRAKE_COST_SWEEP.json", {
        "status": "FRESH_PRE2_Q2_NO_BRAKE_SWEEP_LAUNCHED",
        "source_calibration": str(
            ROOT / (
                "results/stage1_single_ball_t128/"
                "0810_pre2_control_braking_sweep/"
                "NO_BRAKE_SELECTED_B_CALIBRATION.json"
            )
        ),
        "fixed": {
            "pretrain": str(PRETRAIN),
            "pretrained_sha256": _sha256(PRETRAIN / "pretrained.pt"),
            "task_config": str(TASK_CONFIG),
            "task_config_sha256": _sha256(TASK_CONFIG),
            "taskspace_bounds": {
                "x": [-2.5, 1.3],
                "y": [-1.7, 1.8],
                "z": [0.1, 1.7],
            },
            "modeled_sphere_radius_m": 0.2905,
            "seed": SEED,
            "rounds": 5,
            "braking_weight": 0.0,
            "axis_radius_m": 0.8,
            "finite_segment": True,
            "wall_target_m": 0.19,
            "clearance_weight": 2500.0,
            "successful_trajectories_per_gamma": 8,
            "sample_update_mode": [0, 0, 1, 1, 2, 2, 3, 3],
            "K": 16,
            "B": 8,
            "retry_B": 8,
            "retry_verify_all_fast_path": False,
            "beta": 0.1,
            "gp_reference_mode": "sliding_success_per_gamma_current_phi",
            "acquisition_feature": "learned_phi",
            "fresh_round1_gp_caveat": (
                "no previous successful GP rows; learned success-buffer "
                "tilting begins in round 2"
            ),
            "fixed_eval": {
                "rounds": [0, 1, 2, 3, 4, 5],
                "seed": 91000,
                "episodes_per_gamma": 40,
                "flow_nfe": 12,
                "gallery": "head_on",
            },
        },
        "winner_rule": {
            "primary": "OOB must approach zero",
            "roundwise": [
                "SR nondecreasing preference",
                "CR nonincreasing preference",
                "OOB nonincreasing preference",
                "validity nondecreasing preference",
            ],
            "gamma_trend": (
                "high-gamma group must preserve lower TtG and lower "
                "successful minimum clearance than low-gamma group"
            ),
            "coverage": "4/4 required; retain per-mode counts",
        },
        "arms": records,
    })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
