#!/usr/bin/env python3
"""Launch paired brake100 r5 backups with stronger wall/axis penalties."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage1_single_ball_t128/0810_pre2_control_braking_sweep"
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
ARMS = (
    {
        "name": "backup_brake100_wall750_axis7p5_s82510",
        "gpu": 3,
        "taskspace_weight": 750.0,
        "axis_weight": 7.5,
    },
    {
        "name": "backup_brake100_wall1000_axis10_s82510",
        "gpu": 1,
        "taskspace_weight": 1000.0,
        "axis_weight": 10.0,
    },
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def active(commands: str, output: Path) -> bool:
    return any(
        str(output) in command
        for command in commands.splitlines()
        if "RECIPE_control_braking_run_and_eval.sh" in command
        or "research_ball_expansion_optimization.py" in command
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
        output = STAGE / "backups" / arm["name"]
        raw_eval = output / "fixed_eval_r000_r005/raw_eval.json"
        if raw_eval.is_file() or active(commands, output):
            records.append({**arm, "output": str(output), "status": "ALREADY_ACTIVE_OR_COMPLETE"})
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        log = STAGE / "logs" / f"{arm['name']}.screen.log"
        environment = os.environ.copy()
        environment.update({
            "PRE": str(PRETRAIN),
            "OUT": str(output),
            "HELIOS_GPU": str(arm["gpu"]),
            "ROUNDS": "5",
            "SEED": "82510",
            "CONTROL_WEIGHT": "1",
            "BRAKING_WEIGHT": "100",
            "FINITE_SEGMENT": "1",
            "TASKSPACE_WEIGHT": f"{arm['taskspace_weight']:g}",
            "AXIS_WEIGHT": f"{arm['axis_weight']:g}",
            "SUCCESS_QUOTA": "4",
            "SAMPLE_MODES": "0,1,2,3",
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
    atomic_json(STAGE / "BACKUP_WALL_AXIS_SWEEP.json", {
        "status": "PAIRED_BRAKE100_R5_BACKUPS_LAUNCHED",
        "baseline": {
            "name": "paper_q1_faithful_w5_c1_cap_b100_s82510",
            "taskspace_weight": 500.0,
            "axis_weight": 5.0,
        },
        "fixed": {
            "fresh_pre2": str(PRETRAIN),
            "seed": 82510,
            "rounds": 5,
            "control_weight": 1.0,
            "braking_weight": 100.0,
            "finite_segment": True,
            "successful_trajectories_per_gamma": 4,
            "sample_update_mode": [0, 1, 2, 3],
            "K": 16,
            "B": 8,
            "retry_B": 8,
            "faithful_retry": True,
            "retry_verify_all_fast_path": False,
            "fixed_eval_seed": 91000,
            "flow_nfe": 12,
        },
        "arms": records,
    })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
