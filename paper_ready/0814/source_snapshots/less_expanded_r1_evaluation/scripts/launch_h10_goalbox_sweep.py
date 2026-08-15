#!/usr/bin/env python3
"""Launch the paired brake-free H10/stop/goal-box causal sweep."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage1_single_ball_t128/0811_pre2_h10_goalbox_sweep"
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
TASK_CONFIG = ROOT / "configs/lab_ball_stage1_relaxed_z01_17_r15in_v1.json"
LEGACY = ROOT / (
    "results/stage1_single_ball_t128/0810_pre2_control_braking_sweep/"
    "arms/w5_c1_cap_b100"
)
SEED = 82410
ARMS = (
    {
        "name": "h10_stop02_goalbox_w050_t100", "gpu": 3,
        "weight": 50.0, "stopping_margin_m": 0.02,
    },
    {
        "name": "h10_stop02_goalbox_w100_t100", "gpu": 1,
        "weight": 100.0, "stopping_margin_m": 0.02,
    },
    {
        "name": "h10_stop02_goalbox_w250_t100", "gpu": 1,
        "weight": 250.0, "stopping_margin_m": 0.02,
    },
    {
        "name": "h10_only_goalbox_w050_t100", "gpu": 3,
        "weight": 50.0, "stopping_margin_m": None,
    },
    {
        "name": "h10_only_goalbox_w100_t100", "gpu": 1,
        "weight": 100.0, "stopping_margin_m": None,
    },
    {
        "name": "h10_only_goalbox_w250_t100", "gpu": 1,
        "weight": 250.0, "stopping_margin_m": None,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
            "CONTROL_WEIGHT": "1",
            "TERMINAL_WEIGHT": "80",
            "BRAKING_WEIGHT": "0",
            "FINITE_SEGMENT": "1",
            "TASKSPACE_WEIGHT": "0",
            "TASKSPACE_TARGET": "0.15",
            "GOAL_BOX_WEIGHT": f"{arm['weight']:g}",
            "GOAL_BOX_HALF_EXTENT": "0.2",
            "GOAL_BOX_TEMPERATURE": "1.0",
            "FULL_H_TASKSPACE": "1",
            "STOPPING_MARGIN": (
                "" if arm["stopping_margin_m"] is None
                else f"{arm['stopping_margin_m']:g}"
            ),
            "AXIS_WEIGHT": "5",
            "AXIS_RADIUS": "1.1",
            "CLEARANCE_WEIGHT": "2500",
            "CLEARANCE_TARGET": "0.6",
            "SUCCESS_QUOTA": "12",
            "SAMPLE_MODES": "0,0,0,1,1,1,2,2,2,3,3,3",
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

    calibration = STAGE / "SAVED_GP_B8_CALIBRATION.json"
    _atomic_json(STAGE / "SWEEP.json", {
        "status": "PAIRED_FRESH_PRE2_H10_GOALBOX_SWEEP_LAUNCHED",
        "launched_unix": time.time(),
        "fixed": {
            "pretrain": str(PRETRAIN),
            "pretrained_sha256": _sha256(PRETRAIN / "pretrained.pt"),
            "task_config": str(TASK_CONFIG),
            "task_config_sha256": _sha256(TASK_CONFIG),
            "legacy_brake100_r5": str(
                LEGACY / "fixed_eval_r000_r005/raw_eval.json"
            ),
            "legacy_checkpoint5_sha256": _sha256(
                LEGACY / "checkpoint_005.pt"
            ),
            "saved_gp_b8_calibration": str(calibration),
            "saved_gp_b8_calibration_sha256": (
                _sha256(calibration) if calibration.is_file() else None
            ),
            "stopping_backup_diagnosis": str(
                STAGE / "STOPPING_BACKUP_DIAGNOSIS.json"
            ),
            "seed": SEED,
            "rounds": 5,
            "physical_taskspace_unchanged": {
                "x": [-2.5, 1.3],
                "y": [-1.7, 1.8],
                "z": [0.1, 1.7],
            },
            "goal_centered_box": {
                "lower": [0.5, -1.7, 0.7],
                "upper": [0.9, -1.3, 1.1],
                "half_extent_m": 0.2,
                "temperature_m": 1.0,
            },
            "verifier_full_h_taskspace": True,
            "verifier_taskspace_stopping_designs": {
                "terminal_backup": 0.02,
                "full_h_only": None,
            },
            "braking_weight": 0.0,
            "execution_taskspace_quadratic_weight": 0.0,
            "clearance_quadratic": {"weight": 2500.0, "target_m": 0.6},
            "axis_cylinder": {
                "weight": 5.0, "radius_m": 1.1, "finite_segment": True,
            },
            "control_weight": 1.0,
            "terminal_goal_weight": 80.0,
            "successful_trajectories_per_gamma": 12,
            "sample_update_mode": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "K": 16,
            "B": 8,
            "retry_B": 8,
            "retry_verify_all_fast_path": False,
            "beta": 0.1,
            "gp_reference_mode": "sliding_success_per_gamma_current_phi",
            "optimizer_steps_per_round": 2500,
            "fixed_eval": {
                "rounds": [0, 1, 2, 3, 4, 5],
                "seed": 91000,
                "episodes_per_gamma": 40,
                "flow_nfe": 12,
                "gallery": "head_on",
            },
            "gpu_policy": "GPU0/GPU2 prohibited; shared GPU1/GPU3 only",
        },
        "causal_intervention": (
            "paired 2x3 design: full-H task-space with versus without the "
            "terminal task-space stopping backup, crossed with goal-box "
            "exponential weights 50/100/250; every other setting is fixed"
        ),
        "selection_unblock_rule": (
            "the three full-H-only arms may select and promote a winner "
            "without waiting for stopping-backup arms whose episode-level "
            "terminal-success support remains effectively absent"
        ),
        "winner_rule": {
            "hard_preferences": [
                "coverage 4/4",
                "OOB approaches zero",
                "SR increases and collision decreases from PRE2",
                "validity increases",
            ],
            "gamma_trend": (
                "higher gamma should retain lower TtG and lower successful "
                "minimum clearance than lower gamma"
            ),
            "continuation": (
                "winner resumes exact state in five-round chunks through at "
                "least r15 while improving, with full fixed-bank evaluation "
                "at every multiple of five"
            ),
        },
        "arms": records,
    })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
