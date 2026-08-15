#!/usr/bin/env python3
"""Wait for the live brake100 paper checkpoint 5, evaluate r0-r5 once, and exit."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage1_single_ball_t128/0810_pre2_control_braking_sweep"
EXPANSION = STAGE / "paper_q1_faithful_w5_c1_cap_b100_s82510"
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
EVALUATION = EXPANSION / "fixed_eval_r000_r005"
LOCK = STAGE / "paper_r5_evaluation_waiter.lock"
REFERENCE = STAGE / "arms/w5_c1_cap_b100/fixed_eval_r000_r005/raw_eval.json"
COMPARISON = STAGE / "PAPER_R5_TREND_COMPARISON.json"


def publish_comparison() -> None:
    reference = json.loads(REFERENCE.read_text())
    fresh = json.loads(EVALUATION.joinpath("raw_eval.json").read_text())
    metrics = (
        "SR", "CR", "OOB", "timeout", "window_validity",
        "successful_min_clearance_m", "successful_time_to_goal_s",
        "route_coverage",
    )
    reference_series = {
        metric: [reference["summary"][str(index)]["pooled"][metric] for index in range(6)]
        for metric in metrics
    }
    fresh_series = {
        metric: [fresh["summary"][str(index)]["pooled"][metric] for index in range(6)]
        for metric in metrics
    }
    checks = {
        "r5_SR_above_r0": fresh_series["SR"][-1] > fresh_series["SR"][0],
        "r5_CR_below_r0": fresh_series["CR"][-1] < fresh_series["CR"][0],
        "r5_validity_above_r0": (
            fresh_series["window_validity"][-1] > fresh_series["window_validity"][0]
        ),
        "r5_TtG_not_faster_than_r0": (
            fresh_series["successful_time_to_goal_s"][-1]
            >= fresh_series["successful_time_to_goal_s"][0]
        ),
        "r5_coverage_4_of_4": fresh_series["route_coverage"][-1] == 1.0,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "interpretation": (
            "The fresh q1 faithful run is required to reproduce the directional "
            "r0-r5 behavior, not the exact numeric sequence of the q3 fast-path run."
        ),
        "checks": checks,
        "reference_series": reference_series,
        "fresh_series": fresh_series,
        "reference_raw_eval": str(REFERENCE),
        "fresh_raw_eval": str(EVALUATION / "raw_eval.json"),
    }
    temporary = COMPARISON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(COMPARISON)


def remote_output() -> str | None:
    commands = subprocess.run(
        ["ssh", "helios.robotics.caltech.edu", "ps -eo args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for command in commands.splitlines():
        if EXPANSION.name not in command or "research_ball_expansion_optimization.py" not in command:
            continue
        tokens = shlex.split(command)
        if "--output" in tokens:
            return tokens[tokens.index("--output") + 1]
    return None


def main() -> None:
    if EVALUATION.joinpath("raw_eval.json").is_file():
        publish_comparison()
        return
    try:
        LOCK.mkdir()
    except FileExistsError:
        raise SystemExit("r5 evaluation waiter already exists")
    try:
        remote = None
        while not EVALUATION.joinpath("raw_eval.json").is_file():
            remote = remote or remote_output()
            if remote is not None:
                ready = subprocess.run(
                    [
                        "ssh",
                        "helios.robotics.caltech.edu",
                        f"test -s {shlex.quote(remote + '/checkpoint_005.pt')}",
                    ],
                    check=False,
                ).returncode == 0
                if ready:
                    break
            time.sleep(60)
        if EVALUATION.joinpath("raw_eval.json").is_file():
            publish_comparison()
            return
        EXPANSION.mkdir(parents=True, exist_ok=True)
        metadata = EXPANSION / ".helios_remote.json"
        if not metadata.is_file():
            metadata.write_text(json.dumps({
                "status": "HELIOS_REMOTE_EXPANSION_IN_PROGRESS_R5_MILESTONE",
                "host": "helios.robotics.caltech.edu",
                "physical_gpu": 1,
                "gpu_policy": "shared",
                "detached": True,
                "remote_output": remote,
            }, indent=2) + "\n")
        environment = os.environ.copy()
        environment.update({
            "PRE": str(PRETRAIN),
            "EXPANSION": str(EXPANSION),
            "EVAL_OUT": str(EVALUATION),
            "HELIOS_GPU": "3",
            "TARGET_ROUND": "5",
            "EVALUATION_ROUNDS": "0 1 2 3 4 5",
        })
        subprocess.run(
            ["bash", "scripts/RECIPE_r10_taskspace_eval.sh"],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        publish_comparison()
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
