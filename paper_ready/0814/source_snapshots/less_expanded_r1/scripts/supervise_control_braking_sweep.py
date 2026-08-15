#!/usr/bin/env python3
"""Keep the paired r0-r5 sweep alive and promote its trend winner to q1/r15."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage1_single_ball_t128/0810_pre2_control_braking_sweep"
CONFIG = STAGE / "SWEEP.json"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _commands() -> str:
    return subprocess.run(
        ["ps", "-ax", "-o", "command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _active(commands: str, output: Path) -> bool:
    lock_pid = Path(f"{output}.control_braking_process.lock/pid")
    if lock_pid.is_file():
        try:
            os.kill(int(lock_pid.read_text().strip()), 0)
            return True
        except (OSError, ValueError):
            pass
    target = str(output.resolve())
    return any(
        target in command
        and (
            "RECIPE_control_braking_run_and_eval.sh" in command
            or "research_ball_expansion_optimization.py" in command
            or "research_evaluate_ball_expansion.py" in command
        )
        for command in commands.splitlines()
    )


def _launch_evaluation(
    *, name: str, pretrain: Path, output: Path, gpu: int, rounds: int,
) -> dict:
    if gpu not in {1, 3}:
        raise ValueError("only GPU1/GPU3 are allowed")
    evaluation = output / f"fixed_eval_r000_r{rounds:03d}"
    log = STAGE / "logs" / f"{name}.evaluation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    child_environment.update({
        "PRE": str(pretrain),
        "EXPANSION": str(output),
        "EVAL_OUT": str(evaluation),
        "HELIOS_GPU": str(gpu),
        "TARGET_ROUND": str(rounds),
        "EVALUATION_ROUNDS": " ".join(str(index) for index in range(rounds + 1)),
    })
    stream = log.open("a")
    process = subprocess.Popen(
        ["bash", "scripts/RECIPE_r10_taskspace_eval.sh"],
        cwd=ROOT,
        env=child_environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    return {
        "status": "EVALUATION_LAUNCHED",
        "name": name,
        "pid": process.pid,
        "detached": True,
        "gpu": gpu,
        "output": str(output),
        "evaluation": str(evaluation),
        "log": str(log),
        "rounds": rounds,
        "launched_unix": time.time(),
    }


def _launch(
    *,
    name: str,
    pretrain: Path,
    output: Path,
    gpu: int,
    rounds: int,
    seed: int,
    control_weight: float,
    braking_weight: float,
    finite_segment: bool,
    quota: int,
    modes: list[int],
    faithful_retry: bool,
) -> dict:
    if gpu not in {1, 3}:
        raise ValueError("only GPU1/GPU3 are allowed")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = STAGE / "logs" / f"{name}.screen.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "PRE": str(pretrain),
        "OUT": str(output),
        "HELIOS_GPU": str(gpu),
        "ROUNDS": str(rounds),
        "SEED": str(seed),
        "CONTROL_WEIGHT": f"{control_weight:g}",
        "BRAKING_WEIGHT": f"{braking_weight:g}",
        "FINITE_SEGMENT": "1" if finite_segment else "0",
        "SUCCESS_QUOTA": str(quota),
        "SAMPLE_MODES": ",".join(str(mode) for mode in modes),
        "FAITHFUL_RETRY": "1" if faithful_retry else "0",
    }
    child_environment = os.environ.copy()
    child_environment.update(environment)
    stream = log.open("a")
    process = subprocess.Popen(
        ["bash", "scripts/RECIPE_control_braking_run_and_eval.sh"],
        cwd=ROOT,
        env=child_environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    return {
        "status": "LAUNCHED",
        "name": name,
        "pid": process.pid,
        "detached": True,
        "gpu": gpu,
        "output": str(output),
        "log": str(log),
        "rounds": rounds,
        "seed": seed,
        "control_weight": control_weight,
        "braking_weight": braking_weight,
        "finite_segment": finite_segment,
        "successful_trajectories_per_gamma": quota,
        "sample_update_mode": modes,
        "faithful_retry": faithful_retry,
        "launched_unix": time.time(),
    }


def _metric(summary: dict, key: str) -> float:
    value = summary.get(key)
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _score(raw_eval: Path) -> dict:
    payload = json.loads(raw_eval.read_text())
    summaries = payload["summary"]
    rounds = [summaries[str(index)]["pooled"] for index in range(6)]
    sr = [_metric(row, "SR") for row in rounds]
    cr = [_metric(row, "CR") for row in rounds]
    oob = [_metric(row, "OOB") for row in rounds]
    validity = [_metric(row, "window_validity") for row in rounds]
    clearance = [_metric(row, "successful_min_clearance_m") for row in rounds]
    ttg = [_metric(row, "successful_time_to_goal_s") for row in rounds]
    coverage = [_metric(row, "route_coverage") for row in rounds]

    def wrong_increase(values: list[float]) -> float:
        return float(sum(max(values[i + 1] - values[i], 0.0) for i in range(5)))

    def wrong_decrease(values: list[float]) -> float:
        return float(sum(max(values[i] - values[i + 1], 0.0) for i in range(5)))

    wrong_mass = (
        2.0 * wrong_decrease(sr)
        + wrong_increase(cr)
        + 1.5 * wrong_increase(oob)
        + wrong_decrease(validity)
    )
    favorable_transitions = {
        "SR_non_decreasing": sum(sr[i + 1] >= sr[i] - 0.0125 for i in range(5)),
        "CR_non_increasing": sum(cr[i + 1] <= cr[i] + 0.0125 for i in range(5)),
        "OOB_non_increasing": sum(oob[i + 1] <= oob[i] + 0.0125 for i in range(5)),
        "validity_non_decreasing": sum(
            validity[i + 1] >= validity[i] - 0.005 for i in range(5)
        ),
        "TtG_non_decreasing": sum(ttg[i + 1] >= ttg[i] - 0.05 for i in range(5)),
    }
    endpoint = rounds[-1]
    total = (
        8.0 * sr[-1]
        - 3.0 * cr[-1]
        - 4.0 * oob[-1]
        + 2.0 * validity[-1]
        + 1.5 * coverage[-1]
        + 2.0 * (sr[-1] - sr[0])
        + 1.0 * (cr[0] - cr[-1])
        - 4.0 * wrong_mass
    )
    return {
        "score": float(total),
        "wrong_direction_mass": wrong_mass,
        "favorable_transitions": favorable_transitions,
        "series": {
            "SR": sr,
            "CR": cr,
            "OOB": oob,
            "validity": validity,
            "clearance_m": clearance,
            "time_to_goal_s": ttg,
            "coverage": coverage,
        },
        "r5": endpoint,
    }


def _choose_gpu(commands: str) -> int:
    loads = {}
    for gpu in (1, 3):
        loads[gpu] = sum(
            f"--helios-gpu {gpu}" in line
            for line in commands.splitlines()
            if "research_" in line
        )
    return min((1, 3), key=lambda gpu: (loads[gpu], gpu))


def _paper_summary(raw_eval: Path) -> dict:
    payload = json.loads(raw_eval.read_text())
    summaries = payload["summary"]
    return {
        "status": "Q1_FAITHFUL_R15_EVALUATION_COMPLETE",
        "rounds": {
            round_index: summaries[round_index]["pooled"]
            for round_index in sorted(summaries, key=int)
        },
        "per_gamma_r15": summaries["15"]["per_gamma"],
        "raw_eval": str(raw_eval),
        "completed_unix": time.time(),
    }


def tick(config: dict, forced_winner_name: str | None = None) -> bool:
    commands = _commands()
    pretrain = Path(config["pretrain"])
    launch_records = []
    complete = []
    arms = config["arms"]
    if forced_winner_name is not None:
        arms = [arm for arm in arms if arm["name"] == forced_winner_name]
        if not arms:
            raise ValueError(f"unknown forced winner: {forced_winner_name}")
    for arm in arms:
        output = STAGE / "arms" / arm["name"]
        raw_eval = output / "fixed_eval_r000_r005/raw_eval.json"
        if raw_eval.is_file():
            complete.append((arm, output, raw_eval))
            continue
        if not _active(commands, output):
            if (output / "checkpoint_005.pt").is_file():
                record = _launch_evaluation(
                    name=arm["name"],
                    pretrain=pretrain,
                    output=output,
                    gpu=int(arm["gpu"]),
                    rounds=5,
                )
                marker = STAGE / "launches" / f"{arm['name']}.evaluation.json"
            else:
                record = _launch(
                    name=arm["name"],
                    pretrain=pretrain,
                    output=output,
                    gpu=int(arm["gpu"]),
                    rounds=5,
                    seed=int(config["seed"]),
                    control_weight=float(arm["control_weight"]),
                    braking_weight=float(arm["braking_weight"]),
                    finite_segment=bool(arm["finite_segment"]),
                    quota=12,
                    modes=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                    faithful_retry=False,
                )
                marker = STAGE / "launches" / f"{arm['name']}.json"
            _atomic_json(marker, record)
            launch_records.append(record)

    if len(complete) != len(arms):
        _atomic_json(STAGE / "SUPERVISOR_STATUS.json", {
            "status": "SHORT_SWEEP_RUNNING",
            "completed_arms": [arm["name"] for arm, _, _ in complete],
            "total_arms": len(arms),
            "new_launches": launch_records,
            "checked_unix": time.time(),
        })
        return False

    scored = []
    for arm, output, raw_eval in complete:
        scored.append({
            "arm": arm,
            "output": str(output),
            "raw_eval": str(raw_eval),
            **_score(raw_eval),
        })
    scored.sort(key=lambda row: row["score"], reverse=True)
    winner = scored[0]
    _atomic_json(STAGE / "SWEEP_RESULT.json", {
        "status": (
            "USER_SELECTED_WINNER"
            if forced_winner_name is not None
            else "SHORT_SWEEP_COMPLETE_TREND_WINNER_SELECTED"
        ),
        "winner": winner,
        "ranking": scored,
        "selection_rule": (
            "explicit user selection from completed r5 fixed-bank evidence"
            if forced_winner_name is not None
            else (
                "r5 SR/CR/OOB/validity/coverage plus wrong-direction mass across "
                "every r0-r5 transition; TtG is reported but not a hard cutoff"
            )
        ),
        "selected_unix": time.time(),
    })

    winner_arm = winner["arm"]
    paper_name = f"paper_q1_faithful_{winner_arm['name']}_s82510"
    paper_output = STAGE / paper_name
    paper_eval = paper_output / "fixed_eval_r000_r015/raw_eval.json"
    if paper_eval.is_file():
        _atomic_json(STAGE / "PAPER_RUN_COMPLETE.json", _paper_summary(paper_eval))
        return True
    commands = _commands()
    if not _active(commands, paper_output):
        gpu = _choose_gpu(commands)
        record = _launch(
            name=paper_name,
            pretrain=pretrain,
            output=paper_output,
            gpu=gpu,
            rounds=15,
            seed=82510,
            control_weight=float(winner_arm["control_weight"]),
            braking_weight=float(winner_arm["braking_weight"]),
            finite_segment=bool(winner_arm["finite_segment"]),
            quota=4,
            modes=[0, 1, 2, 3],
            faithful_retry=True,
        )
        _atomic_json(STAGE / "PAPER_RUN_LAUNCHED.json", {
            **record,
            "source_sweep_winner": winner_arm["name"],
            "fresh_pre2_start": True,
            "retry_semantics": (
                "ordinary GP uncertainty-tilted B8-of-K16; "
                "retry_verify_all_fast_path absent"
            ),
        })
    _atomic_json(STAGE / "SUPERVISOR_STATUS.json", {
        "status": "Q1_FAITHFUL_R15_RUNNING",
        "winner": winner_arm["name"],
        "paper_output": str(paper_output),
        "checked_unix": time.time(),
    })
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--winner", help="promote this completed r5 arm immediately")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    while True:
        if tick(config, forced_winner_name=args.winner) or args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
