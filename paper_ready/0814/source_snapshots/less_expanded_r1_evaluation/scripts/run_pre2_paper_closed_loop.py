#!/usr/bin/env python3
"""Detached PRE2 expansion/evaluation closed loop for GPUs 0, 1, and 3."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
RESULTS = ROOT / "results/stage1_single_ball_t128"
PRE = RESULTS / "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
OUT = RESULTS / "0810_pre2_paper_closed_loop"
GPU_ORDER = (0, 1, 3)


@dataclass(frozen=True)
class Arm:
    label: str
    execution: str
    scope: str
    gpu: int
    seed: int
    beta: float = 0.1
    flow_std: float = 1.0
    flow_std_final: float = 1.0
    flow_schedule: str = "none"
    retry_cap: int = 64


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _status(phase: str, **extra) -> None:
    _write(OUT / "STATUS.json", {
        "updated_unix": time.time(), "phase": phase, **extra,
    })


def _manifest_round(path: Path) -> int | None:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    return int(json.loads(manifest.read_text())["config"]["rounds"])


def _run_expansions(stage: str, arms: list[Arm], target_round: int) -> list[Arm]:
    running = []
    complete = []
    for arm in arms:
        arm_out = OUT / stage / arm.label
        if _manifest_round(arm_out) == target_round:
            complete.append(arm)
            continue
        if arm_out.exists() and not (arm_out / "resume_state_latest.pt").is_file():
            continue
        arm_out.parent.mkdir(parents=True, exist_ok=True)
        log = (OUT / "logs" / f"{stage}_{arm.label}_r{target_round:03d}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
        env = os.environ.copy()
        env.update({
            "PRE": str(PRE), "OUT": str(arm_out),
            "HELIOS_GPU": str(arm.gpu), "EXECUTION_ARM": arm.execution,
            "UPDATE_SCOPE": arm.scope, "ROUNDS": str(target_round),
            "SEED": str(arm.seed), "BETA": str(arm.beta),
            "FLOW_BASE_STD": str(arm.flow_std),
            "FLOW_BASE_STD_FINAL": str(arm.flow_std_final),
            "FLOW_BASE_STD_SCHEDULE": arm.flow_schedule,
            "RETRY_RESAMPLE_BATCH_CAP": str(arm.retry_cap),
        })
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_pre2_paper_arm.sh"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((arm, arm_out, process, stream))
    _status("expanding", stage=stage, target_round=target_round,
            running=[arm.label for arm, *_ in running])
    for arm, arm_out, process, stream in running:
        return_code = process.wait()
        stream.close()
        if return_code == 0 and _manifest_round(arm_out) == target_round:
            complete.append(arm)
    return complete


def _evaluation_path(stage: str, arm: Arm, target_round: int) -> Path:
    return OUT / "evaluations" / stage / arm.label / f"r{target_round:03d}"


def _run_evaluations(stage: str, arms: list[Arm], target_round: int) -> list[Arm]:
    running = []
    complete = []
    for arm in arms:
        arm_out = OUT / stage / arm.label
        eval_out = _evaluation_path(stage, arm, target_round)
        raw_eval = eval_out / "raw_eval.json"
        if raw_eval.is_file():
            complete.append(arm)
            continue
        if eval_out.exists():
            continue
        log = OUT / "logs" / f"eval_{stage}_{arm.label}_r{target_round:03d}.log"
        stream = log.open("a")
        env = os.environ.copy()
        env.update({
            "PRE": str(PRE), "EXPANSION": str(arm_out),
            "EVAL_OUT": str(eval_out), "HELIOS_GPU": str(arm.gpu),
            "TARGET_ROUND": str(target_round),
        })
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_pre2_paper_eval.sh"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((arm, raw_eval, process, stream))
    _status("evaluating", stage=stage, target_round=target_round,
            running=[arm.label for arm, *_ in running])
    for arm, raw_eval, process, stream in running:
        return_code = process.wait()
        stream.close()
        if return_code == 0 and raw_eval.is_file():
            complete.append(arm)
    return complete


def _score(stage: str, arms: list[Arm], target_round: int) -> dict:
    output = OUT / "rankings" / f"{stage}_r{target_round:03d}.json"
    command = [
        sys.executable, "scripts/score_pre2_paper_arms.py",
        "--round", str(target_round), "--output", str(output),
    ]
    for arm in arms:
        raw_eval = _evaluation_path(stage, arm, target_round) / "raw_eval.json"
        command.extend(["--arm", f"{arm.label}={raw_eval}"])
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text())


def _paper_ready(stage: str, arm: Arm, target_round: int, ranking: dict) -> None:
    selected = next(
        row for row in ranking["ranking"] if row["label"] == arm.label
    )
    _write(OUT / "PAPER_READY.json", {
        "status": "PAPER_READY_STOP_RULE_FULFILLED",
        "stage": stage, "arm": asdict(arm), "round": target_round,
        "pretrain": str(PRE.resolve()), "evaluation": selected,
        "expansion": str((OUT / stage / arm.label).resolve()),
    })
    _status("paper_ready", stage=stage, arm=arm.label, round=target_round)


def _initial_arms() -> list[Arm]:
    arms = []
    index = 0
    for execution in ("exp_balanced", "quad_fast", "quad_high_sr"):
        for scope in ("head", "trunk1", "trunk3"):
            arms.append(Arm(
                label=f"{execution}_{scope}", execution=execution,
                scope=scope, gpu=GPU_ORDER[index % len(GPU_ORDER)],
                seed=82100 + index,
            ))
            index += 1
    return arms


def _adaptive_arms() -> list[Arm]:
    arms = []
    index = 0
    for execution in ("exp_balanced", "quad_fast", "quad_high_sr"):
        for scope in ("trunk1", "trunk3"):
            arms.append(Arm(
                label=f"{execution}_{scope}_beta003_std130cos",
                execution=execution, scope=scope,
                gpu=GPU_ORDER[index % len(GPU_ORDER)], seed=82200 + index,
                beta=0.03, flow_std=1.3, flow_std_final=1.0,
                flow_schedule="cosine", retry_cap=96,
            ))
            index += 1
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-only", action="store_true",
        help="launch the six bounded adaptive r5 arms, then leave evaluation to the monitor",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_stream = (OUT / "SUPERVISOR.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another PRE2 paper supervisor is already running") from error

    protocol = {
        "pretrain": str(PRE), "GPUs": list(GPU_ORDER),
        "initial_matrix": [asdict(arm) for arm in _initial_arms()],
        "adaptive_matrix": [asdict(arm) for arm in _adaptive_arms()],
        "K": 16, "B": 8, "retry_B": 16, "parallel_episodes": 16,
        "quota": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "guidance": "none", "optimizer_steps_per_round": 2500,
        "learning_rate": 5e-5, "learning_rate_schedule": "constant",
        "batch_size": 256, "GP_buffer_cap": 1536,
        "evaluation": "fixed seed 91000, NFE12, 40 episodes/gamma, head-on",
    }
    _write(OUT / "PROTOCOL.json", protocol)
    while not (PRE / "pretrain_manifest.json").is_file():
        _status("waiting_for_recovered_PRE2")
        time.sleep(30)

    if args.adaptive_only:
        adaptive = _adaptive_arms()
        complete = _run_expansions("adaptive", adaptive, 5)
        _status(
            "adaptive_expansions_complete",
            target_round=5,
            completed=[arm.label for arm in complete],
        )
        return

    stages = [("initial", _initial_arms()), ("adaptive", _adaptive_arms())]
    continuation = None
    for stage, arms in stages:
        expanded = _run_expansions(stage, arms, 5)
        evaluated = _run_evaluations(stage, expanded, 5)
        if not evaluated:
            continue
        ranking = _score(stage, evaluated, 5)
        if ranking["winner"] is not None:
            winner = next(arm for arm in evaluated if arm.label == ranking["winner"])
            _paper_ready(stage, winner, 5, ranking)
            return
        best = next(arm for arm in evaluated if arm.label == ranking["best"])
        if stage == "initial" and not next(
            row for row in ranking["ranking"] if row["label"] == best.label
        )["close"]:
            continue
        continuation = (stage, best)
        break

    if continuation is None:
        _status("no_completed_arm_after_adaptive")
        return

    stage, arm = continuation
    target_round = 5
    while True:
        target_round += 5
        continuing = replace(
            arm, flow_std=1.0, flow_std_final=1.0, flow_schedule="none",
        )
        expanded = _run_expansions(stage, [continuing], target_round)
        if not expanded:
            _status("continuation_failed", stage=stage, arm=arm.label,
                    target_round=target_round)
            return
        evaluated = _run_evaluations(stage, expanded, target_round)
        if not evaluated:
            _status("continuation_evaluation_failed", stage=stage,
                    arm=arm.label, target_round=target_round)
            return
        ranking = _score(stage, evaluated, target_round)
        if ranking["winner"] is not None:
            _paper_ready(stage, arm, target_round, ranking)
            return


if __name__ == "__main__":
    main()
