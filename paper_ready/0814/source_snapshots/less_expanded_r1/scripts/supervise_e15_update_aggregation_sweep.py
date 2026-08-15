#!/usr/bin/env python3
"""One-shot reporter/supervisor for the E15 update/aggregation sweep.

Default mode only writes a status snapshot.  ``--run`` starts the duplicate-
safe launcher once, then observes the five r7 evaluations.  It never resumes,
retries, stops, or promotes an arm.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0811_pre2_e15_update_aggregation_sweep"
)
SWEEP = STAGE / "SWEEP.json"
STATUS = STAGE / "SUPERVISOR_STATUS.json"
LOCK = STAGE / ".supervisor.lock"
CONFIRMATION = "I_UNDERSTAND_THIS_STARTS_FIVE_HELIOS_JOBS"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _acquire_lock() -> None:
    try:
        LOCK.mkdir(parents=True)
    except FileExistsError:
        try:
            pid = int((LOCK / "pid").read_text().strip())
        except (FileNotFoundError, ValueError):
            pid = -1
        if _pid_alive(pid):
            raise RuntimeError(f"supervisor already active as PID {pid}")
        shutil.rmtree(LOCK)
        LOCK.mkdir(parents=True)
    (LOCK / "pid").write_text(f"{os.getpid()}\n")


def _release_lock() -> None:
    shutil.rmtree(LOCK, ignore_errors=True)


def _evaluation_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE":
        return False
    summary = payload.get("summary", {})
    return all(str(index) in summary for index in range(8))


def snapshot() -> dict[str, Any]:
    if not SWEEP.is_file():
        raise FileNotFoundError(
            f"prepare the sweep first with launch_e15_update_aggregation_sweep.py: {SWEEP}"
        )
    sweep = json.loads(SWEEP.read_text())
    arms = []
    for arm in sweep["arms"]:
        output = STAGE / "arms" / arm["name"]
        evaluation = output / "fixed_eval_r000_r007/raw_eval.json"
        checkpoint = output / "checkpoint_007.pt"
        failure = output / "FAILED.json"
        complete = checkpoint.is_file() and _evaluation_complete(evaluation)
        failed = failure.is_file() and (
            not checkpoint.is_file()
            or failure.stat().st_mtime_ns > checkpoint.stat().st_mtime_ns
        )
        arms.append({
            "name": arm["name"], "gpu": arm["gpu"],
            "checkpoint_007": checkpoint.is_file(),
            "fixed_eval_complete": _evaluation_complete(evaluation),
            "failed": failed,
            "state": "COMPLETE" if complete else "FAILED" if failed else "PENDING",
        })
    payload = {
        "status": (
            "ALL_R7_EVALUATIONS_COMPLETE"
            if all(arm["state"] == "COMPLETE" for arm in arms)
            else "TERMINAL_WITH_FAILURE"
            if all(arm["state"] in {"COMPLETE", "FAILED"} for arm in arms)
            else "MONITORING"
        ),
        "updated_unix": time.time(),
        "pid": os.getpid(),
        "continuation_policy": "NO_AUTO_CONTINUATION_BEYOND_R7",
        "arms": arms,
    }
    _atomic_json(STATUS, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-launch", default="")
    parser.add_argument("--interval-seconds", type=int, default=30)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(snapshot(), indent=2))
        return
    if args.confirm_launch != CONFIRMATION:
        parser.error(f"--run requires --confirm-launch {CONFIRMATION}")
    if args.interval_seconds < 5:
        parser.error("--interval-seconds must be at least 5")
    _acquire_lock()
    try:
        subprocess.run([
            "python", "scripts/launch_e15_update_aggregation_sweep.py",
            "--launch", "--confirm-launch", CONFIRMATION,
        ], cwd=ROOT, check=True)
        while True:
            payload = snapshot()
            if payload["status"] in {
                "ALL_R7_EVALUATIONS_COMPLETE", "TERMINAL_WITH_FAILURE",
            }:
                print(json.dumps(payload, indent=2))
                return
            time.sleep(args.interval_seconds)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
