#!/usr/bin/env python3
"""Select and continue the H10/stop/goal-box winner in five-round chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage1_single_ball_t128/0811_pre2_h10_goalbox_sweep"
PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
GAMMAS = ("0.1", "0.3", "0.5", "1")
CONTINUATION_LOG = STAGE / "CONTINUATION_LAUNCHES.jsonl"
CONTINUATION_PENDING = STAGE / "CONTINUATION_PENDING.json"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _acquire_lock(lock: Path) -> None:
    try:
        lock.mkdir()
    except FileExistsError:
        pid_path = lock / "pid"
        try:
            owner = int(pid_path.read_text().strip())
        except (OSError, ValueError):
            owner = None
        if owner is not None and _pid_alive(owner):
            raise SystemExit(
                f"another H10 goal-box supervisor owns the lock (pid {owner})"
            )
        if owner is None and time.time() - lock.stat().st_mtime < 60:
            raise SystemExit("H10 goal-box supervisor lock is initializing")
        pid_path.unlink(missing_ok=True)
        try:
            lock.rmdir()
        except OSError as error:
            raise SystemExit(
                f"cannot recover stale supervisor lock {lock}: {error}"
            ) from error
        lock.mkdir()
    (lock / "pid").write_text(f"{os.getpid()}\n")


def _release_lock(lock: Path) -> None:
    pid_path = lock / "pid"
    try:
        owner = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        owner = None
    if owner == os.getpid():
        pid_path.unlink(missing_ok=True)
        try:
            lock.rmdir()
        except OSError:
            pass


def _refresh() -> dict:
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/summarize_h10_goalbox_progress.py"),
        "--stage", str(STAGE), "--write",
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads((STAGE / "CURRENT_PROGRESS.json").read_text())


def _gamma_row(per_gamma: dict, gamma: str) -> dict | None:
    for alias in (gamma, f"{float(gamma):g}", str(float(gamma))):
        row = per_gamma.get(alias)
        if isinstance(row, dict):
            return row
    return None


def _final_checks(evaluation: dict, *, minimum_round: int = 15) -> dict:
    round_index = str(max(evaluation["rounds"]))
    row = evaluation["roundwise"][round_index]
    pooled = row["pooled"]
    per_gamma = row["per_gamma"]
    gamma_rows = [_gamma_row(per_gamma, gamma) for gamma in GAMMAS]
    checks = {
        "pooled_sr": pooled["SR"] >= 0.95,
        "each_gamma_sr": all(
            gamma_row is not None and gamma_row.get("SR", -1.0) >= 0.93
            for gamma_row in gamma_rows
        ),
        "collision": pooled["CR"] <= 0.03,
        "oob_plus_timeout": pooled["OOB"] + pooled["timeout"] <= 0.03,
        "validity": pooled["window_validity"] >= 0.95,
        "coverage": pooled["route_coverage"] == 1.0,
    }
    thresholds_passed = all(checks.values())
    minimum_round_reached = int(round_index) >= minimum_round
    return {
        "passed": thresholds_passed and minimum_round_reached,
        "thresholds_passed": thresholds_passed,
        "minimum_round": minimum_round,
        "minimum_round_reached": minimum_round_reached,
        "checks": checks,
        "round": int(round_index),
        "pooled": pooled,
        "per_gamma": per_gamma,
        "gamma_trend": row["gamma_trend"],
    }


def _launch_continuation(arm: dict, target_round: int) -> dict:
    output = Path(arm["output"])
    gpu = int(arm["gpu"])
    if gpu not in {1, 3}:
        raise ValueError("H10 goal-box continuation permits only GPU1/GPU3")
    if _continuation_was_launched(arm["name"], target_round):
        raise RuntimeError(
            f"continuation already launched: {arm['name']} -> r{target_round}"
        )
    stopping_margin = arm.get("stopping_margin_m", 0.02)
    environment = os.environ.copy()
    environment.update({
        "PRE": str(PRETRAIN),
        "OUT": str(output),
        "HELIOS_GPU": str(gpu),
        "ROUNDS": str(target_round),
        "SEED": "82410",
        "CONTROL_WEIGHT": "1",
        "TERMINAL_WEIGHT": "80",
        "BRAKING_WEIGHT": "0",
        "FINITE_SEGMENT": "1",
        "TASKSPACE_WEIGHT": "0",
        "TASKSPACE_TARGET": "0.15",
        "GOAL_BOX_WEIGHT": f"{float(arm['weight']):g}",
        "GOAL_BOX_HALF_EXTENT": "0.2",
        "GOAL_BOX_TEMPERATURE": "1.0",
        "FULL_H_TASKSPACE": "1",
        "STOPPING_MARGIN": (
            "" if stopping_margin is None else f"{float(stopping_margin):g}"
        ),
        "AXIS_WEIGHT": "5",
        "AXIS_RADIUS": "1.1",
        "CLEARANCE_WEIGHT": "2500",
        "CLEARANCE_TARGET": "0.6",
        "SUCCESS_QUOTA": "12",
        "SAMPLE_MODES": "0,0,0,1,1,1,2,2,2,3,3,3",
        "FAITHFUL_RETRY": "1",
    })
    log = STAGE / "logs" / (
        f"{arm['name']}.continue_r{target_round:03d}.screen.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    pending = {
        "status": "PREPARED",
        "arm": arm["name"],
        "target_round": target_round,
        "gpu": gpu,
        "log": str(log),
        "prepared_unix": time.time(),
    }
    _atomic_json(CONTINUATION_PENDING, pending)
    stream = log.open("a")
    try:
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_control_braking_run_and_eval.sh"],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        _atomic_json(CONTINUATION_PENDING, {
            **pending,
            "status": "LAUNCH_FAILED",
            "failed_unix": time.time(),
        })
        raise
    finally:
        stream.close()
    record = {
        "arm": arm["name"],
        "target_round": target_round,
        "pid": process.pid,
        "gpu": gpu,
        "log": str(log),
        "launched_unix": time.time(),
    }
    _atomic_json(CONTINUATION_PENDING, {
        **pending,
        "status": "LAUNCHED",
        "pid": process.pid,
        "launched_unix": record["launched_unix"],
    })
    with CONTINUATION_LOG.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def _read_pending() -> dict | None:
    if not CONTINUATION_PENDING.is_file():
        return None
    try:
        payload = json.loads(CONTINUATION_PENDING.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "CORRUPT"}
    return payload if isinstance(payload, dict) else {"status": "CORRUPT"}


def _continuation_was_launched(arm_name: str, target_round: int) -> bool:
    pending = _read_pending()
    if (
        pending is not None
        and pending.get("arm") == arm_name
        and pending.get("target_round") == target_round
    ):
        return True
    if not CONTINUATION_LOG.is_file():
        return False
    for line in CONTINUATION_LOG.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("arm") == arm_name
            and record.get("target_round") == target_round
        ):
            return True
    return False


def _block(status: str, **details) -> None:
    _atomic_json(STAGE / "BLOCKED.json", {
        "status": status,
        **details,
        "updated_unix": time.time(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-round", type=int, default=50)
    parser.add_argument("--minimum-final-round", type=int, default=15)
    args = parser.parse_args()
    lock = STAGE / ".supervisor.lock"
    _acquire_lock(lock)
    try:
        while True:
            progress = _refresh()
            _atomic_json(STAGE / "SUPERVISOR_STATUS.json", {
                "status": "RUNNING",
                "pid": os.getpid(),
                "updated_unix": time.time(),
                "progress_status": progress["status"],
            })
            if (STAGE / "PAPER_READY.json").is_file():
                return
            result_path = STAGE / "SWEEP_RESULT.json"
            if not result_path.is_file():
                if not progress["selection_unblocked"]:
                    time.sleep(args.interval_seconds)
                    continue
                contenders = [
                    arm for arm in progress["arms"] if arm["r5_ready"]
                ]
                if not contenders:
                    if all(arm["resolved"] for arm in progress["arms"]):
                        _block("ALL_R5_ARMS_FAILED")
                        return
                    time.sleep(args.interval_seconds)
                    continue
                ranked = sorted(
                    contenders,
                    key=lambda arm: (-arm["selection_score"], arm["name"]),
                )
                winner = ranked[0]
                _atomic_json(result_path, {
                    "status": "R5_WINNER_SELECTED_FOR_CONTINUATION",
                    "selected_unix": time.time(),
                    "selection_unblock_reason": progress[
                        "selection_unblock_reason"
                    ],
                    "winner": winner,
                    "selection_score": winner["selection_score"],
                    "ranking": [{
                        "name": arm["name"],
                        "score": arm["selection_score"],
                        "endpoint": arm["evaluation"]["endpoint"],
                        "stability": arm["evaluation"]["stability"],
                        "gamma_trend": arm["evaluation"][
                            "endpoint_gamma_trend"
                        ],
                    } for arm in ranked],
                })
            result = json.loads(result_path.read_text())
            winner_name = result["winner"]["name"]
            winner = next((
                arm for arm in progress["arms"] if arm["name"] == winner_name
            ), None)
            if winner is None:
                _block("SELECTED_WINNER_MISSING_FROM_SWEEP", winner=winner_name)
                return
            if winner["active"]:
                time.sleep(args.interval_seconds)
                continue
            if winner["failed"]:
                _block(
                    "WINNER_FAILED_CLOSED",
                    winner=winner_name,
                    committed_round=winner["committed_round"],
                )
                return
            evaluation = winner["evaluation"]
            if evaluation is None:
                time.sleep(args.interval_seconds)
                continue
            boundary = max(evaluation["rounds"])
            pending = _read_pending()
            if pending is not None:
                if pending.get("status") == "CORRUPT":
                    _block("CORRUPT_CONTINUATION_PENDING", winner=winner_name)
                    return
                if pending.get("arm") != winner_name:
                    _block(
                        "CONTINUATION_PENDING_FOR_NON_WINNER",
                        winner=winner_name,
                        pending=pending,
                    )
                    return
                pending_target = int(pending["target_round"])
                if boundary >= pending_target:
                    CONTINUATION_PENDING.unlink(missing_ok=True)
                elif _pid_alive(int(pending.get("pid", -1))):
                    time.sleep(args.interval_seconds)
                    continue
                else:
                    _block(
                        "CONTINUATION_LAUNCH_LOST",
                        winner=winner_name,
                        boundary=boundary,
                        pending=pending,
                    )
                    return
            committed_round = winner["committed_round"]
            if committed_round is not None and committed_round > boundary:
                _atomic_json(STAGE / "SUPERVISOR_STATUS.json", {
                    "status": "WAITING_FOR_FIXED_EVALUATION",
                    "pid": os.getpid(),
                    "winner": winner_name,
                    "committed_round": committed_round,
                    "evaluated_through": boundary,
                    "updated_unix": time.time(),
                })
                time.sleep(args.interval_seconds)
                continue
            final = _final_checks(
                evaluation, minimum_round=args.minimum_final_round,
            )
            if final["passed"]:
                checkpoint = Path(winner["output"]) / (
                    f"checkpoint_{final['round']:03d}.pt"
                )
                raw_eval = Path(evaluation["path"])
                if not checkpoint.is_file() or not raw_eval.is_file():
                    _block(
                        "FINAL_ARTIFACT_MISSING",
                        winner=winner_name,
                        checkpoint=str(checkpoint),
                        raw_eval=str(raw_eval),
                    )
                    return
                _atomic_json(STAGE / "PAPER_READY.json", {
                    "status": "FINAL_OVERNIGHT_ACHIEVED",
                    "winner": winner_name,
                    "weight": winner["weight"],
                    "gpu": winner["gpu"],
                    "final": final,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "raw_eval": str(raw_eval),
                    "raw_eval_sha256": _sha256(raw_eval),
                    "completed_unix": time.time(),
                })
                return
            if boundary >= args.max_round:
                _block(
                    "MAX_ROUND_REACHED_WITHOUT_FINAL_GOAL",
                    winner=winner_name,
                    final=final,
                )
                return
            target_round = boundary + 5
            if _continuation_was_launched(winner_name, target_round):
                _block(
                    "DUPLICATE_CONTINUATION_PREVENTED",
                    winner=winner_name,
                    boundary=boundary,
                    target_round=target_round,
                )
                return
            _launch_continuation(winner, target_round)
            time.sleep(args.interval_seconds)
    finally:
        _release_lock(lock)


if __name__ == "__main__":
    main()
