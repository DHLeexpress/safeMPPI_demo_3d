#!/usr/bin/env python3
"""Summarize H10/stop/goal-box expansion and fixed-bank evaluations."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
DEFAULT_STAGE = ROOT / "results/stage1_single_ball_t128/0811_pre2_h10_goalbox_sweep"
GAMMAS = ("0.1", "0.3", "0.5", "1")
EVALUATION_STATUS = "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
PROCESS_NAMES = (
    "research_ball_expansion_optimization.py",
    "research_evaluate_ball_expansion.py",
    "RECIPE_control_braking_run_and_eval.sh",
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _gamma_row(per_gamma: dict, gamma: str) -> dict:
    aliases = (gamma, f"{float(gamma):g}", str(float(gamma)))
    for alias in aliases:
        row = per_gamma.get(alias)
        if isinstance(row, dict):
            return row
    raise KeyError(f"missing gamma={gamma}; available={sorted(per_gamma)}")


def _finite_mean(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        return None
    return sum(converted) / len(converted)


def _trend(values: list[float], increasing: bool) -> float:
    if len(values) < 2:
        return 0.0
    return sum(
        right >= left if increasing else right <= left
        for left, right in zip(values, values[1:])
    ) / (len(values) - 1)


def _gamma_trend(summary: dict) -> dict:
    per_gamma = summary["per_gamma"]
    low = [_gamma_row(per_gamma, gamma) for gamma in GAMMAS[:2]]
    high = [_gamma_row(per_gamma, gamma) for gamma in GAMMAS[2:]]
    low_ttg = _finite_mean(low, "successful_time_to_goal_s")
    high_ttg = _finite_mean(high, "successful_time_to_goal_s")
    low_clearance = _finite_mean(low, "successful_min_clearance_m")
    high_clearance = _finite_mean(high, "successful_min_clearance_m")
    return {
        "low_gamma_ttg_s": low_ttg,
        "high_gamma_ttg_s": high_ttg,
        "ttg_high_is_lower": (
            None
            if low_ttg is None or high_ttg is None
            else high_ttg < low_ttg
        ),
        "low_gamma_clearance_m": low_clearance,
        "high_gamma_clearance_m": high_clearance,
        "clearance_high_is_lower": (
            None
            if low_clearance is None or high_clearance is None
            else high_clearance < low_clearance
        ),
    }


def evaluate(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
        if raw.get("status") != EVALUATION_STATUS:
            return None
        summaries = raw["summary"]
        rounds = sorted(summaries, key=int)
        numeric_rounds = [int(value) for value in rounds]
        if not rounds or numeric_rounds != list(range(numeric_rounds[-1] + 1)):
            return None
        roundwise = {}
        for round_index in rounds:
            source = summaries[round_index]
            pooled = source["pooled"]
            for key in (
                "SR", "CR", "OOB", "timeout", "window_validity",
                "route_coverage",
            ):
                value = float(pooled[key])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite pooled {key}")
            per_gamma = {
                gamma: _gamma_row(source["per_gamma"], gamma)
                for gamma in GAMMAS
            }
            for row in per_gamma.values():
                if not math.isfinite(float(row["SR"])):
                    raise ValueError("non-finite gamma SR")
            normalized = {"pooled": pooled, "per_gamma": per_gamma}
            roundwise[round_index] = {
                **normalized,
                "gamma_trend": _gamma_trend(normalized),
            }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    pooled = [roundwise[round_index]["pooled"] for round_index in rounds]
    return {
        "status": raw["status"],
        "path": str(path),
        "rounds": numeric_rounds,
        "roundwise": roundwise,
        "endpoint": pooled[-1],
        "endpoint_gamma_trend": roundwise[rounds[-1]]["gamma_trend"],
        "stability": {
            "sr_nondecreasing_fraction": _trend(
                [row["SR"] for row in pooled], True,
            ),
            "cr_nonincreasing_fraction": _trend(
                [row["CR"] for row in pooled], False,
            ),
            "oob_nonincreasing_fraction": _trend(
                [row["OOB"] for row in pooled], False,
            ),
            "validity_nondecreasing_fraction": _trend(
                [row["window_validity"] for row in pooled], True,
            ),
        },
    }


def _latest_evaluation(output: Path) -> dict | None:
    evaluations = [
        result
        for path in output.glob("fixed_eval_r000_r*/raw_eval.json")
        if (result := evaluate(path)) is not None
    ]
    if not evaluations:
        return None
    return max(
        evaluations,
        key=lambda result: (
            max(result["rounds"]), Path(result["path"]).stat().st_mtime,
        ),
    )


def _last_retry(log_paths: list[Path]) -> str | None:
    existing = sorted(
        (path for path in log_paths if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in existing:
        for line in reversed(path.read_text(errors="replace").splitlines()):
            if "[sample-update]" in line:
                return line.strip()
    return None


def _active(commands: str, output: Path) -> bool:
    lock_pid = Path(f"{output}.control_braking_process.lock/pid")
    if lock_pid.is_file():
        try:
            if _pid_alive(int(lock_pid.read_text().strip())):
                return True
        except (OSError, ValueError):
            pass
    target = str(output.resolve())
    for command in commands.splitlines():
        if not any(name in command for name in PROCESS_NAMES):
            continue
        try:
            arguments = shlex.split(command)
        except ValueError:
            continue
        if target in arguments:
            return True
    return False


def _read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _selection_score(evaluation: dict) -> float:
    endpoint = evaluation["endpoint"]
    stability = evaluation["stability"]
    return float(
        2.0 * endpoint["SR"]
        - endpoint["CR"]
        - 1.5 * endpoint["OOB"]
        - endpoint["timeout"]
        + 0.5 * endpoint["window_validity"]
        + 0.10 * stability["sr_nondecreasing_fraction"]
        + 0.10 * stability["cr_nonincreasing_fraction"]
        + 0.15 * stability["oob_nonincreasing_fraction"]
        + 0.10 * stability["validity_nondecreasing_fraction"]
        + (0.5 if endpoint["route_coverage"] == 1.0 else 0.0)
    )


def _failure_state(
    output: Path,
    *,
    active: bool,
    checkpoints: list[Path],
    evaluation: dict | None,
) -> tuple[bool, bool]:
    failure = output / "FAILED.json"
    if not failure.is_file():
        return False, False
    successful_artifacts = [*checkpoints]
    for name in ("resume_state.json", "resume_state_latest.pt"):
        candidate = output / name
        if candidate.is_file():
            successful_artifacts.append(candidate)
    if evaluation is not None:
        successful_artifacts.append(Path(evaluation["path"]))
    latest_success = max(
        (path.stat().st_mtime for path in successful_artifacts),
        default=float("-inf"),
    )
    current = not active and failure.stat().st_mtime >= latest_success
    return current, not current


def build(stage: Path) -> dict:
    sweep = json.loads((stage / "SWEEP.json").read_text())
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    arms = []
    for record in sweep["arms"]:
        output = Path(record["output"])
        stopping_margin = record.get("stopping_margin_m", 0.02)
        family = (
            "full_h_only" if stopping_margin is None else "stopping_backup"
        )
        checkpoints = sorted(output.glob("checkpoint_*.pt"))
        metrics = _read_metrics(output / "metrics.jsonl")
        active = _active(commands, output)
        logs = [
            stage / "logs" / f"{record['name']}.screen.log",
            output / "helios.log",
            *stage.glob(
                f"logs/{record['name']}.continue_r*.screen.log"
            ),
        ]
        committed_round = (
            int(checkpoints[-1].stem.split("_")[-1])
            if checkpoints else None
        )
        evaluation = _latest_evaluation(output)
        failed, failed_stale = _failure_state(
            output,
            active=active,
            checkpoints=checkpoints,
            evaluation=evaluation,
        )
        r5_ready = bool(
            evaluation is not None
            and committed_round is not None
            and committed_round >= 5
            and max(evaluation["rounds"]) >= 5
        )
        arms.append({
            **record,
            "family": family,
            "stopping_margin_m": stopping_margin,
            "active": active,
            "failed": failed,
            "failed_artifact": (output / "FAILED.json").is_file(),
            "failed_stale": failed_stale,
            "committed_round": committed_round,
            "last_training_row": metrics[-1] if metrics else None,
            "last_retry": _last_retry(logs),
            "evaluation": evaluation,
            "r5_ready": r5_ready,
            "resolved": r5_ready or (failed and not active),
            "selection_score": (
                _selection_score(evaluation) if r5_ready else None
            ),
        })
    complete_r5 = [arm for arm in arms if arm["r5_ready"]]
    ranking = sorted(
        complete_r5,
        key=lambda arm: (-arm["selection_score"], arm["name"]),
    )
    all_resolved = all(arm["resolved"] for arm in arms)
    full_h_only = [arm for arm in arms if arm["family"] == "full_h_only"]
    full_h_only_resolved = bool(full_h_only) and all(
        arm["resolved"] for arm in full_h_only
    )
    return {
        "status": (
            "R5_COMPLETE"
            if len(complete_r5) == len(arms)
            else "R5_RESOLVED" if all_resolved else "RUNNING"
        ),
        "arms": arms,
        "ranking": [arm["name"] for arm in ranking],
        "provisional_winner": ranking[0]["name"] if ranking else None,
        "selection_unblocked": all_resolved or full_h_only_resolved,
        "selection_unblock_reason": (
            "all_arms_resolved"
            if all_resolved
            else "full_h_only_family_resolved"
            if full_h_only_resolved
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build(args.stage)
    if args.write:
        _atomic_json(args.stage / "CURRENT_PROGRESS.json", result)
    print(json.dumps({
        "status": result["status"],
        "arms": [{
            "name": arm["name"],
            "family": arm["family"],
            "stopping_margin_m": arm["stopping_margin_m"],
            "active": arm["active"],
            "failed": arm["failed"],
            "failed_stale": arm["failed_stale"],
            "round": arm["committed_round"],
            "r5_ready": arm["r5_ready"],
            "last_retry": arm["last_retry"],
            "endpoint": (
                None if arm["evaluation"] is None
                else arm["evaluation"]["endpoint"]
            ),
        } for arm in result["arms"]],
        "ranking": result["ranking"],
        "selection_unblocked": result["selection_unblocked"],
        "selection_unblock_reason": result["selection_unblock_reason"],
    }, indent=2))


if __name__ == "__main__":
    main()
