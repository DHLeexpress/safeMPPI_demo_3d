#!/usr/bin/env python3
"""Publish a compact live summary for the control/braking expansion sweep."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage1_single_ball_t128/0810_pre2_control_braking_sweep"
MODES = ("below", "above", "left", "right")
SAMPLE_RE = re.compile(
    r"\[sample-update\] r(?P<round>\d+) gamma (?P<gamma>[0-9.]+) "
    r"retry batch (?P<retry>\d+)/(?P<cap>\d+) \| "
    r"guided b(?P<gb>\d+)/a(?P<ga>\d+)/l(?P<gl>\d+)/r(?P<gr>\d+) \| "
    r"unguided b(?P<ub>\d+)/a(?P<ua>\d+)/l(?P<ul>\d+)/r(?P<ur>\d+) \| "
    r"need mode \"(?P<missing>[^\"]*)\""
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, indent=2) + "\n")


def _pid_alive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        os.kill(int(path.read_text().strip()), 0)
    except (OSError, ValueError):
        return False
    return True


def _commands() -> str:
    return subprocess.run(
        ["ps", "-ax", "-o", "command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _active(commands: str, output: Path) -> tuple[bool, bool]:
    target = str(output.resolve())
    expansion = False
    evaluation = False
    for command in commands.splitlines():
        if target not in command:
            continue
        expansion |= "research_ball_expansion_optimization.py" in command
        evaluation |= "research_evaluate_ball_expansion.py" in command
    lock = Path(f"{output}.control_braking_process.lock/pid")
    return expansion or _pid_alive(lock), evaluation


def _last_metric(output: Path) -> dict | None:
    path = output / "metrics.jsonl"
    if not path.is_file():
        return None
    last = None
    with path.open() as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    return last


def _sample_updates(stage: Path, output: Path, arm_name: str) -> dict:
    paths = [stage / "logs" / f"{arm_name}.screen.log", output / "helios.log"]
    found = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(errors="replace") as stream:
            for line in stream:
                match = SAMPLE_RE.search(line)
                if match:
                    found.append(match.groupdict())
    if not found:
        return {"round": None, "by_gamma": {}}
    current_round = max(int(row["round"]) for row in found)
    latest = {}
    for row in found:
        if int(row["round"]) == current_round:
            latest[str(float(row["gamma"])).rstrip("0").rstrip(".")] = row
    by_gamma = {}
    for gamma, row in latest.items():
        missing_ids = [value for value in row["missing"].split(",") if value]
        missing = Counter(MODES[int(value)] for value in missing_ids)
        by_gamma[gamma] = {
            "retry": int(row["retry"]),
            "cap": int(row["cap"]),
            "guided": {
                "below": int(row["gb"]), "above": int(row["ga"]),
                "left": int(row["gl"]), "right": int(row["gr"]),
            },
            "unguided": {
                "below": int(row["ub"]), "above": int(row["ua"]),
                "left": int(row["ul"]), "right": int(row["ur"]),
            },
            "missing": dict(missing),
        }
    return {"round": current_round, "by_gamma": by_gamma}


def _raw_evaluation(output: Path, target_round: int) -> dict | None:
    raw_eval = output / f"fixed_eval_r000_r{target_round:03d}/raw_eval.json"
    if not raw_eval.is_file():
        return None
    payload = json.loads(raw_eval.read_text())
    summary = payload["summary"]
    series = {}
    for round_name in sorted(summary, key=int):
        pooled = summary[round_name]["pooled"]
        series[round_name] = {
            "SR": pooled["SR"],
            "CR": pooled["CR"],
            "OOB": pooled["OOB"],
            "timeout": pooled["timeout"],
            "validity": pooled["window_validity"],
            "clearance_m": pooled["successful_min_clearance_m"],
            "time_to_goal_s": pooled["successful_time_to_goal_s"],
            "route_counts": pooled["route_counts"],
            "coverage": pooled["route_coverage"],
        }
    return {
        "path": str(raw_eval),
        "series": series,
        "final": series[str(target_round)],
        "per_gamma_final": summary[str(target_round)]["per_gamma"],
    }


def _compact_modes(values: dict) -> str:
    return "/".join(str(values.get(mode, 0)) for mode in MODES)


def _compact_missing(values: dict) -> str:
    if not values:
        return "none"
    return ",".join(f"{mode}{count}" for mode, count in values.items())


def _arm_summary(
    stage: Path, arm: dict, output: Path, commands: str, target_round: int,
) -> dict:
    checkpoints = sorted(
        int(path.stem.rsplit("_", 1)[1]) for path in output.glob("checkpoint_*.pt")
    )
    committed = checkpoints[-1] if checkpoints else None
    expansion_active, evaluation_active = _active(commands, output)
    evaluation = _raw_evaluation(output, target_round)
    failed = output / "FAILED.json"
    if evaluation is not None:
        state = "evaluated"
    elif evaluation_active:
        state = "evaluating"
    elif failed.is_file():
        state = "failed"
    elif committed == target_round:
        state = "evaluation_pending"
    elif expansion_active:
        state = "expanding"
    elif committed is not None:
        state = "stopped_partial"
    else:
        state = "not_started"
    metric = _last_metric(output)
    live = _sample_updates(stage, output, arm["name"])
    return {
        "name": arm["name"],
        "design": {
            "control_weight": arm["control_weight"],
            "braking_weight": arm["braking_weight"],
            "finite_segment": arm["finite_segment"],
        },
        "state": state,
        "committed_round": committed,
        "target_round": target_round,
        "live": live,
        "last_committed": None if metric is None else {
            "round": metric.get("round"),
            "retry_batches_by_gamma": metric.get("retry_batches_by_gamma"),
            "attempted_episode_count": metric.get("attempted_episode_count"),
            "positive_loss": metric.get("positive_loss"),
            "round_total_s": metric.get("round_total_s"),
            "gather_s": metric.get("gather_s"),
            "update_s": metric.get("update_s"),
        },
        "evaluation": evaluation,
        "failed_path": str(failed) if failed.is_file() else None,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Control/braking expansion monitor",
        "",
        f"Updated: {payload['updated_local']}",
        "",
        "Mode order in support cells: `below/above/left/right`.",
        "",
        "| arm | design (control/brake/cap) | state | committed | live bottleneck | last loss | r5 raw SR/CR/OOB/V/cov |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for arm in payload["short_sweep"]:
        design = arm["design"]
        live_parts = []
        for gamma, row in sorted(arm["live"]["by_gamma"].items(), key=lambda item: float(item[0])):
            live_parts.append(
                f"g{gamma} r{row['retry']} U{_compact_modes(row['unguided'])} "
                f"miss={_compact_missing(row['missing'])}"
            )
        bottleneck = "; ".join(live_parts) or "—"
        metric = arm["last_committed"] or {}
        loss = metric.get("positive_loss")
        loss_text = "—" if loss is None else f"{loss:.4f}"
        evaluation = arm["evaluation"]
        if evaluation is None:
            raw = "—"
        else:
            final = evaluation["final"]
            raw = (
                f"{final['SR']:.3f}/{final['CR']:.3f}/{final['OOB']:.3f}/"
                f"{final['validity']:.3f}/{final['coverage']:.2f}"
            )
        lines.append(
            f"| {arm['name']} | {design['control_weight']:g}/"
            f"{design['braking_weight']:g}/"
            f"{'yes' if design['finite_segment'] else 'no'} | "
            f"{arm['state']} | {arm['committed_round'] if arm['committed_round'] is not None else '—'}"
            f"/{arm['target_round']} | {bottleneck} | {loss_text} | {raw} |"
        )
    if payload.get("winner"):
        lines += [
            "",
            "## Selected winner",
            "",
            f"- Arm: `{payload['winner']['arm']['name']}`",
            f"- Trend score: `{payload['winner']['score']:.4f}`",
            f"- Wrong-direction mass: `{payload['winner']['wrong_direction_mass']:.4f}`",
        ]
    paper = payload.get("paper_run")
    if paper:
        lines += ["", "## q1 faithful r15", ""]
        lines.append(
            f"- `{paper['name']}`: {paper['state']}, checkpoint "
            f"{paper['committed_round']}/{paper['target_round']}"
        )
        if paper["evaluation"]:
            final = paper["evaluation"]["final"]
            lines.append(
                "- r15 raw: "
                f"SR {final['SR']:.3f}, CR {final['CR']:.3f}, "
                f"OOB {final['OOB']:.3f}, validity {final['validity']:.3f}, "
                f"coverage {final['coverage']:.2f}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = json.loads((STAGE / "SWEEP.json").read_text())
    commands = _commands()
    short_sweep = [
        _arm_summary(STAGE, arm, STAGE / "arms" / arm["name"], commands, 5)
        for arm in config["arms"]
    ]
    result_path = STAGE / "SWEEP_RESULT.json"
    winner = None
    if result_path.is_file():
        winner = json.loads(result_path.read_text()).get("winner")
    paper_run = None
    paper_marker = STAGE / "PAPER_RUN_LAUNCHED.json"
    if paper_marker.is_file():
        marker = json.loads(paper_marker.read_text())
        paper_output = Path(marker["output"])
        paper_arm = {
            "name": marker["name"],
            "control_weight": marker["control_weight"],
            "braking_weight": marker["braking_weight"],
            "finite_segment": marker["finite_segment"],
        }
        paper_run = _arm_summary(
            STAGE, paper_arm, paper_output, commands, 15,
        )
    payload = {
        "status": "CONTROL_BRAKING_PROGRESS",
        "updated_unix": time.time(),
        "updated_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "supervisor_active": any(
            "supervise_control_braking_sweep.py" in line
            for line in commands.splitlines()
        ),
        "short_sweep": short_sweep,
        "winner": winner,
        "paper_run": paper_run,
    }
    markdown = _markdown(payload)
    if args.write:
        _atomic_json(STAGE / "CURRENT_PROGRESS.json", payload)
        _atomic_text(STAGE / "CURRENT_PROGRESS.md", markdown)
    print(markdown)


if __name__ == "__main__":
    main()
