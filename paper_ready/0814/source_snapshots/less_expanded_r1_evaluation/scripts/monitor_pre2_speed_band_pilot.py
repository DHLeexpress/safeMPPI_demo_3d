#!/usr/bin/env python3
"""Read-only Helios sync and summary for the paired r0->r2 speed-band pilots."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_obstacle_speed_band_pilot"
)
QUEUE = STAGE / "QUEUE.json"
HOST = "dohyun@helios.robotics.caltech.edu"
MODES = ("below", "above", "left", "right")
SAMPLE_RE = re.compile(
    r"\[sample-update\] r(?P<round>\d+) gamma (?P<gamma>[0-9.]+) "
    r"retry batch (?P<retry>\d+)/(?P<cap>\d+) \| "
    r"guided b(?P<gb>\d+)/a(?P<ga>\d+)/l(?P<gl>\d+)/r(?P<gr>\d+) \| "
    r"unguided b(?P<ub>\d+)/a(?P<ua>\d+)/l(?P<ul>\d+)/r(?P<ur>\d+) \| "
    r'need mode "(?P<missing>[^"]*)"'
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _ssh(command: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, command],
        text=True,
    )


def _sync_control(queue: dict[str, Any]) -> Path:
    local = STAGE / "remote_mirror/control"
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "--prune-empty-dirs", "-e", "ssh",
        "--include=/status/", "--include=/status/*.json",
        "--include=/logs/", "--include=/logs/*.log", "--exclude=*",
        f"{HOST}:{queue['remote_control']}/", f"{local}/",
    ], check=True)
    return local


def _sync_arm(record: dict[str, Any]) -> Path:
    local = STAGE / "remote_mirror/arms" / record["name"]
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "--prune-empty-dirs", "-e", "ssh",
        "--include=/resume_state.json", "--include=/manifest.json",
        "--include=/metrics.jsonl", "--include=/FAILED.json",
        "--include=/SPOOL_PROVENANCE.json",
        "--include=/fixed_eval_r000_r002/",
        "--include=/fixed_eval_r000_r002/raw_eval.json",
        "--exclude=*", f"{HOST}:{record['remote_output']}/", f"{local}/",
    ], check=True)
    return local


def _queue_states(queue: dict[str, Any]) -> tuple[dict[tuple[int, int], str], dict[str, str]]:
    states: dict[tuple[int, int], str] = {}
    listings: dict[str, str] = {}
    for gpu_text, socket in queue["tsp_sockets"].items():
        command = (
            f"if test -S {shlex.quote(socket)}; then "
            f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=200; tsp -l; "
            "else echo SERVER_ABSENT; fi"
        )
        listing = _ssh(command)
        listings[gpu_text] = listing
        for line in listing.splitlines():
            match = re.match(r"^\s*(\d+)\s+(queued|running|finished|skipped)\b", line)
            if match:
                states[(int(gpu_text), int(match.group(1)))] = match.group(2).upper()
    return states, listings


def _remote_checkpoint_rounds(records: list[dict[str, Any]]) -> dict[str, int | None]:
    snippets = []
    for index, record in enumerate(records):
        directory = shlex.quote(record["remote_output"])
        snippets.append(
            f"d={directory}; latest=''; if test -d \"$d\"; then "
            "latest=$(find \"$d\" -maxdepth 1 -type f -name 'checkpoint_*.pt' "
            "-printf '%f\\n' | sort | tail -1); fi; "
            f"printf '{index}|%s\\n' \"$latest\""
        )
    output = _ssh("set -euo pipefail; " + "; ".join(snippets))
    rounds: dict[str, int | None] = {}
    for line in output.splitlines():
        index_text, filename = line.split("|", 1)
        match = re.fullmatch(r"checkpoint_(\d{3}).pt", filename)
        rounds[records[int(index_text)]["config_hash"]] = (
            int(match.group(1)) if match else None
        )
    return rounds


def _sample_updates(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"round": None, "by_gamma": {}}
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = SAMPLE_RE.search(line)
            if match:
                rows.append(match.groupdict())
    if not rows:
        return {"round": None, "by_gamma": {}}
    current_round = max(int(row["round"]) for row in rows)
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        if int(row["round"]) == current_round:
            gamma = str(float(row["gamma"])).rstrip("0").rstrip(".")
            latest[gamma] = row
    by_gamma = {}
    for gamma, row in latest.items():
        missing_ids = [int(value) for value in row["missing"].split(",") if value]
        by_gamma[gamma] = {
            "retry": int(row["retry"]),
            "cap": int(row["cap"]) + 1,
            "unguided_support": {
                "below": int(row["ub"]), "above": int(row["ua"]),
                "left": int(row["ul"]), "right": int(row["ur"]),
            },
            "missing": dict(Counter(MODES[value] for value in missing_ids)),
        }
    return {"round": current_round, "by_gamma": by_gamma}


def _last_metric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open() as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if last is None:
        return None
    return {
        key: last.get(key) for key in (
            "round", "retry_batches_by_gamma", "attempted_episode_count",
            "positive_loss", "round_total_s", "gather_s", "update_s",
        )
    }


def _raw_evaluation(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = _read_json(path)["summary"]
    series = {}
    for round_name in sorted(summary, key=int):
        pooled = summary[round_name]["pooled"]
        route_counts = pooled["route_counts"]
        series[round_name] = {
            "SR": pooled["SR"], "CR": pooled["CR"],
            "OOB": pooled["OOB"], "timeout": pooled["timeout"],
            "validity": pooled["window_validity"],
            "clearance_m": pooled["successful_min_clearance_m"],
            "time_to_goal_s": pooled["successful_time_to_goal_s"],
            "route_counts": route_counts,
            "coverage_modes": sum(int(route_counts[mode]) > 0 for mode in MODES),
        }
    return {
        "path": str(path),
        "series": series,
        "final": series["2"],
        "per_gamma_final": summary["2"]["per_gamma"],
    }


def _marker(control: Path, record: dict[str, Any], kind: str) -> Path:
    return control / "status" / (
        f"{record['name']}--{record['config_hash'][:12]}.{kind}.json"
    )


def _arm_summary(
    record: dict[str, Any], control: Path, local: Path,
    queue_state: str, checkpoint_round: int | None,
) -> dict[str, Any]:
    complete = _marker(control, record, "COMPLETE")
    failed = _marker(control, record, "FAILED")
    running = _marker(control, record, "RUNNING")
    if complete.is_file():
        state = "COMPLETE"
    elif failed.is_file():
        state = "FAILED_CLOSED"
    elif running.is_file() or queue_state == "RUNNING":
        state = "RUNNING"
    elif queue_state == "QUEUED":
        state = "QUEUED"
    else:
        state = queue_state
    specs = list((STAGE / "specs").glob(f"*_{record['config_hash'][:12]}.json"))
    spec = _read_json(specs[0]) if len(specs) == 1 else {}
    log = control / "logs" / (
        f"{record['name']}--{record['config_hash'][:12]}.log"
    )
    evaluation = _raw_evaluation(local / "fixed_eval_r000_r002/raw_eval.json")
    return {
        "name": record["name"],
        "speed_weight": spec.get("scientific_delta", {}).get(
            "execution_obstacle_speed_weight"
        ),
        "cost_band_fraction": spec.get("scientific_delta", {}).get(
            "execution_cost_band_fraction"
        ),
        "physical_gpu": record["physical_gpu"],
        "state": state,
        "queue_state": queue_state,
        "committed_round": checkpoint_round,
        "live": _sample_updates(log),
        "last_committed": _last_metric(local / "metrics.jsonl"),
        "evaluation": evaluation,
        "failure": _read_json(failed) if failed.is_file() else None,
        "local_output": str(local),
        "remote_output": record["remote_output"],
    }


def _support_text(live: dict[str, Any]) -> str:
    pieces = []
    for gamma, row in sorted(live["by_gamma"].items(), key=lambda item: float(item[0])):
        support = row["unguided_support"]
        counts = "/".join(str(support[mode]) for mode in MODES)
        missing = row["missing"]
        missing_text = "none" if not missing else ",".join(
            f"{mode}{count}" for mode, count in missing.items()
        )
        pieces.append(f"g{gamma}:retry{row['retry']},U={counts},miss={missing_text}")
    return "; ".join(pieces) or "—"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PRE2 obstacle-speed + 5% margin-band pilot",
        "",
        "Mode order: `below/above/left/right`. Evaluation is unchanged raw deployment.",
        "",
        "| arm | W/band | GPU | state | r | live retry/support | loss | r2 SR/CR/OOB/TO/V/cov |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for arm in payload["arms"]:
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
                f"{final['timeout']:.3f}/{final['validity']:.3f}/"
                f"{final['coverage_modes']}/4"
            )
        lines.append(
            f"| {arm['name']} | {arm['speed_weight']}/{arm['cost_band_fraction']} | "
            f"{arm['physical_gpu']} | {arm['state']} | {arm['committed_round']} | "
            f"{_support_text(arm['live'])} | {loss_text} | {raw} |"
        )
    for arm in payload["arms"]:
        if arm["evaluation"] is None:
            continue
        lines.extend(["", f"## {arm['name']} raw r0-r2", ""])
        for round_name, row in arm["evaluation"]["series"].items():
            lines.append(
                f"- r{round_name}: SR {row['SR']:.4f}, CR {row['CR']:.4f}, "
                f"OOB {row['OOB']:.4f}, timeout {row['timeout']:.4f}, "
                f"V {row['validity']:.4f}, clr {row['clearance_m']:.4f}m, "
                f"TtG {row['time_to_goal_s']:.3f}s, coverage "
                f"{row['coverage_modes']}/4, routes {row['route_counts']}"
            )
        lines.append("- r2 gamma trend:")
        for gamma, row in sorted(
            arm["evaluation"]["per_gamma_final"].items(),
            key=lambda item: float(item[0]),
        ):
            lines.append(
                f"  - gamma {gamma}: SR {row['SR']:.4f}, CR {row['CR']:.4f}, "
                f"OOB {row['OOB']:.4f}, timeout {row['timeout']:.4f}, "
                f"V {row['window_validity']:.4f}, "
                f"clr {row['successful_min_clearance_m']:.4f}m, "
                f"TtG {row['successful_time_to_goal_s']:.3f}s, "
                f"routes {row['route_counts']}"
            )
    return "\n".join(lines) + "\n"


def snapshot() -> dict[str, Any]:
    queue = _read_json(QUEUE)
    records = queue["records"]
    control = _sync_control(queue)
    queue_states, listings = _queue_states(queue)
    checkpoint_rounds = _remote_checkpoint_rounds(records)
    arms = []
    for record in records:
        local = _sync_arm(record)
        queue_state = queue_states.get(
            (int(record["physical_gpu"]), int(record["tsp_job_id"])),
            "UNKNOWN",
        )
        arms.append(_arm_summary(
            record, control, local, queue_state,
            checkpoint_rounds.get(record["config_hash"]),
        ))
    counts = {
        state: sum(arm["state"] == state for arm in arms)
        for state in sorted({arm["state"] for arm in arms})
    }
    payload = {
        "status": (
            "ALL_COMPLETE" if arms and all(arm["state"] == "COMPLETE" for arm in arms)
            else "TERMINAL_WITH_FAILURE" if arms and all(
                arm["state"] in {"COMPLETE", "FAILED_CLOSED"} for arm in arms
            ) else "MONITORING"
        ),
        "updated_unix": time.time(),
        "raw_evaluation_policy": queue["raw_evaluation_policy"],
        "counts": counts,
        "arms": arms,
        "queue_listings": listings,
    }
    _atomic_json(STAGE / "CURRENT_PROGRESS.json", payload)
    _atomic_text(STAGE / "CURRENT_PROGRESS.md", _markdown(payload))
    return payload


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(snapshot(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
