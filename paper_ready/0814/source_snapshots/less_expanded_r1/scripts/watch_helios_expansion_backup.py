#!/usr/bin/env python3
"""Continuously mirror committed expansion artifacts without touching its job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


HOST = "dohyun@helios.robotics.caltech.edu"
PROGRESS_PATTERN = re.compile(
    r"\[sample-update\] r(?P<round>\d+) gamma (?P<gamma>[0-9.eE+-]+) "
    r"retry batch (?P<retry>\d+)/(?P<cap>\d+).*?"
    r"need mode \"(?P<needed>[0-3,]*)\""
)


def _parse_progress(log_text: str) -> dict:
    rows = []
    for match in PROGRESS_PATTERN.finditer(log_text):
        rows.append({
            "round": int(match.group("round")),
            "gamma": float(match.group("gamma")),
            "retry_batch": int(match.group("retry")),
            "retry_batch_cap": int(match.group("cap")) + 1,
            "needed_modes": [
                int(value) for value in match.group("needed").split(",")
                if value
            ],
        })
    if not rows:
        return {
            "current_round": None,
            "current_max_retry_batch": None,
            "retry_by_gamma": {},
        }
    current_round = max(row["round"] for row in rows)
    retry_by_gamma = {}
    for row in rows:
        if row["round"] == current_round:
            retry_by_gamma[f"{row['gamma']:.9g}"] = row
    return {
        "current_round": current_round,
        "current_max_retry_batch": max(
            row["retry_batch"] for row in retry_by_gamma.values()
        ),
        "retry_by_gamma": retry_by_gamma,
    }


def _remote_status(remote_output: str) -> dict:
    quoted = shlex.quote(remote_output)
    command = (
        f"d={quoted}; "
        "latest=$(find \"$d\" -maxdepth 1 -name 'checkpoint_[0-9][0-9][0-9].pt' "
        "-print 2>/dev/null | sort | tail -1); "
        "printf '%s\\n' \"${latest##*/}\"; "
        "test -s \"$d/manifest.json\" && "
        "! test -e \"$d/RESUME_IN_PROGRESS.json\" && "
        "printf 'complete\\n' || printf 'running\\n'; "
        "printf '%s\\n' '__PROCESS__'; "
        "ps -eo comm=,args= | awk -v d=\"$d\" "
        "'$1 ~ /python/ && index($0, \"research_ball_expansion_optimization.py\") "
        "&& index($0, \"--output \" d) {print; exit}'; "
        "printf '%s\\n' '__LOG__'; "
        "log=\"${d}.helios.log\"; "
        "test -s \"$log\" || log=\"$d/helios.log\"; "
        "tail -n 5000 \"$log\" 2>/dev/null | "
        "grep -E '\\[sample-update\\]|\\[first-action\\]|"
        "Traceback|RuntimeError|\\[expansion\\]' | tail -n 1000 || true"
    )
    raw = subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=20", HOST, command], text=True,
    )
    metadata, remainder = raw.split("__PROCESS__\n", 1)
    process_text, log_text = remainder.split("__LOG__\n", 1)
    metadata_lines = metadata.splitlines()
    checkpoint = metadata_lines[0] if metadata_lines else ""
    process_line = process_text.strip()
    target_match = re.search(r"(?:^| )--rounds (\d+)(?: |$)", process_line)
    result = {
        "latest_checkpoint": checkpoint or None,
        "completed": len(metadata_lines) > 1 and metadata_lines[1] == "complete",
        "process_alive": bool(process_line),
        "target_rounds": int(target_match.group(1)) if target_match else None,
    }
    result.update(_parse_progress(log_text))
    return result


def _local_metrics(local_backup: Path) -> dict:
    path = local_backup / "metrics.jsonl"
    if not path.is_file():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    row = json.loads(lines[-1])
    return {
        "committed_round": int(row["round"]),
        "optimizer_step": int(row.get("optimizer_step", 0)),
        "last_round_total_s": float(row.get("round_total_s", 0.0)),
        "last_round_retry_batches_by_gamma": row.get(
            "retry_batches_by_gamma", {}
        ),
        "replay_positives": int(row.get("replay_positives", 0)),
        "positive_loss": row.get("positive_loss"),
        "learning_rate": row.get("learning_rate"),
    }


def _write_live_markdown(local_backup: Path, payload: dict) -> None:
    progress = payload.get("retry_by_gamma", {})
    lines = [
        "# Live expansion status",
        "",
        f"- committed: {payload.get('committed_round')} / "
        f"{payload.get('target_rounds')}",
        f"- current round: {payload.get('current_round')}",
        f"- max retry batch: {payload.get('current_max_retry_batch')}",
        f"- optimizer step: {payload.get('optimizer_step')}",
        f"- process alive: {payload.get('process_alive')}",
        f"- exact resume available: {payload.get('exact_resume_available')}",
        "",
        "| gamma | retry | needed modes |",
        "|---:|---:|:---|",
    ]
    for gamma, row in sorted(progress.items(), key=lambda item: float(item[0])):
        needed = ",".join(map(str, row["needed_modes"])) or "complete"
        lines.append(f"| {gamma} | {row['retry_batch']} | {needed} |")
    path = local_backup / "LIVE_STATUS.md"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def _sync(remote_output: str, local_backup: Path) -> None:
    local_backup.mkdir(parents=True, exist_ok=True)
    include = [
        "checkpoint_*.pt", "query_archive_round_*.pt", "metrics.jsonl",
        "resume_state_latest.pt", "resume_state.json", "events_round_*.pt",
        "fa_alloc_log.json", "first_action_stats.json", "manifest.json",
        "manifest_before_resume_round_*.json", "RESUME_IN_PROGRESS.json",
        "query_archive.pt", "gp_evidence.pt", "events.pt", "FAILED.json",
        "task_config_resolved.json", "RECIPE.sh", "command.sh",
    ]
    command = ["rsync", "-az"]
    for pattern in include:
        command.extend(["--include", pattern])
    command.extend([
        "--exclude", ".*.tmp", "--exclude", "*",
        f"{HOST}:{remote_output}/", f"{local_backup}/",
    ])
    subprocess.run(command, check=True)


def _write_status(local_backup: Path, payload: dict) -> None:
    path = local_backup / "BACKUP_STATUS.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--local-backup", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=240.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 30.0 and not args.once:
        parser.error("--interval-seconds must be at least 30")

    while True:
        try:
            status = _remote_status(args.remote_output)
            _sync(args.remote_output, args.local_backup)
            status.update(_local_metrics(args.local_backup))
            status.update({
                "remote_output": args.remote_output,
                "last_sync_unix": time.time(),
                "backup_semantics": (
                    "exact Adam/RNG resume when resume_state_latest.pt exists; "
                    "otherwise model+archive cold recovery only"
                ),
                "exact_resume_available": (
                    args.local_backup / "resume_state_latest.pt"
                ).is_file(),
            })
            _write_status(args.local_backup, status)
            _write_live_markdown(args.local_backup, status)
            print(
                f"[backup] committed r{status.get('committed_round')} "
                f"current r{status.get('current_round')} retry "
                f"{status.get('current_max_retry_batch')} -> {args.local_backup}",
                flush=True,
            )
            if args.once or status["completed"]:
                return 0
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"[backup] transient sync failure: {error}", flush=True)
            if args.once:
                return 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
