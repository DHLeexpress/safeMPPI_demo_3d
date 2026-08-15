#!/usr/bin/env python3
"""Snapshot and selectively mirror the 0812 Helios task-spooler sweep."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_margin_hybrid_spooler_sweep"
)
HOST = "dohyun@helios.robotics.caltech.edu"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _ssh(command: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, command],
        text=True,
    )


def _sync_status(remote_control: str) -> Path:
    local = STAGE / "remote_status"
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh",
        f"{HOST}:{remote_control}/status/", f"{local}/",
    ], check=True)
    return local


def _queue_states(queue: dict[str, Any]) -> tuple[dict[tuple[int, int], str], dict[str, str]]:
    states: dict[tuple[int, int], str] = {}
    listings: dict[str, str] = {}
    for gpu_text, socket in queue["tsp_sockets"].items():
        gpu = int(gpu_text)
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
                states[(gpu, int(match.group(1)))] = match.group(2).upper()
    return states, listings


def _remote_progress(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    snippets = []
    for index, record in enumerate(records):
        directory = shlex.quote(record["remote_output"])
        snippets.append(
            f"d={directory}; latest=''; "
            "if test -d \"$d\"; then "
            "latest=$(find \"$d\" -maxdepth 1 -type f -name 'checkpoint_*.pt' "
            "-printf '%f\\n' | sort | tail -1); fi; "
            f"printf '{index}|%s\\n' \"$latest\""
        )
    if not snippets:
        return {}
    output = _ssh("set -euo pipefail; " + "; ".join(snippets))
    result: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        index_text, checkpoint = line.split("|", 1)
        record = records[int(index_text)]
        round_value = None
        match = re.fullmatch(r"checkpoint_(\d{3}).pt", checkpoint)
        if match:
            round_value = int(match.group(1))
        result[record["config_hash"]] = {
            "latest_checkpoint": checkpoint or None,
            "committed_round": round_value,
        }
    return result


def _marker(status_dir: Path, record: dict[str, Any], kind: str) -> Path:
    return status_dir / (
        f"{record['name']}--{record['config_hash'][:12]}.{kind}.json"
    )


def _sync_compact_arm(record: dict[str, Any], rounds: int) -> Path:
    basename = Path(record["remote_output"]).name
    local = STAGE / "arms" / basename
    local.mkdir(parents=True, exist_ok=True)
    evaluation = f"fixed_eval_r000_r{rounds:03d}"
    includes = [
        "checkpoint_*.pt", "query_archive_round_*.pt", "metrics.jsonl",
        "resume_state_latest.pt", "resume_state.json", "query_archive.pt",
        "gp_evidence.pt", "fa_alloc_log.json", "first_action_stats.json",
        "manifest.json", "task_config_resolved.json", "SPOOL_PROVENANCE.json",
        "FAILED.json", f"{evaluation}/", f"{evaluation}/raw_eval.json",
        f"{evaluation}/raw_trajectories.pt", f"{evaluation}/*.png",
        f"{evaluation}/*.pdf", f"{evaluation}/*.jsonl",
    ]
    command = ["rsync", "-az", "--prune-empty-dirs"]
    for pattern in includes:
        command += ["--include", pattern]
    command += [
        "--exclude", "*", "-e", "ssh",
        f"{HOST}:{record['remote_output']}/", f"{local}/",
    ]
    subprocess.run(command, check=True)
    _atomic_json(local / ".helios_spooler.json", {
        "status": "COMPACT_MIRROR_COMPLETE", "host": HOST,
        "physical_gpu": record["physical_gpu"],
        "config_hash": record["config_hash"],
        "remote_output": record["remote_output"],
        "excluded_large_artifacts": ["events.pt", "events_round_*.pt"],
        "synced_unix": time.time(),
    })
    return local


def snapshot(*, sync_complete: bool) -> dict[str, Any]:
    queue_path = STAGE / "QUEUE.json"
    if not queue_path.is_file():
        smoke = STAGE / "SMOKE_QUEUE.json"
        if not smoke.is_file():
            raise FileNotFoundError("QUEUE.json/SMOKE_QUEUE.json is missing")
        queue_path = smoke
    queue = _read_json(queue_path)
    records = queue.get("records", [])
    status_dir = _sync_status(queue["remote_control"])
    queue_states, listings = _queue_states(queue)
    progress = _remote_progress(records)
    arms = []
    for record in records:
        complete_marker = _marker(status_dir, record, "COMPLETE")
        failed_marker = _marker(status_dir, record, "FAILED")
        running_marker = _marker(status_dir, record, "RUNNING")
        queue_state = queue_states.get(
            (int(record["physical_gpu"]), int(record["tsp_job_id"])),
            "UNKNOWN",
        )
        if complete_marker.is_file():
            state = "COMPLETE"
        elif failed_marker.is_file():
            state = "FAILED_CLOSED"
        elif running_marker.is_file() or queue_state == "RUNNING":
            state = "RUNNING"
        elif queue_state == "QUEUED":
            state = "QUEUED"
        elif queue_state == "FINISHED":
            state = "FINISHED_WITHOUT_TERMINAL_MARKER"
        else:
            state = "UNKNOWN"
        local_output = None
        if sync_complete and state == "COMPLETE":
            local_output = str(_sync_compact_arm(record, 1 if queue.get("smoke") else 8))
        arms.append({
            **record, "state": state, "queue_state": queue_state,
            **progress.get(record["config_hash"], {}),
            "local_output": local_output,
            "failure": _read_json(failed_marker) if failed_marker.is_file() else None,
        })
    counts = {
        state: sum(arm["state"] == state for arm in arms)
        for state in sorted({arm["state"] for arm in arms})
    }
    payload = {
        "status": (
            "ALL_COMPLETE" if arms and all(arm["state"] == "COMPLETE" for arm in arms)
            else "TERMINAL_WITH_FAILURE" if arms and all(
                arm["state"] in {"COMPLETE", "FAILED_CLOSED", "FINISHED_WITHOUT_TERMINAL_MARKER"}
                for arm in arms
            ) else "MONITORING"
        ),
        "updated_unix": time.time(), "queue_file": str(queue_path),
        "counts": counts, "arms": arms, "queue_listings": listings,
    }
    _atomic_json(STAGE / "CURRENT_PROGRESS.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(snapshot(sync_complete=args.sync_complete), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
