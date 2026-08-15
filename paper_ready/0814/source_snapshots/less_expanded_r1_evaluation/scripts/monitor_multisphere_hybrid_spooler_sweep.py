#!/usr/bin/env python3
"""Mirror and summarize the GPU0 multi-sphere hybrid task-spooler sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_hybrid_spooler_sweep"
QUEUE = STAGE / "QUEUE.json"
HOST = "dohyun@helios.robotics.caltech.edu"
CUTOFF_SECONDS = 6 * 60 * 60


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _ssh(command: str) -> str:
    return subprocess.check_output(["ssh", HOST, command], text=True)


def _tsp_state(socket: str, job_id: int) -> str:
    command = (
        f"export TS_SOCKET={shlex.quote(socket)}; "
        f"tsp -s {int(job_id)}"
    )
    return _ssh(command).strip().lower()


def _mirror_control(queue: dict[str, Any]) -> None:
    mirror = STAGE / "remote_mirror/control"
    mirror.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "--delete", "--prune-empty-dirs", "-e", "ssh",
        "--include=/status/", "--include=/status/*.json",
        "--include=/logs/", "--include=/logs/*.log", "--exclude=*",
        f"{HOST}:{queue['remote_control']}/", f"{mirror}/",
    ], check=True)


def _mirror_arm(record: dict[str, Any]) -> None:
    destination = STAGE / "remote_mirror/arms" / record["name"]
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "--prune-empty-dirs", "-e", "ssh",
        "--include=/resume_state.json", "--include=/manifest.json",
        "--include=/metrics.jsonl", "--include=/FAILED.json",
        "--include=/SPOOL_PROVENANCE.json", "--include=/checkpoint_005.pt",
        "--include=/eval_m50_m8_hybrid_r0_r5/",
        "--include=/eval_m50_m8_hybrid_r0_r5/raw_eval.json",
        "--exclude=*", f"{HOST}:{record['remote_output']}/", f"{destination}/",
    ], check=True)


def _manifest_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        rounds = value.get("rounds", [])
        if not rounds:
            return {}
        latest = rounds[-1]
        by_gamma = latest.get("successful_executed_commit_by_gamma", {})
        committed = {
            str(gamma): int(row.get("selected_trajectory_count", row.get(
                "committed_trajectory_count", len(row.get("selected_trajectory_ids", [])),
            )))
            for gamma, row in by_gamma.items() if isinstance(row, dict)
        }
        if not any(committed.values()):
            ids = latest.get("committed_trajectory_ids", [])
            committed = {
                gamma: sum(f":g{gamma}:" in str(identifier) for identifier in ids)
                for gamma in ("0.1", "0.3", "0.5", "1")
            }
        cumulative_by_gamma: dict[str, int] = {}
        for round_row in rounds:
            for gamma, detail in round_row.get(
                "successful_executed_commit_by_gamma", {}
            ).items():
                cumulative_by_gamma[str(gamma)] = (
                    cumulative_by_gamma.get(str(gamma), 0)
                    + int(detail.get("committed_trajectory_count", 0))
                )
        return {
            "latest_committed_round": int(latest.get("round", 0)),
            "committed_trajectories_by_gamma": committed,
            "committed_trajectory_count": len(latest.get("committed_trajectory_ids", [])),
            "cumulative_committed_trajectories_by_gamma": cumulative_by_gamma,
            "cumulative_committed_trajectory_count": sum(
                cumulative_by_gamma.values()
            ),
            "retry_batches_by_gamma": latest.get("retry_batches_by_gamma", {}),
            "attempted_episode_count": latest.get("attempted_episode_count"),
            "quota_complete_by_gamma": {
                str(gamma): detail.get("quota_complete")
                for gamma, detail in by_gamma.items()
            },
            "exact_quota_complete": bool(by_gamma) and all(
                detail.get("quota_complete") is True
                for detail in by_gamma.values()
            ),
            "round_transaction_committed": latest.get(
                "round_success_commit_complete"
            ),
            "optimizer_step": latest.get("optimizer_step"),
            "positive_loss": latest.get("positive_loss"),
            "round_total_s": latest.get("round_total_s"),
        }
    except (OSError, ValueError, TypeError):
        return {"manifest_parse_error": True}


def _evaluation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        summaries = value.get("summary", {})
        latest_round = max(map(int, summaries))
        latest = summaries[str(latest_round)]
        pooled = latest["pooled"]
        return {
            "latest_evaluated_round": latest_round,
            "pooled": {key: pooled.get(key) for key in (
                "episodes", "SR", "CR", "OOB", "timeout", "window_validity",
                "successful_min_clearance_m", "successful_time_to_goal_s",
                "status_counts",
            )},
            "per_gamma": {
                gamma: {key: row.get(key) for key in (
                    "SR", "CR", "OOB", "timeout", "window_validity",
                    "successful_min_clearance_m", "successful_time_to_goal_s",
                    "status_counts",
                )}
                for gamma, row in latest.get("per_gamma", {}).items()
            },
        }
    except (OSError, ValueError, TypeError, KeyError):
        return {"evaluation_parse_error": True}


def _remove_unstarted_at_cutoff(queue: dict[str, Any]) -> list[dict[str, Any]]:
    marker = STAGE / "SIX_HOUR_CUTOFF.json"
    if marker.is_file() or time.time() - float(queue["created_unix"]) < CUTOFF_SECONDS:
        return []
    removed = []
    candidates = [{
        "name": record["name"],
        "job_id": int(record["tsp_job_id"]),
        "socket": record["tsp_socket"],
        "config_hash": record["config_hash"],
        "kind": "expansion_arm",
    } for record in queue["records"]]
    recovery_path = STAGE / "EVAL_RECOVERY_QUEUE.json"
    if recovery_path.is_file():
        recovery = json.loads(recovery_path.read_text())
        candidates.extend({
            "name": record["name"],
            "job_id": int(record["recovery_job_id"]),
            "socket": queue["tsp_sockets"]["0"],
            "config_hash": record["config_hash"],
            "kind": "evaluation_recovery",
        } for record in recovery.get("records", []))
    for record in candidates:
        state = _tsp_state(record["socket"], record["job_id"])
        if state != "queued":
            continue
        command = (
            f"export TS_SOCKET={shlex.quote(record['socket'])}; "
            f"tsp -r {int(record['job_id'])}"
        )
        _ssh(command)
        removed.append({
            "name": record["name"],
            "tsp_job_id": int(record["job_id"]),
            "config_hash": record["config_hash"],
            "kind": record["kind"],
        })
    payload = {
        "status": "SIX_HOUR_GPU0_WINDOW_CLOSED",
        "created_unix": time.time(),
        "policy": "running jobs preserved; only never-started queued jobs removed",
        "removed_unstarted": removed,
    }
    _write(marker, payload)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enforce-six-hour-cutoff", action="store_true")
    args = parser.parse_args()
    queue = json.loads(QUEUE.read_text())
    if queue.get("gpu_policy", "").find("GPU0 only") < 0:
        raise RuntimeError("refusing to monitor a non-GPU0-only queue")
    _mirror_control(queue)
    tsp_lists = {
        gpu: _ssh(
            f"export TS_SOCKET={shlex.quote(socket)}; tsp -l"
        )
        for gpu, socket in queue["tsp_sockets"].items()
    }
    states: dict[str, int] = {}
    marker_states: dict[str, int] = {}
    arms = []
    for record in queue["records"]:
        state = _tsp_state(record["tsp_socket"], record["tsp_job_id"])
        states[state] = states.get(state, 0) + 1
        if state in {"running", "finished", "failed", "skipped"}:
            _mirror_arm(record)
        local = STAGE / "remote_mirror/arms" / record["name"]
        marker_stem = (
            f"{record['name']}--{record['config_hash'][:12]}"
        )
        status_dir = STAGE / "remote_mirror/control/status"
        marker_state = "no_marker"
        marker_payload: dict[str, Any] = {}
        for suffix, candidate_state in (
            ("COMPLETE", "complete"),
            ("RUNNING", "running"),
            ("FAILED", "failed_closed"),
        ):
            marker = status_dir / f"{marker_stem}.{suffix}.json"
            if marker.is_file():
                marker_state = candidate_state
                try:
                    marker_payload = json.loads(marker.read_text())
                except (OSError, ValueError):
                    marker_payload = {"marker_parse_error": True}
                break
        marker_states[marker_state] = marker_states.get(marker_state, 0) + 1
        progress = _manifest_progress(local / "manifest.json")
        evaluation = _evaluation(
            local / "eval_m50_m8_hybrid_r0_r5/raw_eval.json"
        )
        arms.append({
            "name": record["name"],
            "config_hash": record["config_hash"],
            "tsp_job_id": record["tsp_job_id"],
            "state": state,
            "marker_state": marker_state,
            "failure_error": marker_payload.get("error"),
            **progress,
            **evaluation,
        })
    removed = (
        _remove_unstarted_at_cutoff(queue)
        if args.enforce_six_hour_cutoff else []
    )
    recovery_states: dict[str, int] = {}
    recovery_path = STAGE / "EVAL_RECOVERY_QUEUE.json"
    if recovery_path.is_file():
        recovery = json.loads(recovery_path.read_text())
        socket = queue["tsp_sockets"]["0"]
        for record in recovery.get("records", []):
            state = _tsp_state(socket, int(record["recovery_job_id"]))
            recovery_states[state] = recovery_states.get(state, 0) + 1
    completed = [arm for arm in arms if arm.get("latest_evaluated_round") == 5]
    for arm in completed:
        pooled = arm["pooled"]
        arm["ranking_key"] = [
            pooled.get("SR", 0.0) < 0.15,
            pooled.get("CR", 1.0),
            pooled.get("OOB", 1.0) + pooled.get("timeout", 1.0),
            -pooled.get("window_validity", 0.0),
            -pooled.get("SR", 0.0),
            -(pooled.get("successful_min_clearance_m") or -1.0),
            pooled.get("successful_time_to_goal_s") or 1.0e9,
        ]
    completed.sort(key=lambda arm: arm["ranking_key"])
    payload = {
        "schema_version": 1,
        "updated_unix": time.time(),
        "elapsed_seconds": time.time() - float(queue["created_unix"]),
        "gpu_policy": queue["gpu_policy"],
        "state_counts": states,
        "marker_state_counts": marker_states,
        "evaluation_recovery_state_counts": recovery_states,
        "tsp_lists": tsp_lists,
        "arms": arms,
        "completed_r5_eval_count": len(completed),
        "provisional_winner": completed[0]["name"] if completed else None,
        "cutoff_removed_this_call": removed,
    }
    _write(STAGE / "CURRENT_PROGRESS.json", payload)
    print(json.dumps({
        key: payload[key] for key in (
            "updated_unix", "elapsed_seconds", "state_counts",
            "marker_state_counts",
            "evaluation_recovery_state_counts",
            "completed_r5_eval_count", "provisional_winner",
            "cutoff_removed_this_call",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
