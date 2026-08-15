#!/usr/bin/env python3
"""Extract bounded terminal-SUCCESS traces from a failed expansion event log."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_theta import theta_name, trajectory_crossing_theta
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


MODE_NAMES = ("below", "above", "left", "right")


def _episode_key(event: dict) -> tuple[int, float, int]:
    return int(event["round"]), float(event["gamma"]), int(event["episode"])


def _trajectory_mode(env: TaskEnvironment, events: list[dict]) -> str:
    positions = np.concatenate([
        np.asarray(env.start[:3], np.float32)[None],
        np.stack([
            np.asarray(event["robot_after"], np.float32)[:3]
            for event in events
        ]),
    ])
    return theta_name(trajectory_crossing_theta(env, positions))


def _authoritative_counts(failed: dict, round_index: int) -> dict:
    rows = [
        row for row in failed.get("fa_alloc_diagnostics", {}).get(
            "retry_progress", []
        )
        if int(row.get("round", -1)) == round_index
    ]
    latest: dict[float, dict] = {}
    for row in rows:
        gamma = float(row["gamma"])
        if (
            gamma not in latest
            or int(row["retry_batch"]) > int(latest[gamma]["retry_batch"])
        ):
            latest[gamma] = row
    return {
        f"{gamma:g}": {
            cohort: {
                MODE_NAMES[index]: int(count)
                for index, count in enumerate(row[f"{cohort}_mode_counts"])
            }
            for cohort in ("guided", "unguided")
        }
        for gamma, row in sorted(latest.items())
    }


def extract(
    *,
    failed_events: Path,
    failed_json: Path,
    task_config: Path,
    round_index: int,
    output: Path,
    per_stratum_cap: int,
) -> dict:
    events = torch.load(failed_events, map_location="cpu", weights_only=False)
    success_keys = {
        _episode_key(event)
        for event in events
        if int(event["round"]) == round_index and event.get("status") == "SUCCESS"
    }
    grouped: dict[tuple[int, float, int], list[dict]] = defaultdict(list)
    for event in events:
        key = _episode_key(event)
        if key in success_keys:
            grouped[key].append(event)
    del events

    env = TaskEnvironment(load_config(task_config))
    classified: dict[tuple[float, str, str], list[tuple[tuple, list[dict]]]] = (
        defaultdict(list)
    )
    rejected: list[dict] = []
    for key, trace in grouped.items():
        trace.sort(key=lambda event: int(event["step"]))
        steps = [int(event["step"]) for event in trace]
        if steps != list(range(len(trace))) or trace[-1].get("status") != "SUCCESS":
            rejected.append({"key": list(key), "reason": "non_contiguous_trace"})
            continue
        mode = _trajectory_mode(env, trace)
        cohort = str(trace[-1].get("sample_update_cohort", "unknown"))
        classified[(key[1], cohort, mode)].append((key, trace))

    selected: dict[tuple[int, float, int], list[dict]] = {}
    available_counts = Counter()
    selected_counts = Counter()
    for stratum, traces in sorted(classified.items()):
        traces.sort(key=lambda item: (
            int(item[1][-1].get("retry_batch", -1)), item[0][2]
        ))
        available_counts[stratum] = len(traces)
        for key, trace in traces[:per_stratum_cap]:
            selected[key] = trace
            selected_counts[stratum] += 1

    failed = json.loads(failed_json.read_text())
    authoritative = _authoritative_counts(failed, round_index)
    derived = {
        f"{gamma:g}": {
            cohort: {
                mode: int(available_counts[(gamma, cohort, mode)])
                for mode in (*MODE_NAMES, "none")
            }
            for cohort in ("guided", "unguided")
        }
        for gamma in sorted({key[0] for key in available_counts})
    }
    metadata = {
        "status": "FAILED_ROUND_SUCCESS_TRACES_EXTRACTED",
        "source_failed_events": str(failed_events),
        "source_failed_json": str(failed_json),
        "round": round_index,
        "per_stratum_cap": per_stratum_cap,
        "terminal_success_episodes": len(success_keys),
        "selected_episodes": len(selected),
        "selected_event_count": sum(map(len, selected.values())),
        "rejected": rejected,
        "mode_classifier": (
            "trajectory_crossing_theta over start plus sparse robot_after states; "
            "audit-only until checked against authoritative retry counts"
        ),
        "authoritative_commit_capable_counts": authoritative,
        "derived_terminal_success_counts": derived,
        "selected_counts": {
            f"gamma={gamma:g},cohort={cohort},mode={mode}": int(count)
            for (gamma, cohort, mode), count in sorted(selected_counts.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": metadata,
        "trajectories": selected,
    }, output)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-events", type=Path, required=True)
    parser.add_argument("--failed-json", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum-cap", type=int, default=12)
    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round must be positive")
    if args.per_stratum_cap < 1:
        parser.error("--per-stratum-cap must be positive")
    metadata = extract(
        failed_events=args.failed_events,
        failed_json=args.failed_json,
        task_config=args.task_config,
        round_index=args.round,
        output=args.output,
        per_stratum_cap=args.per_stratum_cap,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
