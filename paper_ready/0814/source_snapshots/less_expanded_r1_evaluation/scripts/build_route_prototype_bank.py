#!/usr/bin/env python3
"""Build route/action-window prototypes from committed expansion archives."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.route_prototype_guidance import (
    ROUTE_PROTOTYPE_FORMAT,
    prototype_group_key,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=4)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("--per-group must be positive")

    env = TaskEnvironment(load_config(args.task_config))
    grouped_rows: dict[tuple[str, str], list] = defaultdict(list)
    for archive_path in args.archive:
        for row in torch.load(
            archive_path, map_location="cpu", weights_only=False,
        ):
            if row.sample_update_mode is None or row.trajectory_id is None:
                continue
            source_id = archive_path.parent.name
            grouped_rows[(source_id, str(row.trajectory_id))].append(row)

    candidates: dict[str, list[dict]] = defaultdict(list)
    for (source_id, trajectory_id), rows in grouped_rows.items():
        rows.sort(key=lambda row: int(row.window_start))
        starts = [int(row.window_start) for row in rows]
        if starts != list(range(len(rows))):
            continue
        first = rows[0]
        route = int(first.sample_update_mode)
        gamma = float(first.gamma)
        if any(
            int(row.sample_update_mode) != route or float(row.gamma) != gamma
            for row in rows
        ):
            raise ValueError(f"mixed route/gamma trajectory {trajectory_id}")
        contexts = torch.stack([row.context.float() for row in rows])
        states = torch.cat([
            torch.as_tensor(env.goal, dtype=torch.float32)[None]
            - contexts[:, :3],
            contexts[:, 3:6],
        ], dim=1)
        action_windows = torch.stack([row.candidate.float() for row in rows])
        valid_horizons = torch.as_tensor([
            int(row.valid_horizon) for row in rows
        ], dtype=torch.long)
        candidates[prototype_group_key(gamma, route)].append({
            "trajectory_id": f"{source_id}:{trajectory_id}",
            "round": int(first.round),
            "states": states,
            "action_windows": action_windows,
            "valid_horizons": valid_horizons,
        })

    groups = {}
    for key, rows in sorted(candidates.items()):
        rows.sort(key=lambda row: (-row["round"], row["trajectory_id"]))
        groups[key] = rows[:args.per_group]
    expected = {
        prototype_group_key(gamma, route)
        for gamma in (0.1, 0.3, 0.5, 1.0)
        for route in range(4)
    }
    missing = sorted(expected.difference(groups))
    if missing:
        raise RuntimeError(f"prototype bank is missing groups: {missing}")
    payload = {
        "format": ROUTE_PROTOTYPE_FORMAT,
        "guidance_contract": (
            "nearest (position,velocity) success state; score the complete "
            "verified candidate action window against its executed suffix"
        ),
        "source_archives": [str(path) for path in args.archive],
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"wrote {args.output}: {len(groups)} groups, "
        f"{sum(map(len, groups.values()))} prototypes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
