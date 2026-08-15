#!/usr/bin/env python3
"""Render committed legacy/band/speed150 trajectories as a 3-D fragment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = Path(
    "/Users/dhl/.codex/visualizations/2026/08/10/"
    "019fe90f-b8eb-7f52-bd16-2bb83e11672e/"
    "speed-band-acquisition-comparison.html"
)
TASK = ROOT / (
    "configs/lab_ball_stage1_goalspace_yminus04_z01_17_"
    "r15in_reach03_v1.json"
)
PILOT = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_obstacle_speed_band_pilot/"
    "acquisition_archives"
)
LEGACY = ROOT / (
    "results/stage1_single_ball_t128/0811_pre2_r1_inward_oob_probe/"
    "reach03_continuations/"
    "fresh_r1_exp15_goalbox100_t150_s82410_reach03_r1_to_r5"
)
TEMPLATE = ROOT / "assets/templates/speed-band-acquisition-fragment.html"
MODE_NAMES = {0: "below", 1: "above", 2: "left", 3: "right"}


def _archive_path(arm: str, round_i: int) -> Path:
    if arm == "legacy":
        root = LEGACY
    elif arm == "band05":
        root = PILOT / "speed000"
    elif arm == "speed150":
        root = PILOT / "speed150"
    else:
        raise KeyError(arm)
    return root / f"query_archive_round_{round_i:03d}.pt"


def _trajectory_rows(path: Path, goal: np.ndarray, sphere: np.ndarray) -> list[dict[str, Any]]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        if not record.replay_eligible or record.trajectory_id is None:
            continue
        grouped[str(record.trajectory_id)].append(record)

    rows = []
    for trajectory_id, values in sorted(grouped.items()):
        values.sort(key=lambda record: int(record.window_start))
        starts = [int(record.window_start) for record in values]
        if starts != list(range(len(starts))):
            raise ValueError(f"non-contiguous committed windows: {trajectory_id}")
        modes = {record.sample_update_mode for record in values}
        gammas = {float(record.gamma) for record in values}
        retries = {int(record.retry_batch) for record in values}
        if len(modes) != 1 or len(gammas) != 1 or len(retries) != 1:
            raise ValueError(f"trajectory metadata changes within {trajectory_id}")
        mode = MODE_NAMES[int(modes.pop())]
        points = np.stack([
            goal - record.context[:3].detach().cpu().numpy()
            for record in values
        ])
        velocity = np.stack([
            record.context[3:6].detach().cpu().numpy()
            for record in values
        ])
        clearance = np.linalg.norm(points - sphere[:3], axis=1) - sphere[3]
        rows.append({
            "id": trajectory_id,
            "gamma": gammas.pop(),
            "mode": mode,
            "retry": retries.pop(),
            "points": np.round(points, 4).tolist(),
            "minClearance": round(float(clearance.min()), 4),
            "meanSpeed": round(float(np.linalg.norm(velocity, axis=1).mean()), 4),
            "length": round(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), 4),
            "terminalDistance": round(float(np.linalg.norm(points[-1] - goal)), 4),
        })
    if len(rows) != 48:
        raise ValueError(f"expected 48 committed trajectories in {path}, found {len(rows)}")
    counts = defaultdict(int)
    for row in rows:
        counts[(row["gamma"], row["mode"])] += 1
    if set(counts.values()) != {3} or len(counts) != 16:
        raise ValueError(f"committed quota is not 3 per mode/gamma: {dict(counts)}")
    return rows


def build(output: Path) -> dict[str, Any]:
    task = json.loads(TASK.read_text())
    origin = np.asarray(task["taskspace"]["origin"], float)
    size = np.asarray(task["taskspace"]["size"], float)
    goal = np.asarray(task["taskspace"]["goal"], float)
    sphere = np.asarray(task["obstacles"]["spheres"][0], float)
    arms = {
        "legacy": {
            "label": "Legacy E15 · minimum execution cost",
            "short": "legacy E15",
        },
        "band05": {
            "label": "5% cost band · max step margin",
            "short": "band05",
        },
        "speed150": {
            "label": "5% band + obstacle-speed W150",
            "short": "speed150",
        },
    }
    for arm in arms:
        arms[arm]["rounds"] = {
            str(round_i): _trajectory_rows(
                _archive_path(arm, round_i), goal, sphere,
            )
            for round_i in (1, 2)
        }
    payload = {
        "armOrder": ["legacy", "band05", "speed150"],
        "modeOrder": ["below", "above", "left", "right"],
        "arms": arms,
        "bounds": np.stack([origin, origin + size], axis=1).round(4).tolist(),
        "start": task["taskspace"]["start"][:3],
        "goal": task["taskspace"]["goal"],
        "sphere": task["obstacles"]["spheres"][0],
    }
    fragment = TEMPLATE.read_text().replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")),
    )
    if '\\"' in fragment or "\\n" in fragment:
        raise ValueError("fragment contains escaped markup")
    if len(fragment.encode()) >= 1024 * 1024:
        raise ValueError("interactive fragment exceeds 1 MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "arms": list(arms),
        "rounds": [1, 2],
        "trajectories_per_arm_round": 48,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
