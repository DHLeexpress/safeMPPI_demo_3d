#!/usr/bin/env python3
"""Build the faithful-M1 speed400 R1/R3/R5/R7/R9 trajectory audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROUNDS = (1, 3, 5, 7, 9)
GAMMAS = (0.1, 0.3, 0.5, 1.0)
STATUSES = ("SUCCESS", "COLLISION", "OOB")
ROUTES = ("LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _encode_states(states: Any) -> dict[str, Any]:
    values = np.asarray(states, np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2:
        raise ValueError(f"states must be [N,6], found {values.shape}")
    positions = np.rint(values[:, :3] * 1000.0).astype(np.int32)
    deltas = np.diff(positions, axis=0).reshape(-1)
    speed = np.rint(np.linalg.norm(values[:, 3:6], axis=1) * 1000.0).astype(
        np.int32
    )
    return {
        "p0": positions[0].tolist(),
        "dp": deltas.tolist(),
        "speed": speed.tolist(),
    }


def _route_code(row: dict[str, Any]) -> str | None:
    route = row.get("bowling_route")
    if not isinstance(route, dict):
        return None
    code = str(route.get("stable_code", ""))
    return code if code in ROUTES else None


def _path_row(row: dict[str, Any], kind: str) -> dict[str, Any]:
    states = np.asarray(row["states"], np.float64)
    result = {
        "id": (
            f"{kind}:r{int(row['round'])}:g{float(row['gamma']):g}:"
            f"e{int(row['episode'])}"
        ),
        "kind": kind,
        "round": int(row["round"]),
        "gamma": float(row["gamma"]),
        "episode": int(row["episode"]),
        "status": str(row["status"]),
        "route": _route_code(row),
        "states": _encode_states(states),
        "steps": int(len(states) - 1),
        "minClearance": row.get("min_clearance_m"),
        "ttg": row.get("time_to_goal_s"),
        "validity": row.get("window_validity"),
        "meanAbsZ09": float(np.mean(np.abs(states[:, 2] - 0.9))),
    }
    if kind == "random":
        result["spheres"] = np.round(
            np.asarray(row["spheres"], np.float64), 4,
        ).tolist()
        result["sceneHash"] = str(row["scene_hash"])[:12]
    return result


def _select_random(raw: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for round_i in ROUNDS:
        rows = raw[str(round_i)]
        for status in STATUSES:
            for gamma in GAMMAS:
                candidates = [
                    row for row in rows
                    if str(row["status"]) == status
                    and np.isclose(float(row["gamma"]), gamma)
                ]
                if candidates:
                    selected.append(_path_row(candidates[0], "random"))
    return selected


def _select_bowling(raw: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for round_i in ROUNDS:
        rows = raw[str(round_i)]
        successes = [row for row in rows if str(row["status"]) == "SUCCESS"]
        for route in ROUTES:
            candidates = [row for row in successes if _route_code(row) == route]
            candidates.sort(key=lambda row: float(np.mean(np.abs(
                np.asarray(row["states"], np.float64)[:, 2] - 0.9
            ))))
            selected.extend(_path_row(row, "bowling") for row in candidates[:3])
        for status in ("COLLISION", "OOB"):
            for gamma in GAMMAS:
                candidates = [
                    row for row in rows
                    if str(row["status"]) == status
                    and np.isclose(float(row["gamma"]), gamma)
                ]
                if candidates:
                    selected.append(_path_row(candidates[0], "bowling"))
    return selected


def _metric_rows(
    random_eval: dict[str, Any], bowling_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for round_i in ROUNDS:
        random = random_eval["summary"][str(round_i)]["pooled"]
        bowling = bowling_eval["summary"][str(round_i)]
        coverage = bowling["bowling"]
        values.append({
            "round": round_i,
            "SR": random["SR"],
            "CR": random["CR"],
            "OOB": random["OOB"],
            "validity": random["window_validity"],
            "clearance": random["successful_min_clearance_m"],
            "ttg": random["successful_time_to_goal_s"],
            "bowlingSR": bowling["pooled"]["SR"],
            "coverage": coverage["coverage_count"],
            "routes": coverage["observed_routes"],
        })
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-eval", type=Path, required=True)
    parser.add_argument("--random-trajectories", type=Path, required=True)
    parser.add_argument("--bowling-eval", type=Path, required=True)
    parser.add_argument("--bowling-trajectories", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    args = parser.parse_args()

    random_eval = _read(args.random_eval)
    bowling_eval = _read(args.bowling_eval)
    random_raw = torch.load(
        args.random_trajectories, map_location="cpu", weights_only=False,
    )
    bowling_raw = torch.load(
        args.bowling_trajectories, map_location="cpu", weights_only=False,
    )
    if tuple(int(value) for value in random_eval["checkpoint_rounds"]) != ROUNDS:
        raise ValueError("random evaluation rounds changed")
    if tuple(int(value) for value in bowling_eval["checkpoint_rounds"]) != ROUNDS:
        raise ValueError("bowling evaluation rounds changed")
    if random_eval["samples_per_step"] != 1 or random_eval["NFE"] != 16:
        raise ValueError("random evaluation is not faithful M1/NFE16")
    if bowling_eval["deployment_contract"]["samples_per_step"] != 1:
        raise ValueError("bowling evaluation is not faithful M1")
    payload = {
        "contract": "faithful raw NFE16 · M1 · speed400 · 200 attempts/checkpoint",
        "dt": 0.1,
        "rounds": list(ROUNDS),
        "gammas": list(GAMMAS),
        "routes": list(ROUTES),
        "start": bowling_eval["scene"]["start"],
        "goal": bowling_eval["scene"]["goal"],
        "bounds": bowling_eval["scene"]["bounds"],
        "bowlingSpheres": bowling_eval["scene"]["spheres"],
        "metrics": _metric_rows(random_eval, bowling_eval),
        "random": _select_random(random_raw),
        "bowling": _select_bowling(bowling_raw),
        "binding": {
            "random": random_eval["artifact_binding"],
            "bowling": bowling_eval["artifact_binding"],
        },
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    template = args.template.read_text()
    marker = "__SPEED400_M1_TRAJECTORY_DATA__"
    if template.count(marker) != 1:
        raise ValueError("template must contain exactly one data marker")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace(marker, encoded))
    if args.data_output is not None:
        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        args.data_output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output.stat().st_size >= 1_000_000:
        raise ValueError(f"visualization exceeds 1 MB: {args.output.stat().st_size}")
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "random_paths": len(payload["random"]),
        "bowling_paths": len(payload["bowling"]),
    }, indent=2))


if __name__ == "__main__":
    main()
