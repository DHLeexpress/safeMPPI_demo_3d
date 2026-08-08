#!/usr/bin/env python3
"""Re-run Reserve G successes and measure monotone-progress trajectory quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def turn_angles_deg(displacements: np.ndarray) -> np.ndarray:
    if len(displacements) < 2:
        return np.empty(0, dtype=float)
    first = displacements[:-1]
    second = displacements[1:]
    denom = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid = denom > 1.0e-10
    cosine = np.sum(first[valid] * second[valid], axis=1) / denom[valid]
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile)) if len(values) else 0.0


def quality_metrics(result: dict, goal: np.ndarray) -> dict:
    states = np.asarray(result["states"][:, :3], dtype=float)
    controls = np.asarray(result["controls"], dtype=float)
    applied = np.asarray(result["applied_controls"], dtype=float)
    distance = np.linalg.norm(states - goal[None, :], axis=1)
    progress = distance[:-1] - distance[1:]
    displacements = np.diff(states, axis=0)
    step_length = np.linalg.norm(displacements, axis=1)
    turns = turn_angles_deg(displacements)
    applied_delta = np.linalg.norm(np.diff(applied, axis=0), axis=1)
    raw_delta = np.linalg.norm(np.diff(controls, axis=0), axis=1)

    tail_steps = min(10, len(displacements))
    tail_start = len(states) - tail_steps - 1
    tail_length = float(step_length[-tail_steps:].sum()) if tail_steps else 0.0
    tail_displacement = float(np.linalg.norm(states[-1] - states[tail_start]))
    tail_efficiency = tail_displacement / tail_length if tail_length > 0.0 else 1.0
    tail_turns = turn_angles_deg(displacements[-tail_steps:])
    tail_progress = progress[-tail_steps:]

    straight = float(np.linalg.norm(states[-1] - states[0]))
    path_length = float(step_length.sum())
    return {
        "monotone_goal_progress": bool(len(progress) and np.all(progress > 0.0)),
        "minimum_goal_progress_m": float(progress.min()) if len(progress) else None,
        "nonpositive_progress_steps": int(np.count_nonzero(progress <= 0.0)),
        "minimum_terminal_goal_progress_m": (
            float(tail_progress.min()) if len(tail_progress) else None
        ),
        "terminal_nonpositive_progress_steps": int(
            np.count_nonzero(tail_progress <= 0.0)
        ),
        "path_length_m": path_length,
        "path_efficiency": straight / path_length if path_length > 0.0 else 1.0,
        "terminal_path_efficiency": float(tail_efficiency),
        "mean_turn_angle_deg": float(turns.mean()) if len(turns) else 0.0,
        "p95_turn_angle_deg": percentile(turns, 95.0),
        "max_turn_angle_deg": float(turns.max()) if len(turns) else 0.0,
        "terminal_max_turn_angle_deg": (
            float(tail_turns.max()) if len(tail_turns) else 0.0
        ),
        "mean_applied_control_delta": (
            float(applied_delta.mean()) if len(applied_delta) else 0.0
        ),
        "p95_applied_control_delta": percentile(applied_delta, 95.0),
        "max_applied_control_delta": (
            float(applied_delta.max()) if len(applied_delta) else 0.0
        ),
        "terminal_mean_applied_control_delta": (
            float(applied_delta[-tail_steps:].mean())
            if len(applied_delta) and tail_steps else 0.0
        ),
        "mean_raw_control_delta": (
            float(raw_delta.mean()) if len(raw_delta) else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-nfe", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_sha = sha256_file(args.checkpoint)
    config_sha = sha256_file(args.config)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {checkpoint_sha}")
    if config_sha != args.expected_config_sha256:
        raise SystemExit(f"config SHA mismatch: {config_sha}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    packaged_nfe = int(payload["arch"]["nfe"])
    if packaged_nfe != args.expected_nfe:
        raise SystemExit(f"packaged NFE mismatch: {packaged_nfe}")

    from safe_mppi.config import load_config
    from safe_mppi.lab_reference_flow_task import raw_reference_rollout
    from safe_mppi.lab_visual_flow import load_lab_reference_policy

    config = load_config(args.config)
    goal = np.asarray(config.taskspace.goal, dtype=float)
    policy = load_lab_reference_policy(args.checkpoint).to(args.device).eval()
    requested_gammas = set(args.gamma or [])
    source_rows = list(csv.DictReader(args.candidate_csv.open()))
    candidates = []
    seen = set()
    for row in source_rows:
        if row["status"] != "SUCCESS":
            continue
        gamma = float(row["gamma"])
        if requested_gammas and not any(
            abs(gamma - requested) < 1.0e-9 for requested in requested_gammas
        ):
            continue
        key = (gamma, int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "gamma": gamma,
            "seed": int(row["seed"]),
            "expected_mode": row["mode"],
            "source_min_clearance_m": float(row["min_clearance_m"]),
            "source_time_to_goal_s": float(row["time_to_goal_s"]),
        })

    records = []
    for index, candidate in enumerate(candidates, start=1):
        gamma = candidate["gamma"]
        seed = candidate["seed"]
        with torch.no_grad():
            result = raw_reference_rollout(
                policy,
                config,
                gamma,
                seed,
                device=args.device,
                sampling_temperature=args.sampling_temperature,
            )
        record = {
            **candidate,
            "status": result["status"],
            "mode": result["mode"],
            "min_clearance_m": result["min_clearance_m"],
            "time_to_goal_s": result["time_to_goal_s"],
            "steps": int(len(result["controls"])),
            "physical_gpu": args.physical_gpu,
        }
        if result["status"] == "SUCCESS" and result["mode"] == candidate["expected_mode"]:
            record.update(quality_metrics(result, goal))
        records.append(record)
        if index % 50 == 0 or index == len(candidates):
            print(f"quality search {index}/{len(candidates)}", flush=True)

    manifest = {
        "schema": "paper_ready_expanded_quality_search_v1",
        "checkpoint_sha256": checkpoint_sha,
        "packaged_nfe": packaged_nfe,
        "config_sha256": config_sha,
        "source_id": args.source_id,
        "physical_gpu": args.physical_gpu,
        "sampling_temperature": args.sampling_temperature,
        "candidate_count": len(candidates),
        "quality_definition": {
            "goal_progress": (
                "Euclidean goal distance at every 0.1 s executed state knot must "
                "strictly decrease."
            ),
            "terminal_window": "last 10 executed 0.1 s intervals",
            "smoothness": (
                "state-knot turn angles, applied-control deltas, path efficiency, "
                "and terminal-window variants"
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": len(candidates),
        "record_count": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
