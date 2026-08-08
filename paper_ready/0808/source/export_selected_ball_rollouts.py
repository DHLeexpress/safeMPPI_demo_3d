#!/usr/bin/env python3
"""Export deterministic single-ball policy rollouts with strict provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gamma_tag(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def normalized_degrees(theta: float) -> float:
    return float((math.degrees(theta) + 180.0) % 360.0 - 180.0)


def sector_index(theta_deg: float, count: int) -> int:
    width = 360.0 / count
    return min(count - 1, int(math.floor((theta_deg + 180.0) / width)))


def bit_identical(first: dict, second: dict) -> bool:
    keys = ("states", "controls", "applied_controls", "dense_steps")
    return (
        first["status"] == second["status"]
        and first["mode"] == second["mode"]
        and all(np.array_equal(first[key], second[key]) for key in keys)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-nfe", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--theta-tolerance-deg", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 2:
        raise SystemExit("--repeat must be at least 2 for deterministic verification")
    checkpoint_sha = sha256_file(args.checkpoint)
    config_sha = sha256_file(args.config)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {checkpoint_sha}")
    if config_sha != args.expected_config_sha256:
        raise SystemExit(f"config SHA mismatch: {config_sha}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    packaged_nfe = int(payload["arch"]["nfe"])
    if packaged_nfe != args.expected_nfe:
        raise SystemExit(
            f"packaged NFE mismatch: {packaged_nfe} != {args.expected_nfe}"
        )

    from safe_mppi.ball_flow_theta import trajectory_crossing_theta
    from safe_mppi.config import load_config
    from safe_mppi.environment import TaskEnvironment
    from safe_mppi.lab_reference_flow_task import raw_reference_rollout
    from safe_mppi.lab_visual_flow import load_lab_reference_policy

    selections = json.loads(args.selections.read_text())
    selected = [
        row for row in selections["trajectories"]
        if int(row["physical_gpu"]) == args.physical_gpu
    ]
    if not selected:
        raise SystemExit(f"no selections for physical GPU {args.physical_gpu}")

    config = load_config(args.config)
    env = TaskEnvironment(config)
    policy = load_lab_reference_policy(args.checkpoint).to(args.device).eval()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for selection in selected:
        gamma = float(selection["gamma"])
        seed = int(selection["seed"])
        expected_mode = str(selection["expected_mode"])
        expected_theta = float(selection["expected_theta_deg"])
        runs = []
        for _ in range(args.repeat):
            with torch.no_grad():
                runs.append(raw_reference_rollout(
                    policy,
                    config,
                    gamma,
                    seed,
                    device=args.device,
                    sampling_temperature=args.sampling_temperature,
                ))
        if any(not bit_identical(runs[0], run) for run in runs[1:]):
            raise SystemExit(
                f"repeatability failure for gamma={gamma:g}, seed={seed}"
            )

        result = runs[0]
        dense_path = np.concatenate([
            result["states"][:1, :3],
            result["dense_steps"].reshape(-1, 3),
        ])
        theta = trajectory_crossing_theta(env, dense_path)
        if theta is None:
            raise SystemExit(f"missing first crossing for gamma={gamma:g}, seed={seed}")
        theta_deg = normalized_degrees(theta)
        if result["status"] != "SUCCESS":
            raise SystemExit(
                f"expected SUCCESS, got {result['status']} for gamma={gamma:g}, seed={seed}"
            )
        if result["mode"] != expected_mode:
            raise SystemExit(
                f"mode mismatch for gamma={gamma:g}, seed={seed}: "
                f"{result['mode']} != {expected_mode}"
            )
        theta_error = abs(theta_deg - expected_theta)
        if theta_error > args.theta_tolerance_deg:
            raise SystemExit(
                f"theta mismatch for gamma={gamma:g}, seed={seed}: "
                f"{theta_deg:.6f} vs {expected_theta:.6f} deg"
            )

        filename = (
            f"gamma_{gamma_tag(gamma)}_mode_{result['mode']}_seed_{seed}.npz"
        )
        path = output / filename
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        sector_16 = sector_index(theta_deg, 16)
        sector_8 = sector_index(theta_deg, 8)
        np.savez_compressed(
            path,
            states=result["states"],
            controls=result["controls"],
            applied_controls=result["applied_controls"],
            dense_steps=result["dense_steps"],
            status=np.str_(result["status"]),
            gamma=np.float32(gamma),
            seed=np.int64(seed),
            sampling_temperature=np.float32(args.sampling_temperature),
            mode=np.str_(result["mode"]),
            theta_deg=np.float64(theta_deg),
            sector_16=np.int64(sector_16),
            sector_8=np.int64(sector_8),
            min_clearance_m=np.float64(result["min_clearance_m"]),
            time_to_goal_s=np.float64(result["time_to_goal_s"]),
            checkpoint_sha256=np.str_(checkpoint_sha),
            config_sha256=np.str_(config_sha),
            source_id=np.str_(args.source_id),
            physical_gpu=np.int64(args.physical_gpu),
        )
        records.append({
            "file": filename,
            "sha256": sha256_file(path),
            "gamma": gamma,
            "seed": seed,
            "status": result["status"],
            "mode": result["mode"],
            "theta_deg": theta_deg,
            "theta_error_from_selection_deg": theta_error,
            "sector_16": sector_16,
            "sector_8": sector_8,
            "min_clearance_m": result["min_clearance_m"],
            "time_to_goal_s": result["time_to_goal_s"],
            "steps": int(len(result["controls"])),
            "physical_gpu": args.physical_gpu,
            "repeat_count": args.repeat,
            "repeat_verification": "BIT_IDENTICAL",
        })

    manifest = {
        "schema": "paper_ready_selected_ball_rollouts_v1",
        "policy_label": args.policy_label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "packaged_nfe": packaged_nfe,
        "config": str(args.config),
        "config_sha256": config_sha,
        "source_id": args.source_id,
        "device": args.device,
        "physical_gpu": args.physical_gpu,
        "sampling_temperature": args.sampling_temperature,
        "governor": "ReferenceGovernor, exactly once",
        "seed_formula": "episode_seed * 100000 + closed_loop_step",
        "records": records,
    }
    manifest_path = output / f"manifest_gpu{args.physical_gpu}.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
