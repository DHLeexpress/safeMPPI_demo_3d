#!/usr/bin/env python3
"""Evaluate the unchanged stage-1 policy on a fixed randomized clutter bank."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import ObstacleConfig, load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    scene_sha256,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402


DEFAULT_GAMMAS = (0.1, 0.3, 0.5, 1.0)
SCENE_SEED_STRIDE = 1009
ROLLOUT_SEED_STRIDE = 37


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_config(config, spheres: np.ndarray):
    return replace(
        config,
        obstacles=ObstacleConfig(
            spheres=tuple(tuple(map(float, row)) for row in spheres),
            cylinders=(),
        ),
    )


def _summary(rows: list[dict]) -> dict:
    statuses = ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
    counts = {
        status.lower(): sum(row["status"] == status for row in rows)
        for status in statuses
    }
    successes = [row for row in rows if row["status"] == "SUCCESS"]
    return {
        "trials": len(rows),
        "failures": len(rows) - counts["success"],
        **counts,
        "SR": counts["success"] / len(rows),
        "mean_successful_min_clearance_m": (
            float(np.mean([row["min_clearance_m"] for row in successes]))
            if successes else None
        ),
        "mean_successful_time_to_goal_s": (
            float(np.mean([row["time_to_goal_s"] for row in successes]))
            if successes else None
        ),
        "mean_window_validity": float(np.mean([
            row["window_validity"] for row in rows
        ])),
    }


def _json_row(row: dict) -> dict:
    return {
        key: (
            np.round(value[:, :3], 5).tolist()
            if key == "states"
            else value
        )
        for key, value in row.items()
        if key not in {"controls", "applied_controls", "dense_steps"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        default=(
            ROOT
            / "results/stage1_single_ball_t128/pretrain_hp100_t128_d3_e52"
        ),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=(
            ROOT
            / "configs/lab_clutter_spheres_double_hourglass_n6_stage1box_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials-per-gamma", type=int, default=20)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.trials_per_gamma < 1:
        parser.error("--trials-per-gamma must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires an available MPS backend")

    config = load_config(args.task_config)
    template_env = TaskEnvironment(config)
    scene_spec = sphere_scene_spec_from_config(config)
    args.output.mkdir(parents=True)
    policy_path = args.pretrain_dir / "pretrained.pt"
    policy = load_lab_reference_policy(policy_path).to(device).eval()

    scenes = []
    for episode in range(args.trials_per_gamma):
        scene_seed = int(args.seed) + SCENE_SEED_STRIDE * episode
        spheres = scene_spec.sample(template_env, scene_seed)
        scenes.append({
            "episode": episode,
            "scene_seed": scene_seed,
            "scene_hash": scene_sha256(template_env, spheres),
            "spheres": spheres.tolist(),
        })

    started = time.time()
    rows = []
    total = len(args.gammas) * args.trials_per_gamma
    with torch.no_grad():
        for gamma in args.gammas:
            gamma_rows = []
            for scene in scenes:
                spheres = np.asarray(scene["spheres"], np.float32)
                rollout_seed = (
                    int(args.seed)
                    + ROLLOUT_SEED_STRIDE * int(scene["episode"])
                )
                result = raw_reference_rollout(
                    policy,
                    _fixed_config(config, spheres),
                    float(gamma),
                    rollout_seed,
                    device=device,
                    sampling_temperature=float(args.sampling_temperature),
                )
                row = {
                    "gamma": float(gamma),
                    "episode": int(scene["episode"]),
                    "rollout_seed": rollout_seed,
                    "scene_seed": int(scene["scene_seed"]),
                    "scene_hash": str(scene["scene_hash"]),
                    "spheres": scene["spheres"],
                    **result,
                }
                rows.append(row)
                gamma_rows.append(row)
                print(
                    f"[{len(rows):03d}/{total}] gamma={gamma:g} "
                    f"episode={scene['episode']:02d} status={result['status']} "
                    f"steps={len(result['controls'])}",
                    flush=True,
                )
            current = {
                f"{value:g}": _summary([
                    row for row in rows if row["gamma"] == value
                ])
                for value in args.gammas
                if any(row["gamma"] == value for row in rows)
            }
            (args.output / "progress.json").write_text(
                json.dumps(current, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"gamma={gamma:g} complete: "
                + json.dumps(_summary(gamma_rows), allow_nan=False),
                flush=True,
            )

    summary = {
        f"{gamma:g}": _summary([
            row for row in rows if row["gamma"] == gamma
        ])
        for gamma in args.gammas
    }
    pair_distances = [
        float(np.linalg.norm(
            np.asarray(scene["spheres"])[first, :3]
            - np.asarray(scene["spheres"])[second, :3]
        ))
        for scene in scenes
        for first in range(len(scene["spheres"]))
        for second in range(first + 1, len(scene["spheres"]))
    ]
    z_centers = [
        sphere[2] for scene in scenes for sphere in scene["spheres"]
    ]
    manifest = {
        "task_config": str(args.task_config.resolve()),
        "task_config_sha256": _sha256(args.task_config),
        "pretrained_checkpoint": str(policy_path.resolve()),
        "pretrained_checkpoint_sha256": _sha256(policy_path),
        "context_schema": str(policy.context_schema),
        "nfe": int(policy.nfe),
        "device": str(device),
        "sampling_temperature": float(args.sampling_temperature),
        "gammas": list(map(float, args.gammas)),
        "trials_per_gamma": int(args.trials_per_gamma),
        "total_rollouts": len(rows),
        "evaluation_seed": int(args.seed),
        "shared_scene_bank_across_gamma": True,
        "shared_rollout_seed_across_gamma": True,
        "elapsed_seconds": time.time() - started,
        "geometry_audit": {
            "scene_count": len(scenes),
            "sphere_counts": sorted({len(scene["spheres"]) for scene in scenes}),
            "minimum_observed_center_distance_m": min(pair_distances),
            "minimum_required_center_distance_m": (
                2.0 * float(np.float32(scene_spec.radius))
                + float(scene_spec.minimum_surface_margin)
            ),
            "observed_z_center_range_m": [min(z_centers), max(z_centers)],
            "configured_z_center_guard_m": list(map(
                float,
                config.raw["scene_randomization"]["sphere_z_center_range_m"],
            )),
        },
    }
    torch.save(
        {"manifest": manifest, "scenes": scenes, "rows": rows},
        args.output / "raw_rollouts.pt",
    )
    (args.output / "rollouts.json").write_text(
        json.dumps({
            "manifest": manifest,
            "scenes": scenes,
            "summary": summary,
            "rows": [_json_row(row) for row in rows],
        }, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
