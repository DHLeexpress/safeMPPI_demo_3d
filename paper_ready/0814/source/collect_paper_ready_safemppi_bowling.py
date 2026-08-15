#!/usr/bin/env python3
"""Run the pinned paper-ready SafeMPPI controller on a fixed bowling scene."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PINNED_SOURCE_SHA = "dabb5011dfc674864e1de275a1e1c2adab58f4af"
PINNED_CONTROLLER_SHA256 = "dfc91a26ccac2818c902215bf4d9a06e405d5878e5c6af0be2f75c4f68106dad"
ROW_PROGRESS = (0.25, 0.45, 0.65)
LANE_DELTA_M = 0.345
STABILITY_MARGIN_M = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crossing(progress: np.ndarray, values: np.ndarray, target: float) -> np.ndarray:
    for index in range(1, len(progress)):
        before, after = float(progress[index - 1]), float(progress[index])
        if before < target <= after and after - before > 1e-12:
            weight = (target - before) / (after - before)
            return (1.0 - weight) * values[index - 1] + weight * values[index]
    raise ValueError(f"path never crosses progress={target:g}")


def _route_signature(states: np.ndarray, start: np.ndarray, goal: np.ndarray) -> dict:
    path = np.asarray(states, np.float64)
    start3, goal3 = np.asarray(start)[:3], np.asarray(goal)[:3]
    direction = goal3 - start3
    length = float(np.linalg.norm(direction))
    forward = direction / length
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    progress = (path[:, :3] - start3[None]) @ forward / length
    crossings = np.asarray([_crossing(progress, path[:, :3], target) for target in ROW_PROGRESS])
    lateral_values = (crossings - start3[None]) @ lateral
    vertical_values = crossings[:, 2] - start3[2]
    target_lateral, bits, margins, deltas = 0.0, [], [], []
    for value in lateral_values:
        delta = float(value - target_lateral)
        bit = "L" if delta >= 0.0 else "R"
        bits.append(bit)
        margins.append(abs(delta))
        deltas.append(delta)
        target_lateral += LANE_DELTA_M if bit == "L" else -LANE_DELTA_M
    stable = "".join(bit if margin >= STABILITY_MARGIN_M else "X" for bit, margin in zip(bits, margins))
    return {
        "code": "".join(bits),
        "stable_code": stable,
        "decision_xyz_m": crossings.tolist(),
        "decision_lateral_m": lateral_values.tolist(),
        "decision_lateral_delta_m": deltas,
        "decision_vertical_m": vertical_values.tolist(),
        "decision_vertical_dominant": (np.abs(vertical_values) > np.abs(deltas)).tolist(),
        "minimum_decision_margin_m": float(min(margins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attempts-per-gamma", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=240000)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="run these exact controller seeds for every requested gamma",
    )
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--gammas", type=float, nargs="+")
    parser.add_argument(
        "--as-built-scene-json",
        type=Path,
        help="use measured per-ball radii +0.16 m and post-check 0.10 m strings",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    controller_path = args.source_root / "safe_mppi/controller.py"
    if _sha256(controller_path) != PINNED_CONTROLLER_SHA256:
        raise RuntimeError("pinned SafeMPPI controller hash mismatch")

    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from safe_mppi.acquire import run_episode  # pylint: disable=import-outside-toplevel
    from safe_mppi.config import load_config  # pylint: disable=import-outside-toplevel
    from safe_mppi.controller import Mode1SafeMPPI  # pylint: disable=import-outside-toplevel
    from safe_mppi.environment import TaskEnvironment  # pylint: disable=import-outside-toplevel
    from real_bowling_scene import (  # pylint: disable=import-outside-toplevel
        hard_path_diagnostics,
        load_as_built_geometry,
    )

    config = load_config(args.config)
    geometry = (
        load_as_built_geometry(args.as_built_scene_json)
        if args.as_built_scene_json is not None else None
    )
    if geometry is not None:
        config = replace(
            config,
            obstacles=replace(
                config.obstacles,
                spheres=tuple(
                    tuple(map(float, row))
                    for row in geometry["effective_spheres"]
                ),
                cylinders=(),
            ),
        )
    env = TaskEnvironment(config)
    args.output.mkdir(parents=True)
    rows = []
    configured_gammas = [float(gamma) for gamma in config.data.gammas]
    selected_gammas = configured_gammas if args.gammas is None else args.gammas
    for gamma in selected_gammas:
        if not any(np.isclose(gamma, configured) for configured in configured_gammas):
            raise ValueError(f"requested gamma={gamma:g} is absent from config")
    for gamma_index, gamma in enumerate(selected_gammas):
        rollout_seeds = (
            list(args.seeds)
            if args.seeds is not None else
            [
                args.seed_start + 1000 * gamma_index + local_episode
                for local_episode in range(args.attempts_per_gamma)
            ]
        )
        for local_episode, seed in enumerate(rollout_seeds):
            episode = args.episode_offset + local_episode
            controller = Mode1SafeMPPI(config.safemppi, env, device=args.device)
            summary, arrays = run_episode(
                env, controller, gamma, seed,
                rollout_dynamics=config.data.rollout_dynamics,
            )
            status = (
                "SUCCESS" if summary["success"] else
                "COLLISION" if summary["collision"] else
                "OOB" if summary["taskspace_violation"] else "TIMEOUT"
            )
            hard_constraints = None
            if geometry is not None:
                dense_path = np.asarray(
                    arrays["dense_positions"], np.float32,
                ).reshape(-1, 3)
                hard_constraints = hard_path_diagnostics(
                    dense_path,
                    geometry["effective_spheres"],
                    geometry["physical_spheres"],
                    geometry["string_radius_m"],
                )
                if status == "SUCCESS" and not hard_constraints["hard_valid"]:
                    status = "COLLISION"
            route = None
            if status == "SUCCESS":
                try:
                    route = _route_signature(arrays["states"], env.start, env.goal)
                except ValueError:
                    route = None
            rows.append({
                "round": "safemppi",
                "gamma": float(gamma),
                "episode": episode,
                "rollout_seed": seed,
                "bowling_route": route,
                "status": status,
                "states": arrays["states"],
                "controls": arrays["controls"],
                "applied_controls": arrays["executed_controls"],
                "dense_positions": arrays["dense_positions"],
                "min_clearance_m": summary["min_clearance_m"],
                "time_to_goal_s": summary["time_to_goal_s"],
                "window_validity": None,
                "hard_constraints": hard_constraints,
            })
            if (local_episode + 1) % 10 == 0 or local_episode + 1 == len(rollout_seeds):
                print(
                    f"[SafeMPPI] gamma={gamma:g} "
                    f"{local_episode + 1}/{len(rollout_seeds)}",
                    flush=True,
                )

    raw_path = args.output / "raw_safemppi_trajectories.pt"
    torch.save({"safemppi": rows}, raw_path)
    counts = {}
    for gamma in selected_gammas:
        group = [row for row in rows if np.isclose(row["gamma"], gamma)]
        counts[str(gamma)] = {
            "attempts": len(group),
            "success": sum(row["status"] == "SUCCESS" for row in group),
            "collision": sum(row["status"] == "COLLISION" for row in group),
            "oob": sum(row["status"] == "OOB" for row in group),
            "timeout": sum(row["status"] == "TIMEOUT" for row in group),
        }
    manifest = {
        "kind": "paper-ready exact-source SafeMPPI bowling search",
        "source_git_sha": PINNED_SOURCE_SHA,
        "controller_path": "safe_mppi/controller.py",
        "controller_sha256": PINNED_CONTROLLER_SHA256,
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "wrapper_sha256": _sha256(Path(__file__)),
        "device": args.device,
        "attempts_per_gamma": (
            len(args.seeds) if args.seeds is not None else args.attempts_per_gamma
        ),
        "seed_start": None if args.seeds is not None else args.seed_start,
        "seeds": args.seeds,
        "episode_offset": args.episode_offset,
        "selected_gammas": selected_gammas,
        "as_built_scene": (
            None if geometry is None else {
                "scene_json": str(args.as_built_scene_json),
                "scene_json_sha256": _sha256(args.as_built_scene_json),
                "physical_spheres": geometry["physical_spheres"].tolist(),
                "effective_spheres": geometry["effective_spheres"].tolist(),
                "effective_margin_m": geometry["effective_margin_m"],
                "string_radius_m": geometry["string_radius_m"],
                "string_gate_stage": "post-execution hard failure",
            }
        ),
        "counts": counts,
        "raw_trajectories": raw_path.name,
        "raw_sha256": _sha256(raw_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
