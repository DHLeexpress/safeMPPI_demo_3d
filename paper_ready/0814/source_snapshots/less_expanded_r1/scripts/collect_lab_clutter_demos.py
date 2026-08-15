#!/usr/bin/env python3
"""Collect paired-gamma SafeMPPI demos in randomized three-cylinder lab scenes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.lab_clutter import collect_clutter_demos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_clutter_cylinders_pretrain.json",
        help="base Minhyuk-governed lab config; obstacles are replaced per run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/lab_clutter_demos",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=None,
        help="scene count per gamma (default: config episodes_per_gamma)",
    )
    parser.add_argument(
        "--domain-seed",
        type=int,
        default=None,
        help="override scene_randomization.seed",
    )
    parser.add_argument("--rollout-seed-start", type=int, default=None)
    parser.add_argument(
        "--cylinder-count",
        type=int,
        default=None,
        help="override scene_randomization.count",
    )
    parser.add_argument(
        "--cylinder-radius-m",
        type=float,
        default=None,
        help="override scene_randomization.radius_m",
    )
    parser.add_argument(
        "--min-surface-gap-m",
        type=float,
        default=None,
        help="override minimum_obstacle_surface_gap_m",
    )
    parser.add_argument(
        "--endpoint-surface-gap-m",
        type=float,
        default=None,
        help="override both start and goal surface clearances",
    )
    parser.add_argument(
        "--start-surface-gap-m",
        type=float,
        default=None,
        help="override minimum_start_surface_clearance_m",
    )
    parser.add_argument(
        "--goal-surface-gap-m",
        type=float,
        default=None,
        help="override minimum_goal_surface_clearance_m",
    )
    parser.add_argument(
        "--boundary-surface-gap-m",
        type=float,
        default=None,
        help="override minimum_taskspace_wall_surface_clearance_m",
    )
    parser.add_argument("--max-layout-attempts", type=int, default=20_000)
    parser.add_argument(
        "--max-rollout-attempts-per-scene",
        type=int,
        default=None,
        help=(
            "controller-noise retries for a fixed scene/gamma; default uses "
            "data.max_attempts_per_gamma from the config"
        ),
    )
    parser.add_argument(
        "--max-candidate-scenes",
        type=int,
        default=None,
        help=(
            "deterministic proposal bound for all-gamma scene admission; "
            "default is 4x the requested scene count"
        ),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = collect_clutter_demos(
        config,
        args.output,
        scene_count=args.scenes,
        domain_seed=args.domain_seed,
        rollout_seed_start=args.rollout_seed_start,
        cylinder_count=args.cylinder_count,
        cylinder_radius_m=args.cylinder_radius_m,
        min_surface_gap_m=args.min_surface_gap_m,
        endpoint_surface_gap_m=args.endpoint_surface_gap_m,
        start_surface_gap_m=args.start_surface_gap_m,
        goal_surface_gap_m=args.goal_surface_gap_m,
        boundary_surface_gap_m=args.boundary_surface_gap_m,
        max_layout_attempts=args.max_layout_attempts,
        max_rollout_attempts_per_scene=(
            args.max_rollout_attempts_per_scene
        ),
        max_candidate_scenes=args.max_candidate_scenes,
        device=args.device,
    )
    print(
        "[scene admission] "
        f"{manifest['admitted_scene_count']}/"
        f"{manifest['requested_scene_count']} admitted from "
        f"{manifest['candidate_scenes_evaluated']} candidates; "
        f"{manifest['rejected_scene_count']} rejected",
        flush=True,
    )
    path_summary = manifest["scene_bank"]["start_goal_path_summary"]
    print(
        "[path relevance, diagnostic only] "
        f"modeled-hard intersections "
        f"{path_summary['modeled_hard_path_intersection_scene_count']}/"
        f"{path_summary['scene_count']}; within "
        f"{path_summary['soft_clearance_target_m']:.3g} m soft tube "
        f"{path_summary['within_soft_clearance_tube_scene_count']}/"
        f"{path_summary['scene_count']}",
        flush=True,
    )
    print(
        f"[output] {args.output.resolve()} "
        f"({len(manifest['runs'])} accepted trajectories)",
        flush=True,
    )


if __name__ == "__main__":
    main()
