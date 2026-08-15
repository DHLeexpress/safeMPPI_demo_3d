#!/usr/bin/env python3
"""Run a fixed-scene sanity audit or resumable exact-success collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.path_focused_collection import (  # noqa: E402
    collect_path_focused_clutter_demos,
    collect_path_focused_success_quota,
    path_focused_collection_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_clutter_cylinders_path_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--target-successes-per-gamma",
        type=int,
        help="resume until exactly this many accepted trajectories per gamma",
    )
    mode.add_argument(
        "--fixed-scenes",
        type=int,
        help="non-resumable one-rollout-per-scene sanity audit",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="required finite geometry budget for success-quota mode",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=None)
    parser.add_argument("--domain-seed", type=int, default=None)
    parser.add_argument("--rollout-seed-start", type=int, default=None)
    parser.add_argument("--centroid-gain", type=float, default=None)
    parser.add_argument("--centroid-smooth", type=float, default=None)
    parser.add_argument("--sigma-aniso", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    base = load_config(args.config)
    gamma_values = (
        None if args.gammas is None else tuple(args.gammas)
    )
    if args.fixed_scenes is not None:
        if args.max_scenes is not None:
            parser.error("--max-scenes only applies to success-quota mode")
        config = path_focused_collection_config(
            base,
            gammas=gamma_values,
            episodes_per_gamma=args.fixed_scenes,
            max_attempts_per_gamma=args.fixed_scenes,
            centroid_gain=args.centroid_gain,
            centroid_smooth=args.centroid_smooth,
            sigma_aniso=args.sigma_aniso,
        )
        manifest = collect_path_focused_clutter_demos(
            config,
            args.output,
            scene_count=args.fixed_scenes,
            domain_seed=args.domain_seed,
            rollout_seed_start=args.rollout_seed_start,
            device=args.device,
        )
    else:
        if args.max_scenes is None:
            parser.error("success-quota mode requires --max-scenes")
        manifest = collect_path_focused_success_quota(
            base,
            args.output,
            target_successes_per_gamma=args.target_successes_per_gamma,
            max_scene_count=args.max_scenes,
            gammas=gamma_values,
            domain_seed=args.domain_seed,
            rollout_seed_start=args.rollout_seed_start,
            centroid_gain=args.centroid_gain,
            centroid_smooth=args.centroid_smooth,
            sigma_aniso=args.sigma_aniso,
            device=args.device,
        )
    print(json.dumps({
        "status": manifest["status"],
        "gammas": manifest["gammas"],
        "attempts": len(manifest["attempts"]),
        "accepted": len(manifest["runs"]),
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
