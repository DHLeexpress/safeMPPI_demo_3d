#!/usr/bin/env python3
"""Collect exact paired SafeMPPI expert demos on dense-z sphere scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.helios_remote import (  # noqa: E402
    add_helios_arguments,
    run_collection_on_helios,
)
from safe_mppi.multisphere_axis180_expert_collection import (  # noqa: E402
    collect_multisphere_axis180_expert_demos,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_helios_arguments(parser)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs" /
            "lab_clutter_spheres_double_hourglass_n6_dense_z0711_"
            "goalspace_yminus04_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-trajectories-per-gamma", type=int, default=50,
        help=(
            "Exact admitted trajectory quota per gamma; must be even because "
            "each admitted original/axis-180 pair contributes two."
        ),
    )
    parser.add_argument(
        "--max-pairs-per-gamma", type=int, required=True,
        help="Finite fail-closed pair-attempt budget for each gamma.",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=None)
    parser.add_argument("--domain-seed", type=int, default=None)
    parser.add_argument("--rollout-seed-start", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.helios:
        raise SystemExit(run_collection_on_helios(
            args,
            sys.argv[1:],
            ROOT,
            script_name=Path(__file__).name,
        ))

    manifest = collect_multisphere_axis180_expert_demos(
        load_config(args.config),
        args.output,
        target_trajectories_per_gamma=(
            args.target_trajectories_per_gamma
        ),
        max_pairs_per_gamma=args.max_pairs_per_gamma,
        gammas=None if args.gammas is None else tuple(args.gammas),
        domain_seed=args.domain_seed,
        rollout_seed_start=args.rollout_seed_start,
        device=args.device,
    )
    print(json.dumps({
        "status": manifest["status"],
        "gammas": manifest["gammas"],
        "attempted_pairs": len(manifest["scene_bank"]["pairs"]),
        "attempted_trajectories": len(manifest["attempts"]),
        "training_trajectories": len(manifest["runs"]),
        "accepted_counts_by_gamma": (
            manifest["accepted_counts_by_gamma"]
        ),
        "accepted_pair_counts_by_gamma": (
            manifest["accepted_pair_counts_by_gamma"]
        ),
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
