#!/usr/bin/env python3
"""Collect gamma-exclusive mirrored pairs with exact paired-success quotas."""
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
from safe_mppi.mirrored_pair_collection import (  # noqa: E402
    collect_mirrored_pair_success_quota,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_helios_arguments(parser)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs" /
            "lab_clutter_cylinders_path_midpoint_uniform_mirrored_z01_17_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-successes-per-gamma", type=int, required=True,
    )
    parser.add_argument(
        "--max-pair-attempts-per-gamma", type=int, required=True,
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

    manifest = collect_mirrored_pair_success_quota(
        load_config(args.config),
        args.output,
        target_successes_per_gamma=args.target_successes_per_gamma,
        max_pair_attempts_per_gamma=args.max_pair_attempts_per_gamma,
        gammas=None if args.gammas is None else tuple(args.gammas),
        domain_seed=args.domain_seed,
        rollout_seed_start=args.rollout_seed_start,
        device=args.device,
    )
    print(json.dumps({
        "status": manifest["status"],
        "gammas": manifest["gammas"],
        "attempts": len(manifest["attempts"]),
        "training_runs": len(manifest["runs"]),
        "accepted_counts_by_gamma": manifest["accepted_counts_by_gamma"],
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
