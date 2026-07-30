#!/usr/bin/env python3
"""Collect one honest SafeMPPI rollout per path-focused scene/gamma cell."""
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
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_clutter_cylinders_path_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--domain-seed", type=int, default=None)
    parser.add_argument("--rollout-seed-start", type=int, default=None)
    parser.add_argument("--transverse-std-m", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = collect_path_focused_clutter_demos(
        load_config(args.config),
        args.output,
        scene_count=args.scenes,
        domain_seed=args.domain_seed,
        rollout_seed_start=args.rollout_seed_start,
        transverse_std_m=args.transverse_std_m,
        device=args.device,
    )
    print(json.dumps({
        "status": manifest["status"],
        "scene_count": manifest["evaluated_scene_count"],
        "accepted_trajectory_count": len(manifest["runs"]),
        "expert_outcomes": manifest["metrics"],
        "behavior": manifest["behavior_metrics"],
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
