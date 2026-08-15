#!/usr/bin/env python3
"""Build a fixed, geometry-only multi-sphere evaluation scene bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.lab_clutter_evaluation import _fixed_evaluation_scene_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.task_config)
    scene_bank = _fixed_evaluation_scene_bank(
        config, args.episodes, args.seed,
    )
    scene_bank.update({
        "evaluation_unique_scene_count": len(scene_bank["scenes"]) + 1,
        "expansion_unique_scene_count": None,
        "overlap_count": None,
        "disjointness_check": (
            "deferred until each expansion arm has committed its scene ledger"
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "FIXED_MULTISPHERE_SCENE_BANK_COMPLETE",
        "task_config": str(args.task_config.resolve()),
        "scene_bank": scene_bank,
    }, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
