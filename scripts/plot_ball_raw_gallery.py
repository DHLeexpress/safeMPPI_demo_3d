"""Render only a fixed-seed raw temperature-1 side-view gallery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ball_expansion import (checkpoint_policy, raw_eval,
                                     side_rollout_gallery)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.pretrain_dir / "demo_config.json")
    env = TaskEnvironment(config)
    gammas = list(config.data.gammas)
    manifest = json.loads((args.expansion / "manifest.json").read_text())
    tight_corridor = bool(
        manifest.get("ball_verifier_corridor", {}).get("enabled", False)
    )
    rows = {}
    for round_i in args.rounds:
        policy = checkpoint_policy(args.expansion, args.pretrain_dir, round_i)
        rows[round_i] = raw_eval(
            policy, config, gammas, args.episodes, args.seed,
            tight_corridor=tight_corridor,
        )
        print(f"[raw gallery] round {round_i}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    side_rollout_gallery(env, rows, args.rounds, gammas, args.output)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
