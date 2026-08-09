#!/usr/bin/env python3
"""Screen a deterministic SafeMPPI seed bank for single-sphere route modes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-git-sha", required=True)
    parser.add_argument("--gammas", type=float, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    config = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_file(config) != args.expected_config_sha256:
        raise ValueError("config SHA mismatch")
    subprocess.run(
        ["git", "cat-file", "-e", f"{args.expected_source_git_sha}^{{commit}}"],
        cwd=repo,
        check=True,
    )
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", args.expected_source_git_sha, "--", "safe_mppi"],
        cwd=repo,
    )
    if source_diff.returncode != 0:
        raise ValueError(
            "working-tree safe_mppi does not match the expected 0806 source commit"
        )

    sys.path.insert(0, str(repo))
    from safe_mppi.acquire import run_episode
    from safe_mppi.ball_flow_theta import trajectory_crossing_theta, theta_name
    from safe_mppi.config import load_config
    from safe_mppi.controller import Mode1SafeMPPI
    from safe_mppi.environment import TaskEnvironment

    cfg = load_config(config)
    env = TaskEnvironment(cfg)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for gamma in args.gammas:
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            controller = Mode1SafeMPPI(cfg.safemppi, env, device=args.device)
            row, arrays = run_episode(
                env,
                controller,
                float(gamma),
                int(seed),
                rollout_dynamics=cfg.data.rollout_dynamics,
            )
            theta = trajectory_crossing_theta(env, arrays["dense_positions"])
            status = (
                "COLLISION" if row["collision"] else
                "OOB" if row["taskspace_violation"] else
                "SUCCESS" if row["success"] else "TIMEOUT"
            )
            rows.append({
                "gamma": float(gamma),
                "seed": int(seed),
                "status": status,
                "mode": theta_name(theta),
                "theta_deg": None if theta is None else float(np.degrees(theta)),
                "steps": int(row["steps"]),
                "min_clearance_m": row["min_clearance_m"],
                "time_to_goal_s": row["time_to_goal_s"],
                "mean_feasible_fraction": row["mean_feasible_fraction"],
            })
        print(f"completed gamma={gamma:g}", flush=True)

    output.mkdir(parents=True)
    columns = list(rows[0])
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    per_gamma = []
    pooled_success_modes: Counter[str] = Counter()
    for gamma in args.gammas:
        subset = [row for row in rows if row["gamma"] == float(gamma)]
        status_counts = Counter(str(row["status"]) for row in subset)
        success_modes = Counter(
            str(row["mode"]) for row in subset if row["status"] == "SUCCESS"
        )
        pooled_success_modes.update(success_modes)
        per_gamma.append({
            "gamma": float(gamma),
            "attempts": len(subset),
            "status_counts": dict(sorted(status_counts.items())),
            "successful_mode_counts": dict(sorted(success_modes.items())),
        })
    summary = {
        "schema": "paper_ready_0808_safemppi_mode_screen_v1",
        "expected_source_git_sha": args.expected_source_git_sha,
        "safe_mppi_tree_matches_expected_commit": True,
        "config": (
            config.relative_to(repo).as_posix()
            if config.is_relative_to(repo)
            else str(config)
        ),
        "config_sha256": args.expected_config_sha256,
        "device": args.device,
        "seed_start": args.seed_start,
        "seed_count_per_gamma": args.seed_count,
        "gammas": args.gammas,
        "wall_time_s": time.perf_counter() - started,
        "per_gamma": per_gamma,
        "pooled_successful_mode_counts": dict(sorted(pooled_success_modes.items())),
        "interpretation": (
            "Empirical finite-seed distribution used only to identify prominent "
            "SafeMPPI symmetry-broken route classes; not an expansion evaluation."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
