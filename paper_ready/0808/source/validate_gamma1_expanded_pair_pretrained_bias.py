#!/usr/bin/env python3
"""Validate the frozen gamma-1 expanded pair and pretrained bias references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPANDED_SHA = "c1a3c77fc956c57d02a0970c4e54fca942cee391a68275a58134361e00828056"
PRETRAINED_SHA = "cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff"
CONFIG_SHA = "7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2"
EXPANDED_SEEDS = (131629, 135403)
PRETRAINED_SEEDS = (91407, 91703, 91777, 92333, 92369, 92407, 92851, 93888)


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path: Path, expected_checkpoint: str, goal: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        status = str(scalar(data, "status"))
        gamma = float(scalar(data, "gamma"))
        seed = int(scalar(data, "seed"))
        mode = str(scalar(data, "mode"))
        theta = float(scalar(data, "theta_deg"))
        checkpoint = str(scalar(data, "checkpoint_sha256"))
        config_sha = str(scalar(data, "config_sha256"))
        states = np.asarray(data["states"][:, :3], dtype=float)
        dense = np.concatenate([
            states[:1], np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3)
        ])
        clearance = float(scalar(data, "min_clearance_m"))
        time_to_goal = float(scalar(data, "time_to_goal_s"))
    if status != "SUCCESS" or not np.isclose(gamma, 1.0):
        raise ValueError(f"expected gamma-1 terminal success: {path}")
    if checkpoint != expected_checkpoint or config_sha != CONFIG_SHA:
        raise ValueError(f"frozen identity mismatch: {path}")
    state_progress = np.diff(-np.linalg.norm(states - goal[None], axis=1))
    dense_progress = np.diff(-np.linalg.norm(dense - goal[None], axis=1))
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "seed": seed,
        "mode": mode,
        "theta_deg": theta,
        "minimum_clearance_m": clearance,
        "time_to_goal_s": time_to_goal,
        "minimum_state_goal_progress_m": float(state_progress.min()),
        "minimum_dense_goal_progress_m": float(dense_progress.min()),
        "strictly_goal_monotone": bool(
            np.all(state_progress > 0.0) and np.all(dense_progress > 0.0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expanded-string-safe", type=Path)
    parser.add_argument("--expanded-mirrored", type=Path)
    parser.add_argument("--pretrained-biased", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config = json.loads((bundle / "config" / "task_config_resolved.json").read_text())
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    expanded_roots = (
        args.expanded_string_safe.resolve() if args.expanded_string_safe else
        bundle / "trajectories" / "expanded_string_safe_v1",
        args.expanded_mirrored.resolve() if args.expanded_mirrored else
        bundle / "trajectories" / "expanded_mirrored_above_v1",
    )
    pretrained_root = (
        args.pretrained_biased.resolve() if args.pretrained_biased else
        bundle / "trajectories" / "pretrained_gamma1_biased_left_v1"
    )
    expanded = sorted(
        (audit(path, EXPANDED_SHA, goal) for root in expanded_roots
         for path in root.glob("*.npz")),
        key=lambda row: int(row["seed"]),
    )
    pretrained = sorted(
        (audit(path, PRETRAINED_SHA, goal) for path in pretrained_root.glob("*.npz")),
        key=lambda row: int(row["seed"]),
    )
    if tuple(int(row["seed"]) for row in expanded) != EXPANDED_SEEDS:
        raise ValueError("expanded pair does not match the frozen seed contract")
    if tuple(int(row["seed"]) for row in pretrained) != PRETRAINED_SEEDS:
        raise ValueError("pretrained bias set does not match the frozen seed contract")
    if any(row["mode"] != "above" for row in expanded):
        raise ValueError("expanded pair must contain only above trajectories")
    if any(row["mode"] != "left" or not 0.0 <= row["theta_deg"] < 45.0
           for row in pretrained):
        raise ValueError("pretrained bias set must remain in the left crossing sector")

    bank_path = bundle / "quality" / "pretrained_initial_480_rows.json"
    bank = json.loads(bank_path.read_text())
    gamma1 = [row for row in bank if np.isclose(float(row["gamma"]), 1.0)]
    counts = {
        "total": len(gamma1),
        "success": sum(row["status"] == "SUCCESS" for row in gamma1),
        "collision": sum(row["status"] == "COLLISION" for row in gamma1),
        "oob": sum(row["status"] == "OOB" for row in gamma1),
        "successful_left": sum(
            row["status"] == "SUCCESS" and row["mode"] == "left" for row in gamma1
        ),
        "successful_above": sum(
            row["status"] == "SUCCESS" and row["mode"] == "above" for row in gamma1
        ),
    }
    expected_counts = {
        "total": 120, "success": 23, "collision": 89, "oob": 8,
        "successful_left": 21, "successful_above": 2,
    }
    if counts != expected_counts:
        raise ValueError(f"fixed-bank counts changed: {counts}")
    target = 180.0 - 120.56390067783855
    summary = {
        "schema": "paper_ready_0808_gamma1_expanded_pair_pretrained_bias_v1",
        "status": "PASS",
        "expanded_pair": expanded,
        "old_s6_theta_deg": 120.56390067783855,
        "mirrored_target_theta_deg": target,
        "seed_135403_mirror_error_deg": abs(float(expanded[1]["theta_deg"]) - target),
        "pretrained_biased_left": pretrained,
        "source_bank": "quality/pretrained_initial_480_rows.json",
        "source_bank_sha256": sha256_file(bank_path),
        "source_bank_gamma1_counts": counts,
        "interpretation": (
            "The eight pretrained paths are a disclosed qualitative subset of the "
            "fixed bank, selected to show the learned left-route bias; they are not an SR estimate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
