#!/usr/bin/env python3
"""Search PRE2, R1, and S4 rollouts on the measured bowling scene."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


BUNDLE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BUNDLE / "source"), str(BUNDLE / "runtime_snapshot")]

from evaluate_multisphere_min_cost_deployment import _rollout  # noqa: E402
from real_bowling_scene import (  # noqa: E402
    RealBowlingTask,
    hard_path_diagnostics,
    load_as_built_geometry,
)
from safe_mppi.bowling_coverage import bowling_route_signature  # noqa: E402
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.lab_clutter_expansion import LAB_CLUTTER_GOVERNOR_DIM  # noqa: E402
from safe_mppi.lab_clutter_pre2_expansion import (  # noqa: E402
    load_lab_clutter_pre2_expansion_policy,
)


CHECKPOINTS = {
    "pre2": BUNDLE / "checkpoints/pre2/pretrained.pt",
    "less-expanded": BUNDLE / "checkpoints/less_expanded/checkpoint_001.pt",
    "expanded": BUNDLE / "checkpoints/expanded/checkpoint_004.pt",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dense_path(row: dict) -> np.ndarray:
    states = np.asarray(row["states"], np.float32)
    dense = np.asarray(row["dense_steps"], np.float32).reshape(-1, 3)
    return np.concatenate([states[:1, :3], dense], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--seed-start", type=int, default=814300000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="run these exact rollout seeds instead of a consecutive range",
    )
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    geometry = load_as_built_geometry(args.scene_json)
    config = load_config(BUNDLE / "config/task_config_resolved.json")
    wrapped = load_lab_clutter_pre2_expansion_policy(
        CHECKPOINTS["pre2"],
        verifier_suffix_dim=LAB_CLUTTER_GOVERNOR_DIM + 25,
    ).to(args.device).eval()
    wrapped.policy.nfe = 16
    wrapped.policy.flow.nfe = 16
    if args.model != "pre2":
        checkpoint = torch.load(
            CHECKPOINTS[args.model], map_location="cpu", weights_only=False,
        )
        wrapped.policy.load_state_dict(checkpoint["model"], strict=True)

    task = RealBowlingTask(
        config,
        context_schema=wrapped.context_schema,
        device=args.device,
        tight_corridor=False,
        verifier_mode="full_polytope",
        verifier_solver="analytic",
        execution_clearance_exp_weight=15.0,
        execution_clearance_target_m=0.6,
        execution_clearance_exp_temperature=0.15,
        execution_taskspace_quadratic_weight=250.0,
        execution_taskspace_quadratic_target_m=0.15,
        execution_axis_cylinder_quadratic_weight=5.0,
        execution_axis_cylinder_radius_m=1.1,
        execution_control_weight=0.05,
        execution_obstacle_speed_weight=400.0,
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=0,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        fixed_scene_layout="none",
        physical_spheres=geometry["physical_spheres"],
        effective_spheres=geometry["effective_spheres"],
        string_radius_m=geometry["string_radius_m"],
    )

    seeds = (
        list(args.seeds)
        if args.seeds is not None else
        [args.seed_start + trial for trial in range(args.trials)]
    )
    rows = []
    for trial, seed in enumerate(seeds):
        row = _rollout(
            wrapped, task, config, geometry["effective_spheres"],
            float(args.gamma), seed,
            samples_per_step=1,
            sampling_temperature=float(args.sampling_temperature),
            execution_cost_band_fraction=0.05,
        )
        diagnostics = hard_path_diagnostics(
            _dense_path(row),
            geometry["effective_spheres"],
            geometry["physical_spheres"],
            geometry["string_radius_m"],
        )
        route = None
        if row["status"] == "SUCCESS" and diagnostics["hard_valid"]:
            route = bowling_route_signature(
                row["states"], task.env.start, task.env.goal,
                sphere_radius_m=float(np.max(geometry["effective_spheres"][:, 3])),
            )
        rows.append({
            "method": args.model,
            "gamma": float(args.gamma),
            "trial": trial,
            "rollout_seed": seed,
            "bowling_route": route,
            "hard_constraints": diagnostics,
            **row,
        })
        print(
            f"[{trial + 1}/{len(seeds)}] {args.model} gamma={args.gamma:g} "
            f"seed={seed} {row['status']} hard={diagnostics['hard_valid']} "
            f"route={None if route is None else route.get('stable_code')}",
            flush=True,
        )

    args.output.mkdir(parents=True)
    raw_path = args.output / "raw_trajectories.pt"
    torch.save(rows, raw_path)
    summary = {
        "status": "REAL_BOWLING_POLICY_SEARCH_COMPLETE",
        "method": args.model,
        "gamma": float(args.gamma),
        "trials": len(seeds),
        "seed_start": None if args.seeds is not None else args.seed_start,
        "seeds": seeds,
        "sampling_temperature": float(args.sampling_temperature),
        "successes": sum(row["status"] == "SUCCESS" for row in rows),
        "hard_valid_successes": sum(
            row["status"] == "SUCCESS"
            and row["hard_constraints"]["hard_valid"]
            for row in rows
        ),
        "hard_valid_route_counts": {
            route: sum(
                (row.get("bowling_route") or {}).get("stable_code") == route
                for row in rows
            )
            for route in ("LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR")
        },
        "geometry": {
            "physical_spheres": geometry["physical_spheres"].tolist(),
            "effective_spheres": geometry["effective_spheres"].tolist(),
            "effective_margin_m": geometry["effective_margin_m"],
            "string_radius_m": geometry["string_radius_m"],
            "string_start_z_m": geometry["string_start_z_m"].tolist(),
        },
        "artifact_binding": {
            "checkpoint": str(CHECKPOINTS[args.model].relative_to(BUNDLE)),
            "checkpoint_sha256": _sha256(CHECKPOINTS[args.model]),
            "scene_json_sha256": _sha256(args.scene_json),
            "wrapper_sha256": _sha256(Path(__file__)),
            "geometry_adapter_sha256": _sha256(BUNDLE / "source/real_bowling_scene.py"),
            "raw_sha256": _sha256(raw_path),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
