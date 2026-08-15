#!/usr/bin/env python3
"""Calibrate the terminal x_max/y_min wall cost on saved GP-selected B8."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.lab_flow_expansion import LabFlowExpansionTask  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task(
    config, wall_weight: float, braking_weight: float,
) -> LabFlowExpansionTask:
    return LabFlowExpansionTask(
        config,
        context_schema="lab_spherical_hp3d_uniform_radial100_planepack_v1",
        device="cpu",
        tight_corridor=True,
        verifier_mode="full_polytope",
        verifier_solver="analytic",
        execution_obstacle_cost="quadratic",
        execution_clearance_quadratic_weight=2500.0,
        execution_clearance_quadratic_target_m=0.60,
        execution_control_weight=1.0,
        execution_terminal_goal_weight=80.0,
        execution_taskspace_quadratic_weight=0.0,
        execution_goal_side_wall_quadratic_weight=wall_weight,
        execution_goal_side_wall_target_m=0.60,
        execution_goal_box_exp_weight=50.0,
        execution_goal_box_half_extent_m=0.20,
        execution_goal_box_exp_temperature_m=1.0,
        execution_axis_cylinder_quadratic_weight=5.0,
        execution_axis_cylinder_radius_m=1.10,
        execution_axis_cylinder_finite_segment=True,
        execution_goal_braking_weight=braking_weight,
        execution_goal_braking_distance_m=0.60,
        execution_goal_braking_temperature_m=0.15,
        verifier_full_h_taskspace=True,
        verifier_stopping_margin_m=None,
    )


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {str(q): float(np.quantile(array, q)) for q in (0.5, 0.9, 0.99)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=(5,))
    parser.add_argument("--events-per-gamma-round", type=int, default=80)
    parser.add_argument("--weights", type=float, nargs="+", default=(10, 25, 50, 100))
    parser.add_argument(
        "--braking-weights", type=float, nargs="+", default=(0, 25),
    )
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    config = load_config(args.task_config)
    baseline = _task(config, 0.0, 0.0)
    designs = {
        (float(weight), float(braking)): _task(
            config, float(weight), float(braking),
        )
        for weight in args.weights
        for braking in args.braking_weights
    }
    expected_bounds = np.asarray([
        [-2.5, 1.3], [-2.1, 1.8], [0.1, 1.7],
    ])
    if not np.allclose(baseline.env.bounds, expected_bounds, rtol=0.0, atol=1e-9):
        raise ValueError("calibration requires the approved y_min-expanded bounds")
    goal_clearance = np.asarray([
        baseline.env.bounds[0, 1] - baseline.env.goal[0],
        baseline.env.goal[1] - baseline.env.bounds[1, 0],
    ])
    if not np.allclose(goal_clearance, [0.6, 0.6], rtol=0.0, atol=1e-6):
        raise ValueError("goal-side clearances are not symmetric at 0.6 m")

    rng = np.random.default_rng(args.seed)
    sampled: list[dict] = []
    sample_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for round_i in args.rounds:
        path = args.events_dir / f"events_round_{round_i:03d}.pt"
        source_hashes[path.name] = _sha256(path)
        events = torch.load(path, map_location="cpu", weights_only=False)
        for gamma in config.data.gammas:
            available = [
                event for event in events
                if float(event["gamma"]) == float(gamma)
                and len(event.get("selected", ())) == 8
            ]
            count = min(args.events_per_gamma_round, len(available))
            indices = rng.choice(len(available), count, replace=False)
            sampled.extend(available[int(index)] for index in indices)
            sample_counts[f"r{round_i}:g{float(gamma):g}"] = count
        del events

    results = {design: [] for design in designs}
    candidate_counts = Counter()
    context_counts = Counter()
    feature_values: list[float] = []
    for event in sampled:
        context = torch.as_tensor(event["context"], dtype=torch.float32)
        candidates = torch.as_tensor(event["candidates"], dtype=torch.float32)
        selected = [int(index) for index in event["selected"]]
        selected_candidates = candidates[selected]
        verification = baseline.verify(
            context, selected_candidates, float(event["gamma"]),
        )
        state6, previous_applied, previous_raw = baseline._decode_context(context)
        context_goal_distance = float(np.linalg.norm(
            state6[:3] - baseline.env.goal
        ))
        rows = []
        for local, (candidate, result) in enumerate(zip(selected_candidates, verification)):
            states, _, dense_steps = baseline._rollout_plan(
                state6, previous_applied, candidate.numpy(),
            )
            dense = np.concatenate([states[:1, :3], dense_steps.reshape(-1, 3)])
            terminal = np.asarray(states[-1, :3], np.float64)
            clearances = np.asarray([
                baseline.env.bounds[0, 1] - terminal[0],
                terminal[1] - baseline.env.bounds[1, 0],
            ])
            shortfall = np.maximum(0.60 - clearances, 0.0)
            feature = float(np.square(shortfall).sum())
            feature_values.append(feature)
            rows.append({
                "local": local,
                "candidate": candidate,
                "states": states,
                "result": result,
                "base_cost": float(result.execution_cost),
                "eligible": bool(result.valid and result.progress_eligible),
                "terminal": terminal,
                "clearances": clearances,
                "wall_feature": feature,
                "new_full_h_inside": bool(
                    baseline.env.inside_taskspace(dense).all()
                ),
            })
        candidate_counts["selected"] += len(rows)
        candidate_counts["eligible"] += sum(row["eligible"] for row in rows)
        candidate_counts["new_full_h_inside"] += sum(
            row["new_full_h_inside"] for row in rows
        )
        eligible = [row for row in rows if row["eligible"]]
        context_counts["contexts"] += 1
        context_counts["retained"] += bool(eligible)
        if not eligible:
            for design in results:
                results[design].append({
                    "retained": False,
                    "context_goal_distance_m": context_goal_distance,
                })
            continue
        base_choice = min(eligible, key=lambda row: row["base_cost"])
        for (weight, braking), task in designs.items():
            choice = min(
                eligible,
                key=lambda row: task._execution_cost(
                    row["states"], row["candidate"].numpy(), previous_raw,
                ),
            )
            weighted_cost = task._execution_cost(
                choice["states"], choice["candidate"].numpy(), previous_raw,
            )
            results[(weight, braking)].append({
                "retained": True,
                "context_goal_distance_m": context_goal_distance,
                "choice_changed": choice["local"] != base_choice["local"],
                "wall_feature": choice["wall_feature"],
                "wall_cost": weight * choice["wall_feature"],
                "total_added_cost": weighted_cost - choice["base_cost"],
                "x_max_clearance_m": float(choice["clearances"][0]),
                "y_min_clearance_m": float(choice["clearances"][1]),
                "terminal_goal_distance_m": float(np.linalg.norm(
                    choice["terminal"] - baseline.env.goal
                )),
                "terminal_beyond_goal_side": bool(
                    choice["terminal"][0] > baseline.env.goal[0]
                    or choice["terminal"][1] < baseline.env.goal[1]
                ),
            })

    def summarize_rows(rows: list[dict]) -> dict:
        retained = [row for row in rows if row["retained"]]
        return {
            "contexts": len(rows),
            "retained_contexts": len(retained),
            "choice_change_rate": _mean(retained, "choice_changed"),
            "terminal_beyond_goal_side_rate": _mean(
                retained, "terminal_beyond_goal_side"
            ),
            "x_max_clearance_mean_m": _mean(retained, "x_max_clearance_m"),
            "y_min_clearance_mean_m": _mean(retained, "y_min_clearance_m"),
            "terminal_goal_distance_mean_m": _mean(
                retained, "terminal_goal_distance_m"
            ),
            "wall_feature_quantiles": _quantiles([
                row["wall_feature"] for row in retained
            ]),
            "wall_cost_quantiles": _quantiles([
                row["wall_cost"] for row in retained
            ]),
            "total_added_cost_quantiles": _quantiles([
                row["total_added_cost"] for row in retained
            ]),
        }

    summaries = {}
    for (weight, braking), rows in results.items():
        summaries[f"wall{weight:g}_brake{braking:g}"] = {
            "wall_weight": weight,
            "braking_weight": braking,
            "all": summarize_rows(rows),
            "remaining_distance_le_1p2m": summarize_rows([
                row for row in rows
                if row["context_goal_distance_m"] <= 1.2
            ]),
            "remaining_distance_le_0p8m": summarize_rows([
                row for row in rows
                if row["context_goal_distance_m"] <= 0.8
            ]),
        }

    payload = {
        "status": "SAVED_GP_SELECTED_B8_GOAL_SIDE_WALL_CALIBRATION_COMPLETE",
        "selection_contract": (
            "saved uncertainty-selected B8 are reused; candidates are "
            "reverified under the y_min-expanded physical task space"
        ),
        "events_dir": str(args.events_dir.resolve()),
        "task_config": str(args.task_config.resolve()),
        "source_hashes": source_hashes,
        "sample_counts": sample_counts,
        "bounds": baseline.env.bounds.tolist(),
        "goal": baseline.env.goal.tolist(),
        "goal_side_faces": ["x_max", "y_min"],
        "goal_side_clearance_m": goal_clearance.tolist(),
        "target_m": 0.60,
        "candidate_counts": dict(candidate_counts),
        "context_counts": dict(context_counts),
        "all_candidate_wall_feature_quantiles": _quantiles(feature_values),
        "results": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
