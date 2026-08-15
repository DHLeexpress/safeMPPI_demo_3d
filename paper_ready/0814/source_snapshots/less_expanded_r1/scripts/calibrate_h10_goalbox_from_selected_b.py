#!/usr/bin/env python3
"""Audit H10/stop viability and goal-box costs on saved GP-selected B8."""
from __future__ import annotations

import argparse
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


WEIGHTS = (50.0, 100.0, 250.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task(config, weight: float, *, stopping: bool) -> LabFlowExpansionTask:
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
        execution_goal_box_exp_weight=weight,
        execution_goal_box_half_extent_m=0.20,
        execution_goal_box_exp_temperature_m=1.0,
        execution_axis_cylinder_quadratic_weight=5.0,
        execution_axis_cylinder_radius_m=1.10,
        execution_axis_cylinder_finite_segment=True,
        execution_goal_braking_weight=0.0,
        verifier_full_h_taskspace=stopping,
        verifier_stopping_margin_m=0.02 if stopping else None,
    )


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=(5, 9, 10))
    parser.add_argument("--events-per-gamma-round", type=int, default=80)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    config = load_config(args.task_config)
    baseline = _task(config, 0.0, stopping=False)
    guarded = {weight: _task(config, weight, stopping=True) for weight in WEIGHTS}
    rng = np.random.default_rng(args.seed)
    events: list[dict] = []
    sample_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for round_index in args.rounds:
        path = args.events_dir / f"events_round_{round_index:03d}.pt"
        source_hashes[path.name] = _sha256(path)
        round_events = torch.load(path, map_location="cpu", weights_only=False)
        for gamma in config.data.gammas:
            eligible = [
                event for event in round_events
                if float(event["gamma"]) == float(gamma)
                and len(event.get("selected", ())) == 8
            ]
            count = min(args.events_per_gamma_round, len(eligible))
            indices = rng.choice(len(eligible), count, replace=False)
            events.extend(eligible[int(index)] for index in indices)
            sample_counts[f"r{round_index}:g{float(gamma):g}"] = count

    arm_rows = {weight: [] for weight in WEIGHTS}
    total_candidates = 0
    viable_candidates = 0
    retained_contexts = 0
    for event in events:
        context = torch.as_tensor(event["context"], dtype=torch.float32)
        state6, previous_applied, previous_raw = baseline._decode_context(context)
        candidates = np.asarray(event["candidates"], np.float32)
        selected = [int(index) for index in event["selected"]]
        rows: dict[int, dict] = {}
        for candidate_index in selected:
            states, applied, dense_steps = baseline._rollout_plan(
                state6, previous_applied, candidates[candidate_index],
            )
            dense = np.concatenate([
                states[:1, :3], dense_steps.reshape(-1, 3),
            ])
            full_h_inside = bool(baseline.env.inside_taskspace(dense).all())
            stop_inside = bool(
                full_h_inside
                and guarded[WEIGHTS[0]]._stopping_backup_inside(
                    states[-1], applied[-1],
                )
            )
            displacement = np.asarray(states[-1, :3], np.float64) - baseline.axis_origin
            axial = float(displacement @ baseline.forward)
            radial = displacement - axial * baseline.forward
            rows[candidate_index] = {
                "states": states,
                "base_cost": baseline._execution_cost(
                    states, candidates[candidate_index], previous_raw,
                ),
                "full_h_inside": full_h_inside,
                "stop_inside": stop_inside,
                "progress_m": float(
                    np.linalg.norm(state6[:3] - baseline.env.goal)
                    - np.linalg.norm(states[-1, :3] - baseline.env.goal)
                ),
                "lateral_terminal_m": float(np.linalg.norm(radial)),
            }
        total_candidates += len(rows)
        viable = [index for index, row in rows.items() if row["stop_inside"]]
        viable_candidates += len(viable)
        retained_contexts += bool(viable)
        base_choice = min(rows, key=lambda index: rows[index]["base_cost"])
        for weight, task in guarded.items():
            if not viable:
                arm_rows[weight].append({"retained": False})
                continue
            choice = min(
                viable,
                key=lambda index: task._execution_cost(
                    rows[index]["states"], candidates[index], previous_raw,
                ),
            )
            base = rows[choice]["base_cost"]
            cost = task._execution_cost(
                rows[choice]["states"], candidates[choice], previous_raw,
            )
            arm_rows[weight].append({
                "retained": True,
                "choice_changed": choice != base_choice,
                "goal_box_feature": (cost - base) / weight,
                "progress_m": rows[choice]["progress_m"],
                "lateral_terminal_m": rows[choice]["lateral_terminal_m"],
            })

    results = {}
    for weight, rows in arm_rows.items():
        retained = [row for row in rows if row["retained"]]
        features = np.asarray([
            row["goal_box_feature"] for row in retained
        ], np.float64)
        results[f"w{weight:g}_t1"] = {
            "weight": weight,
            "temperature_m": 1.0,
            "half_extent_m": 0.20,
            "retained_contexts": len(retained),
            "retained_context_rate": len(retained) / max(len(rows), 1),
            "choice_change_rate": _mean(retained, "choice_changed"),
            "progress_mean_m": _mean(retained, "progress_m"),
            "lateral_terminal_mean_m": _mean(retained, "lateral_terminal_m"),
            "goal_box_feature_quantiles": {
                str(q): float(np.quantile(features, q))
                for q in (0.5, 0.9, 0.99)
            },
        }

    payload = {
        "status": "SAVED_GP_SELECTED_B8_GOALBOX_CALIBRATION_COMPLETE",
        "selection_contract": (
            "only event contexts with exactly eight uncertainty-acquisition "
            "selected indices are included; retry all-K fast-path contexts are "
            "excluded"
        ),
        "events_dir": str(args.events_dir.resolve()),
        "source_hashes": source_hashes,
        "sample_counts": sample_counts,
        "contexts": len(events),
        "selected_candidates": total_candidates,
        "full_h_plus_stop_viable_candidates": viable_candidates,
        "full_h_plus_stop_candidate_rate": (
            viable_candidates / max(total_candidates, 1)
        ),
        "full_h_plus_stop_retained_contexts": retained_contexts,
        "full_h_plus_stop_retained_context_rate": (
            retained_contexts / max(len(events), 1)
        ),
        "goal": baseline.env.goal.tolist(),
        "goal_box": {
            "lower": (baseline.env.goal - 0.20).tolist(),
            "upper": (baseline.env.goal + 0.20).tolist(),
            "physical_taskspace_unchanged": baseline.env.bounds.tolist(),
        },
        "retry_contract": {
            "K": 16,
            "B": 8,
            "retry_B": 8,
            "retry_verify_all_fast_path": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
