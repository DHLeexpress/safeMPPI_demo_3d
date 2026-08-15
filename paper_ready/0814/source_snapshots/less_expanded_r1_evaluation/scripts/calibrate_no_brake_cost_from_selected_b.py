#!/usr/bin/env python3
"""Rescore the actual GP-selected B candidates under no-brake costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.lab_flow_expansion import LabFlowExpansionTask  # noqa: E402


ARMS = (
    {
        "name": "baseline_revised",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 750.0,
        "wall_target_m": 0.18,
        "axis_weight": 5.0,
        "axis_radius_m": 0.80,
        "control_weight": 5.0,
        "terminal_weight": 80.0,
    },
    {
        "name": "terminal120",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 750.0,
        "wall_target_m": 0.18,
        "axis_weight": 5.0,
        "axis_radius_m": 0.80,
        "control_weight": 5.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "control10_terminal120",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 750.0,
        "wall_target_m": 0.18,
        "axis_weight": 5.0,
        "axis_radius_m": 0.80,
        "control_weight": 10.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "wall1250",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 1250.0,
        "wall_target_m": 0.18,
        "axis_weight": 5.0,
        "axis_radius_m": 0.80,
        "control_weight": 10.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "clearance050",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.50,
        "wall_weight": 750.0,
        "wall_target_m": 0.18,
        "axis_weight": 5.0,
        "axis_radius_m": 0.80,
        "control_weight": 10.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "axis10",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 750.0,
        "wall_target_m": 0.18,
        "axis_weight": 10.0,
        "axis_radius_m": 0.80,
        "control_weight": 10.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "wall5000_axis25",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 5000.0,
        "wall_target_m": 0.19,
        "axis_weight": 25.0,
        "axis_radius_m": 0.80,
        "control_weight": 5.0,
        "terminal_weight": 120.0,
    },
    {
        "name": "wall20000_axis100",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.55,
        "wall_weight": 20000.0,
        "wall_target_m": 0.19,
        "axis_weight": 100.0,
        "axis_radius_m": 0.80,
        "control_weight": 10.0,
        "terminal_weight": 160.0,
    },
    {
        "name": "wall50000_axis250",
        "clearance_weight": 2500.0,
        "clearance_target_m": 0.50,
        "wall_weight": 50000.0,
        "wall_target_m": 0.19,
        "axis_weight": 250.0,
        "axis_radius_m": 0.80,
        "control_weight": 20.0,
        "terminal_weight": 240.0,
    },
)


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"contexts": 0}
    return {
        "contexts": len(rows),
        "choice_change_from_previous_rate": _mean(
            rows, "changed_from_previous"
        ),
        "predicted_h_oob_rate": _mean(rows, "predicted_h_oob"),
        "predicted_near_wall_lt_005_rate": _mean(rows, "near_wall_lt_005"),
        "predicted_min_face_clearance_mean_m": _mean(
            rows, "min_face_clearance_m"
        ),
        "terminal_goal_distance_mean_m": _mean(
            rows, "terminal_goal_distance_m"
        ),
        "terminal_axis_distance_mean_m": _mean(
            rows, "terminal_axis_distance_m"
        ),
        "terminal_forward_speed_mean_mps": _mean(
            rows, "terminal_forward_speed_mps"
        ),
        "control_energy_mean": _mean(rows, "control_energy"),
        "predicted_progress_mean_m": _mean(rows, "predicted_progress_m"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=range(1, 6))
    parser.add_argument("--events-per-gamma-round", type=int, default=120)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()

    config = load_config(args.task_config)
    task = LabFlowExpansionTask(
        config,
        context_schema="lab_spherical_hp3d_uniform_radial100_planepack_v1",
        device="cpu",
        tight_corridor=True,
        execution_obstacle_cost="none",
        execution_control_weight=float(config.safemppi.control_weight),
        execution_terminal_goal_weight=float(
            config.safemppi.terminal_goal_weight
        ),
        execution_taskspace_quadratic_weight=0.0,
        execution_axis_cylinder_quadratic_weight=0.0,
        execution_goal_braking_weight=0.0,
    )
    rng = np.random.default_rng(args.seed)
    events: list[tuple[int, dict]] = []
    counts = {}
    for round_index in args.rounds:
        path = args.events_dir / f"events_round_{round_index:03d}.pt"
        round_events = torch.load(
            path, map_location="cpu", weights_only=False
        )
        for gamma in config.data.gammas:
            candidates = [
                event for event in round_events
                if float(event["gamma"]) == float(gamma)
                and len(event.get("selected", ())) > 0
            ]
            take = min(args.events_per_gamma_round, len(candidates))
            indices = rng.choice(len(candidates), take, replace=False)
            events.extend((round_index, candidates[index]) for index in indices)
            counts[f"r{round_index}:g{float(gamma):g}"] = take

    selected_rows: dict[str, list[dict]] = {
        arm["name"]: [] for arm in ARMS
    }
    for round_index, event in events:
        context = torch.as_tensor(event["context"], dtype=torch.float32)
        state6, previous_applied, previous_raw = task._decode_context(context)
        candidates = np.asarray(event["candidates"], np.float32)
        selected = [int(index) for index in event["selected"]]
        previous_candidate = selected[int(event["chosen_local"])]
        metrics = {}
        for candidate_index in selected:
            plan = candidates[candidate_index]
            states, _, _ = task._rollout_plan(
                state6, previous_applied, plan
            )
            positions = np.asarray(states[1:, :3], np.float64)
            face_clearance = np.concatenate([
                positions - task.env.bounds[:, 0],
                task.env.bounds[:, 1] - positions,
            ], axis=1)
            obstacle_clearance = np.asarray(
                task.env.obstacle_clearance(positions), np.float64
            )
            displacement = positions[-1] - task.axis_origin
            axial = float(displacement @ task.forward)
            closest = float(np.clip(axial, 0.0, task.axis_length))
            radial = displacement - closest * task.forward
            metrics[candidate_index] = {
                "native_cost": task._native_cost(
                    states, plan, previous_raw
                ),
                "face_clearance": face_clearance,
                "obstacle_clearance": obstacle_clearance,
                "terminal_axis_distance_m": float(np.linalg.norm(radial)),
                "terminal_goal_distance_m": float(np.linalg.norm(
                    positions[-1] - task.env.goal
                )),
                "terminal_forward_speed_mps": float(
                    np.asarray(states[-1, 3:6], np.float64) @ task.forward
                ),
                "control_energy": float(np.square(plan).sum()),
                "predicted_progress_m": float(
                    np.linalg.norm(state6[:3] - task.env.goal)
                    - np.linalg.norm(positions[-1] - task.env.goal)
                ),
            }
        for arm in ARMS:
            def cost(candidate_index: int) -> float:
                row = metrics[candidate_index]
                wall_shortfall = np.maximum(
                    arm["wall_target_m"] - row["face_clearance"], 0.0
                )
                obstacle_shortfall = np.maximum(
                    arm["clearance_target_m"]
                    - row["obstacle_clearance"],
                    0.0,
                )
                return float(
                    row["native_cost"]
                    + arm["clearance_weight"]
                    * np.square(obstacle_shortfall).mean()
                    + arm["wall_weight"]
                    * np.square(wall_shortfall).sum(axis=1).mean()
                    + arm["axis_weight"]
                    * (
                        row["terminal_axis_distance_m"]
                        / arm["axis_radius_m"]
                    ) ** 2
                    + (
                        arm["control_weight"]
                        - float(config.safemppi.control_weight)
                    ) * row["control_energy"]
                    + (
                        arm["terminal_weight"]
                        - float(config.safemppi.terminal_goal_weight)
                    ) * row["terminal_goal_distance_m"] ** 2
                )

            chosen = min(selected, key=cost)
            row = metrics[chosen]
            min_face = float(row["face_clearance"].min())
            selected_rows[arm["name"]].append({
                "round": int(round_index),
                "gamma": float(event["gamma"]),
                "retry_batch": int(event["retry_batch"]),
                "chosen_candidate": int(chosen),
                "previous_candidate": int(previous_candidate),
                "changed_from_previous": bool(chosen != previous_candidate),
                "predicted_h_oob": bool(min_face < 0.0),
                "near_wall_lt_005": bool(min_face < 0.05),
                "min_face_clearance_m": min_face,
                **{
                    key: float(row[key]) for key in (
                        "terminal_goal_distance_m",
                        "terminal_axis_distance_m",
                        "terminal_forward_speed_mps",
                        "control_energy",
                        "predicted_progress_m",
                    )
                },
            })

    summary = {}
    for arm in ARMS:
        rows = selected_rows[arm["name"]]
        summary[arm["name"]] = {
            "config": arm,
            "pooled": _summarize(rows),
            "per_gamma": {
                f"{float(gamma):g}": _summarize([
                    row for row in rows
                    if float(row["gamma"]) == float(gamma)
                ])
                for gamma in config.data.gammas
            },
        }
    output = {
        "status": "ACTUAL_GP_SELECTED_B_NO_BRAKE_CALIBRATION_COMPLETE",
        "source_events": str(args.events_dir.resolve()),
        "source_contract": (
            "events preserve the actual uncertainty-acquisition selected B8 "
            "indices from K16; only those B candidates are rescored"
        ),
        "gp_caveat": (
            "fresh round 1 has no prior successful GP buffer; prior-success "
            "uncertainty tilting starts in round 2"
        ),
        "braking_weight": 0.0,
        "sample_counts": counts,
        "contexts": len(events),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        name: values["pooled"] for name, values in summary.items()
    }, indent=2))


if __name__ == "__main__":
    main()
