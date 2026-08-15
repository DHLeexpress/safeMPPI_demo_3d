#!/usr/bin/env python3
"""Calibrate a start-goal-axis cylinder cost at r10 lateral contexts."""
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
from safe_mppi.lab_flow_expansion import (  # noqa: E402
    LabFlowExpansionTask,
    load_lab_expansion_policy,
)
from safe_mppi.lab_reference_flow_task import _raw_history_before  # noqa: E402


def _axis_distance(points: np.ndarray, origin: np.ndarray, axis: np.ndarray):
    displacement = np.asarray(points, np.float64) - origin
    axial = displacement @ axis
    radial = displacement - axial[:, None] * axis[None]
    return np.linalg.norm(radial, axis=1)


def _summary(probes: list[dict], arm: str) -> dict:
    rows = [row["selections"][arm] for row in probes if arm in row["selections"]]
    if not rows:
        return {"contexts": 0}
    values = lambda key: np.asarray([row[key] for row in rows], np.float64)
    return {
        "contexts": len(rows),
        "choice_change_rate": float(np.mean(values("changed_from_native"))),
        "predicted_max_axis_distance_mean_m": float(np.mean(
            values("predicted_max_axis_distance_m")
        )),
        "predicted_outside_cylinder_rate": float(np.mean(
            values("predicted_max_axis_distance_m") > rows[0]["radius_m"]
        )),
        "terminal_goal_distance_mean_m": float(np.mean(
            values("terminal_goal_distance_m")
        )),
        "native_execution_cost_mean": float(np.mean(values("native_cost"))),
        "taskspace_penalty_mean": float(np.mean(values("taskspace_penalty"))),
        "cylinder_penalty_mean": float(np.mean(values("cylinder_penalty"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--raw-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contexts-per-gamma", type=int, default=4)
    parser.add_argument("--lookback-steps", type=int, default=5)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--radius-m", type=float, default=1.10)
    parser.add_argument("--taskspace-weight", type=float, default=500.0)
    parser.add_argument("--taskspace-target-m", type=float, default=0.15)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=(25, 50, 100, 250, 500, 1000, 2500),
    )
    parser.add_argument("--seed", type=int, default=94100)
    args = parser.parse_args()

    if args.radius_m <= 0.0 or not np.isfinite(args.radius_m):
        parser.error("radius must be finite and positive")
    if any(weight < 0.0 or not np.isfinite(weight) for weight in args.weights):
        parser.error("weights must be finite and nonnegative")

    device = torch.device(args.device)
    config = load_config(args.task_config)
    policy = load_lab_expansion_policy(
        args.pretrain_dir / "pretrained.pt"
    ).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.policy.nfe = int(args.flow_nfe)
    policy.policy.flow.nfe = int(args.flow_nfe)
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        device=device,
        tight_corridor=True,
        execution_obstacle_cost="quadratic",
        execution_clearance_quadratic_weight=2500.0,
        execution_clearance_quadratic_target_m=0.6,
        execution_taskspace_quadratic_weight=float(args.taskspace_weight),
        execution_taskspace_quadratic_target_m=float(args.taskspace_target_m),
    )
    saved = torch.load(args.raw_trajectories, map_location="cpu", weights_only=False)
    rows = list(saved.get(args.round, saved.get(str(args.round), [])))
    selected: list[tuple[dict, int, float]] = []
    for gamma in config.data.gammas:
        ranked = []
        for row in rows:
            if float(row["gamma"]) != float(gamma):
                continue
            if row["status"] not in {"SUCCESS", "OOB"}:
                continue
            states = np.asarray(row["states"], np.float32)
            distances = _axis_distance(states[:, :3], task.axis_origin, task.forward)
            max_step = int(np.argmax(distances))
            ranked.append((float(distances[max_step]), row, max_step))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected.extend(
            (row, step, distance)
            for distance, row, step in ranked[:args.contexts_per_gamma]
        )
    expected = len(config.data.gammas) * args.contexts_per_gamma
    if len(selected) != expected:
        raise RuntimeError(f"needed {expected} lateral contexts, found {len(selected)}")

    probes = []
    with torch.no_grad():
        for probe_index, (source, max_step, source_max_distance) in enumerate(selected):
            states = np.asarray(source["states"], np.float32)
            controls = np.asarray(source["controls"], np.float32)
            applied = np.asarray(source["applied_controls"], np.float32)
            context_step = max(0, min(max_step - args.lookback_steps, len(controls) - 1))
            state = task.reset(float(source["gamma"]), probe_index, args.seed)
            state["x"] = states[context_step].copy()
            state["steps"] = int(context_step)
            state["raw_history"] = _raw_history_before(controls, context_step)
            if context_step:
                state["previous_raw"] = controls[context_step - 1].copy()
                state["previous_applied"] = applied[context_step - 1].copy()
            context = task.context(state, float(source["gamma"]))
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1009 * probe_index
            )
            plans = policy.sample(context, args.K, generator, base_std=1.0)
            verdicts = task.verify(context, plans, float(source["gamma"]))
            eligible = [
                index for index, verdict in enumerate(verdicts)
                if not verdict.error and verdict.valid and verdict.progress_eligible
            ]
            metrics = {}
            for index in eligible:
                plan = plans[index].detach().cpu().numpy().reshape(-1, 3)
                planned_states, _, _ = task._rollout_plan(
                    state["x"], state["previous_applied"], plan,
                )
                distances = _axis_distance(
                    planned_states[1:, :3], task.axis_origin, task.forward,
                )
                cylinder_feature = float(np.square(
                    distances[-1] / args.radius_m
                ))
                face_clearance = np.concatenate([
                    planned_states[1:, :3] - task.env.bounds[:, 0],
                    task.env.bounds[:, 1] - planned_states[1:, :3],
                ], axis=1)
                taskspace_feature = float(np.square(np.maximum(
                    args.taskspace_target_m - face_clearance, 0.0,
                )).sum(axis=1).mean())
                metrics[index] = {
                    "candidate": int(index),
                    "native_cost": float(verdicts[index].execution_cost),
                    "cylinder_feature": cylinder_feature,
                    "predicted_max_axis_distance_m": float(distances.max()),
                    "terminal_goal_distance_m": float(np.linalg.norm(
                        planned_states[-1, :3] - task.env.goal
                    )),
                    "taskspace_penalty": float(
                        args.taskspace_weight * taskspace_feature
                    ),
                    "planned_path": np.round(
                        planned_states[:, :3], 5,
                    ).tolist(),
                }
            selections = {}
            if eligible:
                native = min(eligible, key=lambda index: metrics[index]["native_cost"])
                for weight in (0.0, *args.weights):
                    chosen = min(
                        eligible,
                        key=lambda index: (
                            metrics[index]["native_cost"]
                            + weight * metrics[index]["cylinder_feature"]
                        ),
                    )
                    name = "native" if weight == 0.0 else f"w{weight:g}"
                    selection = dict(metrics[chosen])
                    selection.update({
                        "weight": float(weight),
                        "radius_m": float(args.radius_m),
                        "cylinder_penalty": float(
                            weight * metrics[chosen]["cylinder_feature"]
                        ),
                        "total_cost": float(
                            metrics[chosen]["native_cost"]
                            + weight * metrics[chosen]["cylinder_feature"]
                        ),
                        "changed_from_native": bool(chosen != native),
                    })
                    selections[name] = selection
            probes.append({
                "probe": probe_index,
                "gamma": float(source["gamma"]),
                "episode": int(source["episode"]),
                "status": str(source["status"]),
                "mode": str(source.get("mode", "none")),
                "source_max_axis_distance_m": source_max_distance,
                "context_step": int(context_step),
                "context_position": np.round(states[context_step, :3], 5).tolist(),
                "eligible": len(eligible),
                "selections": selections,
            })
            print(
                f"[{probe_index + 1:02d}/{len(selected)}] "
                f"gamma={source['gamma']:g} mode={source.get('mode')} "
                f"eligible={len(eligible)}/{args.K}",
                flush=True,
            )

    arms = ["native", *(f"w{weight:g}" for weight in args.weights)]
    output = {
        "status": "R10_AXIS_CYLINDER_EXECUTION_COST_CALIBRATION_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "taskspace_bounds": task.env.bounds.tolist(),
        "start": task.env.start[:3].tolist(),
        "goal": task.env.goal.tolist(),
        "axis": task.forward.tolist(),
        "radius_m": float(args.radius_m),
        "diameter_m": float(2.0 * args.radius_m),
        "taskspace_weight": float(args.taskspace_weight),
        "taskspace_target_m": float(args.taskspace_target_m),
        "K": int(args.K),
        "summary": {arm: _summary(probes, arm) for arm in arms},
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
