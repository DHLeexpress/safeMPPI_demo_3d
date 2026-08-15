#!/usr/bin/env python3
"""Calibrate an anticipatory taskspace cost at exact r10 OOB contexts."""
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


def _signed_margins(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.concatenate([
        points - bounds[:, 0],
        bounds[:, 1] - points,
    ], axis=1)


def _summary(probes: list[dict], arm: str) -> dict:
    rows = [row["selections"][arm] for row in probes if arm in row["selections"]]
    if not rows:
        return {"contexts": 0}
    values = lambda key: np.asarray([row[key] for row in rows], np.float64)
    margin = values("predicted_min_taskspace_margin_m")
    return {
        "contexts": len(rows),
        "choice_change_rate": float(np.mean(values("changed_from_native"))),
        "predicted_min_taskspace_margin_mean_m": float(np.mean(margin)),
        "predicted_min_taskspace_margin_median_m": float(np.median(margin)),
        "predicted_oob_plan_rate": float(np.mean(margin < 0.0)),
        "terminal_goal_distance_mean_m": float(np.mean(
            values("terminal_goal_distance_m")
        )),
        "native_execution_cost_mean": float(np.mean(values("native_cost"))),
        "taskspace_penalty_mean": float(np.mean(values("taskspace_penalty"))),
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
    parser.add_argument("--lookback-steps", type=int, default=10)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--target-m", type=float, default=0.15)
    parser.add_argument(
        "--weights", type=float, nargs="+", default=(50, 100, 250, 500, 1000),
    )
    parser.add_argument("--seed", type=int, default=93100)
    args = parser.parse_args()

    if args.contexts_per_gamma < 1 or args.lookback_steps < 1 or args.K < 1:
        parser.error("contexts, lookback, and K must be positive")
    if args.target_m < 0.0 or not np.isfinite(args.target_m):
        parser.error("target must be finite and nonnegative")
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
        execution_taskspace_quadratic_weight=0.0,
        execution_taskspace_quadratic_target_m=float(args.target_m),
    )
    saved = torch.load(args.raw_trajectories, map_location="cpu", weights_only=False)
    rows = list(saved.get(args.round, saved.get(str(args.round), [])))
    selected = []
    for gamma in config.data.gammas:
        candidates = [
            row for row in rows
            if float(row["gamma"]) == float(gamma) and row["status"] == "OOB"
        ]
        selected.extend(candidates[:args.contexts_per_gamma])
    expected = len(config.data.gammas) * args.contexts_per_gamma
    if len(selected) != expected:
        raise RuntimeError(f"needed {expected} OOB contexts, found {len(selected)}")

    probes = []
    with torch.no_grad():
        for probe_index, source in enumerate(selected):
            states = np.asarray(source["states"], np.float32)
            controls = np.asarray(source["controls"], np.float32)
            applied = np.asarray(source["applied_controls"], np.float32)
            dense = np.asarray(source["dense_steps"], np.float32)
            inside = task.env.inside_taskspace(dense.reshape(-1, 3))
            first_dense = int(np.flatnonzero(~inside)[0])
            first_step = min(first_dense // config.safemppi.integration_substeps, len(controls))
            context_step = max(0, first_step - args.lookback_steps)
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
                margins = _signed_margins(
                    planned_states[1:, :3], task.env.bounds,
                )
                shortfall = np.maximum(args.target_m - margins, 0.0)
                metrics[index] = {
                    "candidate": int(index),
                    "native_cost": float(verdicts[index].execution_cost),
                    "feature": float(np.square(shortfall).sum(axis=1).mean()),
                    "predicted_min_taskspace_margin_m": float(margins.min()),
                    "terminal_goal_distance_m": float(np.linalg.norm(
                        planned_states[-1, :3] - task.env.goal
                    )),
                    "full_h_margin": float(verdicts[index].margin),
                    "first_step_margin": float(verdicts[index].step_margin),
                }
            selections = {}
            if eligible:
                native = min(eligible, key=lambda index: metrics[index]["native_cost"])
                for weight in (0.0, *args.weights):
                    chosen = min(
                        eligible,
                        key=lambda index: (
                            metrics[index]["native_cost"]
                            + weight * metrics[index]["feature"]
                        ),
                    )
                    name = "native" if weight == 0.0 else f"w{weight:g}"
                    selection = dict(metrics[chosen])
                    selection.update({
                        "weight": float(weight),
                        "taskspace_penalty": float(weight * metrics[chosen]["feature"]),
                        "total_cost": float(
                            metrics[chosen]["native_cost"]
                            + weight * metrics[chosen]["feature"]
                        ),
                        "changed_from_native": bool(chosen != native),
                    })
                    selections[name] = selection
            probes.append({
                "probe": probe_index,
                "gamma": float(source["gamma"]),
                "episode": int(source["episode"]),
                "source_oob_step": int(first_step),
                "context_step": int(context_step),
                "eligible": len(eligible),
                "selections": selections,
            })
            print(
                f"[{probe_index + 1:02d}/{len(selected)}] gamma={source['gamma']:g} "
                f"episode={source['episode']} eligible={len(eligible)}/{args.K}",
                flush=True,
            )

    arms = ["native", *(f"w{weight:g}" for weight in args.weights)]
    output = {
        "status": "R10_TASKSPACE_EXECUTION_COST_CALIBRATION_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "taskspace_bounds": task.env.bounds.tolist(),
        "target_m": args.target_m,
        "K": args.K,
        "lookback_steps": args.lookback_steps,
        "summary": {arm: _summary(probes, arm) for arm in arms},
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
