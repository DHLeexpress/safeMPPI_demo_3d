#!/usr/bin/env python3
"""Calibrate symmetric execution penalties at paired PRE collision contexts."""
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


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    result = np.quantile(np.asarray(values, np.float64), [0.25, 0.5, 0.75])
    return {"q25": float(result[0]), "median": float(result[1]), "q75": float(result[2])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--collision-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=(0.1, 0.3, 0.5, 1.0),
    )
    parser.add_argument("--contexts-per-gamma", type=int, default=4)
    parser.add_argument("--lookback-steps", type=int, default=15)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--flow-base-std", type=float, default=1.0)
    parser.add_argument(
        "--exponential-weights", type=float, nargs="+", default=(30.0, 100.0, 300.0),
    )
    parser.add_argument(
        "--quadratic-weights", type=float, nargs="+", default=(1000.0, 3000.0, 10000.0),
    )
    parser.add_argument("--exponential-temperature", type=float, default=0.15)
    parser.add_argument("--exponential-target-m", type=float, default=0.30)
    parser.add_argument("--quadratic-target-m", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=92000)
    args = parser.parse_args()

    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output}")
    if args.contexts_per_gamma < 1 or args.lookback_steps < 1 or args.K < 1:
        parser.error("contexts, lookback steps, and K must be positive")
    if args.flow_nfe < 1 or args.flow_base_std < 0.0:
        parser.error("flow NFE must be positive and base std nonnegative")
    weights = [*args.exponential_weights, *args.quadratic_weights]
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        parser.error("weights must be finite and nonnegative")
    if (
        args.exponential_temperature <= 0.0
        or args.exponential_target_m < 0.0
        or args.quadratic_target_m < 0.0
    ):
        parser.error("temperature must be positive and targets nonnegative")

    device = torch.device(args.device)
    config = load_config(args.task_config)
    policy = load_lab_expansion_policy(
        args.pretrain_dir / "pretrained.pt"
    ).to(device).eval()
    policy.policy.nfe = int(args.flow_nfe)
    policy.policy.flow.nfe = int(args.flow_nfe)
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        device=device,
        tight_corridor=True,
        execution_obstacle_cost="none",
    )
    source = torch.load(
        args.collision_rollouts, map_location="cpu", weights_only=False,
    )
    source_rows = source["rows"]

    selected_sources = []
    for gamma in args.gammas:
        collisions = [
            row for row in source_rows
            if float(row["gamma"]) == float(gamma)
            and row["status"] == "COLLISION"
        ]
        selected_sources.extend(collisions[:args.contexts_per_gamma])
    expected = len(args.gammas) * args.contexts_per_gamma
    if len(selected_sources) != expected:
        raise RuntimeError(
            f"needed {expected} collision rows, found {len(selected_sources)}"
        )

    arms = [("min_cost", "native_cost", 0.0)]
    arms.extend(
        (f"exponential_cost_w{weight:g}", "clearance_exp_feature", float(weight))
        for weight in args.exponential_weights
    )
    arms.extend(
        (f"quadratic_cost_w{weight:g}", "clearance_quadratic_feature", float(weight))
        for weight in args.quadratic_weights
    )
    probes = []
    with torch.no_grad():
        for probe_index, source_row in enumerate(selected_sources):
            states = np.asarray(source_row["states"], np.float32)
            controls = np.asarray(source_row["controls"], np.float32)
            applied = np.asarray(source_row["applied_controls"], np.float32)
            context_step = max(0, len(controls) - int(args.lookback_steps))
            state = task.reset(float(source_row["gamma"]), probe_index, args.seed)
            state["x"] = states[context_step].copy()
            state["steps"] = int(context_step)
            if context_step:
                state["previous_raw"] = controls[context_step - 1].copy()
                state["previous_applied"] = applied[context_step - 1].copy()
            context = task.context(state, float(source_row["gamma"]))
            generator = torch.Generator(device=device).manual_seed(
                int(args.seed) + 1009 * probe_index
            )
            candidates = policy.sample(
                context, int(args.K), generator,
                base_std=float(args.flow_base_std),
            )
            verdicts = task.verify(context, candidates, float(source_row["gamma"]))
            eligible = [
                index for index, verdict in enumerate(verdicts)
                if not verdict.error and verdict.valid and verdict.progress_eligible
            ]

            source_dense = np.asarray(source_row["dense_steps"], np.float32)
            collision_clearance = task.env.obstacle_clearance(source_dense[-1])
            collision_index = int(np.argmin(collision_clearance))
            metrics = {}
            for index in eligible:
                plan = candidates[index].detach().cpu().numpy().reshape(-1, 3)
                planned_states, _, dense_steps = task._rollout_plan(
                    state["x"], state["previous_applied"], plan,
                )
                knot_clearance = task.env.obstacle_clearance(
                    planned_states[1:, :3]
                )
                dense_clearance = task.env.obstacle_clearance(
                    dense_steps.reshape(-1, 3)
                )
                shortfall = np.maximum(
                    float(args.quadratic_target_m) - knot_clearance, 0.0,
                )
                metrics[index] = {
                    "candidate": int(index),
                    "native_cost": float(verdicts[index].execution_cost),
                    "clearance_exp_feature": float(np.exp(
                        (float(args.exponential_target_m) - knot_clearance)
                        / float(args.exponential_temperature)
                    ).mean()),
                    "clearance_quadratic_feature": float(np.square(shortfall).mean()),
                    "predicted_min_knot_clearance_m": float(knot_clearance.min()),
                    "predicted_min_dense_clearance_m": float(dense_clearance.min()),
                    "terminal_goal_distance_m": float(np.linalg.norm(
                        planned_states[-1, :3] - task.env.goal
                    )),
                    "full_h_margin": float(verdicts[index].margin),
                    "first_step_margin": float(verdicts[index].step_margin),
                }

            selections = {}
            native_best = None
            crossovers = {"exponential": None, "quadratic": None}
            if eligible:
                native_best = min(eligible, key=lambda index: metrics[index]["native_cost"])
                for arm_name, feature, weight in arms:
                    chosen = (
                        native_best if arm_name == "min_cost" else min(
                            eligible,
                            key=lambda index: (
                                metrics[index]["native_cost"]
                                + weight * metrics[index][feature]
                            ),
                        )
                    )
                    choice = dict(metrics[chosen])
                    choice.update({
                        "weight": float(weight),
                        "total_cost": float(
                            metrics[chosen]["native_cost"]
                            + (0.0 if arm_name == "min_cost" else weight * metrics[chosen][feature])
                        ),
                        "changed_from_native": chosen != native_best,
                    })
                    selections[arm_name] = choice

                base = metrics[native_best]
                for label, feature in (
                    ("exponential", "clearance_exp_feature"),
                    ("quadratic", "clearance_quadratic_feature"),
                ):
                    candidate_crossovers = []
                    for index in eligible:
                        other = metrics[index]
                        safer = (
                            other["predicted_min_knot_clearance_m"]
                            > base["predicted_min_knot_clearance_m"] + 0.01
                        )
                        denominator = base[feature] - other[feature]
                        numerator = other["native_cost"] - base["native_cost"]
                        if safer and denominator > 0.0 and numerator >= 0.0:
                            candidate_crossovers.append(numerator / denominator)
                    if candidate_crossovers:
                        crossovers[label] = float(min(candidate_crossovers))

            native_costs = [metrics[index]["native_cost"] for index in eligible]
            record = {
                "probe": int(probe_index),
                "gamma": float(source_row["gamma"]),
                "source_episode": int(source_row["episode"]),
                "source_collision_step": int(len(controls)),
                "context_step": int(context_step),
                "source_collision_point": source_dense[-1, collision_index].tolist(),
                "source_collision_clearance_m": float(collision_clearance[collision_index]),
                "eligible": int(len(eligible)),
                "candidate_count": int(args.K),
                "native_cost_min": float(min(native_costs)) if native_costs else None,
                "native_cost_median": float(np.median(native_costs)) if native_costs else None,
                "native_cost_max": float(max(native_costs)) if native_costs else None,
                "safer_candidate_crossover_weight": crossovers,
                "selections": selections,
            }
            probes.append(record)
            print(
                f"[{probe_index + 1:02d}/{len(selected_sources)}] "
                f"gamma={source_row['gamma']:g} source_ep={source_row['episode']} "
                f"eligible={len(eligible)}/{args.K} "
                f"cross_exp={crossovers['exponential']} "
                f"cross_quad={crossovers['quadratic']}",
                flush=True,
            )

    summary = {}
    for arm_name, _, weight in arms:
        rows = [probe["selections"].get(arm_name) for probe in probes]
        rows = [row for row in rows if row is not None]
        summary[arm_name] = {
            "weight": float(weight),
            "contexts_with_eligible_candidate": len(rows),
            "selection_change_rate_from_native": (
                float(np.mean([row["changed_from_native"] for row in rows]))
                if rows else None
            ),
            "mean_native_cost": _mean(rows, "native_cost"),
            "mean_total_cost": _mean(rows, "total_cost"),
            "mean_predicted_min_knot_clearance_m": _mean(
                rows, "predicted_min_knot_clearance_m"
            ),
            "mean_predicted_min_dense_clearance_m": _mean(
                rows, "predicted_min_dense_clearance_m"
            ),
            "mean_terminal_goal_distance_m": _mean(
                rows, "terminal_goal_distance_m"
            ),
            "mean_first_step_margin": _mean(rows, "first_step_margin"),
        }

    payload = {
        "contract": {
            "source": str(args.collision_rollouts.resolve()),
            "pretrain": str(args.pretrain_dir.resolve()),
            "task_config": str(args.task_config.resolve()),
            "paired_candidate_sets": True,
            "contexts": f"{args.lookback_steps} steps before observed PRE sphere collision",
            "verifier_and_progress_gate_unchanged": True,
            "flow_nfe": int(args.flow_nfe),
            "flow_base_std": float(args.flow_base_std),
            "K": int(args.K),
            "exponential_formula": (
                "weight * mean_h exp((target_m-clearance_h)/temperature)"
            ),
            "quadratic_formula": "weight * mean_h max(target_m-clearance_h,0)^2",
            "exponential_target_m": float(args.exponential_target_m),
            "exponential_temperature": float(args.exponential_temperature),
            "quadratic_target_m": float(args.quadratic_target_m),
        },
        "candidate_cost_scale": {
            "mean_native_min": _mean(probes, "native_cost_min"),
            "mean_native_median": _mean(probes, "native_cost_median"),
            "mean_native_max": _mean(probes, "native_cost_max"),
            "safer_exponential_crossover_weight": _quantiles([
                probe["safer_candidate_crossover_weight"]["exponential"]
                for probe in probes
                if probe["safer_candidate_crossover_weight"]["exponential"] is not None
            ]),
            "safer_quadratic_crossover_weight": _quantiles([
                probe["safer_candidate_crossover_weight"]["quadratic"]
                for probe in probes
                if probe["safer_candidate_crossover_weight"]["quadratic"] is not None
            ]),
        },
        "source_collision_summary": {
            "count": len(probes),
            "mean_terminal_clearance_m": float(np.mean([
                probe["source_collision_clearance_m"] for probe in probes
            ])),
            "mean_collision_point_xyz": np.mean([
                probe["source_collision_point"] for probe in probes
            ], axis=0).tolist(),
        },
        "summary": summary,
        "probes": probes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "execution_cost_probe.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "candidate_cost_scale": payload["candidate_cost_scale"],
        "summary": summary,
    }, indent=2, allow_nan=False), flush=True)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
