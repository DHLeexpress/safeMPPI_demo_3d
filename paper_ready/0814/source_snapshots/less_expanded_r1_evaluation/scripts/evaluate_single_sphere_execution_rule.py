#!/usr/bin/env python3
"""Paired pure-sampling audit of symmetric single-sphere execution costs."""
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


MODE_NAMES = {0: "below", 1: "above", 2: "left", 3: "right", None: "none"}


def _summary(rows: list[dict]) -> dict:
    statuses = ("SUCCESS", "COLLISION", "OOB", "NVP", "TIMEOUT")
    counts = {status: sum(row["status"] == status for row in rows) for status in statuses}
    route_counts = {
        mode: sum(row["status"] == "SUCCESS" and row["mode"] == mode for row in rows)
        for mode in ("below", "above", "left", "right")
    }
    clearances = [
        row["min_clearance_m"] for row in rows
        if row["min_clearance_m"] is not None
    ]
    success_times = [
        row["time_to_goal_s"] for row in rows
        if row["time_to_goal_s"] is not None
    ]
    success_path_ratios = [
        row["path_length_ratio"] for row in rows
        if row["status"] == "SUCCESS"
    ]
    successes = max(counts["SUCCESS"], 1)
    probabilities = np.asarray(list(route_counts.values()), np.float64) / successes
    nonzero = probabilities[probabilities > 0.0]
    return {
        "episodes": len(rows),
        "status_counts": counts,
        "SR": counts["SUCCESS"] / len(rows),
        "CR": counts["COLLISION"] / len(rows),
        "OOB": counts["OOB"] / len(rows),
        "NVP": counts["NVP"] / len(rows),
        "timeout": counts["TIMEOUT"] / len(rows),
        "route_counts": route_counts,
        "route_coverage": sum(count > 0 for count in route_counts.values()),
        "route_entropy_nats": (
            float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_min_clearance_m": (
            float(np.mean(clearances)) if clearances else None
        ),
        "mean_success_time_to_goal_s": (
            float(np.mean(success_times)) if success_times else None
        ),
        "median_success_time_to_goal_s": (
            float(np.median(success_times)) if success_times else None
        ),
        "mean_success_path_length_ratio": (
            float(np.mean(success_path_ratios))
            if success_path_ratios else None
        ),
    }


def _arm_specs(args: argparse.Namespace) -> list[dict]:
    if args.arm:
        arms = []
        for raw in args.arm:
            fields = [field.strip() for field in raw.split(",")]
            if len(fields) != 5:
                raise ValueError(
                    "--arm must be name,rule,weight,target_m,temperature "
                    "(use '-' for an unused target or temperature)"
                )
            name, rule, raw_weight, raw_target, raw_temperature = fields
            if rule not in {"none", "exponential", "quadratic"}:
                raise ValueError("--arm rule must be none, exponential, or quadratic")
            arms.append({
                "name": name,
                "execution_obstacle_cost": rule,
                "weight": float(raw_weight),
                "target_m": None if raw_target == "-" else float(raw_target),
                "temperature": (
                    None if raw_temperature == "-" else float(raw_temperature)
                ),
            })
        return arms
    arms = [{
        "name": "min_cost",
        "execution_obstacle_cost": "none",
        "weight": 0.0,
        "target_m": None,
        "temperature": None,
    }]
    arms.extend({
        "name": f"exponential_cost_w{weight:g}",
        "execution_obstacle_cost": "exponential",
        "weight": float(weight),
        "target_m": float(args.exponential_target_m),
        "temperature": float(args.exponential_temperature),
    } for weight in args.exponential_weights)
    arms.extend({
        "name": f"quadratic_cost_w{weight:g}",
        "execution_obstacle_cost": "quadratic",
        "weight": float(weight),
        "target_m": float(args.quadratic_target_m),
        "temperature": None,
    } for weight in args.quadratic_weights)
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=(0.1, 0.3, 0.5, 1.0),
    )
    parser.add_argument("--episodes-per-gamma", type=int, default=2)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--flow-base-std", type=float, default=1.0)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help=(
            "explicit arm as name,rule,weight,target_m,temperature; rule is "
            "none/exponential/quadratic and '-' marks an unused field. If "
            "provided, replaces the default min/exp/quadratic grid"
        ),
    )
    parser.add_argument(
        "--exponential-weights", type=float, nargs="*", default=(30.0, 100.0, 300.0),
    )
    parser.add_argument(
        "--quadratic-weights", type=float, nargs="*", default=(1000.0, 3000.0, 10000.0),
    )
    parser.add_argument("--exponential-temperature", type=float, default=0.15)
    parser.add_argument("--exponential-target-m", type=float, default=0.30)
    parser.add_argument("--quadratic-target-m", type=float, default=0.30)
    parser.add_argument("--capture-branches", action="store_true")
    parser.add_argument("--capture-gamma", type=float, default=0.5)
    parser.add_argument(
        "--capture-episode", type=int, action="append", default=None,
    )
    parser.add_argument("--branch-capture-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=93000)
    args = parser.parse_args()

    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output}")
    if args.episodes_per_gamma < 1 or args.K < 1 or args.flow_nfe < 1:
        parser.error("episodes, K, and flow NFE must be positive")
    capture_episodes = sorted(set(args.capture_episode or [0]))
    if any(episode < 0 for episode in capture_episodes) or args.branch_capture_stride < 1:
        parser.error("capture episode must be nonnegative and stride positive")
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

    rows = []
    try:
        arms = _arm_specs(args)
    except ValueError as error:
        parser.error(str(error))
    for arm in arms:
        if (
            not np.isfinite(arm["weight"])
            or arm["weight"] < 0.0
            or (
                arm["execution_obstacle_cost"] != "none"
                and (arm["target_m"] is None or arm["target_m"] < 0.0)
            )
            or (
                arm["execution_obstacle_cost"] == "exponential"
                and (arm["temperature"] is None or arm["temperature"] <= 0.0)
            )
        ):
            parser.error(f"invalid explicit arm: {arm}")
    with torch.no_grad():
        for arm in arms:
            task = LabFlowExpansionTask(
                config,
                context_schema=policy.context_schema,
                device=device,
                tight_corridor=True,
                execution_obstacle_cost=arm["execution_obstacle_cost"],
                execution_clearance_exp_weight=(
                    arm["weight"]
                    if arm["execution_obstacle_cost"] == "exponential" else 0.0
                ),
                execution_clearance_exp_temperature=float(
                    arm["temperature"]
                    if arm["execution_obstacle_cost"] == "exponential"
                    else args.exponential_temperature
                ),
                execution_clearance_target_m=float(
                    arm["target_m"]
                    if arm["execution_obstacle_cost"] == "exponential"
                    else args.exponential_target_m
                ),
                execution_clearance_quadratic_weight=(
                    arm["weight"]
                    if arm["execution_obstacle_cost"] == "quadratic" else 0.0
                ),
                execution_clearance_quadratic_target_m=float(
                    arm["target_m"]
                    if arm["execution_obstacle_cost"] == "quadratic"
                    else args.quadratic_target_m
                ),
            )
            for gamma_index, gamma in enumerate(args.gammas):
                for episode in range(args.episodes_per_gamma):
                    rollout_seed = (
                        int(args.seed) + 10_007 * gamma_index + 37 * episode
                    )
                    generator = torch.Generator(device=device).manual_seed(
                        rollout_seed
                    )
                    state = task.reset(float(gamma), episode, rollout_seed)
                    executed = []
                    path = [state["x"][:3].copy()]
                    min_clearance = float("inf")
                    status = "TIMEOUT"
                    branch_frames = []
                    for _ in range(config.taskspace.max_steps):
                        context = task.context(state, float(gamma))
                        candidates = policy.sample(
                            context,
                            int(args.K),
                            generator,
                            base_std=float(args.flow_base_std),
                        )
                        verdicts = task.verify(context, candidates, float(gamma))
                        eligible = [
                            index for index, verdict in enumerate(verdicts)
                            if (
                                not verdict.error
                                and verdict.valid
                                and verdict.progress_eligible
                            )
                        ]
                        if not eligible:
                            status = "NVP"
                            break
                        chosen = min(
                            eligible,
                            key=lambda index: verdicts[index].execution_cost,
                        )
                        if (
                            args.capture_branches
                            and float(gamma) == float(args.capture_gamma)
                            and episode in capture_episodes
                            and int(state["steps"]) % int(args.branch_capture_stride) == 0
                        ):
                            proposed_paths = []
                            for candidate in candidates:
                                plan = candidate.detach().cpu().numpy().reshape(-1, 3)
                                planned_states, _, _ = task._rollout_plan(
                                    state["x"], state["previous_applied"], plan,
                                )
                                proposed_paths.append(planned_states[:, :3])
                            branch_frames.append({
                                "control_step": int(state["steps"]),
                                "origin": state["x"][:3].copy(),
                                "proposed_paths": np.asarray(
                                    proposed_paths, np.float32,
                                ),
                                "eligible": np.asarray(eligible, np.int32),
                                "chosen": int(chosen),
                                "execution_costs": np.asarray([
                                    verdict.execution_cost for verdict in verdicts
                                ], np.float32),
                            })
                        # Receding-horizon contract: sample/verify H=10, execute
                        # only action 0, then form a new context and repeat.
                        state = task.advance(state, candidates[chosen])
                        executed.append(state)
                        path.append(state["x"][:3].copy())
                        step_clearance = task.env.obstacle_clearance(
                            np.asarray(state["last_dense"], np.float32)
                        )
                        if np.isfinite(step_clearance).any():
                            min_clearance = min(
                                min_clearance, float(step_clearance.min())
                            )
                        terminal = task.terminal(state)
                        if terminal is not None:
                            status = terminal
                            break
                    mode_id = task.successful_trajectory_mode(executed)
                    path_array = np.asarray(path, np.float32)
                    segment_lengths = np.linalg.norm(
                        np.diff(path_array, axis=0), axis=1,
                    )
                    direct_distance = float(np.linalg.norm(
                        path_array[-1] - path_array[0]
                    ))
                    row = {
                        "arm": arm["name"],
                        "rule": arm["execution_obstacle_cost"],
                        "weight": float(arm["weight"]),
                        "gamma": float(gamma),
                        "episode": int(episode),
                        "rollout_seed": int(rollout_seed),
                        "status": status,
                        "mode": MODE_NAMES[mode_id],
                        "steps": int(state["steps"]),
                        "time_to_goal_s": (
                            float(state["steps"] * config.safemppi.dt)
                            if status == "SUCCESS" else None
                        ),
                        "path_length_m": float(segment_lengths.sum()),
                        "path_length_ratio": float(
                            segment_lengths.sum() / max(direct_distance, 1.0e-9)
                        ),
                        "min_clearance_m": (
                            float(min_clearance)
                            if np.isfinite(min_clearance) else None
                        ),
                        "path": path_array,
                        "branch_frames": branch_frames,
                    }
                    rows.append(row)
                    print(
                        f"arm={arm['name']} gamma={gamma:g} episode={episode} "
                        f"status={status} mode={row['mode']} "
                        f"steps={row['steps']} "
                        f"min_clearance={row['min_clearance_m']}",
                        flush=True,
                    )

    summary = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm["name"]]
        summary[arm["name"]] = {
            "spec": arm,
            "pooled": _summary(arm_rows),
            "per_gamma": {
                f"{gamma:g}": _summary([
                    row for row in arm_rows if row["gamma"] == float(gamma)
                ])
                for gamma in args.gammas
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": {
            "kind": "selection-only execution audit; no GP/acquisition/update",
            "paired_rollout_seeds_across_arms": True,
            "beta_tilting": False,
            "B_equals_K_all_verified": int(args.K),
            "receding_horizon": "sample H=10; execute action 0; repeat",
            "flow_nfe": int(args.flow_nfe),
            "flow_base_std": float(args.flow_base_std),
            "exponential_formula": (
                "native + w*mean_h exp((target-clearance_h)/temperature)"
            ),
            "quadratic_formula": (
                "native + w*mean_h max(target-clearance_h,0)^2"
            ),
            "exponential_target_m": float(args.exponential_target_m),
            "exponential_temperature": float(args.exponential_temperature),
            "quadratic_target_m": float(args.quadratic_target_m),
            "GP_buffer": "empty/unused",
            "branch_capture": {
                "enabled": bool(args.capture_branches),
                "gamma": float(args.capture_gamma),
                "episode": int(capture_episodes[0]),
                "episodes": capture_episodes,
                "stride": int(args.branch_capture_stride),
            },
        },
        "summary": summary,
        "rows": rows,
    }
    torch.save(payload, args.output / "execution_rule_rollouts.pt")
    slim = [
        {
            key: value for key, value in row.items()
            if key not in {"path", "branch_frames"}
        }
        for row in rows
    ]
    (args.output / "execution_rule_rollouts.json").write_text(
        json.dumps({
            "contract": payload["contract"],
            "summary": summary,
            "rows": slim,
        }, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
