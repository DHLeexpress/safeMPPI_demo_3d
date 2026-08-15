#!/usr/bin/env python3
"""Calibrate control/end-cap/braking execution costs on saved trajectories."""
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


def _rows_for_round(saved: dict, round_index: int) -> list[dict]:
    return list(saved.get(round_index, saved.get(str(round_index), [])))


def _finite_axis_feature(
    point: np.ndarray,
    *,
    origin: np.ndarray,
    axis: np.ndarray,
    length: float,
    radius: float,
) -> tuple[float, float]:
    displacement = np.asarray(point, np.float64) - origin
    axial = float(displacement @ axis)
    infinite_offset = displacement - axial * axis
    finite_offset = displacement - np.clip(axial, 0.0, length) * axis
    infinite = float(np.square(np.linalg.norm(infinite_offset) / radius))
    finite = float(np.square(np.linalg.norm(finite_offset) / radius))
    return infinite, finite


def _braking_feature(
    state: np.ndarray,
    *,
    origin: np.ndarray,
    axis: np.ndarray,
    length: float,
    distance: float,
    temperature: float,
) -> tuple[float, float, float]:
    displacement = np.asarray(state[:3], np.float64) - origin
    remaining = length - float(displacement @ axis)
    gate_argument = (distance - remaining) / temperature
    gate = 1.0 / (1.0 + np.exp(-np.clip(gate_argument, -60.0, 60.0)))
    forward_speed = max(float(np.asarray(state[3:6]) @ axis), 0.0)
    return float(gate * forward_speed ** 2), remaining, forward_speed


def _parse_arm(value: str) -> tuple[str, float, float, bool]:
    fields = value.split(":")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "arm must be name:control_weight:braking_weight:finite(0|1)"
        )
    name, control, braking, finite = fields
    try:
        control_value = float(control)
        braking_value = float(braking)
        finite_value = bool(int(finite))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not name or control_value < 0.0 or braking_value < 0.0:
        raise argparse.ArgumentTypeError("arm values must be nonnegative")
    return name, control_value, braking_value, finite_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--raw-trajectories", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contexts-per-gamma", type=int, default=8)
    parser.add_argument("--lookback-steps", type=int, default=7)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--axis-weight", type=float, default=5.0)
    parser.add_argument("--axis-radius-m", type=float, default=1.10)
    parser.add_argument("--braking-distance-m", type=float, default=0.60)
    parser.add_argument("--braking-temperature-m", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=95100)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        default=[],
        help="name:control_weight:braking_weight:finite(0|1)",
    )
    args = parser.parse_args()
    arms = args.arm or [
        ("native", 0.05, 0.0, False),
        ("control_low", 0.25, 0.0, False),
        ("control_mid", 1.0, 0.0, False),
        ("control_cap_brake", 0.50, 25.0, True),
    ]

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
        execution_taskspace_quadratic_weight=500.0,
        execution_taskspace_quadratic_target_m=0.15,
        execution_axis_cylinder_quadratic_weight=float(args.axis_weight),
        execution_axis_cylinder_radius_m=float(args.axis_radius_m),
    )
    saved = torch.load(
        args.raw_trajectories, map_location="cpu", weights_only=False,
    )
    rows = _rows_for_round(saved, args.round)
    sources: list[dict] = []
    for gamma in config.data.gammas:
        eligible = [
            row for row in rows
            if float(row["gamma"]) == float(gamma)
            and row["status"] in {"SUCCESS", "OOB"}
        ]
        eligible.sort(
            key=lambda row: (row["status"] != "OOB", -len(row["controls"]))
        )
        sources.extend(eligible[:args.contexts_per_gamma])
    expected = len(config.data.gammas) * args.contexts_per_gamma
    if len(sources) != expected:
        raise RuntimeError(f"needed {expected} contexts, found {len(sources)}")

    probes = []
    with torch.no_grad():
        for probe_index, source in enumerate(sources):
            states = np.asarray(source["states"], np.float32)
            controls = np.asarray(source["controls"], np.float32)
            applied = np.asarray(source["applied_controls"], np.float32)
            context_step = max(0, len(controls) - args.lookback_steps)
            context_step = min(context_step, len(controls) - 1)
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
            candidates = []
            for index, verdict in enumerate(verdicts):
                if verdict.error or not verdict.valid or not verdict.progress_eligible:
                    continue
                plan = plans[index].detach().cpu().numpy().reshape(-1, 3)
                planned_states, _, _ = task._rollout_plan(
                    state["x"], state["previous_applied"], plan,
                )
                infinite, finite = _finite_axis_feature(
                    planned_states[-1, :3],
                    origin=task.axis_origin,
                    axis=task.forward,
                    length=task.axis_length,
                    radius=args.axis_radius_m,
                )
                braking, remaining, forward_speed = _braking_feature(
                    planned_states[-1],
                    origin=task.axis_origin,
                    axis=task.forward,
                    length=task.axis_length,
                    distance=args.braking_distance_m,
                    temperature=args.braking_temperature_m,
                )
                candidates.append({
                    "candidate": int(index),
                    "base_cost": float(verdict.execution_cost),
                    "control_energy": float(np.square(plan).sum()),
                    "end_cap_delta": float(finite - infinite),
                    "braking_feature": float(braking),
                    "terminal_remaining_m": float(remaining),
                    "terminal_forward_speed_mps": float(forward_speed),
                    "terminal_goal_distance_m": float(np.linalg.norm(
                        planned_states[-1, :3] - task.env.goal
                    )),
                })
            selections = {}
            if candidates:
                native_choice = min(candidates, key=lambda item: item["base_cost"])
                for name, control_weight, braking_weight, finite in arms:
                    def score(item: dict) -> float:
                        value = item["base_cost"]
                        value += (control_weight - 0.05) * item["control_energy"]
                        if finite:
                            value += args.axis_weight * item["end_cap_delta"]
                        value += braking_weight * item["braking_feature"]
                        return float(value)

                    chosen = min(candidates, key=score)
                    selections[name] = {
                        **chosen,
                        "score": score(chosen),
                        "control_weight": control_weight,
                        "braking_weight": braking_weight,
                        "finite_segment": finite,
                        "changed_from_native": bool(
                            chosen["candidate"] != native_choice["candidate"]
                        ),
                    }
            probes.append({
                "probe": probe_index,
                "gamma": float(source["gamma"]),
                "source_status": str(source["status"]),
                "source_mode": str(source.get("mode", "none")),
                "context_step": int(context_step),
                "eligible_candidates": len(candidates),
                "selections": selections,
            })

    summary = {}
    for name, control_weight, braking_weight, finite in arms:
        selected = [row["selections"][name] for row in probes if name in row["selections"]]
        def mean(key: str) -> float:
            return float(np.mean([row[key] for row in selected]))
        summary[name] = {
            "contexts": len(selected),
            "control_weight": control_weight,
            "braking_weight": braking_weight,
            "finite_segment": finite,
            "choice_change_rate": mean("changed_from_native"),
            "selected_control_energy_mean": mean("control_energy"),
            "selected_braking_feature_mean": mean("braking_feature"),
            "selected_terminal_forward_speed_mean_mps": mean(
                "terminal_forward_speed_mps"
            ),
            "selected_terminal_remaining_mean_m": mean("terminal_remaining_m"),
            "selected_terminal_goal_distance_mean_m": mean(
                "terminal_goal_distance_m"
            ),
        }

    output = {
        "status": "EXECUTION_CONTROL_END_CAP_BRAKING_CALIBRATION_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "raw_trajectories": str(args.raw_trajectories.resolve()),
        "round": int(args.round),
        "configured_control_weight": float(config.safemppi.control_weight),
        "axis_weight": float(args.axis_weight),
        "axis_radius_m": float(args.axis_radius_m),
        "braking_distance_m": float(args.braking_distance_m),
        "braking_temperature_m": float(args.braking_temperature_m),
        "K": int(args.K),
        "summary": summary,
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
