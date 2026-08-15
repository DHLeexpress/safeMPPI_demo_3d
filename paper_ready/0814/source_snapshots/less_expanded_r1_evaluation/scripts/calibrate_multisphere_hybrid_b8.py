#!/usr/bin/env python3
"""Counterfactually re-rank saved multi-sphere B8 decision contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_GOVERNOR_DIM,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in raw.split(",") if token.strip())
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("grid values must be finite and nonnegative")
    return values


def _choose(costs, margins, fraction: float) -> int:
    minimum = min(costs)
    threshold = minimum + fraction * (max(costs) - minimum)
    shortlist = [index for index, value in enumerate(costs) if value <= threshold]
    return max(shortlist, key=lambda index: (margins[index], -costs[index]))


def _mean(values) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--speed-weights", default="0,25,50,100,200,400,800",
    )
    parser.add_argument("--cost-band-fractions", default="0,0.03,0.05,0.075")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error("stride and limit must be positive")
    speed_weights = _parse_grid(args.speed_weights)
    fractions = _parse_grid(args.cost_band_fractions)
    if any(value > 1.0 for value in fractions):
        parser.error("cost-band fractions must lie in [0,1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = load_config(args.task_config)
    scene_spec = sphere_scene_spec_from_config(config)
    policy = load_lab_clutter_pre2_expansion_policy(
        args.pretrain_dir / "pretrained.pt",
        verifier_suffix_dim=LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim,
    )
    task = LabClutterPre2MultiPairExpansionTask(
        config,
        context_schema=policy.context_schema,
        device="cpu",
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
        execution_obstacle_speed_weight=1.0,
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=0,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        scene_spec=scene_spec,
    )
    events = torch.load(args.events, map_location="cpu", weights_only=False)
    if not isinstance(events, list):
        raise TypeError("events artifact must contain a list")
    selected_events = events[::args.stride]
    if args.limit is not None:
        selected_events = selected_events[:args.limit]

    variants = {
        (weight, fraction): {
            "changed": 0,
            "changed_from_saved_actual": 0,
            "saved_actual_comparable": 0,
            "changed_near_060": 0,
            "near_060": 0,
            "delta_first_speed": [],
            "delta_terminal_speed": [],
            "delta_min_clearance": [],
            "delta_base_cost": [],
            "gamma_changed": {},
            "gamma_total": {},
        }
        for weight in speed_weights for fraction in fractions
    }
    usable = 0
    saved_choice_matches = 0
    reconstructed_cost_errors = []
    started = time.time()
    for event_index, event in enumerate(selected_events):
        verification = event.get("verification")
        selected = event.get("selected")
        if not verification or not selected:
            continue
        eligible = [
            local for local, result in enumerate(verification)
            if (
                not result.get("error", False)
                and result.get("valid", False)
                and result.get("progress_eligible", False)
                and result.get("target_eligible", True)
            )
        ]
        if not eligible:
            continue
        context = np.asarray(event["context"], np.float32)
        if event.get("context_compacted", False):
            suffix = context[-task.verifier_suffix_dim:]
            state6 = np.asarray(event["robot"], np.float32)
            previous_applied = suffix[:3].copy()
            previous_raw = suffix[3:6].copy()
            spheres = scene_spec.unpack(task.env, suffix[6:])
            env = task._environment(spheres)
        else:
            (
                state6,
                previous_applied,
                previous_raw,
                env,
            ) = task._decode_context(torch.from_numpy(context))
        base_costs = []
        unit_speed_costs = []
        step_margins = []
        first_speeds = []
        terminal_speeds = []
        min_clearances = []
        for local in eligible:
            candidate_index = int(selected[local])
            plan = np.asarray(event["candidates"][candidate_index], np.float32)
            states, _, _ = task._rollout_plan(state6, previous_applied, plan)
            breakdown = task.execution_cost_breakdown(
                env, states, plan, previous_raw,
            )
            base = float(verification[local]["execution_cost"])
            base_costs.append(base)
            unit_speed_costs.append(float(
                breakdown["obstacle_conditioned_speed"]
            ))
            step_margins.append(float(verification[local]["step_margin"]))
            first_speeds.append(float(np.linalg.norm(states[1, 3:6])))
            terminal_speeds.append(float(np.linalg.norm(states[-1, 3:6])))
            clearances = np.asarray(
                env.obstacle_clearance(states[1:, :3]), np.float64,
            )
            min_clearances.append(float(clearances.min()))
            reconstructed_cost_errors.append(abs(
                float(breakdown["total"])
                - float(breakdown["obstacle_conditioned_speed"])
                - base
            ))
        baseline = int(np.argmin(base_costs))
        chosen_local = int(event.get("chosen_local", -1))
        saved_actual = (
            eligible.index(chosen_local) if chosen_local in eligible else None
        )
        if chosen_local in eligible and eligible[baseline] == chosen_local:
            saved_choice_matches += 1
        usable += 1
        gamma = f"{float(event['gamma']):g}"
        near_060 = min_clearances[baseline] < 0.6
        for (weight, fraction), stats in variants.items():
            costs = [
                base + weight * unit
                for base, unit in zip(base_costs, unit_speed_costs)
            ]
            chosen = _choose(costs, step_margins, fraction)
            changed = chosen != baseline
            stats["changed"] += int(changed)
            if saved_actual is not None:
                stats["saved_actual_comparable"] += 1
                stats["changed_from_saved_actual"] += int(
                    chosen != saved_actual
                )
            stats["near_060"] += int(near_060)
            stats["changed_near_060"] += int(changed and near_060)
            stats["gamma_total"][gamma] = stats["gamma_total"].get(gamma, 0) + 1
            stats["gamma_changed"][gamma] = (
                stats["gamma_changed"].get(gamma, 0) + int(changed)
            )
            stats["delta_first_speed"].append(
                first_speeds[chosen] - first_speeds[baseline]
            )
            stats["delta_terminal_speed"].append(
                terminal_speeds[chosen] - terminal_speeds[baseline]
            )
            stats["delta_min_clearance"].append(
                min_clearances[chosen] - min_clearances[baseline]
            )
            stats["delta_base_cost"].append(
                base_costs[chosen] - base_costs[baseline]
            )
        if (event_index + 1) % 2000 == 0:
            print(
                f"[hybrid-calibration] {event_index + 1}/{len(selected_events)}",
                flush=True,
            )

    if usable < 1:
        raise RuntimeError("no usable verifier-eligible saved B8 contexts")
    rows = []
    for (weight, fraction), stats in variants.items():
        rows.append({
            "obstacle_speed_weight": weight,
            "cost_band_fraction": fraction,
            "decision_change_fraction": stats["changed"] / usable,
            "decision_change_fraction_from_saved_actual": (
                stats["changed_from_saved_actual"]
                / stats["saved_actual_comparable"]
                if stats["saved_actual_comparable"] else None
            ),
            "near_clearance_lt_060_count": stats["near_060"],
            "near_clearance_lt_060_change_fraction": (
                stats["changed_near_060"] / stats["near_060"]
                if stats["near_060"] else None
            ),
            "gamma_change_fraction": {
                gamma: stats["gamma_changed"][gamma] / total
                for gamma, total in sorted(stats["gamma_total"].items())
            },
            "mean_delta_first_step_speed_mps": _mean(
                stats["delta_first_speed"]
            ),
            "mean_delta_terminal_speed_mps": _mean(
                stats["delta_terminal_speed"]
            ),
            "mean_delta_min_predicted_clearance_m": _mean(
                stats["delta_min_clearance"]
            ),
            "mean_delta_E15_execution_cost": _mean(
                stats["delta_base_cost"]
            ),
        })
    payload = {
        "schema_version": 1,
        "status": "MULTISPHERE_SAVED_B8_HYBRID_CALIBRATION_COMPLETE",
        "scope": (
            "decision-local counterfactual on exact verifier-eligible, "
            "GP-selected B8 candidates; trajectories are not advanced"
        ),
        "events": str(args.events.resolve()),
        "events_sha256": _sha256(args.events),
        "task_config": str(args.task_config.resolve()),
        "task_config_sha256": _sha256(args.task_config),
        "source_event_count": len(events),
        "selected_event_count": len(selected_events),
        "usable_decision_count": usable,
        "saved_min_cost_choice_match_fraction": saved_choice_matches / usable,
        "max_reconstructed_base_cost_abs_error": (
            max(reconstructed_cost_errors) if reconstructed_cost_errors else None
        ),
        "reconstructed_base_cost_abs_error_quantiles": (
            {
                str(quantile): float(np.quantile(
                    reconstructed_cost_errors, quantile,
                ))
                for quantile in (0.5, 0.9, 0.99)
            }
            if reconstructed_cost_errors else None
        ),
        "hybrid": {
            "base": "R1 native + wall250 + axis5 + control0.05 + E15",
            "speed": (
                "w * mean_h sigmoid((0.6-clearance_h)/0.15) * ||v_h||^2"
            ),
            "margin": (
                "within J_min + f*(J_max-J_min), maximize first-step nominal "
                "H_P margin; this is not a verifier label"
            ),
        },
        "elapsed_seconds": time.time() - started,
        "variants": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
