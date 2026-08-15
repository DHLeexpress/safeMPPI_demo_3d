#!/usr/bin/env python3
"""Rescore saved uncertainty-selected B8 with the goal-braking cost."""
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


BRAKING_WEIGHTS = (0.0, 25.0, 50.0, 100.0)
ALL_H_AXIS_WEIGHTS = (5.0, 10.0, 20.0, 40.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _quantile(rows: list[dict], key: str, q: float) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.quantile(values, q)) if values else None


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"contexts": 0}
    original_near = [row for row in rows if row["original_brake_gate"] >= 0.5]
    original_far = [row for row in rows if row["original_brake_gate"] < 0.5]
    return {
        "contexts": len(rows),
        "contexts_when_original_gate_ge_0p5": len(original_near),
        "contexts_when_original_gate_lt_0p5": len(original_far),
        "choice_change_from_original_rate": _mean(rows, "choice_changed"),
        "choice_change_when_original_gate_ge_0p5_rate": _mean(
            original_near, "choice_changed"
        ),
        "choice_change_when_original_gate_lt_0p5_rate": _mean(
            original_far, "choice_changed"
        ),
        "selected_full_h_inside_rate": _mean(rows, "full_h_inside"),
        "terminal_forward_speed_positive_mean_mps": _mean(
            rows, "terminal_forward_speed_positive_mps"
        ),
        "terminal_forward_speed_positive_p90_mps": _quantile(
            rows, "terminal_forward_speed_positive_mps", 0.9
        ),
        "terminal_forward_speed_positive_when_original_gate_ge_0p5_mean_mps": (
            _mean(original_near, "terminal_forward_speed_positive_mps")
        ),
        "terminal_forward_speed_positive_when_original_gate_lt_0p5_mean_mps": (
            _mean(original_far, "terminal_forward_speed_positive_mps")
        ),
        "predicted_progress_mean_m": _mean(rows, "predicted_progress_m"),
        "predicted_progress_when_original_gate_ge_0p5_mean_m": _mean(
            original_near, "predicted_progress_m"
        ),
        "predicted_progress_when_original_gate_lt_0p5_mean_m": _mean(
            original_far, "predicted_progress_m"
        ),
        "horizon_axis_distance_mean_m": _mean(
            rows, "horizon_axis_distance_mean_m"
        ),
        "horizon_axis_distance_max_mean_m": _mean(
            rows, "horizon_axis_distance_max_m"
        ),
        "horizon_min_face_clearance_mean_m": _mean(
            rows, "horizon_min_face_clearance_m"
        ),
        "horizon_min_face_clearance_p10_m": _quantile(
            rows, "horizon_min_face_clearance_m", 0.1
        ),
        "horizon_near_wall_lt_0p05_rate": _mean(
            rows, "horizon_near_wall_lt_0p05"
        ),
        "terminal_nearest_wall_clearance_mean_m": _mean(
            rows, "terminal_nearest_wall_clearance_m"
        ),
        "terminal_goal_distance_mean_m": _mean(
            rows, "terminal_goal_distance_m"
        ),
        "terminal_remaining_axial_mean_m": _mean(
            rows, "terminal_remaining_axial_m"
        ),
        "selected_brake_gate_mean": _mean(rows, "brake_gate"),
    }


def _task(config) -> LabFlowExpansionTask:
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
        execution_goal_box_exp_weight=50.0,
        execution_goal_box_half_extent_m=0.20,
        execution_goal_box_exp_temperature_m=1.0,
        execution_axis_cylinder_quadratic_weight=5.0,
        execution_axis_cylinder_radius_m=1.10,
        execution_axis_cylinder_finite_segment=True,
        execution_goal_braking_weight=0.0,
        verifier_full_h_taskspace=True,
    )


def _candidate_metrics(
    task: LabFlowExpansionTask,
    state6: np.ndarray,
    previous_applied: np.ndarray,
    plan: np.ndarray,
    *,
    braking_distance_m: float,
    braking_temperature_m: float,
) -> dict:
    states, _, dense_steps = task._rollout_plan(
        state6, previous_applied, plan,
    )
    dense_positions = np.concatenate([
        states[:1, :3], dense_steps.reshape(-1, 3),
    ])
    face_clearance = np.concatenate([
        dense_positions - task.env.bounds[:, 0],
        task.env.bounds[:, 1] - dense_positions,
    ], axis=1)
    terminal = np.asarray(states[-1], np.float64)
    displacement = terminal[:3] - task.axis_origin
    axial = float(displacement @ task.forward)
    remaining = task.axis_length - axial
    gate_argument = (
        braking_distance_m - remaining
    ) / braking_temperature_m
    gate = float(1.0 / (1.0 + np.exp(-np.clip(
        gate_argument, -60.0, 60.0,
    ))))
    forward_speed = max(float(terminal[3:6] @ task.forward), 0.0)
    closest_axial = float(np.clip(axial, 0.0, task.axis_length))
    radial = displacement - closest_axial * task.forward
    radial_sq = float(np.square(np.linalg.norm(radial)))
    knot_displacements = np.asarray(states[1:, :3], np.float64) - task.axis_origin
    knot_axial = knot_displacements @ task.forward
    knot_closest = np.clip(knot_axial, 0.0, task.axis_length)
    knot_radial = knot_displacements - knot_closest[:, None] * task.forward
    knot_axis_distances = np.linalg.norm(knot_radial, axis=1)
    return {
        "braking_feature": gate * forward_speed ** 2,
        "brake_gate": gate,
        "terminal_forward_speed_positive_mps": forward_speed,
        "terminal_remaining_axial_m": remaining,
        "terminal_goal_distance_m": float(np.linalg.norm(
            terminal[:3] - task.env.goal
        )),
        "terminal_nearest_wall_clearance_m": float(
            face_clearance[-1].min()
        ),
        "horizon_min_face_clearance_m": float(face_clearance.min()),
        "horizon_near_wall_lt_0p05": bool(face_clearance.min() < 0.05),
        "full_h_inside": bool(task.env.inside_taskspace(
            dense_positions
        ).all()),
        "predicted_progress_m": float(
            np.linalg.norm(state6[:3] - task.env.goal)
            - np.linalg.norm(terminal[:3] - task.env.goal)
        ),
        "horizon_axis_distance_mean_m": float(knot_axis_distances.mean()),
        "horizon_axis_distance_max_m": float(knot_axis_distances.max()),
        "axis_terminal_r1p1_w5_feature": 5.0 * radial_sq / 1.10 ** 2,
        "axis_all_h_r1p0_feature": float(np.square(
            knot_axis_distances / 1.0
        ).mean()),
        "axis_radius_1p0_delta": 5.0 * radial_sq * (
            1.0 / 1.0 ** 2 - 1.0 / 1.10 ** 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--events-per-gamma", type=int, default=80)
    parser.add_argument("--seed", type=int, default=81150)
    parser.add_argument("--braking-distance-m", type=float, default=0.60)
    parser.add_argument("--braking-temperature-m", type=float, default=0.15)
    args = parser.parse_args()

    manifest_path = args.events_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    recipe = manifest["config"]
    expected = {
        "K": 16,
        "B": 8,
        "retry_B": 8,
        "retry_verify_all_fast_path": False,
    }
    actual = {key: recipe[key] for key in expected}
    if actual != expected:
        raise ValueError(f"saved event selection contract mismatch: {actual}")

    config = load_config(args.task_config)
    task = _task(config)
    path = args.events_dir / f"events_round_{args.round:03d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(args.seed)
    sampled: list[dict] = []
    sample_counts = {}
    available_counts = {}
    for gamma in config.data.gammas:
        eligible_events = [
            event for event in saved
            if float(event["gamma"]) == float(gamma)
            and len(event.get("selected", ())) == 8
            and event.get("chosen_local") is not None
            and len(event.get("verification", ())) == 8
        ]
        take = min(args.events_per_gamma, len(eligible_events))
        indices = rng.choice(len(eligible_events), take, replace=False)
        sampled.extend(eligible_events[int(index)] for index in indices)
        key = f"{float(gamma):g}"
        available_counts[key] = len(eligible_events)
        sample_counts[key] = take

    arms = {f"brake_w{weight:g}": [] for weight in BRAKING_WEIGHTS}
    arms["axis_radius_1p0"] = []
    for weight in ALL_H_AXIS_WEIGHTS:
        arms[f"axis_all_h_r1p0_w{weight:g}"] = []
    original_reproduction_mismatches = 0
    selected_b_candidates = 0
    selected_b_full_h_inside = 0
    verifier_eligible_candidates = 0
    verifier_eligible_full_h_inside = 0
    for event in sampled:
        context = torch.as_tensor(event["context"], dtype=torch.float32)
        state6, previous_applied, _ = task._decode_context(context)
        candidates = np.asarray(event["candidates"], np.float32)
        selected = [int(index) for index in event["selected"]]
        verdicts = event["verification"]
        metrics = []
        eligible_local = []
        for local, candidate_index in enumerate(selected):
            row = _candidate_metrics(
                task,
                state6,
                previous_applied,
                candidates[candidate_index],
                braking_distance_m=args.braking_distance_m,
                braking_temperature_m=args.braking_temperature_m,
            )
            row["base_cost"] = float(verdicts[local]["execution_cost"])
            metrics.append(row)
            selected_b_candidates += 1
            selected_b_full_h_inside += int(row["full_h_inside"])
            verdict = verdicts[local]
            if (
                not verdict["error"]
                and verdict["valid"]
                and verdict["progress_eligible"]
                and (
                    not event.get("target_gate_active", False)
                    or verdict["target_eligible"]
                )
            ):
                eligible_local.append(local)
                verifier_eligible_candidates += 1
                verifier_eligible_full_h_inside += int(row["full_h_inside"])
        if not eligible_local:
            continue
        original_local = int(event["chosen_local"])
        reproduced = min(
            eligible_local, key=lambda local: metrics[local]["base_cost"]
        )
        original_reproduction_mismatches += int(reproduced != original_local)
        original_gate = metrics[original_local]["brake_gate"]

        choices = {
            f"brake_w{weight:g}": min(
                eligible_local,
                key=lambda local, weight=weight: (
                    metrics[local]["base_cost"]
                    + weight * metrics[local]["braking_feature"]
                ),
            )
            for weight in BRAKING_WEIGHTS
        }
        choices["axis_radius_1p0"] = min(
            eligible_local,
            key=lambda local: (
                metrics[local]["base_cost"]
                + metrics[local]["axis_radius_1p0_delta"]
            ),
        )
        for weight in ALL_H_AXIS_WEIGHTS:
            choices[f"axis_all_h_r1p0_w{weight:g}"] = min(
                eligible_local,
                key=lambda local, weight=weight: (
                    metrics[local]["base_cost"]
                    - metrics[local]["axis_terminal_r1p1_w5_feature"]
                    + weight * metrics[local]["axis_all_h_r1p0_feature"]
                ),
            )
        for name, local in choices.items():
            arms[name].append({
                **metrics[local],
                "gamma": float(event["gamma"]),
                "choice_changed": bool(local != original_local),
                "original_brake_gate": float(original_gate),
            })

    summary = {}
    for name, rows in arms.items():
        summary[name] = {
            "pooled": _summarize(rows),
            "per_gamma": {
                f"{float(gamma):g}": _summarize([
                    row for row in rows
                    if float(row["gamma"]) == float(gamma)
                ])
                for gamma in config.data.gammas
            },
        }

    baseline = summary["brake_w0"]["pooled"]
    for values in summary.values():
        pooled = values["pooled"]
        pooled["delta_vs_brake_w0"] = {
            key: (
                None if pooled.get(key) is None or baseline.get(key) is None
                else pooled[key] - baseline[key]
            )
            for key in (
                "terminal_forward_speed_positive_mean_mps",
                "predicted_progress_mean_m",
                "horizon_min_face_clearance_mean_m",
                "terminal_nearest_wall_clearance_mean_m",
                "terminal_goal_distance_mean_m",
            )
        }

    payload = {
        "status": "W50_R5_SAVED_GP_B8_BRAKING_CALIBRATION_COMPLETE",
        "source_events": str(path.resolve()),
        "source_events_sha256": _sha256(path),
        "source_manifest": str(manifest_path.resolve()),
        "source_contract": (
            "Only the actual uncertainty-selected B8 indices saved from K16 "
            "are rescored; no policy sampling or experiment process is run."
        ),
        "selection_contract": actual,
        "round": int(args.round),
        "seed": int(args.seed),
        "available_contexts_by_gamma": available_counts,
        "sample_counts_by_gamma": sample_counts,
        "contexts": len(sampled),
        "selected_b_candidates": selected_b_candidates,
        "selected_b_full_h_inside_rate": (
            selected_b_full_h_inside / max(selected_b_candidates, 1)
        ),
        "verifier_eligible_candidates": verifier_eligible_candidates,
        "verifier_eligible_full_h_inside_rate": (
            verifier_eligible_full_h_inside
            / max(verifier_eligible_candidates, 1)
        ),
        "original_choice_reproduction_mismatches": (
            original_reproduction_mismatches
        ),
        "braking_distance_m": float(args.braking_distance_m),
        "braking_temperature_m": float(args.braking_temperature_m),
        "braking_weights": list(BRAKING_WEIGHTS),
        "all_h_axis_comparators": {
            "radius_m": 1.0,
            "weights": list(ALL_H_AXIS_WEIGHTS),
            "finite_segment": True,
            "aggregation": "mean_squared_normalized_distance_over_10_H_knots",
            "replacement_contract": (
                "subtract the saved terminal-only radius1.1 weight5 term, "
                "then add the stated all-H term"
            ),
        },
        "axis_comparator": {
            "weight": 5.0,
            "baseline_radius_m": 1.10,
            "comparison_radius_m": 1.0,
            "finite_segment": True,
        },
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        name: values["pooled"] for name, values in summary.items()
    }, indent=2))


if __name__ == "__main__":
    main()
