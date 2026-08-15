#!/usr/bin/env python3
"""CPU-only fixed-context audit of all-K versus GP-selected-B route support.

This script never advances or mutates an expansion run.  It loads a committed
checkpoint and GP evidence, resamples deterministic K banks at decision-zone
states retained in an earlier immutable event archive, reverifies every K
candidate, and compares it with the B candidates selected by the exact RBF-GP
acquisition rule.  The result is a counterfactual fixed-context audit, not a
claim about candidates currently held in a live remote process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_mppi.ball_flow_theta import (  # noqa: E402
    theta_name,
    trajectory_crossing_theta,
)
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.expansion import (  # noqa: E402
    RBFPosterior,
    _embed_records,
    _sliding_success_gp_rows,
)
from safe_mppi.lab_flow_expansion import (  # noqa: E402
    LabExpansionPolicyAdapter,
    LabFlowExpansionTask,
)
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402


MODES = ("below", "above", "left", "right", "none")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_for_contract(arm: Path) -> Path:
    candidates = [arm / "manifest.json", *sorted(
        arm.glob("manifest_before_resume_round_*.json"), reverse=True,
    )]
    for path in candidates:
        if path.is_file() and "lab_execution_cost" in json.loads(path.read_text()):
            return path
    raise FileNotFoundError("no manifest with a lab execution contract")


def _task_from_manifest(config, manifest: dict) -> LabFlowExpansionTask:
    conditioning = manifest["lab_conditioning"]
    cost = manifest["lab_execution_cost"]
    verifier = manifest["lab_verifier"]
    obstacle = cost["execution_clearance_exponential"]
    quadratic = cost["execution_clearance_quadratic"]
    taskspace = cost["execution_taskspace_quadratic"]
    goal_wall = cost["execution_goal_side_wall_quadratic"]
    goal_box = cost["execution_goal_box_exponential"]
    cylinder = cost["execution_axis_cylinder_quadratic"]
    control = cost["execution_control"]
    terminal = cost["execution_terminal_goal"]
    braking = cost["execution_goal_braking"]
    stopping = verifier["taskspace_stopping_backup"]
    rule = str(cost["execution_rule"])
    obstacle_kind = {
        "min_cost": "none",
        "exponential_cost": "exponential",
        "quadratic_cost": "quadratic",
    }[rule]
    return LabFlowExpansionTask(
        config,
        context_schema=str(conditioning["context_schema"]),
        device="cpu",
        tight_corridor=bool(verifier["tight_corridor_flag"]),
        verifier_mode=str(verifier["variant"]),
        verifier_solver=str(verifier["face_solver"]),
        execution_obstacle_cost=obstacle_kind,
        execution_clearance_exp_weight=float(obstacle["effective_weight"]),
        execution_clearance_exp_temperature=float(obstacle["temperature"]),
        execution_clearance_target_m=float(obstacle["target_m"]),
        execution_clearance_quadratic_weight=float(quadratic["effective_weight"]),
        execution_clearance_quadratic_target_m=float(quadratic["target_m"]),
        execution_control_weight=float(control["effective_weight"]),
        execution_terminal_goal_weight=float(terminal["effective_weight"]),
        execution_taskspace_quadratic_weight=float(taskspace["weight"]),
        execution_taskspace_quadratic_target_m=float(taskspace["target_m"]),
        execution_goal_side_wall_quadratic_weight=float(goal_wall["weight"]),
        execution_goal_side_wall_target_m=float(goal_wall["target_m"]),
        execution_goal_box_exp_weight=float(goal_box["weight"]),
        execution_goal_box_half_extent_m=float(goal_box["half_extent_m"]),
        execution_goal_box_exp_temperature_m=float(goal_box["temperature_m"]),
        execution_axis_cylinder_quadratic_weight=float(cylinder["weight"]),
        execution_axis_cylinder_radius_m=float(cylinder["radius_m"]),
        execution_axis_cylinder_finite_segment=bool(cylinder["finite_segment"]),
        execution_goal_braking_weight=float(braking["weight"]),
        execution_goal_braking_distance_m=float(braking["distance_m"]),
        execution_goal_braking_temperature_m=float(braking["temperature_m"]),
        verifier_full_h_taskspace=bool(verifier["unexecuted_tail_taskspace_gate"]),
        verifier_stopping_margin_m=(
            float(stopping["face_margin_m"])
            if bool(stopping["enabled"]) else None
        ),
    )


def _episode_groups(events: list[dict]) -> dict[tuple[float, int], list[dict]]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for event in events:
        grouped[(float(event["gamma"]), int(event["episode"]))].append(event)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def _trace_mode(task: LabFlowExpansionTask, rows: list[dict]) -> str:
    path = np.asarray(
        [row["robot"][:3] for row in rows] + [rows[-1]["robot_after"][:3]],
        np.float64,
    )
    return theta_name(trajectory_crossing_theta(task.env, path))


def select_decision_contexts(
    task: LabFlowExpansionTask,
    events: list[dict],
    *,
    contexts_per_mode_gamma: int,
    axial_before_ball_m: float,
) -> list[dict[str, Any]]:
    """Choose deterministic source-mode-balanced states near a fixed axial plane."""
    grouped = _episode_groups(events)
    cells: dict[tuple[float, str], list[tuple[int, list[dict]]]] = defaultdict(list)
    for (gamma, episode), rows in grouped.items():
        mode = _trace_mode(task, rows)
        cells[(gamma, mode)].append((episode, rows))
    selected = []
    center = np.asarray(task.env.spheres[0, :3], np.float64)
    target_axial = -float(axial_before_ball_m)
    for (gamma, mode), trajectories in sorted(cells.items()):
        if mode not in MODES[:-1]:
            continue
        for episode, rows in sorted(trajectories)[:contexts_per_mode_gamma]:
            event = min(
                rows,
                key=lambda row: abs(
                    float((np.asarray(row["robot"][:3]) - center) @ task.forward)
                    - target_axial
                ),
            )
            prefix = np.asarray([
                row["robot"][:3]
                for row in rows if int(row["step"]) <= int(event["step"])
            ], np.float64)
            selected.append({
                "gamma": gamma,
                "source_mode": mode,
                "episode": episode,
                "event": event,
                "prefix": prefix,
                "source_axial_m": float(
                    (np.asarray(event["robot"][:3]) - center) @ task.forward
                ),
            })
    return selected


def select_terminal_contexts(
    task: LabFlowExpansionTask,
    events: list[dict],
    *,
    contexts_per_mode_gamma: int,
    goal_distance_m: float,
) -> list[dict[str, Any]]:
    """Choose source-mode-balanced states near a fixed distance to goal."""
    grouped = _episode_groups(events)
    cells: dict[tuple[float, str], list[tuple[int, list[dict]]]] = defaultdict(list)
    for (gamma, episode), rows in grouped.items():
        mode = _trace_mode(task, rows)
        cells[(gamma, mode)].append((episode, rows))
    selected = []
    for (gamma, mode), trajectories in sorted(cells.items()):
        if mode not in MODES[:-1]:
            continue
        for episode, rows in sorted(trajectories)[:contexts_per_mode_gamma]:
            event = min(
                rows,
                key=lambda row: abs(
                    float(np.linalg.norm(
                        np.asarray(row["robot"][:3]) - task.env.goal
                    )) - float(goal_distance_m)
                ),
            )
            prefix = np.asarray([
                row["robot"][:3]
                for row in rows if int(row["step"]) <= int(event["step"])
            ], np.float64)
            selected.append({
                "gamma": gamma,
                "source_mode": mode,
                "episode": episode,
                "event": event,
                "prefix": prefix,
                "source_goal_distance_m": float(np.linalg.norm(
                    np.asarray(event["robot"][:3]) - task.env.goal
                )),
            })
    return selected


def _candidate_closest_plane_mode(
    task: LabFlowExpansionTask, positions: np.ndarray,
) -> str:
    """Classify one H10 by its governed knot nearest the ball axial plane."""
    positions = np.asarray(positions, np.float64).reshape(-1, 3)
    center = np.asarray(task.env.spheres[0, :3], np.float64)
    relative = positions - center
    index = int(np.argmin(np.abs(relative @ task.forward)))
    transverse = relative[index] - float(
        relative[index] @ task.forward
    ) * task.forward
    vertical = np.asarray([0.0, 0.0, 1.0], np.float64)
    vertical -= float(vertical @ task.forward) * task.forward
    vertical /= np.linalg.norm(vertical)
    lateral = np.cross(vertical, task.forward)
    lateral /= np.linalg.norm(lateral)
    coordinates = np.asarray([
        transverse @ lateral, transverse @ vertical,
    ])
    if float(np.linalg.norm(coordinates)) <= 1.0e-12:
        return "none"
    return theta_name(float(np.arctan2(coordinates[1], coordinates[0])))


def _mode_counts(rows: list[dict], *, selected_only: bool) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if selected_only and not row["selected"]:
            continue
        if row["execution_eligible"]:
            counts[row["mode"]] += 1
    return {mode: int(counts[mode]) for mode in MODES}


def _aggregate_contexts(contexts: list[dict]) -> dict:
    candidates = [row for context in contexts for row in context["candidates"]]
    k_counts = _mode_counts(candidates, selected_only=False)
    b_counts = _mode_counts(candidates, selected_only=True)
    return {
        "contexts": len(contexts),
        "all_K_execution_eligible_modes": k_counts,
        "selected_B_execution_eligible_modes": b_counts,
        "all_K_horizon_reach_modes": {
            mode: sum(
                row["execution_eligible"] and row["horizon_reach"]
                and row["mode"] == mode for row in candidates
            ) for mode in MODES
        },
        "selected_B_horizon_reach_modes": {
            mode: sum(
                row["selected"] and row["execution_eligible"]
                and row["horizon_reach"] and row["mode"] == mode
                for row in candidates
            ) for mode in MODES
        },
        "all_K_one_step_reach_modes": {
            mode: sum(
                row["execution_eligible"] and row["one_step_reach"]
                and row["mode"] == mode for row in candidates
            ) for mode in MODES
        },
        "selected_B_one_step_reach_modes": {
            mode: sum(
                row["selected"] and row["execution_eligible"]
                and row["one_step_reach"] and row["mode"] == mode
                for row in candidates
            ) for mode in MODES
        },
        "contexts_K_has_right_B_misses_right": sum(
            any(row["execution_eligible"] and row["mode"] == "right"
                for row in context["candidates"])
            and not any(row["selected"] and row["execution_eligible"]
                        and row["mode"] == "right"
                        for row in context["candidates"])
            for context in contexts
        ),
        "contexts_K_has_reach_right_B_misses_reach_right": sum(
            any(row["execution_eligible"] and row["horizon_reach"]
                and row["mode"] == "right" for row in context["candidates"])
            and not any(row["selected"] and row["execution_eligible"]
                        and row["horizon_reach"] and row["mode"] == "right"
                        for row in context["candidates"])
            for context in contexts
        ),
    }


@torch.no_grad()
def run_audit(
    *,
    arm: Path,
    pretrain_dir: Path,
    source_round: int,
    checkpoint_round: int,
    contexts_per_mode_gamma: int,
    context_region: str,
    axial_before_ball_m: float,
    terminal_goal_distance_m: float,
    seed: int,
) -> dict:
    arm = arm.resolve()
    pretrain_dir = pretrain_dir.resolve()
    manifest_path = _manifest_for_contract(arm)
    manifest = json.loads(manifest_path.read_text())
    config = load_config(arm / "task_config_resolved.json")
    task = _task_from_manifest(config, manifest)

    raw_policy = load_lab_reference_policy(pretrain_dir / "pretrained.pt")
    checkpoint_path = arm / f"checkpoint_{checkpoint_round:03d}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_policy.load_state_dict(checkpoint["model"], strict=True)
    raw_policy.flow.nfe = int(manifest["config"]["flow_nfe"])
    policy = LabExpansionPolicyAdapter(raw_policy).eval()

    state_path = arm / "resume_state_latest.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state["completed_round"]) != checkpoint_round:
        raise RuntimeError("resume state and checkpoint round do not match")
    gp_rows = _sliding_success_gp_rows(
        state["gp_evidence"],
        manifest["config"]["gammas"],
        int(manifest["config"]["gp_buffer_cap"]),
        through_round=checkpoint_round,
        selector=str(manifest["config"]["gp_sliding_row_selector"]),
    )
    gps = {}
    for gamma in map(float, manifest["config"]["gammas"]):
        rows = [row for row in gp_rows if float(row.gamma) == gamma]
        gp = RBFPosterior(
            float(manifest["rbf_lengthscale"]),
            float(manifest["config"]["gp_noise"]),
        )
        gp.set_buffer(_embed_records(policy, rows, torch.device("cpu")))
        gps[gamma] = gp

    events_path = arm / f"events_round_{source_round:03d}.pt"
    events = torch.load(events_path, map_location="cpu", weights_only=False)
    if context_region == "decision":
        contexts = select_decision_contexts(
            task,
            events,
            contexts_per_mode_gamma=contexts_per_mode_gamma,
            axial_before_ball_m=axial_before_ball_m,
        )
    elif context_region == "terminal":
        contexts = select_terminal_contexts(
            task,
            events,
            contexts_per_mode_gamma=contexts_per_mode_gamma,
            goal_distance_m=terminal_goal_distance_m,
        )
    else:  # argparse and direct-call guard
        raise ValueError(f"unknown context region: {context_region}")
    output_contexts = []
    K = int(manifest["config"]["K"])
    B = int(manifest["config"]["B"])
    beta = float(state["beta"])
    for context_index, source in enumerate(contexts):
        event = source["event"]
        compact = torch.as_tensor(event["context"], dtype=torch.float32)
        state6, previous_applied, previous_raw = task._decode_context(compact)
        rollout_state = {
            "x": state6,
            "previous_applied": previous_applied,
            "previous_raw": previous_raw,
            "steps": int(event["step"]),
            "collided": False,
            "oob": False,
        }
        full_context = task.context(rollout_state, float(source["gamma"]))
        generator = torch.Generator(device="cpu").manual_seed(seed + context_index)
        candidates, bases = policy.sample_with_base(
            full_context, K, generator, base_std=1.0,
        )
        features = policy.embed(full_context, candidates, base=bases)
        selected, _, _, sigma = gps[float(source["gamma"])].acquire_with_sigma(
            features, B, beta, generator,
        )
        selected_set = set(selected)
        verification = task.verify(
            full_context, candidates, float(source["gamma"]),
        )
        candidate_rows = []
        for index, (candidate, verdict) in enumerate(zip(candidates, verification)):
            states, _, _ = task._rollout_plan(
                state6, previous_applied, candidate.detach().cpu().numpy(),
            )
            path = np.concatenate([source["prefix"], states[1:, :3]], axis=0)
            crossing_mode = theta_name(trajectory_crossing_theta(task.env, path))
            mode = (
                _candidate_closest_plane_mode(task, states[:, :3])
                if context_region == "decision"
                else str(source["source_mode"])
            )
            execution_eligible = bool(
                verdict.valid and verdict.progress_eligible
                and (not bool(event.get("target_gate_active", False))
                     or verdict.target_eligible)
            )
            after = task.advance(rollout_state, candidate)
            candidate_rows.append({
                "index": index,
                "selected": index in selected_set,
                "sigma": float(sigma[index]),
                "mode": mode,
                "prefix_plus_plan_crossing_mode": crossing_mode,
                "valid": bool(verdict.valid),
                "progress_eligible": bool(verdict.progress_eligible),
                "execution_eligible": execution_eligible,
                "horizon_reach": bool(task.env.reached(states[-1, :3])),
                "one_step_reach": task.terminal(after) == "SUCCESS",
                "terminal_distance_m": float(np.linalg.norm(
                    states[-1, :3] - task.env.goal
                )),
            })
        output_contexts.append({
            "gamma": float(source["gamma"]),
            "source_mode": source["source_mode"],
            "source_episode": int(source["episode"]),
            "source_step": int(event["step"]),
            "source_axial_m": source.get("source_axial_m"),
            "source_goal_distance_m": source.get("source_goal_distance_m"),
            "candidates": candidate_rows,
        })

    by_gamma = {}
    for gamma in map(float, manifest["config"]["gammas"]):
        rows = [row for row in output_contexts if row["gamma"] == gamma]
        by_gamma[f"{gamma:g}"] = _aggregate_contexts(rows)
    by_gamma_source_mode = {}
    for gamma in map(float, manifest["config"]["gammas"]):
        by_gamma_source_mode[f"{gamma:g}"] = {
            mode: _aggregate_contexts([
                row for row in output_contexts
                if row["gamma"] == gamma and row["source_mode"] == mode
            ])
            for mode in MODES[:-1]
        }
    return {
        "status": "CHECKPOINT_FIXED_CONTEXT_CANDIDATE_SUPPORT_COMPLETE",
        "scope": {
            "checkpoint_round": checkpoint_round,
            "future_acquisition_round": checkpoint_round + 1,
            "source_context_round": source_round,
            "contexts_per_mode_gamma_cap": contexts_per_mode_gamma,
            "context_region": context_region,
            "axial_before_ball_m": axial_before_ball_m,
            "terminal_goal_distance_m": terminal_goal_distance_m,
            "K": K,
            "B": B,
            "beta": beta,
            "device": "cpu",
            "interpretation": (
                "counterfactual deterministic candidate banks sampled by the "
                "committed checkpoint and selected by its next-round GP; not "
                "the hidden candidate stream of a live remote continuation"
            ),
        },
        "inputs": {
            "arm": str(arm),
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "resume_state": str(state_path),
            "events": str(events_path),
            "pretrained": str((pretrain_dir / "pretrained.pt").resolve()),
            "diagnostic_script": str(Path(__file__).resolve()),
        },
        "hashes": {
            path.name: _sha256(path) for path in (
                manifest_path, checkpoint_path, state_path, events_path,
                pretrain_dir / "pretrained.pt", Path(__file__).resolve(),
            )
        },
        "gp_rows_by_gamma": {
            f"{gamma:g}": sum(float(row.gamma) == gamma for row in gp_rows)
            for gamma in map(float, manifest["config"]["gammas"])
        },
        "summary_by_gamma": by_gamma,
        "summary_by_gamma_and_source_mode": by_gamma_source_mode,
        "contexts": output_contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--source-round", type=int, default=2)
    parser.add_argument("--checkpoint-round", type=int, default=2)
    parser.add_argument("--contexts-per-mode-gamma", type=int, default=3)
    parser.add_argument(
        "--context-region", choices=("decision", "terminal"),
        default="decision",
    )
    parser.add_argument("--axial-before-ball-m", type=float, default=0.45)
    parser.add_argument("--terminal-goal-distance-m", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=93510)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.contexts_per_mode_gamma < 1:
        parser.error("--contexts-per-mode-gamma must be positive")
    if args.axial_before_ball_m <= 0.0:
        parser.error("--axial-before-ball-m must be positive")
    payload = run_audit(
        arm=args.arm,
        pretrain_dir=args.pretrain_dir,
        source_round=args.source_round,
        checkpoint_round=args.checkpoint_round,
        contexts_per_mode_gamma=args.contexts_per_mode_gamma,
        context_region=args.context_region,
        axial_before_ball_m=args.axial_before_ball_m,
        terminal_goal_distance_m=args.terminal_goal_distance_m,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for gamma, row in payload["summary_by_gamma"].items():
        print(
            f"[g{gamma}] K-right={row['all_K_execution_eligible_modes']['right']} "
            f"B-right={row['selected_B_execution_eligible_modes']['right']} "
            f"K-right/B-miss contexts={row['contexts_K_has_right_B_misses_right']}"
        )
    print(f"[output] {args.output.resolve()}")


if __name__ == "__main__":
    main()
