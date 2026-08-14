"""Per-gamma data acquisition, rollout storage, metrics, and figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .controller import Mode1SafeMPPI
from .environment import ReferenceGovernor, TaskEnvironment
from .visualize import make_all_figures


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _object_array(items):
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def _json_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _governed_one_step_slack(info, next_position, gamma):
    A = np.asarray(info["A"], float)
    b = np.asarray(info["b"], float)
    center = np.asarray(info["center"], float)
    margins = np.maximum(b - A @ center, 1.0e-3)
    hp = np.min((b - A @ np.asarray(next_position, float)) / margins)
    return float(hp - (1.0 - float(gamma)))


def run_episode(
    env, controller, gamma, seed, rollout_dynamics="double_integrator",
):
    controller.reset()
    governor = (
        ReferenceGovernor(env.mppi)
        if rollout_dynamics == "minhyuk_reference_governor"
        else None
    )
    state = env.start.copy()
    states, controls, executed_controls = [state.copy()], [], []
    poly_A, poly_b = [], []
    feasible, controller_slacks, governed_slacks, plan_times = [], [], [], []
    all_infeasible_steps = []
    dense_chunks = [state[:3][None].copy()]
    for step in range(env.task.max_steps):
        action, info = controller.plan(state, env.goal, gamma, seed=seed * 100_000 + step)
        poly_A.append(info["A"])
        poly_b.append(info["b"])
        feasible.append(info["feasible_fraction"])
        controller_slacks.append(info["online_one_step_slack"])
        plan_times.append(info["plan_time_s"])
        all_infeasible_steps.append(bool(info["all_infeasible"]))
        if governor is None:
            next_state = env.step(state, action)
            applied = np.asarray(action, np.float32)
            dense_step = env.dense_positions(
                np.asarray([state, next_state], np.float32),
                applied[None],
            )[1:]
        else:
            next_state, applied, dense_step = governor.step(state, action)
            if not np.allclose(
                applied, np.asarray(info["applied_action"], np.float32),
                atol=1.0e-6, rtol=0.0,
            ):
                raise RuntimeError(
                    "planner and deployment reference governor became unsynchronized"
                )
            if not np.allclose(
                next_state,
                np.asarray(info["predicted_next_state"], np.float32),
                atol=1.0e-6, rtol=0.0,
            ):
                raise RuntimeError(
                    "planner and deployment reference dynamics became unsynchronized"
                )
        governed_slacks.append(
            _governed_one_step_slack(info, next_state[:3], gamma)
        )
        state = next_state
        states.append(state.copy())
        controls.append(action.copy())
        executed_controls.append(applied.copy())
        dense_chunks.append(dense_step)
        if governor is not None:
            step_clearance = env.obstacle_clearance(dense_step)
            if np.isfinite(step_clearance).any() and np.min(step_clearance) < 0.0:
                break
            if not env.inside_taskspace(dense_step).all():
                break
        if env.reached(state[:3]):
            break
    states = np.asarray(states, np.float32)
    controls = np.asarray(controls, np.float32).reshape(-1, 3)
    executed_controls = np.asarray(executed_controls, np.float32).reshape(-1, 3)
    dense = np.concatenate(dense_chunks, axis=0)
    clearance = env.obstacle_clearance(dense)
    collision = bool(np.any(clearance < 0.0)) if np.isfinite(clearance).any() else False
    taskspace_violation = bool(np.any(~env.inside_taskspace(dense)))
    reached = bool(np.any(np.linalg.norm(states[:, :3] - env.goal[None], axis=1)
                          < env.task.reach_radius))
    success = reached and not collision and not taskspace_violation
    reach_indices = np.flatnonzero(np.linalg.norm(states[:, :3] - env.goal[None], axis=1)
                                   < env.task.reach_radius)
    time_to_goal = (float(reach_indices[0] * env.mppi.dt) if len(reach_indices) else None)
    finite_clearance = clearance[np.isfinite(clearance)]
    min_clearance = float(finite_clearance.min()) if len(finite_clearance) else None
    internal_z = states[1:-1, 2]
    control_variation = np.linalg.norm(np.diff(controls, axis=0), axis=1)
    applied_variation = np.linalg.norm(
        np.diff(executed_controls, axis=0), axis=1,
    )
    speeds = np.linalg.norm(states[:, 3:6], axis=1)
    vertical_speeds = np.abs(states[:, 5])
    max_control = float(np.max(np.abs(controls))) if len(controls) else 0.0
    max_applied_control = (
        float(np.max(np.abs(executed_controls))) if len(executed_controls) else 0.0
    )
    speed_compatible = (
        bool(float(speeds.max()) <= env.mppi.max_speed + 1.0e-6
             and float(vertical_speeds.max())
             <= env.mppi.max_vertical_speed + 1.0e-6)
        if env.mppi.max_speed is not None else None
    )
    row = {
        "gamma": float(gamma), "seed": int(seed), "success": bool(success),
        "collision": collision, "taskspace_violation": taskspace_violation,
        "steps": int(len(controls)), "time_to_goal_s": time_to_goal,
        "min_clearance_m": min_clearance,
        "mean_feasible_fraction": float(np.mean(feasible)),
        "minimum_feasible_fraction": float(np.min(feasible)),
        "all_infeasible": bool(any(all_infeasible_steps)),
        "all_infeasible_steps": int(sum(all_infeasible_steps)),
        "minimum_controller_one_step_slack": float(np.min(controller_slacks)),
        "minimum_online_one_step_slack": (
            float(np.min(governed_slacks)) if governed_slacks else None
        ),
        "mean_plan_time_ms": float(1000.0 * np.mean(plan_times)),
        "mean_control_variation_mps2": (
            float(control_variation.mean()) if len(control_variation) else 0.0
        ),
        "mean_applied_control_variation_mps2": (
            float(applied_variation.mean()) if len(applied_variation) else 0.0
        ),
        "max_abs_control_mps2": max_control,
        "max_abs_applied_control_mps2": max_applied_control,
        "peak_speed_mps": float(speeds.max()),
        "peak_vertical_speed_mps": float(vertical_speeds.max()),
        "deployment_speed_compatible": speed_compatible,
        "fraction_internal_states_below_z_bias_plane": (
            float(np.mean(internal_z < env.mppi.z_bias_plane)) if len(internal_z) else None
        ),
        "mean_internal_z_m": float(internal_z.mean()) if len(internal_z) else None,
    }
    arrays = dict(states=states, controls=controls,
                  executed_controls=executed_controls,
                  dense_positions=dense.astype(np.float32),
                  poly_A=_object_array(poly_A),
                  poly_b=_object_array(poly_b), feasible_fraction=np.asarray(feasible, np.float32),
                  controller_one_step_slack=np.asarray(
                      controller_slacks, np.float32,
                  ),
                  online_one_step_slack=np.asarray(governed_slacks, np.float32),
                  all_infeasible=np.asarray(all_infeasible_steps, bool),
                  gamma=np.float32(gamma), seed=np.int64(seed))
    return row, arrays


def aggregate_metrics(rows, gammas):
    output = []
    for gamma in gammas:
        group = [row for row in rows if abs(row["gamma"] - gamma) < 1e-9]
        clearances = [row["min_clearance_m"] for row in group if row["min_clearance_m"] is not None]
        times = [row["time_to_goal_s"] for row in group if row["time_to_goal_s"] is not None]
        below = [row["fraction_internal_states_below_z_bias_plane"] for row in group
                 if row["fraction_internal_states_below_z_bias_plane"] is not None]
        mean_z = [row["mean_internal_z_m"] for row in group
                  if row["mean_internal_z_m"] is not None]
        output.append({
            "gamma": float(gamma), "episodes": len(group),
            "SR": sum(row["success"] for row in group) / len(group),
            "CR": sum(row["collision"] for row in group) / len(group),
            "taskspace_violation_rate": sum(row["taskspace_violation"] for row in group) / len(group),
            "avg_min_clearance_m": (float(np.mean(clearances)) if clearances else None),
            "avg_time_to_goal_s": (float(np.mean(times)) if times else None),
            "avg_plan_time_ms": float(np.mean([row["mean_plan_time_ms"] for row in group])),
            "avg_control_variation_mps2": float(np.mean([
                row["mean_control_variation_mps2"] for row in group
            ])),
            "max_abs_control_mps2": float(max(
                row["max_abs_control_mps2"] for row in group
            )),
            "max_abs_applied_control_mps2": float(max(
                row["max_abs_applied_control_mps2"] for row in group
            )),
            "max_peak_speed_mps": float(max(
                row["peak_speed_mps"] for row in group
            )),
            "max_peak_vertical_speed_mps": float(max(
                row["peak_vertical_speed_mps"] for row in group
            )),
            "deployment_speed_compatible_rate": (
                float(np.mean([
                    row["deployment_speed_compatible"] for row in group
                ]))
                if group[0]["deployment_speed_compatible"] is not None else None
            ),
            "avg_fraction_internal_states_below_z_bias_plane": (
                float(np.mean(below)) if below else None
            ),
            "avg_internal_z_m": float(np.mean(mean_z)) if mean_z else None,
        })
    return output


def acquire(config_path, output_dir, device="cpu"):
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    env = TaskEnvironment(config)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        print(f"[GPU] visible cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)
    with (output_dir / "resolved_config.json").open("w") as f:
        json.dump(config.raw, f, indent=2)

    controller = Mode1SafeMPPI(config.safemppi, env, device=device)
    rows = []
    attempts = []
    for gamma in config.data.gammas:
        target = config.data.episodes_per_gamma
        max_attempts = config.data.max_attempts_per_gamma or target
        accepted = 0
        for attempt in range(max_attempts):
            seed = config.data.seed_start + attempt
            row, arrays = run_episode(
                env, controller, gamma, seed,
                rollout_dynamics=config.data.rollout_dynamics,
            )
            is_accepted = (
                config.data.acceptance == "all"
                or (
                    row["success"]
                    and row["minimum_online_one_step_slack"] is not None
                    and row["minimum_online_one_step_slack"] >= -1.0e-6
                    and row["deployment_speed_compatible"]
                )
            )
            row["accepted"] = bool(is_accepted)
            row["attempt_index"] = int(attempt)
            row["file"] = None
            if is_accepted:
                name = f"run_g{gamma:g}_s{seed}.npz"
                np.savez_compressed(output_dir / name, **arrays)
                row["file"] = name
                rows.append(row)
                accepted += 1
            attempts.append(row)
            print(
                f"gamma={gamma:g} seed={seed} accepted={is_accepted} "
                f"success={row['success']} collision={row['collision']} "
                f"all_infeasible={row['all_infeasible']} steps={row['steps']} "
                f"clearance={row['min_clearance_m']} "
                f"time={row['time_to_goal_s']}s",
                flush=True,
            )
            if accepted == target:
                break
        if accepted != target:
            failure = {
                "status": "FAILED_INSUFFICIENT_ACCEPTED_DEMOS",
                "gamma": float(gamma),
                "required": int(target),
                "accepted": int(accepted),
                "attempted": int(max_attempts),
                "attempts": attempts,
            }
            with (output_dir / "FAILED_collection.json").open("w") as f:
                json.dump(failure, f, indent=2)
            raise RuntimeError(
                f"gamma={gamma:g}: collected {accepted}/{target} accepted "
                f"demos in {max_attempts} attempts"
            )

    metrics = aggregate_metrics(rows, config.data.gammas)
    attempt_metrics = aggregate_metrics(attempts, config.data.gammas)
    manifest = {
        "kind": "standalone mode-1 SafeMPPI actual rollouts",
        "config": "resolved_config.json", "gammas": list(config.data.gammas),
        "rollout_dynamics": config.data.rollout_dynamics,
        "acceptance": config.data.acceptance,
        "runs": rows,
        "attempts": attempts,
        "metrics": metrics,
        "attempt_metrics": attempt_metrics,
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    figures = make_all_figures(manifest, env, output_dir)
    print("[outputs]", output_dir / "metrics.csv", *(str(path) for path in figures), sep="\n  ")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "default_config.json"))
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "outputs" / "default_run"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    acquire(args.config, args.output, device=args.device)


if __name__ == "__main__":
    main()
