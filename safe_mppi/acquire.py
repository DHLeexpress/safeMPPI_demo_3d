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
from .environment import TaskEnvironment
from .visualize import make_all_figures


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _object_array(items):
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def _json_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def run_episode(env, controller, gamma, seed):
    controller.reset()
    state = env.start.copy()
    states, controls, poly_A, poly_b = [state.copy()], [], [], []
    feasible, slacks, plan_times = [], [], []
    for step in range(env.task.max_steps):
        action, info = controller.plan(state, env.goal, gamma, seed=seed * 100_000 + step)
        poly_A.append(info["A"])
        poly_b.append(info["b"])
        feasible.append(info["feasible_fraction"])
        slacks.append(info["online_one_step_slack"])
        plan_times.append(info["plan_time_s"])
        state = env.step(state, action)
        states.append(state.copy())
        controls.append(action.copy())
        if env.reached(state[:3]):
            break
    states = np.asarray(states, np.float32)
    controls = np.asarray(controls, np.float32)
    dense = env.dense_positions(states, controls)
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
    row = {
        "gamma": float(gamma), "seed": int(seed), "success": bool(success),
        "collision": collision, "taskspace_violation": taskspace_violation,
        "steps": int(len(controls)), "time_to_goal_s": time_to_goal,
        "min_clearance_m": min_clearance,
        "mean_feasible_fraction": float(np.mean(feasible)),
        "minimum_online_one_step_slack": float(np.min(slacks)),
        "mean_plan_time_ms": float(1000.0 * np.mean(plan_times)),
    }
    arrays = dict(states=states, controls=controls, poly_A=_object_array(poly_A),
                  poly_b=_object_array(poly_b), feasible_fraction=np.asarray(feasible, np.float32),
                  online_one_step_slack=np.asarray(slacks, np.float32),
                  gamma=np.float32(gamma), seed=np.int64(seed))
    return row, arrays


def aggregate_metrics(rows, gammas):
    output = []
    for gamma in gammas:
        group = [row for row in rows if abs(row["gamma"] - gamma) < 1e-9]
        clearances = [row["min_clearance_m"] for row in group if row["min_clearance_m"] is not None]
        times = [row["time_to_goal_s"] for row in group if row["time_to_goal_s"] is not None]
        output.append({
            "gamma": float(gamma), "episodes": len(group),
            "SR": sum(row["success"] for row in group) / len(group),
            "CR": sum(row["collision"] for row in group) / len(group),
            "taskspace_violation_rate": sum(row["taskspace_violation"] for row in group) / len(group),
            "avg_min_clearance_m": (float(np.mean(clearances)) if clearances else None),
            "avg_time_to_goal_s": (float(np.mean(times)) if times else None),
            "avg_plan_time_ms": float(np.mean([row["mean_plan_time_ms"] for row in group])),
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
    for gamma in config.data.gammas:
        for episode in range(config.data.episodes_per_gamma):
            seed = config.data.seed_start + episode
            row, arrays = run_episode(env, controller, gamma, seed)
            name = f"run_g{gamma:g}_s{seed}.npz"
            np.savez_compressed(output_dir / name, **arrays)
            row["file"] = name
            rows.append(row)
            print(f"gamma={gamma:g} seed={seed} success={row['success']} "
                  f"collision={row['collision']} steps={row['steps']} "
                  f"clearance={row['min_clearance_m']} time={row['time_to_goal_s']}s", flush=True)

    metrics = aggregate_metrics(rows, config.data.gammas)
    manifest = {
        "kind": "standalone mode-1 SafeMPPI actual rollouts",
        "config": "resolved_config.json", "gammas": list(config.data.gammas),
        "runs": rows, "metrics": metrics,
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
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
