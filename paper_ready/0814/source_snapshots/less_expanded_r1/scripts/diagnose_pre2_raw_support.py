#!/usr/bin/env python3
"""Reproduce PRE2 raw γ=0.1 rollouts and expose local unfiltered support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.ball_flow_task import PLAN_H
from safe_mppi.config import load_config
from safe_mppi.environment import ReferenceGovernor, TaskEnvironment
from safe_mppi.lab_clutter import config_for_scene
from safe_mppi.lab_reference_flow_task import (
    _raw_history_before,
    policy_context,
    raw_reference_rollout,
)
from safe_mppi.lab_visual_flow import load_lab_reference_policy
from safe_mppi.path_focused_clutter import path_focused_scene_bank


def _candidate_path(config, state, previous_applied, plan):
    governor = ReferenceGovernor(config.safemppi)
    governor.previous_applied = np.asarray(previous_applied, np.float32).copy()
    current = np.asarray(state, np.float32).copy()
    dense = []
    for command in np.asarray(plan, np.float32).reshape(PLAN_H, 3):
        current, _, step_dense = governor.step(current, command)
        dense.append(step_dense)
    return np.concatenate([state[None, :3], np.asarray(dense).reshape(-1, 3)])


def _risk(env, path):
    clearance = env.obstacle_clearance(path)
    finite = clearance[np.isfinite(clearance)]
    min_clearance = float(finite.min()) if len(finite) else None
    return {
        "collision": bool(len(finite) and min_clearance < 0.0),
        "oob": bool(not env.inside_taskspace(path).all()),
        "min_clearance_m": min_clearance,
    }


@torch.no_grad()
def _capture_case(policy, config, result, seed, device, branches):
    env = TaskEnvironment(config)
    controls = np.asarray(result["controls"], np.float32)
    states = np.asarray(result["states"], np.float32)
    applied = np.asarray(result["applied_controls"], np.float32)
    raw_path = np.concatenate([
        states[:1, :3], np.asarray(result["dense_steps"], np.float32).reshape(-1, 3),
    ])
    frame_steps = sorted(set(
        [0, max(0, len(controls) // 2), max(0, len(controls) - 1)]
    ))
    frames = []
    for step in frame_steps:
        history = _raw_history_before(controls, step)
        previous_raw = (
            np.clip(controls[step - 1], -config.safemppi.demo_u_max, config.safemppi.demo_u_max)
            if step else np.zeros(3, np.float32)
        )
        previous_applied = applied[step - 1] if step else np.zeros(3, np.float32)
        context = torch.from_numpy(policy_context(
            policy, env, states[step], 0.1, raw_history=history,
            previous_raw=previous_raw, previous_applied=previous_applied,
        )).to(device)
        # The first path is the exact unfiltered K=1 draw actually used by raw deployment.
        actual_generator = torch.Generator(device=device).manual_seed(seed * 100_000 + step)
        actual_plan = policy.sample(context, 1, actual_generator, base_std=1.0)[0]
        diagnostic_generator = torch.Generator(device=device).manual_seed(
            seed * 100_000 + step + 30_000_000
        )
        diagnostic_plans = policy.sample(
            context, branches, diagnostic_generator, base_std=1.0
        )
        plans = np.concatenate([
            actual_plan.detach().cpu().numpy()[None],
            diagnostic_plans.detach().cpu().numpy(),
        ])
        paths, risks = [], []
        for plan in plans:
            path = _candidate_path(config, states[step], previous_applied, plan)
            paths.append(path[::2])
            risks.append(_risk(env, path))
        frames.append({
            "step": int(step),
            "origin": states[step, :3],
            "paths": np.asarray(paths, np.float32),
            "risks": risks,
            "raw_draw": 0,
            "ensemble_count": int(branches),
        })
    return {
        "status": result["status"],
        "mode": result["mode"],
        "steps": int(len(controls)),
        "min_clearance_m": result["min_clearance_m"],
        "time_to_goal_s": result["time_to_goal_s"],
        "window_validity": float(result["window_validity"]),
        "path": raw_path[::2],
        "frames": frames,
    }


def _condition(policy, name, config, scenes, seed0, expected, device, branches):
    cases = []
    for episode in range(10):
        seed = seed0 + 37 * episode
        scene_config = config_for_scene(config, scenes[episode]) if scenes else config
        result = raw_reference_rollout(policy, scene_config, 0.1, seed, device=device)
        declared = expected[episode]
        if result["status"] != declared["status"]:
            raise RuntimeError(
                f"{name} ep{episode}: raw replay {result['status']} != manifest {declared['status']}"
            )
        captured = _capture_case(
            policy, scene_config, result, seed, device, branches,
        )
        captured.update({
            "episode": episode,
            "seed": seed,
            "scene_id": scenes[episode].scene_id if scenes else "single_sphere",
            "spheres": np.asarray(scene_config.obstacles.spheres, np.float32),
            "cylinders": np.asarray(scene_config.obstacles.cylinders, np.float32),
        })
        cases.append(captured)
        print(f"[{name}] ep={episode} {captured['status']} steps={captured['steps']}", flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--cylinder-config", type=Path, required=True)
    parser.add_argument("--sphere-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branches", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.branches < 1:
        parser.error("--branches must be positive")

    manifest = json.loads((args.pretrain_dir / "pretrain_manifest.json").read_text())
    policy = load_lab_reference_policy(args.pretrain_dir / "pretrained.pt").to(args.device).eval()
    cylinder_config = load_config(args.cylinder_config)
    sphere_config = load_config(args.sphere_config)
    cylinder_scenes = path_focused_scene_bank(
        cylinder_config, 10, seed=int(manifest["raw_audit_seed"]),
    )
    cylinder_expected = [
        row for row in manifest["raw_audit"] if float(row["gamma"]) == 0.1 and row["episode"] < 10
    ]
    sphere_expected = [
        row for row in manifest["ood_raw_audit"] if float(row["gamma"]) == 0.1 and row["episode"] < 10
    ]
    cylinder_expected.sort(key=lambda row: row["episode"])
    sphere_expected.sort(key=lambda row: row["episode"])
    if len(cylinder_expected) != 10 or len(sphere_expected) != 10:
        raise RuntimeError("manifest does not contain the requested first ten γ=0.1 rows")
    payload = {
        "contract": {
            "model": str((args.pretrain_dir / "pretrained.pt").resolve()),
            "gamma": 0.1,
            "raw_deployment": "K=1, base_std=1.0, no verifier; sample H10 and execute action 0",
            "diagnostic_branches": int(args.branches),
            "branch_draw": "64 extra K=1-equivalent draws at the same context; not used for control",
        },
        "bounds": np.asarray(sphere_config.taskspace.bounds, np.float32),
        "start": np.asarray(sphere_config.taskspace.start[:3], np.float32),
        "goal": np.asarray(sphere_config.taskspace.goal, np.float32),
        "sphere_ood": _condition(
            policy, "sphere_ood", sphere_config, None,
            int(manifest["ood_raw_audit_seed"]), sphere_expected, args.device, args.branches,
        ),
        "cylinder_id": _condition(
            policy, "cylinder_id", cylinder_config, cylinder_scenes,
            int(manifest["raw_audit_seed"]), cylinder_expected, args.device, args.branches,
        ),
        "manifest_summaries": {
            "sphere_ood": manifest["ood_raw_audit_summary"],
            "cylinder_id": manifest["raw_audit_summary"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[output] {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
