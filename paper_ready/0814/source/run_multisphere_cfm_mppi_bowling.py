#!/usr/bin/env python3
"""Run the PRE2-based CFM--MPPI baseline on the fixed bowling scene.

The controller uses 32 guided-flow proposals and a reduced 8x32 refinement.
No verifier, H_P gate, or progress label participates in selection.  All three
ranking stages use the native bowling SafeMPPI soft cost.
"""
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

from safe_mppi.bowling_coverage import bowling_route_signature
from safe_mppi.config import load_config
from safe_mppi.lab_clutter_cfm_mppi import (
    CfmMppiConfig,
    attach_reference_config,
    guided_flow_proposals,
    refine_plans,
)
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_GOVERNOR_DIM,
    scene_sha256,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (
    bowling_123_spheres,
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)
from safe_mppi.lab_reference_flow_task import reference_window_validity_fraction
from safe_mppi.lab_clutter_evaluation import _scene_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN = ROOT / (
    "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
DEFAULT_TASK_CONFIG = ROOT / (
    "results/stage2_multi_sphere_n6/0812_pre2_speed400_faithful_gpu1/"
    "s4_bowling_visual/stability_expansion/task_config_resolved.json"
)
DEFAULT_REGIMES = {
    "safety_dominant": {"goal": 0.00, "safety": 1.00},
    "reward_dominant": {"goal": 1.00, "safety": 0.00},
    "balanced": {"goal": 1.00, "safety": 1.00},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_regimes(raw: str | None) -> dict[str, dict[str, float]]:
    values = DEFAULT_REGIMES if raw is None else json.loads(raw)
    if not isinstance(values, dict) or not values:
        raise ValueError("regimes must be a nonempty JSON object")
    parsed = {}
    for name, item in values.items():
        if not isinstance(item, dict) or set(item) != {"goal", "safety"}:
            raise ValueError(f"regime {name!r} must contain goal and safety")
        goal, safety = float(item["goal"]), float(item["safety"])
        if not 0 <= goal <= 1 or not 0 <= safety <= 1:
            raise ValueError("normalized coefficients must lie in [0,1]")
        parsed[str(name)] = {"goal": goal, "safety": safety}
    return parsed


def _state(task, spheres, seed: int) -> dict:
    env = task._environment(spheres)
    return {
        "x": env.start.copy(),
        "previous_applied": np.zeros(3, np.float32),
        "previous_raw": np.zeros(3, np.float32),
        "spheres": np.asarray(spheres, np.float32),
        "scene_seed": int(seed),
        "scene_hash": scene_sha256(env, spheres),
        "steps": 0,
        "collided": False,
        "oob": False,
    }


def _rollout(wrapped, task, config, spheres, gamma, seed, controller):
    env = task._environment(spheres)
    state = _state(task, spheres, seed)
    states = [state["x"].copy()]
    controls, applied, dense = [], [], []
    guidance_rows = []
    previous_plan = None
    status = "TIMEOUT"
    started = time.perf_counter()
    device = task.device
    goal_t = torch.as_tensor(env.goal, device=device, dtype=torch.float32)
    sphere_t = torch.as_tensor(spheres, device=device, dtype=torch.float32)
    bounds_t = torch.as_tensor(env.bounds, device=device, dtype=torch.float32)
    for step in range(config.taskspace.max_steps):
        context = task.context(state, float(gamma))
        state_t = torch.as_tensor(state["x"], device=device, dtype=torch.float32)
        previous_applied_t = torch.as_tensor(
            state["previous_applied"], device=device, dtype=torch.float32,
        )
        previous_raw_t = torch.as_tensor(
            state["previous_raw"], device=device, dtype=torch.float32,
        )
        proposal_generator = torch.Generator(device=device).manual_seed(
            int(seed) * 100_000 + step * 2,
        )
        proposals, guidance = guided_flow_proposals(
            wrapped, context, state_t, previous_applied_t, goal_t, sphere_t,
            bounds_t, proposal_generator, controller, previous_plan,
        )
        refinement_generator = torch.Generator(device=device).manual_seed(
            int(seed) * 100_000 + step * 2 + 1,
        )
        selected, diagnostics = refine_plans(
            proposals, state_t, previous_raw_t, previous_applied_t,
            goal_t, sphere_t, bounds_t, config.safemppi,
            refinement_generator, controller,
        )
        predicted, applied_plan, dense_plan = task._rollout_plan(
            state["x"], state["previous_applied"],
            selected.detach().cpu().numpy(),
        )
        updated = task.advance(state, selected)
        if not np.allclose(updated["x"], predicted[1], rtol=0, atol=2e-6):
            raise RuntimeError("CFM--MPPI execution disagrees with reference dynamics")
        states.append(updated["x"].copy())
        controls.append(selected[0].detach().cpu().numpy().copy())
        applied.append(applied_plan[0].copy())
        dense.append(dense_plan[0].copy())
        guidance_rows.append({
            **guidance,
            "generated_cost_min": float(diagnostics["generated_costs"].min().cpu()),
            "refined_cost_min": float(diagnostics["refined_costs"].min().cpu()),
        })
        previous_plan = selected.detach()
        state = updated
        terminal = task.terminal(state)
        if terminal is not None:
            status = terminal
            break
    states_array = np.asarray(states, np.float32)
    controls_array = np.asarray(controls, np.float32).reshape(-1, 3)
    applied_array = np.asarray(applied, np.float32).reshape(-1, 3)
    dense_array = np.asarray(dense, np.float32).reshape(
        len(controls_array), config.safemppi.integration_substeps, 3,
    )
    dense_path = (
        np.concatenate((states_array[:1, :3], dense_array.reshape(-1, 3)))
        if len(dense_array) else states_array[:, :3]
    )
    clearance = env.obstacle_clearance(dense_path)
    finite = clearance[np.isfinite(clearance)]
    route = None
    if status == "SUCCESS":
        route = bowling_route_signature(
            states_array, env.start, env.goal,
            sphere_radius_m=float(spheres[0, 3]),
        )
    mean_guidance = {
        name: float(np.mean([row[name] for row in guidance_rows]))
        for name in guidance_rows[0]
    } if guidance_rows else {}
    return {
        "status": status,
        "states": states_array,
        "controls": controls_array,
        "applied_controls": applied_array,
        "dense_steps": dense_array,
        "min_clearance_m": float(finite.min()) if len(finite) else None,
        "time_to_goal_s": (
            len(controls_array) * float(config.safemppi.dt)
            if status == "SUCCESS" else None
        ),
        "window_validity": reference_window_validity_fraction(
            _scene_config(config, spheres), states_array, dense_array, float(gamma),
        ),
        "bowling_route": route,
        "mean_guidance": mean_guidance,
        "wall_time_s": time.perf_counter() - started,
    }


def _summary(rows):
    status_counts = {
        name: sum(row["status"] == name for row in rows)
        for name in ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
    }
    successes = [row for row in rows if row["status"] == "SUCCESS"]
    return {
        "attempts": len(rows),
        "status_counts": status_counts,
        "success_rate": status_counts["SUCCESS"] / max(len(rows), 1),
        "collision_rate": status_counts["COLLISION"] / max(len(rows), 1),
        "oob_rate": status_counts["OOB"] / max(len(rows), 1),
        "timeout_rate": status_counts["TIMEOUT"] / max(len(rows), 1),
        "mean_window_validity": float(np.mean([row["window_validity"] for row in rows])),
        "successful_mean_clearance_m": (
            float(np.mean([row["min_clearance_m"] for row in successes]))
            if successes else None
        ),
        "successful_mean_time_to_goal_s": (
            float(np.mean([row["time_to_goal_s"] for row in successes]))
            if successes else None
        ),
        "successful_route_counts": {
            route: sum(
                row.get("bowling_route", {}).get("stable_code") == route
                for row in successes
            )
            for route in ("LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gammas", default="0.1,0.3,0.5,1.0")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--regimes-json")
    parser.add_argument("--proposal-count", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--copies-per-elite", type=int, default=32)
    parser.add_argument("--mppi-sigma", type=float, default=0.20)
    parser.add_argument("--mppi-lambda", type=float, default=0.10)
    parser.add_argument("--alpha-cbf", type=float, default=1.0)
    parser.add_argument("--cbf-margin-m", type=float, default=0.0)
    parser.add_argument("--goal-coefficient-max", type=float, default=0.25)
    parser.add_argument("--safety-coefficient-max", type=float, default=0.50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.trials < 1:
        parser.error("trials must be positive")
    regimes = _parse_regimes(args.regimes_json)
    gammas = tuple(float(value) for value in args.gammas.split(","))

    config = load_config(args.task_config)
    scene_spec = sphere_scene_spec_from_config(config)
    wrapped = load_lab_clutter_pre2_expansion_policy(
        args.pretrain_dir / "pretrained.pt",
        verifier_suffix_dim=LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim,
    ).to(args.device).eval()
    wrapped.policy.nfe = 16
    wrapped.policy.flow.nfe = 16
    attach_reference_config(wrapped, config.safemppi)
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    task = LabClutterPre2MultiPairExpansionTask(
        config,
        context_schema=wrapped.context_schema,
        device=args.device,
        verifier_mode="full_polytope",
        verifier_solver="analytic",
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=args.seed,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        scene_spec=scene_spec,
    )
    spheres = bowling_123_spheres(task.env.start, task.env.goal, scene_spec.radius)
    rows = []
    total = len(regimes) * len(gammas) * args.trials
    for regime_name, coefficients in regimes.items():
        controller = CfmMppiConfig(
            proposal_count=args.proposal_count,
            elite_count=args.elite_count,
            copies_per_elite=args.copies_per_elite,
            mppi_sigma=args.mppi_sigma,
            mppi_lambda=args.mppi_lambda,
            alpha_cbf=args.alpha_cbf,
            cbf_margin_m=args.cbf_margin_m,
            goal_coefficient_max=args.goal_coefficient_max,
            safety_coefficient_max=args.safety_coefficient_max,
            normalized_goal=coefficients["goal"],
            normalized_safety=coefficients["safety"],
        )
        for gamma in gammas:
            for trial in range(args.trials):
                rollout_seed = args.seed + trial * 37
                result = _rollout(
                    wrapped, task, config, spheres, gamma, rollout_seed, controller,
                )
                rows.append({
                    "regime": regime_name,
                    "normalized_goal": coefficients["goal"],
                    "normalized_safety": coefficients["safety"],
                    "raw_goal_coefficient": controller.goal_coefficient,
                    "raw_safety_coefficient": controller.safety_coefficient,
                    "gamma": gamma,
                    "trial": trial,
                    "rollout_seed": rollout_seed,
                    **result,
                })
                print(
                    f"[{len(rows)}/{total}] {regime_name} gamma={gamma:g} "
                    f"trial={trial} {result['status']} "
                    f"clear={result['min_clearance_m']:.3f}",
                    flush=True,
                )
    summaries = {
        regime: {
            "pooled": _summary([row for row in rows if row["regime"] == regime]),
            "per_gamma": {
                f"{gamma:g}": _summary([
                    row for row in rows
                    if row["regime"] == regime and np.isclose(row["gamma"], gamma)
                ])
                for gamma in gammas
            },
        }
        for regime in regimes
    }
    args.output.mkdir(parents=True)
    torch.save(rows, args.output / "raw_trajectories.pt")
    source_files = (
        ROOT / "safe_mppi/lab_clutter_cfm_mppi.py",
        Path(__file__).resolve(),
    )
    payload = {
        "status": "CFM_MPPI_BOWLING_COMPLETE",
        "method_contract": {
            "generative_prior": "PRE2",
            "nfe": 16,
            "proposal_count": args.proposal_count,
            "elite_count": args.elite_count,
            "copies_per_elite": args.copies_per_elite,
            "refinement_cost": "native_bowling_safemppi_soft_cost",
            "verifier_or_hp_used_for_selection": False,
            "cbf": "signed-clearance hdot+alpha*h; six spheres + six walls",
            "gradient_normalization": "batch_global_reference_faithful",
            "alpha_cbf": args.alpha_cbf,
            "cbf_margin_m": args.cbf_margin_m,
            "goal_coefficient_max": args.goal_coefficient_max,
            "safety_coefficient_max": args.safety_coefficient_max,
            "mppi_sigma": args.mppi_sigma,
            "mppi_lambda": args.mppi_lambda,
            "regimes": regimes,
        },
        "scene": {
            "start": task.env.start[:3].tolist(),
            "goal": task.env.goal.tolist(),
            "bounds": task.env.bounds.tolist(),
            "spheres": spheres.tolist(),
        },
        "summaries": summaries,
        "artifact_binding": {
            "pretrained_sha256": _sha256(args.pretrain_dir / "pretrained.pt"),
            "task_config_sha256": _sha256(args.task_config),
            "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
