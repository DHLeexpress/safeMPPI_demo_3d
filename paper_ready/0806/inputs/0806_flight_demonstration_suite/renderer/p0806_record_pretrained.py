"""Pretrained-policy closed-loop recorder + expansion-verifier certification.

Mirrors safe_mppi.lab_reference_flow_task.raw_reference_rollout exactly (one
policy sample per step, generator seed = seed*100_000 + step, ReferenceGovernor
execution, identical termination checks), so the executed trajectory is
bit-identical to scripts/export_lab_flow_frozen_references.py for the same
inputs — and this is asserted against the exporter's saved NPZ when provided.

Per step it additionally preserves the raw H=10 candidate window: its
governor-rolled knots, in-taskspace / collision-free checks, and the real
expansion verifier fit (fit_verifier_polytope A/b/margins/kinds/feasible/
worst_slack). A candidate is positive only if all three hold. The nominal
polytope chain (build_nominal_polytope) is used ONLY inside the policy's
frozen context encoder, never for certification or visualization data.

Private tooling; lives only under /data3/research1/paper_ready_0806_private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def z_departure(dense_positions, knot_positions, plane=0.9):
    dz = np.abs(np.asarray(dense_positions)[:, 2] - plane)
    kz = np.abs(np.asarray(knot_positions)[:, 2] - plane)
    return {
        "plane_z_m": plane,
        "max_abs_dz_dense_m": float(dz.max()),
        "mean_abs_dz_dense_m": float(dz.mean()),
        "max_abs_dz_knots_m": float(kz.max()),
        "mean_abs_dz_knots_m": float(kz.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--gammas", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-dir", default=None,
                        help="exporter output dir; states are asserted equal")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    from safe_mppi.config import load_config
    from safe_mppi.environment import TaskEnvironment, ReferenceGovernor
    from safe_mppi.lab_reference_flow_task import (
        PLAN_H, policy_context, _raw_history_before, _append_raw_history,
    )
    from safe_mppi.verifier_polytope import fit_verifier_polytope
    from flow_deployment.lab_pretrained import load_lab_deployment_controller

    config_sha = sha256_file(args.config)
    if config_sha != args.expected_config_sha256:
        raise SystemExit(f"concrete config sha mismatch: {config_sha}")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"checkpoint sha mismatch: {checkpoint_sha}")

    config = load_config(args.config)
    env = TaskEnvironment(config)
    controller, policy_contract = load_lab_deployment_controller(
        args.checkpoint, env,
        sampling_temperature=args.sampling_temperature,
        device=args.device,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    policy = controller.policy

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    substeps = int(config.safemppi.integration_substeps)
    sensing_range = float(config.safemppi.sensing_range)

    summary = {
        "kind": "0806 pretrained-policy recording with expansion-verifier certification",
        "config": str(args.config),
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "sampling_temperature": float(args.sampling_temperature),
        "seed": int(args.seed),
        "device": str(args.device),
        "verifier": {"mode": "full_polytope_fit_verifier_polytope",
                     "face_solver": "analytic"},
        "runs": [],
    }

    for gamma in args.gammas:
        governor = ReferenceGovernor(config.safemppi)
        state = env.start.copy()
        raw_history = _raw_history_before(np.empty((0, 3)), 0)
        states = [state.copy()]
        controls, applied_controls, dense_steps = [], [], []
        status = "TIMEOUT"
        records = []
        with torch.no_grad():
            for step in range(config.taskspace.max_steps):
                context_np = policy_context(
                    policy, env, state, float(gamma),
                    raw_history=raw_history,
                    previous_raw=(
                        np.clip(
                            controls[-1],
                            -float(config.safemppi.demo_u_max),
                            float(config.safemppi.demo_u_max),
                        ).astype(np.float32)
                        if controls else np.zeros(3, np.float32)
                    ),
                    previous_applied=governor.previous_applied,
                )
                context = torch.from_numpy(context_np).to(args.device)
                generator = torch.Generator(device=torch.device(args.device))
                generator.manual_seed(int(args.seed) * 100_000 + step)
                plan = policy.sample(
                    context, 1, generator,
                    base_std=float(args.sampling_temperature),
                )[0].detach().cpu().numpy()
                plan = plan.reshape(PLAN_H, 3)

                # certify the full H=10 candidate window from the current state
                probe = ReferenceGovernor(config.safemppi)
                probe.previous_applied = governor.previous_applied.copy()
                knots = [np.asarray(state, np.float32).copy()]
                window_dense = []
                for h in range(PLAN_H):
                    k_state, _, k_dense = probe.step(knots[-1], plan[h])
                    knots.append(k_state)
                    window_dense.append(k_dense)
                knots = np.asarray(knots, np.float32)              # (H+1, 6)
                window_dense = np.concatenate(
                    [knots[:1, :3], np.concatenate(window_dense, axis=0)], axis=0,
                )
                clearance = env.obstacle_clearance(window_dense)
                collision_free = bool(
                    np.isinf(clearance).all()
                    or (np.isfinite(clearance).any() and float(clearance.min()) > 0.0)
                )
                inside = bool(env.inside_taskspace(window_dense).all())
                vp = fit_verifier_polytope(
                    knots[:, :3], env.spheres, env.cylinders,
                    float(gamma), sensing_range,
                )
                positive = bool(inside and collision_free and vp.feasible)
                # first geometric failure knot, for the red-x marker
                per_point_bad = (
                    ~env.inside_taskspace(window_dense)
                    | (np.isfinite(clearance) & (clearance <= 0.0))
                )
                if per_point_bad.any():
                    first_bad_point = int(np.flatnonzero(per_point_bad)[0])
                    first_fail_knot = min(
                        PLAN_H, int(np.ceil(first_bad_point / substeps)),
                    )
                    if not inside and not collision_free:
                        failure_reason = "taskspace_and_collision"
                    elif not inside:
                        failure_reason = "taskspace"
                    else:
                        failure_reason = "collision"
                elif not vp.feasible:
                    first_fail_knot = PLAN_H
                    failure_reason = "verifier_infeasible"
                else:
                    first_fail_knot = -1
                    failure_reason = None

                records.append({
                    "context": context_np.astype(np.float32),
                    "plan": plan.astype(np.float32),
                    "window_knots": knots,
                    "window_dense": window_dense.astype(np.float32),
                    "in_taskspace": inside,
                    "collision_free": collision_free,
                    "verifier_feasible": bool(vp.feasible),
                    "verifier_worst_slack": float(vp.worst_slack),
                    "verifier_A": np.asarray(vp.A, np.float32),
                    "verifier_b": np.asarray(vp.b, np.float32),
                    "verifier_center": np.asarray(vp.center, np.float32),
                    "verifier_margins": np.asarray(vp.margins, np.float32),
                    "verifier_kinds": list(vp.kinds),
                    "verifier_sensing_radius": float(vp.sensing_radius),
                    "positive": positive,
                    "failure_reason": failure_reason,
                    "first_fail_knot": int(first_fail_knot),
                })

                # execute exactly like raw_reference_rollout
                control = plan[0]
                raw_history = _append_raw_history(raw_history, control)
                state, applied, dense = governor.step(state, control)
                states.append(state.copy())
                controls.append(control.copy())
                applied_controls.append(applied.copy())
                dense_steps.append(dense.copy())
                clearance = env.obstacle_clearance(dense)
                if np.isfinite(clearance).any() and float(clearance.min()) < 0.0:
                    status = "COLLISION"
                    break
                if not env.inside_taskspace(dense).all():
                    status = "OOB"
                    break
                if env.reached(state[:3]):
                    status = "SUCCESS"
                    break

        states_array = np.asarray(states, np.float32)
        controls_array = np.asarray(controls, np.float32).reshape(-1, 3)
        applied_array = np.asarray(applied_controls, np.float32).reshape(-1, 3)
        dense_array = np.asarray(dense_steps, np.float32).reshape(
            len(controls_array), substeps, 3,
        )
        dense_positions = np.concatenate(
            [states_array[:1, :3], dense_array.reshape(-1, 3)],
        )

        gamma_tag = f"{gamma:g}".replace(".", "p")
        reference_check = None
        if args.reference_dir:
            ref = Path(args.reference_dir) / f"gamma_{gamma_tag}_seed_{args.seed}.npz"
            with np.load(ref) as data:
                same_states = np.array_equal(data["states"], states_array)
                same_controls = np.array_equal(data["controls"], controls_array)
                same_status = str(data["status"]) == status
            if not (same_states and same_controls and same_status):
                raise SystemExit(
                    f"RECORDING DIVERGED from frozen exporter reference {ref}"
                )
            reference_check = {"file": str(ref), "bit_identical": True}

        zrow = z_departure(dense_positions, states_array[:, :3])
        clearance_all = env.obstacle_clearance(dense_positions)
        finite = clearance_all[np.isfinite(clearance_all)]
        name = f"run_g{gamma:g}_s{args.seed}"
        np.savez_compressed(
            out / f"{name}.npz",
            dense_positions=dense_positions.astype(np.float32),
            states=states_array,
            controls=controls_array,
            executed_controls=applied_array,
            dense_steps=dense_array,
            gamma=np.float32(gamma),
            seed=np.int64(args.seed),
            sampling_temperature=np.float32(args.sampling_temperature),
            status=np.asarray(status),
            spheres=env.spheres,
            cylinders=env.cylinders,
        )
        torch.save(
            {
                "meta": {
                    "gamma": float(gamma),
                    "seed": int(args.seed),
                    "status": status,
                    "sampling_temperature": float(args.sampling_temperature),
                    "config_sha256": config_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "verifier_mode": "full_polytope_fit_verifier_polytope",
                    "face_solver": "analytic",
                    "reference_check": reference_check,
                },
                "steps": records,
            },
            out / f"events_{name}.pt",
        )
        positives = sum(r["positive"] for r in records)
        run_row = {
            "gamma": float(gamma),
            "seed": int(args.seed),
            "status": status,
            "steps": int(len(controls_array)),
            "min_clearance_m": float(finite.min()) if len(finite) else None,
            "candidate_windows": len(records),
            "verifier_positive_windows": int(positives),
            "file": f"{name}.npz",
            "events": f"events_{name}.pt",
            "reference_check": reference_check,
            "z_departure": zrow,
        }
        summary["runs"].append(run_row)
        print(json.dumps(run_row), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[done]", out / "summary.json")


if __name__ == "__main__":
    main()
