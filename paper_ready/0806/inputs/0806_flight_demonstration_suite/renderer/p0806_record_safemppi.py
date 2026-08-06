"""SafeMPPI closed-loop recorder for the 0806 prep.

Runs the frozen Mode1SafeMPPI controller on a concrete five-cylinder scene and
preserves, per replanning step: all MPPI candidate trajectories, the multi-step
nominal safety verdicts, the online nominal polytope (poly_A/poly_b), costs,
weights, the executed averaged plan and its multi-step verdict.

The recording controller copies the frozen plan() body verbatim and only adds
tensor captures (no extra RNG draws, no reordering). Every episode is
re-verified by running the vanilla acquire.run_episode with the vanilla
controller and asserting bit-identical states, dense path, and polytopes.

Private tooling; lives only under /data3/research1/paper_ready_0806_private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_recorder(base_cls):
    class RecordingMode1SafeMPPI(base_cls):
        """Frozen plan() body + candidate/verdict capture. No behavior change."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.last_record = None

        def plan(self, state, goal, gamma, seed):
            import numpy as np
            import torch
            from safe_mppi.geometry import build_nominal_polytope, polytope_centroid

            started = time.perf_counter()
            cfg = self.cfg
            state_np = np.asarray(state, np.float32).reshape(6)
            goal_np = np.asarray(goal, np.float32).reshape(3)
            poly = build_nominal_polytope(
                state_np[:3], self.env.spheres, self.env.cylinders, self.env.bounds,
                sensing_range=cfg.sensing_range, obstacle_margin=cfg.obstacle_margin)
            A = torch.tensor(poly.A, device=self.device, dtype=torch.float32)
            b = torch.tensor(poly.b, device=self.device, dtype=torch.float32)
            center = torch.tensor(poly.center, device=self.device, dtype=torch.float32)
            margins = (b - A @ center).clamp_min(1e-3)
            centroid_np = polytope_centroid(poly)
            centroid_delta = torch.tensor(centroid_np - poly.center, device=self.device,
                                          dtype=torch.float32)
            centroid_norm = torch.linalg.norm(centroid_delta)
            centroid_direction = (centroid_delta / centroid_norm if float(centroid_norm) > 1e-6
                                  else torch.zeros(3, device=self.device))
            size = float(margins.min().detach().cpu())
            trapped = max(0.0, (cfg.sensing_range - size) / (size + cfg.centroid_eps))

            H, N = cfg.horizon, cfg.num_samples
            if cfg.warm_start and self._previous_sequence is not None:
                nominal = torch.cat([self._previous_sequence[1:], self._previous_sequence[-1:]], dim=0)
            else:
                nominal = torch.tensor(
                    cfg.initial_control, dtype=torch.float32, device=self.device
                ).repeat(H, 1)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            noise = torch.randn(N, H, 3, generator=generator, device=self.device) * self.sigma
            centers = nominal[None].expand(N, -1, -1).clone()

            mix = 0.0
            if float(torch.linalg.norm(centroid_direction)) > 1e-6:
                mix = min(max(cfg.centroid_gain * trapped, cfg.urgency_floor), 1.0)
                if self._previous_mix is not None:
                    mix = (1.0 - cfg.centroid_smooth) * mix + cfg.centroid_smooth * self._previous_mix
                self._previous_mix = mix
                n_centroid = min(int(round(mix * N)), N)
                if n_centroid:
                    start = N - n_centroid
                    target = cfg.demo_u_max * centroid_direction
                    centers[start:] += target[None, None]
                    direction = centroid_direction[None, None]
                    component = (noise[start:] * direction).sum(dim=-1, keepdim=True)
                    noise[start:] += (cfg.sigma_aniso - 1.0) * component * direction
            controls = torch.clamp(centers + noise, self.u_min, self.u_max)

            state0 = torch.tensor(state_np, device=self.device).view(1, 6).expand(N, -1)
            goal_t = torch.tensor(goal_np, device=self.device)
            bounds = torch.as_tensor(
                self.env.bounds, device=self.device, dtype=torch.float32,
            )
            states = state0.clone()
            costs = torch.zeros(N, device=self.device)
            infeasible = torch.zeros(N, dtype=torch.bool, device=self.device)
            minimum_hp = torch.full((N,), float("inf"), device=self.device)
            initial_distance = torch.linalg.norm(states[:, :3] - goal_t, dim=1)
            previous_raw = (
                torch.zeros(N, 3, device=self.device)
                if self._last_action is None
                else self._last_action[None].expand(N, -1)
            )
            previous_applied = (
                torch.zeros(N, 3, device=self.device)
                if self._last_applied_action is None
                else self._last_applied_action[None].expand(N, -1)
            )
            # --- capture-only additions ---
            candidate_positions = torch.empty(N, H, 3, device=self.device)
            fail_step = torch.full((N,), -1, dtype=torch.int64, device=self.device)
            initial_prev_applied = previous_applied[0].detach().clone()
            # ------------------------------
            from safe_mppi.controller import (
                _clearance, _taskspace_exponential_penalty,
            )
            for step in range(H):
                applied_controls = (
                    cfg.deployment_accel_smooth * controls[:, step]
                    + (1.0 - cfg.deployment_accel_smooth) * previous_applied
                )
                next_states = self._reference_step(states, applied_controls)
                old_hp = self._hp(states[:, :3], A, b, margins)
                new_hp = self._hp(next_states[:, :3], A, b, margins)
                minimum_hp = torch.minimum(minimum_hp, new_hp)
                step_bad = new_hp < (1.0 - float(gamma)) * old_hp
                fail_step = torch.where(
                    step_bad & (fail_step < 0),
                    torch.full_like(fail_step, step),
                    fail_step,
                )
                infeasible |= step_bad
                distance = torch.linalg.norm(next_states[:, :3] - goal_t, dim=1)
                costs += cfg.running_goal_weight * distance.square()
                costs += cfg.control_weight * controls[:, step].square().sum(dim=1)
                costs += cfg.smooth_weight * (
                    controls[:, step] - previous_raw
                ).square().sum(dim=1)
                costs -= cfg.progress_weight * (initial_distance - distance)
                clearance = _clearance(next_states[:, :3], self.spheres, self.cylinders)
                costs += cfg.soft_clearance_weight * torch.relu(
                    cfg.soft_clearance_target - clearance).square()
                if cfg.z_bias_weight:
                    exponent = (
                        (next_states[:, 2] - cfg.z_bias_plane) / cfg.z_bias_temperature
                    ).clamp(max=20.0)
                    costs += cfg.z_bias_weight * torch.exp(exponent)
                if cfg.taskspace_exponential_weight:
                    costs += _taskspace_exponential_penalty(
                        next_states[:, :3],
                        bounds,
                        cfg.taskspace_exponential_weight,
                        cfg.taskspace_exponential_temperature,
                    )
                candidate_positions[:, step] = next_states[:, :3]
                states = next_states
                previous_raw = controls[:, step]
                previous_applied = applied_controls
            costs += cfg.terminal_goal_weight * torch.linalg.norm(states[:, :3] - goal_t, dim=1).square()
            raw_costs = costs.clone()
            costs = torch.where(infeasible, torch.full_like(costs, float("inf")), costs)
            all_infeasible = bool(torch.isinf(costs).all())
            if all_infeasible:
                costs = -minimum_hp + 1e-3 * raw_costs
            weights = torch.softmax(-(costs - costs.min()) / max(cfg.temperature, 1e-6), dim=0)
            weights = torch.nan_to_num(weights, nan=0.0)
            if float(weights.sum()) < 1e-8:
                chosen = int(torch.argmin(costs))
                averaged_sequence = controls[chosen]
            else:
                weights /= weights.sum()
                averaged_sequence = torch.sum(weights[:, None, None] * controls, dim=0)
            action = torch.clamp(averaged_sequence[0], self.u_min, self.u_max)
            last_applied = (
                torch.zeros(3, device=self.device)
                if self._last_applied_action is None
                else self._last_applied_action
            )
            applied_action = (
                cfg.deployment_accel_smooth * action
                + (1.0 - cfg.deployment_accel_smooth) * last_applied
            )
            self._previous_sequence = averaged_sequence.detach()
            self._last_action = action.detach()
            self._last_applied_action = applied_action.detach()

            predicted_next_state = self._reference_step(
                torch.tensor(state_np, device=self.device).view(1, 6),
                applied_action.view(1, 3),
            )[0]
            next_state = predicted_next_state[:3]
            center_hp = self._hp(center.view(1, 3), A, b, margins)[0]
            next_hp = self._hp(next_state.view(1, 3), A, b, margins)[0]
            info = {
                "A": poly.A.astype(np.float32),
                "b": poly.b.astype(np.float32),
                "center": poly.center.astype(np.float32),
                "feasible_fraction": float((~infeasible).float().mean().detach().cpu()),
                "all_infeasible": all_infeasible,
                "applied_action": applied_action.detach().cpu().numpy(),
                "predicted_next_state": predicted_next_state.detach().cpu().numpy(),
                "mixture_fraction": float(mix),
                "online_one_step_slack": float((next_hp - (1.0 - gamma) * center_hp).detach().cpu()),
                "plan_time_s": time.perf_counter() - started,
                "averaged_control_sequence": averaged_sequence.detach().cpu().numpy(),
            }
            action_np = action.detach().cpu().numpy().astype(np.float32)

            # --- capture-only: executed averaged plan multi-step verdict ---
            with torch.no_grad():
                exec_states = torch.tensor(state_np, device=self.device).view(1, 6)
                exec_prev_applied = initial_prev_applied.view(1, 3)
                exec_positions = torch.empty(H, 3, device=self.device)
                exec_fail = -1
                exec_infeasible = False
                for step in range(H):
                    exec_applied = (
                        cfg.deployment_accel_smooth * averaged_sequence[step].view(1, 3)
                        + (1.0 - cfg.deployment_accel_smooth) * exec_prev_applied
                    )
                    exec_next = self._reference_step(exec_states, exec_applied)
                    e_old = self._hp(exec_states[:, :3], A, b, margins)
                    e_new = self._hp(exec_next[:, :3], A, b, margins)
                    if bool(e_new < (1.0 - float(gamma)) * e_old) and not exec_infeasible:
                        exec_infeasible = True
                        exec_fail = step
                    exec_positions[step] = exec_next[0, :3]
                    exec_states = exec_next
                    exec_prev_applied = exec_applied

            self.last_record = {
                "poly_A": poly.A.astype(np.float32),
                "poly_b": poly.b.astype(np.float32),
                "poly_center": poly.center.astype(np.float32),
                "candidate_positions": candidate_positions.detach().cpu().numpy().astype(np.float32),
                "candidate_infeasible": infeasible.detach().cpu().numpy(),
                "candidate_fail_step": fail_step.detach().cpu().numpy().astype(np.int16),
                "candidate_minimum_hp": minimum_hp.detach().cpu().numpy().astype(np.float32),
                "candidate_raw_costs": raw_costs.detach().cpu().numpy().astype(np.float32),
                "candidate_weights": weights.detach().cpu().numpy().astype(np.float32),
                "best_weight_index": int(torch.argmax(weights)),
                "averaged_control_sequence": info["averaged_control_sequence"].astype(np.float32),
                "executed_plan_positions": exec_positions.detach().cpu().numpy().astype(np.float32),
                "executed_plan_infeasible": bool(exec_infeasible),
                "executed_plan_fail_step": int(exec_fail),
                "action": action_np,
                "applied_action": info["applied_action"].astype(np.float32),
                "feasible_fraction": info["feasible_fraction"],
                "all_infeasible": info["all_infeasible"],
                "online_one_step_slack": info["online_one_step_slack"],
                "mixture_fraction": info["mixture_fraction"],
                "plan_time_s": info["plan_time_s"],
            }
            return action_np, info

    return RecordingMode1SafeMPPI


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
    parser.add_argument("--config", required=True, help="concrete scene config")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--gammas", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    from safe_mppi.config import load_config
    from safe_mppi.environment import TaskEnvironment
    from safe_mppi.controller import Mode1SafeMPPI
    from safe_mppi.acquire import run_episode, _object_array

    config_sha = sha256_file(args.config)
    if config_sha != args.expected_config_sha256:
        raise SystemExit(f"concrete config sha mismatch: {config_sha}")

    config = load_config(args.config)
    env = TaskEnvironment(config)
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        print(f"[GPU] visible cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    Recording = build_recorder(Mode1SafeMPPI)

    summary = {
        "kind": "0806 SafeMPPI full-candidate recording",
        "config": str(args.config),
        "config_sha256": config_sha,
        "device": str(args.device),
        "rollout_dynamics": config.data.rollout_dynamics,
        "seed": int(args.seed),
        "runs": [],
    }
    for gamma in args.gammas:
        controller = Recording(config.safemppi, env, device=args.device)
        records = []
        original_plan = controller.plan

        def capturing_plan(state, goal, g, seed, _c=controller, _r=records, _o=original_plan):
            action, info = _o(state, goal, g, seed)
            _r.append(_c.last_record)
            return action, info

        controller.plan = capturing_plan
        row, arrays = run_episode(
            env, controller, gamma, args.seed,
            rollout_dynamics=config.data.rollout_dynamics,
        )
        # verification: vanilla controller, vanilla episode, must be bit-identical
        vanilla = Mode1SafeMPPI(config.safemppi, env, device=args.device)
        row_v, arrays_v = run_episode(
            env, vanilla, gamma, args.seed,
            rollout_dynamics=config.data.rollout_dynamics,
        )
        identical = (
            np.array_equal(arrays["states"], arrays_v["states"])
            and np.array_equal(arrays["dense_positions"], arrays_v["dense_positions"])
            and len(arrays["poly_A"]) == len(arrays_v["poly_A"])
            and all(np.array_equal(a, b) for a, b in zip(arrays["poly_A"], arrays_v["poly_A"]))
            and all(np.array_equal(a, b) for a, b in zip(arrays["poly_b"], arrays_v["poly_b"]))
        )
        if not identical:
            raise SystemExit(
                f"RECORDING DIVERGED from vanilla controller at gamma={gamma}; aborting"
            )
        if row["collision"]:
            status = "COLLISION"
        elif row["taskspace_violation"]:
            status = "OOB"
        elif row["success"]:
            status = "SUCCESS"
        else:
            status = "TIMEOUT"
        zrow = z_departure(arrays["dense_positions"], arrays["states"][:, :3])
        name = f"run_g{gamma:g}_s{args.seed}"
        np.savez_compressed(
            out / f"{name}.npz",
            **arrays,
            spheres=env.spheres,
            cylinders=env.cylinders,
        )
        torch.save(
            {
                "meta": {
                    "gamma": float(gamma),
                    "seed": int(args.seed),
                    "status": status,
                    "num_samples": int(config.safemppi.num_samples),
                    "horizon": int(config.safemppi.horizon),
                    "config_sha256": config_sha,
                    "verified_bit_identical_to_vanilla": True,
                },
                "steps": records,
            },
            out / f"events_{name}.pt",
        )
        summary["runs"].append({
            **{k: v for k, v in row.items()},
            "status": status,
            "file": f"{name}.npz",
            "events": f"events_{name}.pt",
            "recorded_steps": len(records),
            "verified_bit_identical_to_vanilla": True,
            "z_departure": zrow,
        })
        print(json.dumps({"gamma": gamma, "status": status,
                          "steps": row["steps"],
                          "min_clearance_m": row["min_clearance_m"],
                          "verified": identical,
                          **zrow}), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[done]", out / "summary.json")


if __name__ == "__main__":
    main()
