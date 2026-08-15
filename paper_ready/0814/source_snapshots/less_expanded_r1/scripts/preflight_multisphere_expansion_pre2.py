#!/usr/bin/env python3
"""Pure-sampling PRE2 multi-sphere expansion preflight (no gradient update)."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.expansion import (  # noqa: E402
    RBFPosterior,
    mean_pairwise_lengthscale,
    normalized_ess,
)
from safe_mppi.geometry import build_nominal_polytope, hp_values  # noqa: E402
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    LAB_CLUTTER_GOVERNOR_DIM,
    scene_sha256,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (  # noqa: E402
    LabClutterPre2ExpansionTask,
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_visual_flow import (  # noqa: E402
    LAB_HP100_GRID_SHAPE,
    LabUniformHp100Rasterizer,
    pack_uniform_hp100_planes,
    uniform_hp100_grid_points,
)
from scripts.research_multisphere_expansion_pre2 import (  # noqa: E402
    _learned_phi_calibration,
    _set_flow_nfe,
)


@dataclass(frozen=True)
class CostArm:
    name: str
    wall_weight: float
    wall_target_m: float
    axis_weight: float
    axis_radius_m: float
    control_weight: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter_seed(master: int, *coordinates: Any) -> int:
    digest = hashlib.sha256(str(int(master)).encode())
    for value in coordinates:
        digest.update(b":")
        digest.update(str(value).encode())
    return int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)


def _parse_arm(raw: str) -> CostArm:
    fields = [value.strip() for value in raw.split(",")]
    if len(fields) != 6:
        raise ValueError(
            "--arm must be name,wall_weight,wall_target_m,axis_weight,"
            "axis_radius_m,control_weight"
        )
    arm = CostArm(
        name=fields[0],
        wall_weight=float(fields[1]),
        wall_target_m=float(fields[2]),
        axis_weight=float(fields[3]),
        axis_radius_m=float(fields[4]),
        control_weight=float(fields[5]),
    )
    values = tuple(asdict(arm).values())[1:]
    if (
        not arm.name
        or not all(np.isfinite(value) for value in values)
        or arm.wall_weight < 0.0
        or arm.wall_target_m < 0.0
        or arm.axis_weight < 0.0
        or arm.axis_radius_m <= 0.0
        or arm.control_weight < 0.0
    ):
        raise ValueError("invalid --arm values")
    return arm


def _task(config, policy, scene_spec, arm: CostArm, device):
    return LabClutterPre2ExpansionTask(
        config,
        context_schema=policy.context_schema,
        device=device,
        execution_z_bias_mode="none",
        verifier_mode="full_polytope",
        verifier_solver="analytic",
        execution_clearance_exp_weight=0.0,
        execution_soft_clearance_weight=0.0,
        execution_taskspace_quadratic_weight=arm.wall_weight,
        execution_taskspace_quadratic_target_m=arm.wall_target_m,
        execution_axis_cylinder_quadratic_weight=arm.axis_weight,
        execution_axis_cylinder_radius_m=arm.axis_radius_m,
        execution_control_weight=arm.control_weight,
        scene_spec=scene_spec,
    )


def _verification_dict(value) -> dict[str, Any]:
    return {
        "valid": bool(value.valid),
        "hp_eligible": bool(value.hp_eligible),
        "margin": float(value.margin),
        "progress": float(value.progress),
        "progress_eligible": bool(value.progress_eligible),
        "target_eligible": bool(value.target_eligible),
        "step_margin": (
            None if value.step_margin is None else float(value.step_margin)
        ),
        "execution_cost": float(value.execution_cost),
        "error": bool(value.error),
    }


def _fresh_shared_scene_state(task, base_state: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(base_state)
    state["x"] = task.env.start.copy()
    state["previous_applied"] = np.zeros(3, np.float32)
    state["previous_raw"] = np.zeros(3, np.float32)
    state["steps"] = 0
    state["collided"] = False
    state["oob"] = False
    state.pop("raw_history", None)
    return state


def _hp_snapshot(task, state: dict[str, Any], rasterizer) -> dict[str, Any]:
    env = task._environment(state["spheres"])
    position = np.asarray(state["x"][:3], np.float64)
    planes = pack_uniform_hp100_planes(env, position)
    with torch.no_grad():
        raster_with_channel = rasterizer(torch.from_numpy(
            planes.reshape(1, -1)
        ).to(task.device))[0].detach().cpu().numpy()
    if tuple(raster_with_channel.shape) != tuple(LAB_HP100_GRID_SHAPE):
        raise RuntimeError("uniform H_P raster changed shape")
    raster = raster_with_channel[0]
    points = uniform_hp100_grid_points(position)
    polytope = build_nominal_polytope(
        position,
        env.spheres,
        env.cylinders,
        env.bounds,
        sensing_range=env.mppi.sensing_range,
        obstacle_margin=0.0,
    )
    direct = np.clip(
        hp_values(polytope, points.reshape(-1, 3)), -1.0, 1.0,
    ).reshape(LAB_HP100_GRID_SHAPE[1:])
    max_error = float(np.max(np.abs(raster - direct)))
    if max_error > 2.0e-5:
        raise RuntimeError(
            f"exact H_P raster/direct mismatch: {max_error:.3e}"
        )
    chosen = np.s_[::4, ::4, ::5]
    display_points = points[chosen].reshape(-1, 3)
    display_hp = raster[chosen].reshape(-1)
    return {
        "position": position.astype(np.float32),
        "points": display_points.astype(np.float32),
        "clipped_hp": display_hp.astype(np.float32),
        "inside_taskspace": env.inside_taskspace(
            display_points
        ).astype(np.bool_),
        "inside_obstacle": (
            env.obstacle_clearance(display_points) < 0.0
        ).astype(np.bool_),
        "full_grid_shape": tuple(
            int(value) for value in raster_with_channel.shape
        ),
        "display_stride": (4, 4, 5),
        "full_grid_max_abs_error": max_error,
        "active_dynamic_faces": int(np.count_nonzero(planes[:, 4] > 0.5)),
        "query_center_collision": bool(
            env.obstacle_clearance(position[None])[0] < 0.0
        ),
    }


def _candidate_geometry(task, context, candidates) -> list[dict[str, Any]]:
    state6, previous_applied, previous_raw, env = task._decode_context(context)
    output = []
    for candidate in candidates:
        plan = candidate.detach().cpu().numpy().reshape(-1, 3)
        states, _, dense_steps = task._rollout_plan(
            state6, previous_applied, plan,
        )
        dense = dense_steps.reshape(-1, 3)
        full_h_inside = bool(env.inside_taskspace(dense).all())
        output.append({
            "path": states[:, :3].astype(np.float32),
            "full_h_taskspace_inside": full_h_inside,
            "tail_oob": not full_h_inside,
            "cost_breakdown": task.execution_cost_breakdown(
                env, states, plan, previous_raw,
            ),
        })
    return output


def _rollout_batch(
    *,
    task,
    policy,
    lengthscale: float,
    episodes: list[dict[str, Any]],
    K: int,
    B: int,
    beta: float,
    base_std: float,
    master_seed: int,
    max_steps: int,
    capture_stride: int,
    rasterizer,
) -> list[dict[str, Any]]:
    for episode in episodes:
        episode.setdefault("status", None)
        episode.setdefault("path", [
            np.asarray(episode["state"]["x"][:3], np.float32)
        ])
        episode.setdefault("dense_path", [])
        episode.setdefault("commands", [])
        episode.setdefault("frames", [])
        episode.setdefault("diagnostics", {
            "contexts": 0,
            "all_k_valid": 0,
            "selected_b_valid": 0,
            "all_k_progress_only_reject": 0,
            "selected_positive_recall_numerator": 0,
            "selected_positive_recall_denominator": 0,
            "all_k_tail_oob": 0,
            "selected_b_tail_oob": 0,
            "chosen_tail_oob": 0,
            "initial_sigma_tied_contexts": 0,
            "marginal_ess": [],
        })
    gp = RBFPosterior(lengthscale, 1.0e-2)
    gp.set_buffer(None)
    for control_step in range(max_steps):
        active = [row for row in episodes if row["status"] is None]
        if not active:
            break
        contexts = torch.stack([
            task.context(row["state"], row["gamma"]) for row in active
        ])
        generators = [
            torch.Generator(device=task.device).manual_seed(_counter_seed(
                master_seed,
                "preflight_gather",
                row["stream"],
                row["gamma"],
                control_step,
            ))
            for row in active
        ]
        with torch.inference_mode():
            candidates, bases = policy.sample_many_with_base(
                contexts, K, generators, base_std=base_std,
            )
            features = policy.embed_many(
                contexts, candidates, bases=bases,
            )
        for local, row in enumerate(active):
            context = contexts[local]
            candidate_block = candidates[local]
            acquisition_generator = generators[local]
            selected, selected_sigma, _, sigma_k_device = (
                gp.acquire_with_sigma(
                    features[local], B, beta, acquisition_generator,
                )
            )
            sigma_k = sigma_k_device.detach().cpu().numpy()
            selected_tensor = torch.as_tensor(
                selected, dtype=torch.long, device=candidate_block.device,
            )
            selected_candidates = candidate_block[selected_tensor]
            selected_results = task.verify(
                context, selected_candidates, row["gamma"],
            )
            eligible = [
                index for index, result in enumerate(selected_results)
                if (
                    not result.error
                    and result.valid
                    and result.progress_eligible
                    and result.target_eligible
                )
            ]
            chosen_local = (
                min(
                    eligible,
                    key=lambda index: selected_results[index].execution_cost,
                )
                if eligible else None
            )
            chosen_global = (
                None if chosen_local is None else selected[chosen_local]
            )
            capture = bool(
                row.get("capture", False)
                and (
                    control_step % capture_stride == 0
                    or chosen_global is None
                )
            )
            all_results = None
            geometry = None
            if capture:
                all_results = task.verify(
                    context, candidate_block, row["gamma"],
                )
                geometry = _candidate_geometry(
                    task, context, candidate_block,
                )
                for selected_local, global_index in enumerate(selected):
                    expected = _verification_dict(
                        all_results[global_index]
                    )
                    actual = _verification_dict(
                        selected_results[selected_local]
                    )
                    if expected != actual:
                        raise RuntimeError(
                            "offline all-K verifier disagrees with selected-B"
                        )
                frame = {
                    "control_step": int(control_step),
                    "robot": np.asarray(row["state"]["x"], np.float32),
                    "candidates": candidate_block.detach().cpu().numpy(),
                    "flow_bases": bases[local].detach().cpu().numpy(),
                    "sigma_K": sigma_k.astype(np.float32),
                    "selected": np.asarray(selected, np.int32),
                    "selected_sigma": np.asarray(
                        selected_sigma, np.float32,
                    ),
                    "verification_all_K": [
                        _verification_dict(value) for value in all_results
                    ],
                    "chosen": chosen_global,
                    "candidate_geometry": geometry,
                    "hp": _hp_snapshot(
                        task, row["state"], rasterizer,
                    ),
                    "gp_buffer_rows": 0,
                }
                row["frames"].append(frame)
            diagnostics = row["diagnostics"]
            diagnostics["contexts"] += 1
            tied = bool(np.max(sigma_k) - np.min(sigma_k) <= 1.0e-5)
            diagnostics["initial_sigma_tied_contexts"] += int(tied)
            diagnostics["marginal_ess"].append(
                normalized_ess(
                    torch.from_numpy(sigma_k), beta,
                )
            )
            if all_results is None:
                all_results = task.verify(
                    context, candidate_block, row["gamma"],
                )
            if geometry is None:
                geometry = _candidate_geometry(
                    task, context, candidate_block,
                )
            all_valid = [
                index for index, result in enumerate(all_results)
                if not result.error and result.valid
            ]
            selected_valid = [
                selected[index] for index, result in enumerate(selected_results)
                if not result.error and result.valid
            ]
            diagnostics["all_k_valid"] += len(all_valid)
            diagnostics["selected_b_valid"] += len(selected_valid)
            diagnostics["all_k_progress_only_reject"] += sum(
                bool(result.valid and not result.progress_eligible)
                for result in all_results
            )
            diagnostics["selected_positive_recall_numerator"] += len(
                set(all_valid).intersection(selected)
            )
            diagnostics["selected_positive_recall_denominator"] += len(
                all_valid
            )
            diagnostics["all_k_tail_oob"] += sum(
                bool(value["tail_oob"]) for value in geometry
            )
            diagnostics["selected_b_tail_oob"] += sum(
                bool(geometry[index]["tail_oob"]) for index in selected
            )
            diagnostics["chosen_tail_oob"] += int(
                chosen_global is not None
                and geometry[chosen_global]["tail_oob"]
            )
            if chosen_global is None:
                row["status"] = "NVP"
                row["nvp_reason"] = (
                    "PROGRESS"
                    if any(result.valid for result in selected_results)
                    else "VERIFIER"
                )
                continue
            chosen = candidate_block[chosen_global]
            state6, previous_applied, _, _ = task._decode_context(context)
            _, _, dense_steps = task._rollout_plan(
                state6,
                previous_applied,
                chosen.detach().cpu().numpy().reshape(-1, 3)[:1],
            )
            row["dense_path"].extend(
                dense_steps.reshape(-1, 3).astype(np.float32)
            )
            row["commands"].append(
                chosen.detach().cpu().numpy().reshape(-1, 3)[0].astype(
                    np.float32
                )
            )
            row["state"] = task.advance(row["state"], chosen)
            row["path"].append(
                np.asarray(row["state"]["x"][:3], np.float32)
            )
            status = task.terminal(row["state"])
            if status is not None:
                row["status"] = status
        # Keep the same status semantics as expansion: an episode that reaches
        # max_steps without a terminal event is a timeout.
    for row in episodes:
        if row["status"] is None:
            row["status"] = "TIMEOUT"
    return episodes


def _finalize_episode(task, row: dict[str, Any]) -> dict[str, Any]:
    path = np.asarray(row["path"], np.float32)
    dense = np.asarray(row["dense_path"], np.float32).reshape(-1, 3)
    commands = np.asarray(row["commands"], np.float32).reshape(-1, 3)
    env = task._environment(row["state"]["spheres"])
    clearance = (
        env.obstacle_clearance(dense) if len(dense) else np.asarray([])
    )
    command_norm = (
        np.linalg.norm(commands, axis=1) if len(commands) else np.asarray([])
    )
    jerk = (
        np.linalg.norm(np.diff(commands, axis=0), axis=1)
        if len(commands) > 1 else np.asarray([])
    )
    diagnostics = dict(row["diagnostics"])
    contexts = max(int(diagnostics["contexts"]), 1)
    all_queries = contexts * int(row["K"])
    selected_queries = contexts * int(row["B"])
    denominator = diagnostics.pop("selected_positive_recall_denominator")
    numerator = diagnostics.pop("selected_positive_recall_numerator")
    marginal_ess = diagnostics.pop("marginal_ess")
    diagnostics.update({
        "all_k_valid_rate": diagnostics["all_k_valid"] / all_queries,
        "selected_b_valid_rate": (
            diagnostics["selected_b_valid"] / selected_queries
        ),
        "selected_b_recall_of_all_k_positives": (
            numerator / denominator if denominator else None
        ),
        "all_k_progress_only_reject_rate": (
            diagnostics["all_k_progress_only_reject"] / all_queries
        ),
        "all_k_tail_oob_rate": diagnostics["all_k_tail_oob"] / all_queries,
        "selected_b_tail_oob_rate": (
            diagnostics["selected_b_tail_oob"] / selected_queries
        ),
        "chosen_tail_oob_rate": (
            diagnostics["chosen_tail_oob"] / contexts
        ),
        "initial_sigma_tied_rate": (
            diagnostics["initial_sigma_tied_contexts"] / contexts
        ),
        "mean_marginal_ess": (
            float(np.mean(marginal_ess)) if marginal_ess else None
        ),
    })
    return {
        "gamma": float(row["gamma"]),
        "episode": int(row["episode"]),
        "stream": str(row["stream"]),
        "retry_batch": int(row.get("retry_batch", 0)),
        "status": str(row["status"]),
        "nvp_reason": row.get("nvp_reason"),
        "steps": len(commands),
        "scene": task.scene_metadata(row["state"]),
        "path": path,
        "dense_path": dense,
        "commands": commands,
        "min_clearance_m": (
            float(clearance.min()) if len(clearance) else None
        ),
        "command_rms": (
            float(np.sqrt(np.mean(np.square(command_norm))))
            if len(command_norm) else None
        ),
        "command_peak": (
            float(command_norm.max()) if len(command_norm) else None
        ),
        "command_saturation_rate": (
            float(np.mean(np.abs(commands) >= 0.3 - 1.0e-6))
            if len(commands) else None
        ),
        "mean_command_jerk": (
            float(jerk.mean()) if len(jerk) else None
        ),
        "diagnostics": diagnostics,
        "frames": row["frames"],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = ("SUCCESS", "COLLISION", "OOB", "NVP", "TIMEOUT")
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in statuses
    }

    def mean(key, *, nested=False):
        values = [
            (row["diagnostics"][key] if nested else row[key])
            for row in rows
            if (row["diagnostics"][key] if nested else row[key]) is not None
        ]
        return float(np.mean(values)) if values else None

    return {
        "episodes": len(rows),
        "status_counts": counts,
        "SR": counts["SUCCESS"] / len(rows) if rows else 0.0,
        "CR": counts["COLLISION"] / len(rows) if rows else 0.0,
        "OOB": counts["OOB"] / len(rows) if rows else 0.0,
        "NVP": counts["NVP"] / len(rows) if rows else 0.0,
        "timeout": counts["TIMEOUT"] / len(rows) if rows else 0.0,
        "mean_steps": mean("steps"),
        "mean_min_clearance_m": mean("min_clearance_m"),
        "mean_command_rms": mean("command_rms"),
        "mean_command_peak": mean("command_peak"),
        "mean_command_saturation_rate": mean("command_saturation_rate"),
        "mean_command_jerk": mean("mean_command_jerk"),
        "all_k_valid_rate": mean("all_k_valid_rate", nested=True),
        "selected_b_valid_rate": mean("selected_b_valid_rate", nested=True),
        "selected_b_recall_of_all_k_positives": mean(
            "selected_b_recall_of_all_k_positives", nested=True,
        ),
        "all_k_progress_only_reject_rate": mean(
            "all_k_progress_only_reject_rate", nested=True,
        ),
        "all_k_tail_oob_rate": mean("all_k_tail_oob_rate", nested=True),
        "selected_b_tail_oob_rate": mean(
            "selected_b_tail_oob_rate", nested=True,
        ),
        "chosen_tail_oob_rate": mean("chosen_tail_oob_rate", nested=True),
        "initial_sigma_tied_rate": mean(
            "initial_sigma_tied_rate", nested=True,
        ),
        "mean_marginal_ess": mean("mean_marginal_ess", nested=True),
    }


def _new_episode(
    task,
    *,
    gamma: float,
    episode: int,
    reset_seed: int,
    stream: str,
    K: int,
    B: int,
    retry_batch: int = 0,
    capture: bool = False,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "gamma": float(gamma),
        "episode": int(episode),
        "stream": stream,
        "retry_batch": int(retry_batch),
        "capture": bool(capture),
        "K": int(K),
        "B": int(B),
        "state": (
            task.reset(gamma, episode, reset_seed)
            if state is None else state
        ),
    }


def _calibration_phase(args, config, policy, scene_spec, lengthscale, rasterizer):
    arms = [_parse_arm(value) for value in args.arm]
    if not arms:
        arms = [
            CostArm("base", 0.0, 0.15, 0.0, 1.1, 0.05),
            CostArm("wall250_axis5", 250.0, 0.15, 5.0, 1.1, 0.05),
            CostArm("wall500_axis5", 500.0, 0.15, 5.0, 1.1, 0.05),
            CostArm("wall250_axis5_ctrl02", 250.0, 0.15, 5.0, 1.1, 0.20),
            CostArm("wall500_axis5_ctrl05", 500.0, 0.15, 5.0, 1.1, 0.50),
        ]
    artifact = {"arms": {}, "rows": {}}
    for arm in arms:
        task = _task(config, policy, scene_spec, arm, args.device)
        episodes = []
        for scene_index in range(args.calibration_scenes):
            for gamma in args.gammas:
                episode = len(episodes)
                episodes.append(_new_episode(
                    task,
                    gamma=gamma,
                    episode=episode,
                    reset_seed=_counter_seed(
                        args.seed, "calibration_scene", scene_index,
                    ),
                    stream=f"cal_s{scene_index}_g{gamma:g}",
                    K=args.K,
                    B=args.B,
                ))
        _rollout_batch(
            task=task,
            policy=policy,
            lengthscale=lengthscale,
            episodes=episodes,
            K=args.K,
            B=args.B,
            beta=args.beta,
            base_std=args.flow_base_std,
            master_seed=args.seed,
            max_steps=args.max_steps,
            capture_stride=args.capture_stride,
            rasterizer=rasterizer,
        )
        rows = [_finalize_episode(task, row) for row in episodes]
        artifact["arms"][arm.name] = {
            "spec": asdict(arm),
            "pooled": _summary(rows),
            "by_gamma": {
                f"{gamma:g}": _summary([
                    row for row in rows if row["gamma"] == gamma
                ])
                for gamma in args.gammas
            },
        }
        artifact["rows"][arm.name] = rows
        print(
            f"[calibration] {arm.name}: "
            f"SR={artifact['arms'][arm.name]['pooled']['SR']:.3f} "
            f"OOB={artifact['arms'][arm.name]['pooled']['OOB']:.3f} "
            f"NVP={artifact['arms'][arm.name]['pooled']['NVP']:.3f} "
            f"tail={artifact['arms'][arm.name]['pooled']['chosen_tail_oob_rate']:.3f}",
            flush=True,
        )
    return artifact


def _true_dr_quota(
    args, task, policy, lengthscale, rasterizer,
) -> dict[str, Any]:
    committed = {float(gamma): [] for gamma in args.gammas}
    all_rows = {float(gamma): [] for gamma in args.gammas}
    retry_batches = {float(gamma): 0 for gamma in args.gammas}
    next_episode = 0
    for retry_batch in range(args.max_retry_batches):
        missing = [
            gamma for gamma in args.gammas
            if len(committed[float(gamma)]) < args.quota_per_gamma
        ]
        if not missing:
            break
        episodes = []
        for gamma in missing:
            for replica in range(args.parallel_episodes):
                episodes.append(_new_episode(
                    task,
                    gamma=gamma,
                    episode=next_episode,
                    reset_seed=_counter_seed(
                        args.seed,
                        "true_dr_reset",
                        retry_batch,
                        replica,
                    ),
                    stream=(
                        f"quota_b{retry_batch}_g{gamma:g}_r{replica}"
                    ),
                    K=args.K,
                    B=args.B,
                    retry_batch=retry_batch,
                ))
                next_episode += 1
            retry_batches[float(gamma)] = retry_batch + 1
        _rollout_batch(
            task=task,
            policy=policy,
            lengthscale=lengthscale,
            episodes=episodes,
            K=args.K,
            B=args.B,
            beta=args.beta,
            base_std=args.flow_base_std,
            master_seed=args.seed,
            max_steps=args.max_steps,
            capture_stride=args.capture_stride,
            rasterizer=rasterizer,
        )
        rows = [_finalize_episode(task, row) for row in episodes]
        for row in rows:
            gamma = float(row["gamma"])
            all_rows[gamma].append(row)
            if (
                row["status"] == "SUCCESS"
                and len(committed[gamma]) < args.quota_per_gamma
            ):
                committed[gamma].append(row)
        progress = " ".join(
            f"g{gamma:g}={len(committed[float(gamma)])}/"
            f"{args.quota_per_gamma}"
            for gamma in args.gammas
        )
        print(f"[quota] retry batch {retry_batch}: {progress}", flush=True)
    scene_hashes = {
        f"{gamma:g}": [row["scene"]["scene_hash"] for row in rows]
        for gamma, rows in all_rows.items()
    }
    for first_index, first in enumerate(args.gammas):
        for second in args.gammas[first_index + 1:]:
            overlap = set(scene_hashes[f"{first:g}"]).intersection(
                scene_hashes[f"{second:g}"]
            )
            if overlap:
                raise RuntimeError(
                    "true-DR scene hashes overlap across gamma cells"
                )
    return {
        "complete": all(
            len(rows) >= args.quota_per_gamma
            for rows in committed.values()
        ),
        "quota_per_gamma": args.quota_per_gamma,
        "parallel_episodes": args.parallel_episodes,
        "max_retry_batches": args.max_retry_batches,
        "retry_batches_used": {
            f"{gamma:g}": retry_batches[float(gamma)]
            for gamma in args.gammas
        },
        "committed": {
            f"{gamma:g}": rows for gamma, rows in committed.items()
        },
        "all_rows": {
            f"{gamma:g}": rows for gamma, rows in all_rows.items()
        },
        "summary": {
            f"{gamma:g}": _summary(all_rows[float(gamma)])
            for gamma in args.gammas
        },
        "scene_hashes_pairwise_disjoint": True,
    }


def _matched_gallery(
    args, task, policy, lengthscale, rasterizer,
) -> dict[str, Any]:
    episodes = []
    scenes = []
    next_episode = 100_000
    for scene_index in range(args.gallery_scenes):
        reset_seed = _counter_seed(args.seed, "matched_scene", scene_index)
        base = task.reset(0.5, next_episode, reset_seed)
        scenes.append({
            "scene_index": scene_index,
            "scene_seed": int(base["scene_seed"]),
            "scene_hash": str(base["scene_hash"]),
            "spheres": np.asarray(base["spheres"], np.float32),
        })
        for gamma in args.gammas:
            state = _fresh_shared_scene_state(task, base)
            state["scene_hash"] = scene_sha256(
                task._environment(state["spheres"]), state["spheres"],
            )
            episodes.append(_new_episode(
                task,
                gamma=gamma,
                episode=next_episode,
                reset_seed=reset_seed,
                stream=f"matched_s{scene_index}_g{gamma:g}",
                K=args.K,
                B=args.B,
                capture=math.isclose(gamma, 0.5),
                state=state,
            ))
            next_episode += 1
    _rollout_batch(
        task=task,
        policy=policy,
        lengthscale=lengthscale,
        episodes=episodes,
        K=args.K,
        B=args.B,
        beta=args.beta,
        base_std=args.flow_base_std,
        master_seed=args.seed,
        max_steps=args.max_steps,
        capture_stride=args.capture_stride,
        rasterizer=rasterizer,
    )
    rows = [_finalize_episode(task, row) for row in episodes]
    for scene_index, scene in enumerate(scenes):
        hashes = {
            row["scene"]["scene_hash"] for row in rows
            if row["stream"].startswith(f"matched_s{scene_index}_")
        }
        if hashes != {scene["scene_hash"]}:
            raise RuntimeError("matched-scene gallery changed geometry by gamma")
    return {
        "provenance": (
            "fixed-scene gamma counterfactual only; never eligible for replay "
            "or true-DR quota evidence"
        ),
        "scenes": scenes,
        "rows": rows,
        "summary": _summary(rows),
        "by_gamma": {
            f"{gamma:g}": _summary([
                row for row in rows if row["gamma"] == gamma
            ])
            for gamma in args.gammas
        },
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "preflight"), required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.1, 0.3, 0.5, 1.0))
    parser.add_argument("--flow-nfe", type=int, default=16)
    parser.add_argument("--flow-base-std", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=81231)
    parser.add_argument("--capture-stride", type=int, default=10)
    parser.add_argument("--calibration-scenes", type=int, default=2)
    parser.add_argument("--gallery-scenes", type=int, default=5)
    parser.add_argument("--parallel-episodes", type=int, default=8)
    parser.add_argument("--quota-per-gamma", type=int, default=2)
    parser.add_argument("--max-retry-batches", type=int, default=20)
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"--device {args.device!r} requires CUDA")
    if not 1 <= args.B < args.K:
        parser.error("preflight requires 1 <= B < K to preserve acquisition")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing to overwrite nonempty output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.task_config)
    scene_spec = sphere_scene_spec_from_config(config)
    policy = load_lab_clutter_pre2_expansion_policy(
        args.pretrain_dir / "pretrained.pt",
        verifier_suffix_dim=(
            LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim
        ),
    ).to(args.device)
    policy.eval()
    _set_flow_nfe(policy, args.flow_nfe)
    manifest = json.loads(
        (args.pretrain_dir / "pretrain_manifest.json").read_text()
    )
    calibration = _learned_phi_calibration(
        policy,
        args.pretrain_dir,
        manifest,
        config,
        args.seed,
        lab_profile=True,
        flow_base_std=args.flow_base_std,
        paired_noised_representation=True,
    )
    measured_lengthscale = mean_pairwise_lengthscale(calibration)
    recorded_lengthscale = manifest.get("rbf_lengthscale")
    if not np.isfinite(measured_lengthscale) or measured_lengthscale <= 0.0:
        raise RuntimeError("PRE2 calibration did not define a lengthscale")
    rasterizer = LabUniformHp100Rasterizer().to(args.device).eval()
    contract = {
        "kind": "pure sampling; no optimizer and no expansion buffer writes",
        "phase": args.phase,
        "pretrained_sha256": _sha256(args.pretrain_dir / "pretrained.pt"),
        "pretrain_manifest_sha256": _sha256(
            args.pretrain_dir / "pretrain_manifest.json"
        ),
        "task_config_sha256": _sha256(args.task_config),
        "source_sha256": _sha256(Path(__file__)),
        "policy_nfe": args.flow_nfe,
        "K": args.K,
        "B": args.B,
        "retry_B": args.B,
        "beta": args.beta,
        "paired_noised_representation": True,
        "scene_reset_rng": "upstream reset seed salted by exact gamma cell",
        "flow_latent_rng": "upstream gather seed salted by context gamma",
        "gp_buffer_rows": 0,
        "round1_uncertainty_interpretation": (
            "initial sigma_K is tied at one; first draw is uniform and later "
            "B slots use conditional within-K RBF diversity"
        ),
        "rbf_lengthscale_measured": measured_lengthscale,
        "rbf_lengthscale_manifest": recorded_lengthscale,
        "rbf_lengthscale_ratio_to_manifest": (
            measured_lengthscale / float(recorded_lengthscale)
            if recorded_lengthscale is not None else None
        ),
        "taskspace_bounds": config.raw["safety"],
        "sphere_count": [scene_spec.spec.count_min, scene_spec.spec.count_max],
        "mode_quota": None,
        "fa_alloc": "none",
        "fast_path": False,
    }
    if args.phase == "calibration":
        artifact = _calibration_phase(
            args, config, policy, scene_spec, measured_lengthscale, rasterizer,
        )
        filename = "calibration.pt"
    else:
        if len(args.arm) != 1:
            parser.error("--phase preflight requires exactly one --arm")
        arm = _parse_arm(args.arm[0])
        task = _task(config, policy, scene_spec, arm, args.device)
        artifact = {
            "arm": asdict(arm),
            "true_dr_quota": _true_dr_quota(
                args, task, policy, measured_lengthscale, rasterizer,
            ),
            "matched_gallery": _matched_gallery(
                args, task, policy, measured_lengthscale, rasterizer,
            ),
        }
        filename = "preflight.pt"
    output = {"contract": contract, **artifact}
    torch.save(output, args.output / filename)
    summary = _json_ready(output)
    # Frames and full paths live only in the PT artifact; JSON remains a compact
    # inspectable summary for monitoring and provenance.
    if args.phase == "calibration":
        for rows in summary["rows"].values():
            for row in rows:
                for key in ("path", "dense_path", "commands", "frames"):
                    row.pop(key, None)
    else:
        for section in ("committed", "all_rows"):
            for rows in summary["true_dr_quota"][section].values():
                for row in rows:
                    for key in ("path", "dense_path", "commands", "frames"):
                        row.pop(key, None)
        for row in summary["matched_gallery"]["rows"]:
            for key in ("dense_path", "commands", "frames"):
                row.pop(key, None)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"[done] {args.output / filename}", flush=True)


if __name__ == "__main__":
    main()
