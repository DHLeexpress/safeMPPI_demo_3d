#!/usr/bin/env python3
"""Visualize raw r0 failures in the fixed Stage-2 triangle scene."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.ball_flow_theta import start_goal_frame  # noqa: E402
from safe_mppi.config import ObstacleConfig, load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    PerpendicularTriangleSphereScene,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402


GAMMAS = (0.1, 0.3, 0.5, 1.0)
FIXED_SEED_OFFSET = 1_000_003
GAMMA_SEED_STRIDE = 10_007
ROLLOUT_SEED_STRIDE = 37


def _fixed_config(config, spheres: np.ndarray):
    return replace(
        config,
        obstacles=ObstacleConfig(
            spheres=tuple(tuple(map(float, row)) for row in spheres),
            cylinders=(),
        ),
    )


def _dense_path(result: dict) -> np.ndarray:
    states = np.asarray(result["states"], float)
    dense = np.asarray(result["dense_steps"], float)
    if not len(dense):
        return states[:, :3]
    return np.concatenate([states[:1, :3], dense.reshape(-1, 3)])


def _first_failure_point(result: dict, env: TaskEnvironment) -> np.ndarray | None:
    path = _dense_path(result)
    if result["status"] == "COLLISION":
        clearance = env.obstacle_clearance(path)
        hit = np.flatnonzero(np.isfinite(clearance) & (clearance < 0.0))
        return path[int(hit[0])] if len(hit) else path[-1]
    if result["status"] == "OOB":
        outside = np.flatnonzero(~env.inside_taskspace(path))
        return path[int(outside[0])] if len(outside) else path[-1]
    if result["status"] == "TIMEOUT":
        return path[-1]
    return None


def _collision_sphere(result: dict, env: TaskEnvironment) -> int | None:
    if result["status"] != "COLLISION":
        return None
    point = _first_failure_point(result, env)
    if point is None or not len(env.spheres):
        return None
    surface = np.linalg.norm(env.spheres[:, :3] - point[None], axis=1)
    surface -= env.spheres[:, 3]
    return int(np.argmin(surface))


def _draw_projected_spheres(
    axis,
    local_spheres: np.ndarray,
    first: int,
    second: int,
    physical_radius: float,
) -> None:
    for sphere in local_spheres:
        center = (sphere[first], sphere[second])
        axis.add_patch(Circle(
            center,
            physical_radius,
            facecolor="#8d9295",
            edgecolor="#555b60",
            linewidth=0.8,
            alpha=0.48,
            zorder=2,
        ))
        axis.add_patch(Circle(
            center,
            sphere[3],
            fill=False,
            edgecolor="#4f83a1",
            linestyle="--",
            linewidth=1.0,
            alpha=0.72,
            zorder=2,
        ))


def _status_summary(rows: list[dict]) -> dict:
    count = len(rows)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
    }
    successful = [row for row in rows if row["status"] == "SUCCESS"]
    return {
        "episodes": count,
        "success": counts["SUCCESS"],
        "collision": counts["COLLISION"],
        "oob": counts["OOB"],
        "timeout": counts["TIMEOUT"],
        "SR": counts["SUCCESS"] / count,
        "CR": counts["COLLISION"] / count,
        "OOB": counts["OOB"] / count,
        "window_validity": float(np.mean([
            row["window_validity"] for row in rows
        ])),
        "successful_min_clearance_m": (
            float(np.mean([row["min_clearance_m"] for row in successful]))
            if successful else None
        ),
        "successful_time_to_goal_s": (
            float(np.mean([row["time_to_goal_s"] for row in successful]))
            if successful else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        default=(
            ROOT
            / "results/stage1_single_ball_t128/pretrain_hp100_t128_d3_e52"
        ),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=(
            ROOT / "configs/lab_clutter_spheres_stage2_triangle15_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/stage2_sphere_clutter_triangle15/preexpansion_r0_s070_p030_010"
        ),
    )
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires an available MPS backend")
    config = load_config(args.task_config)
    template_env = TaskEnvironment(config)
    scene_spec = sphere_scene_spec_from_config(config)
    if not isinstance(scene_spec, PerpendicularTriangleSphereScene):
        parser.error("visualization requires perpendicular_triangle_anisotropic_v1")
    spheres = scene_spec.base_scene(template_env)
    fixed_config = _fixed_config(config, spheres)
    env = TaskEnvironment(fixed_config)
    policy = load_lab_reference_policy(
        args.pretrain_dir / "pretrained.pt"
    ).to(device).eval()

    rows_by_gamma: dict[float, list[dict]] = {}
    all_rows = []
    with torch.no_grad():
        for gamma_index, gamma in enumerate(GAMMAS):
            rows = []
            for rollout in range(args.rollouts):
                rollout_seed = (
                    int(args.seed)
                    + FIXED_SEED_OFFSET
                    + GAMMA_SEED_STRIDE * gamma_index
                    + ROLLOUT_SEED_STRIDE * rollout
                )
                result = raw_reference_rollout(
                    policy,
                    fixed_config,
                    gamma,
                    rollout_seed,
                    device=device,
                    sampling_temperature=1.0,
                )
                row = {
                    **result,
                    "gamma": gamma,
                    "rollout": rollout,
                    "rollout_seed": rollout_seed,
                    "collision_sphere": _collision_sphere(result, env),
                }
                rows.append(row)
                all_rows.append(row)
            rows_by_gamma[gamma] = rows

    frame = start_goal_frame(env).astype(float)
    start = np.asarray(env.start[:3], float)
    goal_local = (np.asarray(env.goal, float) - start) @ frame
    sphere_local = np.column_stack([
        (spheres[:, :3] - start[None]) @ frame,
        spheres[:, 3],
    ])
    physical_radius = float(
        config.raw["scene_randomization"]["physical_radius_m"]
    )
    views = (
        (0, 1, r"longitudinal $e_\parallel$ [m]", r"lateral $e_1$ [m]", "top"),
        (0, 2, r"longitudinal $e_\parallel$ [m]", r"vertical $e_2$ [m]", "side"),
        (1, 2, r"lateral $e_1$ [m]", r"vertical $e_2$ [m]", "head-on"),
    )
    gamma_colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(GAMMAS) - 1, 1)
        )
        for index, gamma in enumerate(GAMMAS)
    }
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })
    fig, axes = plt.subplots(
        len(GAMMAS), len(views),
        figsize=(15.2, 15.5),
        squeeze=False,
    )
    for row_index, gamma in enumerate(GAMMAS):
        rows = rows_by_gamma[gamma]
        summary = _status_summary(rows)
        for column_index, (first, second, xlabel, ylabel, view_name) in enumerate(views):
            axis = axes[row_index, column_index]
            _draw_projected_spheres(
                axis, sphere_local, first, second, physical_radius,
            )
            for record in rows:
                path = (_dense_path(record) - start[None]) @ frame
                success = record["status"] == "SUCCESS"
                axis.plot(
                    path[:, first], path[:, second],
                    color=gamma_colors[gamma],
                    linewidth=(1.45 if success else 0.85),
                    linestyle=("-" if success else "--"),
                    alpha=(0.72 if success else 0.40),
                    zorder=3,
                )
                failure = _first_failure_point(record, env)
                if failure is not None:
                    failure_local = (failure - start) @ frame
                    axis.scatter(
                        failure_local[first], failure_local[second],
                        marker="x", s=28, linewidths=1.2,
                        color="#c8321b", alpha=0.80, zorder=5,
                    )
            axis.scatter(
                0.0, 0.0, marker="s", s=30,
                color="black", zorder=6,
            )
            axis.scatter(
                goal_local[first], goal_local[second],
                marker="*", s=100, color="#f1c40f",
                edgecolor="black", linewidth=0.5, zorder=6,
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.20)
            axis.set_aspect("equal", adjustable="datalim")
            if row_index == 0:
                axis.set_title(view_name)
            if column_index == 0:
                axis.text(
                    -0.23, 0.5,
                    rf"$\gamma={gamma:g}$" + "\n"
                    + rf"SR={summary['SR']:.2f}, CR={summary['CR']:.2f}" + "\n"
                    + rf"$V^{{\rm win}}={summary['window_validity']:.2f}$",
                    transform=axis.transAxes,
                    ha="center", va="center", rotation=90,
                    fontsize=12,
                )
    legend = (
        Line2D([0], [0], color="#444", lw=1.5, label="SUCCESS"),
        Line2D([0], [0], color="#777", lw=1.0, ls="--", label="failed rollout"),
        Line2D(
            [0], [0], marker="x", color="none",
            markeredgecolor="#c8321b", label="first failure point",
        ),
        Line2D([0], [0], color="#555b60", lw=5, alpha=0.48, label="physical sphere"),
        Line2D([0], [0], color="#4f83a1", lw=1.0, ls="--", label="robot-inflated sphere"),
    )
    fig.legend(
        handles=legend, loc="upper center", ncol=len(legend),
        frameon=False, bbox_to_anchor=(0.5, 0.925),
    )
    fig.suptitle(
        "Raw pretrained policy in the unjittered tight-triangle scene\n"
        "temperature 1; independent fixed seed bank",
        fontsize=18, weight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.875))

    args.output.mkdir(parents=True, exist_ok=False)
    png = args.output / "pretrained_r0_nominal_triangle_failures.png"
    pdf = args.output / "pretrained_r0_nominal_triangle_failures.pdf"
    fig.savefig(png, dpi=210, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    per_gamma = {
        f"{gamma:g}": _status_summary(rows_by_gamma[gamma])
        for gamma in GAMMAS
    }
    collision_counts = {
        str(index): sum(
            row["collision_sphere"] == index for row in all_rows
        )
        for index in range(len(spheres))
    }
    metrics = {
        "status": "STAGE2_PREEXPANSION_VISUALIZATION_COMPLETE",
        "pretrain_dir": str(args.pretrain_dir.resolve()),
        "task_config": str(args.task_config.resolve()),
        "device": str(device),
        "sampling_temperature": 1.0,
        "seed": int(args.seed),
        "rollouts_per_gamma": int(args.rollouts),
        "fixed_scene": "unjittered perpendicular equilateral triangle",
        "physical_radius_m": physical_radius,
        "modeled_radius_m": float(scene_spec.radius),
        "spheres": spheres.tolist(),
        "per_gamma": per_gamma,
        "collision_count_by_sphere_index": collision_counts,
        "rollouts": [{
            "gamma": float(row["gamma"]),
            "rollout": int(row["rollout"]),
            "rollout_seed": int(row["rollout_seed"]),
            "status": str(row["status"]),
            "collision_sphere": row["collision_sphere"],
            "min_clearance_m": row["min_clearance_m"],
            "time_to_goal_s": row["time_to_goal_s"],
            "window_validity": float(row["window_validity"]),
        } for row in all_rows],
    }
    metrics_path = args.output / "pretrained_r0_nominal_triangle_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"[failure gallery] {png}")
    print(f"[vector PDF] {pdf}")
    print(f"[metrics] {metrics_path}")


if __name__ == "__main__":
    main()
