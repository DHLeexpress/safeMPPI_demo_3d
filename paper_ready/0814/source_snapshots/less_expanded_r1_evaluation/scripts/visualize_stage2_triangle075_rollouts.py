#!/usr/bin/env python3
"""Render accepted 3-D center clouds and raw pretrained gamma overlays."""
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
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import ObstacleConfig, load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    PerpendicularTriangleSphereScene,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402


GAMMAS = (0.1, 0.3, 0.5, 1.0)
COLORS = {
    0.1: "#3e63dd",
    0.3: "#00a2c7",
    0.5: "#30a46c",
    1.0: "#e5484d",
}


def _fixed_config(config, spheres: np.ndarray):
    return replace(config, obstacles=ObstacleConfig(
        spheres=tuple(tuple(map(float, row)) for row in spheres),
        cylinders=(),
    ))


def _dense_path(result: dict) -> np.ndarray:
    states = np.asarray(result["states"], float)
    dense = np.asarray(result["dense_steps"], float)
    return states[:, :3] if not len(dense) else np.concatenate([
        states[:1, :3], dense.reshape(-1, 3),
    ])


def _draw_sphere(axis, center, radius, *, color="#8d9295", alpha=0.23):
    u = np.linspace(0.0, 2.0 * np.pi, 22)
    v = np.linspace(0.0, np.pi, 13)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _style_3d(axis, env: TaskEnvironment, title: str) -> None:
    axis.set(
        xlim=env.bounds[0], ylim=env.bounds[1], zlim=env.bounds[2],
        xlabel=r"$x$ [m]", ylabel=r"$y$ [m]", zlabel=r"$z$ [m]",
        title=title,
    )
    axis.set_box_aspect(tuple(env.bounds[:, 1] - env.bounds[:, 0]))
    axis.view_init(elev=23, azim=-54)


def _plot_overlay(axis, env, spheres, records, title, physical_radius):
    for sphere in spheres:
        _draw_sphere(axis, sphere[:3], physical_radius)
    for gamma, result in records.items():
        path = _dense_path(result)
        success = result["status"] == "SUCCESS"
        axis.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color=COLORS[gamma], linewidth=2.0,
            linestyle="-" if success else "--",
            alpha=0.92,
        )
        if not success:
            axis.scatter(*path[-1], marker="x", s=28, color=COLORS[gamma])
    axis.scatter(*env.start[:3], marker="s", s=28, color="black")
    axis.scatter(
        *env.goal, marker="*", s=95, color="#f5c542",
        edgecolor="black", linewidth=0.5,
    )
    _style_3d(axis, env, title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-dir", type=Path,
        default=ROOT / "results/stage1_single_ball_t128/pretrain_hp100_t128_d3_e52",
    )
    parser.add_argument(
        "--task-config", type=Path,
        default=ROOT / "configs/lab_clutter_spheres_stage2_triangle075_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/stage2_sphere_clutter_triangle075/pretrained_overlays",
    )
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--cloud-scenes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    config = load_config(args.task_config)
    template_env = TaskEnvironment(config)
    spec = sphere_scene_spec_from_config(config)
    if not isinstance(spec, PerpendicularTriangleSphereScene):
        parser.error("requires perpendicular_triangle_anisotropic_v1")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    policy = load_lab_reference_policy(args.pretrain_dir / "pretrained.pt").to(device).eval()

    cloud_scenes = []
    for i in range(args.cloud_scenes):
        cloud_scenes.append(spec.sample(template_env, args.seed + 1009 * i))
    cloud = np.stack(cloud_scenes)
    selected = cloud[:args.scenes]
    canonical = np.column_stack([
        spec.base_centers(template_env),
        np.full(3, spec.radius, dtype=float),
    ])

    rollout_bank = []
    with torch.no_grad():
        for scene_index, spheres in enumerate(selected):
            fixed = _fixed_config(config, spheres)
            records = {}
            for gamma_index, gamma in enumerate(GAMMAS):
                rollout_seed = args.seed + 1_000_003 + 10_007 * gamma_index + 37 * scene_index
                records[gamma] = raw_reference_rollout(
                    policy, fixed, gamma, rollout_seed,
                    device=device, sampling_temperature=1.0,
                )
            rollout_bank.append((fixed, spheres, records))
        fixed = _fixed_config(config, canonical)
        canonical_records = {}
        for gamma_index, gamma in enumerate(GAMMAS):
            canonical_records[gamma] = raw_reference_rollout(
                policy, fixed, gamma,
                args.seed + 2_000_003 + 10_007 * gamma_index,
                device=device, sampling_temperature=1.0,
            )

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.titlesize": 11, "axes.labelsize": 9,
    })
    args.output.mkdir(parents=True, exist_ok=False)

    fig = plt.figure(figsize=(11.8, 8.9))
    axis = fig.add_subplot(111, projection="3d")
    for identity, color in enumerate(("#e5484d", "#30a46c", "#3e63dd")):
        axis.scatter(
            cloud[:, identity, 0], cloud[:, identity, 1], cloud[:, identity, 2],
            s=5, alpha=0.16, color=color, label=f"center {identity + 1}",
        )
        axis.scatter(
            canonical[identity, 0], canonical[identity, 1], canonical[identity, 2],
            s=60, marker="D", color=color, edgecolor="black", linewidth=0.7,
        )
    axis.plot(
        [template_env.start[0], template_env.goal[0]],
        [template_env.start[1], template_env.goal[1]],
        [template_env.start[2], template_env.goal[2]],
        linestyle="--", color="black", linewidth=1.1,
    )
    axis.scatter(*template_env.start[:3], marker="s", s=34, color="black")
    axis.scatter(*template_env.goal, marker="*", s=110, color="#f5c542", edgecolor="black")
    _style_3d(
        axis, template_env,
        r"Accepted center clouds: $d_0=0.75$, $\sigma_\parallel=0.30$, $\sigma_\perp=0.10$ m",
    )
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "accepted_center_clouds_3d.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "accepted_center_clouds_3d.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(16.2, 10.5))
    for scene_index, (fixed, spheres, records) in enumerate(rollout_bank):
        axis = fig.add_subplot(2, 3, scene_index + 1, projection="3d")
        status_code = {
            "SUCCESS": "S", "COLLISION": "C", "OOB": "O", "TIMEOUT": "T",
        }
        status = r"$\gamma$ .1/.3/.5/1: " + "/".join(
            status_code[records[gamma]["status"]] for gamma in GAMMAS
        )
        _plot_overlay(
            axis, TaskEnvironment(fixed), spheres, records,
            f"scene {scene_index + 1}\n{status}",
            float(config.raw["scene_randomization"]["physical_radius_m"]),
        )
    legend = [Line2D([0], [0], color=COLORS[g], lw=2, label=rf"$\gamma={g:g}$") for g in GAMMAS]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Raw pretrained policy: six accepted randomized scenes", fontsize=18, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output / "pretrained_gamma_overlay_6_scenes.png", dpi=210, bbox_inches="tight")
    fig.savefig(args.output / "pretrained_gamma_overlay_6_scenes.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11.8, 8.9))
    axis = fig.add_subplot(111, projection="3d")
    _plot_overlay(
        axis, TaskEnvironment(fixed), canonical, canonical_records,
        "Raw pretrained policy: canonical unjittered nominal triangle\n"
        "display-only: inflated lower bodies extend 7 mm below the geofence",
        float(config.raw["scene_randomization"]["physical_radius_m"]),
    )
    axis.legend(handles=legend, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "pretrained_gamma_overlay_canonical_triangle.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "pretrained_gamma_overlay_canonical_triangle.pdf", bbox_inches="tight")
    plt.close(fig)

    def row(scene_index, gamma, result):
        return {
            "scene": scene_index, "gamma": gamma,
            "status": result["status"],
            "min_clearance_m": result["min_clearance_m"],
            "time_to_goal_s": result["time_to_goal_s"],
            "window_validity": result["window_validity"],
        }
    metrics = [
        row(i + 1, gamma, records[gamma])
        for i, (_, _, records) in enumerate(rollout_bank)
        for gamma in GAMMAS
    ]
    canonical_metrics = [row("canonical", g, canonical_records[g]) for g in GAMMAS]
    (args.output / "metrics.json").write_text(json.dumps({
        "task_config": str(args.task_config.resolve()),
        "pretrain_dir": str(args.pretrain_dir.resolve()),
        "sampling_temperature": 1.0,
        "cloud_scene_count": args.cloud_scenes,
        "randomized_scene_metrics": metrics,
        "canonical_metrics": canonical_metrics,
    }, indent=2, allow_nan=False) + "\n")
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
