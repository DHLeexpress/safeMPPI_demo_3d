"""Preview the Stage-2 tight-triangle scene and its accepted perturbation bank."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_clutter_expansion import (
    PerpendicularTriangleSphereScene,
    scene_sha256,
    sphere_scene_spec_from_config,
)


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def _assign_to_base(local_centers: np.ndarray, base_local: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        local_centers[:, None, 1:3] - base_local[None, :, 1:3], axis=2,
    )
    permutations = tuple(itertools.permutations(range(3)))
    costs = [
        sum(distances[row, identity] for row, identity in enumerate(order))
        for order in permutations
    ]
    return np.asarray(permutations[int(np.argmin(costs))], dtype=int)


def _draw_sphere(axis, center, radius, *, color, alpha, wire=False):
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    if wire:
        axis.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.45)
    else:
        axis.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _ellipse(axis, center, std_x, std_y, color):
    axis.add_patch(Ellipse(
        center,
        width=4.0 * std_x,
        height=4.0 * std_y,
        fill=False,
        linestyle="--",
        linewidth=1.25,
        edgecolor=color,
        alpha=0.9,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-config", type=Path,
        default=ROOT / "configs/lab_clutter_spheres_stage2_triangle15_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            ROOT
            / "results/stage2_sphere_clutter_triangle15/scene_preview_s070_p030_010"
        ),
    )
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=91_000)
    args = parser.parse_args()
    if args.scenes < 1:
        parser.error("--scenes must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    config = load_config(args.task_config)
    env = TaskEnvironment(config)
    spec = sphere_scene_spec_from_config(config)
    if not isinstance(spec, PerpendicularTriangleSphereScene):
        parser.error("preview requires perpendicular_triangle_anisotropic_v1")

    frame = spec.local_frame(env)
    midpoint = (
        np.asarray(env.start[:3], float)
        + spec.center_fraction
        * (np.asarray(env.goal, float) - np.asarray(env.start[:3], float))
    )
    base = spec.base_scene(env)
    base_local = (base[:, :3] - midpoint[None]) @ frame
    scenes = []
    local_by_identity = [[] for _ in range(3)]
    for episode in range(args.scenes):
        scene_seed = int(args.seed) + 1009 * episode
        spheres = spec.sample(env, scene_seed)
        local = (spheres[:, :3] - midpoint[None]) @ frame
        identities = _assign_to_base(local, base_local)
        for row, identity in zip(local, identities):
            local_by_identity[int(identity)].append(row)
        scenes.append({
            "episode": episode,
            "scene_seed": scene_seed,
            "scene_hash": scene_sha256(env, spheres),
            "spheres": spheres.tolist(),
        })

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
    })
    fig = plt.figure(figsize=(15.5, 11.5))
    scene_axis = fig.add_subplot(2, 2, 1, projection="3d")
    for index, sphere in enumerate(base):
        _draw_sphere(
            scene_axis, sphere[:3], spec.radius,
            color=COLORS[index], alpha=0.11, wire=True,
        )
        _draw_sphere(
            scene_axis, sphere[:3],
            config.raw["scene_randomization"]["physical_radius_m"],
            color=COLORS[index], alpha=0.38,
        )
    scene_axis.plot(
        [env.start[0], env.goal[0]],
        [env.start[1], env.goal[1]],
        [env.start[2], env.goal[2]],
        color="black", linestyle="--", linewidth=1.1,
    )
    scene_axis.scatter(*env.start[:3], marker="s", s=35, color="black")
    scene_axis.scatter(
        *env.goal, marker="*", s=110, color="#f1c40f", edgecolor="black",
    )
    scene_axis.set(
        xlim=env.bounds[0], ylim=env.bounds[1], zlim=env.bounds[2],
        xlabel="$x$ [m]", ylabel="$y$ [m]", zlabel="$z$ [m]",
        title="Fixed tight-triangle challenge",
    )
    scene_axis.set_box_aspect(tuple(env.bounds[:, 1] - env.bounds[:, 0]))

    panels = (
        (fig.add_subplot(2, 2, 2), 0, 1,
         r"longitudinal $e_\parallel$ [m]", r"lateral $e_1$ [m]"),
        (fig.add_subplot(2, 2, 3), 0, 2,
         r"longitudinal $e_\parallel$ [m]", r"vertical $e_2$ [m]"),
        (fig.add_subplot(2, 2, 4), 1, 2,
         r"lateral $e_1$ [m]", r"vertical $e_2$ [m]"),
    )
    for axis, first, second, xlabel, ylabel in panels:
        for identity, values in enumerate(local_by_identity):
            values = np.asarray(values, float)
            axis.scatter(
                values[:, first], values[:, second],
                s=15, alpha=0.27, color=COLORS[identity],
            )
            axis.scatter(
                base_local[identity, first], base_local[identity, second],
                marker="x", s=75, linewidths=2.0, color=COLORS[identity],
            )
            std = (
                spec.parallel_std_m if first == 0 else spec.plane_std_m,
                spec.parallel_std_m if second == 0 else spec.plane_std_m,
            )
            _ellipse(
                axis,
                (base_local[identity, first], base_local[identity, second]),
                std[0], std[1], COLORS[identity],
            )
            if first != 0 and second != 0:
                axis.add_patch(Circle(
                    (base_local[identity, first], base_local[identity, second]),
                    spec.radius,
                    fill=False, linestyle=":", linewidth=0.9,
                    edgecolor=COLORS[identity], alpha=0.5,
                ))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.set_aspect("equal", adjustable="datalim")
    panels[0][0].set_title(
        "Accepted centers; dashed ellipses = proposal $2\\sigma$\n"
        + rf"$\sigma_\parallel={spec.parallel_std_m:g}$ m, "
        + rf"$\sigma_\perp={spec.plane_std_m:g}$ m"
    )
    panels[1][0].set_title("Longitudinal–vertical perturbation")
    panels[2][0].set_title(
        "Perpendicular triangle plane\n"
        "dotted circles = robot-inflated bodies"
    )
    fig.suptitle(
        "Stage-2 15-inch sphere geometry before expansion",
        fontsize=19,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    args.output.mkdir(parents=True, exist_ok=False)
    png = args.output / "stage2_scene_distribution_preview.png"
    pdf = args.output / "stage2_scene_distribution_preview.pdf"
    fig.savefig(png, dpi=210, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    (args.output / "scene_bank.json").write_text(json.dumps({
        "status": "STAGE2_SCENE_PREVIEW_COMPLETE",
        "task_config": str(args.task_config.resolve()),
        "scene_count": int(args.scenes),
        "seed": int(args.seed),
        "physical_radius_m": float(
            config.raw["scene_randomization"]["physical_radius_m"]
        ),
        "robot_inflation_m": float(
            config.raw["scene_randomization"]["vehicle_inflation_m"]
        ),
        "modeled_radius_m": float(spec.radius),
        "base_modeled_surface_gap_m": float(
            spec.center_spacing_m - 2.0 * spec.radius
        ),
        "base_center_clearance_m": float(
            spec.center_spacing_m / np.sqrt(3.0) - spec.radius
        ),
        "sampling_contract": spec.sampling_contract(),
        "fixed_challenge_spheres": base.tolist(),
        "scenes": scenes,
    }, indent=2, allow_nan=False) + "\n")
    print(f"[preview] {png}")
    print(f"[preview] {pdf}")
    print(f"[scene bank] {args.output / 'scene_bank.json'}")


if __name__ == "__main__":
    main()
