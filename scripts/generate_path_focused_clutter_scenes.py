#!/usr/bin/env python3
"""Preview the canonical path-focused cylinder and sphere scene banks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.path_focused_clutter import (  # noqa: E402
    PATH_FOCUSED_DISTRIBUTION,
    PATH_FOCUSED_MIDPOINT_UNIFORM_DISTRIBUTION,
    PathFocusedClutterSpec,
    path_focused_scene_bank,
)


BOUNDS = np.asarray(
    [[-2.5, 1.3], [-1.7, 1.8], [0.4, 2.0]], dtype=np.float64,
)
START = np.asarray([-2.1, 1.5, 0.9], dtype=np.float64)
GOAL = np.asarray([0.7, -1.5, 0.9], dtype=np.float64)
DEFAULT_CYLINDER_CONFIG = ROOT / "configs/lab_clutter_cylinders_path_v2.json"
DEFAULT_SPHERE_CONFIG = ROOT / "configs/lab_clutter_spheres_path_v2.json"


def _scene_payload(scene, spec: PathFocusedClutterSpec) -> dict:
    obstacles = scene.cylinders or scene.spheres
    return {
        "scene_index": int(scene.index),
        "scene_seed": int(scene.seed),
        "scene_hash": scene.scene_hash,
        "obstacle_family": spec.obstacle_family,
        "obstacle_count": len(obstacles),
        "physical_radius_m": spec.physical_radius_m,
        "vehicle_inflation_m": spec.vehicle_inflation_m,
        "modeled_radius_m": spec.modeled_radius_m,
        "minimum_obstacle_surface_gap_m": spec.minimum_surface_gap_m,
        "obstacles": [list(map(float, row)) for row in obstacles],
    }


def _sampling_payload(spec: PathFocusedClutterSpec) -> dict:
    payload = {
        "distribution": spec.distribution,
        "longitudinal_fraction": [
            spec.longitudinal_min,
            spec.longitudinal_max,
        ],
        "minimum_obstacle_surface_gap_m": spec.minimum_surface_gap_m,
        "minimum_taskspace_wall_surface_clearance_m": (
            spec.boundary_surface_gap_m
        ),
        "minimum_start_goal_surface_clearance_m": spec.endpoint_surface_gap_m,
    }
    if spec.distribution == PATH_FOCUSED_DISTRIBUTION:
        payload.update({
            "transverse_distribution": "normal",
            "transverse_std_m": spec.transverse_std_m,
        })
    elif spec.distribution == PATH_FOCUSED_MIDPOINT_UNIFORM_DISTRIBUTION:
        payload.update({
            "transverse_distribution": (
                "uniform_symmetric_with_longitudinal_halfwidth"
            ),
            "transverse_halfwidth_formula": (
                "scale_m * (0.5 - abs(lambda - 0.5))"
            ),
            "transverse_halfwidth_scale_m": (
                spec.transverse_halfwidth_scale_m
            ),
        })
    else:  # pragma: no cover - PathFocusedClutterSpec rejects this earlier.
        raise ValueError(f"unsupported preview distribution {spec.distribution!r}")
    if spec.obstacle_family == "spheres" and spec.sphere_z_mean_m is not None:
        payload.update({
            "sphere_z_distribution": "truncated_normal",
            "sphere_z_mean_m": spec.sphere_z_mean_m,
            "sphere_z_std_m": spec.sphere_z_std_m,
            "sphere_z_center_range_m": [
                spec.sphere_z_lower_m,
                spec.sphere_z_upper_m,
            ],
        })
    return payload


def _override_legacy_transverse_std(config, value: float):
    raw = copy.deepcopy(config.raw)
    randomization = raw["scene_randomization"]
    if randomization.get("distribution") != PATH_FOCUSED_DISTRIBUTION:
        raise ValueError(
            "--transverse-std-m only applies to the legacy "
            f"{PATH_FOCUSED_DISTRIBUTION} scene law"
        )
    randomization["transverse_std_m"] = float(value)
    return type(config)(
        config.taskspace,
        config.obstacles,
        config.safemppi,
        config.data,
        raw,
    )


def _plot_cylinder_scene(ax, scene: dict) -> None:
    physical = float(scene["physical_radius_m"])
    modeled = float(scene["modeled_radius_m"])
    for x, y, _ in scene["obstacles"]:
        ax.add_patch(Circle((x, y), modeled, fill=False, ls="--",
                            lw=1.2, color="#d95f02"))
        ax.add_patch(Circle((x, y), physical, color="0.35", alpha=0.8))
    ax.plot(
        [START[0], GOAL[0]], [START[1], GOAL[1]],
        "--", color="0.3", lw=1.2,
    )
    ax.scatter(*START[:2], marker="s", s=35, color="black", zorder=5)
    ax.scatter(*GOAL[:2], marker="*", s=90, color="#f1c40f",
               edgecolor="black", zorder=5)
    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(*BOUNDS[1])
    ax.set_aspect("equal")
    ax.set_title(f"Cylinders, $N={scene['obstacle_count']}$")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid(alpha=0.2)


def _sphere_wireframe(ax, center: np.ndarray, radius: float, **kwargs) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, **kwargs)


def _plot_sphere_scene(ax, scene: dict) -> None:
    physical = float(scene["physical_radius_m"])
    modeled = float(scene["modeled_radius_m"])
    for row in scene["obstacles"]:
        center = np.asarray(row[:3])
        _sphere_wireframe(
            ax, center, modeled, color="#d95f02", alpha=0.25, linewidth=0.5,
        )
        _sphere_wireframe(
            ax, center, physical, color="0.25", alpha=0.65, linewidth=0.7,
        )
    ax.plot(
        [START[0], GOAL[0]], [START[1], GOAL[1]], [START[2], GOAL[2]],
        "--", color="0.3", lw=1.2,
    )
    ax.scatter(*START, marker="s", s=30, color="black")
    ax.scatter(*GOAL, marker="*", s=80, color="#f1c40f", edgecolor="black")
    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(*BOUNDS[1])
    ax.set_zlim(*BOUNDS[2])
    ax.set_title(f"Spheres, $N={scene['obstacle_count']}$")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_zlabel("$z$ [m]")
    ax.view_init(elev=24, azim=-56)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/path_focused_clutter_preview"),
    )
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cylinder-config",
        type=Path,
        default=DEFAULT_CYLINDER_CONFIG,
        help="explicit cylinder scene config (legacy default is preserved)",
    )
    parser.add_argument(
        "--sphere-config",
        type=Path,
        default=DEFAULT_SPHERE_CONFIG,
        help="explicit sphere scene config (legacy default is preserved)",
    )
    parser.add_argument(
        "--transverse-std-m",
        type=float,
        default=None,
        help=(
            "preview-only override for both legacy truncated-normal configs; "
            "not valid for midpoint-uniform v2"
        ),
    )
    args = parser.parse_args()
    if args.scenes < 1:
        raise ValueError("--scenes must be positive")

    cylinder_config = load_config(args.cylinder_config)
    sphere_config = load_config(args.sphere_config)
    if args.transverse_std_m is not None:
        cylinder_config = _override_legacy_transverse_std(
            cylinder_config, args.transverse_std_m,
        )
        sphere_config = _override_legacy_transverse_std(
            sphere_config, args.transverse_std_m,
        )
    cylinder_spec = PathFocusedClutterSpec.from_config(cylinder_config)
    sphere_spec = PathFocusedClutterSpec.from_config(sphere_config)
    cylinders = [
        _scene_payload(scene, cylinder_spec)
        for scene in path_focused_scene_bank(
            cylinder_config, args.scenes, seed=args.seed,
        )
    ]
    spheres = [
        _scene_payload(scene, sphere_spec)
        for scene in path_focused_scene_bank(
            sphere_config, args.scenes, seed=args.seed,
        )
    ]
    sampling = _sampling_payload(cylinder_spec)
    sampling["sphere_sampling"] = _sampling_payload(sphere_spec)

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "path_focused_variable_clutter_preview_v2",
        "config_paths": {
            "cylinders": str(args.cylinder_config.resolve()),
            "spheres": str(args.sphere_config.resolve()),
        },
        "taskspace_bounds": BOUNDS.tolist(),
        "start": START.tolist(),
        "goal": GOAL.tolist(),
        "sampling": sampling,
        "cylinder_scenes": cylinders,
        "sphere_scenes": spheres,
    }
    (args.output / "scene_bank.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    shown = min(4, args.scenes)
    figure = plt.figure(figsize=(4.2 * shown, 8.2), constrained_layout=True)
    for column in range(shown):
        _plot_cylinder_scene(
            figure.add_subplot(2, shown, column + 1),
            cylinders[column],
        )
        _plot_sphere_scene(
            figure.add_subplot(2, shown, shown + column + 1, projection="3d"),
            spheres[column],
        )
    figure.suptitle(
        "Path-focused variable clutter | solid: physical, dashed/wire: modeled",
        fontsize=16,
    )
    figure.savefig(args.output / "scene_preview.png", dpi=180)
    figure.savefig(args.output / "scene_preview.pdf")
    plt.close(figure)

    print(args.output / "scene_bank.json")
    print(args.output / "scene_preview.png")
    print(args.output / "scene_preview.pdf")


if __name__ == "__main__":
    main()
