#!/usr/bin/env python3
"""Audit the untrained 3-D visual input on one sphere and one cylinder."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_visual_flow import (  # noqa: E402
    LAB_VISUAL_LOW_DIM,
    LabVisualFlowPolicy,
    build_visual_context,
    spherical_grid_points,
    spherical_safety_grid,
)


AUDIT_CYLINDER = (-1.15, 0.85, 0.22)
AUDIT_OBSERVER = (-1.55, 0.25, 0.9)


def obstacle_labels(points: np.ndarray, env: TaskEnvironment) -> tuple[np.ndarray, np.ndarray]:
    sphere = env.spheres[0]
    sphere_clearance = np.linalg.norm(points - sphere[:3], axis=1) - sphere[3]
    cylinder = env.cylinders[0]
    cylinder_clearance = (
        np.linalg.norm(points[:, :2] - cylinder[:2], axis=1) - cylinder[2]
    )
    return sphere_clearance, cylinder_clearance


def boundary_mask(occupancy: np.ndarray) -> np.ndarray:
    occupancy = occupancy.astype(bool)
    boundary = np.zeros_like(occupancy)
    for axis in range(3):
        for shift in (-1, 1):
            neighbor = np.roll(occupancy, shift, axis=axis)
            if axis != 0:
                index = [slice(None)] * 3
                index[axis] = 0 if shift == 1 else -1
                neighbor[tuple(index)] = False
            boundary |= occupancy & ~neighbor
    return boundary


def visual_token(policy: LabVisualFlowPolicy, context: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        encoded = policy.encode_context(torch.from_numpy(context))
    return encoded[LAB_VISUAL_LOW_DIM:].numpy()


def add_sphere(axis, sphere, *, alpha=0.22):
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 20)
    axis.plot_surface(
        sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
        sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
        sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
        color="#8a8a8a",
        alpha=alpha,
        linewidth=0,
    )


def add_cylinder(axis, cylinder, z_bounds, *, alpha=0.18):
    angle = np.linspace(0.0, 2.0 * np.pi, 48)
    z = np.asarray(z_bounds)
    x = cylinder[0] + cylinder[2] * np.cos(angle)[:, None]
    y = cylinder[1] + cylinder[2] * np.sin(angle)[:, None]
    axis.plot_surface(
        np.broadcast_to(x, (len(angle), 2)),
        np.broadcast_to(y, (len(angle), 2)),
        np.broadcast_to(z[None], (len(angle), 2)),
        color="#b0b0b0",
        alpha=alpha,
        linewidth=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_ball_pretrain.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    audit_config = replace(
        config,
        obstacles=replace(
            config.obstacles,
            cylinders=(AUDIT_CYLINDER,),
        ),
    )
    env = TaskEnvironment(audit_config)
    observer = np.asarray(AUDIT_OBSERVER, np.float32)
    grid = spherical_safety_grid(env, observer)
    points_grid = spherical_grid_points(
        observer,
        env.mppi.sensing_range,
    )
    points = points_grid.reshape(-1, 3)
    occupancy = grid[0].astype(bool)
    occupied_points = points[occupancy.reshape(-1)]
    sphere_clearance, cylinder_clearance = obstacle_labels(points, env)
    expected_occupancy = (sphere_clearance < 0.0) | (cylinder_clearance < 0.0)
    union_match = float(np.mean(occupancy.reshape(-1) == expected_occupancy))

    sphere_owned = occupancy.reshape(-1) & (
        sphere_clearance <= cylinder_clearance
    )
    cylinder_owned = occupancy.reshape(-1) & (
        cylinder_clearance < sphere_clearance
    )
    boundary = boundary_mask(occupancy)
    boundary_clearance = np.abs(
        np.minimum(sphere_clearance, cylinder_clearance)[boundary.reshape(-1)]
    )
    hp_consistency = float(np.mean(
        grid[1].astype(bool) == (grid[2] >= 0.0)
    ))

    torch.manual_seed(0)
    policy = LabVisualFlowPolicy(
        hidden=48,
        representation_dim=32,
        grid_token_dim=32,
        control_limit=config.safemppi.demo_u_max,
        nfe=16,
    )
    state = np.concatenate([observer, np.zeros(3, np.float32)])
    contexts = []
    names = ("empty", "sphere", "cylinder", "combined")
    for include_sphere, include_cylinder in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        local_config = replace(
            audit_config,
            obstacles=replace(
                audit_config.obstacles,
                spheres=(
                    (tuple(map(float, env.spheres[0])),)
                    if include_sphere else ()
                ),
                cylinders=(
                    (AUDIT_CYLINDER,) if include_cylinder else ()
                ),
            ),
        )
        contexts.append(build_visual_context(
            TaskEnvironment(local_config),
            state,
            0.3,
        ))
    tokens = np.stack([
        visual_token(policy, context) for context in contexts
    ])
    token_distance = np.linalg.norm(
        tokens[:, None] - tokens[None, :],
        axis=2,
    )

    fig = plt.figure(figsize=(14.5, 8.6))
    axis3d = fig.add_subplot(2, 3, 1, projection="3d")
    add_sphere(axis3d, env.spheres[0])
    add_cylinder(
        axis3d,
        env.cylinders[0],
        (
            observer[2] - env.mppi.sensing_range,
            observer[2] + env.mppi.sensing_range,
        ),
    )
    axis3d.scatter(
        occupied_points[:, 0],
        occupied_points[:, 1],
        occupied_points[:, 2],
        c=np.where(sphere_owned[occupancy.reshape(-1)], "#3b82f6", "#f97316"),
        s=15,
        alpha=0.9,
    )
    axis3d.scatter(*observer, marker="x", s=65, color="black")
    axis3d.set(
        title="Analytic obstacles + occupied grid cells",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )

    top = fig.add_subplot(2, 3, 2)
    top.scatter(
        occupied_points[:, 0],
        occupied_points[:, 1],
        c=np.where(sphere_owned[occupancy.reshape(-1)], "#3b82f6", "#f97316"),
        s=18,
        alpha=0.85,
    )
    for center, radius, color in (
        (env.spheres[0][:2], env.spheres[0][3], "#2563eb"),
        (env.cylinders[0][:2], env.cylinders[0][2], "#ea580c"),
    ):
        top.add_patch(plt.Circle(center, radius, fill=False, color=color, lw=2))
    top.scatter(*observer[:2], marker="x", color="black", s=65)
    top.set_aspect("equal")
    top.set(title="Top-view reconstruction", xlabel="x [m]", ylabel="y [m]")
    top.grid(alpha=0.2)

    side = fig.add_subplot(2, 3, 3)
    side.scatter(
        occupied_points[:, 0],
        occupied_points[:, 2],
        c=np.where(sphere_owned[occupancy.reshape(-1)], "#3b82f6", "#f97316"),
        s=18,
        alpha=0.85,
    )
    sphere = env.spheres[0]
    side.add_patch(plt.Circle(
        (sphere[0], sphere[2]),
        sphere[3],
        fill=False,
        color="#2563eb",
        lw=2,
    ))
    cylinder = env.cylinders[0]
    for x in (cylinder[0] - cylinder[2], cylinder[0] + cylinder[2]):
        side.axvline(x, color="#ea580c", linewidth=2, alpha=0.8)
    side.scatter(observer[0], observer[2], marker="x", color="black", s=65)
    side.set_aspect("equal")
    side.set(title="Side-view reconstruction", xlabel="x [m]", ylabel="z [m]")
    side.grid(alpha=0.2)

    hp = fig.add_subplot(2, 3, 4)
    hp_values = grid[2].reshape(-1)
    scatter = hp.scatter(
        points[:, 0],
        points[:, 1],
        c=hp_values,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        s=8,
        alpha=0.6,
    )
    hp.set_aspect("equal")
    hp.set(title=r"Clipped nominal $H_P$", xlabel="x [m]", ylabel="y [m]")
    fig.colorbar(scatter, ax=hp, fraction=0.046)

    counts = fig.add_subplot(2, 3, 5)
    values = [
        int(sphere_owned.sum()),
        int(cylinder_owned.sum()),
        int(boundary.sum()),
    ]
    counts.bar(
        ("sphere cells", "cylinder cells", "boundary cells"),
        values,
        color=("#3b82f6", "#f97316", "#64748b"),
    )
    counts.set_title("Grid support")
    counts.tick_params(axis="x", rotation=18)
    counts.grid(alpha=0.2, axis="y")

    token = fig.add_subplot(2, 3, 6)
    image = token.imshow(token_distance, cmap="magma")
    token.set_xticks(range(4), names, rotation=25)
    token.set_yticks(range(4), names)
    token.set_title("Untrained encoder token distance")
    fig.colorbar(image, ax=token, fraction=0.046)

    fig.suptitle(
        "3-D visual input audit (geometry reconstruction before training)",
        fontsize=15,
    )
    fig.tight_layout()
    png = args.output / "visual_encoder_reconstruction.png"
    pdf = args.output / "visual_encoder_reconstruction.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "LAB_VISUAL_INPUT_AUDIT_COMPLETE",
        "scope": (
            "Deterministic input-grid reconstruction and pathway sensitivity; "
            "not a learned decoder or trained-policy qualification."
        ),
        "observer": observer.tolist(),
        "sphere": env.spheres[0].tolist(),
        "cylinder": env.cylinders[0].tolist(),
        "grid_shape": list(grid.shape),
        "occupied_cells": int(occupancy.sum()),
        "sphere_cells": int(sphere_owned.sum()),
        "cylinder_cells": int(cylinder_owned.sum()),
        "occupancy_union_accuracy": union_match,
        "hp_mask_consistency": hp_consistency,
        "boundary_abs_clearance_median_m": float(np.median(boundary_clearance)),
        "boundary_abs_clearance_p95_m": float(np.quantile(boundary_clearance, 0.95)),
        "token_names": list(names),
        "untrained_token_distance": token_distance.tolist(),
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
    }
    (args.output / "visual_encoder_reconstruction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
