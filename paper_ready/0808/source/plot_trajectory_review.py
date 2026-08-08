#!/usr/bin/env python3
"""Create approval galleries for the 16/4+2 single-sphere trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


MODE_COLORS = {
    "below": "#D55E00",
    "above": "#0072B2",
    "left": "#009E73",
    "right": "#CC79A7",
}
PRETRAINED_COLOR = "#E69F00"
COLLISION_COLOR = "#B2182B"
START_COLOR = "#009E73"
GOAL_COLOR = "#0072B2"
GAMMAS = (0.1, 0.3, 0.5, 1.0)
MODES = ("below", "above", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--collisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def load_path(path: Path) -> tuple[np.lib.npyio.NpzFile, np.ndarray]:
    data = np.load(path)
    dense = np.concatenate([
        np.asarray(data["states"][:1, :3], dtype=float),
        np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3),
    ])
    return data, dense


def gamma_value(data: np.lib.npyio.NpzFile) -> float:
    raw = float(scalar(data, "gamma"))
    gamma = min(GAMMAS, key=lambda value: abs(value - raw))
    if abs(gamma - raw) > 1.0e-5:
        raise ValueError(f"unexpected gamma {raw:g}")
    return gamma


def projected_coordinates(
    dense: np.ndarray,
    center: np.ndarray,
    direction_xy: np.ndarray,
    left_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative_xy = dense[:, :2] - center[:2]
    along = relative_xy @ direction_xy
    lateral = relative_xy @ left_xy
    vertical = dense[:, 2] - center[2]
    return along, lateral, vertical


def first_plane_crossing(along: np.ndarray) -> int | None:
    indices = np.flatnonzero((along[:-1] <= 0.0) & (along[1:] > 0.0))
    return int(indices[0] + 1) if indices.size else None


def draw_panel(
    ax: plt.Axes,
    dense: np.ndarray,
    center: np.ndarray,
    radius: float,
    direction_xy: np.ndarray,
    left_xy: np.ndarray,
    color: str,
    collision: bool,
) -> None:
    along, lateral, vertical = projected_coordinates(
        dense, center, direction_xy, left_xy
    )
    local = np.abs(along) <= 0.9
    ax.add_patch(Circle(
        (0.0, 0.0), radius,
        facecolor="#E6E6E6", edgecolor="#333333", linewidth=0.9, zorder=0,
    ))
    ax.plot(lateral[local], vertical[local], color=color, linewidth=1.45, zorder=2)
    crossing = first_plane_crossing(along)
    if crossing is not None and not collision:
        ax.scatter(
            lateral[crossing], vertical[crossing], marker="*", s=50,
            color=color, edgecolor="white", linewidth=0.55, zorder=4,
        )
    if collision:
        distance = np.linalg.norm(dense - center[None, :], axis=1)
        contact = int(np.argmin(distance))
        ax.scatter(
            lateral[contact], vertical[contact], marker="X", s=43,
            color=COLLISION_COLOR, edgecolor="white", linewidth=0.6, zorder=5,
        )
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="#AAAAAA", linewidth=0.45, zorder=0)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.82, 0.82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-0.8, 0.0, 0.8))
    ax.set_yticks((-0.6, 0.0, 0.6))
    ax.grid(color="#EEEEEE", linewidth=0.45)


def draw_collision_side_panel(
    ax: plt.Axes,
    dense: np.ndarray,
    center: np.ndarray,
    radius: float,
    direction_xy: np.ndarray,
    left_xy: np.ndarray,
) -> None:
    along, _, vertical = projected_coordinates(dense, center, direction_xy, left_xy)
    local = np.abs(along) <= 0.9
    ax.add_patch(Circle(
        (0.0, 0.0), radius,
        facecolor="#E6E6E6", edgecolor="#333333", linewidth=0.9, zorder=0,
    ))
    ax.plot(along[local], vertical[local], color=COLLISION_COLOR, linewidth=1.6)
    distance = np.linalg.norm(dense - center[None, :], axis=1)
    contact = int(np.argmin(distance))
    ax.scatter(
        along[contact], vertical[contact], marker="X", s=43,
        color=COLLISION_COLOR, edgecolor="white", linewidth=0.6, zorder=5,
    )
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="#AAAAAA", linewidth=0.45, zorder=0)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.82, 0.82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-0.8, 0.0, 0.8))
    ax.set_yticks((-0.6, 0.0, 0.6))
    ax.grid(color="#EEEEEE", linewidth=0.45)


def draw_3d_panel(
    ax: plt.Axes,
    dense: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    radius: float,
    color: str,
    collision: bool,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 14)
    sphere_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        sphere_x, sphere_y, sphere_z,
        color="#BDBDBD", alpha=0.42, linewidth=0.0, shade=True, zorder=0,
    )
    ax.plot(
        dense[:, 0], dense[:, 1], dense[:, 2],
        color=color, linewidth=1.65, zorder=3,
    )
    ax.scatter(
        *start, marker="o", s=28, color=START_COLOR,
        edgecolor="white", linewidth=0.55, depthshade=False, zorder=6,
    )
    ax.scatter(
        *goal, marker="*", s=48, color=GOAL_COLOR,
        edgecolor="white", linewidth=0.55, depthshade=False, zorder=6,
    )
    ax.text(*start, "  S", color=START_COLOR, fontsize=7, weight="bold")
    ax.text(*goal, "  G", color=GOAL_COLOR, fontsize=7, weight="bold")
    if collision:
        distance = np.linalg.norm(dense - center[None, :], axis=1)
        contact = dense[int(np.argmin(distance))]
        ax.scatter(
            *contact, marker="X", s=34, color=COLLISION_COLOR,
            edgecolor="white", linewidth=0.55, depthshade=False, zorder=7,
        )
    ax.set_xlim(-2.3, 1.05)
    ax.set_ylim(-1.65, 1.65)
    ax.set_zlim(0.35, 1.85)
    ax.set_box_aspect((3.35, 3.3, 1.65))
    ax.set_xticks((-2.0, -0.7, 0.6))
    ax.set_yticks((-1.5, 0.0, 1.5))
    ax.set_zticks((0.4, 0.9, 1.4))
    ax.tick_params(labelsize=5.7, pad=-1)
    ax.set_xlabel("x", fontsize=6.5, labelpad=-5)
    ax.set_ylabel("y", fontsize=6.5, labelpad=-5)
    ax.set_zlabel("z", fontsize=6.5, labelpad=-4)
    ax.view_init(elev=24, azim=28)
    ax.grid(True, linewidth=0.35, alpha=0.45)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.04)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    start = np.asarray(config["taskspace"]["start"][:3], dtype=float)
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    sphere = np.asarray(config["obstacles"]["spheres"][0], dtype=float)
    center, radius = sphere[:3], float(sphere[3])
    direction_xy = goal[:2] - start[:2]
    direction_xy /= np.linalg.norm(direction_xy)
    left_xy = np.asarray([-direction_xy[1], direction_xy[0]])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.3,
        "axes.titlesize": 9.8,
        "axes.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })

    expanded_records = {}
    for path in args.expanded.glob("*.npz"):
        data, dense = load_path(path)
        gamma = gamma_value(data)
        mode = str(scalar(data, "mode"))
        if str(scalar(data, "status")) != "SUCCESS":
            raise SystemExit(f"expanded non-success: {path}")
        key = (gamma, mode)
        if key in expanded_records:
            raise SystemExit(f"duplicate expanded stratum: {key}")
        expanded_records[key] = (path, data, dense)
    expected_keys = {(gamma, mode) for gamma in GAMMAS for mode in MODES}
    if set(expanded_records) != expected_keys:
        raise SystemExit("expanded gallery is not exactly 4 gamma x 4 mode")

    fig, axes = plt.subplots(4, 4, figsize=(10.0, 8.55), sharex=True, sharey=True)
    for row, gamma in enumerate(GAMMAS):
        for column, mode in enumerate(MODES):
            ax = axes[row, column]
            _, data, dense = expanded_records[(gamma, mode)]
            draw_panel(
                ax, dense, center, radius, direction_xy, left_xy,
                MODE_COLORS[mode], collision=False,
            )
            theta = float(scalar(data, "theta_deg"))
            seed = int(scalar(data, "seed"))
            clearance = float(scalar(data, "min_clearance_m"))
            ax.set_title(
                rf"{mode}: seed {seed}" + "\n" +
                rf"$\theta={theta:.1f}^\circ$, clr. {clearance:.2f} m",
                color=MODE_COLORS[mode], pad=4,
            )
            if column == 0:
                ax.set_ylabel(rf"$\gamma={gamma:g}$" + "\nvertical (m)")
            if row == 3:
                ax.set_xlabel("local left (m)")
    fig.suptitle(
        "Expanded policy v1: 16 successful gamma–mode representatives",
        fontsize=14, y=0.995,
    )
    fig.text(
        0.5, 0.002,
        "Head-on projection near the sphere; stars mark first axial-plane crossing.",
        ha="center", fontsize=9.5, color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.965))
    args.output.mkdir(parents=True, exist_ok=True)
    expanded_out = args.output / "expanded_16_trajectory_review.png"
    fig.savefig(expanded_out, bbox_inches="tight")
    fig.savefig(expanded_out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11.6, 10.25))
    for index, (gamma, mode) in enumerate(
        ((gamma, mode) for gamma in GAMMAS for mode in MODES), start=1
    ):
        ax = fig.add_subplot(4, 4, index, projection="3d")
        _, data, dense = expanded_records[(gamma, mode)]
        draw_3d_panel(
            ax, dense, start, goal, center, radius,
            MODE_COLORS[mode], collision=False,
        )
        seed = int(scalar(data, "seed"))
        ax.set_title(
            rf"$\gamma={gamma:g}$  {mode}" + "\n" + rf"seed {seed}",
            color=MODE_COLORS[mode], fontsize=8.8, pad=-1,
        )
    fig.suptitle(
        "Expanded policy v1: full 3D trajectories (16 successful representatives)",
        fontsize=14, y=0.982,
    )
    fig.text(
        0.5, 0.015,
        "S = start, G = goal; all panels use the same physical axes and camera.",
        ha="center", fontsize=9.5, color="#444444",
    )
    fig.subplots_adjust(
        left=0.015, right=0.985, bottom=0.045, top=0.94,
        wspace=0.02, hspace=0.05,
    )
    expanded_3d_out = args.output / "expanded_16_trajectory_review_3d.png"
    fig.savefig(expanded_3d_out, bbox_inches="tight")
    fig.savefig(expanded_3d_out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    successes = []
    for path in args.pretrained.glob("*.npz"):
        data, dense = load_path(path)
        if str(scalar(data, "status")) != "SUCCESS":
            raise SystemExit(f"pretrained showcase non-success: {path}")
        successes.append((gamma_value(data), path, data, dense))
    successes.sort(key=lambda row: row[0])
    if len(successes) != 4 or [row[0] for row in successes] != list(GAMMAS):
        raise SystemExit("pretrained success gallery must contain one per gamma")

    collisions = []
    for path in args.collisions.glob("*.npz"):
        data, dense = load_path(path)
        if str(scalar(data, "status")) != "COLLISION":
            raise SystemExit(f"pretrained collision case did not collide: {path}")
        collisions.append((gamma_value(data), path, data, dense))
    collisions.sort(key=lambda row: row[0])
    if len(collisions) != 2:
        raise SystemExit("pretrained collision gallery must contain exactly two cases")

    entries = [("success", *row) for row in successes] + [
        ("collision", *row) for row in collisions
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 6.75), sharey=True)
    for ax, (kind, gamma, _, data, dense) in zip(axes.flat, entries):
        collision = kind == "collision"
        color = COLLISION_COLOR if collision else PRETRAINED_COLOR
        if collision:
            draw_collision_side_panel(
                ax, dense, center, radius, direction_xy, left_xy
            )
        else:
            draw_panel(
                ax, dense, center, radius, direction_xy, left_xy,
                color, collision=False,
            )
        seed = int(scalar(data, "seed"))
        clearance = float(scalar(data, "min_clearance_m"))
        if collision:
            ax.set_title(
                rf"COLLISION: $\gamma={gamma:g}$, seed {seed}" + "\n" +
                rf"penetration ${-clearance:.3f}$ m",
                color=COLLISION_COLOR, pad=4,
            )
        else:
            theta = float(scalar(data, "theta_deg"))
            ax.set_title(
                rf"SUCCESS: $\gamma={gamma:g}$, seed {seed}" + "\n" +
                rf"$\theta={theta:.1f}^\circ$, clr. {clearance:.2f} m",
                color=PRETRAINED_COLOR, pad=4,
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("vertical (m)")
    for ax, (kind, *_rest) in zip(axes.flat, entries):
        ax.set_xlabel("forward (m)" if kind == "collision" else "local left (m)")
    fig.suptitle(
        "P0806 pretrained policy: four showcase successes and two collisions",
        fontsize=14, y=0.995,
    )
    fig.text(
        0.5, 0.002,
        "Successes: head-on projection and first-crossing stars.  "
        "Collisions: longitudinal projection and deepest-point crosses.",
        ha="center", fontsize=9.5, color="#444444",
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.095, top=0.88,
        wspace=0.08, hspace=0.38,
    )
    pretrained_out = args.output / "pretrained_4_success_2_collision_review.png"
    fig.savefig(pretrained_out, bbox_inches="tight")
    fig.savefig(pretrained_out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10.1, 6.9))
    for index, (kind, gamma, _, data, dense) in enumerate(entries, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        collision = kind == "collision"
        color = COLLISION_COLOR if collision else PRETRAINED_COLOR
        draw_3d_panel(
            ax, dense, start, goal, center, radius, color, collision=collision
        )
        seed = int(scalar(data, "seed"))
        status = "COLLISION" if collision else "SUCCESS"
        ax.set_title(
            rf"{status}: $\gamma={gamma:g}$" + "\n" + rf"seed {seed}",
            color=color, fontsize=9.5, pad=0,
        )
    fig.suptitle(
        "P0806 pretrained policy: full 3D trajectories (4 success + 2 collision)",
        fontsize=14, y=0.98,
    )
    fig.text(
        0.5, 0.018,
        "S = start, G = goal; red crosses mark the deepest collision points.",
        ha="center", fontsize=9.5, color="#444444",
    )
    fig.subplots_adjust(
        left=0.015, right=0.985, bottom=0.06, top=0.91,
        wspace=0.02, hspace=0.04,
    )
    pretrained_3d_out = args.output / "pretrained_4_success_2_collision_review_3d.png"
    fig.savefig(pretrained_3d_out, bbox_inches="tight")
    fig.savefig(pretrained_3d_out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema": "paper_ready_trajectory_approval_gallery_v1",
        "status": "AWAITING_USER_APPROVAL",
        "counts": {
            "expanded_success": 16,
            "pretrained_success": 4,
            "pretrained_collision": 2,
            "total": 22,
        },
        "expanded": [
            {
                "gamma": gamma,
                "mode": mode,
                "file": expanded_records[(gamma, mode)][0].name,
                "seed": int(scalar(expanded_records[(gamma, mode)][1], "seed")),
            }
            for gamma in GAMMAS for mode in MODES
        ],
        "pretrained_success": [
            {"gamma": gamma, "file": path.name, "seed": int(scalar(data, "seed"))}
            for gamma, path, data, _ in successes
        ],
        "pretrained_collision": [
            {
                "gamma": gamma,
                "file": path.name,
                "seed": int(scalar(data, "seed")),
                "min_clearance_m": float(scalar(data, "min_clearance_m")),
            }
            for gamma, path, data, _ in collisions
        ],
    }
    (args.output / "trajectory_review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
