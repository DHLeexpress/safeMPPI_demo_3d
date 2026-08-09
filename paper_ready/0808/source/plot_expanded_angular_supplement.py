#!/usr/bin/env python3
"""Plot the gamma-1 octants and gamma-0.1 side-above frozen trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


SECTOR_COLORS = (
    "#6A3D9A", "#1F78B4", "#33A02C", "#B2DF8A",
    "#FDBF6F", "#FF7F00", "#E31A1C", "#B15928",
)


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def task_frame(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    forward = goal - start
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    up -= float(up @ forward) * forward
    up /= np.linalg.norm(up)
    lateral = np.cross(up, forward)
    lateral /= np.linalg.norm(lateral)
    return np.column_stack([forward, lateral, up])


def load(path: Path, center: np.ndarray, frame: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        dense = np.concatenate([
            np.asarray(data["states"][:1, :3], dtype=float),
            np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3),
        ])
        local = (dense - center[None]) @ frame
        return {
            "path": path,
            "local": local,
            "gamma": float(scalar(data, "gamma")),
            "seed": int(scalar(data, "seed")),
            "mode": str(scalar(data, "mode")),
            "theta_deg": float(scalar(data, "theta_deg")),
            "sector_8": int(scalar(data, "sector_8")),
        }


def crossing_point(local: np.ndarray) -> np.ndarray:
    along = local[:, 0]
    indices = np.flatnonzero((along[:-1] <= 0.0) & (along[1:] >= 0.0))
    if not len(indices):
        raise ValueError("trajectory has no obstacle-plane crossing")
    index = int(indices[0])
    delta = along[index + 1] - along[index]
    fraction = 0.0 if abs(delta) < 1.0e-12 else -along[index] / delta
    return local[index] + fraction * (local[index + 1] - local[index])


def setup_head_on(ax: plt.Axes, physical_radius: float, modeled_radius: float) -> None:
    ax.add_patch(Circle(
        (0.0, 0.0), modeled_radius, facecolor="#BDBDBD", alpha=0.16,
        edgecolor="#555555", linestyle="--", linewidth=1.2,
    ))
    ax.add_patch(Circle(
        (0.0, 0.0), physical_radius, facecolor="#888888", alpha=0.55,
        edgecolor="#333333", linewidth=1.0,
    ))
    for angle in np.arange(-180.0, 180.0, 45.0):
        radians = np.radians(angle)
        ax.plot(
            [0.0, np.cos(radians)], [0.0, np.sin(radians)],
            color="#B5B5B5", linestyle=(0, (3, 4)), linewidth=0.8, zorder=0,
        )
    ax.axhline(0.0, color="#8F8F8F", linewidth=0.55, zorder=0)
    ax.axvline(0.0, color="#8F8F8F", linewidth=0.55, zorder=0)
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.82, 0.82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("local left [m]")
    ax.set_ylabel("vertical [m]")
    ax.grid(color="#EEEEEE", linewidth=0.45)


def plot_record(ax: plt.Axes, row: dict[str, object], color: str, label: str) -> None:
    local = np.asarray(row["local"])
    visible = np.abs(local[:, 0]) <= 0.9
    ax.plot(local[visible, 1], local[visible, 2], color=color, linewidth=2.1, label=label)
    crossing = crossing_point(local)
    ax.scatter(
        crossing[1], crossing[2], marker="o", s=34, color=color,
        edgecolor="white", linewidth=0.65, zorder=5,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expanded", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config = json.loads((bundle / "config" / "task_config_resolved.json").read_text())
    start = np.asarray(config["taskspace"]["start"][:3], dtype=float)
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    sphere = np.asarray(config["obstacles"]["spheres"][0], dtype=float)
    frame = task_frame(start, goal)
    physical_radius = float(config["stage1"]["physical_sphere_radius_m"])
    modeled_radius = float(config["stage1"]["modeled_radius_m"])

    expanded_root = (
        args.expanded.resolve()
        if args.expanded is not None
        else bundle / "trajectories" / "expanded_quality_v2"
    )
    supplement_root = (
        args.supplement.resolve()
        if args.supplement is not None
        else bundle / "trajectories" / "expanded_supplement_v1"
    )
    existing = [
        load(path, sphere[:3], frame)
        for path in sorted(expanded_root.glob("*.npz"))
    ]
    supplement = [
        load(path, sphere[:3], frame)
        for path in sorted(supplement_root.glob("*.npz"))
    ]
    gamma1 = [
        row for row in existing + supplement
        if np.isclose(row["gamma"], 1.0, atol=1.0e-7, rtol=0.0)
    ]
    gamma1.sort(key=lambda row: int(row["sector_8"]))
    if [row["sector_8"] for row in gamma1] != list(range(8)):
        raise ValueError("gamma=1 trajectories do not occupy all eight sectors")

    straight_above = next(
        row for row in existing
        if np.isclose(row["gamma"], 0.1, atol=1.0e-7, rtol=0.0)
        and row["seed"] == 108992
    )
    side_above = sorted([
        row for row in supplement
        if np.isclose(row["gamma"], 0.1, atol=1.0e-7, rtol=0.0)
    ], key=lambda row: float(row["theta_deg"]))
    gamma01_above = [side_above[0], straight_above, side_above[1]]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 9.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), constrained_layout=True)
    for ax in axes:
        setup_head_on(ax, physical_radius, modeled_radius)

    for row in gamma1:
        sector = int(row["sector_8"])
        plot_record(
            axes[0], row, SECTOR_COLORS[sector],
            rf"S{sector}: {float(row['theta_deg']):.1f}$^\circ$ (seed {row['seed']})",
        )
    axes[0].set_title(r"Expanded $\gamma=1.0$: all eight sections")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    side_colors = ("#009E73", "#0072B2", "#D55E00")
    side_labels = ("left-side above", "straight above", "right-side above")
    for row, color, name in zip(gamma01_above, side_colors, side_labels):
        plot_record(
            axes[1], row, color,
            rf"{name}: {float(row['theta_deg']):.1f}$^\circ$ (seed {row['seed']})",
        )
    axes[1].set_title(r"Expanded $\gamma=0.1$: above alternatives")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)

    figure_dir = args.output.resolve() if args.output is not None else bundle / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "expanded_angular_supplement_headon.png"
    pdf = figure_dir / "expanded_angular_supplement_headon.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
