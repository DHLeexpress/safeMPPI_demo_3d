#!/usr/bin/env python3
"""Plot the SafeMPPI seed-bank support and four frozen representatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MODE_COLORS = {
    "below": "#7B2CBF",
    "above": "#2A9D8F",
    "left": "#E76F51",
    "right": "#3A86FF",
    "failure": "#D8D8D8",
}
GAMMA_COLORS = {
    0.1: "#3A86FF",
    0.3: "#2A9D8F",
    0.5: "#F4A261",
    1.0: "#D62828",
}


def draw_sphere(ax, center: np.ndarray, radius: float, **kwargs) -> None:
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    out = bundle / "safemppi" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    screen = json.loads(
        (bundle / "safemppi" / "mode_screen" / "summary.json").read_text()
    )
    selection = json.loads(
        (bundle / "safemppi" / "selection.json").read_text()
    )
    config = json.loads(
        (bundle / "config" / "task_config_resolved.json").read_text()
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
    })
    fig = plt.figure(figsize=(13.2, 5.8), constrained_layout=True)
    ax_count = fig.add_subplot(1, 2, 1)
    ax_traj = fig.add_subplot(1, 2, 2, projection="3d")

    gammas = [float(item["gamma"]) for item in screen["per_gamma"]]
    bottoms = np.zeros(len(gammas))
    for mode in ("below", "above", "left", "right"):
        values = np.array([
            item["successful_mode_counts"].get(mode, 0)
            for item in screen["per_gamma"]
        ])
        ax_count.bar(
            np.arange(len(gammas)), values, bottom=bottoms,
            color=MODE_COLORS[mode], edgecolor="white", linewidth=0.8, label=mode,
        )
        bottoms += values
    failures = np.array([
        item["attempts"] - sum(item["successful_mode_counts"].values())
        for item in screen["per_gamma"]
    ])
    ax_count.bar(
        np.arange(len(gammas)), failures, bottom=bottoms,
        color=MODE_COLORS["failure"], edgecolor="white", linewidth=0.8,
        label="collision",
    )
    for index, successes in enumerate(bottoms.astype(int)):
        ax_count.text(index, 66, f"{successes}/64 success", ha="center", va="bottom")
    ax_count.set_xticks(np.arange(len(gammas)), [f"{g:g}" for g in gammas])
    ax_count.set_ylim(0, 72)
    ax_count.set_xlabel(r"Safety level $\gamma$")
    ax_count.set_ylabel("Seed count")
    ax_count.set_title("Finite-seed SafeMPPI support")
    ax_count.grid(axis="y", alpha=0.2)
    ax_count.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)

    start = np.asarray(config["taskspace"]["start"][:3], dtype=float)
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    sphere = np.asarray(config["obstacles"]["spheres"][0], dtype=float)
    physical_radius = float(config["stage1"]["physical_sphere_radius_m"])
    draw_sphere(ax_traj, sphere[:3], sphere[3], color="#B0B0B0", alpha=0.10)
    draw_sphere(ax_traj, sphere[:3], physical_radius, color="#666666", alpha=0.44)

    handles: list[Line2D] = []
    for item in selection["representatives"]:
        gamma = float(item["gamma"])
        source = bundle / item["recording"]
        with np.load(source, allow_pickle=True) as archive:
            path = np.asarray(archive["dense_positions"], dtype=float)
        color = GAMMA_COLORS[gamma]
        ax_traj.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.5)
        handles.append(Line2D(
            [0], [0], color=color, linewidth=2.5,
            label=rf"$\gamma={gamma:g}$  {item['mode']}  (seed {item['seed']})",
        ))
    ax_traj.scatter(*start, marker="s", s=55, color="black", depthshade=False)
    ax_traj.scatter(
        *goal, marker="*", s=150, color="#F6C945", edgecolor="#6B5A00",
        linewidth=0.8, depthshade=False,
    )
    ax_traj.set_xlabel(r"$x$ [m]", labelpad=8)
    ax_traj.set_ylabel(r"$y$ [m]", labelpad=8)
    ax_traj.set_zlabel(r"$z$ [m]", labelpad=8)
    ax_traj.set_title("Frozen qualitative representatives")
    ax_traj.view_init(elev=24, azim=-53)
    ax_traj.set_box_aspect((1.35, 1.35, 0.7))
    ax_traj.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.04), frameon=False)

    png = out / "safemppi_support_and_representatives.png"
    pdf = out / "safemppi_support_and_representatives.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
