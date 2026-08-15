#!/usr/bin/env python3
"""Render paired-gamma SafeMPPI trajectories in randomized cylinder worlds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402


def _cylinder(axis, values, z_bounds, physical_radius):
    x, y, effective_radius = map(float, values)
    theta = np.linspace(0.0, 2.0 * np.pi, 30)
    z = np.linspace(float(z_bounds[0]), float(z_bounds[1]), 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    axis.plot_surface(
        x + effective_radius * np.cos(theta_grid),
        y + effective_radius * np.sin(theta_grid),
        z_grid,
        color="#78a9c4",
        alpha=0.16,
        linewidth=0.0,
    )
    axis.plot_surface(
        x + physical_radius * np.cos(theta_grid),
        y + physical_radius * np.sin(theta_grid),
        z_grid,
        color="#8d9295",
        alpha=0.72,
        linewidth=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=6)
    args = parser.parse_args()

    manifest = json.loads((args.demo_dir / "manifest.json").read_text())
    config = load_config(args.demo_dir / "resolved_config.json")
    physical_radius = float(
        config.raw["scene_randomization"]["physical_radius_m"]
    )
    scene_ids = [
        row["scene_id"]
        for row in manifest["scene_bank"]["scenes"][:args.scenes]
    ]
    rows = {
        (row["scene_id"], float(row["gamma"])): row
        for row in manifest["runs"]
        if row["scene_id"] in scene_ids
    }
    gammas = list(map(float, manifest["gammas"]))
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(gammas) - 1, 1)
        )
        for index, gamma in enumerate(gammas)
    }
    columns = 3
    row_count = int(np.ceil(len(scene_ids) / columns))
    fig = plt.figure(figsize=(5.2 * columns, 4.5 * row_count))
    for index, scene_id in enumerate(scene_ids):
        axis = fig.add_subplot(
            row_count, columns, index + 1, projection="3d",
        )
        representative = rows[(scene_id, gammas[0])]
        geometry = np.load(args.demo_dir / representative["file"])
        for cylinder in geometry["cylinders"]:
            _cylinder(
                axis,
                cylinder,
                config.taskspace.bounds[2],
                physical_radius,
            )
        for gamma in gammas:
            record = rows[(scene_id, gamma)]
            data = np.load(args.demo_dir / record["file"])
            trajectory = np.asarray(data["dense_positions"], float)
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=colors[gamma],
                linewidth=1.7,
                label=rf"$\gamma={gamma:g}$",
            )
        axis.scatter(
            *config.taskspace.start[:3], marker="s", s=28, color="black",
        )
        axis.scatter(
            *config.taskspace.goal, marker="*", s=90, color="#f1c40f",
            edgecolor="black",
        )
        axis.set(
            xlim=config.taskspace.bounds[0],
            ylim=config.taskspace.bounds[1],
            zlim=config.taskspace.bounds[2],
            xlabel=r"$x$ [m]",
            ylabel=r"$y$ [m]",
            zlabel=r"$z$ [m]",
            title=scene_id.replace("_", " "),
        )
        axis.view_init(elev=24, azim=-56)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.extend([
        Patch(
            facecolor="#8d9295",
            alpha=0.72,
            label="physical cylinder",
        ),
        Patch(
            facecolor="#78a9c4",
            alpha=0.16,
            label="inflated safety shell",
        ),
    ])
    labels.extend(["physical cylinder", "inflated safety shell"])
    fig.legend(
        handles, labels, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 0.962), frameon=False,
    )
    fig.suptitle(
        "Paired-gamma SafeMPPI demonstrations | randomized cylinders",
        fontsize=17,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[output] {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
