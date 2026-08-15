"""Plot all replicas of selected expansion rounds in 3-D, side, and head-on views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import plan_states
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.expansion_visualize import (
    round_sigma_statistics,
    within_round_normalized_sigma,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, nargs="+", default=(1, 5, 10))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = TaskEnvironment(load_config(args.pretrain_dir / "demo_config.json"))
    recipe = json.loads((args.expansion / "manifest.json").read_text())["config"]
    events = torch.load(args.expansion / "events.pt", weights_only=False)
    events = [event for event in events
              if event["round"] in args.rounds
              and abs(float(event["gamma"]) - args.gamma) < 1.0e-7]
    sigma_statistics = round_sigma_statistics(events)
    normalized_color_scale = Normalize(0.0, 1.0, clip=True)
    cmap = plt.get_cmap("viridis")
    sphere = np.asarray(env.spheres[0], float)

    fig = plt.figure(figsize=(16.8, 4.9 * len(args.rounds)))
    grid = fig.add_gridspec(len(args.rounds), 3, wspace=0.18, hspace=0.22)
    for row_i, round_i in enumerate(args.rounds):
        round_sigma = sigma_statistics[round_i]

        def sigma_color(value):
            normalized = within_round_normalized_sigma(
                float(value), round_sigma,
            )
            return cmap(normalized_color_scale(normalized))

        grouped = {}
        for event in events:
            if event["round"] == round_i:
                grouped.setdefault(event["episode"], []).append(event)
        for rows in grouped.values():
            rows.sort(key=lambda event: event["step"])

        ax3 = fig.add_subplot(grid[row_i, 0], projection="3d")
        ax_side = fig.add_subplot(grid[row_i, 1])
        ax_head = fig.add_subplot(grid[row_i, 2])
        u = np.linspace(0, 2 * np.pi, 28)
        v = np.linspace(0, np.pi, 16)
        ax3.plot_surface(sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
                         sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
                         sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
                         color="#8f969f", alpha=0.45, linewidth=0)
        theta = np.linspace(0, 2 * np.pi, 120)
        ax_side.fill(sphere[0] + sphere[3] * np.cos(theta),
                     sphere[2] + sphere[3] * np.sin(theta),
                     color="#8f969f", alpha=0.45)
        ax_head.fill(sphere[1] + sphere[3] * np.cos(theta),
                     sphere[2] + sphere[3] * np.sin(theta),
                     color="#8f969f", alpha=0.45)

        success = nvp = 0
        for episode, rows in sorted(grouped.items()):
            segments, segment_colors = [], []
            for index, event in enumerate(rows):
                if event["chosen_local"] is None:
                    continue
                start = np.asarray(event["robot"][:3], float)
                if "robot_after" in event:
                    stop = np.asarray(event["robot_after"][:3], float)
                elif index + 1 < len(rows):
                    stop = np.asarray(rows[index + 1]["robot"][:3], float)
                else:
                    continue
                segments.append(np.stack([start, stop]))
                chosen = event["selected"][event["chosen_local"]]
                segment_colors.append(sigma_color(event["sigma_K"][chosen]))
            if segments:
                segments = np.asarray(segments)
                ax3.add_collection3d(Line3DCollection(
                    segments, colors=segment_colors, linewidths=2.0, alpha=0.90))
                ax_side.add_collection(LineCollection(
                    segments[:, :, [0, 2]], colors=segment_colors,
                    linewidths=2.0, alpha=0.90))
                ax_head.add_collection(LineCollection(
                    segments[:, :, [1, 2]], colors=segment_colors,
                    linewidths=2.0, alpha=0.90))

            terminal = rows[-1]["status"] or "TIMEOUT"
            terminal_point = np.asarray(
                rows[-1].get("robot_after", rows[-1]["robot"])[:3], float)
            marker = "o"
            marker_color = "#555555"
            if terminal == "NVP":
                nvp += 1
                marker, marker_color = "x", "#c8321b"
            elif terminal == "SUCCESS":
                success += 1
                marker, marker_color = "*", "#17964b"
            ax3.scatter(*terminal_point, marker=marker, color=marker_color, s=48)
            ax_side.scatter(terminal_point[0], terminal_point[2], marker=marker,
                            color=marker_color, s=48)
            ax_head.scatter(terminal_point[1], terminal_point[2], marker=marker,
                            color=marker_color, s=48)

            final = rows[-1]
            for candidate in final["selected"]:
                state6 = np.asarray(final["robot"][:6], float)
                path = plan_states(env, state6, final["candidates"][candidate])[:, :3]
                plan_color = sigma_color(final["sigma_K"][candidate])
                ax3.plot(*path.T, color=plan_color, lw=1.5, ls="--")
                ax_side.plot(path[:, 0], path[:, 2], color=plan_color, lw=1.5, ls="--")
                ax_head.plot(path[:, 1], path[:, 2], color=plan_color, lw=1.5, ls="--")

        ax3.scatter(*env.start[:3], marker="s", color="#111111", s=25)
        ax3.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=95)
        ax3.set_box_aspect((3.0, 1.8, 1.8))
        ax3.view_init(elev=20, azim=-118)
        timeout = len(grouped) - success - nvp
        total = len(grouped)
        ax3.set_title(f"round {round_i}: 3-D   success {success}/{total}, "
                      f"NVP {nvp}/{total}, timeout {timeout}/{total}\n"
                      r"raw $\sigma$ "
                      f"q02/med/q98={round_sigma['q02']:.3g}/"
                      f"{round_sigma['median']:.3g}/{round_sigma['q98']:.3g}")
        ax_side.set_title(r"side view ($x$--$z$)")
        ax_head.set_title(r"head-on view ($y$--$z$)")
        ax_side.set_xlim(-0.15, 3.2)
        ax_side.set_ylim(1.1, 2.9)
        ax_head.set_xlim(0.95, -0.95)
        ax_head.set_ylim(1.1, 2.9)
        ax_side.set_xlabel(r"$x$ [m]")
        ax_side.set_ylabel(r"$z$ [m]")
        ax_head.set_xlabel(r"$y$ [m] ($+y$ left)")
        ax_head.set_ylabel(r"$z$ [m]")
        for axis in (ax_side, ax_head):
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.20)

    scalar = plt.cm.ScalarMappable(norm=normalized_color_scale, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=fig.axes, fraction=0.018, pad=0.02)
    colorbar.set_label(
        r"within-round normalized $\widetilde{\sigma}_n(\phi_s)$"
    )
    fig.suptitle(rf"Fast arm: perturbation ${recipe['candidate_perturb_std']:g}\,"
                 rf"\mathrm{{m/s^2}}$, $B={recipe['B']}$ ({recipe['archive_rule']}), "
                 rf"$\gamma={args.gamma:g}$; all {recipe['parallel_episodes']} replicas per round",
                 fontsize=18, weight="bold", y=0.995)
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
