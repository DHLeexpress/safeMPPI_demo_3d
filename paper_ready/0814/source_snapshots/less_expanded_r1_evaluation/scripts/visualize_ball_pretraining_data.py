"""Render every trajectory in one ball-flow pretraining archive from multiple views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import ROUTE_MODES, route_mode
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.demo_dir / "manifest.json").read_text())
    config = load_config(args.demo_dir / "resolved_config.json")
    env = TaskEnvironment(config)
    gammas = sorted(float(gamma) for gamma in config.data.gammas)
    colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * i / max(len(gammas) - 1, 1))
        for i, gamma in enumerate(gammas)
    }
    trajectories = []
    counts = {(gamma, mode): 0 for gamma in gammas for mode in ROUTE_MODES}
    for row in manifest["runs"]:
        states = np.load(args.demo_dir / row["file"], allow_pickle=True)["states"][:, :3]
        gamma = float(row["gamma"])
        mode = route_mode(env, states)
        trajectories.append((gamma, mode, states))
        if mode in ROUTE_MODES:
            counts[(gamma, mode)] += 1

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 17, "axes.labelsize": 14})
    fig = plt.figure(figsize=(15.2, 10.8))
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.19)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax_side = fig.add_subplot(grid[0, 1])
    ax_head = fig.add_subplot(grid[1, 0])
    ax_count = fig.add_subplot(grid[1, 1])

    sphere = np.asarray(env.spheres[0], float)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 18)
    ax3.plot_surface(sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
                     sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
                     sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
                     color="#8f969f", alpha=0.42, linewidth=0)
    theta = np.linspace(0, 2 * np.pi, 160)
    ax_side.fill(sphere[0] + sphere[3] * np.cos(theta),
                 sphere[2] + sphere[3] * np.sin(theta), color="#8f969f", alpha=0.42)
    ax_head.fill(sphere[1] + sphere[3] * np.cos(theta),
                 sphere[2] + sphere[3] * np.sin(theta), color="#8f969f", alpha=0.42)
    ax_head.axhline(sphere[2], color="#c8321b", lw=0.9, ls="--")

    for gamma, _, states in trajectories:
        color = colors[gamma]
        ax3.plot(*states.T, color=color, lw=0.72, alpha=0.32)
        ax_side.plot(states[:, 0], states[:, 2], color=color, lw=0.72, alpha=0.32)
        ax_head.plot(states[:, 1], states[:, 2], color=color, lw=0.72, alpha=0.32)

    ax3.scatter(*env.start[:3], marker="s", color="#111111", s=28)
    ax3.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=120)
    ax3.set_box_aspect((3.0, 1.8, 1.8))
    ax3.view_init(elev=21, azim=-118)
    ax3.set_title("3-D view")
    ax3.set_xlabel(r"$x$ [m]")
    ax3.set_ylabel(r"$y$ [m]")
    ax3.set_zlabel(r"$z$ [m]")

    ax_side.scatter(env.start[0], env.start[2], marker="s", color="#111111", s=24)
    ax_side.scatter(env.goal[0], env.goal[2], marker="*", color="#ffca28",
                    edgecolor="#6a4e00", s=95)
    ax_side.set_title(r"Side view ($x$--$z$)")
    ax_side.set_xlabel(r"$x$ [m]")
    ax_side.set_ylabel(r"$z$ [m]")
    ax_side.set_aspect("equal", adjustable="box")

    ax_head.set_title(r"Head-on view at the ball ($y$--$z$)")
    ax_head.set_xlabel(r"$y$ [m] ($+y$ left)")
    ax_head.set_ylabel(r"$z$ [m]")
    ax_head.set_xlim(0.95, -0.95)
    ax_head.set_ylim(sphere[2] - 0.95, sphere[2] + 0.95)
    ax_head.set_aspect("equal", adjustable="box")

    values = np.asarray([[counts[(gamma, mode)] for mode in ROUTE_MODES]
                         for gamma in gammas])
    image = ax_count.imshow(values, cmap="Blues", vmin=0, vmax=max(1, int(values.max())))
    for i in range(len(gammas)):
        for j in range(len(ROUTE_MODES)):
            ax_count.text(j, i, str(int(values[i, j])), ha="center", va="center",
                          color="white" if values[i, j] > values.max() / 2 else "#111111",
                          fontsize=13)
    ax_count.set_xticks(range(len(ROUTE_MODES)), ROUTE_MODES)
    ax_count.set_yticks(range(len(gammas)), [rf"$\gamma={gamma:g}$" for gamma in gammas])
    ax_count.set_title("Demonstration route counts")
    fig.colorbar(image, ax=ax_count, fraction=0.046, pad=0.04, label="trajectories")

    for axis in (ax_side, ax_head, ax_count):
        axis.grid(alpha=0.20)
    handles = [plt.Line2D([0], [0], color=colors[gamma], lw=2.2,
                          label=rf"$\gamma={gamma:g}$") for gamma in gammas]
    fig.legend(handles=handles, ncol=len(gammas), loc="upper center", frameon=False)
    fig.suptitle(f"SafeMPPI pretraining archive: all {len(trajectories)} trajectories "
                 "(no trajectory subsampling)", fontsize=19, weight="bold", y=0.985)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[output] {args.output}")
    print("[counts]", {f"gamma={gamma:g}": {mode: counts[(gamma, mode)]
                                             for mode in ROUTE_MODES}
                       for gamma in gammas})


if __name__ == "__main__":
    main()
