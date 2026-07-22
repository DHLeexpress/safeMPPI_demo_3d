"""Per-gamma rollout GIFs for the ball-below task.

Each GIF animates all seeds of one gamma: growing trajectory trails in 3D while the BLUE nominal
polytope of a representative seed evolves step by step. The polytope interior is a translucent
fill so the ten H_P level-set lines stay readable, and the camera orbits so the polytope is seen
from different angles. A second panel shows the head-on view from the start toward the goal,
where the below-latitude crossing fan builds up.

Usage: python -m safe_mppi.ball_gif --run <output_dir> [--fps 11]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import numpy as np

from .ball_analysis import _ball, _draw_latitude0, _load, draw_polytope_soft
from .visualize import PLASMA, _draw_box, _draw_obstacles, _poly_from_saved, _style


def _episode(env, data):
    states = np.asarray(data["states"], float)
    controls = np.asarray(data["controls"], float)
    dense = env.dense_positions(states, controls)
    return states, controls, dense


def _passage_angle(dense, center):
    crossing = int(np.argmin(np.abs(dense[:, 0] - center[0])))
    dy, dz = dense[crossing, 1] - center[1], dense[crossing, 2] - center[2]
    return float(np.degrees(np.arctan2(-dz, -dy)))


def make_gamma_gif(env, gamma, runs, output_dir, fps=11):
    center, radius = _ball(env)
    episodes = [(row, *_episode(env, data), data) for row, data in runs]
    angles = [_passage_angle(dense, center) for _, _, _, dense, _ in episodes]
    highlight = int(np.argsort(angles)[len(angles) // 2])
    max_steps = max(len(controls) for _, _, controls, _, _ in episodes)
    stride = max(1, int(np.ceil(max_steps / 36)))
    motion_frames = int(np.ceil(max_steps / stride)) + 1
    n_frames = motion_frames + 9
    color = PLASMA(Normalize(0.0, 1.0)(gamma))

    fig = plt.figure(figsize=(10.6, 4.7), facecolor="white")
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    def draw(frame):
        step = min(frame, motion_frames - 1) * stride
        ax3d.cla()
        ax2d.cla()
        _draw_box(ax3d, env.bounds, alpha=0.22, linewidth=0.5)
        _draw_obstacles(ax3d, env, alpha=0.26)
        _draw_latitude0(ax3d, center, radius)
        for index, (row, states, controls, dense, _) in enumerate(episodes):
            k = min(step, len(controls))
            trail = dense[:1 + 10 * k]
            is_highlight = index == highlight
            ax3d.plot(*trail.T, color=color, lw=2.1 if is_highlight else 1.2,
                      alpha=0.95 if is_highlight else 0.45)
            ax3d.scatter(*states[min(k, len(states) - 1), :3],
                         color=color, edgecolor="#222222", linewidth=0.5,
                         s=44 if is_highlight else 20, depthshade=False)
            ax2d.plot(trail[:, 1], trail[:, 2], color=color,
                      lw=1.9 if is_highlight else 1.0,
                      alpha=0.9 if is_highlight else 0.4)
            ax2d.scatter(states[min(k, len(states) - 1), 1],
                         states[min(k, len(states) - 1), 2], color=color,
                         edgecolor="#222222", linewidth=0.5, s=34 if is_highlight else 16,
                         zorder=6)
        _, _, hl_controls, _, hl_data = episodes[highlight]
        poly_index = min(min(step, len(hl_controls)), len(hl_data["poly_A"]) - 1)
        draw_polytope_soft(ax3d, _poly_from_saved(hl_data, poly_index), gamma)
        ax3d.scatter(*env.start[:3], marker="s", color="#111111", s=40)
        ax3d.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=170)
        _style(ax3d, env)
        progress = frame / max(n_frames - 1, 1)
        ax3d.view_init(elev=21.0 + 6.0 * np.sin(2.0 * np.pi * progress),
                       azim=-110.0 + 100.0 * progress)
        ax3d.set_title(rf"$\gamma={gamma:g}$ — t={step * env.mppi.dt:.1f}s, BLUE $P_k$"
                       " + $H_P$ levels (orbiting view)", fontsize=10, weight="bold")

        theta = np.linspace(0.0, 2.0 * np.pi, 160)
        ax2d.fill(center[1] + radius * np.cos(theta), center[2] + radius * np.sin(theta),
                  color="#8f969f", alpha=0.45, zorder=2)
        ax2d.axhline(center[2], color="#cc3311", lw=1.0, ls="--", zorder=3)
        ax2d.scatter(0.0, center[2], marker="*", color="#ffca28", edgecolor="#6a4e00",
                     s=150, zorder=5)
        ax2d.set_xlim(1.15, -1.15)
        ax2d.set_ylim(center[2] - 1.05, center[2] + 0.55)
        ax2d.set_aspect("equal")
        ax2d.grid(alpha=0.22)
        ax2d.set_xlabel("y [m]  (+y left: viewed from start)")
        ax2d.set_ylabel("z [m]")
        ax2d.set_title("head-on: fan below latitude 0", fontsize=10)
        return []

    draw(int(0.55 * motion_frames))
    preview = Path(output_dir) / f"ball_evolve_g{gamma:g}_preview.png"
    fig.savefig(preview, dpi=92, bbox_inches="tight")
    animation = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    out = Path(output_dir) / f"ball_evolve_g{gamma:g}.gif"
    animation.save(out, writer=PillowWriter(fps=fps), dpi=92)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="finished run.py output directory")
    parser.add_argument("--fps", type=int, default=11)
    args = parser.parse_args()
    run_dir = Path(args.run).resolve()
    manifest, config, env, runs = _load(run_dir)
    by_gamma = {}
    for row, data in runs:
        by_gamma.setdefault(float(row["gamma"]), []).append((row, data))
    for gamma in sorted(by_gamma):
        out = make_gamma_gif(env, gamma, by_gamma[gamma], run_dir, fps=args.fps)
        print(f"[gif] {out}", flush=True)


if __name__ == "__main__":
    main()
