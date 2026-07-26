"""Figures for the temporary canonical-policy to lab-frame bridge."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bridge import EndpointSimilarity


def _sphere(ax, sphere, color="#9aa0a6", alpha=0.42):
    center = np.asarray(sphere[:3], float)
    radius = float(sphere[3])
    u = np.linspace(0.0, 2.0 * np.pi, 38)
    v = np.linspace(0.0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _task(ax, start, goal, sphere, title, path=None):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    _sphere(ax, sphere)
    ax.scatter(*start, marker="s", s=48, color="black", zorder=5)
    ax.scatter(*goal, marker="*", s=115, color="#f2c94c",
               edgecolor="black", linewidth=0.6, zorder=5)
    ax.plot(*np.stack([start, goal]).T, color="#9aa0a6",
            linestyle="--", linewidth=1.2)
    if path is not None and len(path):
        path = np.asarray(path, float)
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color="#0077b6", linewidth=2.2)
    ax.set_title(title)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_zlabel("$z$ [m]")
    ax.view_init(elev=24, azim=-58)


def save_frame_bridge_figure(
    output_png: str | Path,
    frame: EndpointSimilarity,
    source_sphere: np.ndarray,
    target_sphere: np.ndarray,
    target_path: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Show the trained task, lab task, residual mismatch, and simulated path."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    actual_target_in_source = frame.target_sphere_to_source(target_sphere)
    mapped_path = (
        frame.target_to_source_position(target_path)
        if target_path is not None and len(target_path)
        else None
    )

    with plt.rc_context({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 13,
    }):
        fig = plt.figure(figsize=(13.2, 10.0), constrained_layout=True)
        axes = [fig.add_subplot(2, 2, index + 1, projection="3d")
                for index in range(4)]
        _task(
            axes[0], frame.source_start, frame.source_goal, source_sphere,
            "Canonical policy frame (training task)",
        )
        _task(
            axes[1], frame.target_start, frame.target_goal, target_sphere,
            "Minhyuk lab frame (deployment task)", path=target_path,
        )
        _task(
            axes[2], frame.source_start, frame.source_goal, source_sphere,
            "Lab task expressed in canonical coordinates", path=mapped_path,
        )
        _sphere(axes[2], actual_target_in_source, color="#e76f51", alpha=0.42)
        axes[2].text(
            actual_target_in_source[0],
            actual_target_in_source[1],
            actual_target_in_source[2] + actual_target_in_source[3],
            "lab sphere",
            color="#b23a2b",
        )
        _task(
            axes[3], frame.target_start, frame.target_goal, target_sphere,
            "Unchanged deploy_sim trajectory", path=target_path,
        )
        residual = actual_target_in_source[:3] - np.asarray(source_sphere[:3])
        fig.suptitle(
            "Temporary frozen-flow deployment bridge\n"
            f"endpoint scale={frame.scale:.4f}; lab-sphere residual in policy frame="
            f"({residual[0]:+.3f}, {residual[1]:+.3f}, {residual[2]:+.3f}) m",
            fontsize=15,
        )
        fig.savefig(output_png, dpi=180)
        output_pdf = output_png.with_suffix(".pdf")
        fig.savefig(output_pdf)
        plt.close(fig)
    return output_png, output_pdf
