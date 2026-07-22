"""Reusable result, gallery, and mechanism-video skeletons for 3D expansion tasks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
import numpy as np


@dataclass(frozen=True)
class MechanismFrame:
    round: int
    gamma: float
    robot: np.ndarray
    candidate_paths: np.ndarray       # [K,H+1,3], supplied by the task adapter.
    sigma: np.ndarray                 # [K]
    selected: tuple[int, ...]         # B indices in the K pool.
    positive: tuple[int, ...]         # selected indices passing the full verifier.
    executed: int | None
    nvp: bool
    positive_states: np.ndarray       # [N+,3]
    negative_states: np.ndarray       # [N-,3]


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def plot_expansion_results(metrics_jsonl: str | Path, output: str | Path) -> Path:
    rows = _rows(metrics_jsonl)
    rounds = np.asarray([row["round"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    axes[0, 0].plot(rounds, [row["success"] for row in rows], label="success")
    axes[0, 0].plot(rounds, [row["NVP"] for row in rows], label="NVP")
    axes[0, 0].set_title("Closed-loop outcomes")
    axes[0, 1].plot(rounds, [row["positives"] / max(row["queries"], 1) for row in rows])
    axes[0, 1].set_title("Selected-query validity")
    axes[1, 0].plot(rounds, [row["ESS_over_K"] for row in rows])
    axes[1, 0].set_title(r"Acquisition ESS$/K$")
    axes[1, 1].plot(rounds, [np.nan if row["positive_loss"] is None
                             else row["positive_loss"] for row in rows])
    axes[1, 1].set_title("Positive CFM loss")
    for ax in axes.flat:
        ax.grid(alpha=0.20)
        ax.set_xlabel("Expansion round")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_rollout_gallery(
    rollouts: dict[tuple[int, float], Sequence[np.ndarray]],
    rounds: Sequence[int],
    gammas: Sequence[float],
    output: str | Path,
    *,
    draw_scene: Callable[[object], None] | None = None,
) -> Path:
    fig = plt.figure(figsize=(4.5 * len(gammas), 4.2 * len(rounds)))
    for row, round_i in enumerate(rounds):
        for column, gamma in enumerate(gammas):
            ax = fig.add_subplot(len(rounds), len(gammas),
                                 row * len(gammas) + column + 1, projection="3d")
            if draw_scene is not None:
                draw_scene(ax)
            for trajectory in rollouts.get((round_i, float(gamma)), []):
                xyz = np.asarray(trajectory, float)
                ax.plot(*xyz.T, lw=1.15, alpha=0.72)
            if row == 0:
                ax.set_title(rf"$\gamma={gamma:g}$")
            if column == 0:
                ax.text2D(-0.12, 0.5, rf"round {round_i}", transform=ax.transAxes,
                          rotation=90, va="center")
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def render_expansion_mechanism(
    frames: Sequence[MechanismFrame],
    output: str | Path,
    *,
    draw_scene: Callable[[object], None] | None = None,
    fps: int = 7,
) -> Path:
    """Render K plans, selected B uncertainty, verifier labels, and NVP states."""
    output = Path(output)
    all_sigma = np.concatenate([np.asarray(frame.sigma, float) for frame in frames])
    norm = Normalize(float(all_sigma.min()), float(all_sigma.max()) + 1.0e-12)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8.8, 7.2))
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2200)
    with writer.saving(fig, str(output), dpi=150):
        for index, frame in enumerate(frames):
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            if draw_scene is not None:
                draw_scene(ax)
            paths = np.asarray(frame.candidate_paths, float)
            for candidate, path in enumerate(paths):
                if candidate in frame.selected:
                    ax.plot(*path.T, color=cmap(norm(frame.sigma[candidate])),
                            lw=1.35, ls="--", alpha=0.90)
                else:
                    ax.plot(*path.T, color="#9aa0a6", lw=0.55, alpha=0.28)
            for candidate in frame.positive:
                ax.plot(*paths[candidate].T, color="#17964b", lw=1.8, alpha=0.80)
            if frame.executed is not None:
                ax.plot(*paths[frame.executed, :2].T, color="#1468b3", lw=3.0)
            if len(frame.positive_states):
                ax.scatter(*np.asarray(frame.positive_states).T, color="#1468b3", s=13)
            if len(frame.negative_states):
                ax.scatter(*np.asarray(frame.negative_states).T, color="#c8321b", s=22)
            ax.scatter(*np.asarray(frame.robot), color="#111111", marker="s", s=30)
            ax.set_title(rf"frame {index:04d}   round {frame.round}   $\gamma={frame.gamma:g}$")
            scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.08)
            colorbar.set_label(r"$\sigma(\phi_s)$")
            writer.grab_frame()
    plt.close(fig)
    return output
