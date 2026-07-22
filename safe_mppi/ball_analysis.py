"""Ball-below task analysis: per-gamma overlays with the nominal polytope, and gamma trends.

Reads a finished `run.py` output directory (manifest + NPZ rollouts) for the ball-below task:
start (0,0,2) -> goal (3,0,2) with one sphere obstacle whose center sits on the z=2 plane. All
trajectories are expected to pass *below* the sphere's latitude-0 circle (its z=2 equator).

Produces, inside the run directory:
  ball_gamma_<g>.png        per-gamma seed overlay: 3D view with BLUE nominal polytope + level
                            sets near the ball, and an x-z side view with the ball cross-section
  ball_all_gammas.png       all gammas overlaid (3D + side view), colored by gamma
  ball_gamma_trends.png     avg clearance / time-to-goal / smoothness / saturation versus gamma
  ball_metrics.json/.csv    per-run and per-gamma aggregate numbers used in the figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .config import load_config
from .environment import TaskEnvironment
from .visualize import PLASMA, _add_levelsets, _draw_box, _draw_obstacles, _poly_from_saved, _style


def _load(run_dir: Path):
    with (run_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    config = load_config(run_dir / "resolved_config.json")
    env = TaskEnvironment(config)
    runs = []
    for row in manifest["runs"]:
        runs.append((row, np.load(run_dir / row["file"], allow_pickle=True)))
    return manifest, config, env, runs


def _ball(env: TaskEnvironment):
    if len(env.spheres) != 1:
        raise ValueError("ball analysis expects exactly one sphere obstacle")
    sphere = np.asarray(env.spheres[0], float)
    return sphere[:3], float(sphere[3])


def run_metrics(env: TaskEnvironment, row, data, half_window=0.75):
    """Symmetry, below-latitude compliance, and smoothness numbers for one rollout."""
    center, radius = _ball(env)
    states = np.asarray(data["states"], float)
    controls = np.asarray(data["controls"], float)
    dense = env.dense_positions(states, controls)
    clearance = env.obstacle_clearance(dense)

    footprint = np.abs(dense[:, 0] - center[0]) <= radius
    max_z_under = float(dense[footprint, 2].max()) if footprint.any() else None
    below_latitude = bool(max_z_under is not None and max_z_under < center[2])
    axis_distance = np.hypot(dense[:, 0] - center[0], dense[:, 1] - center[1])
    under_disk = axis_distance <= 0.6 * radius
    crossed_under_disk = bool(under_disk.any()
                              and float(dense[under_disk, 2].max()) < center[2] - 0.5 * radius)
    max_abs_y_at_ball = (float(np.abs(dense[footprint, 1]).max()) if footprint.any() else None)

    window = np.abs(dense[:, 0] - center[0]) <= half_window
    symmetry_error = None
    if window.sum() >= 8:
        x, z = dense[window, 0], dense[window, 2]
        order = np.argsort(x)
        x, z = x[order], z[order]
        offsets = np.linspace(0.0, half_window, 40)
        left = np.interp(center[0] - offsets, x, z)
        right = np.interp(center[0] + offsets, x, z)
        symmetry_error = float(np.mean(np.abs(left - right)))

    du = np.diff(controls, axis=0)
    smoothness = float(np.linalg.norm(du, axis=1).mean()) if len(du) else 0.0
    saturation = float((np.abs(controls) > 0.95 * env.mppi.demo_u_max).mean())
    return {
        "gamma": row["gamma"], "seed": row["seed"], "success": row["success"],
        "collision": row["collision"], "time_to_goal_s": row["time_to_goal_s"],
        "min_clearance_m": float(clearance.min()),
        "max_z_under_ball_m": max_z_under, "below_latitude0": below_latitude,
        "crossed_under_disk": crossed_under_disk, "max_abs_y_at_ball_m": max_abs_y_at_ball,
        "max_abs_y_m": float(np.abs(dense[:, 1]).max()),
        "symmetry_error_m": symmetry_error,
        "mean_delta_u": smoothness, "saturation_fraction": saturation,
    }


def _mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def aggregate(per_run, gammas):
    out = []
    for gamma in gammas:
        rows = [r for r in per_run if abs(r["gamma"] - gamma) < 1e-9]
        out.append({
            "gamma": float(gamma), "episodes": len(rows),
            "SR": _mean(rows, "success"), "CR": _mean(rows, "collision"),
            "all_below_latitude0": bool(all(r["below_latitude0"] for r in rows)),
            "all_crossed_under_disk": bool(all(r["crossed_under_disk"] for r in rows)),
            "avg_max_abs_y_at_ball_m": _mean(rows, "max_abs_y_at_ball_m"),
            "avg_min_clearance_m": _mean(rows, "min_clearance_m"),
            "avg_time_to_goal_s": _mean(rows, "time_to_goal_s"),
            "avg_max_z_under_ball_m": _mean(rows, "max_z_under_ball_m"),
            "avg_symmetry_error_m": _mean(rows, "symmetry_error_m"),
            "avg_max_abs_y_m": _mean(rows, "max_abs_y_m"),
            "avg_mean_delta_u": _mean(rows, "mean_delta_u"),
            "avg_saturation_fraction": _mean(rows, "saturation_fraction"),
        })
    return out


def _draw_latitude0(ax3d, center, radius, color="#cc3311"):
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    ax3d.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
              np.full_like(theta, center[2]), color=color, lw=1.4, ls="--",
              label="ball latitude 0 (z=%.0f m)" % center[2])


def _side_view(ax, env, center, radius, paths, colors, labels=None):
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    ax.fill(center[0] + radius * np.cos(theta), center[2] + radius * np.sin(theta),
            color="#8f969f", alpha=0.5, zorder=3)
    ax.axhline(center[2], color="#cc3311", lw=1.0, ls="--", zorder=2)
    for k, path in enumerate(paths):
        label = labels[k] if labels else None
        ax.plot(path[:, 0], path[:, 2], color=colors[k], lw=1.5, alpha=0.85, label=label, zorder=4)
    ax.scatter(*env.start[[0, 2]], marker="s", color="#111111", s=40, zorder=5)
    ax.scatter(env.goal[0], env.goal[2], marker="*", color="#ffca28", edgecolor="#6a4e00",
               s=170, zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlim(env.start[0] - 0.4, env.goal[0] + 0.4)
    ax.set_ylim(center[2] - radius - 0.7, center[2] + radius + 0.4)


def _polytope_index(data, ball_x):
    states = np.asarray(data["states"], float)
    n_poly = len(data["poly_A"])
    return int(np.argmin(np.abs(states[:n_poly, 0] - ball_x)))


def plot_gamma_panel(env, gamma, runs, output_dir):
    center, radius = _ball(env)
    fig = plt.figure(figsize=(13.6, 5.6), facecolor="white")
    ax3d = fig.add_subplot(121, projection="3d")
    _draw_box(ax3d, env.bounds, alpha=0.25, linewidth=0.5)
    _draw_obstacles(ax3d, env, alpha=0.28)
    _draw_latitude0(ax3d, center, radius)
    color = PLASMA(Normalize(0.0, 1.0)(gamma))
    paths = []
    for row, data in runs:
        dense = env.dense_positions(np.asarray(data["states"], float),
                                    np.asarray(data["controls"], float))
        paths.append(dense)
        ax3d.plot(*dense.T, color=color, lw=1.5, alpha=0.75)
        if not row["success"]:
            ax3d.scatter(*dense[-1], marker="x", color="#cc3311", s=55, linewidth=1.7)
    reference = next((d for r, d in runs if r["success"]), runs[0][1])
    index = _polytope_index(reference, center[0])
    polytope = _poly_from_saved(reference, index)
    ax3d.scatter(*polytope.center, marker="o", color="#111111", s=18)
    _add_levelsets(ax3d, polytope, gamma)
    ax3d.scatter(*env.start[:3], marker="s", color="#111111", s=42, label="start")
    ax3d.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=180, label="goal")
    _style(ax3d, env)
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.set_title(rf"$\gamma={gamma:g}$ — {len(runs)} seeds, BLUE nominal $P_k$ + levels"
                   f" at step {index}", fontsize=11, weight="bold")

    ax2d = fig.add_subplot(122)
    _side_view(ax2d, env, center, radius, paths, [color] * len(paths),
               labels=[f"seed {row['seed']}" for row, _ in runs])
    ax2d.legend(loc="lower left", fontsize=8, ncol=2)
    ax2d.set_title("x-z side view — pass below the dashed latitude-0 line", fontsize=11)
    fig.tight_layout()
    out = Path(output_dir) / f"ball_gamma_{gamma:g}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all_gammas(env, manifest, runs_by_gamma, output_dir):
    center, radius = _ball(env)
    gammas = sorted(runs_by_gamma)
    norm = Normalize(min(gammas), max(gammas))
    fig = plt.figure(figsize=(13.6, 5.6), facecolor="white")
    ax3d = fig.add_subplot(121, projection="3d")
    _draw_box(ax3d, env.bounds, alpha=0.25, linewidth=0.5)
    _draw_obstacles(ax3d, env, alpha=0.28)
    _draw_latitude0(ax3d, center, radius)
    paths, colors = [], []
    for gamma in gammas:
        for row, data in runs_by_gamma[gamma]:
            dense = env.dense_positions(np.asarray(data["states"], float),
                                        np.asarray(data["controls"], float))
            paths.append(dense)
            colors.append(PLASMA(norm(gamma)))
            ax3d.plot(*dense.T, color=colors[-1], lw=1.4, alpha=0.7)
    ax3d.scatter(*env.start[:3], marker="s", color="#111111", s=42, label="start")
    ax3d.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=180, label="goal")
    _style(ax3d, env)
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.set_title("All gammas, all seeds — below-the-ball rollouts", fontsize=12, weight="bold")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=PLASMA)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax3d, fraction=0.03, pad=0.09).set_label(r"safety level $\gamma$")

    ax2d = fig.add_subplot(122)
    _side_view(ax2d, env, center, radius, paths, colors)
    ax2d.set_title("x-z side view — all gammas", fontsize=11)
    fig.tight_layout()
    out = Path(output_dir) / "ball_all_gammas.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trends(per_run, aggregates, output_dir):
    gammas = [row["gamma"] for row in aggregates]
    panels = [
        ("avg_min_clearance_m", "min_clearance_m", "avg min clearance [m]"),
        ("avg_time_to_goal_s", "time_to_goal_s", "avg time to goal [s]"),
        ("avg_mean_delta_u", "mean_delta_u", r"smoothness: mean $\|\Delta u\|$ [m/s$^2$]"),
        ("avg_saturation_fraction", "saturation_fraction", "bang-bang: |u| > 0.95 cap fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), facecolor="white")
    norm = Normalize(min(gammas), max(gammas))
    for ax, (agg_key, run_key, label) in zip(axes.ravel(), panels):
        for row in per_run:
            if row[run_key] is not None:
                ax.scatter(row["gamma"], row[run_key], color=PLASMA(norm(row["gamma"])),
                           s=22, alpha=0.55, zorder=3)
        values = [row[agg_key] for row in aggregates]
        mask = [v is not None for v in values]
        xs = [g for g, m in zip(gammas, mask) if m]
        ys = [v for v in values if v is not None]
        ax.plot(xs, ys, "-o", color="#174f92", lw=1.8, ms=5, zorder=4)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(label)
        ax.set_xticks(gammas)
        ax.grid(alpha=0.25)
    fig.suptitle("Gamma trends — dots are single seeds, line is the per-gamma mean",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(output_dir) / "ball_gamma_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def analyze(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest, config, env, runs = _load(run_dir)
    per_run = [run_metrics(env, row, data) for row, data in runs]
    aggregates = aggregate(per_run, manifest["gammas"])

    with (run_dir / "ball_metrics.json").open("w") as f:
        json.dump({"per_run": per_run, "per_gamma": aggregates}, f, indent=2)
    with (run_dir / "ball_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    runs_by_gamma = {}
    for row, data in runs:
        runs_by_gamma.setdefault(float(row["gamma"]), []).append((row, data))
    figures = [plot_gamma_panel(env, gamma, runs_by_gamma[gamma], run_dir)
               for gamma in sorted(runs_by_gamma)]
    figures.append(plot_all_gammas(env, manifest, runs_by_gamma, run_dir))
    figures.append(plot_trends(per_run, aggregates, run_dir))

    for row in aggregates:
        print(f"gamma={row['gamma']:g} SR={row['SR']} all_below_latitude0="
              f"{row['all_below_latitude0']} under_disk={row['all_crossed_under_disk']} "
              f"y_at_ball={row['avg_max_abs_y_at_ball_m']:.3f} "
              f"clearance={row['avg_min_clearance_m']} "
              f"time={row['avg_time_to_goal_s']} sym_err={row['avg_symmetry_error_m']} "
              f"mean_du={row['avg_mean_delta_u']} sat={row['avg_saturation_fraction']}",
              flush=True)
    print("[outputs]", *(str(path) for path in figures), sep="\n  ")
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="finished run.py output directory")
    args = parser.parse_args()
    analyze(args.run)


if __name__ == "__main__":
    main()
