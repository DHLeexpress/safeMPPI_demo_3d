"""Qualify and render the Minhyuk-frame SafeMPPI pretraining archive."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_sim.harness import geofence
from safe_mppi.ball_flow_theta import (
    start_goal_frame,
    theta_name,
    trajectory_crossing_theta,
    world_to_local,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.visualize import _draw_box, _draw_obstacles, _style


ROUTES = ("below", "above", "left", "right")


def _signed_box_margin(points, box):
    points = np.asarray(points, float)
    box = np.asarray(box, float)
    return float(np.min(np.minimum(
        points - box[:, 0],
        box[:, 1] - points,
    )))


def _crossing_point(env, positions, frame):
    positions = np.asarray(positions, float)
    center = np.asarray(env.spheres[0, :3], float)
    axial = (positions - center) @ frame[:, 0]
    indices = np.flatnonzero((axial[:-1] <= 0.0) & (axial[1:] >= 0.0))
    if not len(indices):
        return None
    index = int(indices[0])
    delta = axial[index + 1] - axial[index]
    fraction = 0.0 if abs(delta) <= 1.0e-12 else -axial[index] / delta
    return positions[index] + fraction * (positions[index + 1] - positions[index])


def build_qualification(demo_dir):
    demo_dir = Path(demo_dir)
    manifest = json.loads((demo_dir / "manifest.json").read_text())
    config = load_config(demo_dir / "resolved_config.json")
    env = TaskEnvironment(config)
    frame = start_goal_frame(env)
    soft, hard = geofence(config, env, env.start, env.goal)
    per_run = []
    trajectories = []
    for row in manifest["runs"]:
        data = np.load(demo_dir / row["file"], allow_pickle=True)
        states = np.asarray(data["states"], float)
        dense = np.asarray(data["dense_positions"], float)
        raw = np.asarray(data["controls"], float)
        applied = np.asarray(data["executed_controls"], float)
        theta = trajectory_crossing_theta(env, dense, frame)
        route = theta_name(theta)
        crossing = _crossing_point(env, dense, frame)
        crossing_local = (
            None
            if crossing is None
            else world_to_local(crossing - env.spheres[0, :3], frame)
        )
        per_run.append({
            "gamma": float(row["gamma"]),
            "seed": int(row["seed"]),
            "route": route,
            "crossing_below_plane": (
                None if crossing_local is None else bool(crossing_local[2] < 0.0)
            ),
            "success": bool(row["success"]),
            "min_clearance_m": float(row["min_clearance_m"]),
            "time_to_goal_s": row["time_to_goal_s"],
            "max_abs_raw_accel_mps2": float(np.max(np.abs(raw))),
            "max_abs_applied_accel_mps2": float(np.max(np.abs(applied))),
            "peak_speed_mps": float(np.linalg.norm(states[:, 3:6], axis=1).max()),
            "peak_vertical_speed_mps": float(np.abs(states[:, 5]).max()),
            "taskspace_margin_m": _signed_box_margin(dense, env.bounds),
            "soft_geofence_margin_m": _signed_box_margin(dense, soft),
            "hard_geofence_margin_m": _signed_box_margin(dense, hard),
            "all_infeasible_steps": int(row["all_infeasible_steps"]),
            "minimum_executed_one_step_slack": row[
                "minimum_online_one_step_slack"
            ],
        })
        trajectories.append({
            "gamma": float(row["gamma"]),
            "states": states,
            "dense": dense,
            "route": route,
            "crossing": crossing,
        })

    per_gamma = []
    for gamma in config.data.gammas:
        runs = [row for row in per_run if np.isclose(row["gamma"], gamma)]
        attempts = [
            row for row in manifest["attempts"]
            if np.isclose(float(row["gamma"]), gamma)
        ]
        counts = {route: sum(row["route"] == route for row in runs)
                  for route in ROUTES}
        crossing_rows = [
            row for row in runs if row["crossing_below_plane"] is not None
        ]
        per_gamma.append({
            "gamma": float(gamma),
            "attempts": len(attempts),
            "accepted": len(runs),
            "attempt_SR": float(np.mean([row["success"] for row in attempts])),
            "attempt_CR": float(np.mean([row["collision"] for row in attempts])),
            "attempt_OOB_rate": float(np.mean([
                row["taskspace_violation"] for row in attempts
            ])),
            "attempt_all_infeasible_rate": float(np.mean([
                row["all_infeasible"] for row in attempts
            ])),
            "crossing_below_plane_fraction": float(np.mean([
                row["crossing_below_plane"] for row in crossing_rows
            ])),
            "below_fraction": counts["below"] / len(runs),
            "above_fraction": counts["above"] / len(runs),
            "left_fraction": counts["left"] / len(runs),
            "right_fraction": counts["right"] / len(runs),
            "mean_min_clearance_m": float(np.mean([
                row["min_clearance_m"] for row in runs
            ])),
            "mean_time_to_goal_s": float(np.mean([
                row["time_to_goal_s"] for row in runs
            ])),
            "max_abs_raw_accel_mps2": float(max(
                row["max_abs_raw_accel_mps2"] for row in runs
            )),
            "max_abs_applied_accel_mps2": float(max(
                row["max_abs_applied_accel_mps2"] for row in runs
            )),
            "max_speed_mps": float(max(row["peak_speed_mps"] for row in runs)),
            "max_vertical_speed_mps": float(max(
                row["peak_vertical_speed_mps"] for row in runs
            )),
            "minimum_taskspace_margin_m": float(min(
                row["taskspace_margin_m"] for row in runs
            )),
            "minimum_soft_geofence_margin_m": float(min(
                row["soft_geofence_margin_m"] for row in runs
            )),
            "minimum_hard_geofence_margin_m": float(min(
                row["hard_geofence_margin_m"] for row in runs
            )),
            **{f"{route}_count": int(counts[route]) for route in ROUTES},
        })
    return config, env, frame, trajectories, {
        "contract": "lab-frame reference-dynamics SafeMPPI demonstrations",
        "attempt_metrics_are_pre_retry": True,
        "accepted_archive_is_success_conditioned": True,
        "per_gamma": per_gamma,
        "per_run": per_run,
    }


def plot_overlay(env, frame, trajectories, output):
    gammas = sorted({row["gamma"] for row in trajectories})
    colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / (len(gammas) - 1))
        for index, gamma in enumerate(gammas)
    }
    center = np.asarray(env.spheres[0, :3], float)
    radius = float(env.spheres[0, 3])
    start = np.asarray(env.start[:3], float)
    length = float(np.linalg.norm(env.goal - start))

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.titlesize": 15, "axes.labelsize": 13,
    })
    fig = plt.figure(figsize=(14.8, 10.0))
    grid = fig.add_gridspec(2, 2, hspace=0.27, wspace=0.22)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax_side = fig.add_subplot(grid[0, 1])
    ax_head = fig.add_subplot(grid[1, 0])
    ax_count = fig.add_subplot(grid[1, 1])

    _draw_box(ax3, env.bounds)
    _draw_obstacles(ax3, env, alpha=0.27)
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ax_side.fill(
        length / 2 + radius * np.cos(theta),
        radius * np.sin(theta),
        color="#9299a2", alpha=0.35,
    )
    ax_head.fill(
        radius * np.cos(theta), radius * np.sin(theta),
        color="#9299a2", alpha=0.35,
    )

    counts = np.zeros((len(gammas), len(ROUTES)), int)
    for row in trajectories:
        gamma = row["gamma"]
        color = colors[gamma]
        positions = row["states"][:, :3]
        local = world_to_local(positions - start, frame)
        ax3.plot(*positions.T, color=color, lw=0.85, alpha=0.40)
        ax_side.plot(local[:, 0], local[:, 2], color=color, lw=0.85, alpha=0.40)
        ax_head.plot(
            local[:, 1],
            local[:, 2] - center[2] + start[2],
            color=color, lw=0.85, alpha=0.40,
        )
        if row["crossing"] is not None:
            crossing = world_to_local(row["crossing"] - center, frame)
            ax_head.scatter(
                crossing[1], crossing[2], color=color, s=10, alpha=0.72,
            )
        if row["route"] in ROUTES:
            counts[gammas.index(gamma), ROUTES.index(row["route"])] += 1

    ax3.scatter(*env.start[:3], marker="s", color="#111111", s=32)
    ax3.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=130)
    _style(ax3, env)
    ax3.view_init(elev=23, azim=-50)
    ax3.set_title("Lab-frame 3-D trajectories")

    ax_side.scatter(0.0, 0.0, marker="s", color="#111111", s=28)
    ax_side.scatter(length, 0.0, marker="*", color="#ffca28",
                    edgecolor="#6a4e00", s=105)
    ax_side.set(
        title="Start--goal longitudinal view",
        xlabel=r"$s$ [m]",
        ylabel=r"$z-z_{\mathrm{start}}$ [m]",
        xlim=(-0.1, length + 0.1),
    )
    ax_side.set_aspect("equal", adjustable="box")

    ax_head.axhline(0.0, color="#777777", lw=0.8, ls="--")
    ax_head.set(
        title="Crossing plane at the sphere",
        xlabel=r"lateral displacement [m]",
        ylabel=r"$z-z_{\mathrm{sphere}}$ [m]",
        xlim=(-1.0, 1.0), ylim=(-0.55, 0.95),
    )
    ax_head.set_aspect("equal", adjustable="box")

    image = ax_count.imshow(
        counts, cmap="Blues", vmin=0, vmax=max(1, int(counts.max())),
    )
    for row in range(len(gammas)):
        for column in range(len(ROUTES)):
            ax_count.text(
                column, row, str(counts[row, column]),
                ha="center", va="center",
                color="white" if counts[row, column] > counts.max() / 2 else "#111111",
                fontsize=13,
            )
    ax_count.set_xticks(range(len(ROUTES)), ROUTES)
    ax_count.set_yticks(
        range(len(gammas)), [rf"$\gamma={gamma:g}$" for gamma in gammas],
    )
    ax_count.set_title("Accepted dominant crossing modes")
    fig.colorbar(image, ax=ax_count, fraction=0.046, pad=0.04)

    for axis in (ax_side, ax_head, ax_count):
        axis.grid(alpha=0.20)
    handles = [
        plt.Line2D([0], [0], color=colors[gamma], lw=2.2,
                   label=rf"$\gamma={gamma:g}$")
        for gamma in gammas
    ]
    fig.legend(
        handles=handles, ncol=len(gammas), loc="upper center",
        bbox_to_anchor=(0.5, 0.945), frameon=False,
    )
    fig.suptitle(
        "Minhyuk-frame SafeMPPI pretraining demonstrations",
        fontsize=18, weight="bold", y=0.992,
    )
    fig.subplots_adjust(top=0.87)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, env, frame, trajectories, qualification = build_qualification(
        args.demo_dir,
    )
    json_path = args.output_dir / "lab_ball_qualification.json"
    csv_path = args.output_dir / "lab_ball_qualification.csv"
    json_path.write_text(json.dumps(qualification, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(qualification["per_gamma"][0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(qualification["per_gamma"])
    plot_overlay(
        env, frame, trajectories,
        args.output_dir / "lab_ball_demo_overlay.png",
    )
    print(json.dumps(qualification["per_gamma"], indent=2))
    print("[outputs]", json_path, csv_path,
          args.output_dir / "lab_ball_demo_overlay.png", sep="\n  ")


if __name__ == "__main__":
    main()
