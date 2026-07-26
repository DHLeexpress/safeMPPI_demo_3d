#!/usr/bin/env python3
"""Replay accepted lab SafeMPPI references through the calibrated plant."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_mppi.ball_flow_theta import start_goal_frame, world_to_local  # noqa: E402
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_plant_replay import replay_demo_on_plant  # noqa: E402


def _group_summary(rows, gamma):
    group = [row for row in rows if np.isclose(row["gamma"], gamma)]
    return {
        "gamma": float(gamma),
        "episodes": len(group),
        "reference_SR": float(np.mean([row["reference_success"] for row in group])),
        "reference_CR": float(np.mean([row["reference_collision"] for row in group])),
        "plant_SR": float(np.mean([row["plant_success"] for row in group])),
        "plant_goal_reach_rate": float(np.mean([
            row["plant_reached"] for row in group
        ])),
        "plant_CR": float(np.mean([row["plant_collision"] for row in group])),
        "plant_OOB_rate": float(np.mean([
            row["plant_taskspace_violation"] for row in group
        ])),
        "plant_measured_soft_fence_violation_rate": float(np.mean([
            row["plant_measured_soft_geofence_violation"] for row in group
        ])),
        "plant_true_hard_fence_violation_rate": float(np.mean([
            row["plant_true_hard_geofence_violation"] for row in group
        ])),
        "plant_measured_hard_fence_violation_rate": float(np.mean([
            row["plant_measured_hard_geofence_violation"] for row in group
        ])),
        "mean_reference_clearance_m": float(np.mean([
            row["reference_min_clearance_m"] for row in group
        ])),
        "mean_plant_clearance_m": float(np.mean([
            row["plant_min_clearance_m"] for row in group
        ])),
        "mean_tracking_RMSE_m": float(np.mean([
            row["tracking_rmse_m"] for row in group
        ])),
        "p95_tracking_max_error_m": float(np.percentile([
            row["tracking_max_error_m"] for row in group
        ], 95)),
        "mean_clearance_erosion_m": float(np.mean([
            row["clearance_erosion_m"] for row in group
        ])),
        "mean_peak_reference_speed_mps": float(np.mean([
            row["peak_reference_speed_mps"] for row in group
        ])),
        "mean_peak_plant_speed_mps": float(np.mean([
            row["peak_plant_speed_mps"] for row in group
        ])),
        "mean_raw_action_cap_fraction": float(np.mean([
            row["raw_action_cap_fraction"] for row in group
        ])),
        "mean_applied_action_cap_fraction": float(np.mean([
            row["applied_action_cap_fraction"] for row in group
        ])),
        "mean_reference_speed_cap_fraction": float(np.mean([
            row["reference_speed_cap_fraction"] for row in group
        ])),
        "mean_reference_vertical_speed_cap_fraction": float(np.mean([
            row["reference_vertical_speed_cap_fraction"] for row in group
        ])),
        "mean_peak_abs_plant_vertical_speed_mps": float(np.mean([
            row["peak_abs_plant_vertical_speed_mps"] for row in group
        ])),
        "max_abs_applied_acceleration_mps2": float(max(
            row["max_abs_applied_acceleration_mps2"] for row in group
        )),
    }


def evaluate_archive(demo_dir: Path, clip_commanded_position: bool):
    manifest = json.loads((demo_dir / "manifest.json").read_text())
    config = load_config(demo_dir / "resolved_config.json")
    env = TaskEnvironment(config)
    records = []
    rows = []
    for row in manifest["runs"]:
        data = np.load(demo_dir / row["file"])
        replay = replay_demo_on_plant(
            config,
            data["dense_positions"],
            data["executed_controls"],
            seed=int(row["seed"]),
            clip_commanded_position=clip_commanded_position,
        )
        expected_reference = np.asarray(data["dense_positions"], np.float32)[1:]
        expected_applied = np.repeat(
            np.asarray(data["executed_controls"], np.float32),
            config.safemppi.integration_substeps,
            axis=0,
        )
        if not np.allclose(
            replay["reference_positions"],
            expected_reference,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(f"{row['file']}: reference reconstruction mismatch")
        if not np.allclose(
            replay["applied_controls"],
            expected_applied,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(f"{row['file']}: applied-action reconstruction mismatch")
        raw_controls = np.asarray(data["controls"], np.float32)
        reference_speed = np.linalg.norm(
            replay["reference_velocities"], axis=1,
        )
        reference_vertical_speed = np.abs(
            replay["reference_velocities"][:, 2],
        )
        applied_controls = np.asarray(data["executed_controls"], np.float32)
        if bool(row["success"]) != bool(replay["reference_success"]):
            raise RuntimeError(
                f"{row['file']}: accepted manifest and stored-reference "
                "success disagree"
            )
        scalar = {
            "gamma": float(row["gamma"]),
            "seed": int(row["seed"]),
            "file": row["file"],
            "reference_success": bool(replay["reference_success"]),
            "reference_reached": bool(replay["reference_reached"]),
            "reference_collision": bool(replay["reference_collision"]),
            "reference_taskspace_violation": bool(
                replay["reference_taskspace_violation"]
            ),
            "plant_success": bool(replay["geometric_success"]),
            "plant_reached": bool(replay["reached"]),
            "plant_collision": bool(replay["collision"]),
            "plant_taskspace_violation": bool(replay["taskspace_violation"]),
            "plant_true_soft_geofence_violation": bool(
                replay["true_soft_geofence_violation"]
            ),
            "plant_measured_soft_geofence_violation": bool(
                replay["measured_soft_geofence_violation"]
            ),
            "plant_true_hard_geofence_violation": bool(
                replay["true_hard_geofence_violation"]
            ),
            "plant_measured_hard_geofence_violation": bool(
                replay["measured_hard_geofence_violation"]
            ),
            "raw_action_cap_fraction": float(np.mean(np.any(
                np.abs(raw_controls) >= config.safemppi.demo_u_max - 1.0e-6,
                axis=1,
            ))),
            "reference_speed_cap_fraction": float(np.mean(
                reference_speed >= config.safemppi.max_speed - 1.0e-5
            )),
            "reference_vertical_speed_cap_fraction": float(np.mean(
                reference_vertical_speed
                >= config.safemppi.max_vertical_speed - 1.0e-5
            )),
            "applied_action_cap_fraction": float(np.mean(np.any(
                np.abs(applied_controls)
                >= config.safemppi.demo_u_max - 1.0e-5,
                axis=1,
            ))),
            "clearance_erosion_m": float(
                replay["reference_min_clearance_m"]
                - replay["plant_min_clearance_m"]
            ),
            **{
                key: replay[key]
                for key in (
                    "reference_min_clearance_m",
                    "plant_min_clearance_m",
                    "minimum_true_soft_geofence_margin_m",
                    "minimum_measured_soft_geofence_margin_m",
                    "minimum_true_hard_geofence_margin_m",
                    "minimum_measured_hard_geofence_margin_m",
                    "tracking_rmse_m",
                    "tracking_max_error_m",
                    "reference_command_rmse_m",
                    "command_clip_fraction",
                    "peak_reference_speed_mps",
                    "peak_plant_speed_mps",
                    "peak_abs_plant_vertical_speed_mps",
                    "max_abs_applied_acceleration_mps2",
                    "time_to_goal_s",
                )
            },
        }
        rows.append(scalar)
        records.append({**scalar, "arrays": replay})
    summaries = [
        _group_summary(rows, gamma) for gamma in config.data.gammas
    ]
    return config, env, records, rows, summaries


def _ball_circle(axis, center, radius):
    angle = np.linspace(0.0, 2.0 * np.pi, 180)
    axis.fill(
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
        color="#9ca3af",
        alpha=0.28,
        zorder=0,
    )


def plot_trajectories(config, env, records, output: Path):
    gammas = list(config.data.gammas)
    frame = start_goal_frame(env)
    start = env.start[:3]
    center_local = world_to_local(env.spheres[0, :3] - start, frame)
    radius = float(env.spheres[0, 3])
    colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.92, len(gammas)))
    fig, axes = plt.subplots(
        3, len(gammas), figsize=(16.0, 10.2), squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.72]},
    )
    for column, (gamma, color) in enumerate(zip(gammas, colors)):
        group = sorted(
            (row for row in records if np.isclose(row["gamma"], gamma)),
            key=lambda row: row["seed"],
        )
        shown = group[:10]
        top, side, error_axis = axes[:, column]
        _ball_circle(top, center_local[:2], radius)
        _ball_circle(side, center_local[[0, 2]], radius)
        error_curves = []
        for index, row in enumerate(shown):
            replay = row["arrays"]
            reference = world_to_local(
                replay["reference_positions"] - start, frame,
            )
            plant = world_to_local(
                replay["plant_positions"] - start, frame,
            )
            label_reference = "governed reference" if index == 0 else None
            label_plant = "calibrated plant" if index == 0 else None
            top.plot(
                reference[:, 0], reference[:, 1],
                color="#30343b", lw=0.8, alpha=0.38,
                label=label_reference,
            )
            top.plot(
                plant[:, 0], plant[:, 1],
                color=color, lw=1.0, alpha=0.55,
                label=label_plant,
            )
            side.plot(
                reference[:, 0], reference[:, 2],
                color="#30343b", lw=0.8, alpha=0.38,
            )
            side.plot(
                plant[:, 0], plant[:, 2],
                color=color, lw=1.0, alpha=0.55,
            )
            error = np.linalg.norm(
                replay["plant_positions"] - replay["reference_positions"],
                axis=1,
            )
            error_curves.append(error)
            error_axis.plot(
                np.linspace(0.0, 1.0, len(error)),
                error, color=color, lw=0.7, alpha=0.20,
            )
        shortest = min(map(len, error_curves))
        mean_error = np.mean([curve[:shortest] for curve in error_curves], axis=0)
        error_axis.plot(
            np.linspace(0.0, 1.0, shortest),
            mean_error, color=color, lw=2.2,
        )
        top.set_title(rf"$\gamma={gamma:g}$")
        top.set_xlabel(r"longitudinal $s$ [m]")
        top.set_ylabel("lateral [m]")
        side.set_xlabel(r"longitudinal $s$ [m]")
        side.set_ylabel(r"$z-z_{\rm start}$ [m]")
        error_axis.set_xlabel("normalized reference time")
        error_axis.set_ylabel(r"$\|p_{\rm plant}-p_{\rm ref}\|$ [m]")
        for axis in (top, side, error_axis):
            axis.grid(alpha=0.20)
        top.set_aspect("equal", adjustable="box")
        side.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        "Accepted SafeMPPI reference tracking through the calibrated plant",
        fontsize=18,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(summaries, output: Path):
    gammas = [row["gamma"] for row in summaries]
    x = np.arange(len(gammas))
    width = 0.34
    colors = ("#40464f", "#1f9e89")
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.6), squeeze=False)
    labels = [rf"$\gamma={gamma:g}$" for gamma in gammas]

    axes[0, 0].bar(
        x - width / 2, [row["reference_SR"] for row in summaries],
        width, color=colors[0], label="reference SR",
    )
    axes[0, 0].bar(
        x + width / 2, [row["plant_SR"] for row in summaries],
        width, color=colors[1], label="plant SR",
    )
    axes[0, 0].set(title="Success rate", ylim=(0.0, 1.05))
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(
        x - width / 2, [row["reference_CR"] for row in summaries],
        width, color=colors[0], label="reference CR",
    )
    axes[0, 1].bar(
        x + width / 2, [row["plant_CR"] for row in summaries],
        width, color="#d1495b", label="plant CR",
    )
    axes[0, 1].set(title="Collision rate", ylim=(0.0, 1.05))
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        x, [row["mean_reference_clearance_m"] for row in summaries],
        "o-", color=colors[0], label="reference",
    )
    axes[1, 0].plot(
        x, [row["mean_plant_clearance_m"] for row in summaries],
        "o-", color=colors[1], label="plant",
    )
    axes[1, 0].axhline(0.0, color="#777777", lw=0.8)
    axes[1, 0].set(title="Mean minimum clearance", ylabel="clearance [m]")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        x, [row["mean_tracking_RMSE_m"] for row in summaries],
        color=colors[1],
    )
    axes[1, 1].set(title="Position-tracking RMSE", ylabel="RMSE [m]")

    axes[0, 2].bar(
        x, [row["plant_goal_reach_rate"] for row in summaries],
        color="#3c78a8",
    )
    axes[0, 2].set(title="Plant goal-reach rate", ylim=(0.0, 1.05))

    axes[1, 2].plot(
        x, [row["mean_raw_action_cap_fraction"] for row in summaries],
        "o-", color="#d1495b", label="raw action at cap",
    )
    axes[1, 2].plot(
        x, [row["mean_reference_speed_cap_fraction"] for row in summaries],
        "o-", color="#6f4e7c", label="reference speed at cap",
    )
    axes[1, 2].set(
        title="How often reference caps are active",
        ylabel="fraction of reference steps",
        ylim=(0.0, 1.0),
    )
    axes[1, 2].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.20, axis="y")
    fig.suptitle("Reference-domain safety versus calibrated-plant tracking")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clip-commanded-position",
        action="store_true",
        help="also apply the harness command-box clamp; off isolates pure tracking",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, env, records, rows, summaries = evaluate_archive(
        args.demo_dir,
        clip_commanded_position=args.clip_commanded_position,
    )
    contract = {
        "experiment": (
            "accepted once-governed reference streamed to calibrated plant; "
            "no replanning and no second governor"
        ),
        "clip_commanded_position": bool(args.clip_commanded_position),
        "plant": "unchanged deploy_sim.plant",
        "per_gamma": summaries,
    }
    (args.output_dir / "reference_vs_plant.json").write_text(
        json.dumps({"contract": contract, "per_run": rows}, indent=2) + "\n"
    )
    with (args.output_dir / "reference_vs_plant.csv").open(
        "w", newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    plot_trajectories(
        config, env, records,
        args.output_dir / "reference_vs_plant_trajectories.png",
    )
    plot_metrics(
        summaries,
        args.output_dir / "reference_vs_plant_metrics.png",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
