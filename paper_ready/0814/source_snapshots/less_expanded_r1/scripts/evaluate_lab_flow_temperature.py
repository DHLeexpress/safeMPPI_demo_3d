"""Sweep raw flow sampling scale in the reference sim and native vehicle.

``sampling temperature`` is the latent standard deviation in
``x0 ~ N(0, tau^2 I)``.  It is not the SafeMPPI softmax temperature.
Each cell is evaluated both as a raw governed reference and through Minhyuk's
unchanged online state-feedback deployment harness.
"""
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
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "deploy_sim"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deploy_sim import run_offline as native  # noqa: E402
from flow_deployment.bridge import sha256_file, verify_deploy_sim_lock  # noqa: E402
from safe_mppi.ball_flow_theta import theta_name, trajectory_crossing_theta  # noqa: E402
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_reference_flow_task import (  # noqa: E402
    LAB_ROUTE_MODES,
    LabReferenceFlowController,
    raw_reference_rollout,
    reference_window_validity_fraction,
)
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402


def _status(success, collision, oob):
    if success:
        return "SUCCESS"
    if collision:
        return "COLLISION"
    if oob:
        return "OOB"
    return "TIMEOUT"


def _dense_reference(result):
    return np.concatenate([
        result["states"][:1, :3],
        result["dense_steps"].reshape(-1, 3),
    ])


def _vehicle_window_validity(config, result, gamma):
    substeps = config.safemppi.integration_substeps
    positions = np.asarray([
        [row["x"], row["y"], row["z"]] for row in result["log"]
    ], np.float32)
    if not len(positions) or len(positions) % substeps:
        return 0.0
    dense = positions.reshape(-1, substeps, 3)
    states = np.concatenate([
        np.asarray(config.taskspace.start[:3], np.float32)[None],
        dense[:, -1],
    ])
    state6 = np.concatenate([states, np.zeros_like(states)], axis=1)
    return reference_window_validity_fraction(
        config,
        state6,
        dense,
        float(gamma),
    )


def evaluate_cell(policy, config, tau, gamma, episodes, seed0):
    env = TaskEnvironment(config)
    controller = LabReferenceFlowController(
        policy,
        env,
        sampling_temperature=float(tau),
    )
    records = []
    for episode in range(episodes):
        seed = int(seed0 + 37 * episode)
        reference = raw_reference_rollout(
            policy,
            config,
            float(gamma),
            seed,
            sampling_temperature=float(tau),
        )
        dense_reference = _dense_reference(reference)
        swarm = native.Swarm(
            np.array([env.start[0], env.start[1], 0.0]),
            seed=seed,
        )
        vehicle = native.harness.run(
            controller,
            env,
            config,
            swarm,
            gamma=float(gamma),
            seed=seed,
            verbose=False,
        )
        metrics = native.harness.summarize(vehicle, env, verbose=False)
        vehicle_path = np.concatenate([
            np.asarray(config.taskspace.start[:3], np.float32)[None],
            np.asarray([
                [row["x"], row["y"], row["z"]]
                for row in vehicle["log"]
            ], np.float32).reshape(-1, 3),
        ])
        clearance = env.obstacle_clearance(vehicle_path)
        finite_clearance = clearance[np.isfinite(clearance)]
        vehicle_mode = theta_name(
            trajectory_crossing_theta(env, vehicle_path)
        )
        vehicle_status = _status(
            bool(vehicle["reached"]),
            bool(len(finite_clearance) and finite_clearance.min() < 0.0),
            bool(not env.inside_taskspace(vehicle_path).all()),
        )
        commanded = np.asarray([
            [row["cx"], row["cy"], row["cz"]]
            for row in vehicle["log"]
        ], np.float32).reshape(-1, 3)
        measured = vehicle_path[1:]
        tracking_error = np.linalg.norm(measured - commanded, axis=1)
        soft = np.asarray(vehicle["soft"], float)
        hard = np.asarray(vehicle["hard"], float)
        soft_violation = bool(
            np.any(
                (vehicle_path < soft[:, 0])
                | (vehicle_path > soft[:, 1])
            )
        )
        hard_violation = bool(
            np.any(
                (vehicle_path < hard[:, 0])
                | (vehicle_path > hard[:, 1])
            )
        )
        vehicle_min_clearance = (
            float(finite_clearance.min()) if len(finite_clearance) else None
        )
        records.append({
            "temperature": float(tau),
            "gamma": float(gamma),
            "episode": int(episode),
            "seed": seed,
            "reference_status": reference["status"],
            "reference_mode": reference["mode"],
            "reference_window_validity": float(reference["window_validity"]),
            "reference_min_clearance_m": reference["min_clearance_m"],
            "reference_time_to_goal_s": reference["time_to_goal_s"],
            "vehicle_status": vehicle_status,
            "vehicle_mode": vehicle_mode,
            "vehicle_window_validity": _vehicle_window_validity(
                config, vehicle, gamma,
            ),
            "vehicle_min_clearance_m": vehicle_min_clearance,
            "vehicle_time_to_goal_s": (
                float(metrics["duration_s"])
                if vehicle_status == "SUCCESS" else None
            ),
            "vehicle_soft_geofence_violation": soft_violation,
            "vehicle_hard_geofence_violation": hard_violation,
            "vehicle_outcome": vehicle["outcome"],
            "tracking_rmse_m": (
                float(np.sqrt(np.mean(tracking_error ** 2)))
                if len(tracking_error) else None
            ),
            "tracking_max_error_m": (
                float(tracking_error.max()) if len(tracking_error) else None
            ),
            "clearance_erosion_m": (
                reference["min_clearance_m"] - vehicle_min_clearance
                if (
                    reference["min_clearance_m"] is not None
                    and vehicle_min_clearance is not None
                )
                else None
            ),
            "reference_positions": dense_reference,
            "vehicle_positions": vehicle_path,
        })
    return records


def _mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize(records):
    summaries = []
    cells = sorted({
        (row["temperature"], row["gamma"]) for row in records
    }, reverse=True)
    for tau, gamma in cells:
        group = [
            row for row in records
            if row["temperature"] == tau and row["gamma"] == gamma
        ]
        reference_successes = [
            row for row in group if row["reference_status"] == "SUCCESS"
        ]
        vehicle_successes = [
            row for row in group if row["vehicle_status"] == "SUCCESS"
        ]
        reference_modes = {
            row["reference_mode"] for row in reference_successes
            if row["reference_mode"] in LAB_ROUTE_MODES
        }
        vehicle_modes = {
            row["vehicle_mode"] for row in vehicle_successes
            if row["vehicle_mode"] in LAB_ROUTE_MODES
        }
        summary = {
            "temperature": tau,
            "gamma": gamma,
            "episodes": len(group),
            "reference_SR": float(np.mean([
                row["reference_status"] == "SUCCESS" for row in group
            ])),
            "reference_CR": float(np.mean([
                row["reference_status"] == "COLLISION" for row in group
            ])),
            "reference_OOB": float(np.mean([
                row["reference_status"] == "OOB" for row in group
            ])),
            "reference_window_validity": _mean([
                row["reference_window_validity"] for row in group
            ]),
            "reference_route_coverage": len(reference_modes) / len(LAB_ROUTE_MODES),
            "reference_clearance_success_m": _mean([
                row["reference_min_clearance_m"] for row in reference_successes
            ]),
            "reference_time_success_s": _mean([
                row["reference_time_to_goal_s"] for row in reference_successes
            ]),
            "vehicle_SR": float(np.mean([
                row["vehicle_status"] == "SUCCESS" for row in group
            ])),
            "vehicle_CR": float(np.mean([
                row["vehicle_status"] == "COLLISION" for row in group
            ])),
            "vehicle_OOB": float(np.mean([
                row["vehicle_status"] == "OOB" for row in group
            ])),
            "vehicle_soft_geofence_violation": float(np.mean([
                row["vehicle_soft_geofence_violation"] for row in group
            ])),
            "vehicle_hard_geofence_violation": float(np.mean([
                row["vehicle_hard_geofence_violation"] for row in group
            ])),
            "vehicle_window_validity": _mean([
                row["vehicle_window_validity"] for row in group
            ]),
            "vehicle_route_coverage": (
                len(vehicle_modes) / len(LAB_ROUTE_MODES)
            ),
            "vehicle_clearance_success_m": _mean([
                row["vehicle_min_clearance_m"] for row in vehicle_successes
            ]),
            "vehicle_time_success_s": _mean([
                row["vehicle_time_to_goal_s"] for row in vehicle_successes
            ]),
            "tracking_RMSE_m": _mean([
                row["tracking_rmse_m"] for row in group
            ]),
            "tracking_max_error_p95_m": float(np.percentile([
                row["tracking_max_error_m"] for row in group
            ], 95)),
            "clearance_erosion_m": _mean([
                row["clearance_erosion_m"] for row in group
            ]),
            "reference_route_counts": {
                mode: sum(
                    row["reference_status"] == "SUCCESS"
                    and row["reference_mode"] == mode
                    for row in group
                )
                for mode in LAB_ROUTE_MODES
            },
            "vehicle_route_counts": {
                mode: sum(
                    row["vehicle_status"] == "SUCCESS"
                    and row["vehicle_mode"] == mode
                    for row in group
                )
                for mode in LAB_ROUTE_MODES
            },
        }
        summaries.append(summary)
    return summaries


def _draw_sphere(axis, sphere):
    u = np.linspace(0.0, 2.0 * np.pi, 16)
    v = np.linspace(0.0, np.pi, 9)
    axis.plot_surface(
        sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
        sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
        sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
        color="#a3a8ae",
        alpha=0.30,
        linewidth=0,
    )


def plot_trajectories(records, config, temperatures, gammas, output, shown=3):
    env = TaskEnvironment(config)
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(4.1 * len(gammas), 3.8 * len(temperatures)))
    for row_index, tau in enumerate(temperatures):
        for column, gamma in enumerate(gammas):
            axis = fig.add_subplot(
                len(temperatures),
                len(gammas),
                row_index * len(gammas) + column + 1,
                projection="3d",
            )
            group = sorted(
                (
                    row for row in records
                    if row["temperature"] == tau and row["gamma"] == gamma
                ),
                key=lambda row: row["seed"],
            )[:shown]
            _draw_sphere(axis, env.spheres[0])
            for index, record in enumerate(group):
                reference = record["reference_positions"]
                vehicle = record["vehicle_positions"]
                axis.plot(
                    reference[:, 0],
                    reference[:, 1],
                    reference[:, 2],
                    color="#2e3440",
                    ls="--",
                    lw=1.2,
                    alpha=0.70,
                    label="reference sim" if index == 0 else None,
                )
                axis.plot(
                    vehicle[:, 0],
                    vehicle[:, 1],
                    vehicle[:, 2],
                    color="#0f9d8a",
                    lw=1.45,
                    alpha=0.80,
                    label="native vehicle" if index == 0 else None,
                )
                for path, status in (
                    (reference, record["reference_status"]),
                    (vehicle, record["vehicle_status"]),
                ):
                    if status != "SUCCESS":
                        axis.scatter(
                            *path[-1],
                            marker="x",
                            color="#c43b2b",
                            s=26,
                        )
            axis.scatter(
                *env.start[:3],
                marker="s",
                color="#111111",
                s=20,
            )
            axis.scatter(
                *env.goal,
                marker="*",
                color="#ffca28",
                edgecolor="#6a4e00",
                s=70,
            )
            axis.set_xlim(*env.bounds[0])
            axis.set_ylim(*env.bounds[1])
            axis.set_zlim(*env.bounds[2])
            axis.set_box_aspect(tuple(env.bounds[:, 1] - env.bounds[:, 0]))
            axis.tick_params(labelsize=7)
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$", fontsize=14)
            if column == 0:
                axis.text2D(
                    -0.18,
                    0.5,
                    rf"$\tau={tau:g}$",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=14,
                )
            if row_index == len(temperatures) - 1:
                axis.set_xlabel("$x$ [m]")
                axis.set_ylabel("$y$ [m]")
                axis.set_zlabel("$z$ [m]")
    fig.axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(
        "Lab-native flow: reference simulation and native vehicle",
        fontsize=18,
    )
    fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.975))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(summaries, temperatures, gammas, output):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })
    specs = (
        ("SR", "Success rate"),
        ("CR", "Collision rate"),
        ("OOB", "Task-space violation"),
        ("window_validity", "Window validity"),
        ("route_coverage", "Route coverage"),
        ("clearance_success_m", "Successful clearance [m]"),
        ("time_success_s", "Successful time [s]"),
        ("tracking_RMSE_m", "Vehicle-reference RMSE [m]"),
    )
    colors = plt.get_cmap("plasma")(
        np.linspace(0.08, 0.92, len(gammas))
    )
    x = np.asarray(sorted(temperatures))
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6), squeeze=False)
    for axis, (suffix, title) in zip(axes.flat, specs):
        for gamma, color in zip(gammas, colors):
            cells = {
                row["temperature"]: row
                for row in summaries if row["gamma"] == gamma
            }
            if suffix == "tracking_RMSE_m":
                axis.plot(
                    x,
                    [cells[t][suffix] for t in x],
                    color=color,
                    marker="o",
                    lw=1.5,
                    label=rf"$\gamma={gamma:g}$",
                )
            else:
                axis.plot(
                    x,
                    [cells[t][f"reference_{suffix}"] for t in x],
                    color=color,
                    ls="--",
                    marker=".",
                    lw=1.1,
                )
                axis.plot(
                    x,
                    [cells[t][f"vehicle_{suffix}"] for t in x],
                    color=color,
                    marker="o",
                    lw=1.5,
                    label=rf"$\gamma={gamma:g}$",
                )
        axis.set_title(title)
        axis.set_xlabel(r"latent scale $\tau$")
        axis.grid(alpha=0.22)
        if suffix in {"SR", "CR", "OOB", "window_validity", "route_coverage"}:
            axis.set_ylim(-0.03, 1.03)
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    axes[0, 1].plot([], [], color="#333333", ls="--", label="reference sim")
    axes[0, 1].plot([], [], color="#333333", label="native vehicle")
    axes[0, 1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[1.0, 0.7, 0.5, 0.3],
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument("--shown-trajectories", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.episodes < 1 or args.shown_trajectories < 1:
        raise ValueError("episodes and shown trajectories must be positive")
    if any(not np.isfinite(value) or value < 0.0 for value in args.temperatures):
        raise ValueError("temperatures must be finite and nonnegative")
    args.output.mkdir(parents=True)

    lock = verify_deploy_sim_lock(ROOT)
    manifest = json.loads(
        (args.pretrain_dir / "pretrain_manifest.json").read_text()
    )
    if manifest["policy_output"] != "pre_smoothing_raw_acceleration_command":
        raise ValueError("temperature sweep requires the raw-command checkpoint")
    demo_dir = Path(manifest["source_demo_dir"])
    config = load_config(demo_dir / "resolved_config.json")
    policy = load_lab_reference_policy(args.pretrain_dir / "pretrained.pt")
    checkpoint = torch.load(
        args.pretrain_dir / "pretrained.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["contract"]["stateful_governor_in_policy"]:
        raise ValueError("policy checkpoint unexpectedly contains governor state")

    temperatures = [float(value) for value in args.temperatures]
    gammas = [float(value) for value in config.data.gammas]
    records = []
    for tau in temperatures:
        for gamma in gammas:
            print(f"[evaluate] tau={tau:g} gamma={gamma:g}", flush=True)
            records.extend(evaluate_cell(
                policy,
                config,
                tau,
                gamma,
                args.episodes,
                args.seed,
            ))
    summaries = summarize(records)

    scalar_fields = [
        key for key, value in records[0].items()
        if not isinstance(value, np.ndarray)
    ]
    with (args.output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows([
            {key: row[key] for key in scalar_fields} for row in records
        ])
    summary_fields = [
        key for key, value in summaries[0].items()
        if not isinstance(value, dict)
    ]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows([
            {key: row[key] for key in summary_fields} for row in summaries
        ])
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    plot_trajectories(
        records,
        config,
        temperatures,
        gammas,
        args.output / "temperature_gamma_3d.png",
        shown=args.shown_trajectories,
    )
    plot_metrics(
        summaries,
        temperatures,
        gammas,
        args.output / "temperature_metrics.png",
    )
    delivery = {
        "kind": "lab flow sampling-temperature deployment sweep",
        "sampling_temperature_definition": "x0 ~ N(0, tau^2 I)",
        "temperatures": temperatures,
        "gammas": gammas,
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "pretrained_checkpoint": str(
            (args.pretrain_dir / "pretrained.pt").resolve()
        ),
        "pretrained_sha256": sha256_file(
            args.pretrain_dir / "pretrained.pt"
        ),
        "deploy_sim_lock": lock,
        "deploy_sim_modified": False,
    }
    (args.output / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(delivery, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
