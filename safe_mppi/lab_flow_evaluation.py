"""Raw temperature-1 evaluation for Minhyuk-frame flow expansion."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import torch

from .ball_flow_theta import start_goal_frame
from .committed_success_visualize import (
    COMMITTED_COLOR,
    COMMITTED_WINDOW_COLOR,
    FAILED_COLOR,
    OTHER_SUCCESS_COLOR,
    resolve_committed_success,
)
from .environment import TaskEnvironment
from .lab_flow_expansion import (
    LabExpansionPolicyAdapter,
    LabFlowExpansionTask,
)
from .lab_reference_flow_task import raw_reference_rollout
from .lab_visual_flow import load_lab_reference_policy


ROUTE_MODES = ("below", "above", "left", "right")
MODE_COLORS = {
    "below": "#1468b3",
    "above": "#c8321b",
    "left": "#17964b",
    "right": "#8a3ffc",
    "none": "#9aa0a6",
}
Z95 = 1.959963984540054


def _checkpoint_policy(
    pretrain_dir: Path,
    expansion_dir: Path,
    round_i: int,
):
    policy = load_lab_reference_policy(pretrain_dir / "pretrained.pt")
    payload = torch.load(
        expansion_dir / f"checkpoint_{round_i:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy.load_state_dict(payload["model"], strict=True)
    return policy.eval()


def _summary(rows: list[dict], probe_rows: list[dict]) -> dict:
    episodes = len(rows)
    successes = [row for row in rows if row["status"] == "SUCCESS"]
    counts = {
        mode: sum(
            row["status"] == "SUCCESS" and row["mode"] == mode
            for row in rows
        )
        for mode in ROUTE_MODES
    }
    clearances = [
        row["min_clearance_m"] for row in successes
        if row["min_clearance_m"] is not None
    ]
    times = [
        row["time_to_goal_s"] for row in successes
        if row["time_to_goal_s"] is not None
    ]
    observed_modes = {
        row["mode"] for row in successes if row["mode"] in ROUTE_MODES
    }
    return {
        "episodes": episodes,
        "SR": float(len(successes) / episodes),
        "CR": float(sum(row["status"] == "COLLISION" for row in rows) / episodes),
        "OOB": float(sum(row["status"] == "OOB" for row in rows) / episodes),
        "timeout": float(sum(row["status"] == "TIMEOUT" for row in rows) / episodes),
        "window_validity": float(np.mean([
            row["window_validity"] for row in rows
        ])),
        "probe_validity": float(np.mean([
            row["valid"] for row in probe_rows
        ])) if probe_rows else None,
        "route_counts": counts,
        "route_coverage": float(len(observed_modes) / len(ROUTE_MODES)),
        "above_success_rate": float(counts["above"] / episodes),
        "above_share_among_success": (
            float(counts["above"] / len(successes)) if successes else 0.0
        ),
        "minimum_successful_mode_share": (
            float(min(counts.values()) / len(successes))
            if successes else 0.0
        ),
        "successful_min_clearance_m": (
            float(np.mean(clearances)) if clearances else None
        ),
        "successful_time_to_goal_s": (
            float(np.mean(times)) if times else None
        ),
    }


def _raw_rows(policy, config, gammas, episodes, seed):
    rows = []
    for gamma in gammas:
        for episode in range(episodes):
            result = raw_reference_rollout(
                policy,
                config,
                float(gamma),
                int(seed) + 37 * episode,
                sampling_temperature=1.0,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(episode),
                **result,
            })
    return rows


@torch.no_grad()
def _probe_rows(policy, config, gammas, samples, seed):
    env = TaskEnvironment(config)
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        tight_corridor=True,
    )
    wrapped = LabExpansionPolicyAdapter(policy)
    delta = env.goal - env.start[:3]
    probes = [
        env.start.copy(),
        np.concatenate([
            env.start[:3] + 0.25 * delta,
            np.zeros(3, np.float32),
        ]),
        np.concatenate([
            env.start[:3] + 0.75 * delta,
            np.zeros(3, np.float32),
        ]),
    ]
    generator = torch.Generator().manual_seed(int(seed))
    rows = []
    for gamma in gammas:
        for probe_index, state6 in enumerate(probes):
            state = task.reset(float(gamma), probe_index, int(seed))
            state["x"] = np.asarray(state6, np.float32)
            context = task.context(state, float(gamma))
            candidates = wrapped.sample(
                context, int(samples), generator, base_std=1.0,
            )
            results = task.verify(context, candidates, float(gamma))
            rows.extend({
                "gamma": float(gamma),
                "probe": int(probe_index),
                "sample": int(sample_index),
                "valid": bool(result.valid),
                "margin": float(result.margin),
            } for sample_index, result in enumerate(results))
    return rows


def _mean_se(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, 0
    se = (
        float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1 else 0.0
    )
    return float(values.mean()), se, int(len(values))


def _wilson(mean, count):
    mean = np.asarray(mean, float)
    count = np.asarray(count, float)
    denominator = 1.0 + Z95 ** 2 / count
    center = (mean + Z95 ** 2 / (2.0 * count)) / denominator
    radius = (
        Z95
        * np.sqrt(
            mean * (1.0 - mean) / count
            + Z95 ** 2 / (4.0 * count ** 2)
        )
        / denominator
    )
    return center - radius, center + radius


def _paper_rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": [
            "cmr10", "Computer Modern Roman", "DejaVu Serif",
        ],
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
    })


def _plot_curves(per_round_rows, summaries, gammas, output):
    del summaries
    rounds = sorted(per_round_rows)
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(gammas) - 1, 1)
        )
        for index, gamma in enumerate(gammas)
    }
    specs = (
        ("CR", "Collision rate", (-0.03, 1.03)),
        ("validity", "Validity", (-0.03, 1.03)),
        ("clearance", "Min. clearance [m]", None),
        ("time", "Time-to-goal [s]", None),
    )
    _paper_rcparams()

    def values(round_i, gamma, metric):
        rows = [
            row for row in per_round_rows[round_i]
            if row["gamma"] == gamma
        ]
        if metric == "CR":
            return [row["status"] == "COLLISION" for row in rows]
        if metric == "validity":
            return [row["window_validity"] for row in rows]
        successes = [row for row in rows if row["status"] == "SUCCESS"]
        key = (
            "min_clearance_m"
            if metric == "clearance" else "time_to_goal_s"
        )
        return [
            row[key] for row in successes if row[key] is not None
        ]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.8))
    for axis, (metric, title, ylim) in zip(axes.flat, specs):
        for gamma in gammas:
            statistics = [
                _mean_se(values(round_i, gamma, metric))
                for round_i in rounds
            ]
            mean = np.asarray([item[0] for item in statistics], float)
            se = np.asarray([item[1] for item in statistics], float)
            count = np.asarray([item[2] for item in statistics], float)
            axis.plot(
                rounds, mean, color=colors[gamma], lw=1.35, alpha=0.75,
            )
            if metric == "CR":
                lower, upper = _wilson(mean, count)
            else:
                lower, upper = mean - Z95 * se, mean + Z95 * se
                if metric == "validity":
                    lower = np.clip(lower, 0.0, 1.0)
                    upper = np.clip(upper, 0.0, 1.0)
            axis.fill_between(
                rounds, lower, upper, color=colors[gamma],
                alpha=0.18, linewidth=0,
            )
        pooled_statistics = [
            _mean_se([
                value
                for gamma in gammas
                for value in values(round_i, gamma, metric)
            ])
            for round_i in rounds
        ]
        pooled = np.asarray(
            [item[0] for item in pooled_statistics], float,
        )
        pooled_se = np.asarray(
            [item[1] for item in pooled_statistics], float,
        )
        pooled_count = np.asarray(
            [item[2] for item in pooled_statistics], float,
        )
        axis.plot(rounds, pooled, color="black", lw=3.0)
        if metric == "CR":
            lower, upper = _wilson(pooled, pooled_count)
        else:
            lower = pooled - Z95 * pooled_se
            upper = pooled + Z95 * pooled_se
            if metric == "validity":
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)
        axis.fill_between(
            rounds, lower, upper, color="black",
            alpha=0.14, linewidth=0,
        )
        axis.set_title(title)
        axis.set_xlabel("Expansion round")
        axis.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
    handles = [
        Line2D(
            [0], [0], color=colors[gamma], lw=2.2,
            label=rf"$\gamma={gamma:g}$",
        )
        for gamma in gammas
    ]
    handles.append(
        Line2D([0], [0], color="black", lw=3.0, label="pooled"),
    )
    fig.legend(
        handles=handles, ncol=5, loc="upper center", frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output = Path(output)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".figure.json").write_text(json.dumps({
        "status": "LAB_RAW_TEMPERATURE1_B1_STYLE_COMPLETE",
        "rounds": rounds,
        "gammas": [float(gamma) for gamma in gammas],
        "raw_rollouts_per_gamma_round": len([
            row for row in per_round_rows[rounds[0]]
            if row["gamma"] == gammas[0]
        ]),
        "claim": (
            "fixed raw temperature-1 rollouts; Validity is mean executed "
            "window validity; acquisition tilt and execution-time verifier "
            "are not used in this evaluation; Collision rate means physical "
            "sphere collision only, while taskspace/geofence OOB is reported "
            "separately in raw_eval.json and raw_sr_coverage"
        ),
        "confidence_bands": (
            "95% Wilson for collision rate; trajectory-level mean +/- "
            "1.96 SE for validity, successful clearance, and time-to-goal"
        ),
    }, indent=2) + "\n")


def _plot_gallery(env, per_round_rows, rounds, gammas, output):
    _paper_rcparams()
    frame = start_goal_frame(env)
    center_local = (env.spheres[0, :3] - env.start[:3]) @ frame
    goal_local = (env.goal - env.start[:3]) @ frame
    radius = float(env.spheres[0, 3])
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    fig, axes = plt.subplots(
        len(rounds),
        len(gammas),
        figsize=(4.0 * len(gammas), 3.0 * len(rounds)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_index, round_i in enumerate(rounds):
        for column_index, gamma in enumerate(gammas):
            axis = axes[row_index, column_index]
            axis.fill(
                center_local[0] + radius * np.cos(theta),
                center_local[2] + radius * np.sin(theta),
                color="#9aa0a6",
                alpha=0.45,
            )
            for row in per_round_rows[round_i]:
                if row["gamma"] != gamma:
                    continue
                local = (
                    np.asarray(row["states"])[:, :3] - env.start[:3]
                ) @ frame
                axis.plot(
                    local[:, 0],
                    local[:, 2],
                    color=MODE_COLORS[row["mode"]],
                    lw=1.0,
                    alpha=0.65,
                )
                if row["status"] != "SUCCESS":
                    axis.scatter(
                        local[-1, 0],
                        local[-1, 2],
                        marker="x",
                        color="#c8321b",
                        s=18,
                    )
            axis.scatter(
                0.0, 0.0, marker="s", color="#111111", s=28, zorder=8,
            )
            axis.scatter(
                goal_local[0], goal_local[2], marker="*",
                color="#ffca28", edgecolor="#6a4e00", s=115, zorder=8,
            )
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="box")
            corners = np.asarray([
                [x, y, z]
                for x in env.bounds[0]
                for y in env.bounds[1]
                for z in env.bounds[2]
            ])
            longitudinal = (corners - env.start[:3]) @ frame[:, 0]
            axis.set_xlim(
                float(longitudinal.min()) - 0.1,
                float(longitudinal.max()) + 0.1,
            )
            axis.set_ylim(
                env.bounds[2, 0] - env.start[2] - 0.1,
                env.bounds[2, 1] - env.start[2] + 0.1,
            )
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$")
            if column_index == 0:
                axis.set_ylabel(f"round {round_i}\nvertical [m]")
            if row_index == len(rounds) - 1:
                axis.set_xlabel("start-goal axis [m]")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(Path(output).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_crossing_fan(env, per_round_rows, rounds, gammas, output):
    """Head-on view of raw successful routes, colored by safety level."""
    _paper_rcparams()
    frame = start_goal_frame(env)
    center_local = (env.spheres[0, :3] - env.start[:3]) @ frame
    radius = float(env.spheres[0, 3])
    gamma_colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(gammas) - 1, 1)
        )
        for index, gamma in enumerate(gammas)
    }
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    fig, axes = plt.subplots(
        1, len(rounds), figsize=(5.0 * len(rounds), 4.5),
        squeeze=False, sharex=True, sharey=True,
    )
    for axis, round_i in zip(axes.flat, rounds):
        axis.fill(
            center_local[1] + radius * np.cos(theta),
            center_local[2] + radius * np.sin(theta),
            color="#9aa0a6", alpha=0.45,
        )
        for row in per_round_rows[round_i]:
            if row["status"] != "SUCCESS":
                continue
            local = (
                np.asarray(row["states"])[:, :3] - env.start[:3]
            ) @ frame
            axis.plot(
                local[:, 1], local[:, 2],
                color=gamma_colors[float(row["gamma"])],
                lw=1.0, alpha=0.58,
            )
        axis.set_title(f"round {round_i}")
        axis.set_xlabel("transverse [m]")
        axis.grid(alpha=0.2)
        axis.set_aspect("equal", adjustable="box")
    axes[0, 0].set_ylabel("vertical [m]")
    handles = [
        Line2D(
            [0], [0], color=gamma_colors[gamma], lw=2.2,
            label=rf"$\gamma={gamma:g}$",
        )
        for gamma in gammas
    ]
    fig.legend(
        handles=handles, ncol=len(gammas), loc="upper center",
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output = Path(output)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_sr_coverage(per_round_rows, gammas, output):
    rounds = sorted(per_round_rows)
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(gammas) - 1, 1)
        )
        for index, gamma in enumerate(gammas)
    }
    fig, axes = plt.subplots(1, 4, figsize=(17.8, 4.1), sharex=True)
    specs = (
        ("SR", "Raw success rate"),
        ("OOB", "Taskspace OOB rate"),
        ("coverage", "Successful route coverage"),
        ("above", "Above success rate"),
    )
    for gamma in gammas:
        series = {key: [] for key, _ in specs}
        for round_i in rounds:
            rows = [
                row for row in per_round_rows[round_i]
                if row["gamma"] == gamma
            ]
            successes = [
                row for row in rows if row["status"] == "SUCCESS"
            ]
            modes = {
                row["mode"] for row in successes
                if row["mode"] in ROUTE_MODES
            }
            series["SR"].append(len(successes) / len(rows))
            series["OOB"].append(
                sum(row["status"] == "OOB" for row in rows) / len(rows)
            )
            series["coverage"].append(len(modes) / len(ROUTE_MODES))
            series["above"].append(
                sum(row["mode"] == "above" for row in successes) / len(rows)
            )
        for axis, (key, title) in zip(axes, specs):
            axis.plot(
                rounds, series[key], color=colors[gamma],
                marker="o", ms=3, lw=1.5,
                label=rf"$\gamma={gamma:g}$",
            )
            axis.set_title(title)
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlabel("Expansion round")
            axis.grid(alpha=0.22)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _local_path(path, env, frame):
    return (np.asarray(path, float) - env.start[:3]) @ frame


def _draw_local_scene(side, head, env, frame):
    center = (env.spheres[0, :3] - env.start[:3]) @ frame
    goal = (env.goal - env.start[:3]) @ frame
    radius = float(env.spheres[0, 3])
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    side.fill(
        center[0] + radius * np.cos(theta),
        center[2] + radius * np.sin(theta),
        color="#9aa0a6", alpha=0.45,
    )
    head.fill(
        center[1] + radius * np.cos(theta),
        center[2] + radius * np.sin(theta),
        color="#9aa0a6", alpha=0.45,
    )
    for axis, first, second in (
        (side, 0, 2),
        (head, 1, 2),
    ):
        axis.scatter(
            0.0, 0.0, marker="s", color="#111111", s=27, zorder=10,
        )
        axis.scatter(
            goal[first], goal[second], marker="*", color="#ffca28",
            edgecolor="#6a4e00", s=110, zorder=10,
        )
        axis.grid(alpha=0.20)
        axis.set_aspect("equal", adjustable="box")
    length = float(np.linalg.norm(env.goal - env.start[:3]))
    side.set_xlim(-0.25, length + 0.35)
    side.set_ylim(
        env.bounds[2, 0] - env.start[2] - 0.1,
        env.bounds[2, 1] - env.start[2] + 0.1,
    )
    transverse = max(
        abs(float(env.bounds[0, 1] - env.bounds[0, 0])),
        abs(float(env.bounds[1, 1] - env.bounds[1, 0])),
    )
    head.set_xlim(0.5 * transverse, -0.5 * transverse)
    head.set_ylim(side.get_ylim())


def _draw_episode(
    side, head, episode, env, frame, *,
    color, linewidth, alpha=1.0, terminal=True,
):
    local = _local_path(episode.path, env, frame)
    side.plot(
        local[:, 0], local[:, 2], color=color,
        lw=linewidth, alpha=alpha,
    )
    head.plot(
        local[:, 1], local[:, 2], color=color,
        lw=linewidth, alpha=alpha,
    )
    if not terminal:
        return
    if episode.status == "SUCCESS":
        marker, marker_color, size = "*", "#159447", 55
    else:
        marker, marker_color, size = "x", "#c8321b", 42
    side.scatter(
        local[-1, 0], local[-1, 2], marker=marker,
        color=marker_color, s=size, zorder=9,
    )
    head.scatter(
        local[-1, 1], local[-1, 2], marker=marker,
        color=marker_color, s=size, zorder=9,
    )


def _replay_cells(committed_cells, round_i, gamma, replay_rounds):
    first = max(1, int(round_i) - int(replay_rounds) + 1)
    return [
        committed_cells[(prior_round, float(gamma))]
        for prior_round in range(first, int(round_i) + 1)
    ]


def _validate_replay_provenance(manifest, committed_cells):
    config = manifest["config"]
    if config["replay_selector"] != "uniform":
        raise ValueError(
            "exact Adam membership visualization requires uniform replay"
        )
    if config.get("inner_steps") is not None:
        raise ValueError(
            "exact Adam membership visualization requires no optimizer-row "
            "cap (--optimizer-steps-per-round must be unset)"
        )
    gammas = [float(value) for value in config["gammas"]]
    replay_rounds = int(config["replay_rounds"])
    for row in manifest["rounds"]:
        round_i = int(row["round"])
        count = sum(
            len(cell.committed_window_ids)
            for gamma in gammas
            for cell in _replay_cells(
                committed_cells, round_i, gamma, replay_rounds,
            )
        )
        if count != int(row["replay_positives"]):
            raise ValueError(
                "committed-window count does not equal the declared Adam "
                f"replay set at round {round_i}: {count} != "
                f"{row['replay_positives']}"
            )


def _plot_committed_gallery(
    env, committed_cells, rounds, gammas, output,
):
    """Current-round admissions: exact trajectories/windows entering replay."""
    _paper_rcparams()
    frame = start_goal_frame(env)
    fig, axes = plt.subplots(
        len(rounds), len(gammas),
        figsize=(4.2 * len(gammas), 3.15 * len(rounds)),
        squeeze=False, sharex=True, sharey=True,
    )
    for row_index, round_i in enumerate(rounds):
        for column_index, gamma in enumerate(gammas):
            axis = axes[row_index, column_index]
            dummy = axis.inset_axes([0, 0, 0.001, 0.001])
            _draw_local_scene(axis, dummy, env, frame)
            dummy.remove()
            cell = committed_cells[(int(round_i), float(gamma))]
            for episode in cell.episodes:
                local = _local_path(episode.path, env, frame)
                color = (
                    OTHER_SUCCESS_COLOR
                    if episode.status == "SUCCESS" else FAILED_COLOR
                )
                axis.plot(
                    local[:, 0], local[:, 2], color=color,
                    lw=0.7, alpha=0.32,
                )
                if episode.status != "SUCCESS":
                    axis.scatter(
                        local[-1, 0], local[-1, 2], marker="x",
                        color="#c8321b", s=14, alpha=0.55,
                    )
            for episode in cell.committed_episodes:
                local = _local_path(episode.path, env, frame)
                axis.plot(
                    local[:, 0], local[:, 2], color=COMMITTED_COLOR,
                    lw=2.8, alpha=0.95,
                )
            positions = _local_path(
                cell.committed_window_positions, env, frame,
            )
            if len(positions):
                axis.scatter(
                    positions[:, 0], positions[:, 2], s=13,
                    color=COMMITTED_WINDOW_COLOR, edgecolor="white",
                    linewidth=0.25, zorder=8,
                )
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$")
            if column_index == 0:
                axis.set_ylabel(f"round {round_i}\nvertical [m]")
            if row_index == len(rounds) - 1:
                axis.set_xlabel("start-goal axis [m]")
    handles = (
        Line2D(
            [0], [0], color=COMMITTED_COLOR, lw=3.0,
            label="committed SUCCESS trajectory",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=COMMITTED_WINDOW_COLOR,
            markeredgecolor="white", markersize=7,
            label="exact current-round Adam-admitted window start",
        ),
        Line2D(
            [0], [0], color=OTHER_SUCCESS_COLOR, lw=1.0,
            label="other gathered SUCCESS",
        ),
        Line2D(
            [0], [0], color=FAILED_COLOR, lw=1.0,
            label="failed gathering attempt",
        ),
    )
    fig.legend(
        handles=handles, ncol=4, loc="upper center", frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(output)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_mechanism_multiview(
    env, manifest, committed_cells, rounds, gamma, output,
):
    """Static exact W-round Adam replay provenance for one gamma."""
    frame = start_goal_frame(env)
    replay_rounds = int(manifest["config"]["replay_rounds"])
    fig, axes = plt.subplots(
        len(rounds), 2, figsize=(12.4, 4.0 * len(rounds)),
        squeeze=False,
    )
    for row_index, round_i in enumerate(rounds):
        side, head = axes[row_index]
        _draw_local_scene(side, head, env, frame)
        cells = _replay_cells(
            committed_cells, round_i, gamma, replay_rounds,
        )
        current = cells[-1]
        for cell in cells[:-1]:
            for episode in cell.committed_episodes:
                _draw_episode(
                    side, head, episode, env, frame,
                    color="#48a9b8", linewidth=1.25, alpha=0.55,
                    terminal=False,
                )
            positions = _local_path(
                cell.committed_window_positions, env, frame,
            )
            if len(positions):
                side.scatter(
                    positions[:, 0], positions[:, 2], s=12,
                    facecolors="none", edgecolors="#48a9b8",
                    linewidth=0.7,
                )
                head.scatter(
                    positions[:, 1], positions[:, 2], s=12,
                    facecolors="none", edgecolors="#48a9b8",
                    linewidth=0.7,
                )
        for episode in current.committed_episodes:
            _draw_episode(
                side, head, episode, env, frame,
                color=COMMITTED_COLOR, linewidth=2.8,
            )
        positions = _local_path(
            current.committed_window_positions, env, frame,
        )
        if len(positions):
            side.scatter(
                positions[:, 0], positions[:, 2], s=15,
                color=COMMITTED_WINDOW_COLOR, edgecolor="white",
                linewidth=0.25, zorder=8,
            )
            head.scatter(
                positions[:, 1], positions[:, 2], s=15,
                color=COMMITTED_WINDOW_COLOR, edgecolor="white",
                linewidth=0.25, zorder=8,
            )
        gamma_count = sum(len(cell.committed_window_ids) for cell in cells)
        row = manifest["rounds"][int(round_i) - 1]
        side.set_title(
            rf"round {round_i}, $\gamma={gamma:g}$: "
            f"{gamma_count} exact W={replay_rounds} replay windows\n"
            f"all gammas: {row['replay_positives']} unique rows, "
            f"{row['steps']} Adam steps"
        )
        head.set_title("head-on")
        side.set_xlabel("start-goal axis [m]")
        side.set_ylabel("vertical [m]")
        head.set_xlabel("transverse [m]")
        head.set_ylabel("vertical [m]")
    handles = (
        Line2D(
            [0], [0], color="#48a9b8", lw=1.5,
            label="prior committed trajectory retained by W",
        ),
        Line2D(
            [0], [0], color=COMMITTED_COLOR, lw=3.0,
            label="current committed trajectory",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=COMMITTED_WINDOW_COLOR,
            markeredgecolor="white", markersize=7,
            label="current Adam window start",
        ),
    )
    fig.legend(
        handles=handles, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, 0.005), frameon=False,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output = Path(output)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _compose_summary(output_dir):
    paths = (
        ("raw_crossing_fan.png", "Raw crossing fan"),
        ("raw_gallery.png", "Raw temperature-1 gallery"),
        ("raw_curves.png", "Raw temperature-1 metrics"),
        ("mechanism_multiview.png", "Exact Adam replay provenance"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 13.0))
    for axis, (name, title) in zip(axes.flat, paths):
        image = plt.imread(Path(output_dir) / name)
        axis.imshow(image)
        axis.set_title(title, fontsize=17)
        axis.axis("off")
    fig.tight_layout()
    output = Path(output_dir) / "ball_flow_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _mechanism_video(
    env,
    task,
    events,
    rounds,
    gamma,
    output,
    *,
    manifest,
    committed_cells,
):
    """Reference-style gathering movie with exact Adam replay provenance."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 11.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    selected_events = [
        event for event in events
        if int(event["round"]) in rounds
        and abs(float(event["gamma"]) - float(gamma)) < 1.0e-8
    ]
    if not selected_events:
        raise ValueError("no mechanism events match requested rounds/gamma")
    frame = start_goal_frame(env)
    replay_rounds = int(manifest["config"]["replay_rounds"])
    repeats = int(manifest["config"]["microbatch_repeats"])
    by_round_episode = {}
    for event in selected_events:
        key = (int(event["round"]), int(event["episode"]))
        by_round_episode.setdefault(key, []).append(event)
    for rows in by_round_episode.values():
        rows.sort(key=lambda event: int(event["step"]))

    round_sigma = {}
    for round_i in rounds:
        values = np.concatenate([
            np.asarray(event["sigma_K"], float)
            for event in selected_events
            if int(event["round"]) == int(round_i)
        ])
        low, high = np.quantile(values, [0.02, 0.98])
        if high <= low:
            high = low + 1.0e-12
        round_sigma[int(round_i)] = {
            "low": float(low),
            "high": float(high),
            "median": float(np.median(values)),
            "flat": bool(
                float(np.ptp(values))
                <= 1.0e-6 * max(1.0, abs(float(np.median(values))))
            ),
        }

    cmap = plt.get_cmap("viridis")
    normalized = Normalize(0.0, 1.0, clip=True)

    def sigma_fraction(value, round_i):
        stats = round_sigma[int(round_i)]
        if stats["flat"]:
            return 0.5
        return float(np.clip(
            (float(value) - stats["low"])
            / (stats["high"] - stats["low"]),
            0.0,
            1.0,
        ))

    fig = plt.figure(figsize=(13.2, 7.2))
    writer = FFMpegWriter(
        fps=9,
        codec="libx264",
        bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=135):
        for round_i in sorted(set(int(value) for value in rounds)):
            episodes = {
                episode: rows
                for (event_round, episode), rows in by_round_episode.items()
                if event_round == round_i
            }
            steps = sorted({
                int(event["step"])
                for rows in episodes.values()
                for event in rows
            })
            frame_count = min(36, len(steps))
            frame_indices = np.unique(np.linspace(
                0, len(steps) - 1, frame_count, dtype=int,
            ))
            frame_steps = [steps[index] for index in frame_indices]
            for frame_step in frame_steps:
                fig.clear()
                grid = fig.add_gridspec(
                    1, 2, width_ratios=(2.15, 1.0), wspace=0.18,
                )
                side = fig.add_subplot(grid[0, 0])
                head = fig.add_subplot(grid[0, 1])
                _draw_local_scene(side, head, env, frame)
                active = nvp = success = 0
                for episode, rows in sorted(episodes.items()):
                    past = [
                        event for event in rows
                        if int(event["step"]) <= frame_step
                    ]
                    if not past:
                        continue
                    segments_side = []
                    segments_head = []
                    segment_colors = []
                    for event in past:
                        if event["chosen_local"] is None:
                            continue
                        start = (
                            np.asarray(event["robot"][:3], float)
                            - env.start[:3]
                        ) @ frame
                        stop = (
                            np.asarray(event["robot_after"][:3], float)
                            - env.start[:3]
                        ) @ frame
                        chosen = event["selected"][event["chosen_local"]]
                        color = cmap(normalized(sigma_fraction(
                            event["sigma_K"][chosen], round_i,
                        )))
                        segments_side.append(
                            [[start[0], start[2]], [stop[0], stop[2]]],
                        )
                        segments_head.append(
                            [[start[1], start[2]], [stop[1], stop[2]]],
                        )
                        segment_colors.append(color)
                    if segments_side:
                        side.add_collection(LineCollection(
                            np.asarray(segments_side),
                            colors=segment_colors, linewidths=1.8,
                            alpha=0.88,
                        ))
                        head.add_collection(LineCollection(
                            np.asarray(segments_head),
                            colors=segment_colors, linewidths=1.8,
                            alpha=0.88,
                        ))
                    current = past[-1]
                    if (
                        current["status"] is None
                        and int(current["step"]) == frame_step
                    ):
                        active += 1
                    elif current["status"] == "NVP":
                        nvp += 1
                        local = (
                            np.asarray(current["robot"][:3], float)
                            - env.start[:3]
                        ) @ frame
                        side.scatter(
                            local[0], local[2], marker="x",
                            color="#c8321b", s=48, linewidth=1.7,
                        )
                        head.scatter(
                            local[1], local[2], marker="x",
                            color="#c8321b", s=48, linewidth=1.7,
                        )
                    elif current["status"] == "SUCCESS":
                        success += 1
                        local = (
                            np.asarray(current["robot_after"][:3], float)
                            - env.start[:3]
                        ) @ frame
                        side.scatter(
                            local[0], local[2], marker="*",
                            color="#159447", edgecolor="#0b5f2c", s=62,
                        )
                        head.scatter(
                            local[1], local[2], marker="*",
                            color="#159447", edgecolor="#0b5f2c", s=62,
                        )
                    if int(current["step"]) != frame_step:
                        continue
                    state6 = np.asarray(current["robot"], np.float32)
                    previous_applied = np.asarray(
                        current["context"][-6:-3], np.float32,
                    )
                    paths = []
                    for candidate in current["candidates"]:
                        states, _, _ = task._rollout_plan(
                            state6,
                            previous_applied,
                            np.asarray(candidate, np.float32),
                        )
                        paths.append(_local_path(
                            states[:, :3], env, frame,
                        ))
                    paths = np.asarray(paths)
                    for candidate_index, path in enumerate(paths):
                        if candidate_index in current["selected"]:
                            continue
                        side.plot(
                            path[:, 0], path[:, 2],
                            color="#9aa0a6", lw=0.35, alpha=0.16,
                        )
                        head.plot(
                            path[:, 1], path[:, 2],
                            color="#9aa0a6", lw=0.35, alpha=0.16,
                        )
                    for local_index, candidate_index in enumerate(
                        current["selected"],
                    ):
                        verification = current["verification"][local_index]
                        path = paths[candidate_index]
                        if verification["valid"]:
                            side.plot(
                                path[:, 0], path[:, 2],
                                color="#17964b", lw=3.0, alpha=0.42,
                            )
                            head.plot(
                                path[:, 1], path[:, 2],
                                color="#17964b", lw=3.0, alpha=0.42,
                            )
                        color = cmap(normalized(sigma_fraction(
                            current["sigma_K"][candidate_index], round_i,
                        )))
                        side.plot(
                            path[:, 0], path[:, 2],
                            color=color, lw=1.15, ls="--", alpha=0.96,
                        )
                        head.plot(
                            path[:, 1], path[:, 2],
                            color=color, lw=1.15, ls="--", alpha=0.96,
                        )
                    if current["chosen_local"] is not None:
                        chosen = current["selected"][
                            current["chosen_local"]
                        ]
                        path = paths[chosen]
                        side.plot(
                            path[:2, 0], path[:2, 2],
                            color="#1468b3", lw=3.2,
                        )
                        head.plot(
                            path[:2, 1], path[:2, 2],
                            color="#1468b3", lw=3.2,
                        )
                stats = round_sigma[round_i]
                manifest_row = manifest["rounds"][round_i - 1]
                side.set_title(
                    rf"round {round_i} | $\gamma={gamma:g}$ | "
                    f"synchronized step {frame_step} | "
                    f"active {active} | NVP {nvp} | success {success} | "
                    "\n" + r"raw $\sigma$ "
                    f"q02/med/q98={stats['low']:.3g}/"
                    f"{stats['median']:.3g}/{stats['high']:.3g} | "
                    f"ESS/K={manifest_row['ESS_over_K']:.3f}, "
                    f"uplift={manifest_row['uncertainty_uplift']:+.3f}",
                    fontsize=10.5,
                )
                head.set_title("head-on", fontsize=10.5)
                side.set_xlabel("start-goal axis [m]")
                side.set_ylabel("vertical [m]")
                head.set_xlabel("transverse [m]")
                head.set_ylabel("vertical [m]")
                legend = (
                    Line2D(
                        [0], [0], color="#9aa0a6", lw=0.8,
                        label=r"$K$ generated plans",
                    ),
                    Line2D(
                        [0], [0], color="#5b8fd6", lw=1.2, ls="--",
                        label=r"queried $B$ (normalized $\sigma$ color)",
                    ),
                    Line2D(
                        [0], [0], color="#17964b", lw=3.0, alpha=0.42,
                        label="full verifier positive",
                    ),
                    Line2D(
                        [0], [0], color="#1468b3", lw=3.2,
                        label="executed first step",
                    ),
                    Line2D(
                        [0], [0], color="#c8321b", marker="x", lw=0,
                        markersize=7, label="NVP",
                    ),
                )
                side.legend(
                    handles=legend, loc="lower right",
                    fontsize=7.5, framealpha=0.88,
                )
                scalar = plt.cm.ScalarMappable(norm=normalized, cmap=cmap)
                colorbar = fig.colorbar(
                    scalar, ax=(side, head), fraction=0.025, pad=0.035,
                )
                colorbar.set_label(
                    r"within-round normalized "
                    r"$\widetilde{\sigma}_n(\phi_s)$",
                    fontsize=10,
                )
                colorbar.ax.tick_params(labelsize=9)
                writer.grab_frame()

            fig.clear()
            grid = fig.add_gridspec(
                1, 2, width_ratios=(2.15, 1.0), wspace=0.18,
            )
            side = fig.add_subplot(grid[0, 0])
            head = fig.add_subplot(grid[0, 1])
            _draw_local_scene(side, head, env, frame)
            current_cell = committed_cells[(round_i, float(gamma))]
            for episode in current_cell.episodes:
                _draw_episode(
                    side, head, episode, env, frame,
                    color=(
                        OTHER_SUCCESS_COLOR
                        if episode.status == "SUCCESS" else FAILED_COLOR
                    ),
                    linewidth=0.75, alpha=0.32,
                )
            replay_cells = _replay_cells(
                committed_cells, round_i, gamma, replay_rounds,
            )
            for prior_cell in replay_cells[:-1]:
                for episode in prior_cell.committed_episodes:
                    _draw_episode(
                        side, head, episode, env, frame,
                        color="#48a9b8", linewidth=1.3, alpha=0.55,
                        terminal=False,
                    )
                prior_positions = _local_path(
                    prior_cell.committed_window_positions, env, frame,
                )
                if len(prior_positions):
                    side.scatter(
                        prior_positions[:, 0], prior_positions[:, 2],
                        s=12, facecolors="none", edgecolors="#48a9b8",
                        linewidth=0.7,
                    )
                    head.scatter(
                        prior_positions[:, 1], prior_positions[:, 2],
                        s=12, facecolors="none", edgecolors="#48a9b8",
                        linewidth=0.7,
                    )
            for episode in current_cell.committed_episodes:
                _draw_episode(
                    side, head, episode, env, frame,
                    color=COMMITTED_COLOR, linewidth=3.0,
                )
            positions = _local_path(
                current_cell.committed_window_positions, env, frame,
            )
            if len(positions):
                side.scatter(
                    positions[:, 0], positions[:, 2], s=16,
                    color=COMMITTED_WINDOW_COLOR, edgecolor="white",
                    linewidth=0.25, zorder=9,
                )
                head.scatter(
                    positions[:, 1], positions[:, 2], s=16,
                    color=COMMITTED_WINDOW_COLOR, edgecolor="white",
                    linewidth=0.25, zorder=9,
                )
            gamma_rows = sum(
                len(cell.committed_window_ids) for cell in replay_cells
            )
            manifest_row = manifest["rounds"][round_i - 1]
            side.set_title(
                rf"round {round_i} | $\gamma={gamma:g}$ | exact Adam input"
                "\n"
                f"{gamma_rows} gamma-slice rows | "
                f"{manifest_row['replay_positives']} all-gamma rows | "
                f"microbatch repeat {repeats}x",
                fontsize=10.5,
            )
            head.set_title("head-on", fontsize=10.5)
            side.set_xlabel("start-goal axis [m]")
            side.set_ylabel("vertical [m]")
            head.set_xlabel("transverse [m]")
            head.set_ylabel("vertical [m]")
            summary_legend = (
                Line2D(
                    [0], [0], color=COMMITTED_COLOR, lw=3.0,
                    label="current committed SUCCESS",
                ),
                Line2D(
                    [0], [0], marker="o", color="none",
                    markerfacecolor=COMMITTED_WINDOW_COLOR,
                    markeredgecolor="white", markersize=7,
                    label="current replay-window start",
                ),
                Line2D(
                    [0], [0], color="#48a9b8", lw=1.5,
                    label="prior committed evidence retained by W",
                ),
                Line2D(
                    [0], [0], color=FAILED_COLOR, lw=0.8,
                    label="failed gathering attempt (not Adam input)",
                ),
            )
            side.legend(
                handles=summary_legend, loc="lower right",
                fontsize=7.5, framealpha=0.88,
            )
            fig.suptitle(
                "Expansion gathering evidence - not raw evaluation",
                fontsize=13, weight="bold",
            )
            for _ in range(14):
                writer.grab_frame()
    plt.close(fig)


def evaluate_lab_expansion(args, config, pretrain_manifest, manifest):
    del pretrain_manifest
    env = TaskEnvironment(config)
    gammas = [float(value) for value in config.data.gammas]
    total_rounds = int(manifest["config"]["rounds"])
    eval_rounds = sorted({
        0,
        *range(0, total_rounds + 1, int(args.stride)),
        total_rounds,
    })
    per_round_rows, per_round_probes, summaries = {}, {}, {}
    for round_i in eval_rounds:
        policy = _checkpoint_policy(
            args.pretrain_dir, args.expansion, round_i,
        )
        rows = _raw_rows(
            policy, config, gammas, int(args.episodes), int(args.seed),
        )
        probes = _probe_rows(
            policy,
            config,
            gammas,
            int(args.probe_samples),
            int(args.seed) + 7,
        )
        per_round_rows[round_i] = rows
        per_round_probes[round_i] = probes
        per_gamma = {}
        for gamma in gammas:
            gamma_rows = [
                row for row in rows if row["gamma"] == gamma
            ]
            gamma_probes = [
                row for row in probes if row["gamma"] == gamma
            ]
            per_gamma[f"{gamma:g}"] = _summary(
                gamma_rows, gamma_probes,
            )
        summaries[str(round_i)] = {
            "pooled": _summary(rows, probes),
            "per_gamma": per_gamma,
        }
        pooled = summaries[str(round_i)]["pooled"]
        print(
            f"round {round_i:2d}: SR {pooled['SR']:.2f} "
            f"CR {pooled['CR']:.2f} validity "
            f"{pooled['window_validity']:.2f} above "
            f"{pooled['above_success_rate']:.2f}",
            flush=True,
        )

    output = args.expansion / "eval"
    output.mkdir(exist_ok=True)
    slim_rows = {
        str(round_i): [
            {
                key: value
                for key, value in row.items()
                if key not in {
                    "states",
                    "controls",
                    "applied_controls",
                    "dense_steps",
                }
            }
            for row in rows
        ]
        for round_i, rows in per_round_rows.items()
    }
    (output / "raw_eval.json").write_text(json.dumps({
        "status": "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE",
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "raw_tight_corridor_interpretation": (
            "configured lab taskspace/geofence only; no canonical ball corridor"
        ),
        "summary": summaries,
        "rows": slim_rows,
        "probe_rows": per_round_probes,
    }, indent=2) + "\n")
    if args.metrics_only:
        print("[outputs]", output, flush=True)
        return

    _plot_curves(
        per_round_rows,
        summaries,
        gammas,
        output / "raw_curves.png",
    )
    gallery_rounds = sorted({
        eval_rounds[0],
        eval_rounds[len(eval_rounds) // 2],
        eval_rounds[-1],
    })
    _plot_gallery(
        env,
        per_round_rows,
        gallery_rounds,
        gammas,
        output / "raw_gallery.png",
    )
    _plot_crossing_fan(
        env,
        per_round_rows,
        gallery_rounds,
        gammas,
        output / "raw_crossing_fan.png",
    )
    _plot_sr_coverage(
        per_round_rows,
        gammas,
        output / "raw_sr_coverage.png",
    )
    if not args.screening_only:
        events_path = args.expansion / "events.pt"
        if not events_path.is_file():
            raise FileNotFoundError(
                "mechanism video requires --event-log full during expansion"
            )
        video_rounds = (
            sorted(set(args.video_rounds))
            if args.video_rounds is not None
            else list(range(1, total_rounds + 1))
        )
        if any(
            round_i < 1 or round_i > total_rounds
            for round_i in video_rounds
        ):
            raise ValueError(
                f"--video-rounds must lie in [1,{total_rounds}]"
            )
        events = torch.load(events_path, weights_only=False)
        committed_cells = resolve_committed_success(manifest, events)
        _validate_replay_provenance(manifest, committed_cells)
        policy = _checkpoint_policy(
            args.pretrain_dir, args.expansion, total_rounds,
        )
        task = LabFlowExpansionTask(
            config,
            context_schema=policy.context_schema,
            tight_corridor=True,
        )
        committed_rounds = sorted({
            1,
            max(1, total_rounds // 2),
            total_rounds,
        })
        _plot_committed_gallery(
            env,
            committed_cells,
            committed_rounds,
            gammas,
            output / "gathering_committed_success_gallery.png",
        )
        _plot_mechanism_multiview(
            env,
            manifest,
            committed_cells,
            committed_rounds,
            float(args.video_gamma),
            output / "mechanism_multiview.png",
        )
        _mechanism_video(
            env,
            task,
            events,
            video_rounds,
            float(args.video_gamma),
            output / "mechanism.mp4",
            manifest=manifest,
            committed_cells=committed_cells,
        )
        _compose_summary(output)
    print("[outputs]", output, flush=True)
