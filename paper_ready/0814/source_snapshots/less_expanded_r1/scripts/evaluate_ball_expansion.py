"""Evaluate saved expansion checkpoints with an untilted raw-policy seed bank.

Per checkpoint (round): closed-loop temperature-1 rollouts (no verifier, no tilt) give raw
success/collision rates, route coverage, clearance, and time-to-goal; a raw open-loop probe at
fixed contexts gives untilted GREEN-verifier validity. Also renders: result curves, mode-colored
rollout galleries, the crossing fan per round, sigma-tilt mechanism figures, and the mechanism
video built from logged events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import (
    BallFlowTask, ROUTE_MODES, build_context, inside_above_halfspace,
    inside_above_wedge, load_policy, plan_states, raw_rollout,
    raw_window_validity_fraction, route_mode,
)
from safe_mppi.ball_flow_theta import (
    THETA_VALUES, ThetaBallFlowTask, raw_rollout_theta, requested_theta,
)
from safe_mppi.config import load_config
from safe_mppi.committed_success_visualize import (
    COMMITTED_COLOR,
    COMMITTED_WINDOW_COLOR,
    FAILED_COLOR,
    OTHER_SUCCESS_COLOR,
    draw_gathering_cell,
    has_committed_success_schema,
    plot_gathering_commit_gallery,
    resolve_committed_success,
)
from safe_mppi.environment import TaskEnvironment
from safe_mppi.expansion_visualize import (
    MechanismFrame,
    plot_expansion_results,
    plot_rollout_gallery,
    render_expansion_mechanism,
    round_sigma_statistics,
    within_round_normalized_sigma,
)
from safe_mppi.helios_remote import (
    add_helios_arguments,
    run_evaluation_on_helios,
)
from safe_mppi.visualize import PLASMA

ROOT = Path(__file__).resolve().parents[1]
MODE_COLORS = {"below": "#1468b3", "above": "#c8321b", "left": "#17964b",
               "right": "#8a3ffc", "none": "#9aa0a6"}
Z95 = 1.959963984540054


def checkpoint_policy(expansion_dir: Path, pretrain_dir: Path, round_i: int):
    policy = load_policy(pretrain_dir / "pretrained.pt")
    payload = torch.load(expansion_dir / f"checkpoint_{round_i:03d}.pt", weights_only=False)
    policy.load_state_dict(payload["model"])
    return policy


def raw_eval(policy, task, gammas, episodes: int, seed0: int,
             tight_corridor: bool = False):
    config = task.config
    rows = []
    for gamma in gammas:
        for episode in range(episodes):
            if isinstance(task, ThetaBallFlowTask):
                result = raw_rollout_theta(
                    policy, config, gamma, seed0 + 37 * episode,
                    requested_theta(episode),
                    tight_corridor=tight_corridor,
                )
            else:
                result = raw_rollout(
                    policy, config, gamma, seed0 + 37 * episode,
                    tight_corridor=tight_corridor,
                )
            rows.append({
                "gamma": float(gamma), "episode": episode, "status": result["status"],
                "mode": result["mode"], "min_clearance_m": result["min_clearance_m"],
                "time_to_goal_s": result["time_to_goal_s"],
                "physical_collision": bool(result["physical_collision"]),
                "corridor_violation": bool(result["corridor_violation"]),
                "requested_theta": result.get("requested_theta"),
                "requested_route": result.get("requested_route"),
                "v_safe": raw_window_validity_fraction(
                    config, result["states"], result["controls"], float(gamma)),
                "states": result["states"],
            })
    return rows


def validity_probe(policy, task: BallFlowTask, gammas, samples: int, seed: int):
    """Raw open-loop plans at fixed probe states, judged by the untilted GREEN verifier."""
    env = task.env
    probes = [np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], np.float32),
              np.array([1.0, 0.0, 2.0, 0.55, 0.0, 0.0], np.float32),
              np.array([1.35, 0.0, 1.9, 0.5, 0.0, -0.1], np.float32)]
    generator = torch.Generator().manual_seed(seed)
    out = []
    for gamma in gammas:
        for index, state in enumerate(probes):
            thetas = THETA_VALUES if isinstance(task, ThetaBallFlowTask) else (None,)
            for theta in thetas:
                if theta is None:
                    context = torch.from_numpy(build_context(env, state, float(gamma)))
                else:
                    context = task.context({
                        "x": state, "steps": 0, "collided": False, "oob": False,
                        "theta": float(theta), "requested_route": None,
                    }, float(gamma))
                candidates = policy.sample(context, samples, generator)
                results = task.verify(context, candidates, float(gamma))
                world_candidates = (
                    task.world_plan(candidates)
                    if isinstance(task, ThetaBallFlowTask) else candidates
                )
                for candidate, result in zip(world_candidates, results):
                    states = plan_states(env, state, candidate.numpy())
                    out.append({
                        "gamma": float(gamma), "probe": index,
                        "requested_theta": theta,
                        "valid": bool(result.valid), "margin": float(result.margin),
                        "mode": route_mode(env, states[:, :3]),
                    })
    return out


def summarize(rows, probe_rows):
    def rate(rows, key, value):
        return float(np.mean([row[key] == value for row in rows])) if rows else None
    modes = sorted({row["mode"] for row in rows
                    if row["status"] == "SUCCESS" and row["mode"] in ROUTE_MODES})
    clearances = [row["min_clearance_m"] for row in rows if row["min_clearance_m"] is not None]
    times = [row["time_to_goal_s"] for row in rows if row["time_to_goal_s"] is not None]
    physical_cr = float(np.mean([
        row.get("physical_collision", row["status"] == "COLLISION") for row in rows
    ])) if rows else None
    corridor_rate = float(np.mean([
        row.get("corridor_violation", row["status"] == "CORRIDOR_VIOLATION")
        for row in rows
    ])) if rows else None
    return {
        "SR": rate(rows, "status", "SUCCESS"),
        "CR": float(np.mean([
            row.get("physical_collision", row["status"] == "COLLISION")
            or row.get("corridor_violation", row["status"] == "CORRIDOR_VIOLATION")
            for row in rows
        ])) if rows else None,
        "physical_CR": physical_cr,
        "corridor_violation": corridor_rate,
        "OOB": rate(rows, "status", "OOB"), "timeout": rate(rows, "status", "TIMEOUT"),
        "modes": modes, "coverage": len(modes) / len(ROUTE_MODES),
        "theta_adherence": (
            float(np.mean([
                row["mode"] == row["requested_route"]
                for row in rows
                if row["status"] == "SUCCESS"
                and row.get("requested_route") is not None
            ]))
            if any(
                row["status"] == "SUCCESS"
                and row.get("requested_route") is not None
                for row in rows
            ) else None
        ),
        "avg_min_clearance_m": (float(np.mean(clearances)) if clearances else None),
        "avg_time_to_goal_s": (float(np.mean(times)) if times else None),
        "raw_validity": (float(np.mean([row["v_safe"] for row in rows]))
                         if rows else None),
        "open_loop_probe_validity": (
            float(np.mean([row["valid"] for row in probe_rows])) if probe_rows else None),
    }


def above_start_probe(policy, task, gammas, episodes: int, seed0: int,
                      tight_corridor: bool = False):
    """Raw closed-loop rollouts from a fixed above/ahead start: is the above corridor learned?"""
    start = np.array([1.05, 0.0, 2.42, 0.4, 0.0, 0.0], np.float32)
    rows = []
    for gamma in gammas:
        for episode in range(episodes):
            if isinstance(task, ThetaBallFlowTask):
                result = raw_rollout_theta(
                    policy, task.config, gamma, seed0 + 61 * episode,
                    requested_theta(episode), start=start,
                    tight_corridor=tight_corridor,
                )
            else:
                result = raw_rollout(
                    policy, task.config, gamma, seed0 + 61 * episode, start=start,
                    tight_corridor=tight_corridor,
                )
            rows.append({"gamma": float(gamma), "status": result["status"],
                         "mode": result["mode"]})
    above = [row for row in rows if row["mode"] == "above" and row["status"] == "SUCCESS"]
    return {"SR": float(np.mean([row["status"] == "SUCCESS" for row in rows])),
            "above_success_share": len(above) / len(rows)}


def draw_scene_factory(env: TaskEnvironment):
    def draw_scene(ax):
        sphere = np.asarray(env.spheres[0], float)
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 10)
        ax.plot_surface(sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
                        sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
                        sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
                        color="#8f969f", alpha=0.30, linewidth=0)
        ax.scatter(*env.start[:3], marker="s", color="#111111", s=25)
        ax.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=110)
        ax.set_xlim(*env.bounds[0])
        ax.set_ylim(*env.bounds[1])
        ax.set_zlim(*env.bounds[2])
        ax.set_box_aspect(tuple(env.bounds[:, 1] - env.bounds[:, 0]))
    return draw_scene


def _mean_se(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, 0
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), se, int(len(values))


def _wilson(mean, n):
    mean = np.asarray(mean, float)
    n = np.asarray(n, float)
    denominator = 1.0 + Z95 ** 2 / n
    center = (mean + Z95 ** 2 / (2.0 * n)) / denominator
    radius = Z95 * np.sqrt(mean * (1.0 - mean) / n + Z95 ** 2 / (4.0 * n ** 2)) / denominator
    return center - radius, center + radius


def raw_metric_figure(per_round_rows, per_round_probes, gammas, output):
    """Exact B1 paper style, using only fixed-bank raw temperature-1 evaluations."""
    rounds = sorted(per_round_rows)
    specs = (
        ("CR", "Collision rate", (-0.03, 1.03)),
        ("validity", "Validity", (-0.03, 1.03)),
        ("clearance", "Min. clearance [m]", None),
        ("time", "Time-to-goal [s]", None),
    )
    colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / (len(gammas) - 1))
        for index, gamma in enumerate(gammas)
    }
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.titlesize": 24, "axes.labelsize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
        "legend.fontsize": 16, "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
    })

    def values(round_i, gamma, metric):
        rows = [row for row in per_round_rows[round_i] if row["gamma"] == gamma]
        if metric == "CR":
            return [
                row.get("physical_collision", row["status"] == "COLLISION")
                or row.get("corridor_violation", row["status"] == "CORRIDOR_VIOLATION")
                for row in rows
            ]
        if metric == "validity":
            return [row["v_safe"] for row in rows]
        successes = [row for row in rows if row["status"] == "SUCCESS"]
        key = "min_clearance_m" if metric == "clearance" else "time_to_goal_s"
        return [row[key] for row in successes if row[key] is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.8), squeeze=False)
    for axis, (metric, title, ylim) in zip(axes.flat, specs):
        for gamma in gammas:
            statistics = [_mean_se(values(round_i, gamma, metric)) for round_i in rounds]
            mean = np.asarray([item[0] for item in statistics], float)
            se = np.asarray([item[1] for item in statistics], float)
            count = np.asarray([item[2] for item in statistics], float)
            axis.plot(rounds, mean, color=colors[gamma], lw=1.35, alpha=0.75)
            if metric == "CR":
                lower, upper = _wilson(mean, count)
            else:
                lower, upper = mean - Z95 * se, mean + Z95 * se
                if metric == "validity":
                    lower, upper = np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
            axis.fill_between(rounds, lower, upper, color=colors[gamma],
                              alpha=0.18, linewidth=0)
        pooled_statistics = [
            _mean_se([
                value
                for gamma in gammas
                for value in values(round_i, gamma, metric)
            ])
            for round_i in rounds
        ]
        pooled = np.asarray([item[0] for item in pooled_statistics], float)
        pooled_se = np.asarray([item[1] for item in pooled_statistics], float)
        pooled_count = np.asarray([item[2] for item in pooled_statistics], float)
        axis.plot(rounds, pooled, color="black", lw=3.0)
        if metric == "CR":
            pooled_lower, pooled_upper = _wilson(pooled, pooled_count)
        else:
            pooled_lower, pooled_upper = pooled - Z95 * pooled_se, pooled + Z95 * pooled_se
            if metric == "validity":
                pooled_lower = np.clip(pooled_lower, 0.0, 1.0)
                pooled_upper = np.clip(pooled_upper, 0.0, 1.0)
        axis.fill_between(rounds, pooled_lower, pooled_upper,
                          color="black", alpha=0.14, linewidth=0)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_xlabel("Expansion round")
    handles = [
        plt.Line2D([0], [0], color=colors[gamma], lw=2.2,
                   label=rf"$\gamma={gamma:g}$") for gamma in gammas
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=3.0, label="pooled"))
    fig.legend(handles=handles, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output = Path(output)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".figure.json").write_text(json.dumps({
        "status": "BALL_RAW_TEMPERATURE1_B1_STYLE_COMPLETE",
        "rounds": rounds, "gammas": [float(gamma) for gamma in gammas],
        "raw_rollouts_per_gamma_round": len([
            row for row in per_round_rows[rounds[0]] if row["gamma"] == gammas[0]
        ]),
        "claim": ("fixed raw temperature-1 rollouts; Validity is the mean of all executed "
                  "window indicators, with every start included and terminal horizons truncated "
                  "to H_t=min(10,T-t); each indicator requires task-space bounds, collision "
                  "avoidance, and GREEN certification only; no RBF acquisition, execution-time "
                  "verifier controller, candidate perturbation, or fallback; headline CR is "
                  "physical collision OR an enabled hard-corridor violation, while raw_eval.json "
                  "also retains both components"),
        "confidence_bands": ("95% Wilson for CR/corridor; trajectory-level mean +/- 1.96 SE "
                             "for fractional window validity, successful-trajectory clearance, "
                             "and time-to-goal"),
        "style_source": ("safe_flow_expansion/scripts/"
                         "paper_b1_margin50_trends.py"),
    }, indent=2) + "\n")


def mode_share_figure(per_round_rows, output):
    rounds = sorted(per_round_rows)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bottom = np.zeros(len(rounds))
    for mode in (*ROUTE_MODES, "none"):
        shares = []
        for round_i in rounds:
            rows = [row for row in per_round_rows[round_i] if row["status"] == "SUCCESS"]
            shares.append(np.mean([row["mode"] == mode for row in rows]) if rows else 0.0)
        shares = np.asarray(shares)
        ax.bar(rounds, shares, bottom=bottom, color=MODE_COLORS[mode], label=mode, width=0.8)
        bottom += shares
    ax.set_xlabel("expansion round")
    ax.set_ylabel("share of raw successful episodes")
    ax.set_title("Route-mode composition of raw successes", weight="bold")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def raw_sr_coverage_figure(per_round_rows, gammas, output):
    """Raw success, realized route coverage, and requested-theta adherence."""
    rounds = sorted(per_round_rows)
    colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / max(len(gammas) - 1, 1))
        for index, gamma in enumerate(gammas)
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), sharex=True)
    for gamma in gammas:
        sr, coverage, adherence = [], [], []
        for round_i in rounds:
            rows = [
                row for row in per_round_rows[round_i]
                if float(row["gamma"]) == float(gamma)
            ]
            successes = [row for row in rows if row["status"] == "SUCCESS"]
            sr.append(float(np.mean([row["status"] == "SUCCESS" for row in rows])))
            coverage.append(
                len({row["mode"] for row in successes if row["mode"] in ROUTE_MODES})
                / len(ROUTE_MODES)
            )
            adherence.append(
                float(np.mean([
                    row["mode"] == row["requested_route"] for row in successes
                ])) if successes and successes[0].get("requested_route") is not None
                else np.nan
            )
        for axis, values in zip(axes, (sr, coverage, adherence)):
            axis.plot(rounds, values, color=colors[gamma], lw=1.7,
                      label=rf"$\gamma={gamma:g}$")
    for axis, title in zip(
        axes, ("Success rate", "Realized route coverage", r"$\theta$ adherence"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Expansion round")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.23)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Fixed-bank raw temperature-1 policy evaluation")
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def crossing_fan_figure(env, per_round_rows, rounds, output):
    sphere = np.asarray(env.spheres[0], float)
    gammas = sorted({float(row["gamma"]) for rows in per_round_rows.values() for row in rows})
    gamma_colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / max(len(gammas) - 1, 1))
        for index, gamma in enumerate(gammas)
    }
    fig, axes = plt.subplots(1, len(rounds), figsize=(4.4 * len(rounds), 4.4),
                             sharey=True)
    theta = np.linspace(0, 2 * np.pi, 120)
    for ax, round_i in zip(np.atleast_1d(axes), rounds):
        ax.fill(sphere[1] + sphere[3] * np.cos(theta), sphere[2] + sphere[3] * np.sin(theta),
                color="#8f969f", alpha=0.45, zorder=2)
        ax.axhline(sphere[2], color="#cc3311", lw=0.9, ls="--")
        for row in per_round_rows[round_i]:
            if row["status"] != "SUCCESS":
                continue
            states = row["states"]
            ax.plot(states[:, 1], states[:, 2], color=gamma_colors[float(row["gamma"])],
                    lw=1.0, alpha=0.55)
        ax.set_xlim(0.9, -0.9)
        ax.set_ylim(sphere[2] - 0.9, sphere[2] + 0.9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"round {round_i}")
        ax.set_xlabel("y [m] (+y left)")
    np.atleast_1d(axes)[0].set_ylabel("z [m]")
    handles = [plt.Line2D([0], [0], color=gamma_colors[gamma], lw=2.0,
                          label=rf"$\gamma={gamma:g}$") for gamma in gammas]
    fig.legend(handles=handles, ncol=len(gammas), loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(r"Head-on raw successful trajectories by round (color = $\gamma$)",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def side_rollout_gallery(env, per_round_rows, rounds, gammas, output):
    """Readable x-z side view; unlike a flattened 3-D axis, it has no overlapping ticks."""
    sphere = np.asarray(env.spheres[0], float)
    fig, axes = plt.subplots(len(rounds), len(gammas),
                             figsize=(4.2 * len(gammas), 3.2 * len(rounds)),
                             sharex=True, sharey=True, squeeze=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    for row_i, round_i in enumerate(rounds):
        for column_i, gamma in enumerate(gammas):
            axis = axes[row_i, column_i]
            axis.fill(sphere[0] + sphere[3] * np.cos(theta),
                      sphere[2] + sphere[3] * np.sin(theta),
                      color="#8f969f", alpha=0.46)
            for row in per_round_rows[round_i]:
                if row["gamma"] != gamma:
                    continue
                states = np.asarray(row["states"], float)
                color = MODE_COLORS.get(row["mode"], "#9aa0a6")
                axis.plot(states[:, 0], states[:, 2], color=color, lw=1.05, alpha=0.64)
                if row["status"] != "SUCCESS":
                    axis.scatter(states[-1, 0], states[-1, 2], marker="x",
                                 color="#c8321b", s=23, linewidth=1.1)
            axis.scatter(env.start[0], env.start[2], marker="s", color="#111111", s=22)
            axis.scatter(env.goal[0], env.goal[2], marker="*", color="#ffca28",
                         edgecolor="#6a4e00", s=95)
            axis.set_xlim(-0.15, 3.2)
            axis.set_ylim(1.1, 2.95)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.20)
            if row_i == 0:
                axis.set_title(rf"$\gamma={gamma:g}$")
            if column_i == 0:
                axis.set_ylabel(f"round {round_i}\n" + r"$z$ [m]")
            if row_i == len(rounds) - 1:
                axis.set_xlabel(r"$x$ [m]")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def green_verifier_figure(env, config, events, output):
    """Legacy rebuilt-nominal-chain diagnostic; not the fitted-face verifier label."""
    from safe_mppi.ball_analysis import draw_polytope_soft
    from safe_mppi.ball_flow_task import nominal_hp_chain_margins
    from safe_mppi.geometry import NominalPolytope, build_nominal_polytope
    candidates_events = [event for event in events
                         if event["chosen_local"] is not None
                         and 1.05 <= event["robot"][0] <= 1.35
                         and abs(event["gamma"] - 0.3) < 1e-9]
    if not candidates_events:
        return False
    event = candidates_events[len(candidates_events) // 2]
    state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
    plan = event["candidates"][event["selected"][event["chosen_local"]]]
    states = plan_states(env, state6, plan)
    margins = nominal_hp_chain_margins(
        env, states[:, :3], float(event["gamma"]),
        config.safemppi.sensing_range,
    )
    fig = plt.figure(figsize=(12.8, 5.2))
    ax = fig.add_subplot(121, projection="3d")
    draw_scene_factory(env)(ax)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from safe_mppi.geometry import hull_edges, polytope_vertices
    for h in (0, 3, 6, 9):
        polytope = build_nominal_polytope(states[h, :3], env.spheres, env.cylinders,
                                          env.bounds,
                                          sensing_range=config.safemppi.sensing_range,
                                          obstacle_margin=0.0)
        if h == 3:
            draw_polytope_soft(ax, polytope, float(event["gamma"]), color="#1d7a3e",
                               cmap=plt.get_cmap("Greens"))
        else:
            vertices = polytope_vertices(polytope)
            if vertices is not None:
                _, edges = hull_edges(vertices)
                ax.add_collection3d(Line3DCollection(vertices[edges], colors="#1d7a3e",
                                                     linewidths=0.35, alpha=0.30))
    ax.plot(*states[:, :3].T, color="#111111", lw=2.4, marker="o", ms=3)
    robot = states[0, :3]
    ax.set_xlim(robot[0] - 0.6, robot[0] + 1.2)
    ax.set_ylim(robot[1] - 0.8, robot[1] + 0.8)
    ax.set_zlim(max(robot[2] - 0.8, 1.0), robot[2] + 0.8)
    ax.set_box_aspect((1.8, 1.6, 1.6))
    ax.set_title("Legacy rebuilt nominal-polytope chain (diagnostic only)\n"
                 rf"($\gamma={event['gamma']:g}$, round {event['round']}; soft fill at h=3, "
                 "outlines at h=0,6,9)", fontsize=10)
    ax2 = fig.add_subplot(122)
    ax2.bar(range(1, len(margins) + 1), margins, color="#1d7a3e", alpha=0.8)
    ax2.axhline(0.0, color="#c8321b", lw=1.2)
    ax2.set_xlabel("plan step h")
    ax2.set_ylabel(r"$H_P(q_{h+1}) - (1-\gamma)$ under the rebuilt polytope at $q_h$")
    ax2.set_title("per-knot verifier margins (all must stay above the red line)",
                  fontsize=10)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def mode_timeline_figure(env, events, per_round_rows, output):
    """First appearance of each route mode: tilted acquisition (D+) vs raw sampling."""
    acquisition = {}
    for event in events:
        state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
        if not (0.7 <= state6[0] <= 1.5):
            continue
        for local, k in enumerate(event["selected"]):
            if event["verification"][local]["valid"]:
                mode = route_mode(env, plan_states(env, state6,
                                                   event["candidates"][k])[:, :3])
                acquisition.setdefault(event["round"], set()).add(mode)
    raw = {round_i: {row["mode"] for row in rows if row["status"] == "SUCCESS"}
           for round_i, rows in per_round_rows.items()}
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    for row_i, mode in enumerate(ROUTE_MODES):
        acq_rounds = [r for r in sorted(acquisition) if mode in acquisition[r]]
        raw_rounds = [r for r in sorted(raw) if mode in raw[r]]
        ax.scatter(acq_rounds, [row_i + 0.16] * len(acq_rounds), marker="o", s=26,
                   facecolor="none", edgecolor=MODE_COLORS[mode],
                   label="tilted acquisition D+" if row_i == 0 else None)
        ax.scatter(raw_rounds, [row_i - 0.16] * len(raw_rounds), marker="s", s=26,
                   color=MODE_COLORS[mode],
                   label="raw temperature-1" if row_i == 0 else None)
    ax.set_yticks(range(len(ROUTE_MODES)), ROUTE_MODES)
    ax.set_xlabel("expansion round")
    ax.set_title("Mode presence: verifier-positive tilted queries (open) vs raw sampling "
                 "(filled)", weight="bold", fontsize=10)
    ax.grid(alpha=0.2, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gamma_trend_figure(rows_final, demo_metrics_csv, gammas, output):
    """Final-round raw clearance / time-to-goal vs gamma, next to the SafeMPPI demo trend."""
    import csv as csv_module
    with open(demo_metrics_csv) as stream:
        demo = {float(row["gamma"]): row for row in csv_module.DictReader(stream)}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key, demo_key, label in (
            (axes[0], "min_clearance_m", "avg_min_clearance_m", "avg min clearance [m]"),
            (axes[1], "time_to_goal_s", "avg_time_to_goal_s", "avg time to goal [s]")):
        ours, reference = [], []
        for gamma in gammas:
            values = [row[key] for row in rows_final
                      if row["gamma"] == gamma and row[key] is not None
                      and row["status"] == "SUCCESS"]
            ours.append(float(np.mean(values)) if values else np.nan)
            reference.append(float(demo[gamma][demo_key]))
        ax.plot(gammas, ours, "-o", color="#174f92", label="expansion raw (final round)")
        ax.plot(gammas, reference, "--s", color="#9aa0a6", label="SafeMPPI demonstrations")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(label)
        ax.set_xticks(gammas)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Gamma trends: raw expansion policy vs the SafeMPPI demonstrator",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sigma_tilt_figures(env, events, output_dir, beta):
    """(1) anatomy of one acquisition step; (2) per-mode sigma decay across rounds."""
    interesting = [event for event in events if 1.0 <= event["robot"][0] <= 1.45]
    if not interesting:
        interesting = list(events)
    event = max(interesting, key=lambda e: float(np.std(e["sigma_K"])))
    paths = np.stack([plan_states(env, np.concatenate([event["robot"][:3],
                                                       event["robot"][3:6]]),
                                  plan)[:, :3] for plan in event["candidates"]])
    sigma = event["sigma_K"]
    norm = Normalize(float(sigma.min()), float(sigma.max()) + 1e-12)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(12.6, 4.8))
    ax = fig.add_subplot(131, projection="3d")
    draw_scene_factory(env)(ax)
    valid_flags = [bool(v["valid"]) for v in event["verification"]]
    for k, path in enumerate(paths):
        ax.plot(*path.T, color=cmap(norm(sigma[k])), lw=2.0 if k in event["selected"] else 0.9,
                ls="--" if k in event["selected"] else "-",
                alpha=0.95 if k in event["selected"] else 0.45)
    for local, k in enumerate(event["selected"]):
        ax.plot(*paths[k].T, color=("#17964b" if valid_flags[local] else "#c8321b"),
                lw=2.6, alpha=0.9)
    robot = event["robot"][:3]
    ax.set_xlim(robot[0] - 0.45, robot[0] + 1.05)
    ax.set_ylim(robot[1] - 0.7, robot[1] + 0.7)
    ax.set_zlim(max(robot[2] - 0.7, 1.0), robot[2] + 0.7)
    ax.set_box_aspect((1.5, 1.4, 1.4))
    ax.set_title(rf"K=16 plans colored by $\sigma(\phi_s)$; selected B dashed"
                 f"\n(green/red = verifier verdict), round {event['round']}", fontsize=9)
    ax2 = fig.add_subplot(132)
    order = np.argsort(sigma)[::-1]
    weights = np.exp((sigma - sigma.max()) / beta)
    weights = weights / weights.sum()
    ax2.bar(range(len(sigma)), sigma[order], color=[cmap(norm(s)) for s in sigma[order]])
    marks = [int(np.where(order == k)[0][0]) for k in event["selected"]]
    ax2.scatter(marks, sigma[order][marks] + 0.002, marker="v", color="#c8321b", s=40,
                label="selected B")
    ax2.set_xlabel("candidate (sorted)")
    ax2.set_ylabel(r"$\sigma(\phi_s)$")
    ax2.legend(fontsize=8)
    ax2.set_title("marginal uncertainty + selection", fontsize=9)
    ax3 = fig.add_subplot(133)
    ax3.bar(range(len(sigma)), weights[order], color="#5b8fd6")
    ax3.set_xlabel("candidate (sorted by sigma)")
    ax3.set_ylabel(r"$\pi(j)\propto\exp((\sigma_j-\max\sigma)/\beta)$")
    ax3.set_title(rf"acquisition softmax, $\beta$ fixed", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "sigma_tilt_anatomy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    by_round_mode = {}
    for event in events:
        state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
        if not (0.8 <= state6[0] <= 1.5):
            continue
        for k, plan in enumerate(event["candidates"]):
            mode = route_mode(env, plan_states(env, state6, plan)[:, :3])
            by_round_mode.setdefault((event["round"], mode), []).append(
                float(event["sigma_K"][k]))
    rounds = sorted({key[0] for key in by_round_mode})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for mode in (*ROUTE_MODES, "none"):
        values = [np.mean(by_round_mode.get((r, mode), [np.nan])) for r in rounds]
        ax.plot(rounds, values, "-o", ms=4, color=MODE_COLORS[mode], label=mode)
    ax.set_xlabel("expansion round")
    ax.set_ylabel(r"mean $\sigma(\phi_s)$ of near-ball candidates")
    ax.set_title(r"$\sigma$ per route mode: novel modes start high, decay as queried",
                 weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "sigma_mode_decay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def acquisition_diagnostics(metrics_jsonl, output):
    """Plot marginal acquisition uncertainty and reference-buffer diagnostics."""
    rows = [
        json.loads(line) for line in Path(metrics_jsonl).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"no acquisition metrics in {metrics_jsonl}")
    rounds = np.asarray([row["round"] for row in rows], int)

    def values(key):
        return np.asarray([
            np.nan if row.get(key) is None else float(row[key]) for row in rows
        ])

    marginal_ess = np.asarray([
        float(row["marginal_ESS_over_K"])
        if row.get("marginal_ESS_over_K") is not None
        else float(row.get("ESS_over_K", np.nan))
        for row in rows
    ])
    used_legacy_ess = any(
        row.get("marginal_ESS_over_K") is None and row.get("ESS_over_K") is not None
        for row in rows
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    axes[0, 0].plot(rounds, values("sigma_pool_mean"), "-o", ms=3.5,
                    label=r"all-$K$ marginal mean")
    axes[0, 0].plot(rounds, values("sigma_selected_mean"), "-o", ms=3.5,
                    label=r"selected-$B$ marginal mean")
    axes[0, 0].set_title(r"Marginal GP uncertainty $\sigma(\phi_s)$")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].axhline(0.0, color="#777777", lw=0.8, ls=":")
    axes[0, 1].plot(rounds, values("uncertainty_uplift"), "-o", ms=3.5,
                    color="#d8741c")
    axes[0, 1].set_title(
        r"Uncertainty uplift: selected-$B$ minus all-$K$"
    )

    axes[1, 0].plot(rounds, marginal_ess, "-o", ms=3.5, color="#6a51a3")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title(
        r"Marginal acquisition ESS$/K$"
        + (" (legacy sequential fallback)" if used_legacy_ess else "")
    )

    axes[1, 1].step(rounds, values("gp_buffer"), where="mid", color="#238b45")
    axes[1, 1].set_title("GP reference-buffer size")

    for axis in axes.flat:
        axis.set_xlabel("Expansion round")
        axis.grid(alpha=0.22)
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def mechanism_video(
    env, events, output, rounds, gamma=0.3, step_stride=2,
    target_region="above_wedge", target_gate_start_round=None,
    committed_cells=None,
):
    """Synchronized canonical replicas for one gamma; no replica start assistance."""
    if target_region not in {"above_wedge", "above_halfspace"}:
        raise ValueError(f"unknown target_region: {target_region}")
    target_equation = (
        r"$z-2\geq |y|$" if target_region == "above_wedge" else r"$z\geq2$"
    )
    target_name = (
        "above wedge" if target_region == "above_wedge" else "above halfspace"
    )
    target_predicate = (
        inside_above_wedge
        if target_region == "above_wedge" else inside_above_halfspace
    )
    chosen_rounds = set(rounds)
    filtered = [event for event in events if event["round"] in chosen_rounds
                and abs(float(event["gamma"]) - float(gamma)) < 1.0e-7]
    if not filtered:
        raise RuntimeError(f"no mechanism events for gamma={gamma:g}")
    sigma_statistics = round_sigma_statistics(filtered)
    normalized_color_scale = Normalize(0.0, 1.0, clip=True)
    cmap = plt.get_cmap("viridis")
    by_round_episode = {}
    for event in filtered:
        by_round_episode.setdefault((event["round"], event["episode"]), []).append(event)
    for rows in by_round_episode.values():
        rows.sort(key=lambda event: event["step"])
    if committed_cells is not None:
        missing = [
            (round_i, float(gamma)) for round_i in sorted(chosen_rounds)
            if (round_i, float(gamma)) not in committed_cells
        ]
        if missing:
            raise ValueError(
                f"mechanism video has no committed-success cells for {missing}"
            )

    fig = plt.figure(figsize=(13.2, 7.2))
    writer = FFMpegWriter(
        fps=9, codec="libx264", bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=135):
        for round_i in sorted(chosen_rounds):
            colorbar = None
            round_sigma = sigma_statistics[round_i]

            def sigma_color(value):
                normalized = within_round_normalized_sigma(
                    float(value), round_sigma,
                )
                return cmap(normalized_color_scale(normalized))

            episodes = {episode: rows for (r, episode), rows in by_round_episode.items()
                        if r == round_i}
            round_gate_active = any(
                bool(event.get(
                    "target_gate_active",
                    target_gate_start_round is not None
                    and round_i >= target_gate_start_round,
                ))
                for rows in episodes.values() for event in rows
            )
            steps = sorted({event["step"] for rows in episodes.values() for event in rows})
            frame_steps = steps[::step_stride]
            if steps and steps[-1] not in frame_steps:
                frame_steps.append(steps[-1])
            for frame_step in frame_steps:
                if colorbar is not None:
                    colorbar.remove()
                    colorbar = None
                fig.clear()
                grid = fig.add_gridspec(1, 2, width_ratios=(2.15, 1.0), wspace=0.18)
                ax = fig.add_subplot(grid[0, 0])
                ax_head = fig.add_subplot(grid[0, 1])
                sphere = np.asarray(env.spheres[0], float)
                theta = np.linspace(0.0, 2.0 * np.pi, 160)
                ax.fill(sphere[0] + sphere[3] * np.cos(theta),
                        sphere[2] + sphere[3] * np.sin(theta),
                        color="#8f969f", alpha=0.46)
                ax_head.fill(sphere[1] + sphere[3] * np.cos(theta),
                             sphere[2] + sphere[3] * np.sin(theta),
                             color="#8f969f", alpha=0.46)
                if round_gate_active:
                    # Both target regions project to z>=2 in x-z.  The head-on
                    # inset distinguishes the wedge from the halfspace.
                    target_z = 2.0
                    ax.axhspan(
                        target_z, 2.95, color="#f0a33a", alpha=0.055, zorder=0,
                    )
                    ax.axhline(
                        target_z, color="#d47b18", lw=1.25, ls=":", alpha=0.85,
                    )
                    target_y = np.linspace(-0.95, 0.95, 160)
                    target_boundary = (
                        target_z + np.abs(target_y)
                        if target_region == "above_wedge"
                        else np.full_like(target_y, target_z)
                    )
                    ax_head.fill_between(
                        target_y, target_boundary, 2.95,
                        color="#f0a33a", alpha=0.085, zorder=0,
                    )
                    ax_head.plot(
                        target_y, target_boundary, color="#d47b18",
                        lw=1.25, ls=":",
                    )
                ax.scatter(env.start[0], env.start[2], marker="s",
                           color="#111111", s=28)
                ax.scatter(env.goal[0], env.goal[2], marker="*",
                           color="#ffca28", edgecolor="#6a4e00", s=110)
                ax_head.scatter(env.start[1], env.start[2], marker="s",
                                color="#111111", s=28)
                ax_head.scatter(env.goal[1], env.goal[2], marker="*",
                                color="#ffca28", edgecolor="#6a4e00", s=110)
                negative_states = []
                negative_reasons = []
                success_states = []
                active = nvp_count = target_nvp_count = success_count = 0
                for episode, rows in sorted(episodes.items()):
                    past = [event for event in rows if event["step"] <= frame_step]
                    if not past:
                        continue
                    current = past[-1]
                    segments, head_segments, segment_colors = [], [], []
                    for index, event in enumerate(past):
                        if event["chosen_local"] is None:
                            continue
                        start = np.asarray(event["robot"][:3], float)
                        if "robot_after" in event:
                            stop = np.asarray(event["robot_after"][:3], float)
                        elif index + 1 < len(past):
                            stop = np.asarray(past[index + 1]["robot"][:3], float)
                        else:
                            continue
                        segments.append([[start[0], start[2]], [stop[0], stop[2]]])
                        head_segments.append([[start[1], start[2]], [stop[1], stop[2]]])
                        chosen = event["selected"][event["chosen_local"]]
                        segment_colors.append(sigma_color(
                            event["sigma_K"][chosen]
                        ))
                    if segments:
                        segment_array = np.asarray(segments)
                        head_segment_array = np.asarray(head_segments)
                        ax.add_collection(LineCollection(
                            segment_array, colors=segment_colors,
                            linewidths=1.8, alpha=0.88))
                        ax_head.add_collection(LineCollection(
                            head_segment_array, colors=segment_colors,
                            linewidths=1.8, alpha=0.88))
                        ax.scatter(
                            segment_array[:, 0, 0], segment_array[:, 0, 1],
                            s=8, color=segment_colors, alpha=0.92, zorder=6,
                        )
                        ax_head.scatter(
                            head_segment_array[:, 0, 0], head_segment_array[:, 0, 1],
                            s=8, color=segment_colors, alpha=0.92, zorder=6,
                        )
                        ax.scatter(
                            segment_array[-1, 1, 0], segment_array[-1, 1, 1],
                            s=8, color=[segment_colors[-1]], alpha=0.92, zorder=6,
                        )
                        ax_head.scatter(
                            head_segment_array[-1, 1, 0], head_segment_array[-1, 1, 1],
                            s=8, color=[segment_colors[-1]], alpha=0.92, zorder=6,
                        )
                    for event in past:
                        if event["status"] == "NVP":
                            negative_states.append(event["robot"][:3])
                            negative_reasons.append(event.get("nvp_reason"))
                    success_states.extend(
                        event["robot_after"][:3] for event in past
                        if event["status"] == "SUCCESS"
                    )
                    nvp_count += int(current["status"] == "NVP")
                    target_nvp_count += int(
                        current["status"] == "NVP"
                        and current.get("nvp_reason") == "TARGET"
                    )
                    success_count += int(current["status"] == "SUCCESS")
                    if current["status"] is None and current["step"] == frame_step:
                        active += 1
                    if current["step"] != frame_step:
                        continue
                    state6 = np.asarray(current["robot"][:6], float)
                    paths = np.stack([plan_states(env, state6, plan)[:, :3]
                                      for plan in current["candidates"]])
                    for candidate, path in enumerate(paths):
                        if candidate not in current["selected"]:
                            ax.plot(path[:, 0], path[:, 2],
                                    color="#9aa0a6", lw=0.35, alpha=0.16)
                            ax_head.plot(path[:, 1], path[:, 2],
                                         color="#9aa0a6", lw=0.35, alpha=0.16)
                    for local, candidate in enumerate(current["selected"]):
                        verification = current["verification"][local]
                        if verification["valid"]:
                            ax.plot(paths[candidate, :, 0], paths[candidate, :, 2],
                                    color="#17964b", lw=3.0, alpha=0.48)
                            ax_head.plot(paths[candidate, :, 1], paths[candidate, :, 2],
                                         color="#17964b", lw=3.0, alpha=0.48)
                        plan_color = sigma_color(current["sigma_K"][candidate])
                        ax.plot(paths[candidate, :, 0], paths[candidate, :, 2],
                                color=plan_color, lw=1.15, ls="--", alpha=0.95)
                        ax_head.plot(paths[candidate, :, 1], paths[candidate, :, 2],
                                     color=plan_color, lw=1.15, ls="--", alpha=0.95)
                        gate_active = current.get("target_gate_active")
                        if gate_active is None:
                            gate_active = (
                                target_gate_start_round is not None
                                and current["round"] >= target_gate_start_round
                            )
                        gate_active = bool(gate_active)
                        target_eligible = verification.get("target_eligible")
                        if target_eligible is None:
                            target_eligible = bool(target_predicate(
                                paths[candidate, 1:2], tolerance=1.0e-7
                            )[0])
                        if (gate_active and verification["valid"]
                                and verification.get("progress_eligible", True)
                                and not target_eligible):
                            proposed = paths[candidate, 1]
                            # The rejected proposal is never executed.  A hollow orange
                            # cross at q1 distinguishes it from the actual terminal state.
                            ax.plot(paths[candidate, :2, 0], paths[candidate, :2, 2],
                                    color="#e67e22", lw=1.8, ls=":", alpha=0.95)
                            ax_head.plot(paths[candidate, :2, 1], paths[candidate, :2, 2],
                                         color="#e67e22", lw=1.8, ls=":", alpha=0.95)
                            ax.scatter(proposed[0], proposed[2], s=42, marker="x",
                                       color="#e67e22", linewidths=1.5, zorder=8)
                            ax_head.scatter(proposed[1], proposed[2], s=42, marker="x",
                                            color="#e67e22", linewidths=1.5, zorder=8)
                    if current["chosen_local"] is not None:
                        chosen = current["selected"][current["chosen_local"]]
                        ax.plot(paths[chosen, :2, 0], paths[chosen, :2, 2],
                                color="#1468b3", lw=3.0)
                        ax_head.plot(paths[chosen, :2, 1], paths[chosen, :2, 2],
                                     color="#1468b3", lw=3.0)
                    ax.scatter(current["robot"][0], current["robot"][2],
                               s=13, color="#111111")
                    ax_head.scatter(current["robot"][1], current["robot"][2],
                                    s=13, color="#111111")
                if negative_states:
                    negative_states = np.asarray(negative_states)
                    wedge_mask = np.asarray([
                        reason == "TARGET" for reason in negative_reasons
                    ], bool)
                    for axis, columns in ((ax, (0, 2)), (ax_head, (1, 2))):
                        if np.any(~wedge_mask):
                            axis.scatter(
                                negative_states[~wedge_mask, columns[0]],
                                negative_states[~wedge_mask, columns[1]],
                                s=55, color="#c8321b", marker="x", linewidth=1.8,
                            )
                        if np.any(wedge_mask):
                            axis.scatter(
                                negative_states[wedge_mask, columns[0]],
                                negative_states[wedge_mask, columns[1]],
                                s=58, color="#e67e22", marker="x", linewidth=2.0,
                            )
                if success_states:
                    success_states = np.asarray(success_states)
                    for axis, columns in ((ax, (0, 2)), (ax_head, (1, 2))):
                        axis.scatter(success_states[:, columns[0]],
                                     success_states[:, columns[1]],
                                     s=56, color="#159447", edgecolor="#0b5f2c",
                                     marker="*", linewidth=0.6)
                ax.set_xlim(-0.15, 3.2)
                ax.set_ylim(1.1, 2.95)
                ax_head.set_xlim(0.95, -0.95)
                ax_head.set_ylim(1.1, 2.95)
                ax.set_aspect("equal", adjustable="box")
                ax_head.set_aspect("equal", adjustable="box")
                ax.set_xlabel(r"$x$ [m]")
                ax.set_ylabel(r"$z$ [m]")
                ax_head.set_xlabel(r"$y$ [m] ($+y$ left)")
                ax_head.set_ylabel(r"$z$ [m]")
                ax.grid(alpha=0.20)
                ax_head.grid(alpha=0.20)
                gate_active = round_gate_active
                gate_caption = "ON" if gate_active else "OFF"
                ax.set_title(
                    rf"round {round_i}   $\gamma={gamma:g}$   synchronized step "
                    f"{frame_step}   target gate {gate_caption}\n"
                    f"active {active} | NVP {nvp_count} "
                    f"(target {target_nvp_count}) | success {success_count} | "
                    r"raw $\sigma$ "
                    f"q02/med/q98={round_sigma['q02']:.3g}/"
                    f"{round_sigma['median']:.3g}/{round_sigma['q98']:.3g}"
                )
                ax_head.set_title(
                    (
                        f"head-on {target_name}: {target_equation}"
                        if gate_active else "head-on"
                    ),
                    fontsize=10.5,
                )
                legend = [
                    Line2D([0], [0], color="#9aa0a6", lw=0.8, label=r"$K$ plans"),
                    Line2D([0], [0], color="#5b8fd6", lw=1.2, ls="--",
                           label=r"queried $B$ (normalized $\sigma$ color)"),
                    Line2D([0], [0], color="#17964b", lw=3.0, alpha=0.48,
                           label=r"GREEN halo = full-$H$ safe"),
                    Line2D([0], [0], color="#1468b3", lw=3.0,
                           label="executed first step"),
                    Line2D([0], [0], color="#c8321b", marker="x", lw=0,
                           markersize=6, label="verifier/progress NVP (actual state)"),
                ]
                if gate_active:
                    legend[3:3] = [
                        Line2D([0], [0], color="#e67e22", lw=1.8, ls=":",
                               label="target-rejected proposed $q_1$"),
                        Line2D([0], [0], color="#e67e22", marker="x", lw=0,
                               markersize=6,
                               label="target-region NVP (actual state)"),
                    ]
                ax.legend(handles=legend, loc="lower right", fontsize=7.5,
                          framealpha=0.88)
                scalar = plt.cm.ScalarMappable(
                    norm=normalized_color_scale, cmap=cmap,
                )
                colorbar = fig.colorbar(
                    scalar, ax=(ax, ax_head), fraction=0.025, pad=0.035
                )
                colorbar.set_label(
                    r"within-round normalized $\widetilde{\sigma}_n(\phi_s)$"
                )
                writer.grab_frame()
            if colorbar is not None:
                colorbar.remove()
                colorbar = None
            if committed_cells is None:
                for _ in range(5):
                    writer.grab_frame()
                continue

            fig.clear()
            summary_grid = fig.add_gridspec(
                1, 2, width_ratios=(2.15, 1.0), wspace=0.18,
            )
            summary_side = fig.add_subplot(summary_grid[0, 0])
            summary_head = fig.add_subplot(summary_grid[0, 1])
            cell = committed_cells[(round_i, float(gamma))]
            draw_gathering_cell(
                summary_side,
                summary_head,
                env,
                cell,
                target_region=target_region,
                gate_active=round_gate_active,
            )
            if cell.committed_trajectory_id is None:
                commit_caption = "NO COMMITTED SUCCESS"
            else:
                commit_caption = (
                    f"{len(cell.committed_trajectory_ids)} committed SUCCESS "
                    "episodes; "
                    f"{len(cell.committed_window_ids)} committed <=H suffix starts"
                )
            summary_side.set_title(
                rf"round {round_i}   $\gamma={gamma:g}$ | gathering summary"
                "\n" + commit_caption,
                fontsize=11,
                weight="bold",
            )
            summary_head.set_title(
                (
                    f"head-on {target_name}: {target_equation}"
                    if round_gate_active else "head-on"
                ),
                fontsize=10.5,
            )
            summary_side.set_xlabel(r"$x$ [m]")
            summary_side.set_ylabel(r"$z$ [m]")
            summary_head.set_xlabel(r"$y$ [m] ($+y$ left)")
            summary_head.set_ylabel(r"$z$ [m]")
            summary_legend = (
                Line2D([0], [0], color=COMMITTED_COLOR, lw=3.0,
                       label="committed SUCCESS"),
                Line2D(
                    [0], [0], marker="o", color="none",
                    markerfacecolor=COMMITTED_WINDOW_COLOR,
                    markeredgecolor="white", markersize=6,
                    label=r"committed $\leq H$ suffix start",
                ),
                Line2D([0], [0], color=OTHER_SUCCESS_COLOR, lw=1.0,
                       label="other gathering SUCCESS"),
                Line2D([0], [0], color=FAILED_COLOR, lw=0.8,
                       label="failed attempt (diagnostic only)"),
            )
            summary_side.legend(
                handles=summary_legend, loc="lower right",
                fontsize=7.5, framealpha=0.88,
            )
            fig.suptitle(
                "Expansion gathering evidence | not raw evaluation",
                fontsize=13,
                weight="bold",
            )
            for _ in range(14):
                writer.grab_frame()
    plt.close(fig)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_helios_arguments(parser)
    parser.add_argument("--pretrain-dir", type=Path, default=ROOT / "outputs" / "ball_flow")
    parser.add_argument("--expansion", type=Path,
                        default=ROOT / "outputs" / "ball_flow" / "expansion")
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help=(
            "optional isolated output directory; explicit nonempty paths are "
            "never overwritten by clutter evaluation"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="lab-clutter policy sampling device, e.g. cpu, cuda, or cuda:0",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--probe-samples", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument(
        "--fixed-scene-rollouts",
        type=int,
        default=10,
        help=(
            "independent raw temperature-1 rollouts per gamma/checkpoint in "
            "one preregistered clutter scene (separate from randomized metrics)"
        ),
    )
    parser.add_argument(
        "--fixed-scene-seed",
        type=int,
        help=(
            "optional preregistered clutter-scene seed; default is a fixed "
            "offset from --seed"
        ),
    )
    parser.add_argument("--video-gamma", type=float, default=0.3)
    parser.add_argument(
        "--video-rounds", type=int, nargs="+",
        help=(
            "displayed evaluation checkpoints; clutter evaluation permits r0 "
            "for raw galleries and uses positive rounds for mechanism video"
        ),
    )
    parser.add_argument(
        "--gallery-rounds", type=int, nargs="+",
        help=(
            "checkpoint rounds shown in raw_gallery / raw_crossing_fan; "
            "default is first, middle, last of the evaluated rounds. Every "
            "listed round must have been evaluated (see --stride)"
        ),
    )
    parser.add_argument(
        "--gallery-view",
        choices=("side", "head_on", "both"),
        default="side",
        help=(
            "raw_gallery projection: 'side' is the canonical start-goal-axis "
            "vs vertical view; 'head_on' looks down the start-goal vector "
            "(transverse vs vertical), the plane where below/above/left/right "
            "separate; 'both' writes raw_gallery.png and "
            "raw_gallery_headon.png"
        ),
    )
    parser.add_argument("--screening-only", action="store_true",
                        help="write raw metrics/curves/gallery but skip event-heavy diagnostics")
    parser.add_argument("--metrics-only", action="store_true",
                        help="write raw_eval.json without rendering figures or video")
    parser.add_argument(
        "--raw-tight-corridor", action="store_true",
        help=(
            "opt-in legacy raw termination at the expansion corridor; canonical "
            "temperature-1 evaluation leaves this disabled"
        ),
    )
    args = parser.parse_args()
    if args.helios:
        return run_evaluation_on_helios(args, sys.argv[1:], ROOT)
    if (
        args.evaluation_output is not None
        and args.evaluation_output.exists()
        and (
            not args.evaluation_output.is_dir()
            or any(args.evaluation_output.iterdir())
        )
    ):
        raise FileExistsError(
            "refusing to overwrite explicit evaluation output "
            f"{args.evaluation_output}"
        )

    pretrain_manifest = json.loads(
        (args.pretrain_dir / "pretrain_manifest.json").read_text()
    )
    if (
        pretrain_manifest.get("kind")
        == "lab raw-command reference-flow pretraining"
        or str(pretrain_manifest.get("context_schema", "")).startswith("lab_")
    ):
        manifest = json.loads(
            (args.expansion / "manifest.json").read_text()
        )
        task_config_path = args.expansion / "task_config_resolved.json"
        from safe_mppi.lab_clutter_evaluation import (
            evaluate_lab_clutter_expansion,
            is_lab_clutter_evaluation_manifest,
        )

        if is_lab_clutter_evaluation_manifest(manifest):
            if not task_config_path.is_file():
                raise FileNotFoundError(
                    "random-three-sphere expansion requires "
                    "task_config_resolved.json"
                )

            evaluate_lab_clutter_expansion(
                args,
                load_config(task_config_path),
                pretrain_manifest,
                manifest,
            )
        else:
            from safe_mppi.lab_flow_evaluation import evaluate_lab_expansion

            if task_config_path.is_file():
                config = load_config(task_config_path)
            else:
                source_demo_dir = Path(pretrain_manifest["source_demo_dir"])
                if not source_demo_dir.is_absolute():
                    source_demo_dir = args.pretrain_dir / source_demo_dir
                config = load_config(source_demo_dir / "resolved_config.json")
            evaluate_lab_expansion(
                args, config, pretrain_manifest, manifest,
            )
        return

    config = load_config(args.pretrain_dir / "demo_config.json")
    env = TaskEnvironment(config)
    gammas = list(config.data.gammas)
    manifest = json.loads((args.expansion / "manifest.json").read_text())
    tight_corridor = bool(args.raw_tight_corridor)
    context_contract = pretrain_manifest.get("context_contract", "legacy10")
    target_gate = manifest.get("ball_target_region_gate")
    if target_gate is None:
        legacy_wedge = manifest.get("ball_above_wedge_gate", {})
        target_region = "above_wedge"
        target_gate_start_round = legacy_wedge.get("start_round")
    else:
        target_region = target_gate.get("target_region", "above_wedge")
        target_gate_start_round = target_gate.get("start_round")
    task_type = (
        ThetaBallFlowTask
        if context_contract == "local_theta12" else BallFlowTask
    )
    task = task_type(
        config, tight_corridor=tight_corridor, target_region=target_region,
    )
    total_rounds = manifest["config"]["rounds"]
    eval_rounds = sorted({0, *range(0, total_rounds + 1, args.stride), total_rounds})

    summary, per_round_rows, per_round_probes = {}, {}, {}
    for round_i in eval_rounds:
        policy = checkpoint_policy(args.expansion, args.pretrain_dir, round_i)
        rows = raw_eval(
            policy, task, gammas, args.episodes, args.seed,
            tight_corridor=tight_corridor,
        )
        probes = validity_probe(policy, task, gammas, args.probe_samples, args.seed + 7)
        per_round_rows[round_i] = rows
        per_round_probes[round_i] = probes
        per_gamma = {}
        for gamma in gammas:
            gamma_rows = [row for row in rows if row["gamma"] == gamma]
            gamma_probe = [row for row in probes if row["gamma"] == gamma]
            per_gamma[f"{gamma:g}"] = summarize(gamma_rows, gamma_probe)
        summary[str(round_i)] = {"pooled": summarize(rows, probes), "per_gamma": per_gamma}
        if round_i in (eval_rounds[0], eval_rounds[-1]):
            summary[str(round_i)]["above_start_probe"] = above_start_probe(
                policy, task, gammas, episodes=8, seed0=args.seed + 500,
                tight_corridor=tight_corridor)
        pooled = summary[str(round_i)]["pooled"]
        print(f"round {round_i:2d}: SR {pooled['SR']:.2f} CR {pooled['CR']:.2f} "
              f"coverage {pooled['coverage']:.2f} modes {pooled['modes']} "
              f"validity {pooled['raw_validity']:.2f}"
              + (f" | above-start probe {summary[str(round_i)]['above_start_probe']}"
                 if "above_start_probe" in summary[str(round_i)] else ""), flush=True)

    eval_dir = (
        args.evaluation_output
        if args.evaluation_output is not None
        else args.expansion / "eval"
    )
    if (
        args.evaluation_output is not None
        and eval_dir.exists()
        and (
            not eval_dir.is_dir()
            or any(eval_dir.iterdir())
        )
    ):
        raise FileExistsError(
            f"refusing to overwrite explicit evaluation output {eval_dir}"
        )
    eval_dir.mkdir(parents=True, exist_ok=True)
    slim = {round_i: [{k: v for k, v in row.items() if k != "states"} for row in rows]
            for round_i, rows in per_round_rows.items()}
    (eval_dir / "raw_eval.json").write_text(json.dumps(
        {"summary": summary, "rows": slim}, indent=2) + "\n")

    if args.metrics_only:
        print("[outputs]", eval_dir, flush=True)
        return

    plot_expansion_results(args.expansion / "metrics.jsonl", eval_dir / "expansion_curves.png")
    acquisition_diagnostics(
        args.expansion / "metrics.jsonl",
        eval_dir / "acquisition_diagnostics.png",
    )
    raw_metric_figure(per_round_rows, per_round_probes, gammas,
                      eval_dir / "raw_curves.png")
    raw_sr_coverage_figure(
        per_round_rows, gammas, eval_dir / "raw_sr_coverage.png",
    )
    mode_share_figure(per_round_rows, eval_dir / "mode_share.png")
    gallery_rounds = sorted({eval_rounds[0], eval_rounds[len(eval_rounds) // 2],
                             eval_rounds[-1]})
    crossing_fan_figure(env, per_round_rows, gallery_rounds, eval_dir / "raw_crossing_fan.png")
    rollouts = {(round_i, float(gamma)): [row["states"][:, :3]
                                          for row in per_round_rows[round_i]
                                          if row["gamma"] == gamma]
                for round_i in gallery_rounds for gamma in gammas}
    side_rollout_gallery(env, per_round_rows, gallery_rounds, gammas,
                         eval_dir / "raw_gallery.png")

    if args.screening_only:
        print("[outputs]", eval_dir, flush=True)
        return

    events = torch.load(args.expansion / "events.pt", weights_only=False)
    committed_cells = None
    if has_committed_success_schema(manifest):
        committed_cells = resolve_committed_success(manifest, events)
        plot_gathering_commit_gallery(
            env,
            manifest,
            events,
            eval_dir / "gathering_committed_success_gallery.png",
            target_region=target_region,
            target_gate_start_round=target_gate_start_round,
        )
    sigma_tilt_figures(env, events, eval_dir, float(manifest["config"]["beta"]))
    green_verifier_figure(
        env, config, events, eval_dir / "legacy_rebuilt_nominal_chain.png")
    mode_timeline_figure(env, events, per_round_rows, eval_dir / "mode_timeline.png")
    gamma_trend_figure(per_round_rows[total_rounds],
                       args.pretrain_dir / "demos" / "metrics.csv", gammas,
                       eval_dir / "gamma_trend_vs_safemppi.png")
    video_rounds = (
        sorted(set(args.video_rounds))
        if args.video_rounds is not None else list(range(1, total_rounds + 1))
    )
    if not video_rounds or any(round_i < 1 or round_i > total_rounds
                               for round_i in video_rounds):
        raise ValueError(
            f"--video-rounds must lie in [1,{total_rounds}], got {video_rounds}"
        )
    mechanism_video(env, events, eval_dir / "mechanism.mp4", video_rounds,
                    gamma=args.video_gamma, target_region=target_region,
                    target_gate_start_round=target_gate_start_round,
                    committed_cells=committed_cells)
    print("[outputs]", eval_dir, flush=True)


if __name__ == "__main__":
    main()
