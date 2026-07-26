"""Plot BLUE nominal and bounded GREEN verifier level sets for one saved event."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.spatial import HalfspaceIntersection
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import plan_states
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.geometry import build_nominal_polytope, hull_edges
from safe_mppi.verifier_polytope import fit_verifier_polytope


def level_field(A, b, center, margins, x, z, y):
    X, Z = np.meshgrid(x, z)
    points = np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], axis=1)
    values = ((b[None] - points @ A.T) / margins[None]).min(axis=1)
    return X, Z, values.reshape(X.shape)


def level_vertices(A, center, margins, beta):
    b = A @ center + float(beta) * margins
    try:
        vertices = HalfspaceIntersection(
            np.hstack([A, -b[:, None]]), center
        ).intersections
        return vertices if len(vertices) >= 4 else None
    except Exception:
        return None


def draw_scene_side(axis, env):
    sphere = np.asarray(env.spheres[0], float)
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    axis.fill(
        sphere[0] + sphere[3] * np.cos(theta),
        sphere[2] + sphere[3] * np.sin(theta),
        color="#aeb4bc",
        alpha=0.52,
        zorder=1,
    )
    axis.scatter(env.goal[0], env.goal[2], marker="*", s=90,
                 color="#ffca28", edgecolor="#6a4e00", zorder=8)
    axis.set_xlim(-0.05, 3.25)
    axis.set_ylim(1.25, 2.70)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel(r"$z$ [m]")
    axis.grid(alpha=0.18)


def draw_level_slice(axis, env, A, b, center, margins, gamma, color_map,
                     paths, focused=None):
    draw_scene_side(axis, env)
    x = np.linspace(-0.05, 3.25, 330)
    z = np.linspace(1.25, 2.70, 240)
    X, Z, H = level_field(A, b, center, margins, x, z, float(center[1]))
    alpha = (1.0 - float(gamma)) ** np.arange(1, 11)
    ordered = np.sort(alpha)
    colors = [color_map(value) for value in np.linspace(0.25, 0.95, len(ordered))]
    axis.contour(X, Z, H, levels=ordered, colors=colors, linewidths=1.05)
    for index, path in enumerate(paths):
        selected = focused is not None and index == focused
        axis.plot(path[:, 0], path[:, 2],
                  color="#171717" if selected else "#8f969f",
                  lw=2.2 if selected else 0.65,
                  alpha=0.95 if selected else 0.34,
                  ls="-" if selected else "--",
                  zorder=5)
    axis.scatter(center[0], center[2], color="#1468b3", s=28, zorder=9)
    axis.text(
        0.02, 0.02,
        r"contours: $H=\alpha_h,\ \alpha_h=(1-\gamma)^h,\ h=1{:}10$",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def draw_sphere(axis, sphere):
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 16)
    axis.plot_surface(
        sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
        sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
        sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
        color="#aeb4bc", alpha=0.42, linewidth=0,
    )


def draw_hull_edges(axis, A, center, margins, gamma, steps, colors,
                    linestyles, linewidths):
    for step, color, linestyle, linewidth in zip(
            steps, colors, linestyles, linewidths):
        alpha = (1.0 - float(gamma)) ** int(step)
        vertices = level_vertices(A, center, margins, 1.0 - alpha)
        if vertices is None:
            continue
        _, edges = hull_edges(vertices)
        axis.add_collection3d(Line3DCollection(
            vertices[edges], colors=color, linestyles=linestyle,
            linewidths=linewidth, alpha=0.74,
        ))


def find_event(events, round_i, gamma, step, episode):
    matches = [
        event for event in events
        if event["round"] == round_i
        and abs(float(event["gamma"]) - float(gamma)) < 1.0e-7
        and event["step"] == step
        and (episode is None or event["episode"] == episode)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one matching event, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--step", type=int, default=39)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.pretrain_dir / "demo_config.json")
    env = TaskEnvironment(config)
    events = torch.load(args.expansion / "events.pt", weights_only=False)
    event = find_event(events, args.round, args.gamma, args.step, args.episode)
    state6 = np.asarray(event["robot"][:6], float)
    center = state6[:3]
    candidates = [int(index) for index in event["selected"]]
    plans = [np.asarray(event["candidates"][index], float) for index in candidates]
    states = [plan_states(env, state6, plan) for plan in plans]
    verifier = [
        fit_verifier_polytope(
            row[:, :3], env.spheres, env.cylinders, args.gamma,
            config.safemppi.sensing_range,
        )
        for row in states
    ]
    nominal = build_nominal_polytope(
        center, env.spheres, env.cylinders, env.bounds,
        sensing_range=config.safemppi.sensing_range,
        obstacle_margin=0.0,
    )
    progress = [float(np.min(np.diff(row[:, 0]))) for row in states]
    sigma = [float(event["sigma_K"][index]) for index in candidates]

    fig = plt.figure(figsize=(18.0, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    overview_axis = fig.add_subplot(grid[0, :2])
    nominal_axis = fig.add_subplot(grid[0, 2:])
    green_axes = tuple(fig.add_subplot(grid[1, local]) for local in range(4))

    draw_scene_side(overview_axis, env)
    for path in states:
        overview_axis.plot(path[:, 0], path[:, 2], "--", color="#5f6368",
                           lw=1.0, alpha=0.75)
    overview_axis.scatter(center[0], center[2], color="#1468b3", s=32, zorder=8)
    overview_axis.set_title(
        rf"saved acquisition: round {args.round}, $\gamma={args.gamma:g}$, "
        rf"step {args.step}, episode {args.episode}" "\n"
        rf"$B={len(candidates)}$; all four GREEN-positive",
        fontsize=11,
    )

    draw_level_slice(
        nominal_axis, env, nominal.A, nominal.b, nominal.center,
        nominal.margins, args.gamma, plt.get_cmap("Blues"), states,
    )
    nominal_axis.set_title(
        "BLUE nominal $H_P$: 80 fixed sensing faces + obstacle/task faces\n"
        "visualization only; not an execution gate",
        color="#145da0", fontsize=11,
    )

    for local, (axis, polytope) in enumerate(zip(green_axes, verifier)):
        draw_level_slice(
            axis, env, polytope.A, polytope.b, polytope.center,
            polytope.margins, args.gamma, plt.get_cmap("Greens"), states,
            focused=local,
        )
        axis.set_title(
            rf"GREEN fitted $B_{{{local + 1}}}$: "
            rf"$\sigma={sigma[local]:.4f}$, "
            rf"slack $={polytope.worst_slack:.4f}$, "
            rf"$\min_h\Delta x_h={progress[local]:+.3f}$ m",
            fontsize=9.5, color="#176b3a",
        )

    fig.suptitle(
        r"$K\rightarrow B$ by uncertainty $\rightarrow$ bounded GREEN positives "
        r"$\rightarrow\min_h\Delta x_h>0\rightarrow\arg\min J_{\rm native}$",
        fontsize=16, weight="bold",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    diagnostic_local = int(np.argmin([
        float(row["execution_cost"]) for row in event["verification"]
    ]))
    diagnostic_path = states[diagnostic_local]
    diagnostic_polytope = verifier[diagnostic_local]
    fig = plt.figure(figsize=(14.5, 6.8))
    for column, (name, A, margins, colors) in enumerate((
            ("BLUE nominal (visualization only)",
             nominal.A, nominal.margins,
             ("#9ecae1", "#4292c6", "#08519c")),
            (rf"GREEN fitted $B_{{{diagnostic_local + 1}}}$ "
             "(lowest native cost; unexecuted)",
             diagnostic_polytope.A, diagnostic_polytope.margins,
             ("#a8ddb5", "#41ab5d", "#006d2c")),
    )):
        axis = fig.add_subplot(1, 2, column + 1, projection="3d")
        draw_sphere(axis, np.asarray(env.spheres[0], float))
        draw_hull_edges(
            axis, A, center, margins, args.gamma,
            steps=(1, 5, 10), colors=colors,
            linestyles=("-", "-", "-"), linewidths=(0.30, 0.48, 0.75),
        )
        axis.plot(*diagnostic_path[:, :3].T,
                  color="#171717", lw=2.2, marker="o", ms=2.5)
        axis.scatter(*center, color="#1468b3", s=25)
        axis.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=90)
        radius = diagnostic_polytope.sensing_radius * 1.08
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=20, azim=-118)
        axis.set_title(
            name + "\n" + rf"level sets $h\in\{{1,5,10\}}$",
            fontsize=11,
        )
    fig.suptitle(
        rf"Bounded polytope geometry at round {args.round}, step {args.step}: "
        rf"GREEN slack $={diagnostic_polytope.worst_slack:.4f}$, "
        rf"$\min_h\Delta x_h={progress[diagnostic_local]:+.3f}$ m",
        fontsize=15, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_3d = args.output.with_name(args.output.stem + "_3d" + args.output.suffix)
    fig.savefig(output_3d, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(output_3d.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[snapshot] {args.output}")
    print(f"[snapshot] {output_3d}")
    for local, polytope in enumerate(verifier):
        print(
            f"  B{local + 1}: faces={len(polytope.A)} "
            f"artificial={polytope.kinds.count('artificial')} "
            f"feasible={polytope.feasible} slack={polytope.worst_slack:.6f} "
            f"min_dx={progress[local]:+.6f}"
        )


if __name__ == "__main__":
    main()
