"""The two requested figures: gamma-colored 3D rollouts and BLUE nominal H_P level sets."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from .environment import TaskEnvironment
from .geometry import NominalPolytope, hull_edges, polytope_vertices


PLASMA = plt.get_cmap("plasma")
BLUE = plt.get_cmap("Blues")


def _draw_box(ax, bounds, color="#777d86", alpha=0.45, linewidth=0.7):
    lo, hi = bounds[:, 0], bounds[:, 1]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    edges = []
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j > i and np.count_nonzero(np.abs(a - b) > 1e-9) == 1:
                edges.append([a, b])
    ax.add_collection3d(Line3DCollection(edges, colors=color, linewidths=linewidth, alpha=alpha))


def _draw_obstacles(ax, env: TaskEnvironment, alpha=0.18):
    for x, y, z, radius in env.spheres:
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 10)
        X = x + radius * np.outer(np.cos(u), np.sin(v))
        Y = y + radius * np.outer(np.sin(u), np.sin(v))
        Z = z + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(X, Y, Z, color="#8f969f", alpha=alpha, linewidth=0, shade=True)
    z0, z1 = env.bounds[2]
    for x, y, radius in env.cylinders:
        theta = np.linspace(0, 2 * np.pi, 20)
        Theta, Z = np.meshgrid(theta, [z0, z1])
        ax.plot_surface(x + radius * np.cos(Theta), y + radius * np.sin(Theta), Z,
                        color="#7d858e", alpha=alpha, linewidth=0, shade=True)


def _style(ax, env: TaskEnvironment):
    ax.set_xlim(*env.bounds[0])
    ax.set_ylim(*env.bounds[1])
    ax.set_zlim(*env.bounds[2])
    ax.set_box_aspect(tuple(env.bounds[:, 1] - env.bounds[:, 0]))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-58)
    ax.grid(alpha=0.16)


def _load_runs(manifest, root):
    runs = []
    for row in manifest["runs"]:
        data = np.load(root / row["file"], allow_pickle=True)
        runs.append((row, data))
    return runs


def plot_gamma_rollouts(manifest, env: TaskEnvironment, output_dir):
    output_dir = Path(output_dir)
    runs = _load_runs(manifest, output_dir)
    gammas = manifest["gammas"]
    norm = Normalize(min(gammas), max(gammas))
    fig = plt.figure(figsize=(10.5, 8.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _draw_box(ax, env.bounds)
    _draw_obstacles(ax, env, alpha=0.22)
    for row, data in runs:
        path = np.asarray(data["states"], float)[:, :3]
        color = PLASMA(norm(row["gamma"]))
        ax.plot(*path.T, color=color, lw=1.8, alpha=0.84)
        ax.scatter(*path[::max(1, len(path) // 18)].T, color=color, s=7, alpha=0.70)
        if not row["success"]:
            ax.scatter(*path[-1], marker="x", color="#cc3311", s=60, linewidth=1.8)
    ax.scatter(*env.start[:3], marker="s", color="#111111", s=45, label="start")
    ax.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=190, label="goal")
    _style(ax, env)
    ax.set_title("Actual mode-1 SafeMPPI rollouts — gamma overlay", fontsize=14, weight="bold")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=PLASMA)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.08)
    colorbar.set_label(r"safety level $\gamma$")
    ax.legend(loc="upper left")
    out = output_dir / "gamma_rollouts_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _poly_from_saved(data, index):
    A = np.asarray(data["poly_A"][index], float)
    b = np.asarray(data["poly_b"][index], float)
    center = np.asarray(data["states"], float)[index, :3]
    return NominalPolytope(A=A, b=b, center=center, margins=b - A @ center)


def _add_levelsets(ax, polytope, gamma):
    vertices = polytope_vertices(polytope)
    if vertices is None:
        ax.text2D(0.04, 0.90, "polytope is not strict-interior", transform=ax.transAxes,
                  color="#b3261e")
        return
    _, edges = hull_edges(vertices)
    raw_segments = vertices[edges]
    ax.add_collection3d(Line3DCollection(raw_segments, colors="#174f92", linewidths=0.75,
                                         alpha=0.82))
    alphas = (1.0 - float(gamma)) ** np.arange(1, 11)
    indices = [9] if np.allclose(alphas, alphas[0]) else range(10)
    for h0 in indices:
        beta = 1.0 - alphas[h0]
        level_vertices = polytope.center + beta * (vertices - polytope.center)
        color = BLUE(0.30 + 0.65 * h0 / 9.0)
        ax.add_collection3d(Line3DCollection(level_vertices[edges], colors=[color],
                                             linewidths=0.38, alpha=0.50))
    if gamma == 1.0:
        ax.text2D(0.04, 0.90, "10 levels coincide at H_P=0", transform=ax.transAxes,
                  fontsize=7.5, color="#174f92")


def plot_nominal_levelsets(manifest, env: TaskEnvironment, output_dir):
    output_dir = Path(output_dir)
    runs = _load_runs(manifest, output_dir)
    by_gamma = {}
    for row, data in runs:
        by_gamma.setdefault(float(row["gamma"]), []).append((row, data))
    gammas = sorted(by_gamma)
    columns = 4
    rows = math.ceil(len(gammas) / columns)
    fig = plt.figure(figsize=(18.0, 4.7 * rows), facecolor="white")
    norm = Normalize(min(gammas), max(gammas))
    for panel, gamma in enumerate(gammas):
        ax = fig.add_subplot(rows, columns, panel + 1, projection="3d")
        candidates = by_gamma[gamma]
        row, data = next(((r, d) for r, d in candidates if r["success"]), candidates[0])
        path = np.asarray(data["states"], float)[:, :3]
        n_poly = len(data["poly_A"])
        index = min(n_poly - 1, max(1, n_poly // 2))
        polytope = _poly_from_saved(data, index)
        _draw_box(ax, env.bounds, alpha=0.20, linewidth=0.4)
        _draw_obstacles(ax, env, alpha=0.09)
        ax.plot(*path.T, color=PLASMA(norm(gamma)), lw=1.35, alpha=0.78)
        ax.scatter(*polytope.center, marker="s", color="#111111", s=20)
        _add_levelsets(ax, polytope, gamma)
        _style(ax, env)
        ax.set_title(rf"$\gamma={gamma:g}$ — BLUE $P_k$ + all H=10 levels", fontsize=10,
                     weight="bold")
    for panel in range(len(gammas), rows * columns):
        ax = fig.add_subplot(rows, columns, panel + 1)
        ax.axis("off")
    fig.suptitle("Nominal polytope and nested level sets by gamma\n"
                 r"raw boundary $H_P=0$ is dark blue; levels enforce $H_P\geq(1-\gamma)^h$",
                 fontsize=14, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = output_dir / "nominal_levelsets_by_gamma.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all_figures(manifest, env, output_dir):
    return [plot_gamma_rollouts(manifest, env, output_dir),
            plot_nominal_levelsets(manifest, env, output_dir)]
