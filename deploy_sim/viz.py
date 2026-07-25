#!/usr/bin/env python3
"""Shared flight rendering: trajectory + BLUE nominal polytope + H_P level sets.

Used by both the offline harness and the hardware log tools, so an offline run
and a real flight are drawn the same way.

The nominal polytope is the repo's own online geometry — the 80-face triangular
sensing polyhedron intersected with obstacle tangent halfspaces and the taskspace
box (`safe_mppi.geometry.build_nominal_polytope`) — and the nested translucent
shells are the ten H_P horizon level sets, drawn with the repo's own
the same BLUE ramp as in the README's ball-below GIFs.

Level shell h is scaled by (1-gamma)^h about the polytope centre, so a small
gamma shows ten tightly nested shells (motion contracts gradually) and gamma=1
collapses them onto the H_P=0 boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from safe_mppi.geometry import (build_nominal_polytope, hull_edges,  # noqa: E402
                                polytope_vertices)
from safe_mppi.visualize import BLUE, _draw_box, _draw_obstacles, _style  # noqa: E402

INK = "#1f2430"
MUTED = "#6b7280"
PATH_CMAP = "viridis"


def draw_polytope_levels(ax, polytope, gamma, n_levels=5):
    """BLUE nominal polytope + its H_P level sets, drawn to stay legible.

    The repo's `draw_polytope_soft` stacks ten 6%-alpha fills, which reads as one
    blob once the taskspace box clips the polytope on every side (our arena is
    smaller than the 2 m sensing range). Here each shown level keeps a soft fill
    but also gets a crisp outline, on the same light->dark BLUE ramp, so the
    nested contraction is visible. Shell h is scaled by (1-gamma)^h about the
    centre: H_P >= (1-gamma) H_P is the admissible one-step contraction.
    """
    vertices = polytope_vertices(polytope)
    if vertices is None:
        ax.text2D(0.03, 0.92, "polytope not strict-interior", transform=ax.transAxes,
                  fontsize=7, color="#b3261e")
        return None
    hull = ConvexHull(vertices)
    _, edges = hull_edges(vertices)
    # Outer boundary P_k: one thin wire frames it; shells are fills only, so the
    # nesting reads as concentric density instead of overlapping wireframes.
    ax.add_collection3d(Line3DCollection(vertices[edges], colors="#7ba3d4",
                                         linewidths=0.3, alpha=0.30))
    betas = 1.0 - (1.0 - float(gamma)) ** np.arange(1, 11)  # shell radius fraction
    picks = [9] if np.allclose(betas, betas[0]) else \
        np.unique(np.linspace(0, 9, n_levels).astype(int))
    for n, h in enumerate(picks):
        lv = polytope.center + betas[h] * (vertices - polytope.center)
        shade = BLUE(0.32 + 0.55 * (n / max(len(picks) - 1, 1)))
        ax.add_collection3d(Poly3DCollection([lv[s] for s in hull.simplices],
                                             facecolor=shade, alpha=0.055,
                                             edgecolor="none"))
    return vertices


def draw_polytope_slice(ax, polytope, gamma, z0, n_levels=10):
    """Horizontal cut of P_k at height z0 with the H_P level sets as contours.

    Nested translucent shells are hard to read in 3D; a slice shows the same ten
    levels as crisp closed curves, which is how level sets are normally read.
    """
    from scipy.spatial import HalfspaceIntersection
    A, b, c = polytope.A, polytope.b, polytope.center
    # a1 x + a2 y + (a3 z0 - b) <= 0
    hs = np.hstack([A[:, :2], (A[:, 2] * float(z0) - b)[:, None]])
    interior = np.asarray(c[:2], float)
    if np.any(hs[:, :2] @ interior + hs[:, 2] >= -1e-9):
        ax.text(0.03, 0.92, "slice empty at this height", transform=ax.transAxes,
                fontsize=7.5, color="#b3261e")
        return
    pts = HalfspaceIntersection(hs, interior).intersections
    ang = np.arctan2(pts[:, 1] - interior[1], pts[:, 0] - interior[0])
    poly = pts[np.argsort(ang)]
    betas = 1.0 - (1.0 - float(gamma)) ** np.arange(1, 11)
    picks = [9] if np.allclose(betas, betas[0]) else \
        np.unique(np.linspace(0, 9, n_levels).astype(int))
    for n, h in enumerate(picks):
        lv = interior + betas[h] * (poly - interior)
        shade = BLUE(0.30 + 0.58 * (n / max(len(picks) - 1, 1)))
        ax.fill(lv[:, 0], lv[:, 1], color=shade, alpha=0.10, zorder=2)
        ax.plot(np.r_[lv[:, 0], lv[0, 0]], np.r_[lv[:, 1], lv[0, 1]],
                color=shade, lw=1.0, alpha=0.85, zorder=3)
    ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
            color="#1f4e8c", lw=1.6, zorder=4)


def polytope_at(env, cfg, position):
    """The online nominal polytope the controller would build at `position`."""
    return build_nominal_polytope(
        np.asarray(position, float)[:3], env.spheres, env.cylinders, env.bounds,
        sensing_range=cfg.sensing_range, obstacle_margin=cfg.obstacle_margin)


def _sphere3d(ax, c, r, *, color="#8f969f", alpha=0.26):
    u, w = np.mgrid[0:2 * np.pi:22j, 0:np.pi:11j]
    ax.plot_surface(c[0] + r * np.cos(u) * np.sin(w), c[1] + r * np.sin(u) * np.sin(w),
                    c[2] + r * np.cos(w), color=color, alpha=alpha, linewidth=0, shade=True)


def _scene(ax, env, *, obstacles=True, spheres=None):
    """`spheres` (N,4) overrides the config obstacles — pass the LOGGED ones so
    the drawing always matches the ball actually used at that instant."""
    _draw_box(ax, env.bounds, alpha=0.30, linewidth=0.6)
    if spheres is not None:
        for sx, sy, sz, sr in np.asarray(spheres, float).reshape(-1, 4):
            if np.isfinite([sx, sy, sz, sr]).all():
                _sphere3d(ax, (sx, sy, sz), sr)
    elif obstacles:
        _draw_obstacles(ax, env, alpha=0.24)
    ax.scatter(*env.start[:3], marker="s", color="#2f9e44", s=42,
               depthshade=False, label="start", zorder=6)
    ax.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00",
               s=185, depthshade=False, label="goal", zorder=6)


def _path(ax, P, lw=2.3, cmap=PATH_CMAP):
    """Trajectory coloured by normalised time (sequential ramp = magnitude)."""
    pts = np.asarray(P, float).reshape(-1, 1, 3)
    if len(pts) < 2:
        return None
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw, zorder=5)
    lc.set_array(np.linspace(0.0, 1.0, len(segs)))
    ax.add_collection3d(lc)
    return lc


def _label(ax, text):
    ax.set_title(text, fontsize=9.5, color=INK, weight="bold", pad=2)


def render_static(P, env, cfg, gamma, out_png, *, title="", at=None, t=None,
                  ceiling=None, obstacles=None):
    """Path overview, the 3D polytope + H_P levels, a level-set slice, altitude."""
    P = np.asarray(P, float)
    if at is None:      # most informative moment: closest approach to an obstacle
        clr = env.obstacle_clearance(P)
        i = int(np.argmin(clr)) if np.isfinite(clr).any() else len(P) // 2
    else:
        i = int(at)
    tt = (np.arange(len(P)) * cfg.dt) if t is None else (np.asarray(t, float) - t[0])
    # Use the obstacle recorded at this instant, not whatever the config says.
    OB = None if obstacles is None else np.asarray(obstacles, float).reshape(len(P), -1, 4)
    if OB is not None and np.isfinite(OB[i]).all():
        env.spheres = OB[i].astype(np.float32)
    ob_i = None if OB is None else OB[i]
    poly = polytope_at(env, cfg, P[i])

    fig = plt.figure(figsize=(12.4, 9.4), facecolor="white")

    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    _scene(ax0, env, spheres=ob_i)
    lc = _path(ax0, P)
    _style(ax0, env)
    _label(ax0, "flown path  (colour = time)")
    ax0.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax0, shrink=0.48, pad=0.14)
        cb.set_label("normalised time", fontsize=8, color=MUTED)
        cb.ax.tick_params(labelsize=7, colors=MUTED)

    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    _scene(ax1, env, spheres=ob_i)
    ax1.plot(P[:, 0], P[:, 1], P[:, 2], color="#c3c9d2", lw=1.0, zorder=3)
    ax1.plot(P[:i + 1, 0], P[:i + 1, 1], P[:i + 1, 2], color="#3f7ad0", lw=1.8, zorder=4)
    try:
        draw_polytope_levels(ax1, poly, gamma)
    except Exception as exc:
        ax1.text2D(0.03, 0.92, f"polytope n/a: {exc}", transform=ax1.transAxes,
                   fontsize=6.5, color="#b3261e")
    ax1.scatter(*P[i], color="#d1495b", s=42, depthshade=False, zorder=7)
    _style(ax1, env)
    _label(ax1, rf"3D $P_k$ + $H_P$ levels   t={tt[i]:.1f}s")

    ax2 = fig.add_subplot(2, 2, 3)
    try:
        draw_polytope_slice(ax2, poly, gamma, P[i, 2])
    except Exception as exc:
        ax2.text(0.03, 0.92, f"slice n/a: {exc}", transform=ax2.transAxes,
                 fontsize=7, color="#b3261e")
    for sx, sy, sz, sr in (env.spheres if ob_i is None else ob_i):   # cut at this height
        d = abs(P[i, 2] - sz)
        if d < sr:
            rr = float(np.sqrt(sr ** 2 - d ** 2))
            ax2.add_patch(plt.Circle((sx, sy), rr, color="#8f969f", alpha=0.55, zorder=5))
    ax2.plot(P[:, 0], P[:, 1], color="#c3c9d2", lw=1.0, zorder=6)
    ax2.plot(P[:i + 1, 0], P[:i + 1, 1], color="#3f7ad0", lw=1.6, zorder=6)
    ax2.scatter(*P[i, :2], color="#d1495b", s=32, zorder=7)
    ax2.scatter(*env.goal[:2], marker="*", color="#ffca28", edgecolor="#6a4e00", s=150, zorder=7)
    ax2.set_xlim(*env.bounds[0]); ax2.set_ylim(*env.bounds[1]); ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]", fontsize=9, color=MUTED); ax2.set_ylabel("y [m]", fontsize=9, color=MUTED)
    ax2.tick_params(labelsize=8, colors=MUTED); ax2.grid(alpha=0.16)
    for s in ax2.spines.values():
        s.set_color("#d7dbe0")
    _label(ax2, f"ten $H_P$ level sets, slice z={P[i, 2]:.2f} m")

    ax3 = fig.add_subplot(2, 2, 4)
    for sx, sy, sz, sr in (env.spheres if ob_i is None else ob_i):
        ax3.axhspan(sz - sr, sz + sr, color="#8f969f", alpha=0.20, zorder=1)
    ax3.plot(tt, P[:, 2], color="#3f7ad0", lw=1.9, zorder=3)
    ax3.axvline(tt[i], color="#d1495b", lw=1.0, alpha=0.7)
    if ceiling is not None:
        ax3.axhline(ceiling, color="#d1495b", lw=1.2, ls="--", alpha=0.85)
        ax3.text(0.01, ceiling, " safety ceiling", va="bottom", fontsize=7.5,
                 color="#d1495b", transform=ax3.get_yaxis_transform())
    ax3.set_xlabel("t [s]", fontsize=9, color=MUTED)
    ax3.set_ylabel("z [m]", fontsize=9, color=MUTED)
    ax3.tick_params(labelsize=8, colors=MUTED); ax3.grid(alpha=0.16)
    for s in ax3.spines.values():
        s.set_color("#d7dbe0")
    _label(ax3, "altitude  (band = obstacle extent)")

    head = title or "SafeMPPI flight"
    fig.suptitle(f"{head}   —   BLUE nominal polytope $P_k$ (80 triangular faces) and its "
                 rf"$H_P$ level sets, $\gamma={gamma:g}$",
                 fontsize=11.5, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    fig.savefig(out_png, dpi=145)
    plt.close(fig)
    return out_png


def render_gif(P, env, cfg, gamma, out_gif, *, t=None, fps=12, frames=44,
               title="", ceiling=None, obstacles=None):
    """Orbiting 3D view with the polytope evolving along the path + altitude panel."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    P = np.asarray(P, float)
    t = np.arange(len(P)) if t is None else np.asarray(t, float)
    steps = np.linspace(0, len(P) - 1, frames).astype(int)
    OB = None if obstacles is None else np.asarray(obstacles, float).reshape(len(P), -1, 4)

    fig = plt.figure(figsize=(11.8, 5.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    def draw(f):
        i = steps[f]
        ax3d.cla(); ax2d.cla()
        ob_i = None
        if OB is not None and np.isfinite(OB[i]).all():
            ob_i = OB[i]
            env.spheres = ob_i.astype(np.float32)   # polytope uses the live ball too
        _scene(ax3d, env, spheres=ob_i)
        ax3d.plot(P[:, 0], P[:, 1], P[:, 2], color="#d3d8df", lw=0.9, zorder=2)
        _path(ax3d, P[:i + 1])
        try:
            draw_polytope_levels(ax3d, polytope_at(env, cfg, P[i]), gamma)
        except Exception:
            pass
        ax3d.scatter(*P[i], color="#d1495b", s=46, depthshade=False, zorder=7)
        _style(ax3d, env)
        prog = f / max(frames - 1, 1)
        ax3d.view_init(elev=20.0 + 7.0 * np.sin(2 * np.pi * prog),
                       azim=-112.0 + 104.0 * prog)
        _label(ax3d, rf"$P_k$ + $H_P$ levels, $\gamma={gamma:g}$   t={t[i] - t[0]:.1f}s")

        ax2d.plot(t - t[0], P[:, 2], color="#c3c9d2", lw=1.4)
        ax2d.plot(t[:i + 1] - t[0], P[:i + 1, 2], color="#3f7ad0", lw=2.0)
        ax2d.scatter(t[i] - t[0], P[i, 2], color="#d1495b", s=34, zorder=5)
        if ceiling is not None:
            ax2d.axhline(ceiling, color="#d1495b", lw=1.1, ls="--", alpha=0.8)
            ax2d.text(0.02, ceiling, " safety ceiling", va="bottom", fontsize=7.5,
                      color="#d1495b", transform=ax2d.get_yaxis_transform())
        for sx, sy, sz, sr in (env.spheres if ob_i is None else ob_i):
            ax2d.axhspan(sz - sr, sz + sr, color="#8f969f", alpha=0.18)
        ax2d.set_xlabel("t [s]", fontsize=9, color=MUTED)
        ax2d.set_ylabel("z [m]", fontsize=9, color=MUTED)
        ax2d.set_title("altitude", fontsize=9.5, color=INK, weight="bold")
        ax2d.tick_params(labelsize=8, colors=MUTED)
        ax2d.grid(alpha=0.18)
        for s in ax2d.spines.values():
            s.set_color("#d7dbe0")
        return []

    if title:
        fig.suptitle(title, fontsize=11, color=INK, weight="bold")
    draw(frames // 2)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1.0))
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=96)
    plt.close(fig)
    return out_gif
