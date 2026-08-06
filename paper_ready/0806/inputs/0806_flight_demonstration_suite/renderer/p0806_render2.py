"""Dual-view (regular 3-D + ego-centric) MP4 renderer, depth-buffered (PyVista).

Revision after Dohyun's Phase-1 feedback (no-go on v1):
- no blinking: continuous motion; every step shows its polytope (SafeMPPI
  nominal always; pretrained verifier only when positive) or the red/False
  state, with the badge every frame
- true occlusion: executed trajectory is a 3-D tube in a depth-buffered
  renderer, so obstacles in front of it correctly hide it
- verifier polytope + level sets are intersected with the taskspace box for
  display (the verifier SOCP has no taskspace faces, unlike the nominal)
- level-set opacity ramps: inner (small h) clear, outer very transparent
- right panel: ego-centric chase camera aligned with the velocity vector
  (records in the robot's direction of motion), world up preserved

Output: one MP4 per policy/gamma, 1440x720 (two 720x720 panels), H.264
yuv420p +faststart, fps 10, 2 frames per replanning step + 1 s terminal hold.
Private tooling; lives only under /data3/research1/paper_ready_0806_private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

import pyvista as pv

pv.OFF_SCREEN = True

POSITIVE_BLUE = "#0057FF"
NEGATIVE_RED = "#E31A1C"
VERIFIER_GREEN = "#00A651"
EXECUTED_BLACK = "#111111"
GOAL_GOLD = "#F0B400"
CYL_MODEL_BLUE = "#2b8cbe"
CYL_PHYS_GRAY = "#747b80"
BOX_GRAY = "#9aa2a8"

STYLE_VERSION = "p0806_3d_video_style_v2_pyvista"
PANEL = 720
FPS = 10
FRAMES_PER_STEP = 2
HOLD_SECONDS = 1.0


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polyline(points):
    points = np.asarray(points, float)
    n = len(points)
    cells = np.concatenate([[n], np.arange(n)]).astype(np.int64)
    return pv.PolyData(points, lines=cells)


def multi_polyline(list_of_points):
    all_pts = []
    cells = []
    offset = 0
    for pts in list_of_points:
        pts = np.asarray(pts, float)
        all_pts.append(pts)
        cells.append(np.concatenate([[len(pts)], offset + np.arange(len(pts))]))
        offset += len(pts)
    return pv.PolyData(np.concatenate(all_pts), lines=np.concatenate(cells).astype(np.int64))


def box_halfspaces(bounds):
    """Taskspace box as A x <= b (6 faces)."""
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.concatenate([bounds[:, 1], -bounds[:, 0]])
    return A, b


def clipped_hull(A, b, interior, box_A=None, box_b=None):
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if box_A is not None:
        A = np.vstack([A, box_A])
        b = np.concatenate([b, box_b])
    interior = np.asarray(interior, float)
    if (b - A @ interior).min() <= 1e-9:
        return None
    try:
        verts = HalfspaceIntersection(np.hstack([A, -b[:, None]]), interior).intersections
        if len(verts) < 4:
            return None
        hull = ConvexHull(verts)
        faces = np.hstack([np.full((len(hull.simplices), 1), 3, np.int64),
                           hull.simplices.astype(np.int64)]).ravel()
        return pv.PolyData(verts, faces=faces)
    except Exception:
        return None


def add_levels(pl, A, center, margins, gamma, horizon, color, box_A, box_b,
               inner_alpha, outer_alpha, edge_alpha):
    """Outer polytope wireframe + nested level-set fills, clipped to the box.

    Level h: {x : A x <= A c + beta_h * margins}, beta_h = 1-(1-gamma)^h.
    Inner (h=1) most opaque -> outer (h=H) most transparent.
    """
    center = np.asarray(center, float)
    margins = np.asarray(margins, float)
    b_full = A @ center + margins
    outer = clipped_hull(A, b_full, center, box_A, box_b)
    if outer is None:
        return False
    pl.add_mesh(outer.extract_all_edges(), color=color, line_width=1.0,
                opacity=edge_alpha)
    betas = 1.0 - (1.0 - float(gamma)) ** np.arange(1, horizon + 1)
    if np.allclose(betas, betas[-1]):
        draw = [horizon - 1]
    else:
        draw = list(range(horizon))
    alphas = np.linspace(inner_alpha, outer_alpha, horizon)
    for h0 in draw:
        b_level = A @ center + betas[h0] * margins
        hull = clipped_hull(A, b_level, center, box_A, box_b)
        if hull is None:
            continue
        pl.add_mesh(hull, color=color, opacity=float(alphas[h0]),
                    smooth_shading=False)
        if h0 == draw[0]:
            pl.add_mesh(hull.extract_all_edges(), color=color, line_width=1.0,
                        opacity=min(0.55, 3.5 * float(alphas[h0])))
    return True


class DualViewRenderer:
    def __init__(self, config, env):
        self.bounds = np.asarray(config.taskspace.bounds, float)
        self.start = np.asarray(env.start[:3], float)
        self.goal = np.asarray(env.goal, float)
        self.cylinders = np.asarray(env.cylinders, float).reshape(-1, 3)
        self.box_A, self.box_b = box_halfspaces(self.bounds)
        self.center = self.bounds.mean(axis=1)
        self.extent = self.bounds[:, 1] - self.bounds[:, 0]
        self.pl = pv.Plotter(off_screen=True, window_size=(PANEL, PANEL))
        # NOTE: depth peeling is broken on this OSMesa build (all translucent
        # actors vanish) -- default alpha blending is used instead.
        self.pl.set_background("white")
        self._fwd = None

    def close(self):
        self.pl.close()

    def _static_scene(self):
        pl = self.pl
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))
        pl.add_mesh(box.extract_all_edges(), color=BOX_GRAY, line_width=1.2,
                    opacity=0.55)
        height = z1 - z0
        zc = 0.5 * (z0 + z1)
        for cx, cy, r_model in self.cylinders:
            phys = pv.Cylinder(center=(cx, cy, zc), direction=(0, 0, 1),
                               radius=0.10, height=height, resolution=48)
            pl.add_mesh(phys, color=CYL_PHYS_GRAY, opacity=1.0,
                        smooth_shading=True)
            model = pv.Cylinder(center=(cx, cy, zc), direction=(0, 0, 1),
                                radius=float(r_model), height=height,
                                resolution=48)
            pl.add_mesh(model, color=CYL_MODEL_BLUE, opacity=0.13,
                        smooth_shading=True)
        pl.add_mesh(pv.Cube(center=self.start, x_length=0.07, y_length=0.07,
                            z_length=0.07), color=EXECUTED_BLACK)
        pl.add_mesh(pv.Sphere(radius=0.07, center=self.goal), color=GOAL_GOLD)

    def badge(self, safe):
        color = POSITIVE_BLUE if safe else NEGATIVE_RED
        actor = self.pl.add_text(
            f"Multi-step safety: {bool(safe)}",
            position=(0.035, 0.93), viewport=True,
            font_size=15, color=color, font="times", shadow=False,
        )
        prop = actor.GetTextProperty()
        prop.SetBold(True)
        prop.SetBackgroundColor(1.0, 1.0, 1.0)
        prop.SetBackgroundOpacity(0.78)
        return actor

    def executed_path(self, dense, upto_index):
        seg = np.asarray(dense, float)[:upto_index]
        if len(seg) >= 2:
            tube = polyline(seg).tube(radius=0.010, n_sides=10)
            self.pl.add_mesh(tube, color=EXECUTED_BLACK, smooth_shading=True)

    def robot(self, position):
        self.pl.add_mesh(pv.Sphere(radius=0.030, center=np.asarray(position, float)),
                         color=EXECUTED_BLACK)

    def update_forward(self, velocity, fallback):
        v = np.asarray(velocity, float)
        speed = np.linalg.norm(v)
        direction = v / speed if speed > 0.05 else None
        if direction is None:
            if self._fwd is None:
                direction = fallback / np.linalg.norm(fallback)
                self._fwd = direction
        else:
            if self._fwd is None:
                self._fwd = direction
            else:
                mixed = 0.65 * self._fwd + 0.35 * direction
                self._fwd = mixed / np.linalg.norm(mixed)
        return self._fwd

    def camera_regular(self):
        cam = self.pl.camera
        self.pl.disable_parallel_projection()
        elev, azim = np.deg2rad(25.0), np.deg2rad(-57.0)
        direction = np.array([np.cos(elev) * np.cos(azim),
                              np.cos(elev) * np.sin(azim),
                              np.sin(elev)])
        distance = 2.05 * float(np.max(self.extent))
        cam.focal_point = self.center
        cam.position = self.center + distance * direction
        cam.up = (0.0, 0.0, 1.0)
        cam.view_angle = 30.0

    def camera_ego(self, position, forward):
        cam = self.pl.camera
        self.pl.disable_parallel_projection()
        p = np.asarray(position, float)
        f = np.asarray(forward, float)
        cam.position = p - 0.55 * f + np.array([0.0, 0.0, 0.22])
        cam.focal_point = p + 1.6 * f
        cam.up = (0.0, 0.0, 1.0)
        cam.view_angle = 78.0

    def grab(self):
        # force a fresh render: screenshot alone returns a stale buffer when
        # only the camera moved since the last render
        self.pl.render()
        return self.pl.screenshot(return_img=True)


def draw_safemppi(view, rec, cur, gamma, horizon, cap):
    pl = view.pl
    pos = np.asarray(rec["candidate_positions"], float)
    infeasible = np.asarray(rec["candidate_infeasible"], bool)
    fail_step = np.asarray(rec["candidate_fail_step"])
    n = len(pos)
    if n > cap:
        keep_pos_idx = np.flatnonzero(~infeasible)
        keep_neg_idx = np.flatnonzero(infeasible)
        k_pos = int(round(cap * len(keep_pos_idx) / n))
        k_neg = cap - k_pos

        def spread(idx, k):
            if k <= 0 or not len(idx):
                return np.empty(0, int)
            sel = np.linspace(0, len(idx) - 1, min(k, len(idx))).round().astype(int)
            return idx[np.unique(sel)]

        keep = np.concatenate([spread(keep_pos_idx, k_pos),
                               spread(keep_neg_idx, k_neg)])
    else:
        keep = np.arange(n)
    cur3 = np.asarray(cur, float)[:3]
    pos_lines = [np.vstack([cur3, pos[i]]) for i in keep if not infeasible[i]]
    neg_lines = [np.vstack([cur3, pos[i]]) for i in keep if infeasible[i]]
    if neg_lines:
        pl.add_mesh(multi_polyline(neg_lines), color=NEGATIVE_RED,
                    line_width=1.4, opacity=0.16)
    if pos_lines:
        pl.add_mesh(multi_polyline(pos_lines), color=POSITIVE_BLUE,
                    line_width=1.4, opacity=0.22)
    fails = [pos[i, int(fail_step[i])] for i in keep
             if infeasible[i] and int(fail_step[i]) >= 0]
    if fails:
        cloud = pv.PolyData(np.asarray(fails, float))
        pl.add_mesh(cloud.glyph(geom=pv.Sphere(radius=0.011), scale=False,
                                orient=False),
                    color=NEGATIVE_RED, opacity=0.85)
    exec_pts = np.vstack([cur3, np.asarray(rec["executed_plan_positions"], float)])
    exec_ok = not rec["executed_plan_infeasible"]
    color = POSITIVE_BLUE if exec_ok else NEGATIVE_RED
    pl.add_mesh(polyline(exec_pts).tube(radius=0.013, n_sides=10), color=color)
    if not exec_ok:
        fs = max(int(rec["executed_plan_fail_step"]), 0)
        pl.add_mesh(pv.Sphere(radius=0.030,
                              center=np.asarray(rec["executed_plan_positions"], float)[fs]),
                    color=NEGATIVE_RED)
    # nominal polytope always (its box faces already bound it to the taskspace)
    A = np.asarray(rec["poly_A"], float)
    b = np.asarray(rec["poly_b"], float)
    center = np.asarray(rec["poly_center"], float)
    margins = np.maximum(b - A @ center, 1e-9)
    add_levels(pl, A, center, margins, gamma, horizon, POSITIVE_BLUE,
               None, None, inner_alpha=0.11, outer_alpha=0.010,
               edge_alpha=0.30)
    return exec_ok


def draw_pretrained(view, steps, t, gamma, horizon):
    pl = view.pl
    history = [np.asarray(steps[j]["window_knots"], float)[:, :3]
               for j in range(t)]
    hist_pos = [k for j, k in enumerate(history) if steps[j]["positive"]]
    hist_neg = [k for j, k in enumerate(history) if not steps[j]["positive"]]
    if hist_pos:
        pl.add_mesh(multi_polyline(hist_pos), color=POSITIVE_BLUE,
                    line_width=1.1, opacity=0.06)
    if hist_neg:
        pl.add_mesh(multi_polyline(hist_neg), color=NEGATIVE_RED,
                    line_width=1.1, opacity=0.08)
    rec = steps[t]
    knots = np.asarray(rec["window_knots"], float)[:, :3]
    positive = bool(rec["positive"])
    if positive:
        A = np.asarray(rec["verifier_A"], float)
        center = np.asarray(rec["verifier_center"], float)
        margins = np.asarray(rec["verifier_margins"], float)
        add_levels(pl, A, center, margins, gamma, horizon, VERIFIER_GREEN,
                   view.box_A, view.box_b, inner_alpha=0.16,
                   outer_alpha=0.008, edge_alpha=0.22)
        pl.add_mesh(polyline(knots).tube(radius=0.013, n_sides=10),
                    color=POSITIVE_BLUE)
        cloud = pv.PolyData(knots[1:])
        pl.add_mesh(cloud.glyph(geom=pv.Sphere(radius=0.016), scale=False,
                                orient=False), color=POSITIVE_BLUE)
    else:
        pl.add_mesh(polyline(knots).tube(radius=0.012, n_sides=10),
                    color=NEGATIVE_RED)
        fk = int(rec["first_fail_knot"])
        fk = fk if fk >= 0 else len(knots) - 1
        pl.add_mesh(pv.Sphere(radius=0.030, center=knots[min(fk, len(knots) - 1)]),
                    color=NEGATIVE_RED)
    return positive


def render(args):
    import torch
    sys.path.insert(0, args.repo)
    from safe_mppi.config import load_config
    from safe_mppi.environment import TaskEnvironment

    config = load_config(args.config)
    env = TaskEnvironment(config)
    substeps = int(config.safemppi.integration_substeps)
    horizon = int(config.safemppi.horizon)

    with np.load(args.run_npz, allow_pickle=True) as data:
        run = {k: data[k] for k in data.files}
    events = torch.load(args.events, map_location="cpu", weights_only=False)
    steps = events["steps"]
    status = str(events["meta"]["status"])
    gamma = float(events["meta"]["gamma"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = f"{2 * PANEL}x{PANEL}"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", size, "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-crf", "19", "-preset", "medium", str(out)],
        stdin=subprocess.PIPE,
    )

    view = DualViewRenderer(config, env)
    dense = np.asarray(run["dense_positions"], float)
    states = np.asarray(run["states"], float)
    goal_dir = view.goal - view.start
    n_steps = len(steps)
    frame_count = 0

    def emit_frame(badge_state, draw_dynamic, ego_anchor, forward):
        nonlocal frame_count
        pl = view.pl
        pl.clear_actors()
        view._static_scene()
        draw_dynamic()
        badge = view.badge(badge_state) if badge_state is not None else None
        view.camera_regular()
        left = view.grab()
        if badge is not None:
            pl.remove_actor(badge)
        view.camera_ego(ego_anchor, forward)
        right = view.grab()
        frame = np.hstack([left, right])
        ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
        frame_count += 1

    for t in range(n_steps):
        rec = steps[t]
        cur = states[t]
        forward = view.update_forward(cur[3:6], goal_dir)

        def dynamic(rec=rec, cur=cur, t=t):
            view.executed_path(dense, 1 + t * substeps)
            view.robot(cur[:3])
            if args.mode == "safemppi":
                dynamic.safe = draw_safemppi(view, rec, cur, gamma, horizon,
                                             args.display_cap)
            else:
                dynamic.safe = draw_pretrained(view, steps, t, gamma, horizon)

        # draw once to know badge state, reuse for both frames of the step
        pl = view.pl
        pl.clear_actors()
        view._static_scene()
        dynamic()
        badge_state = dynamic.safe
        badge = view.badge(badge_state)
        view.camera_regular()
        left = view.grab()
        pl.remove_actor(badge)
        badge2 = view.badge(badge_state)
        view.camera_ego(cur[:3], forward)
        right = view.grab()
        pl.remove_actor(badge2)
        frame = np.ascontiguousarray(np.hstack([left, right])).tobytes()
        for _ in range(FRAMES_PER_STEP):
            ffmpeg.stdin.write(frame)
            frame_count += 1

    # terminal hold: full path + outcome marker
    end = dense[-1]
    forward = view._fwd if view._fwd is not None else goal_dir / np.linalg.norm(goal_dir)

    def hold_dynamic():
        view.executed_path(dense, len(dense))
        if status == "SUCCESS":
            view.robot(end)
        else:
            view.pl.add_mesh(pv.Sphere(radius=0.035, center=end),
                             color=NEGATIVE_RED)

    for _ in range(int(HOLD_SECONDS * FPS)):
        emit_frame(None, hold_dynamic, end, forward)

    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit("ffmpeg encode failed")
    view.close()

    sidecar = {
        "status": "P0806_3D_VIDEO_COMPLETE",
        "style_version": STYLE_VERSION,
        "mode": args.mode,
        "mp4": out.name,
        "mp4_sha256": sha256_file(out),
        "bytes": out.stat().st_size,
        "gamma": gamma,
        "episode_status": status,
        "fps": FPS,
        "frames": frame_count,
        "frames_per_step": FRAMES_PER_STEP,
        "blinking": False,
        "panels": {"left": "regular 3-D, fixed camera (elev 25, azim -57)",
                   "right": "ego-centric chase camera along velocity"},
        "resolution": [2 * PANEL, PANEL],
        "renderer": f"pyvista {pv.__version__} (OSMesa, depth-buffered)",
        "display_cap": args.display_cap if args.mode == "safemppi" else 1,
        "verifier_polytope_clipped_to_taskspace": args.mode == "pretrained",
        "level_set_opacity": "inner clear -> outer transparent",
        "run_npz": Path(args.run_npz).name,
        "events": Path(args.events).name,
        "colors": {"positive": POSITIVE_BLUE, "negative": NEGATIVE_RED,
                   "verifier": VERIFIER_GREEN, "executed": EXECUTED_BLACK},
    }
    Path(str(out) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"mp4": str(out), "frames": frame_count,
                      "bytes": sidecar["bytes"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["safemppi", "pretrained"], required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-npz", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--display-cap", type=int, default=160)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
