#!/usr/bin/env python3
"""Render actual mirrored trajectories and exact uniform-Hp100 evolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.geometry import (  # noqa: E402
    build_nominal_polytope,
    hp_values,
    triangular_geometry,
)
from safe_mppi.lab_clutter import (  # noqa: E402
    ClutterScene,
    config_for_scene,
)
from safe_mppi.lab_visual_flow import (  # noqa: E402
    LAB_HP100_GRID_SHAPE,
    LAB_HP100_SCHEMA,
    LAB_VISUAL_LOW_DIM,
    LabUniformHp100Rasterizer,
    build_uniform_hp100_context,
    uniform_hp100_grid_points,
)


def _box_edges(bounds: np.ndarray):
    low, high = bounds[:, 0], bounds[:, 1]
    corners = np.asarray([
        [x, y, z]
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ])
    for index, first in enumerate(corners):
        for second in corners[index + 1:]:
            if np.count_nonzero(~np.isclose(first, second)) == 1:
                yield first, second


def _draw_world(
    axis,
    env: TaskEnvironment,
    cylinders: np.ndarray,
    dense: np.ndarray,
    current: np.ndarray,
):
    bounds = np.asarray(env.bounds, np.float64)
    for first, second in _box_edges(bounds):
        axis.plot(*np.stack([first, second]).T, color="0.55", linewidth=0.8)
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    z = np.asarray([bounds[2, 0], bounds[2, 1]])
    for x, y, radius in cylinders:
        xx = x + radius * np.cos(theta)
        yy = y + radius * np.sin(theta)
        axis.plot_surface(
            np.broadcast_to(xx[:, None], (len(theta), 2)),
            np.broadcast_to(yy[:, None], (len(theta), 2)),
            np.broadcast_to(z[None], (len(theta), 2)),
            color="#8a8a8a", alpha=0.28, linewidth=0,
        )
        axis.plot(xx, yy, np.full_like(xx, bounds[2, 0]), color="0.35", linewidth=0.7)
        axis.plot(xx, yy, np.full_like(xx, bounds[2, 1]), color="0.35", linewidth=0.7)
    axis.plot(
        dense[:, 0], dense[:, 1], dense[:, 2],
        color="#0072b2", linewidth=2.0,
    )
    axis.scatter(*env.start[:3], marker="o", s=42, color="#009e73", label="start")
    axis.scatter(*env.goal, marker="*", s=70, color="#d55e00", label="goal")
    axis.scatter(*current, marker="o", s=48, color="#cc79a7", label="current")
    axis.set_xlim(*bounds[0])
    axis.set_ylim(*bounds[1])
    axis.set_zlim(*bounds[2])
    axis.set_box_aspect(np.ptp(bounds, axis=1))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("Executed trajectory in exact taskspace")
    axis.view_init(elev=24, azim=-54)


def _hp_snapshot(
    env: TaskEnvironment,
    state: np.ndarray,
    gamma: float,
    rasterizer: LabUniformHp100Rasterizer,
) -> dict:
    packed = build_uniform_hp100_context(env, state, gamma)
    with torch.no_grad():
        raster = rasterizer(torch.from_numpy(
            packed[LAB_VISUAL_LOW_DIM:][None],
        )).cpu().numpy()[0, 0]
    points = uniform_hp100_grid_points(state[:3])
    flat = points.reshape(-1, 3)
    polytope = build_nominal_polytope(
        state[:3],
        env.spheres,
        env.cylinders,
        env.bounds,
        sensing_range=env.mppi.sensing_range,
        obstacle_margin=0.0,
    )
    direct = np.clip(
        hp_values(polytope, flat), -1.0, 1.0,
    ).reshape(LAB_HP100_GRID_SHAPE[1:])
    maximum_error = float(np.max(np.abs(raster - direct)))
    if maximum_error > 2.0e-5:
        raise RuntimeError(
            "packed-plane rasterizer disagrees with direct nominal polytope: "
            f"max_abs_error={maximum_error:.8g}"
        )
    margins = np.maximum(polytope.margins, 1.0e-3)
    all_hp = (
        polytope.b[None] - flat @ polytope.A.T
    ) / margins[None]
    active_face = np.argmin(all_hp, axis=1)
    sensing_count = len(triangular_geometry()[2])
    wall_start = len(polytope.A) - 6
    source = np.full(len(flat), 1, np.int8)
    source[active_face < sensing_count] = 0
    source[active_face >= wall_start] = 2
    return {
        "packed": packed,
        "points": points,
        "hp": raster,
        "source": source.reshape(LAB_HP100_GRID_SHAPE[1:]),
        "maximum_raster_error": maximum_error,
        "polytope_face_count": len(polytope.A),
        "dynamic_face_count": len(polytope.A) - sensing_count,
    }


def _draw_hp(axis, snapshot: dict, state: np.ndarray):
    points = snapshot["points"] - state[None, None, None, :3]
    hp = snapshot["hp"]
    source = snapshot["source"]
    # Deterministic display decimation only. All assertions/statistics above use
    # every one of the 32x32x100 encoder cells.
    selector = np.zeros(hp.shape, bool)
    selector[::2, ::2, 4::5] = True
    shown = points[selector]
    shown_hp = hp[selector]
    shown_source = source[selector]
    ordinary = shown_source != 2
    walls = shown_source == 2
    axis.scatter(
        shown[ordinary, 0], shown[ordinary, 1], shown[ordinary, 2],
        c=shown_hp[ordinary], cmap="coolwarm", norm=Normalize(-1.0, 1.0),
        s=5, alpha=0.38, linewidths=0,
    )
    axis.scatter(
        shown[walls, 0], shown[walls, 1], shown[walls, 2],
        c=shown_hp[walls], cmap="coolwarm", norm=Normalize(-1.0, 1.0),
        s=11, alpha=0.85, edgecolors="black", linewidths=0.28,
        label="taskspace-wall-limited",
    )
    axis.scatter(0.0, 0.0, 0.0, color="black", s=28, marker="x")
    sensing = float(np.max(np.linalg.norm(points.reshape(-1, 3), axis=1)))
    axis.set_xlim(-sensing, sensing)
    axis.set_ylim(-sensing, sensing)
    axis.set_zlim(-sensing, sensing)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("relative x [m]")
    axis.set_ylabel("relative y [m]")
    axis.set_zlabel("relative z [m]")
    axis.set_title("Exact encoder clipped Hp (outlined = wall face)")
    axis.view_init(elev=24, azim=-54)


def _frame_indices(state_count: int, maximum: int) -> np.ndarray:
    count = min(int(maximum), int(state_count))
    return np.unique(np.rint(np.linspace(0, state_count - 1, count)).astype(int))


def _render_run(
    archive: Path,
    row: dict,
    output: Path,
    *,
    max_frames: int,
    frame_duration_ms: int,
) -> dict:
    config = load_config(archive / "resolved_config.json")
    run_path = archive / row["file"]
    with np.load(run_path, allow_pickle=False) as data:
        arrays = {
            key: data[key]
            for key in (
                "states", "dense_positions", "spheres", "cylinders",
                "scene_index", "scene_seed", "scene_hash",
            )
        }
    scene = ClutterScene(
        index=int(np.asarray(arrays["scene_index"]).item()),
        seed=int(np.asarray(arrays["scene_seed"]).item()),
        spheres=tuple(tuple(map(float, item)) for item in arrays["spheres"]),
        cylinders=tuple(tuple(map(float, item)) for item in arrays["cylinders"]),
        scene_hash=str(np.asarray(arrays["scene_hash"]).item()),
    )
    env = TaskEnvironment(config_for_scene(config, scene))
    states = np.asarray(arrays["states"], np.float32)
    dense = np.asarray(arrays["dense_positions"], np.float32)
    cylinders = np.asarray(arrays["cylinders"], np.float32)
    gamma = float(row["gamma"])
    indices = _frame_indices(len(states), max_frames)
    rasterizer = LabUniformHp100Rasterizer().eval()
    images = []
    frame_rows = []
    scalar = ScalarMappable(norm=Normalize(-1.0, 1.0), cmap="coolwarm")
    for frame_number, state_index in enumerate(indices):
        state = states[state_index]
        snapshot = _hp_snapshot(env, state, gamma, rasterizer)
        source = snapshot["source"]
        hp = snapshot["hp"]
        figure = plt.figure(figsize=(11.2, 5.4), dpi=100)
        world_axis = figure.add_subplot(1, 2, 1, projection="3d")
        hp_axis = figure.add_subplot(1, 2, 2, projection="3d")
        _draw_world(world_axis, env, cylinders, dense, state[:3])
        _draw_hp(hp_axis, snapshot, state)
        figure.suptitle(
            f"gamma={gamma:g} · {row['pair_id']} · {row['pair_member']} · "
            f"state {state_index}/{len(states)-1}"
        )
        wall_fraction = float(np.mean(source == 2))
        cylinder_fraction = float(np.mean(source == 1))
        figure.text(
            0.5, 0.018,
            f"z={state[2]:.3f} m · min Hp={hp.min():.3f} · "
            f"wall-limited={100*wall_fraction:.1f}% · "
            f"cylinder-limited={100*cylinder_fraction:.1f}% · "
            f"raster/direct max error={snapshot['maximum_raster_error']:.2e}",
            ha="center", va="bottom", fontsize=9,
        )
        colorbar = figure.colorbar(
            scalar, ax=[world_axis, hp_axis], fraction=0.025, pad=0.02,
        )
        colorbar.set_label("clipped Hp")
        figure.subplots_adjust(left=0.02, right=0.92, top=0.89, bottom=0.11, wspace=0.05)
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        images.append(Image.fromarray(rgba[:, :, :3].copy()))
        plt.close(figure)
        frame_rows.append({
            "frame_number": frame_number,
            "state_index": int(state_index),
            "position_m": state[:3].astype(float).tolist(),
            "minimum_clipped_hp": float(hp.min()),
            "wall_limited_cell_fraction": wall_fraction,
            "cylinder_limited_cell_fraction": cylinder_fraction,
            "sensing_limited_cell_fraction": float(np.mean(source == 0)),
            "maximum_raster_direct_error": snapshot["maximum_raster_error"],
            "polytope_face_count": snapshot["polytope_face_count"],
            "dynamic_face_count": snapshot["dynamic_face_count"],
        })
    stem = (
        f"gamma_{gamma:g}_{row['pair_id']}_{row['pair_member']}_"
        f"{row['scene_hash'][:8]}"
    ).replace(".", "p")
    name = f"{stem}.gif"
    gif_path = output / name
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    return {
        "archive": str(archive),
        "trajectory_file": row["file"],
        "gif": gif_path.name,
        "gamma": gamma,
        "pair_id": row["pair_id"],
        "pair_member": row["pair_member"],
        "scene_hash": row["scene_hash"],
        "success": bool(row["success"]),
        "individual_accepted": bool(row["individual_accepted"]),
        "pair_admitted": bool(row["pair_admitted"]),
        "taskspace_bounds_m": env.bounds.astype(float).tolist(),
        "sensing_range_m": float(env.mppi.sensing_range),
        "encoder_schema": LAB_HP100_SCHEMA,
        "encoder_cells": int(np.prod(LAB_HP100_GRID_SHAPE[1:])),
        "displayed_cells_per_frame": int(16 * 16 * 20),
        "frames": frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frame-duration-ms", type=int, default=140)
    args = parser.parse_args()
    if args.max_frames < 2 or args.frame_duration_ms < 10:
        parser.error("max frames must be >=2 and frame duration >=10 ms")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.gif")):
        raise FileExistsError(f"refusing to overwrite existing GIFs in {output}")
    rows = []
    for archive in args.archive:
        archive = archive.resolve()
        manifest = json.loads((archive / "manifest.json").read_text())
        for row in manifest["runs"]:
            rows.append((archive, row))
    if len(rows) != 8:
        raise ValueError(
            f"first-check visualization requires exactly 8 admitted runs; got {len(rows)}"
        )
    observed = {}
    for _, row in rows:
        observed.setdefault(float(row["gamma"]), []).append(row)
    if set(observed) != {0.1, 0.3, 0.5, 1.0} or any(
        len(group) != 2 for group in observed.values()
    ):
        raise ValueError("expected one admitted mirrored pair for each of four gammas")
    hashes = [str(row["scene_hash"]) for _, row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("a first-check scene was reused across gamma or pair members")
    report = [
        _render_run(
            archive,
            row,
            output,
            max_frames=args.max_frames,
            frame_duration_ms=args.frame_duration_ms,
        )
        for archive, row in sorted(
            rows,
            key=lambda item: (
                float(item[1]["gamma"]), int(item[1]["pair_member_index"]),
            ),
        )
    ]
    verification = {
        "status": "VERIFIED_EIGHT_MIRRORED_TRAJECTORY_HP100_GIFS",
        "encoder_schema": LAB_HP100_SCHEMA,
        "full_encoder_grid_shape": list(LAB_HP100_GRID_SHAPE),
        "full_encoder_cells_per_frame": int(np.prod(LAB_HP100_GRID_SHAPE[1:])),
        "direct_polytope_assertion_tolerance": 2.0e-5,
        "display_decimation": {
            "azimuth_stride": 2,
            "elevation_stride": 2,
            "radial_bins": "indices 4,9,...,99",
            "statistics_and_assertions_use_full_grid": True,
        },
        "runs": report,
    }
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps({
        "status": verification["status"],
        "gif_count": len(report),
        "output": str(output),
        "maximum_raster_direct_error": max(
            frame["maximum_raster_direct_error"]
            for run in report for frame in run["frames"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
