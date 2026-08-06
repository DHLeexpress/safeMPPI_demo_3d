#!/usr/bin/env python3
"""Render one measured 0806 flight log without touching frozen sim assets.

Expected input is Minhyuk's existing NPZ schema: measured ``x,y,z``, commanded
``cx,cy,cz``, timestamp ``t``, and optional ``cyl``/``sph`` geometry.  Copy
this template into a new ``minhyuk/runs/<run_id>/`` directory before editing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv


PANEL = 720
FPS = 30
MEASURED = "#111111"
REFERENCE = "#0057FF"
GOAL = "#F0B400"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polyline(points: np.ndarray) -> pv.PolyData:
    points = np.asarray(points, float)
    cells = np.concatenate([[len(points)], np.arange(len(points))]).astype(np.int64)
    return pv.PolyData(points, lines=cells)


def camera_from_json(path: Path) -> dict:
    camera = json.loads(path.read_text())
    required = {"position", "focal_point", "up", "view_angle_deg"}
    missing = sorted(required.difference(camera))
    if missing:
        raise ValueError(f"camera calibration is missing {missing}")
    return camera


def load_log(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        required = {"t", "x", "y", "z"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"flight log is missing {missing}")
        measured = np.column_stack([archive["x"], archive["y"], archive["z"]])
        if {"cx", "cy", "cz"}.issubset(archive.files):
            commanded = np.column_stack(
                [archive["cx"], archive["cy"], archive["cz"]]
            )
        else:
            commanded = measured.copy()
        goal = (
            np.asarray([archive["tx"][0], archive["ty"][0], archive["tz"][0]], float)
            if {"tx", "ty", "tz"}.issubset(archive.files)
            else commanded[-1]
        )
        cylinders = (
            np.asarray(archive["cyl"], float).reshape(-1, 3)
            if "cyl" in archive.files
            else np.empty((0, 3), float)
        )
        spheres = (
            np.asarray(archive["sph"], float).reshape(-1, 4)
            if "sph" in archive.files
            else np.empty((0, 4), float)
        )
        time_s = np.asarray(archive["t"], float)
    if not (len(time_s) == len(measured) == len(commanded)):
        raise ValueError("time, measured position, and command arrays are misaligned")
    return {
        "time_s": time_s,
        "measured": measured,
        "commanded": commanded,
        "goal": goal,
        "cylinders": cylinders,
        "spheres": spheres,
    }


def add_static(plotter: pv.Plotter, data: dict) -> None:
    for x, y, radius in data["cylinders"]:
        cylinder = pv.Cylinder(
            center=(x, y, 1.2), direction=(0, 0, 1), radius=radius,
            height=2.4, resolution=48,
        )
        plotter.add_mesh(cylinder, color="#747b80", opacity=0.65)
    for x, y, z, radius in data["spheres"]:
        plotter.add_mesh(
            pv.Sphere(center=(x, y, z), radius=radius),
            color="#747b80", opacity=0.65,
        )
    plotter.add_mesh(pv.Sphere(center=data["goal"], radius=0.07), color=GOAL)


def apply_global_camera(plotter: pv.Plotter, camera: dict) -> None:
    plotter.disable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["up"]
    plotter.camera.view_angle = float(camera["view_angle_deg"])


def apply_ego_camera(plotter: pv.Plotter, positions: np.ndarray, index: int) -> None:
    current = positions[index]
    begin = max(0, index - 8)
    displacement = current - positions[begin]
    if np.linalg.norm(displacement) < 1.0e-4:
        displacement = positions[-1] - positions[0]
    forward = displacement / max(np.linalg.norm(displacement), 1.0e-9)
    plotter.disable_parallel_projection()
    plotter.camera.position = current - 0.55 * forward + np.array([0, 0, 0.22])
    plotter.camera.focal_point = current + 1.6 * forward
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.view_angle = 78.0


def render_panel(data: dict, index: int, camera: dict, *, ego: bool) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=(PANEL, PANEL))
    plotter.set_background("white")
    add_static(plotter, data)
    measured = data["measured"][: index + 1]
    commanded = data["commanded"][: index + 1]
    if len(commanded) >= 2:
        plotter.add_mesh(polyline(commanded).tube(radius=0.007), color=REFERENCE)
    if len(measured) >= 2:
        plotter.add_mesh(polyline(measured).tube(radius=0.011), color=MEASURED)
    plotter.add_mesh(pv.Sphere(center=measured[-1], radius=0.035), color=MEASURED)
    if ego:
        apply_ego_camera(plotter, data["measured"], index)
    else:
        apply_global_camera(plotter, camera)
    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if "0806_flight_demonstration_suite" in args.output.resolve().parts:
        raise ValueError("real-flight output must not be written into frozen inputs")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = load_log(args.log)
    camera = camera_from_json(args.camera)
    indices = list(range(0, len(data["measured"]), max(args.stride, 1)))
    if indices[-1] != len(data["measured"]) - 1:
        indices.append(len(data["measured"]) - 1)
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{2 * PANEL}x{PANEL}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    for index in indices:
        left = render_panel(data, index, camera, ego=False)
        right = render_panel(data, index, camera, ego=True)
        encoder.stdin.write(np.concatenate([left, right], axis=1).astype(np.uint8).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg encode failed")

    manifest = {
        "status": "P0806_REAL_LOG_VIDEO_COMPLETE",
        "flight_log": str(args.log.resolve()),
        "flight_log_sha256": sha256_file(args.log),
        "camera_calibration": str(args.camera.resolve()),
        "camera_calibration_sha256": sha256_file(args.camera),
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "frames": len(indices),
        "fps": FPS,
        "resolution": [2 * PANEL, PANEL],
        "panels": {"left": "calibrated global", "right": "ego-centric"},
    }
    Path(str(args.output) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
