#!/usr/bin/env python3
"""Interactively calibrate one shared global camera against both 0806 scenes.

The approved simulator MP4s remain frozen.  This tool writes a run-local
camera transform used only when rendering measured flight logs beside them.
Press ``s`` to save, ``r`` to reset to the frozen simulation camera, and ``q``
to close the window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def load_scene(path: Path) -> dict:
    return json.loads(path.read_text())


def bounds_from_config(config: dict) -> np.ndarray:
    origin = np.asarray(config["taskspace"]["origin"], float)
    size = np.asarray(config["taskspace"]["size"], float)
    return np.column_stack([origin, origin + size])


def frozen_camera(bounds: np.ndarray) -> dict:
    center = bounds.mean(axis=1)
    extent = bounds[:, 1] - bounds[:, 0]
    elevation, azimuth = np.deg2rad(25.0), np.deg2rad(-57.0)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    return {
        "position": (center + 2.05 * float(np.max(extent)) * direction).tolist(),
        "focal_point": center.tolist(),
        "up": [0.0, 0.0, 1.0],
        "view_angle_deg": 30.0,
    }


def apply_camera(plotter: pv.Plotter, camera: dict) -> None:
    plotter.disable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["up"]
    plotter.camera.view_angle = float(camera["view_angle_deg"])


def add_scene(plotter: pv.Plotter, config: dict, label: str) -> None:
    bounds = bounds_from_config(config)
    (x0, x1), (y0, y1), (z0, z1) = bounds
    box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))
    plotter.add_mesh(box.extract_all_edges(), color="#9aa2a8", opacity=0.55)
    height = z1 - z0
    z_center = 0.5 * (z0 + z1)
    for x, y, modeled_radius in config["obstacles"]["cylinders"]:
        physical = pv.Cylinder(
            center=(x, y, z_center),
            direction=(0.0, 0.0, 1.0),
            radius=0.10,
            height=height,
            resolution=48,
        )
        inflated = pv.Cylinder(
            center=(x, y, z_center),
            direction=(0.0, 0.0, 1.0),
            radius=float(modeled_radius),
            height=height,
            resolution=48,
        )
        plotter.add_mesh(physical, color="#747b80", opacity=1.0)
        plotter.add_mesh(inflated, color="#2b8cbe", opacity=0.13)
    start = np.asarray(config["taskspace"]["start"][:3], float)
    goal = np.asarray(config["taskspace"]["goal"], float)
    plotter.add_mesh(pv.Cube(center=start, x_length=0.07, y_length=0.07, z_length=0.07), color="#111111")
    plotter.add_mesh(pv.Sphere(radius=0.07, center=goal), color="#F0B400")
    plotter.add_text(label, position="upper_left", font_size=12, color="#111111")
    plotter.show_axes()


def camera_payload(plotter: pv.Plotter, source_camera: dict) -> dict:
    camera = plotter.camera
    return {
        "status": "P0806_CAMERA_CALIBRATION_COMPLETE",
        "semantics": (
            "virtual global camera for measured-flight visualization; does "
            "not alter the frozen simulation videos"
        ),
        "position": list(map(float, camera.position)),
        "focal_point": list(map(float, camera.focal_point)),
        "up": list(map(float, camera.up)),
        "view_angle_deg": float(camera.view_angle),
        "clipping_range": list(map(float, camera.clipping_range)),
        "frozen_simulation_camera": source_camera,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-config", type=Path, required=True)
    parser.add_argument("--inner-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="save the default frozen camera without opening a window",
    )
    args = parser.parse_args()

    immutable_marker = "0806_flight_demonstration_suite"
    if immutable_marker in args.output.resolve().parts:
        raise ValueError("camera calibration must be written to an operator run")
    args.output.mkdir(parents=True, exist_ok=True)

    outer = load_scene(args.outer_config)
    inner = load_scene(args.inner_config)
    bounds = bounds_from_config(outer)
    source_camera = frozen_camera(bounds)
    plotter = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1440, 720))
    plotter.set_background("white")
    plotter.subplot(0, 0)
    add_scene(plotter, outer, "symmetric_scene_outer")
    apply_camera(plotter, source_camera)
    plotter.subplot(0, 1)
    add_scene(plotter, inner, "symmetric_scene_inner")
    apply_camera(plotter, source_camera)
    plotter.link_views()

    def save() -> None:
        plotter.subplot(0, 0)
        payload = camera_payload(plotter, source_camera)
        (args.output / "camera_calibration.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        plotter.screenshot(args.output / "camera_calibration.png")
        print(f"saved camera calibration to {args.output}")

    def reset() -> None:
        for column in (0, 1):
            plotter.subplot(0, column)
            apply_camera(plotter, source_camera)
        plotter.render()

    if args.off_screen:
        save()
        plotter.close()
        return 0
    plotter.add_key_event("s", save)
    plotter.add_key_event("r", reset)
    plotter.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
