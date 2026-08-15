from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "paper_ready/0814/source"
RUNTIME = Path(__file__).resolve().parents[1] / "paper_ready/0814/runtime_snapshot"
sys.path[:0] = [str(SOURCE), str(RUNTIME)]

from real_bowling_scene import (  # noqa: E402
    hard_path_diagnostics,
    load_as_built_geometry,
    string_clearance_m,
)


SCENE = Path(__file__).resolve().parents[1] / (
    "paper_ready/0814/claude/runs/20260814_flight_references/scene/"
    "bowling_scene.json"
)


def test_effective_radius_is_measured_plus_point_16() -> None:
    geometry = load_as_built_geometry(SCENE)
    np.testing.assert_allclose(
        geometry["effective_spheres"][:, 3]
        - geometry["physical_spheres"][:, 3],
        0.16,
        rtol=0.0,
        atol=2.0e-7,
    )


def test_string_gate_begins_at_physical_ball_top() -> None:
    geometry = load_as_built_geometry(SCENE)
    ball = geometry["physical_spheres"][0]
    below = np.asarray([[ball[0], ball[1], ball[2] + ball[3] - 1.0e-3]])
    above_center = np.asarray([[ball[0], ball[1], ball[2] + ball[3] + 1.0e-3]])
    above_safe = above_center.copy()
    above_safe[0, 0] += 0.101
    one_ball = geometry["physical_spheres"][:1]
    assert np.isinf(string_clearance_m(below, one_ball)[0])
    assert string_clearance_m(above_center, one_ball)[0] < 0.0
    assert string_clearance_m(above_safe, one_ball)[0] > 0.0


def test_hard_path_requires_both_effective_shell_and_string_clearance() -> None:
    geometry = load_as_built_geometry(SCENE)
    safe = np.asarray([[-2.1, 1.5, 0.9], [0.7, -1.5, 0.9]])
    diagnostic = hard_path_diagnostics(
        safe,
        geometry["effective_spheres"],
        geometry["physical_spheres"],
    )
    assert diagnostic["effective_sphere_valid"]
    assert diagnostic["string_valid"]
    assert diagnostic["hard_valid"]
