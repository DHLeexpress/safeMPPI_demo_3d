from __future__ import annotations

from pathlib import Path

import numpy as np

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.geometry import build_nominal_polytope, hp_values, triangular_geometry
from safe_mppi.expansion import run_safe_expansion


ROOT = Path(__file__).resolve().parents[1]


def test_default_contract():
    cfg = load_config(ROOT / "default_config.json")
    assert cfg.taskspace.size == (5.0, 5.0, 3.0)
    assert cfg.taskspace.start == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    assert cfg.taskspace.goal == (5.0, 5.0, 2.0)
    assert cfg.safemppi.sampling_mode == "mode1_centroid_anisotropic"
    assert cfg.safemppi.horizon == 10
    assert cfg.safemppi.num_samples == 512
    assert cfg.safemppi.obstacle_margin == cfg.safemppi.safety_margin == 0.0


def test_default_config_keeps_new_fields_inactive():
    cfg = load_config(ROOT / "default_config.json")
    assert cfg.safemppi.z_bias_weight == 0.0
    assert cfg.safemppi.warm_start_bias == (0.0, 0.0, 0.0)


def test_ball_below_contract():
    cfg = load_config(ROOT / "ball_below_config.json")
    assert cfg.taskspace.start[:3] == (0.0, 0.0, 2.0)
    assert cfg.taskspace.goal == (3.0, 0.0, 2.0)
    assert cfg.obstacles.spheres == ((1.5, 0.0, 2.0, 0.254),)
    assert cfg.safemppi.demo_u_max == 1.0
    assert cfg.safemppi.warm_start_bias == (0.1, 0.0, 0.0)
    assert cfg.safemppi.z_bias_weight > 0.0 and cfg.safemppi.z_bias_temperature <= 0.05
    assert cfg.safemppi.soft_clearance_weight == 0.0 and cfg.safemppi.progress_weight == 0.0
    assert cfg.data.gammas == (0.1, 0.3, 0.5, 1.0)


def test_uniform_triangular_base():
    vertices, faces, normals, offsets = triangular_geometry()
    assert vertices.shape == (42, 3)
    assert faces.shape == (80, 3)
    assert normals.shape == (80, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    for face, normal, offset in zip(faces, normals, offsets):
        np.testing.assert_allclose(vertices[face] @ normal, offset, atol=1e-12)


def test_zero_margin_obstacle_tangency():
    cfg = load_config(ROOT / "default_config.json")
    env = TaskEnvironment(cfg)
    center = np.array([1.0, 1.0, 1.5])
    sphere = np.array([[2.0, 1.0, 1.5, 0.25]])
    cylinder = np.array([[1.0, 2.0, 0.20]])
    poly = build_nominal_polytope(center, sphere, cylinder, env.bounds,
                                  sensing_range=2.0, obstacle_margin=0.0)
    sphere_tangent = np.array([[1.75, 1.0, 1.5]])
    cylinder_tangent = np.array([[1.0, 1.8, 1.5]])
    assert abs(hp_values(poly, sphere_tangent)[0]) < 1e-12
    assert abs(hp_values(poly, cylinder_tangent)[0]) < 1e-12


def test_expansion_is_an_explicit_placeholder():
    try:
        run_safe_expansion("dataset", "output")
    except NotImplementedError as error:
        assert "next stage" in str(error)
    else:
        raise AssertionError("safe expansion must remain an explicit placeholder")
