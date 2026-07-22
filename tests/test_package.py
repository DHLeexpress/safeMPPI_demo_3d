from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.geometry import build_nominal_polytope, hp_values, triangular_geometry
from safe_mppi.expansion import (
    ExpansionConfig, RBFPosterior, Verification, calibrate_fixed_beta,
    mean_pairwise_lengthscale, run_safe_expansion,
)
from safe_mppi.flow_model import ConditionalFlowMLP


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
    assert cfg.safemppi.initial_control == (0.0, 0.0, 0.0)


def test_ball_below_contract():
    cfg = load_config(ROOT / "ball_below_config.json")
    assert cfg.taskspace.start[:3] == (0.0, 0.0, 2.0)
    assert cfg.taskspace.goal == (3.0, 0.0, 2.0)
    assert cfg.obstacles.spheres == ((1.5, 0.0, 2.0, 0.254),)
    assert cfg.safemppi.demo_u_max == 1.0
    assert cfg.safemppi.initial_control == (0.1, 0.0, 0.0)
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


def test_ball_demo_contract():
    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    assert cfg.taskspace.start[:3] == (0.0, 0.0, 2.0)
    assert cfg.taskspace.goal == (3.0, 0.0, 2.0)
    assert cfg.obstacles.spheres == ((1.5, 0.0, 2.0, 0.254),)
    assert cfg.data.gammas == (0.1, 0.3, 0.5, 1.0)
    assert cfg.safemppi.demo_u_max == 1.0
    assert cfg.safemppi.initial_control == (0.1, 0.0, 0.0)
    assert cfg.safemppi.soft_clearance_weight == cfg.safemppi.progress_weight == 0.0


def test_rbf_uncertainty_and_fixed_beta():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ell = mean_pairwise_lengthscale(features)
    gp = RBFPosterior(ell, 1.0e-2)
    prior = gp.sigma(features)
    gp.set_buffer(features[:1])
    posterior = gp.sigma(features)
    assert posterior[0] < prior[0]
    scores = [torch.tensor([0.0, 0.2, 0.8, 1.0])]
    beta = calibrate_fixed_beta(scores, 0.5)
    assert 0.0 < beta < 10.0


class _OneStepTask:
    def reset(self, gamma, episode, seed):
        return 0

    def context(self, state, gamma):
        return torch.tensor([gamma], dtype=torch.float32)

    def verify(self, context, candidates, gamma):
        return [Verification(True, True, 1.0, float(row.square().sum()))
                for row in candidates]

    def advance(self, state, candidate):
        return state + 1

    def terminal(self, state):
        return "SUCCESS" if state else None


def test_standalone_expansion_smoke(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=2, batch_size=2, gp_buffer_cap=8,
    )
    result = run_safe_expansion(
        policy, _OneStepTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    assert result["D"] == 2
    assert result["D_plus"] == 2
    assert result["rounds"][0]["round"] == 1
    assert result["rounds"][0]["success"] == 1
    assert (tmp_path / "checkpoint_000.pt").is_file()
    assert (tmp_path / "checkpoint_001.pt").is_file()
