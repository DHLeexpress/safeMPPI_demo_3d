from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.geometry import build_nominal_polytope, hp_values, triangular_geometry
from safe_mppi.expansion import (
    ExpansionConfig, QueryRecord, RBFPosterior, Verification,
    _OrderedVerifier,
    _cluster_balanced_replay, _context_kcenter_replay,
    _frozen_phi_farthest_first, _sliding_success_gp_rows, _softmin_choice,
    _top_uncertainty_by_round, calibrate_fixed_beta, mean_pairwise_lengthscale,
    perturb_plan_candidates, run_safe_expansion,
)
from safe_mppi.flow_model import ConditionalFlowMLP
from safe_mppi.verifier_polytope import (
    certify_single_sphere_affine,
    certify_window,
    fit_variable_face,
    fit_variable_face_cvxpy,
    fit_verifier_polytope,
)


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


def test_ball_flow_context_verifier_and_route_modes():
    from safe_mppi.ball_flow_task import (BallFlowTask, build_context, context_state,
                                          inside_expansion_corridor, native_cost, plan_states,
                                          raw_rollout, raw_trajectory_validity, route_mode)
    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    task = BallFlowTask(cfg)
    env = task.env
    state = task.reset(0.5, 0, 0)
    context = task.context(state, 0.5)
    assert context.shape == (10,)
    assert float(context[9]) == 0.5
    recovered = context_state(env, context.numpy())
    np.testing.assert_allclose(recovered, env.start, atol=1e-5)

    near = np.array([1.0, 0.0, 2.0, 0.6, 0.0, 0.0], np.float32)
    ctx = torch.from_numpy(build_context(env, near, 0.5))
    through = torch.zeros(1, 10, 3)
    through[0, :, 0] = 1.0
    verdict = task.verify(ctx, through, 0.5)[0]
    assert not verdict.valid and verdict.margin < 0.0
    gentle = torch.zeros(1, 10, 3)
    gentle[0, :, 0] = 0.1
    start_ctx = task.context(task.reset(0.3, 0, 0), 0.3)
    assert task.verify(start_ctx, gentle, 0.3)[0].valid
    assert float(env.bounds[0, 1]) == 3.5
    beyond_x_bound = np.array([3.45, 0.0, 2.0, 0.8, 0.0, 0.0], np.float32)
    beyond_ctx = torch.from_numpy(build_context(env, beyond_x_bound, 0.3))
    assert not task.verify(beyond_ctx, through, 0.3)[0].valid
    np.testing.assert_array_equal(
        inside_expansion_corridor(np.array([
            [0.0, 10.0, 1.5], [3.0, -10.0, 2.5],
            [-0.01, 0.0, 2.0], [1.5, 0.0, 2.51],
        ])),
        np.array([True, True, False, False]),
    )
    upward = torch.zeros(1, 10, 3)
    upward[0, :, 0] = 0.1
    upward[0, :, 2] = 1.2
    assert task.verify(start_ctx, upward, 0.3)[0].valid
    assert BallFlowTask(cfg, tight_corridor=True).verify(
        start_ctx, upward, 0.3
    )[0].valid

    # The strict corridor constrains only the segment that receding-horizon control will
    # execute. A corridor-violating unexecuted tail must not cause NVP.
    tail_only_state = np.array([2.95, 0.0, 2.0, 0.2, 0.0, 0.0], np.float32)
    tail_only_ctx = torch.from_numpy(build_context(env, tail_only_state, 0.3))
    tail_accel = torch.zeros(1, 10, 3)
    tail_accel[0, :, 0] = 1.0
    tail_states = plan_states(env, tail_only_state, tail_accel[0].numpy())
    assert env.inside_taskspace(tail_states[1:2, :3]).all()
    assert not env.inside_taskspace(tail_states[:, :3]).all()
    assert BallFlowTask(cfg).verify(tail_only_ctx, tail_accel, 0.3)[0].valid
    assert BallFlowTask(cfg, tight_corridor=True).verify(
        tail_only_ctx, tail_accel, 0.3
    )[0].valid

    # A corridor exit on the next executed segment remains fail-closed.
    next_step_state = np.array([2.99, 0.0, 2.0, 0.2, 0.0, 0.0], np.float32)
    next_step_ctx = torch.from_numpy(build_context(env, next_step_state, 0.3))
    coast = torch.zeros(1, 10, 3)
    assert BallFlowTask(cfg).verify(next_step_ctx, coast, 0.3)[0].valid
    assert not BallFlowTask(cfg, tight_corridor=True).verify(
        next_step_ctx, coast, 0.3
    )[0].valid

    class UpwardPolicy:
        def sample(self, context, count, generator):
            plans = torch.zeros(count, 10, 3)
            plans[:, :, 0] = 0.1
            plans[:, :, 2] = 1.2
            return plans

    corridor_rollout = raw_rollout(
        UpwardPolicy(), cfg, 0.3, seed=0, max_steps=20, tight_corridor=True,
    )
    assert corridor_rollout["status"] == "CORRIDOR_VIOLATION"
    assert corridor_rollout["corridor_violation"]
    assert not corridor_rollout["physical_collision"]

    def crossing(second):
        return np.array([[1.4, 0.0, 2.0], [1.4, 0.0, 2.0], second], float)

    assert route_mode(env, np.array([[1.4, 0.0, 1.9], [1.6, 0.0, 1.5]])) == "below"
    assert route_mode(env, np.array([[1.4, 0.0, 2.1], [1.6, 0.0, 2.5]])) == "above"
    assert route_mode(env, np.array([[1.4, 0.0, 2.0], [1.6, 0.8, 2.0]])) == "left"
    assert route_mode(env, np.array([[1.4, 0.0, 2.0], [1.6, -0.8, 2.0]])) == "right"
    assert route_mode(env, np.array([[0.2, 0.0, 2.0], [0.9, 0.0, 2.0]])) == "none"

    # The demo-only exponential z bias is nonzero in this config, but expansion ranking must
    # equal the four native terms exactly and therefore remain independent of that bias.
    assert cfg.safemppi.z_bias_weight > 0.0
    plan = np.zeros((10, 3), np.float32)
    states = plan_states(env, env.start, plan)
    expected = sum(
        cfg.safemppi.running_goal_weight
        * float(((states[h + 1, :3] - env.goal) ** 2).sum())
        for h in range(10)
    )
    expected += cfg.safemppi.terminal_goal_weight * float(
        ((states[-1, :3] - env.goal) ** 2).sum())
    assert native_cost(env, states, plan) == expected
    reverse_increment = cfg.safemppi.z_bias_weight * float(np.exp(np.minimum(
        (cfg.safemppi.z_bias_plane - states[1:, 2])
        / cfg.safemppi.z_bias_temperature,
        20.0,
    )).sum())
    assert np.isclose(
        native_cost(env, states, plan, "favor_above"),
        expected + reverse_increment,
    )
    with pytest.raises(ValueError):
        native_cost(env, states, plan, "unknown")

    biased_task = BallFlowTask(cfg, execution_z_bias_mode="favor_above")
    plain_verdict = task.verify(start_ctx, gentle, 0.3)[0]
    biased_verdict = biased_task.verify(start_ctx, gentle, 0.3)[0]
    assert plain_verdict.valid == biased_verdict.valid
    assert plain_verdict.hp_eligible == biased_verdict.hp_eligible
    assert plain_verdict.margin == biased_verdict.margin
    assert biased_verdict.execution_cost > plain_verdict.execution_cost
    with pytest.raises(ValueError):
        BallFlowTask(cfg, execution_z_bias_mode="unknown")

    controls = np.zeros((12, 3), np.float32)
    controls[:, 0] = 0.1
    executed = plan_states(env, env.start, controls)
    assert raw_trajectory_validity(cfg, executed, controls, 1.0)
    invalid = executed.copy()
    invalid[5, :3] = np.asarray(cfg.obstacles.spheres[0][:3], np.float32)
    assert not raw_trajectory_validity(cfg, invalid, controls, 1.0)


def test_variable_face_matches_cloned_2d_max_margin_problem():
    trajectory = np.array([
        [0.0, 0.0],
        [0.16, -0.03],
        [0.34, -0.06],
        [0.53, -0.04],
    ])
    center = np.array([0.82, 0.42])
    radius = 0.18
    gamma = 0.3
    certificate = fit_variable_face(trajectory, center, radius, gamma)
    assert certificate.feasible

    # Dense angular reference for the exact cloned 2-D formulation.
    theta = np.linspace(-np.pi, np.pi, 200_000, endpoint=False)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    centered = trajectory - trajectory[0]
    d = center - trajectory[0]
    beta = 1.0 - (1.0 - gamma) ** np.arange(len(trajectory))
    margins = normals @ d - radius
    feasible = margins >= 1.0e-4
    feasible &= np.all(
        centered[1:] @ normals.T <= beta[1:, None] * margins[None] + 1.0e-10,
        axis=0,
    )
    assert feasible.any()
    reference_margin = float(margins[feasible].max())
    assert abs(certificate.margin - reference_margin) < 5.0e-5

    lifted = fit_variable_face(
        np.column_stack([trajectory, np.zeros(len(trajectory))]),
        np.r_[center, 0.0], radius, gamma,
    )
    assert lifted.feasible
    assert abs(lifted.margin - certificate.margin) < 1.0e-8


def test_variable_face_radial_shortcut_checks_normalized_slack():
    trajectory = np.array([
        [0.0, 0.0],
        [0.000100005, 0.001],
        [0.000100005, -0.001],
    ])
    certificate = fit_variable_face(
        trajectory,
        obstacle_center=np.array([1.0001, 0.0]),
        radius=1.0,
        gamma=1.0,
    )
    assert not certificate.feasible


def test_cvxpy_face_solver_matches_analytic_solver():
    pytest.importorskip("cvxpy")
    trajectories = [
        np.array([
            [0.0, 0.0, 0.0],
            [0.04, 0.00, 0.01],
            [0.09, 0.01, 0.03],
            [0.15, 0.03, 0.06],
        ]),
        np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.0],
            [0.4, -0.3, 0.2],
            [0.6, 0.4, -0.2],
        ]),
    ]
    for trajectory in trajectories:
        analytic = fit_variable_face(
            trajectory, np.array([1.0, 0.0, 0.0]), 0.25, 0.3,
        )
        cvxpy = fit_variable_face_cvxpy(
            trajectory, np.array([1.0, 0.0, 0.0]), 0.25, 0.3,
        )
        assert cvxpy.feasible == analytic.feasible
        if analytic.feasible:
            assert cvxpy.margin == pytest.approx(
                analytic.margin, abs=2.0e-5,
            )
            assert cvxpy.worst_slack >= -1.0e-6


def test_cvxpy_face_solver_uses_legacy_boundary_tolerance():
    pytest.importorskip("cvxpy")
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [0.7500000375, 0.0, 0.0],
    ])
    kwargs = {
        "obstacle_center": np.array([1.0, 0.0, 0.0]),
        "radius": 0.25,
        "gamma": 1.0,
    }
    analytic = fit_variable_face(trajectory, **kwargs)
    cvxpy = fit_variable_face_cvxpy(trajectory, **kwargs)
    assert not analytic.feasible
    assert cvxpy.feasible == analytic.feasible


def test_cvxpy_strict_path_solves_nonradial_face():
    pytest.importorskip("cvxpy")
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [0.2377367940125078, 0.0543451719190362, 0.0280847177715644],
        [0.3596585436236606, 0.1417806021757651, 0.0442746779532262],
        [0.4172295916608439, 0.1851917073360822, -0.1203400271922104],
        [0.3842839990631271, 0.3485043442226631, -0.1753073868325064],
        [0.5072413276495846, 0.5350556229566190, -0.0288737448795695],
    ])
    kwargs = {
        "obstacle_center": np.array([1.0, 0.0, 0.0]),
        "radius": 0.25,
        "gamma": 0.3,
    }
    analytic = fit_variable_face(trajectory, **kwargs)
    cvxpy = fit_variable_face_cvxpy(
        trajectory, fallback_to_analytic=False, **kwargs,
    )
    assert analytic.feasible and cvxpy.feasible
    assert abs(float(cvxpy.normal @ np.array([1.0, 0.0, 0.0]))) < 0.999
    assert cvxpy.margin == pytest.approx(analytic.margin, abs=2.0e-5)
    assert cvxpy.worst_slack >= -1.0e-8


def test_3d_verifier_restores_eighty_artificial_bounding_faces():
    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    env = TaskEnvironment(cfg)
    positions = np.repeat(env.start[None, :3], 11, axis=0)
    polytope = fit_verifier_polytope(
        positions, env.spheres, env.cylinders, 0.3,
        cfg.safemppi.sensing_range,
    )
    assert polytope.feasible
    assert polytope.kinds.count("artificial") == 80
    assert polytope.kinds.count("real_sphere") == 1
    assert polytope.A.shape == (81, 3)
    assert np.all(polytope.margins > 0.0)


def test_single_sphere_affine_matches_real_face_but_not_bounded_polytope():
    sphere = np.array([[10.0, 0.0, 0.0, 0.25]])
    no_cylinders = np.empty((0, 3))
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, -0.5],
        [0.0, 0.5, 0.5],
        [0.0, -0.5, 0.5],
        [0.0, -0.5, -0.5],
        [0.0, 0.5, -0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    real_face = fit_variable_face(
        positions, sphere[0, :3], sphere[0, 3], 0.1,
    )
    affine = certify_single_sphere_affine(
        positions, sphere, no_cylinders, 0.1, 2.0,
    )
    bounded = certify_window(
        positions, sphere, no_cylinders, 0.1, 2.0,
    )
    assert affine[0] == real_face.feasible
    assert affine[1] == pytest.approx(real_face.worst_slack)
    assert affine[0]
    assert not bounded[0]


def test_ordered_verifier_process_pool_matches_serial_ball_task():
    from safe_mppi.ball_flow_task import BallFlowTask

    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    task = BallFlowTask(cfg, tight_corridor=True)
    blocks = []
    for gamma in (0.1, 0.5):
        context = task.context(task.reset(gamma, 0, 0), gamma)
        plans = torch.zeros(3, 10, 3)
        plans[:, :, 0] = torch.tensor([0.1, 0.2, 0.3])[:, None]
        blocks.append((context, plans, gamma))
    with _OrderedVerifier(task, 1) as serial:
        expected = serial.verify_many(blocks)
    with _OrderedVerifier(task, 2) as parallel:
        actual = parallel.verify_many(blocks)
    assert actual == expected


def test_goal_overshoot_tail_is_eligible_when_every_step_moves_forward():
    from safe_mppi.ball_flow_task import BallFlowTask, build_context

    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    task = BallFlowTask(cfg, tight_corridor=True)
    state = np.array([2.8, 0.0, 2.0, 0.5, 0.0, 0.0], np.float32)
    context = torch.from_numpy(build_context(task.env, state, 0.3))
    coast = torch.zeros(1, 10, 3)
    verdict = task.verify(context, coast, 0.3)[0]
    assert verdict.valid
    assert verdict.hp_eligible
    assert verdict.progress > 0.0
    assert verdict.progress_eligible


def test_plan_that_reverses_x_is_execution_ineligible_but_safety_positive():
    from safe_mppi.ball_flow_task import BallFlowTask, build_context

    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    task = BallFlowTask(cfg, tight_corridor=True)
    state = np.array([2.6, 0.0, 2.4, 0.3, 0.0, 0.0], np.float32)
    context = torch.from_numpy(build_context(task.env, state, 0.3))
    reversing = torch.zeros(1, 10, 3)
    reversing[0, :, 0] = -0.4
    verdict = task.verify(context, reversing, 0.3)[0]
    assert verdict.valid
    assert verdict.progress < 0.0
    assert not verdict.progress_eligible


def test_ball_verifier_accepts_available_terminal_horizons():
    from safe_mppi.ball_flow_task import BallFlowTask

    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    task = BallFlowTask(cfg, tight_corridor=True)
    context = task.context(task.reset(0.3, 0, 0), 0.3)
    for horizon in (1, 4, 10):
        plan = torch.zeros(1, horizon, 3)
        plan[0, :, 0] = 0.2
        verdict = task.verify(context, plan, 0.3)[0]
        assert not verdict.error
        assert verdict.step_margin is not None


def test_above_wedge_geometry_and_next_state_only_eligibility():
    from safe_mppi.ball_flow_task import (
        BallFlowTask, inside_above_halfspace, inside_above_wedge, plan_states,
    )

    np.testing.assert_array_equal(
        inside_above_wedge(np.array([
            [0.0, 0.0, 2.0],
            [1.0, 0.10, 2.10],
            [1.0, -0.10, 2.10],
            [1.0, 0.11, 2.10],
            [1.0, 0.0, 1.99],
        ])),
        np.array([True, True, True, False, False]),
    )
    np.testing.assert_array_equal(
        inside_above_halfspace(np.array([
            [1.0, 10.0, 2.0],
            [1.0, -10.0, 2.1],
            [1.0, 0.0, 1.99],
        ])),
        np.array([True, True, False]),
    )

    cfg = load_config(ROOT / "configs" / "ball_biased_demo.json")
    task = BallFlowTask(cfg)
    assert task.target_region == "above_wedge"
    context = task.context(task.reset(0.3, 0, 0), 0.3)

    above = torch.zeros(1, 10, 3)
    above[0, :, 0] = 0.1
    above[0, 0, 2] = 1.0
    assert bool(task.above_wedge_eligibility(context, above)[0])

    below = above.clone()
    below[0, 0, 2] = -0.1
    assert not bool(task.above_wedge_eligibility(context, below)[0])
    # The wedge is a separate target signal, not a replacement safety label.
    assert task.verify(context, below, 0.3)[0].valid

    tail_exit = above.clone()
    tail_exit[0, 1:, 2] = -1.0
    states = plan_states(task.env, task.env.start, tail_exit[0].numpy())
    assert inside_above_wedge(states[1:2, :3]).all()
    assert not inside_above_wedge(states[-1:, :3]).all()
    assert bool(task.above_wedge_eligibility(context, tail_exit)[0])

    # A q1 with z > 2 but |y| > z-2 separates the halfspace from the wedge.
    lateral = torch.zeros(1, 10, 3)
    lateral[0, :, 0] = 0.1
    lateral[0, 0, 1] = 1.0
    lateral[0, 0, 2] = 0.1
    halfspace_task = BallFlowTask(cfg, target_region="above_halfspace")
    assert not bool(task.target_region_eligibility(context, lateral)[0])
    assert bool(halfspace_task.target_region_eligibility(context, lateral)[0])
    assert not task.verify(context, lateral, 0.3)[0].target_eligible
    assert halfspace_task.verify(context, lateral, 0.3)[0].target_eligible

    with pytest.raises(ValueError, match="unknown target_region"):
        BallFlowTask(cfg, target_region="unknown")


def test_flow_embed_noised_base_changes_features():
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(10, (10, 3), hidden=32, representation_dim=16)
    context = torch.randn(4, 10)
    plans = torch.randn(4, 10, 3)
    plain = policy.embed(context, plans)
    noised = policy.embed(context, plans, base=torch.randn(4, 30))
    assert plain.shape == (4, 16)
    assert not torch.allclose(plain, noised)


def test_flow_shallow_trunk_matches_spec():
    shallow = ConditionalFlowMLP(10, (10, 3), hidden=48, representation_dim=32,
                                 trunk_depth=2, time_features="raw1")
    linear_layers = [m for m in shallow.trunk if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 2
    assert linear_layers[0].in_features == 41
    assert linear_layers[0].out_features == 48
    assert linear_layers[1].out_features == 32
    assert shallow.embed(torch.randn(3, 10), torch.randn(3, 10, 3)).shape == (3, 32)
    assert shallow(torch.randn(3, 30), torch.rand(3), torch.randn(3, 10)).shape == (3, 30)


def test_cfm_loss_mask_preserves_full_loss_and_masks_terminal_tail():
    policy = ConditionalFlowMLP(
        2, (3, 2), hidden=12, representation_dim=6,
    )
    context = torch.randn(2, 2)
    target = torch.randn(2, 3, 2)
    torch.manual_seed(7)
    legacy = policy.cfm_loss(context, target, reduction="none")
    torch.manual_seed(7)
    full_mask = policy.cfm_loss(
        context, target, reduction="none", loss_mask=torch.ones_like(target),
    )
    torch.testing.assert_close(legacy, full_mask)

    mask = torch.zeros_like(target)
    mask[:, :1] = 1.0
    policy.zero_grad()
    torch.manual_seed(11)
    policy.cfm_loss(
        context, target, reduction="mean", loss_mask=mask,
    ).backward()
    head_bias_gradient = policy.head.bias.grad.reshape(3, 2)
    assert torch.count_nonzero(head_bias_gradient[1:]) == 0
    assert torch.count_nonzero(head_bias_gradient[:1]) > 0


def test_candidate_perturbation_scope_and_clamping():
    policy = ConditionalFlowMLP(10, (10, 3), hidden=16, representation_dim=8,
                                control_limit=1.0)
    candidates = torch.zeros(7, 10, 3)
    perturbed = perturb_plan_candidates(
        policy, candidates, 5.0, torch.Generator().manual_seed(4))
    assert perturbed.shape == candidates.shape
    torch.testing.assert_close(perturbed[:, 1:], perturbed[:, :1].expand(-1, 9, -1))
    first = perturb_plan_candidates(
        policy, candidates, 5.0, torch.Generator().manual_seed(4), "first_action")
    torch.testing.assert_close(first[:, 1:], torch.zeros_like(first[:, 1:]))
    assert float(perturbed.abs().max()) <= 1.0


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


def test_rbf_acquisition_uses_standard_deviation_scale():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    gp = RBFPosterior(0.5, 1.0e-2)
    gp.set_buffer(features[:1])
    marginal = gp.sigma(features)
    selected, selected_sigma, _ = gp.acquire(
        features, 1, 0.2, torch.Generator().manual_seed(7)
    )
    assert selected_sigma[0] == pytest.approx(float(marginal[selected[0]]), abs=1.0e-6)


def test_top_uncertainty_includes_complete_boundary_tie():
    verification = Verification(True, True, 1.0, 0.0)
    rows = [
        QueryRecord(1, 0.1, 0, index, torch.zeros(1), torch.zeros(1),
                    verification, acquisition_sigma=1.0)
        for index in range(5)
    ]
    assert len(_top_uncertainty_by_round(rows, 0.2)) == 5
    rows[-1].acquisition_sigma = 0.5
    assert len(_top_uncertainty_by_round(rows, 0.2)) == 4


def test_top_uncertainty_can_preserve_each_round_gamma_group():
    verification = Verification(True, True, 1.0, 0.0)
    rows = [
        QueryRecord(
            1, gamma, 0, index, torch.zeros(1), torch.zeros(1),
            verification, acquisition_sigma=sigma,
        )
        for gamma, sigma in ((0.1, 1.0), (0.3, 0.1))
        for index in range(5)
    ]
    selected = _top_uncertainty_by_round(
        rows, 0.2, group_by_gamma=True,
    )
    assert len(selected) == 10
    assert {row.gamma for row in selected} == {0.1, 0.3}


def test_context_kcenter_retains_geometric_extreme_per_context():
    class EmbedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def embed(self, context, candidates):
            return candidates.reshape(len(candidates), -1)[:, :2]

    verification = Verification(True, True, 1.0, 0.0)
    candidates = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.95, 0.05]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([-1.0, 0.0]),
    )
    rows = [
        QueryRecord(1, 0.1, 0, 3, torch.zeros(1), candidate,
                    verification, acquisition_sigma=0.5)
        for candidate in candidates
    ]
    selected = _context_kcenter_replay(
        rows, EmbedPolicy(), quota=2, action_weight=0.5
    )
    assert len(selected) == 2
    assert any(torch.equal(row.candidate, candidates[-1]) for row in selected)
    assert len({id(row) for row in selected}) == len(selected)


def test_context_kcenter_keeps_small_groups_separate():
    class EmbedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def embed(self, context, candidates):
            return candidates.reshape(len(candidates), -1)

    verification = Verification(True, True, 1.0, 0.0)
    rows = [
        QueryRecord(1, 0.1, episode, episode, torch.zeros(1),
                    torch.tensor([float(episode), float(index)]), verification)
        for episode in (0, 1) for index in (0, 1)
    ]
    selected = _context_kcenter_replay(
        rows, EmbedPolicy(), quota=4, action_weight=0.0
    )
    assert len(selected) == 4
    assert {id(row) for row in selected} == {id(row) for row in rows}


def test_cluster_balanced_replay_retains_sparse_feature_cell():
    class EmbedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def embed(self, context, candidates):
            return candidates.reshape(len(candidates), -1)[:, :2]

    verification = Verification(True, True, 1.0, 0.0)
    rows = []
    for index in range(60):
        candidate = torch.tensor([1.0, 0.01 * (index % 3)])
        rows.append(QueryRecord(
            1, 0.1, index, index, torch.zeros(1), candidate, verification
        ))
    rare = [
        QueryRecord(1, 0.1, 100 + index, 100 + index, torch.zeros(1),
                    torch.tensor([-1.0, 0.1 * index]), verification)
        for index in range(2)
    ]
    rows.extend(rare)
    selected = _cluster_balanced_replay(
        rows, EmbedPolicy(), cluster_count=2, action_weight=0.5, budget=12,
        generator=torch.Generator().manual_seed(2),
    )
    assert len(selected) == 12
    assert any(id(row) in {id(item) for item in selected} for row in rare)
    assert len({id(row) for row in selected}) == len(selected)


def test_softmin_choice_hits_normalized_ess_target():
    _, ess = _softmin_choice(
        list(range(8)), 0.5, torch.Generator().manual_seed(3),
        torch.device("cpu"),
    )
    assert ess == pytest.approx(0.5, abs=1.0e-5)


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


class _StepMarginTask(_OneStepTask):
    def verify(self, context, candidates, gamma):
        return [
            Verification(
                True, True, margin=100.0 - index,
                execution_cost=float(index), progress_eligible=True,
                step_margin=float(index),
            )
            for index, _ in enumerate(candidates)
        ]


def test_max_step_margin_execution_uses_one_step_margin(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    events = []
    run_safe_expansion(
        policy, _StepMarginTask(), tmp_path,
        config=ExpansionConfig(
            rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
            K=4, B=4, batch_size=2, gp_buffer_cap=8,
            execution_rule="max_step_margin",
        ),
        calibration_features=torch.randn(12, 8),
        event_callback=events.append,
    )
    assert events[0]["chosen_local"] == 3


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


def test_round_callback_observes_resolved_row_after_step_events(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    events = []
    rows = []

    def resolved(row):
        assert events
        assert row["round"] == 1
        assert row["success"] == 1
        rows.append(row)

    run_safe_expansion(
        policy,
        _OneStepTask(),
        tmp_path,
        config=ExpansionConfig(
            rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
            K=4, B=2, batch_size=2, gp_buffer_cap=8,
        ),
        calibration_features=torch.randn(12, 8),
        event_callback=events.append,
        round_callback=resolved,
    )

    assert len(rows) == 1
    assert events[-1]["status"] == "SUCCESS"


def test_executed_only_archive_keeps_one_of_b_queries(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8, archive_rule="executed_only",
    )
    result = run_safe_expansion(
        policy, _OneStepTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    assert result["rounds"][0]["verifier_queries"] == 4
    assert result["D"] == 1
    assert result["D_plus"] == 1


class _SuccessfulTrajectoryCommitTask:
    horizon = 3

    def reset(self, gamma, episode, seed):
        return {"episode": int(episode), "step": 0}

    def context(self, state, gamma):
        return torch.tensor(
            [float(gamma), float(state["episode"]), float(state["step"])],
            dtype=torch.float32,
        )

    def verify(self, context, candidates, gamma):
        episode = int(round(float(context[1])))
        if episode == 2:
            return [
                Verification(False, False, -1.0, float(index))
                for index, _ in enumerate(candidates)
            ]
        return [
            Verification(
                True, True, 1.0, float(candidate.square().sum()),
                progress=1.0, progress_eligible=True,
            )
            for candidate in candidates
        ]

    def advance(self, state, candidate):
        return {"episode": state["episode"], "step": state["step"] + 1}

    def terminal(self, state):
        episode, step = state["episode"], state["step"]
        if episode == 0 and step >= 4:
            return "SUCCESS"
        if episode == 1 and step >= 5:
            return "SUCCESS"
        if episode == 3 and step >= 2:
            return "COLLISION"
        if episode == 4 and step >= 2:
            return "OOB"
        return None


def _successful_window_config(**updates):
    values = dict(
        rounds=1, gammas=(0.1, 0.3), parallel_episodes=3, max_steps=5,
        max_retry_batches=1,
        K=4, B=2, batch_size=2, gp_buffer_cap=8,
        replay_selector="uniform", archive_rule="successful_executed_windows",
        negative_alpha=0.0,
        gp_reference_mode="cumulative_accepted_frozen_phi",
    )
    values.update(updates)
    return ExpansionConfig(**values)


def test_successful_executed_windows_commit_one_trajectory_per_gamma(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    events = []
    result = run_safe_expansion(
        policy, _SuccessfulTrajectoryCommitTask(), tmp_path,
        config=_successful_window_config(gammas=(0.1,)),
        calibration_features=torch.randn(12, 8),
        event_callback=events.append,
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)
    evidence = torch.load(tmp_path / "gp_evidence.pt", weights_only=False)
    detail = result["rounds"][0]["successful_executed_commit_by_gamma"]

    # Gamma 0.1 has two successes, but fixed replica order authoritatively
    # selects episode 0. Gamma 0.3 has only collision/OOB/timeout trajectories.
    assert detail["0.1"] == {
        "selector": "lowest_episode_id",
        "requested_trajectory_count": 1,
        "retry_batches_used": 1,
        "attempted_episode_count": 3,
        "success_episode_count": 2,
        "success_episode_above_fractions": {"0": None, "1": None},
        "success_episode_mean_z": {"0": None, "1": None},
        "committed_trajectory_id": "r0001:g0.1:e000000",
        "committed_episode_id": 0,
        "successful_retry_batch": 0,
        "above_fraction": None,
        "mean_z": None,
        "executed_steps": 4,
        "candidate_window_count": 4,
        "full_h_valid_window_count": 4,
        "committed_window_count": 4,
        "committed_window_ids": [
            "r0001:g0.1:e000000:w000000",
            "r0001:g0.1:e000000:w000001",
            "r0001:g0.1:e000000:w000002",
            "r0001:g0.1:e000000:w000003",
        ],
        "committed_trajectory_count": 1,
        "committed_trajectory_ids": ["r0001:g0.1:e000000"],
        "committed_episode_ids": [0],
        "committed_trajectories": [{
            "rank": 0,
            "trajectory_id": "r0001:g0.1:e000000",
            "episode_id": 0,
            "successful_retry_batch": 0,
            "above_fraction": None,
            "mean_z": None,
            "executed_steps": 4,
            "candidate_window_count": 4,
            "full_h_valid_window_count": 4,
            "committed_window_count": 4,
            "committed_window_ids": [
                "r0001:g0.1:e000000:w000000",
                "r0001:g0.1:e000000:w000001",
                "r0001:g0.1:e000000:w000002",
                "r0001:g0.1:e000000:w000003",
            ],
        }],
        "all_committed_window_count": 4,
        "all_committed_window_ids": [
            "r0001:g0.1:e000000:w000000",
            "r0001:g0.1:e000000:w000001",
            "r0001:g0.1:e000000:w000002",
            "r0001:g0.1:e000000:w000003",
        ],
    }
    assert result["rounds"][0]["committed_trajectory_ids"] == [
        "r0001:g0.1:e000000",
    ]
    assert result["rounds"][0]["commit_reverify_queries"] == 4
    assert result["rounds"][0]["commit_reverify_valid"] == 4
    assert result["rounds"][0]["commit_reverify_progress"] == 4
    assert result["D"] == result["D_plus"] == result["D_replay_accepted"] == 4
    assert len(archive) == len(evidence) == 4
    assert {row.episode for row in archive} == {0}
    assert all(row.verification.valid and row.verification.progress_eligible
               for row in archive)
    assert not any(row.nvp_context for row in archive)

    executed = [
        event for event in events
        if event["episode"] == 0 and event["chosen_local"] is not None
    ]
    actions = [
        event["candidates"][
            event["selected"][event["chosen_local"]]
        ][0].detach().cpu()
        for event in executed
    ]
    expected = [
        torch.stack(actions[:3]),
        torch.stack(actions[1:4]),
        torch.cat([torch.stack(actions[2:4]), torch.zeros_like(actions[0])[None]]),
        torch.cat([
            torch.stack(actions[3:4]),
            torch.zeros_like(actions[0])[None].repeat(2, 1),
        ]),
    ]
    assert all(torch.equal(row.candidate, value)
               for row, value in zip(archive, expected))
    assert [row.valid_horizon for row in archive] == [3, 3, 2, 1]
    assert [row.loss_mask[:, 0].tolist() for row in archive] == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert all(torch.equal(row.context, executed[index]["context"].cpu())
               for index, row in enumerate(archive))


def test_successful_executed_windows_commit_requested_trajectories_per_gamma(
        tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    result = run_safe_expansion(
        policy, _SuccessfulTrajectoryCommitTask(), tmp_path,
        config=_successful_window_config(
            gammas=(0.1,),
            parallel_episodes=1,
            max_retry_batches=2,
            gp_reference_mode="recent_current_phi",
            successful_trajectories_per_gamma=2,
        ),
        calibration_features=torch.randn(12, 8),
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)
    detail = result["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"]

    assert detail["requested_trajectory_count"] == 2
    assert detail["retry_batches_used"] == 2
    assert detail["committed_trajectory_count"] == 2
    assert detail["committed_episode_ids"] == [0, 1]
    assert detail["committed_trajectory_ids"] == [
        "r0001:g0.1:e000000",
        "r0001:g0.1:e000001",
    ]
    assert [
        row["committed_window_count"]
        for row in detail["committed_trajectories"]
    ] == [4, 5]
    assert detail["all_committed_window_count"] == 9
    assert result["rounds"][0]["committed_trajectory_ids"] == [
        "r0001:g0.1:e000000",
        "r0001:g0.1:e000001",
    ]
    assert result["rounds"][0]["commit_reverify_queries"] == 9
    assert result["D"] == result["D_plus"] == 9
    assert len(archive) == 9
    assert {row.episode for row in archive} == {0, 1}
    assert len({row.trajectory_id for row in archive}) == 2


class _AboveFractionTrajectoryCommitTask(_SuccessfulTrajectoryCommitTask):
    def verify(self, context, candidates, gamma):
        return [
            Verification(
                True, True, 1.0, float(candidate.square().sum()),
                progress=1.0, progress_eligible=True,
            )
            for candidate in candidates
        ]

    def terminal(self, state):
        if state["episode"] == 0 and state["step"] >= 4:
            return "SUCCESS"
        if state["episode"] in {1, 2} and state["step"] >= 5:
            return "SUCCESS"
        return None

    def successful_trajectory_above_fraction(self, executed_states):
        episode = int(executed_states[0]["episode"])
        return 0.25 if episode == 0 else 0.75


def test_successful_executed_windows_max_above_fraction_tie_breaks_by_episode(
        tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    cfg = _successful_window_config(
        gammas=(0.1,), parallel_episodes=3, max_steps=5,
        gp_reference_mode="recent_current_phi",
        successful_trajectory_selector="max_above_fraction",
    )
    result = run_safe_expansion(
        policy, _AboveFractionTrajectoryCommitTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)
    detail = result["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"]
    assert detail["success_episode_count"] == 3
    assert detail["success_episode_above_fractions"] == {
        "0": 0.25, "1": 0.75, "2": 0.75,
    }
    assert detail["success_episode_mean_z"] == {
        "0": None, "1": None, "2": None,
    }
    assert detail["committed_episode_id"] == 1
    assert detail["committed_trajectory_id"] == "r0001:g0.1:e000001"
    assert detail["above_fraction"] == 0.75
    assert detail["candidate_window_count"] == 5
    assert detail["committed_window_count"] == 5
    assert {row.episode for row in archive} == {1}


class _MeanZTrajectoryCommitTask(_AboveFractionTrajectoryCommitTask):
    def successful_trajectory_mean_z(self, executed_states):
        episode = int(executed_states[0]["episode"])
        return 2.0 if episode == 0 else 2.4


def test_successful_executed_windows_max_mean_z_tie_breaks_by_episode(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    result = run_safe_expansion(
        policy, _MeanZTrajectoryCommitTask(), tmp_path,
        config=_successful_window_config(
            gammas=(0.1,), parallel_episodes=3, max_steps=5,
            gp_reference_mode="recent_current_phi",
            successful_trajectory_selector="max_mean_z",
        ),
        calibration_features=torch.randn(12, 8),
    )
    detail = result["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"]
    assert detail["success_episode_mean_z"] == {
        "0": 2.0, "1": 2.4, "2": 2.4,
    }
    assert detail["committed_episode_id"] == 1
    assert detail["mean_z"] == 2.4


class _AllSuccessfulCommitTask(_SuccessfulTrajectoryCommitTask):
    def terminal(self, state):
        return "SUCCESS" if state["step"] >= 4 else None

    def successful_trajectory_above_fraction(self, executed_states):
        episode = int(executed_states[0]["episode"])
        return float(episode % 8) / 7.0


def _exact_success_gp_config(**updates):
    values = dict(
        rounds=2, gammas=(0.1, 0.3), parallel_episodes=8,
        max_retry_batches=4, max_steps=4, K=64, B=8,
        batch_size=8, gp_buffer_cap=32, replay_selector="uniform",
        archive_rule="successful_executed_windows",
        successful_trajectory_selector="max_above_fraction",
        negative_alpha=0.0,
        gp_reference_mode="cumulative_success_per_gamma_frozen_phi_exact",
        gp_exact_max_rows_per_gamma=1024,
    )
    values.update(updates)
    return ExpansionConfig(**values)


def test_exact_success_gp_is_per_gamma_causal_and_unthinned(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    result = run_safe_expansion(
        policy, _AllSuccessfulCommitTask(), tmp_path,
        config=_exact_success_gp_config(),
        calibration_features=torch.randn(12, 8),
    )
    first, second = result["rounds"]
    evidence = torch.load(tmp_path / "gp_evidence.pt", weights_only=False)

    assert first["gp_buffer_by_gamma"] == {"0.1": 0, "0.3": 0}
    assert second["gp_buffer_by_gamma"] == {"0.1": 4, "0.3": 4}
    assert first["retry_batches_by_gamma"] == {"0.1": 1, "0.3": 1}
    assert first["attempted_episode_count"] == 16
    assert first["successful_executed_commit_by_gamma"]["0.1"][
        "committed_episode_id"
    ] == 7
    assert first["successful_executed_commit_by_gamma"]["0.3"][
        "committed_episode_id"
    ] == 15
    assert len(evidence) == result["D"] == result["D_plus"] == 16
    assert {row.retry_batch for row in evidence} == {0}
    assert {row.replica for row in evidence} == {7}
    assert all(row.window_id and row.trajectory_id for row in evidence)
    assert result["gp_reference"]["exact"] is True
    assert result["gp_reference"]["thinning"] == "none"
    assert result["gp_reference"]["count_by_gamma"] == {"0.1": 4, "0.3": 4}
    assert result["gp_reference"]["evidence_count_by_gamma"] == {
        "0.1": 8, "0.3": 8,
    }


def test_sliding_success_gp_is_per_gamma_fifo_and_causal(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    config = _exact_success_gp_config(
        rounds=4,
        gp_reference_mode="sliding_success_per_gamma_frozen_phi",
        gp_buffer_cap=10,
    )
    result = run_safe_expansion(
        policy, _AllSuccessfulCommitTask(), tmp_path,
        config=config,
        calibration_features=torch.randn(12, 8),
    )
    assert [
        row["gp_buffer_by_gamma"] for row in result["rounds"]
    ] == [
        {"0.1": 0, "0.3": 0},
        {"0.1": 4, "0.3": 4},
        {"0.1": 5, "0.3": 5},
        {"0.1": 5, "0.3": 5},
    ]

    evidence = torch.load(tmp_path / "gp_evidence.pt", weights_only=False)
    forward = _sliding_success_gp_rows(
        evidence, config.gammas, config.gp_buffer_cap, through_round=3,
    )
    reverse = _sliding_success_gp_rows(
        list(reversed(evidence)), config.gammas, config.gp_buffer_cap,
        through_round=3,
    )
    assert [row.window_id for row in forward] == [
        row.window_id for row in reverse
    ]
    assert all(row.round <= 3 for row in forward)
    assert result["gp_reference"]["count_by_gamma"] == {"0.1": 5, "0.3": 5}
    assert result["gp_reference"]["evidence_count_by_gamma"] == {
        "0.1": 16, "0.3": 16,
    }
    assert result["gp_reference"]["per_gamma_cap"] == 5
    assert result["gp_reference"]["exact"] is False


class _SecondBatchSuccessTask(_AllSuccessfulCommitTask):
    def verify(self, context, candidates, gamma):
        episode = int(round(float(context[1])))
        if episode < 8:
            return [
                Verification(False, False, -1.0, float(index))
                for index, _ in enumerate(candidates)
            ]
        return super().verify(context, candidates, gamma)


class _AllRejectCommitTask(_SuccessfulTrajectoryCommitTask):
    def verify(self, context, candidates, gamma):
        return [
            Verification(False, False, -1.0, float(index))
            for index, _ in enumerate(candidates)
        ]


class _ContextIndependentRandomPolicy(torch.nn.Module):
    """Test double exposing whether gather RNG coordinates include gamma."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def sample(self, context, count, generator):
        return torch.randn(count, 3, 1, generator=generator) + 0.0 * self.anchor

    @torch.no_grad()
    def embed(self, context, candidates, flow_time=0.9, base=None):
        return candidates.reshape(len(candidates), -1)

    def cfm_loss(self, contexts, candidates, reduction="none", loss_mask=None):
        target = candidates.reshape(len(candidates), 3)
        squared = (target - self.anchor).square()
        if loss_mask is None:
            values = squared.mean(dim=1)
        else:
            mask = loss_mask.reshape(len(candidates), 3)
            values = (squared * mask).sum(dim=1) / mask.sum(dim=1)
        if reduction == "none":
            return values
        if reduction == "mean":
            return values.mean()
        raise ValueError("reduction must be 'none' or 'mean'")


def test_success_retry_gather_crn_excludes_gamma(tmp_path):
    policy = _ContextIndependentRandomPolicy()
    events = []
    run_safe_expansion(
        policy, _AllSuccessfulCommitTask(), tmp_path,
        config=_exact_success_gp_config(rounds=1),
        calibration_features=torch.randn(12, 3),
        event_callback=events.append,
    )
    first_steps = {
        (event["gamma"], event["replica"]): event
        for event in events
        if event["retry_batch"] == 0 and event["step"] == 0
    }
    for replica in range(8):
        low = first_steps[(0.1, replica)]
        high = first_steps[(0.3, replica)]
        assert torch.equal(low["base_candidates"], high["base_candidates"])
        assert torch.equal(low["candidates"], high["candidates"])
        assert low["selected"] == high["selected"]


def test_success_retry_uses_whole_batches_and_unique_episode_ids(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    events = []
    result = run_safe_expansion(
        policy, _SecondBatchSuccessTask(), tmp_path,
        config=_exact_success_gp_config(
            rounds=1, gammas=(0.1,), max_retry_batches=2,
        ),
        calibration_features=torch.randn(12, 8),
        event_callback=events.append,
    )
    row = result["rounds"][0]
    detail = row["successful_executed_commit_by_gamma"]["0.1"]

    assert row["retry_batches_by_gamma"] == {"0.1": 2}
    assert row["attempted_episode_count"] == 16
    assert detail["successful_retry_batch"] == 1
    assert detail["committed_episode_id"] == 15
    assert detail["attempted_episode_count"] == 16
    assert {event["episode"] for event in events} == set(range(16))
    assert all(len(event["candidates"]) == 64 for event in events)
    assert all(len(event["selected"]) == 8 for event in events)
    assert all(
        event["step"] == 0 for event in events if event["episode"] < 8
    )


def test_success_retry_exhaustion_is_atomic(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    initial = {
        key: value.detach().clone() for key, value in policy.state_dict().items()
    }
    cfg = _exact_success_gp_config(
        rounds=1, gammas=(0.1,), max_retry_batches=2,
    )
    with pytest.raises(RuntimeError, match="round not committed"):
        run_safe_expansion(
            policy, _AllRejectCommitTask(), tmp_path, config=cfg,
            calibration_features=torch.randn(12, 8),
        )
    assert (tmp_path / "checkpoint_000.pt").is_file()
    assert not (tmp_path / "checkpoint_001.pt").exists()
    assert not (tmp_path / "query_archive.pt").exists()
    assert all(
        torch.equal(initial[key], value)
        for key, value in policy.state_dict().items()
    )


def test_exact_success_gp_cap_overflow_is_atomic(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    cfg = _exact_success_gp_config(
        rounds=2, gammas=(0.1,), gp_exact_max_rows_per_gamma=4,
    )
    with pytest.raises(RuntimeError, match="gp_exact_max_rows_per_gamma"):
        run_safe_expansion(
            policy, _AllSuccessfulCommitTask(), tmp_path, config=cfg,
            calibration_features=torch.randn(12, 8),
        )
    first = torch.load(
        tmp_path / "checkpoint_001.pt", weights_only=False
    )["model"]
    assert not (tmp_path / "checkpoint_002.pt").exists()
    assert all(
        torch.equal(first[key], value)
        for key, value in policy.state_dict().items()
    )


def test_successful_executed_windows_enter_gp_one_round_later(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    cfg = _successful_window_config(
        rounds=2, gammas=(0.1,), parallel_episodes=2, max_steps=5,
    )
    result = run_safe_expansion(
        policy, _SuccessfulTrajectoryCommitTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    first, second = result["rounds"]
    assert first["gp_buffer"] == 0
    assert first["gp_evidence_count"] == 4
    assert second["gp_buffer"] == 4
    assert second["gp_anchor_count"] == 4
    assert second["gp_adaptive_count"] == 0
    assert second["gp_evidence_count"] == 8
    assert result["gp_reference"]["through_round"] == 1
    assert result["gp_reference"]["evidence_count"] == 8


def test_successful_executed_windows_seed_fixed_gp_from_round_one_only(
        tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    cfg = _successful_window_config(
        rounds=2, gammas=(0.1,), parallel_episodes=2, max_steps=5,
        gp_reference_mode="round1_fixed_frozen_phi",
    )
    result = run_safe_expansion(
        policy, _SuccessfulTrajectoryCommitTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    first, second = result["rounds"]
    assert first["gp_buffer"] == 0
    assert second["gp_buffer"] == 4
    assert first["committed_window_ids"] == [
        "r0001:g0.1:e000000:w000000",
        "r0001:g0.1:e000000:w000001",
        "r0001:g0.1:e000000:w000002",
        "r0001:g0.1:e000000:w000003",
    ]
    assert second["committed_window_ids"] == [
        "r0002:g0.1:e000000:w000000",
        "r0002:g0.1:e000000:w000001",
        "r0002:g0.1:e000000:w000002",
        "r0002:g0.1:e000000:w000003",
    ]
    assert result["gp_reference"] == {
        "mode": "round1_fixed_frozen_phi",
        "round": 1,
        "count": 4,
        "source_window_count": 4,
        "identity_sha256": second["gp_reference_hash"],
        "frozen_phi": True,
    }


class _ShortSuccessfulTrajectoryTask(_SuccessfulTrajectoryCommitTask):
    def terminal(self, state):
        return "SUCCESS" if state["step"] >= 2 else None


def test_successful_executed_windows_verify_and_mask_short_terminal_suffixes(
        tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    cfg = _successful_window_config(
        gammas=(0.1,), parallel_episodes=1, max_steps=2,
        gp_reference_mode="recent_current_phi",
    )
    result = run_safe_expansion(
        policy, _ShortSuccessfulTrajectoryTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)
    detail = result["rounds"][0]["successful_executed_commit_by_gamma"]["0.1"]
    assert (tmp_path / "checkpoint_000.pt").is_file()
    assert (tmp_path / "checkpoint_001.pt").is_file()
    assert detail["executed_steps"] == 2
    assert detail["candidate_window_count"] == 2
    assert detail["committed_window_count"] == 2
    assert [row.valid_horizon for row in archive] == [2, 1]
    assert [row.loss_mask[:, 0].tolist() for row in archive] == [
        [1.0, 1.0, 0.0], [1.0, 0.0, 0.0],
    ]
    assert torch.count_nonzero(archive[0].candidate[2:]) == 0
    assert torch.count_nonzero(archive[1].candidate[1:]) == 0


class _ReconstructedWindowNoProgressTask(_SuccessfulTrajectoryCommitTask):
    def verify(self, context, candidates, gamma):
        values = super().verify(context, candidates, gamma)
        if len(candidates) != 1:
            return values
        return [
            Verification(
                True, True, 1.0, value.execution_cost,
                progress=-1.0, progress_eligible=False,
            )
            for value in values
        ]


def test_successful_executed_windows_are_reverified_for_progress(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    cfg = _successful_window_config(
        gammas=(0.1,), parallel_episodes=1, max_steps=4,
        gp_reference_mode="recent_current_phi",
    )
    with pytest.raises(RuntimeError, match="round not committed"):
        run_safe_expansion(
            policy, _ReconstructedWindowNoProgressTask(), tmp_path, config=cfg,
            calibration_features=torch.randn(12, 8),
        )
    assert (tmp_path / "checkpoint_000.pt").is_file()
    assert not (tmp_path / "checkpoint_001.pt").exists()


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"negative_alpha": 0.01}, "negative_alpha=0"),
        ({"target_gate_start_round": 2}, "inactive target gate"),
        ({"successful_trajectory_selector": "unknown"},
         "successful_trajectory_selector"),
        ({"successful_trajectories_per_gamma": 0},
         "successful_trajectories_per_gamma"),
        ({"successful_trajectories_per_gamma": 4},
         "parallel_episodes \\* max_retry_batches"),
    ],
)
def test_successful_executed_windows_validate_incompatible_options(
        updates, match):
    with pytest.raises(ValueError, match=match):
        _successful_window_config(**updates).validate()


def test_successful_executed_windows_preserve_composed_paired_flow_base(
        tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(
        3, (3, 1), hidden=16, representation_dim=8,
    )
    events = []
    result = run_safe_expansion(
        policy, _SuccessfulTrajectoryCommitTask(), tmp_path,
        config=_successful_window_config(
            rounds=2, gammas=(0.1,), paired_noised_representation=True,
            gp_reference_mode="sliding_success_per_gamma_current_phi",
        ),
        calibration_features=torch.randn(12, 8),
        event_callback=events.append,
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)
    executed = [
        event for event in events
        if event["episode"] == 0 and event["chosen_local"] is not None
    ]
    base_actions = [
        event["flow_bases"][
            event["selected"][event["chosen_local"]]
        ][0].detach().cpu()
        for event in executed
    ]
    expected = [
        torch.stack(base_actions[:3]),
        torch.stack(base_actions[1:4]),
        torch.cat([
            torch.stack(base_actions[2:4]),
            torch.zeros_like(base_actions[0])[None],
        ]),
        torch.cat([
            torch.stack(base_actions[3:4]),
            torch.zeros_like(base_actions[0])[None].repeat(2, 1),
        ]),
    ]
    assert result["D_plus"] == 8
    assert all(
        torch.equal(row.flow_base, value)
        for row, value in zip(archive[:4], expected)
    )
    assert all(row.flow_base is not None for row in archive)
    assert result["rounds"][1]["gp_buffer_by_gamma"] == {"0.1": 4}


def test_adaptive_ess_target_respects_finite_candidate_lower_bound():
    ExpansionConfig(K=4, B=2, adaptive_beta=True, ess_target=0.25).validate()
    ExpansionConfig(K=4, B=2, adaptive_beta=False, ess_target=0.1).validate()
    with pytest.raises(ValueError, match="at least 1/K"):
        ExpansionConfig(K=4, B=2, adaptive_beta=True, ess_target=0.1).validate()


class _RejectTask(_OneStepTask):
    def verify(self, context, candidates, gamma):
        return [Verification(False, False, -float(index), float(index))
                for index, _ in enumerate(candidates)]


def test_nvp_archive_keeps_one_reject_per_failed_rollout(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        execution_rule="min_cost", archive_rule="executed_plus_nvp_negative",
    )
    result = run_safe_expansion(
        policy, _RejectTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)
    assert result["rounds"][0]["verifier_queries"] == 4
    assert result["rounds"][0]["NVP"] == 1
    assert result["D"] == 1
    assert result["D_plus"] == 0
    assert archive[0].nvp_context
    assert archive[0].verification.execution_cost == 0.0


class _SafeNoProgressTask(_OneStepTask):
    def verify(self, context, candidates, gamma):
        return [Verification(True, True, 1.0, float(index),
                             progress=-1.0, progress_eligible=False)
                for index, _ in enumerate(candidates)]


def test_safe_negative_progress_causes_nvp_without_becoming_unsafe(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        archive_rule="all_queries",
    )
    result = run_safe_expansion(
        policy, _SafeNoProgressTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    assert result["rounds"][0]["NVP"] == 1
    assert result["D"] == 4
    assert result["D_plus"] == 4


class _NominalDiagnosticFailTask(_OneStepTask):
    def verify(self, context, candidates, gamma):
        return [Verification(True, False, 1.0, float(index),
                             progress=1.0, progress_eligible=True)
                for index, _ in enumerate(candidates)]


def test_nominal_diagnostic_does_not_gate_execution(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
    )
    result = run_safe_expansion(
        policy, _NominalDiagnosticFailTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    assert result["rounds"][0]["success"] == 1
    assert result["rounds"][0]["NVP"] == 0


class _SafeTargetRejectTask(_OneStepTask):
    def verify(self, context, candidates, gamma):
        return [
            Verification(
                True, True, 1.0, float(index),
                progress=1.0, progress_eligible=True, target_eligible=False,
            )
            for index, _ in enumerate(candidates)
        ]


def test_target_gate_starts_at_round_two_without_relabeling_safety(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    events = []
    cfg = ExpansionConfig(
        rounds=2, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        replay_rounds=2, replay_selector="uniform",
        archive_rule="all_queries", negative_alpha=0.01,
        target_gate_start_round=2,
    )
    result = run_safe_expansion(
        policy, _SafeTargetRejectTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
        event_callback=events.append,
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)

    assert result["rounds"][0]["success"] == 1
    assert result["rounds"][0]["NVP"] == 0
    assert result["rounds"][0]["replay_accepted_positives"] == 4
    assert result["rounds"][1]["success"] == 0
    assert result["rounds"][1]["NVP"] == 1
    assert result["rounds"][1]["replay_accepted_positives"] == 0
    assert result["D_plus"] == 8
    assert result["D_replay_accepted"] == 4
    assert result["rounds"][1]["negative_loss"] is None
    assert events[-1]["nvp_reason"] == "TARGET"
    assert events[-1]["target_gate_active"]

    round_two = [row for row in archive if row.round == 2]
    assert len(round_two) == 4
    assert all(row.verification.valid for row in round_two)
    assert all(not row.replay_eligible for row in round_two)
    assert all(row.nvp_context for row in round_two)


def test_safety_valid_replay_ignores_progress_and_target_gates(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=2, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        replay_rounds=2, replay_selector="uniform",
        archive_rule="all_queries", replay_acceptance="safety_valid",
        target_gate_start_round=2,
    )
    result = run_safe_expansion(
        policy, _SafeTargetRejectTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)

    # The target gate still terminates execution, but it neither relabels
    # safety nor suppresses valid generative-model supervision.
    assert result["rounds"][1]["NVP"] == 1
    assert result["rounds"][1]["replay_accepted_positives"] == 4
    assert result["D"] == result["D_plus"] == result["D_replay_accepted"] == 8
    assert all(row.verification.valid and row.replay_eligible for row in archive)
    assert "independent of progress and task target gates" in (
        result["semantics"]["replay_acceptance"]
    )
    assert "safety_valid replay remains independent" in (
        result["semantics"]["target_gate"]
    )


def test_safety_valid_replay_keeps_safe_nonprogressing_queries(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=1, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        archive_rule="all_queries", replay_acceptance="safety_valid",
        target_gate_start_round=1,
    )
    result = run_safe_expansion(
        policy, _SafeNoProgressTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    archive = torch.load(tmp_path / "query_archive.pt", weights_only=False)

    assert result["rounds"][0]["NVP"] == 1
    assert result["rounds"][0]["replay_accepted_positives"] == 4
    assert result["D"] == result["D_plus"] == result["D_replay_accepted"] == 4
    assert all(row.verification.valid and row.replay_eligible for row in archive)


def test_safety_valid_replay_requires_complete_query_archive():
    with pytest.raises(ValueError, match="archive_rule=all_queries"):
        ExpansionConfig(
            archive_rule="executed_only",
            replay_acceptance="safety_valid",
        ).validate()


def test_safety_valid_cumulative_gp_keeps_target_rejected_evidence(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=3, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        replay_rounds=2, replay_selector="uniform",
        archive_rule="all_queries", replay_acceptance="safety_valid",
        target_gate_start_round=2,
        gp_reference_mode="cumulative_accepted_frozen_phi",
    )
    result = run_safe_expansion(
        policy, _SafeTargetRejectTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    evidence = torch.load(tmp_path / "gp_evidence.pt", weights_only=False)

    assert [row["NVP"] for row in result["rounds"]] == [0, 1, 1]
    assert [row["gp_buffer"] for row in result["rounds"]] == [0, 4, 6]
    assert [row["gp_evidence_count"] for row in result["rounds"]] == [4, 8, 12]
    assert len(evidence) == 12
    assert result["D"] == result["D_plus"] == result["D_replay_accepted"] == 12
    assert all(row.verification.valid and row.replay_eligible for row in evidence)


def test_round_one_fixed_gp_reference_identity_is_stable(tmp_path):
    torch.manual_seed(0)
    policy = ConditionalFlowMLP(1, (2,), hidden=16, representation_dim=8)
    cfg = ExpansionConfig(
        rounds=3, gammas=(0.1,), parallel_episodes=1, max_steps=1,
        K=4, B=4, batch_size=2, gp_buffer_cap=8,
        replay_selector="uniform",
        gp_reference_mode="round1_fixed_frozen_phi",
        archive_rule="executed_only",
    )
    result = run_safe_expansion(
        policy, _OneStepTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )

    round_one, round_two, round_three = result["rounds"]
    assert round_one["gp_buffer"] == 0
    assert round_one["gp_reference_hash"] is None
    assert round_two["gp_buffer"] == round_three["gp_buffer"] == 4
    assert round_two["gp_reference_hash"]
    assert round_two["gp_reference_hash"] == round_three["gp_reference_hash"]
    for row in (round_one, round_two, round_three):
        assert 0.0 < row["marginal_ESS_over_K"] <= 1.0
        assert np.isfinite(row["sigma_pool_mean"])
        assert np.isfinite(row["sigma_selected_mean"])
        assert np.isfinite(row["uncertainty_uplift"])
    assert result["D"] == 3
    assert result["gp_reference"] == {
        "mode": "round1_fixed_frozen_phi",
        "round": 1,
        "count": 4,
        "source_query_count": 4,
        "identity_sha256": round_two["gp_reference_hash"],
        "frozen_phi": True,
    }


def test_cumulative_frozen_phi_coreset_grows_causally_and_stays_bounded(tmp_path):
    class RecordingFlow(ConditionalFlowMLP):
        embed_log = []

        def embed(self, *args, **kwargs):
            parameters = list(self.parameters())
            frozen = all(not parameter.requires_grad for parameter in parameters)
            checksum = tuple(
                float(parameter.detach().sum()) for parameter in parameters
            )
            type(self).embed_log.append((frozen, checksum))
            return super().embed(*args, **kwargs)

    torch.manual_seed(0)
    RecordingFlow.embed_log.clear()
    policy = RecordingFlow(1, (2,), hidden=16, representation_dim=8)
    initial = {
        key: value.detach().clone() for key, value in policy.state_dict().items()
    }
    cfg = ExpansionConfig(
        rounds=4, gammas=(0.1, 0.3), parallel_episodes=2, max_steps=1,
        K=4, B=4, batch_size=8, gp_buffer_cap=32,
        replay_selector="uniform", archive_rule="executed_only",
        learning_rate=1.0e-3,
        gp_reference_mode="cumulative_accepted_frozen_phi",
    )
    result = run_safe_expansion(
        policy, _OneStepTask(), tmp_path, config=cfg,
        calibration_features=torch.randn(12, 8),
    )
    evidence = torch.load(tmp_path / "gp_evidence.pt", weights_only=False)
    rows = result["rounds"]

    assert [row["gp_buffer"] for row in rows] == [0, 16, 24, 32]
    assert [row["gp_anchor_count"] for row in rows] == [0, 16, 16, 16]
    assert [row["gp_adaptive_count"] for row in rows] == [0, 0, 8, 16]
    assert [row["gp_evidence_count"] for row in rows] == [16, 32, 48, 64]
    hashes = [row["gp_reference_hash"] for row in rows]
    assert hashes[0] is None
    assert len(set(hashes[1:])) == 3

    # Every accepted selected-B query is evidence even though executed_only
    # deliberately stores one row per context in the replay archive.
    assert len(evidence) == 64
    assert result["D"] == 16
    assert all(row.verification.valid and row.replay_eligible for row in evidence)
    assert result["gp_reference"] == {
        "mode": "cumulative_accepted_frozen_phi",
        "round": None,
        "through_round": 3,
        "count": 32,
        "anchor_count": 16,
        "adaptive_count": 16,
        "evidence_count": 64,
        "identity_sha256": hashes[-1],
        "frozen_phi": True,
        "active_cap": 32,
        "per_gamma_total_cap": 16,
        "per_gamma_anchor_cap": 8,
        "per_gamma_adaptive_cap": 8,
        "per_gamma_ingress_cap": 4,
    }

    assert RecordingFlow.embed_log
    assert all(frozen for frozen, _ in RecordingFlow.embed_log)
    assert len({checksum for _, checksum in RecordingFlow.embed_log}) == 1
    assert any(
        not torch.equal(initial[key], value)
        for key, value in policy.state_dict().items()
    )


def test_frozen_phi_farthest_first_is_archive_order_independent():
    class IdentityEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def embed(self, context, candidates, **kwargs):
            return candidates

    verification = Verification(True, True, 1.0, 0.0)
    records = [
        QueryRecord(
            1, 0.1, 0, index, torch.zeros(1), candidate, verification
        )
        for index, candidate in enumerate((
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1.0, 0.0]),
            torch.tensor([0.0, -1.0]),
        ))
    ]
    policy = IdentityEmbed()
    forward = _frozen_phi_farthest_first(
        policy, records, 2, torch.device("cpu")
    )
    reverse = _frozen_phi_farthest_first(
        policy, list(reversed(records)), 2, torch.device("cpu")
    )
    assert [row.context_id for row in forward] == [
        row.context_id for row in reverse
    ]
