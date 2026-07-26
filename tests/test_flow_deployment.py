import json
from pathlib import Path

import numpy as np

from flow_deployment.bridge import (
    EndpointSimilarity,
    FlowDeploymentController,
    load_flow_policy,
    verify_deploy_sim_lock,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = (
    ROOT / "results/global50_reference/pretrain_global10_h48p32_s0"
)
LAB_CONFIG = ROOT / "configs/crazyflie_mppi_corner.json"


def _fixture():
    source = TaskEnvironment(load_config(PRETRAIN / "demo_config.json"))
    target_config = load_config(LAB_CONFIG)
    target = TaskEnvironment(target_config)
    frame = EndpointSimilarity.from_endpoints(
        source.start[:3], source.goal, target.start[:3], target.goal,
    )
    return source, target, target_config, frame


def test_endpoint_similarity_is_exact_and_right_handed():
    source, target, _, frame = _fixture()
    assert np.allclose(
        frame.source_to_target_position(source.start[:3]), target.start[:3],
    )
    assert np.allclose(
        frame.source_to_target_position(source.goal), target.goal,
    )
    assert np.allclose(frame.rotation.T @ frame.rotation, np.eye(3))
    assert np.isclose(np.linalg.det(frame.rotation), 1.0)


def test_endpoint_similarity_does_not_assume_source_forward_is_world_x():
    source_start = np.array([1.0, -2.0, 0.5])
    source_goal = np.array([1.0, 2.0, 1.0])
    target_start = np.array([-3.0, 0.0, 0.4])
    target_goal = np.array([0.0, -2.0, 1.2])
    frame = EndpointSimilarity.from_endpoints(
        source_start, source_goal, target_start, target_goal,
    )
    assert np.allclose(frame.source_to_target_position(source_start), target_start)
    assert np.allclose(frame.source_to_target_position(source_goal), target_goal)
    assert np.allclose(frame.target_to_source_position(target_goal), source_goal)


def test_lab_sphere_residual_is_explicit_not_silently_aligned():
    source, target, _, frame = _fixture()
    sphere = frame.target_sphere_to_source(target.spheres[0])
    assert np.allclose(
        sphere,
        np.array([1.8284991, -0.1212406, 2.1837142, 0.2741550]),
        atol=1.0e-6,
    )
    assert not np.allclose(sphere, source.spheres[0])


def test_controller_loads_canonical_policy_and_is_deterministic():
    _, target, target_config, frame = _fixture()
    policy, provenance = load_flow_policy(PRETRAIN)
    assert provenance["round"] == 0
    controller = FlowDeploymentController(
        policy,
        frame,
        target.spheres[0],
        float(target_config.safemppi.demo_u_max),
    )
    state = target.start.copy()
    action_a, _ = controller.plan(state, target.goal, 0.3, seed=17)
    controller.reset()
    action_b, _ = controller.plan(state, target.goal, 0.3, seed=17)
    assert np.array_equal(action_a, action_b)
    assert np.isfinite(action_a).all()
    assert np.max(np.abs(action_a)) <= 0.3 + 1.0e-7
    assert np.isclose(controller.authority_ratio, 0.3)


def test_minhyuk_deploy_files_match_lock():
    result = verify_deploy_sim_lock(ROOT)
    lock = json.loads(
        (ROOT / "flow_deployment/deploy_sim_lock.json").read_text()
    )
    assert result["status"] == "DEPLOY_SIM_LOCK_VERIFIED"
    assert result["files"] == lock["files"]
