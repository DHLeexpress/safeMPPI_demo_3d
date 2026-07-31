"""Focused contracts for the additive plane-packed uniform H_P encoder."""
from pathlib import Path

import numpy as np
import torch

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.geometry import build_nominal_polytope
from safe_mppi.lab_clutter_expansion import (
    LabClutterExpansionPolicyAdapter,
    LabClutterSphereExpansionTask,
    load_lab_clutter_expansion_policy,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_evaluation import (
    SUPPORTED_LAB_VISUAL_CONTEXT_SCHEMAS,
    _decode_event_scene,
)
from safe_mppi.lab_reference_flow_task import (
    LabReferenceFlowController,
    policy_context,
)
from safe_mppi.lab_visual_flow import (
    LAB_HP100_CHANNELS,
    LAB_HP100_DYNAMIC_FACE_COUNT,
    LAB_HP100_FRAME,
    LAB_HP100_GRID_SHAPE,
    LAB_HP100_HISTORY_PACKED_DIM,
    LAB_HP100_HISTORY_SCHEMA,
    LAB_HP100_PACKED_DIM,
    LAB_HP100_PLANE_CHANNELS,
    LAB_HP100_RADIAL_EDGES,
    LAB_HP100_SCHEMA,
    LabUniformHp100Encoder,
    LabUniformHp100FlowPolicy,
    LabUniformHp100HistoryFlowPolicy,
    LabUniformHp100Rasterizer,
    load_lab_reference_policy,
    uniform_hp100_grid_points,
)
from scripts import pretrain_lab_reference_flow as pretrain
from scripts import run_ball_expansion as run_expansion


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = (
    ROOT
    / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
    / "resolved_config.json"
)
SPHERE_CONFIG = ROOT / "configs/lab_clutter_spheres_path_v2.json"


def _arch(config) -> dict:
    return pretrain.pretraining_arch(
        "uniform_hp100",
        config,
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=0,
        nfe=1,
        trunk_depth=3,
    )


def test_plane_pack_raster_matches_full_nominal_polytope_hp():
    config = load_config(REFERENCE_CONFIG)
    env = TaskEnvironment(config)
    position = env.start[:3] + np.asarray([0.13, -0.07, 0.04])
    state = np.concatenate([position, np.zeros(3, np.float32)])
    packed = policy_context(
        type("Policy", (), {"context_schema": LAB_HP100_SCHEMA})(),
        env,
        state,
        0.3,
    )
    assert packed.shape == (LAB_HP100_PACKED_DIM,)
    grid = LabUniformHp100Rasterizer()(
        torch.from_numpy(packed[7:]).unsqueeze(0)
    )[0, 0].numpy()

    points = uniform_hp100_grid_points(position).reshape(-1, 3)
    polytope = build_nominal_polytope(
        position,
        env.spheres,
        env.cylinders,
        env.bounds,
        sensing_range=env.mppi.sensing_range,
        obstacle_margin=0.0,
    )
    margins = np.maximum(polytope.margins, 1.0e-3)
    expected = (
        polytope.b[None] - points @ polytope.A.T
    ) / margins[None]
    expected = np.clip(expected.min(axis=1), -1.0, 1.0).reshape(
        LAB_HP100_GRID_SHAPE[1:]
    )
    assert np.allclose(grid, expected, rtol=0.0, atol=2.0e-6)


def test_hp100_encoder_has_no_radial_pool_and_receives_gradients():
    assert len(LAB_HP100_RADIAL_EDGES) == 101
    assert np.allclose(np.diff(LAB_HP100_RADIAL_EDGES), 0.02)
    assert LAB_HP100_GRID_SHAPE == (1, 32, 32, 100)
    encoder = LabUniformHp100Encoder(64)
    assert not any(
        isinstance(module, torch.nn.AvgPool3d)
        for module in encoder.modules()
    )
    grid = torch.linspace(-1.0, 1.0, 32 * 32 * 100).reshape(
        1, *LAB_HP100_GRID_SHAPE
    )
    token = encoder(grid)
    assert token.shape == (1, 64)
    token.square().mean().backward()
    convs = [
        module for module in encoder.modules()
        if isinstance(module, torch.nn.Conv3d)
    ]
    assert len(convs) == 3
    for convolution in convs:
        gradient = convolution.weight.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0


def test_hp100_policy_checkpoint_pretrain_and_clutter_routing(tmp_path):
    config = load_config(SPHERE_CONFIG)
    policy = pretrain.build_pretraining_policy(
        "uniform_hp100",
        config,
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=0,
        nfe=1,
        trunk_depth=3,
    )
    assert isinstance(policy, LabUniformHp100FlowPolicy)
    assert policy.context_dim == LAB_HP100_PACKED_DIM
    arch = _arch(config)
    assert arch == {
        "kind": LAB_HP100_SCHEMA,
        "plan_shape": [10, 3],
        "hidden": 8,
        "representation_dim": 4,
        "grid_token_dim": 64,
        "grid_shape": list(LAB_HP100_GRID_SHAPE),
        "grid_channels": list(LAB_HP100_CHANNELS),
        "grid_frame": LAB_HP100_FRAME,
        "radial_edges": list(LAB_HP100_RADIAL_EDGES),
        "plane_face_count": LAB_HP100_DYNAMIC_FACE_COUNT,
        "plane_row_channels": list(LAB_HP100_PLANE_CHANNELS),
        "control_limit": config.safemppi.demo_u_max,
        "nfe": 1,
        "trunk_depth": 3,
        "time_features": "raw1",
    }
    checkpoint = tmp_path / "hp100.pt"
    torch.save({"model": policy.state_dict(), "arch": arch}, checkpoint)
    loaded = load_lab_reference_policy(checkpoint)
    assert isinstance(loaded, LabUniformHp100FlowPolicy)
    assert loaded.context_schema == LAB_HP100_SCHEMA

    scene_spec = sphere_scene_spec_from_config(config)
    wrapped = LabClutterExpansionPolicyAdapter(
        loaded,
        verifier_suffix_dim=6 + scene_spec.packed_dim,
    )
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=wrapped.context_schema,
        scene_spec=scene_spec,
    )
    context = task.context(task.reset(0.3, 0, 17), 0.3)
    assert context.shape == (
        LAB_HP100_PACKED_DIM + task.verifier_suffix_dim,
    )
    assert wrapped._policy_context(context).shape == (
        LAB_HP100_PACKED_DIM,
    )
    assert LAB_HP100_SCHEMA in SUPPORTED_LAB_VISUAL_CONTEXT_SCHEMAS
    assert run_expansion.LAB_CONTEXT_BASE_PACKED_DIMS[LAB_HP100_SCHEMA] == (
        LAB_HP100_PACKED_DIM
    )


def test_lab_calibration_reuses_compact_context_artifact(tmp_path, monkeypatch):
    config = load_config(SPHERE_CONFIG)
    gammas = np.asarray([0.1] * 13 + [0.3] * 13 + [0.5] * 12 + [1.0] * 12)
    contexts = torch.zeros(50, LAB_HP100_PACKED_DIM)
    contexts[:, 6] = torch.from_numpy(gammas)
    artifact = tmp_path / "calibration_contexts.pt"
    torch.save(contexts, artifact)

    class FakePolicy(torch.nn.Module):
        context_dim = LAB_HP100_PACKED_DIM
        policy_context_dim = LAB_HP100_PACKED_DIM

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def sample_with_base(self, context, count, generator, base_std=1.0):
            del context, generator, base_std
            values = torch.zeros(count, 10, 3)
            return values, values.clone()

        def embed(self, contexts, plans, base=None):
            del plans, base
            return torch.ones(len(contexts), 4)

    monkeypatch.setattr(
        run_expansion,
        "lab_reference_demo_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full demo archive must not be rebuilt")
        ),
    )
    features = run_expansion._lab_pretrained_phi_calibration(
        FakePolicy(),
        tmp_path,
        {
            "rbf_calibration": {
                "context_artifact": artifact.name,
                "context_artifact_sha256": run_expansion._sha256_file(
                    artifact
                ),
            },
        },
        config,
        7,
        flow_base_std=1.5,
        paired_noised_representation=True,
    )
    assert features.shape == (50, 4)


def _history_arch(config) -> dict:
    return pretrain.pretraining_arch(
        "uniform_hp100_gru",
        config,
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=32,
        nfe=1,
        trunk_depth=3,
    )


def test_hp100_gru32_packing_and_encoder_gradients():
    config = load_config(SPHERE_CONFIG)
    env = TaskEnvironment(config)
    history = np.zeros((10, 4), np.float32)
    history[-2:, :3] = np.asarray([
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, -0.6],
    ])
    history[-2:, 3] = 1.0
    marker = type(
        "Policy", (), {"context_schema": LAB_HP100_HISTORY_SCHEMA},
    )()
    packed = policy_context(
        marker,
        env,
        env.start,
        0.3,
        raw_history=history,
    )
    base = policy_context(
        type("Policy", (), {"context_schema": LAB_HP100_SCHEMA})(),
        env,
        env.start,
        0.3,
    )
    assert packed.shape == (LAB_HP100_HISTORY_PACKED_DIM,)
    assert np.array_equal(packed[:LAB_HP100_PACKED_DIM], base)
    assert np.array_equal(
        packed[LAB_HP100_PACKED_DIM:].reshape(10, 4), history,
    )

    policy = LabUniformHp100HistoryFlowPolicy(
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=32,
        nfe=1,
        trunk_depth=3,
    )
    encoded = policy.encode_context(torch.from_numpy(packed).unsqueeze(0))
    encoded.square().mean().backward()
    gru_gradient = policy.history_encoder.weight_ih_l0.grad
    visual_gradient = next(policy.grid_encoder.parameters()).grad
    assert gru_gradient is not None and float(gru_gradient.abs().sum()) > 0.0
    assert (
        visual_gradient is not None
        and float(visual_gradient.abs().sum()) > 0.0
    )


def test_hp100_gru32_checkpoint_roundtrip_and_expansion_freeze(tmp_path):
    config = load_config(SPHERE_CONFIG)
    policy = pretrain.build_pretraining_policy(
        "uniform_hp100_gru",
        config,
        hidden=8,
        representation_dim=4,
        grid_token_dim=64,
        history_token_dim=32,
        nfe=1,
        trunk_depth=3,
    )
    checkpoint = tmp_path / "hp100_gru32.pt"
    torch.save({
        "model": policy.state_dict(),
        "arch": _history_arch(config),
    }, checkpoint)
    loaded = load_lab_reference_policy(checkpoint)
    assert isinstance(loaded, LabUniformHp100HistoryFlowPolicy)
    assert loaded.context_schema == LAB_HP100_HISTORY_SCHEMA
    assert loaded.context_dim == LAB_HP100_HISTORY_PACKED_DIM
    assert loaded.history_token_dim == 32

    scene_spec = sphere_scene_spec_from_config(config)
    frozen = load_lab_clutter_expansion_policy(
        checkpoint,
        verifier_suffix_dim=6 + scene_spec.packed_dim,
    )
    assert not any(
        parameter.requires_grad
        for parameter in frozen.policy.history_encoder.parameters()
    )
    trainable = load_lab_clutter_expansion_policy(
        checkpoint,
        verifier_suffix_dim=6 + scene_spec.packed_dim,
        train_history_encoder=True,
    )
    assert all(
        parameter.requires_grad
        for parameter in trainable.policy.history_encoder.parameters()
    )


def test_hp100_gru32_closed_loop_history_and_clutter_event_routing():
    config = load_config(SPHERE_CONFIG)
    env = TaskEnvironment(config)

    class RecordingPolicy(torch.nn.Module):
        context_schema = LAB_HP100_HISTORY_SCHEMA
        context_dim = LAB_HP100_HISTORY_PACKED_DIM
        plan_shape = (10, 3)

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.contexts = []

        def sample(self, context, count, generator, base_std=1.0):
            del generator, base_std
            self.contexts.append(context.detach().cpu().clone())
            plan = torch.zeros(count, 10, 3, device=context.device)
            plan[:, :, :] = torch.tensor(
                [0.1, -0.2, 0.05], device=context.device,
            )
            return plan

    policy = RecordingPolicy()
    controller = LabReferenceFlowController(policy, env)
    controller.plan(env.start, env.goal, 0.3, seed=1)
    controller.plan(env.start, env.goal, 0.3, seed=2)
    first = policy.contexts[0][LAB_HP100_PACKED_DIM:].reshape(10, 4)
    second = policy.contexts[1][LAB_HP100_PACKED_DIM:].reshape(10, 4)
    assert torch.count_nonzero(first) == 0
    assert torch.allclose(second[-1, :3], torch.tensor([0.1, -0.2, 0.05]))
    assert second[-1, 3] == 1.0

    scene_spec = sphere_scene_spec_from_config(config)
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=LAB_HP100_HISTORY_SCHEMA,
        scene_spec=scene_spec,
    )
    state = task.reset(0.3, 0, 17)
    context = task.context(state, 0.3)
    compact_context = torch.cat([
        context[:7],
        context[LAB_HP100_PACKED_DIM:LAB_HP100_HISTORY_PACKED_DIM],
        context[-task.verifier_suffix_dim:],
    ])
    decoded = _decode_event_scene(
        {"context": compact_context, "gamma": 0.3, "robot": state["x"]},
        task,
    )
    assert np.array_equal(decoded["spheres"], state["spheres"])
    assert LAB_HP100_HISTORY_SCHEMA in SUPPORTED_LAB_VISUAL_CONTEXT_SCHEMAS
    assert run_expansion.LAB_CONTEXT_BASE_PACKED_DIMS[
        LAB_HP100_HISTORY_SCHEMA
    ] == LAB_HP100_PACKED_DIM
