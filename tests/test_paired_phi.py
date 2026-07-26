from __future__ import annotations

import torch

from safe_mppi.flow_model import ConditionalFlowMLP


@torch.no_grad()
def _legacy_sample(
    model: ConditionalFlowMLP,
    context: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_context = context.reshape(1, -1).expand(count, -1)
    x = torch.randn(
        count,
        model.plan_dim,
        device=expanded_context.device,
        generator=generator,
    )
    base = x.clone()
    for index in range(model.nfe):
        t = torch.full((count,), index / model.nfe, device=x.device)
        x = x + model(x, t, expanded_context) / model.nfe
    output = x.reshape(count, *model.plan_shape)
    if model.control_limit is not None:
        output = output.clamp(-model.control_limit, model.control_limit)
    return output, base.reshape(count, *model.plan_shape)


def _model() -> ConditionalFlowMLP:
    torch.manual_seed(17)
    return ConditionalFlowMLP(
        context_dim=4,
        plan_shape=(3, 2),
        hidden=12,
        representation_dim=7,
        control_limit=0.2,
        nfe=3,
        trunk_depth=2,
        time_features="raw1",
    )


def test_sample_with_base_matches_legacy_output_and_random_stream():
    model = _model()
    context = torch.tensor([0.2, -0.1, 0.4, 0.7])
    legacy_generator = torch.Generator().manual_seed(1234)
    paired_generator = torch.Generator().manual_seed(1234)

    expected_output, expected_base = _legacy_sample(
        model, context, count=5, generator=legacy_generator,
    )
    output, base = model.sample_with_base(
        context, count=5, generator=paired_generator,
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(base, expected_base)
    assert torch.equal(
        torch.randn(8, generator=paired_generator),
        torch.randn(8, generator=legacy_generator),
    )
    assert output.shape == base.shape == (5, 3, 2)
    assert float(base.abs().max()) > model.control_limit
    assert torch.all(
        output.abs() <= torch.as_tensor(model.control_limit, dtype=output.dtype)
    )
    assert not output.requires_grad
    assert not base.requires_grad


def test_sample_delegates_to_paired_sampler_without_changing_output():
    model = _model()
    context = torch.tensor([-0.3, 0.5, 0.1, -0.8])
    sample_generator = torch.Generator().manual_seed(4321)
    paired_generator = torch.Generator().manual_seed(4321)

    output = model.sample(context, count=4, generator=sample_generator)
    paired_output, _ = model.sample_with_base(
        context, count=4, generator=paired_generator,
    )

    assert torch.equal(output, paired_output)
    assert torch.equal(
        torch.randn(8, generator=sample_generator),
        torch.randn(8, generator=paired_generator),
    )


def test_flow_base_std_scales_the_same_gaussian_draw_before_integration():
    model = _model()
    context = torch.tensor([0.1, 0.2, -0.3, 0.4])
    unit_generator = torch.Generator().manual_seed(2468)
    scaled_generator = torch.Generator().manual_seed(2468)

    _, unit_base = model.sample_with_base(
        context, count=6, generator=unit_generator,
    )
    scaled_output, scaled_base = model.sample_with_base(
        context, count=6, generator=scaled_generator, base_std=1.75,
    )

    assert torch.equal(scaled_base, unit_base * 1.75)
    assert scaled_output.shape == (6, 3, 2)
    assert torch.equal(
        torch.randn(8, generator=unit_generator),
        torch.randn(8, generator=scaled_generator),
    )


def test_flow_base_std_rejects_invalid_scale():
    model = _model()
    context = torch.zeros(4)
    for value in (-0.1, float("inf"), float("nan")):
        try:
            model.sample(
                context, count=1, generator=torch.Generator(), base_std=value,
            )
        except ValueError as error:
            assert "base_std" in str(error)
        else:
            raise AssertionError(f"accepted invalid base_std={value}")
