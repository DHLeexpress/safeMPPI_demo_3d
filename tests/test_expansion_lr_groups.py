import pytest

from safe_mppi.expansion import ExpansionConfig
from safe_mppi.flow_model import ConditionalFlowMLP


def test_expansion_parameter_groups_only_slow_the_first_linear():
    policy = ConditionalFlowMLP(
        context_dim=12,
        plan_shape=(10, 3),
        hidden=48,
        representation_dim=32,
        trunk_depth=2,
        time_features="raw1",
    )
    groups = policy.expansion_parameter_groups(1.0e-4, 0.1)

    assert [group["lr"] for group in groups] == [1.0e-5, 1.0e-4]
    assert {id(value) for value in groups[0]["params"]} == {
        id(value) for value in policy.trunk[0].parameters()
    }
    assert (
        {id(value) for group in groups for value in group["params"]}
        == {id(value) for value in policy.parameters()}
    )


def test_first_layer_lr_scale_must_be_positive_and_at_most_one():
    with pytest.raises(ValueError, match="first_layer_lr_scale"):
        ExpansionConfig(first_layer_lr_scale=0.0).validate()
    with pytest.raises(ValueError, match="first_layer_lr_scale"):
        ExpansionConfig(first_layer_lr_scale=1.1).validate()
