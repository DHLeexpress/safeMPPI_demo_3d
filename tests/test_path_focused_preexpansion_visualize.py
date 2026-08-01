import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "path_focused_preexpansion_visualize",
    ROOT / "scripts/visualize_path_focused_preexpansion.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cross_gamma_spread_is_finite_for_opposite_detours():
    groups = []
    for group_index in range(2):
        items = []
        for gamma_index, gamma in enumerate(MODULE.GAMMAS):
            x = np.linspace(0.0, 1.0, 64)
            path = np.column_stack([
                x,
                (group_index + 1) * (gamma_index - 1.5) * x * (1.0 - x),
                np.zeros_like(x),
            ])
            items.append({
                "gamma": gamma,
                "time_to_goal_s": 1.0,
                "features": {
                    "resampled_path": path,
                    "transverse_rms_m": float(np.std(path[:, 1])),
                    "path_length_excess_ratio": 0.1,
                    "interaction_fraction": 0.5,
                },
            })
        groups.append({"key": group_index, "items": items})

    ranked = MODULE._rank_groups(
        groups,
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )

    assert len(ranked) == 2
    assert all(np.isfinite(row["cross_gamma_spread"]) for row in ranked)
    assert ranked[0]["cross_gamma_spread"] > 0.0
