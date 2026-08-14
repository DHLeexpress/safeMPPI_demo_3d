"""Shared execution-ranking cost terms."""

from __future__ import annotations

import numpy as np


def obstacle_conditioned_speed_cost(
    states: np.ndarray,
    clearances: np.ndarray,
    *,
    weight: float,
    clearance_target_m: float,
    clearance_temperature_m: float,
) -> float:
    """Penalize predicted speed in proportion to obstacle proximity."""
    scaled = np.clip(
        (np.asarray(clearances, np.float64) - clearance_target_m)
        / clearance_temperature_m,
        -60.0,
        60.0,
    )
    proximity = 1.0 / (1.0 + np.exp(scaled))
    speed_squared = np.square(
        np.asarray(states, np.float64)[1:, 3:6]
    ).sum(axis=1)
    return float(weight) * float(np.mean(proximity * speed_squared))
