"""Projected 1-2-3 bowling-route diagnostics for successful 3-D paths."""
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


BOWLING_ROUTE_CODES = tuple(
    "".join(bits) for bits in product("LR", repeat=3)
)
BOWLING_ROW_PROGRESS = (0.25, 0.45, 0.65)
BOWLING_LANE_DELTA_M = 0.345
BOWLING_STABILITY_MARGIN_M = 0.02


def _first_forward_crossing(
    progress: np.ndarray,
    values: np.ndarray,
    target: float,
) -> np.ndarray:
    """Interpolate the first forward crossing, preserving backtracking."""
    for index in range(1, len(progress)):
        before = float(progress[index - 1])
        after = float(progress[index])
        if before < target <= after and after - before > 1.0e-12:
            weight = (target - before) / (after - before)
            return (
                (1.0 - weight) * values[index - 1]
                + weight * values[index]
            )
    raise ValueError(f"successful path never crosses progress={target:g}")


def bowling_route_signature(
    states: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    lane_delta_m: float = BOWLING_LANE_DELTA_M,
    stability_margin_m: float = BOWLING_STABILITY_MARGIN_M,
    sphere_radius_m: float = 0.2405,
) -> dict[str, Any]:
    """Classify sequential L/R choices while retaining 3-D pin offsets."""
    path = np.asarray(states, np.float64)
    start3 = np.asarray(start, np.float64).reshape(-1)[:3]
    goal3 = np.asarray(goal, np.float64).reshape(-1)[:3]
    if (
        path.ndim != 2
        or path.shape[1] < 3
        or len(path) < 2
        or not bool(np.isfinite(path[:, :3]).all())
    ):
        raise ValueError("bowling route requires a finite [T,>=3] path")
    direction = goal3 - start3
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        raise ValueError("bowling route requires distinct start and goal")
    forward = direction / length
    lateral = np.asarray([-forward[1], forward[0], 0.0], np.float64)
    offset = path[:, :3] - start3[None]
    progress = offset @ forward / length
    crossings = np.asarray([
        _first_forward_crossing(progress, path[:, :3], target)
        for target in BOWLING_ROW_PROGRESS
    ])
    crossing_offset = crossings - start3[None]
    lateral_values = crossing_offset @ lateral
    vertical_values = crossings[:, 2] - start3[2]

    target_lateral = 0.0
    bits: list[str] = []
    margins: list[float] = []
    lateral_deltas: list[float] = []
    for value in lateral_values:
        delta = float(value - target_lateral)
        bit = "L" if delta >= 0.0 else "R"
        bits.append(bit)
        margins.append(abs(delta))
        lateral_deltas.append(delta)
        target_lateral += lane_delta_m if bit == "L" else -lane_delta_m
    stable_bits = [
        bit if margin >= stability_margin_m else "X"
        for bit, margin in zip(bits, margins)
    ]
    vertical_signs = [
        "A" if value > stability_margin_m
        else "B" if value < -stability_margin_m
        else "P"
        for value in vertical_values
    ]
    polar_angles = np.degrees(np.arctan2(
        vertical_values, np.asarray(lateral_deltas),
    ))
    vertical_dominant = np.abs(vertical_values) > np.abs(lateral_deltas)
    plane_clearance = np.sqrt(
        np.square(lateral_deltas) + np.square(vertical_values)
    ) - float(sphere_radius_m)
    return {
        "code": "".join(bits),
        "stable_code": "".join(stable_bits),
        "decision_xyz_m": crossings.tolist(),
        "decision_lateral_m": lateral_values.tolist(),
        "decision_target_lateral_m": [
            0.0,
            lane_delta_m if bits[0] == "L" else -lane_delta_m,
            lane_delta_m * (
                (1 if bits[0] == "L" else -1)
                + (1 if bits[1] == "L" else -1)
            ),
        ],
        "decision_lateral_delta_m": lateral_deltas,
        "decision_vertical_m": vertical_values.tolist(),
        "decision_vertical_sign": "".join(vertical_signs),
        "decision_polar_angle_deg": polar_angles.tolist(),
        "decision_vertical_dominant": vertical_dominant.tolist(),
        "decision_plane_clearance_m": plane_clearance.tolist(),
        "minimum_decision_margin_m": float(min(margins)),
    }


def summarize_bowling_coverage(rows: list[dict]) -> dict[str, Any]:
    """Summarize projected stable routes among raw terminal successes."""
    successes = [row for row in rows if row.get("status") == "SUCCESS"]
    signatures = [
        row.get("bowling_route") for row in successes
        if isinstance(row.get("bowling_route"), dict)
    ]
    stable = [
        str(signature["stable_code"])
        for signature in signatures
        if str(signature.get("stable_code")) in BOWLING_ROUTE_CODES
    ]
    counts = {
        code: int(sum(value == code for value in stable))
        for code in BOWLING_ROUTE_CODES
    }
    observed = [code for code in BOWLING_ROUTE_CODES if counts[code] > 0]
    vertical_dominant_successes = sum(
        any(bool(value) for value in signature["decision_vertical_dominant"])
        for signature in signatures
    )
    vertical_values = [
        abs(float(value))
        for signature in signatures
        for value in signature["decision_vertical_m"]
    ]
    return {
        "attempts": int(len(rows)),
        "terminal_successes": int(len(successes)),
        "classified_successes": int(len(signatures)),
        "stable_classified_successes": int(len(stable)),
        "ambiguous_successes": int(len(signatures) - len(stable)),
        "route_counts": counts,
        "observed_routes": observed,
        "coverage_count": int(len(observed)),
        "coverage_fraction": float(len(observed) / len(BOWLING_ROUTE_CODES)),
        "route_attempt_mass": {
            code: float(counts[code] / len(rows)) if rows else None
            for code in BOWLING_ROUTE_CODES
        },
        "route_success_mass": {
            code: float(counts[code] / len(successes)) if successes else None
            for code in BOWLING_ROUTE_CODES
        },
        "vertical_dominant_successes": int(vertical_dominant_successes),
        "mean_abs_decision_vertical_m": (
            float(np.mean(vertical_values)) if vertical_values else None
        ),
        "max_abs_decision_vertical_m": (
            float(np.max(vertical_values)) if vertical_values else None
        ),
        "contract": (
            "SUCCESS-only first-forward crossings at progress "
            "0.25/0.45/0.65; stable sequential XY L/R code with 0.02 m "
            "decision margin. This is projected coverage, not full 3-D "
            "homotopy coverage."
        ),
    }
