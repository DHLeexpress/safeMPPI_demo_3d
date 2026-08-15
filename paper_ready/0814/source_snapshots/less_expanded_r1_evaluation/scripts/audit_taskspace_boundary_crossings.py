#!/usr/bin/env python3
"""Audit the same raw trajectories against legacy and expanded task spaces."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch


FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _parse_bounds(text: str) -> np.ndarray:
    values = [float(value) for value in text.split(",")]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "bounds must be xmin,xmax,ymin,ymax,zmin,zmax"
        )
    bounds = np.asarray(values, np.float64).reshape(3, 2)
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise argparse.ArgumentTypeError("every lower bound must precede upper")
    return bounds


def _points(row: dict) -> np.ndarray:
    dense = np.asarray(row.get("dense_steps", []), np.float64)
    if dense.size:
        return dense.reshape(-1, 3)
    states = np.asarray(row["states"], np.float64)
    return states[:, :3]


def _crossing(points: np.ndarray, bounds: np.ndarray) -> tuple[bool, str | None]:
    tolerance = 1e-7
    for point in points:
        violations = np.asarray([
            bounds[0, 0] - point[0], point[0] - bounds[0, 1],
            bounds[1, 0] - point[1], point[1] - bounds[1, 1],
            bounds[2, 0] - point[2], point[2] - bounds[2, 1],
        ])
        if float(violations.max()) > tolerance:
            return True, FACE_NAMES[int(np.argmax(violations))]
    return False, None


def _counter_json(counter: Counter) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {str(q): float(np.quantile(array, q)) for q in (0.5, 0.9, 0.99)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--old-bounds", type=_parse_bounds,
        default=_parse_bounds("-2.5,1.3,-1.7,1.8,0.1,1.7"),
    )
    parser.add_argument(
        "--new-bounds", type=_parse_bounds,
        default=_parse_bounds("-2.5,1.3,-2.1,1.8,0.1,1.7"),
    )
    args = parser.parse_args()
    trajectories = torch.load(
        args.trajectories,
        map_location="cpu",
        weights_only=False,
    )

    episode_rows = []
    aggregate: dict[str, dict[str, dict]] = {}
    for round_i, rows in sorted(trajectories.items(), key=lambda pair: int(pair[0])):
        by_gamma: dict[str, list[dict]] = {}
        for row in rows:
            points = _points(row)
            old_cross, old_face = _crossing(points, args.old_bounds)
            new_cross, new_face = _crossing(points, args.new_bounds)
            old_y_min = float(args.old_bounds[1, 0])
            crossed_old_y = bool(np.any(points[:, 1] < old_y_min - 1e-7))
            first_old_y = (
                int(np.flatnonzero(points[:, 1] < old_y_min - 1e-7)[0])
                if crossed_old_y else None
            )
            returned_inside_old_y = bool(
                crossed_old_y
                and np.any(points[first_old_y + 1:, 1] >= old_y_min - 1e-7)
            )
            delta_y = np.diff(points[:, 1])
            reversal_indices = (
                np.flatnonzero(delta_y[first_old_y:] > 1e-7) + first_old_y
                if crossed_old_y else np.empty(0, dtype=int)
            )
            reversal_index = (
                int(reversal_indices[0]) if len(reversal_indices) else None
            )
            # The approved reference governor uses dt=0.1 with 10 dense
            # integration substeps, so each dense displacement is 0.01 s.
            turnaround_after_crossing_s = (
                0.01 * (reversal_index - first_old_y)
                if reversal_index is not None else None
            )
            max_old_y_overshoot_m = float(max(
                old_y_min - float(points[:, 1].min()), 0.0,
            ))
            record = {
                "round": int(round_i),
                "gamma": float(row["gamma"]),
                "episode": int(row["episode"]),
                "status": str(row["status"]),
                "mode": str(row.get("mode", "none")),
                "old_boundary_crossing": old_cross,
                "new_boundary_crossing": new_cross,
                "old_only_crossing": bool(old_cross and not new_cross),
                "first_old_exit_face": old_face,
                "first_new_exit_face": new_face,
                "crossed_old_y_min": crossed_old_y,
                "returned_inside_old_y_min": returned_inside_old_y,
                "successful_after_old_y_crossing": bool(
                    crossed_old_y and row["status"] == "SUCCESS"
                ),
                "max_old_y_min_overshoot_m": max_old_y_overshoot_m,
                "turnaround_after_old_y_crossing_s": (
                    turnaround_after_crossing_s
                ),
            }
            episode_rows.append(record)
            by_gamma.setdefault(f"{float(row['gamma']):g}", []).append(record)

        aggregate[str(int(round_i))] = {}
        for gamma, group in sorted(by_gamma.items(), key=lambda pair: float(pair[0])):
            status = Counter(record["status"] for record in group)
            old_faces = Counter(
                record["first_old_exit_face"] for record in group
                if record["first_old_exit_face"] is not None
            )
            new_faces = Counter(
                record["first_new_exit_face"] for record in group
                if record["first_new_exit_face"] is not None
            )
            old_only = [record for record in group if record["old_only_crossing"]]
            old_y = [record for record in group if record["crossed_old_y_min"]]
            aggregate[str(int(round_i))][gamma] = {
                "episodes": len(group),
                "reported_status": _counter_json(status),
                "old_boundary_crossings": sum(
                    record["old_boundary_crossing"] for record in group
                ),
                "new_boundary_crossings": sum(
                    record["new_boundary_crossing"] for record in group
                ),
                "old_only_crossings": len(old_only),
                "old_only_terminal_status": _counter_json(Counter(
                    record["status"] for record in old_only
                )),
                "first_old_exit_faces": _counter_json(old_faces),
                "first_new_exit_faces": _counter_json(new_faces),
                "old_y_min_crossings": len(old_y),
                "old_y_min_crossing_then_return": sum(
                    record["returned_inside_old_y_min"] for record in old_y
                ),
                "old_y_min_crossing_then_success": sum(
                    record["successful_after_old_y_crossing"] for record in old_y
                ),
                "old_y_min_crossing_then_return_and_success": sum(
                    record["returned_inside_old_y_min"]
                    and record["successful_after_old_y_crossing"]
                    for record in old_y
                ),
                "old_y_min_max_overshoot_m_quantiles": _quantiles([
                    record["max_old_y_min_overshoot_m"] for record in old_y
                ]),
                "turnaround_after_old_y_crossing_s_quantiles": _quantiles([
                    record["turnaround_after_old_y_crossing_s"]
                    for record in old_y
                    if record["turnaround_after_old_y_crossing_s"] is not None
                ]),
                "new_outer_y_min_crossings": int(new_faces.get("y_min", 0)),
            }

    payload = {
        "status": "DUAL_TASKSPACE_BOUNDARY_AUDIT",
        "source": str(args.trajectories.resolve()),
        "old_bounds": args.old_bounds.tolist(),
        "new_bounds": args.new_bounds.tolist(),
        "interpretation": (
            "old_boundary_crossing on a new-bound rollout is retrospective, "
            "not a counterfactual continuation of an old-bound terminated episode"
        ),
        "aggregate": aggregate,
        "episodes": episode_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output.resolve()),
        "rounds": sorted(aggregate, key=int),
        "episode_rows": len(episode_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
