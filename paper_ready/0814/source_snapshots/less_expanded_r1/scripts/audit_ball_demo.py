"""Audit the declared below-equator property of the checked-in ball demonstrations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("examples/ball_biased_demo"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = json.loads((args.run / "resolved_config.json").read_text())
    sphere = config["obstacles"]["spheres"][0]
    center_x, _, plane_z, radius = map(float, sphere)
    half_width = radius + 0.146  # declared 1.1 <= x <= 1.9 audit band.
    per_gamma = {}
    maxima = []
    for gamma in manifest["gammas"]:
        values = []
        for row in manifest["runs"]:
            if abs(float(row["gamma"]) - float(gamma)) > 1.0e-9:
                continue
            states = np.load(args.run / row["file"])["states"][:, :3]
            near = states[np.abs(states[:, 0] - center_x) <= half_width]
            values.append(float(near[:, 2].max()))
        per_gamma[f"{gamma:g}"] = {
            "episodes": len(values),
            "maximum_z_near_ball_m": max(values),
            "all_below_equator": bool(max(values) < plane_z),
        }
        maxima.extend(values)
    payload = {
        "status": "BALL_DEMO_AUDIT_COMPLETE",
        "audit_x_interval_m": [center_x - half_width, center_x + half_width],
        "equator_z_m": plane_z,
        "maximum_z_near_ball_m": max(maxima),
        "all_trajectories_below_equator_near_ball": bool(max(maxima) < plane_z),
        "per_gamma": per_gamma,
    }
    output = args.output or args.run / "ball_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
