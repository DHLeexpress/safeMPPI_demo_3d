#!/usr/bin/env python3
"""Export the approved real-scene site roster as exact 100 Hz references."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PAPER_READY = ROOT.parent
sys.path.insert(0, str(PAPER_READY / "0814/source"))

from export_flight_references import reconstruct_reference, sha256_file  # noqa: E402


GAMMA = 0.3
GROUPS = (
    "real-paper-ready-pre2",
    "real-paper-ready-less-expanded",
    "real-paper-ready-expanded",
    "real-paper-ready-cfmmppi",
    "real-paper-ready-safemppi",
)
METHODS = {
    "real-paper-ready-pre2": "PRE2",
    "real-paper-ready-less-expanded": "Expanded_R1",
    "real-paper-ready-expanded": "Expanded_S4",
    "real-paper-ready-cfmmppi": "CFM-MPPI",
    "real-paper-ready-safemppi": "SafeMPPI",
}
CHECKPOINTS = {
    "real-paper-ready-pre2": PAPER_READY / "0814/checkpoints/pre2/pretrained.pt",
    "real-paper-ready-less-expanded": (
        PAPER_READY / "0814/checkpoints/less_expanded/checkpoint_001.pt"
    ),
    "real-paper-ready-expanded": (
        PAPER_READY / "0814/checkpoints/expanded/checkpoint_004.pt"
    ),
    "real-paper-ready-cfmmppi": PAPER_READY / "0814/checkpoints/pre2/pretrained.pt",
}


def selected_rows(payload: dict) -> list[tuple[str, int, dict]]:
    rows: list[tuple[str, int, dict]] = []
    for group in GROUPS:
        values = payload["groups"][group]
        if len(values) != 8:
            raise ValueError(f"{group}: expected 8 rows, got {len(values)}")
        if {float(row["gamma"]) for row in values} != {GAMMA}:
            raise ValueError(f"{group}: expected fixed gamma={GAMMA}")
        rows.extend((group, index, row) for index, row in enumerate(values))
    if len(rows) != 40:
        raise ValueError(f"expected 40 selected rows, got {len(rows)}")
    return rows


def _slug(group: str) -> str:
    return group.removeprefix("real-paper-ready-").replace("-", "_")


def _scalar(value, default):
    return default if value is None else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = (args.output or bundle / "flight_references").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_path = bundle / "trajectories/real_selected_trajectories.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_sha = sha256_file(source_path)
    config_path = bundle / "config/task_config_resolved.json"
    config = json.loads(config_path.read_text())
    config_sha = sha256_file(config_path)
    mppi = config["safemppi"]
    index_rows: list[dict[str, object]] = []

    for group, source_row, row in selected_rows(source):
        method = METHODS[group]
        regime = str(row.get("regime") or "")
        route = str(
            row.get("stable_route")
            or (row.get("bowling_route") or {}).get("stable_code")
            or ""
        )
        reference, diagnostics = reconstruct_reference(
            row,
            dt=float(mppi["dt"]),
            substeps=int(mppi["integration_substeps"]),
            max_speed=float(mppi["max_speed"]),
            max_vertical_speed=float(mppi["max_vertical_speed"]),
        )
        checkpoint = CHECKPOINTS.get(group)
        checkpoint_sha = (
            sha256_file(checkpoint)
            if checkpoint is not None else "NOT_APPLICABLE_SAFEMPPI"
        )
        hard = row.get("hard_constraints") or {}
        status = str(row["status"])
        hard_valid = bool(hard.get("hard_valid", False))
        string_valid = bool(hard.get("string_valid", False))
        hardware_eligible = status == "SUCCESS" and hard_valid and string_valid
        reference.update({
            "view": np.str_(group),
            "method": np.str_(method),
            "regime": np.str_(regime),
            "route": np.str_(route),
            "source_handoff_sha256": np.str_(source_sha),
            "config_sha256": np.str_(config_sha),
            "checkpoint_sha256": np.str_(checkpoint_sha),
            "real_effective_margin_m": np.float32(0.16),
            "string_no_go_radius_m": np.float32(0.10),
            "effective_sphere_min_clearance_m": np.float32(
                _scalar(hard.get("effective_sphere_min_clearance_m"), np.nan)
            ),
            "string_min_clearance_m": np.float32(
                _scalar(hard.get("string_min_clearance_m"), np.nan)
            ),
            "hardware_eligible": np.bool_(hardware_eligible),
        })

        directory = output / _slug(group)
        if regime:
            directory /= regime
        directory.mkdir(parents=True, exist_ok=True)
        episode = int(reference["episode"])
        seed = int(reference["seed"])
        target = directory / (
            f"gamma_0p3_{route or 'unclassified'}_e{episode}_seed_{seed}_100hz.npz"
        )
        np.savez_compressed(target, **reference)
        try:
            relative = target.relative_to(bundle).as_posix()
        except ValueError:
            relative = target.relative_to(output).as_posix()
        index_rows.append({
            "flight_id": f"0815_{_slug(group)}_{regime or 'default'}_e{episode}_s{seed}",
            "view": group,
            "method": method,
            "regime": regime,
            "gamma": GAMMA,
            "route": route,
            "episode": episode,
            "seed": seed,
            "simulated_status": status,
            "hard_real_geometry_pass": hard_valid,
            "string_valid": string_valid,
            "hardware_eligibility": (
                "REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL"
                if hardware_eligible else "SIMULATION_ONLY_DO_NOT_FLY"
            ),
            "source_archive": source_path.relative_to(bundle).as_posix(),
            "source_archive_sha256": source_sha,
            "source_group": group,
            "source_row": source_row,
            "flight_reference": relative,
            "flight_reference_sha256": sha256_file(target),
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            **diagnostics,
        })

    fields = list(index_rows[0])
    with (output / "FLIGHT_INDEX.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    manifest = {
        "schema": "paper_ready_0815_real_bowling_100hz_references_v1",
        "status": "COMPLETE",
        "fixed_gamma": GAMMA,
        "count": len(index_rows),
        "hardware_eligible_count": sum(
            row["hardware_eligibility"]
            == "REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL"
            for row in index_rows
        ),
        "simulation_only_count": sum(
            row["hardware_eligibility"] == "SIMULATION_ONLY_DO_NOT_FLY"
            for row in index_rows
        ),
        "contract": {
            "reference_rate_hz": 100,
            "source_control_rate_hz": 10,
            "position_velocity_acceleration_are_stored": True,
            "differentiate_position_on_hardware": False,
            "interpolate_or_smooth_on_hardware": False,
            "reapply_reference_governor": False,
            "effective_margin_m": 0.16,
            "string_no_go_radius_m": 0.10,
        },
        "source_archive": source_path.relative_to(bundle).as_posix(),
        "source_archive_sha256": source_sha,
        "config_sha256": config_sha,
        "runs": index_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "count": manifest["count"],
        "hardware_eligible_count": manifest["hardware_eligible_count"],
        "simulation_only_count": manifest["simulation_only_count"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
