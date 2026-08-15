#!/usr/bin/env python3
"""Verify the 0815 as-built bowling handoff and all 40 flight references."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

from export_real_flight_references import (  # noqa: E402
    GROUPS,
    selected_rows,
    sha256_file,
)
sys.path.insert(0, str(ROOT.parent / "0814/source"))
from export_flight_references import reconstruct_reference  # noqa: E402


def check_hashes() -> int:
    checked = 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def route_counts(rows: list[dict]) -> Counter:
    return Counter(
        str(
            row.get("stable_route")
            or (row.get("bowling_route") or {}).get("stable_code")
            or ""
        )
        for row in rows
    )


def check_roster(payload: dict) -> None:
    groups = payload["groups"]
    expected = {
        "real-paper-ready-pre2": Counter({"LLL": 6, "RRR": 2}),
        "real-paper-ready-less-expanded": Counter(
            {"LLL": 2, "LLR": 2, "RLL": 2, "RRR": 2}
        ),
        "real-paper-ready-safemppi": Counter(
            {"LLL": 2, "LLR": 2, "RRL": 2, "RRR": 2}
        ),
    }
    for group, counts in expected.items():
        if route_counts(groups[group]) != counts:
            raise RuntimeError(f"unexpected route roster for {group}")
    expanded_seeds = {
        int(row["rollout_seed"])
        for row in groups["real-paper-ready-expanded"]
    }
    if 814321091 in expanded_seeds or 814323086 not in expanded_seeds:
        raise RuntimeError("Expanded LLR replacement is not frozen correctly")
    for group in GROUPS:
        rows = groups[group]
        if len(rows) != 8 or {float(row["gamma"]) for row in rows} != {0.3}:
            raise RuntimeError(f"{group} is not an eight-row gamma=0.3 roster")
    cfm = groups["real-paper-ready-cfmmppi"]
    if Counter(row["status"] for row in cfm) != Counter(
        {"SUCCESS": 6, "COLLISION": 2}
    ):
        raise RuntimeError("CFM site roster status contract changed")
    for group in GROUPS:
        for row in groups[group]:
            hard = row.get("hard_constraints") or {}
            if not bool(hard.get("string_valid", False)):
                raise RuntimeError(f"string-invalid row leaked into {group}")
            if row["status"] == "SUCCESS" and not bool(hard.get("hard_valid", False)):
                raise RuntimeError(f"geometry-invalid success leaked into {group}")
            if group != "real-paper-ready-cfmmppi":
                if not bool(row["paper_quality"]["hard_goal_progress_pass"]):
                    raise RuntimeError(f"negative goal progress leaked into {group}")


def check_references(payload: dict) -> int:
    manifest = json.loads((ROOT / "flight_references/manifest.json").read_text())
    if manifest["count"] != 40 or manifest["hardware_eligible_count"] != 38:
        raise RuntimeError("flight-reference count contract changed")
    if manifest["simulation_only_count"] != 2:
        raise RuntimeError("expected exactly two simulation-only CFM rows")
    config = json.loads((ROOT / "config/task_config_resolved.json").read_text())
    mppi = config["safemppi"]
    roster = {(group, index): row for group, index, row in selected_rows(payload)}
    seen: set[tuple[str, int]] = set()
    required = {
        "time_s", "position_ref", "velocity_ref", "acceleration_ref",
        "raw_controls_10hz", "applied_controls_10hz",
        "executed_controls_10hz", "hardware_eligible",
    }
    for record in manifest["runs"]:
        key = (str(record["source_group"]), int(record["source_row"]))
        if key not in roster or key in seen:
            raise RuntimeError(f"invalid/duplicate source row: {key}")
        path = ROOT / str(record["flight_reference"])
        if sha256_file(path) != record["flight_reference_sha256"]:
            raise RuntimeError(f"reference hash mismatch: {path}")
        expected, _ = reconstruct_reference(
            roster[key],
            dt=float(mppi["dt"]),
            substeps=int(mppi["integration_substeps"]),
            max_speed=float(mppi["max_speed"]),
            max_vertical_speed=float(mppi["max_vertical_speed"]),
        )
        with np.load(path, allow_pickle=False) as archive:
            if required.difference(archive.files):
                raise RuntimeError(f"reference fields missing: {path}")
            for field in (
                "time_s", "position_ref", "velocity_ref", "acceleration_ref",
                "raw_controls_10hz", "applied_controls_10hz",
            ):
                if not np.array_equal(np.asarray(archive[field]), expected[field]):
                    raise RuntimeError(f"reference array mismatch {key}: {field}")
            if not np.array_equal(
                archive["executed_controls_10hz"], archive["applied_controls_10hz"]
            ):
                raise RuntimeError(f"executed-control alias mismatch: {path}")
            if len(archive["time_s"]) > 1 and not np.allclose(
                np.diff(archive["time_s"]), 0.01, atol=1e-9, rtol=0.0
            ):
                raise RuntimeError(f"reference is not 100 Hz: {path}")
            eligible = bool(archive["hardware_eligible"].item())
            if eligible != (
                record["hardware_eligibility"]
                == "REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL"
            ):
                raise RuntimeError(f"hardware eligibility mismatch: {path}")
        seen.add(key)
    if seen != set(roster):
        raise RuntimeError("flight references do not cover the selected roster")
    site = (ROOT / "site/visualization.html").read_text()
    for _, _, row in selected_rows(payload):
        if str(int(row["rollout_seed"])) not in site:
            raise RuntimeError(f"site is missing seed {row['rollout_seed']}")
    return len(seen)


def main() -> None:
    payload = torch.load(
        ROOT / "trajectories/real_selected_trajectories.pt",
        map_location="cpu",
        weights_only=False,
    )
    check_roster(payload)
    references = check_references(payload)
    hashes = check_hashes()
    print(
        f"OK: {hashes} file hashes, 5 views, {references} exact 100 Hz "
        "references, 38 hardware-eligible, 2 simulation-only"
    )


if __name__ == "__main__":
    main()
