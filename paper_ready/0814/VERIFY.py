#!/usr/bin/env python3
"""Verify the 0814 handoff's hashes, seeds, and frozen CFM arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

from export_flight_references import paper_rows, reconstruct_reference  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes() -> int:
    checked = 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}: {actual} != {expected}")
        checked += 1
    return checked


def check_cfm_arrays(handoff: dict) -> int:
    contract = (
        ROOT
        / "trajectories/cfmmppi/site_bank_combined_raw_trajectories.pt"
    )
    actual = {
        (
            str(row["regime"]),
            float(row["gamma"]),
            int(row["trial"]),
            int(row["rollout_seed"]),
        ): row
        for row in torch.load(contract, map_location="cpu", weights_only=False)
    }
    checked = 0
    for group in ("paper-ready-cfmmppi", "not-paper-ready-cfmmppi"):
        for row in handoff["groups"][group]:
            key = (
                row["regime"], float(row["gamma"]), int(row["episode"]),
                int(row["rollout_seed"]),
            )
            source = actual[key]
            for field in ("states", "controls", "applied_controls", "dense_steps"):
                if not np.array_equal(np.asarray(row[field]), np.asarray(source[field])):
                    raise RuntimeError(f"CFM array mismatch {key}: {field}")
            checked += 1
    paper_rows = handoff["groups"]["paper-ready-cfmmppi"]
    not_paper_rows = handoff["groups"]["not-paper-ready-cfmmppi"]
    if len(paper_rows) != 24 or {
        float(row["gamma"]) for row in paper_rows
    } != {0.1}:
        raise RuntimeError("paper-ready CFM bank is not the matched gamma 0.1 bank")
    if any(np.isclose(float(row["gamma"]), 0.1) for row in not_paper_rows):
        raise RuntimeError("old gamma 0.1 CFM rows leaked into not-paper-ready")
    if not any(np.isclose(float(row["gamma"]), 0.3) for row in not_paper_rows):
        raise RuntimeError("prior paper-ready gamma 0.3 was not preserved")
    return checked


def check_fixed_gamma_contracts(handoff: dict) -> None:
    """Enforce the four fixed-gamma paper handoff counts."""
    groups = handoff["groups"]
    for group in (
        "paper-ready-pre2", "paper-ready-less-expanded", "paper-ready-expanded"
    ):
        rows = groups[group]
        if len(rows) != 8 or {float(row["gamma"]) for row in rows} != {0.1}:
            raise RuntimeError(f"{group} must contain exactly 8 gamma 0.1 rows")

    cfm_rows = groups["paper-ready-cfmmppi"]
    regimes = {"safety", "balanced", "performance"}
    if {str(row["regime"]) for row in cfm_rows} != regimes:
        raise RuntimeError("paper-ready CFM bank is missing a required regime")
    for regime in regimes:
        rows = [row for row in cfm_rows if row["regime"] == regime]
        if len(rows) != 8 or {float(row["gamma"]) for row in rows} != {0.1}:
            raise RuntimeError(f"CFM {regime} must contain exactly 8 gamma 0.1 rows")

    safe_rows = [
        row for row in groups["paper-ready-safemppi"]
        if np.isclose(float(row["gamma"]), 0.1)
    ]
    if len(safe_rows) != 8:
        raise RuntimeError("paper-ready SafeMPPI must contain 8 gamma 0.1 rows")


def check_less_expanded_arrays(handoff: dict) -> int:
    payload = torch.load(
        ROOT / "trajectories/less_expanded/r1_only_bowling_raw_trajectories.pt",
        map_location="cpu",
        weights_only=False,
    )
    actual = {
        (float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"])): row
        for row in payload["1"]
    }
    checked = 0
    for row in handoff["groups"]["paper-ready-less-expanded"]:
        key = (float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"]))
        source = actual[key]
        for field in ("states", "controls", "applied_controls", "dense_steps"):
            if not np.array_equal(np.asarray(row[field]), np.asarray(source[field])):
                raise RuntimeError(f"Expanded R1 array mismatch {key}: {field}")
        checked += 1
    return checked


def check_safemppi_arrays(handoff: dict) -> int:
    contract = ROOT / "trajectories/safemppi/site_bank_combined_raw_trajectories.pt"
    payload = torch.load(contract, map_location="cpu", weights_only=False)
    actual = {
        (float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"])): row
        for row in payload["safemppi"]
    }
    checked = 0
    for row in handoff["groups"]["paper-ready-safemppi"]:
        key = (float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"]))
        source = actual[key]
        for field in (
            "states", "controls", "applied_controls", "dense_positions"
        ):
            if not np.array_equal(np.asarray(row[field]), np.asarray(source[field])):
                raise RuntimeError(f"SafeMPPI array mismatch {key}: {field}")
        checked += 1
    return checked


def check_flight_references(handoff: dict) -> int:
    manifest = json.loads((ROOT / "flight_references/manifest.json").read_text())
    if manifest.get("status") != "COMPLETE" or int(manifest.get("count", -1)) != 56:
        raise RuntimeError("flight-reference manifest is not complete at 56 rows")
    config = json.loads((ROOT / "config/task_config_resolved.json").read_text())
    mppi = config["safemppi"]
    roster = {(group, index): row for group, index, row in paper_rows(handoff)}
    seen: set[tuple[str, int]] = set()
    required = {
        "time_s", "position_ref", "velocity_ref", "acceleration_ref",
        "raw_controls_10hz", "applied_controls_10hz", "executed_controls_10hz",
    }
    for run in manifest["runs"]:
        key = (str(run["source_group"]), int(run["source_row"]))
        if key not in roster or key in seen:
            raise RuntimeError(f"invalid or duplicate flight-reference source {key}")
        seen.add(key)
        path = ROOT / str(run["flight_reference"])
        if sha256(path) != str(run["flight_reference_sha256"]):
            raise RuntimeError(f"flight-reference SHA mismatch: {path}")
        expected, _ = reconstruct_reference(
            roster[key],
            dt=float(mppi["dt"]),
            substeps=int(mppi["integration_substeps"]),
            max_speed=float(mppi["max_speed"]),
            max_vertical_speed=float(mppi["max_vertical_speed"]),
        )
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"flight reference {path} is missing {sorted(missing)}")
            for field in (
                "time_s", "position_ref", "velocity_ref", "acceleration_ref",
                "raw_controls_10hz", "applied_controls_10hz",
            ):
                if not np.array_equal(np.asarray(archive[field]), expected[field]):
                    raise RuntimeError(f"flight-reference array mismatch {key}: {field}")
            if not np.array_equal(
                np.asarray(archive["executed_controls_10hz"]),
                np.asarray(archive["applied_controls_10hz"]),
            ):
                raise RuntimeError(f"0806 control alias mismatch: {key}")
            count = len(archive["time_s"])
            for field in ("position_ref", "velocity_ref", "acceleration_ref"):
                if archive[field].shape != (count, 3):
                    raise RuntimeError(f"invalid {field} shape in {path}")
            if count > 1 and not np.allclose(
                np.diff(archive["time_s"]), 0.01, atol=1.0e-9, rtol=0.0
            ):
                raise RuntimeError(f"flight reference is not 100 Hz: {path}")
            if not all(np.isfinite(archive[field]).all() for field in required):
                raise RuntimeError(f"non-finite flight-reference value: {path}")
        seen.add(key)
    if seen != set(roster):
        raise RuntimeError(f"flight references do not cover the frozen roster: {seen ^ set(roster)}")
    return len(seen)


def main() -> None:
    selection = json.loads(
        (ROOT / "selections/paper_ready_bowling_selection.json").read_text()
    )
    handoff = torch.load(
        ROOT / "trajectories/paper_ready_bowling_handoff.pt",
        map_location="cpu",
        weights_only=False,
    )
    site = (ROOT / "site/visualization.inner.html").read_text()
    row_count = 0
    for group, rows in selection["groups"].items():
        if rows is None:
            raise RuntimeError(f"site group is unexpectedly empty: {group}")
        for row in rows:
            seed = int(row["seed"])
            if f'"seed":{seed}' not in site:
                raise RuntimeError(f"site is missing seed {seed} from {group}")
            row_count += 1
    if 'id="regime"' not in site or "exact rollout seed" not in site:
        raise RuntimeError("site is missing interactive regime/seed controls")
    check_fixed_gamma_contracts(handoff)
    less_expanded_count = check_less_expanded_arrays(handoff)
    cfm_count = check_cfm_arrays(handoff)
    safemppi_count = check_safemppi_arrays(handoff)
    flight_reference_count = check_flight_references(handoff)
    hash_count = check_hashes()
    print(
        f"OK: {hash_count} file hashes, {row_count} selected rows, "
        f"{less_expanded_count} exact Expanded R1 rows, {cfm_count} exact CFM rows, "
        f"{safemppi_count} exact SafeMPPI rows, "
        f"{flight_reference_count} exact 100 Hz flight references"
    )


if __name__ == "__main__":
    main()
