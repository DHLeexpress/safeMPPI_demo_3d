#!/usr/bin/env python3
"""Verify the 0814 handoff's hashes, seeds, and frozen CFM arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


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
    hash_count = check_hashes()
    print(
        f"OK: {hash_count} file hashes, {row_count} selected rows, "
        f"{less_expanded_count} exact Expanded R1 rows, {cfm_count} exact CFM rows, "
        f"{safemppi_count} exact SafeMPPI rows"
    )


if __name__ == "__main__":
    main()
