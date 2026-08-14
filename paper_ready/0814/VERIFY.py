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
    contracts = {
        "safety": ROOT / "trajectories/cfmmppi/safety_raw_trajectories.pt",
        "reward": ROOT / "trajectories/cfmmppi/reward_raw_trajectories.pt",
        "balanced": ROOT / "trajectories/cfmmppi/balanced_raw_trajectories.pt",
    }
    actual = {
        (alias, float(row["gamma"]), int(row["trial"]), int(row["rollout_seed"])): row
        for alias, path in contracts.items()
        for row in torch.load(path, map_location="cpu", weights_only=False)
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
    cfm_count = check_cfm_arrays(handoff)
    hash_count = check_hashes()
    print(
        f"OK: {hash_count} file hashes, {row_count} selected rows, "
        f"{cfm_count} exact CFM rows"
    )


if __name__ == "__main__":
    main()
