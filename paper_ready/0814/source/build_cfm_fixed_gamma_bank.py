#!/usr/bin/env python3
"""Replace only CFM gamma 0.1 while preserving the older non-paper gammas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


TARGET_GAMMA = 0.1
REGIME_ALIASES = {
    "safety_alpha05": "safety",
    "balanced_alpha05": "balanced",
    "reward_dominant": "performance",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of rollout rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-gamma01", type=Path, required=True)
    parser.add_argument("--prior-safety", type=Path, required=True)
    parser.add_argument("--prior-balanced", type=Path, required=True)
    parser.add_argument("--prior-performance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite the merged bank or manifest")

    fresh = _load(args.fresh_gamma01)
    expected_regimes = {"safety", "balanced", "performance"}
    if {str(row["regime"]) for row in fresh} != expected_regimes:
        raise ValueError("fresh bank must contain safety, balanced, and performance")
    if any(not np.isclose(float(row["gamma"]), TARGET_GAMMA) for row in fresh):
        raise ValueError("fresh bank must contain only gamma 0.1")
    for regime in expected_regimes:
        selected = [row for row in fresh if row["regime"] == regime]
        if len(selected) != 8 or {int(row["trial"]) for row in selected} != set(range(8)):
            raise ValueError(f"fresh {regime} bank is not the exact 8-trial contract")
    seed_sets = {
        regime: {int(row["rollout_seed"]) for row in fresh if row["regime"] == regime}
        for regime in expected_regimes
    }
    if len({tuple(sorted(value)) for value in seed_sets.values()}) != 1:
        raise ValueError("fresh regimes do not share an identical seed bank")

    merged = [dict(row) for row in fresh]
    prior_paths = (args.prior_safety, args.prior_balanced, args.prior_performance)
    for path in prior_paths:
        for source in _load(path):
            if np.isclose(float(source["gamma"]), TARGET_GAMMA):
                continue
            row = dict(source)
            row["regime"] = REGIME_ALIASES.get(str(row["regime"]), str(row["regime"]))
            merged.append(row)
    merged.sort(key=lambda row: (
        str(row["regime"]), float(row["gamma"]), int(row["trial"]),
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    manifest = {
        "schema": "cfmmppi_fixed_gamma01_site_bank_v1",
        "paper_ready_gamma": TARGET_GAMMA,
        "replacement_scope": "only gamma 0.1; prior gamma 0.3/0.5/1.0 preserved",
        "matched_fresh_seed_bank": [
            int(seed) for seed in sorted(next(iter(seed_sets.values())))
        ],
        "counts": {
            regime: {
                str(gamma): int(sum(
                    row["regime"] == regime and np.isclose(float(row["gamma"]), gamma)
                    for row in merged
                ))
                for gamma in (0.1, 0.3, 0.5, 1.0)
            }
            for regime in sorted(expected_regimes)
        },
        "input_sha256": {
            "fresh_gamma01": _sha256(args.fresh_gamma01),
            "prior_safety": _sha256(args.prior_safety),
            "prior_balanced": _sha256(args.prior_balanced),
            "prior_performance": _sha256(args.prior_performance),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
