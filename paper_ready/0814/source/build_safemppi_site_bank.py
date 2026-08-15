#!/usr/bin/env python3
"""Merge an exact-source SafeMPPI gamma-0.1 extension into the site bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload.get("safemppi") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a safemppi trajectory list")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite the merged bank or manifest")

    base = _rows(args.base)
    extension = _rows(args.extension)
    if not extension or any(
        not np.isclose(float(row["gamma"]), 0.1) for row in extension
    ):
        raise ValueError("extension must contain only gamma 0.1 trajectories")
    keys = {
        (float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"]))
        for row in base
    }
    for row in extension:
        key = (
            float(row["gamma"]),
            int(row["episode"]),
            int(row["rollout_seed"]),
        )
        if key in keys:
            raise ValueError(f"duplicate trajectory key: {key}")
        keys.add(key)

    merged = [*base, *extension]
    merged.sort(key=lambda row: (
        float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"]),
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"safemppi": merged}, args.output)
    manifest = {
        "schema": "safemppi_gamma01_extended_site_bank_v1",
        "base_sha256": _sha256(args.base),
        "extension_sha256": _sha256(args.extension),
        "output_sha256": _sha256(args.output),
        "base_rows": len(base),
        "extension_rows": len(extension),
        "combined_rows": len(merged),
        "gamma01_rows": sum(
            np.isclose(float(row["gamma"]), 0.1) for row in merged
        ),
        "extension_episode_range": [
            min(int(row["episode"]) for row in extension),
            max(int(row["episode"]) for row in extension),
        ],
        "extension_seed_range": [
            min(int(row["rollout_seed"]) for row in extension),
            max(int(row["rollout_seed"]) for row in extension),
        ],
    }
    manifest = {
        key: int(value) if isinstance(value, np.integer) else value
        for key, value in manifest.items()
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
