"""Recover completed-round replay data from a fail-closed quota expansion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=Path, required=True)
    args = parser.parse_args()
    root = args.expansion
    if not (root / "FAILED.json").is_file():
        raise FileNotFoundError("recovery requires FAILED.json")
    if (root / "manifest.json").exists() or (root / "query_archive.pt").exists():
        raise FileExistsError("refusing to overwrite an existing recovered output")

    rows = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("no completed rounds are available to recover")
    completed = [int(row["round"]) for row in rows]
    if completed != list(range(1, len(rows) + 1)):
        raise ValueError(f"completed rounds are not contiguous: {completed}")
    checkpoint = torch.load(
        root / f"checkpoint_{completed[-1]:03d}.pt",
        map_location="cpu", weights_only=False,
    )
    archive = []
    for round_i in completed:
        block = torch.load(
            root / f"query_archive_round_{round_i:03d}.pt",
            map_location="cpu", weights_only=False,
        )
        if any(int(record.round) != round_i for record in block):
            raise ValueError(f"round-{round_i} archive contains another round")
        archive.extend(block)
    torch.save(archive, root / "query_archive.pt")
    failure = json.loads((root / "FAILED.json").read_text())
    manifest = {
        "status": "FAILED_CLOSED_COMPLETED_ROUNDS_RECOVERED",
        "config": checkpoint["config"],
        "rounds": rows,
        "completed_rounds": completed,
        "failed_attempt": failure,
        "D": len(archive),
        "D_plus": sum(record.verification.valid for record in archive),
        "D_replay_accepted": sum(record.replay_eligible for record in archive),
        "recovery_contract": (
            "Only atomically saved completed-round committed replay rows are "
            "included; every row from the uncommitted failed round is excluded."
        ),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
