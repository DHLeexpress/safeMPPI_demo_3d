#!/usr/bin/env python3
"""Create the human-readable group/gamma/trial/seed index for the site."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


BUNDLE = Path(__file__).resolve().parents[1]


def main() -> None:
    handoff = torch.load(
        BUNDLE / "trajectories/paper_ready_bowling_handoff.pt",
        map_location="cpu",
        weights_only=False,
    )
    rows = []
    for group, values in handoff["groups"].items():
        if values is None:
            continue
        for index, row in enumerate(values):
            route = row.get("bowling_route") or {}
            rows.append({
                "group": group,
                "method": (
                    "CFM-MPPI" if "cfmmppi" in group else
                    "SafeMPPI" if "safemppi" in group else
                    "PRE2" if "pre2" in group else "Expanded"
                ),
                "regime": row.get("regime", ""),
                "gamma": row["gamma"],
                "episode": row.get("episode", row.get("trial")),
                "seed": row["rollout_seed"],
                "status": row["status"],
                "route": route.get("stable_code", ""),
                "source": "trajectories/paper_ready_bowling_handoff.pt",
                "row": index,
            })
    legacy = json.loads(
        (BUNDLE / "trajectories/legacy_pre2_s4_site_rows.json").read_text()
    )
    for key, group in (("pre2", "legacy-pre2"), ("s4", "legacy-s4")):
        for index, row in enumerate(legacy[key]):
            rows.append({
                "group": group,
                "method": row["model"],
                "regime": "",
                "gamma": row["gamma"],
                "episode": row["episode"],
                "seed": row["seed"],
                "status": row["status"],
                "route": row.get("route", ""),
                "source": "trajectories/legacy_pre2_s4_site_rows.json",
                "row": index,
            })
        if key == "s4":
            for index, row in enumerate(item for item in legacy[key] if item.get("curated")):
                rows.append({
                    "group": "legacy-s4-distinct",
                    "method": row["model"],
                    "regime": "",
                    "gamma": row["gamma"],
                    "episode": row["episode"],
                    "seed": row["seed"],
                    "status": row["status"],
                    "route": row.get("route", ""),
                    "source": "trajectories/legacy_pre2_s4_site_rows.json",
                    "row": index,
                })
    output = BUNDLE / "SITE_TRAJECTORY_INDEX.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} indexed site rows to {output}")


if __name__ == "__main__":
    main()
