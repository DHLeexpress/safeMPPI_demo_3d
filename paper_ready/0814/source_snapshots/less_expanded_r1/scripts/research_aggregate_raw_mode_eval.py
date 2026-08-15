"""Aggregate pooled raw-evaluation mode metrics across independent seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODES = ("below", "above", "left", "right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval", type=Path, action="append", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    entries = []
    totals = {mode: 0 for mode in MODES}
    episodes = successes = 0
    weighted = {key: 0.0 for key in ("CR", "OOB", "timeout")}
    nfe_values = set()
    for path in args.raw_eval:
        payload = json.loads(path.read_text())
        summary = payload["summary"][str(args.round)]["pooled"]
        count = int(summary["episodes"])
        success_count = int(sum(summary["route_counts"].values()))
        episodes += count
        successes += success_count
        for mode in MODES:
            totals[mode] += int(summary["route_counts"][mode])
        for key in weighted:
            weighted[key] += float(summary[key]) * count
        nfe_values.add(payload.get("flow_nfe_override"))
        entries.append({
            "raw_eval": str(path.resolve()),
            "episodes": count,
            "SR": float(summary["SR"]),
            "route_counts": summary["route_counts"],
            "route_coverage": float(summary["route_coverage"]),
            "minimum_successful_mode_share": float(
                summary["minimum_successful_mode_share"]
            ),
            "OOB_plus_timeout": float(summary["OOB"] + summary["timeout"]),
        })
    shares = {
        mode: (totals[mode] / successes if successes else 0.0)
        for mode in MODES
    }
    entropy = -sum(
        share * math.log(share) for share in shares.values() if share > 0.0
    ) / math.log(len(MODES))
    output = {
        "round": int(args.round),
        "flow_nfe_override_values": sorted(
            nfe_values, key=lambda value: -1 if value is None else value,
        ),
        "seed_evaluations": entries,
        "aggregate": {
            "episodes": episodes,
            "successes": successes,
            "SR": successes / episodes,
            "CR": weighted["CR"] / episodes,
            "OOB": weighted["OOB"] / episodes,
            "timeout": weighted["timeout"] / episodes,
            "route_counts": totals,
            "route_shares_among_success": shares,
            "route_coverage": sum(value > 0 for value in totals.values()) / 4,
            "minimum_successful_mode_share": min(shares.values()),
            "normalized_route_entropy": entropy,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
