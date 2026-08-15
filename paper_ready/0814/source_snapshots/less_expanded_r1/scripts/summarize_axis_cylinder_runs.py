#!/usr/bin/env python3
"""Merge per-round expansion support and fixed-bank raw metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


MODE_NAMES = ("below", "above", "left", "right")


def _raw_by_round(arm: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for path in sorted(arm.rglob("raw_eval.json")):
        payload = json.loads(path.read_text())
        for round_text, summary in payload.get("summary", {}).items():
            result[int(round_text)] = summary
    return result


def _support(row: dict) -> dict:
    per_gamma = {}
    total = Counter()
    for gamma, evidence in row["successful_executed_commit_by_gamma"].items():
        counts = Counter(
            int(mode)
            for mode in evidence["success_episode_sample_update_modes"].values()
        )
        named = {
            name: int(counts.get(index, 0))
            for index, name in enumerate(MODE_NAMES)
        }
        total.update(counts)
        per_gamma[str(gamma)] = {
            "terminal_success_modes": named,
            "retry_batches_used": int(evidence["retry_batches_used"]),
            "attempted_episodes": int(evidence["attempted_episode_count"]),
            "terminal_success_episodes": int(evidence["success_episode_count"]),
        }
    return {
        "per_gamma": per_gamma,
        "pooled_terminal_success_modes": {
            name: int(total.get(index, 0))
            for index, name in enumerate(MODE_NAMES)
        },
    }


def _arm_summary(arm: Path) -> dict:
    raw = _raw_by_round(arm)
    rounds = []
    metrics = arm / "metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            round_i = int(row["round"])
            item = {
                "round": round_i,
                "candidate_support": _support(row),
                "positive_loss": float(row["positive_loss"]),
                "optimizer_step": int(row["optimizer_step"]),
                "replay_rows": int(row["replay_positive_total"]),
                "gather_s": float(row["gather_s"]),
                "update_s": float(row["update_s"]),
                "round_total_s": float(row["round_total_s"]),
            }
            if round_i in raw:
                item["raw_eval"] = raw[round_i]
            rounds.append(item)
    return {
        "arm": str(arm.resolve()),
        "rounds": rounds,
        "raw_evaluation_rounds": sorted(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True,
        help="NAME=PATH; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for spec in args.arm:
        name, separator, path_text = spec.partition("=")
        if not separator or not name or not path_text:
            parser.error("--arm must be NAME=PATH")
        arms[name] = _arm_summary(Path(path_text))
    payload = {
        "status": "AXIS_CYLINDER_ROUNDWISE_SUMMARY",
        "mode_order": list(MODE_NAMES),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
