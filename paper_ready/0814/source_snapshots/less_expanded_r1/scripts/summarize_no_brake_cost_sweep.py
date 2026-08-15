#!/usr/bin/env python3
"""Summarize paired no-brake expansion progress and fixed-bank stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
DEFAULT_STAGE = ROOT / (
    "results/stage1_single_ball_t128/0810_pre2_no_brake_cost_sweep"
)
GAMMAS = ("0.1", "0.3", "0.5", "1")


def _trend(values: list[float], increasing: bool) -> float:
    if len(values) < 2:
        return 0.0
    hits = sum(
        (right >= left if increasing else right <= left)
        for left, right in zip(values, values[1:])
    )
    return hits / (len(values) - 1)


def _gamma_trend(round_summary: dict) -> dict:
    per_gamma = round_summary["per_gamma"]
    low = [per_gamma[gamma] for gamma in GAMMAS[:2]]
    high = [per_gamma[gamma] for gamma in GAMMAS[2:]]
    low_ttg = sum(row["successful_time_to_goal_s"] for row in low) / 2
    high_ttg = sum(row["successful_time_to_goal_s"] for row in high) / 2
    low_clearance = sum(
        row["successful_min_clearance_m"] for row in low
    ) / 2
    high_clearance = sum(
        row["successful_min_clearance_m"] for row in high
    ) / 2
    return {
        "low_gamma_ttg_s": low_ttg,
        "high_gamma_ttg_s": high_ttg,
        "ttg_high_is_lower": high_ttg < low_ttg,
        "low_gamma_clearance_m": low_clearance,
        "high_gamma_clearance_m": high_clearance,
        "clearance_high_is_lower": high_clearance < low_clearance,
    }


def _evaluate(path: Path) -> dict | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    summaries = raw["summary"]
    rounds = sorted(summaries, key=int)
    pooled = [summaries[round_index]["pooled"] for round_index in rounds]
    gamma = {
        round_index: _gamma_trend(summaries[round_index])
        for round_index in rounds
    }
    endpoint = pooled[-1]
    return {
        "status": raw.get("status"),
        "rounds": [int(round_index) for round_index in rounds],
        "roundwise": {
            round_index: {
                "pooled": summaries[round_index]["pooled"],
                "per_gamma": summaries[round_index]["per_gamma"],
                "gamma_trend": gamma[round_index],
            }
            for round_index in rounds
        },
        "stability": {
            "sr_nondecreasing_fraction": _trend(
                [row["SR"] for row in pooled], True
            ),
            "cr_nonincreasing_fraction": _trend(
                [row["CR"] for row in pooled], False
            ),
            "oob_nonincreasing_fraction": _trend(
                [row["OOB"] for row in pooled], False
            ),
            "validity_nondecreasing_fraction": _trend(
                [row["window_validity"] for row in pooled], True
            ),
            "gamma_ttg_rounds_preserved": sum(
                row["ttg_high_is_lower"] for row in gamma.values()
            ),
            "gamma_clearance_rounds_preserved": sum(
                row["clearance_high_is_lower"] for row in gamma.values()
            ),
            "both_gamma_trends_rounds_preserved": sum(
                row["ttg_high_is_lower"] and row["clearance_high_is_lower"]
                for row in gamma.values()
            ),
        },
        "r5": endpoint,
        "r5_oob_zero": endpoint["OOB"] == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    sweep = json.loads((args.stage / "NO_BRAKE_COST_SWEEP.json").read_text())
    arms = []
    for record in sweep["arms"]:
        output = Path(record["output"])
        checkpoints = sorted(output.glob("checkpoint_*.pt"))
        metrics_path = output / "metrics.jsonl"
        metrics = []
        if metrics_path.is_file():
            metrics = [
                json.loads(line) for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
        evaluation = _evaluate(
            output / "fixed_eval_r000_r005/raw_eval.json"
        )
        arms.append({
            **record,
            "committed_round": (
                int(checkpoints[-1].stem.split("_")[-1])
                if checkpoints else None
            ),
            "last_training_row": metrics[-1] if metrics else None,
            "failed": (output / "FAILED.json").is_file(),
            "evaluation": evaluation,
        })
    complete = [arm for arm in arms if arm["evaluation"] is not None]
    ranking = sorted(
        complete,
        key=lambda arm: (
            arm["evaluation"]["r5"]["OOB"],
            -arm["evaluation"]["stability"]["oob_nonincreasing_fraction"],
            -arm["evaluation"]["r5"]["SR"],
            arm["evaluation"]["r5"]["CR"],
            -arm["evaluation"]["r5"]["window_validity"],
            -arm["evaluation"]["stability"][
                "both_gamma_trends_rounds_preserved"
            ],
        ),
    )
    output = {
        "status": (
            "COMPLETE" if len(complete) == len(arms) else "RUNNING"
        ),
        "arms": arms,
        "ranking": [arm["name"] for arm in ranking],
        "provisional_winner": ranking[0]["name"] if ranking else None,
    }
    if args.write:
        (args.stage / "CURRENT_PROGRESS.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
    print(json.dumps({
        "status": output["status"],
        "arms": [
            {
                "name": arm["name"],
                "round": arm["committed_round"],
                "failed": arm["failed"],
                "r5": None if arm["evaluation"] is None else {
                    key: arm["evaluation"]["r5"][key]
                    for key in ("SR", "CR", "OOB", "timeout", "window_validity")
                },
            }
            for arm in arms
        ],
        "ranking": output["ranking"],
    }, indent=2))


if __name__ == "__main__":
    main()
