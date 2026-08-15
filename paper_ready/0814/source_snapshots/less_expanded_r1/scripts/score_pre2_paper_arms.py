#!/usr/bin/env python3
"""Rank fixed-bank PRE2 evaluations against intermediate and final goals."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODES = ("below", "above", "left", "right")


def _arm(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--arm must be label=raw_eval.json")
    label, path = raw.split("=", 1)
    return label, Path(path)


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite evaluation value {value!r}")
    return number


def _pooled_metric(pooled: dict, *names: str) -> float:
    for name in names:
        if name in pooled:
            return _finite(pooled[name])
    raise KeyError(f"none of the pooled metric names are present: {names}")


def _optional_pooled_metric(pooled: dict, *names: str) -> float | None:
    for name in names:
        if name in pooled and pooled[name] is not None:
            return _finite(pooled[name])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranked = []
    for raw in args.arm:
        label, path = _arm(raw)
        payload = json.loads(path.read_text())
        summary = payload["summary"][str(args.round)]
        pooled = summary["pooled"]
        rows = payload["rows"][str(args.round)]
        successes = [row for row in rows if row["status"] == "SUCCESS"]
        counts = {mode: 0 for mode in MODES}
        statuses: dict[str, int] = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        for row in successes:
            if row.get("mode") in counts:
                counts[row["mode"]] += 1
        success_count = len(successes)
        shares = {
            mode: (counts[mode] / success_count if success_count else 0.0)
            for mode in MODES
        }
        entropy = -sum(
            share * math.log(share)
            for share in shares.values() if share > 0.0
        ) / math.log(len(MODES))
        gamma_sr = {
            gamma: _finite(cell["SR"])
            for gamma, cell in summary["per_gamma"].items()
        }
        gamma_time_to_goal = {
            gamma: _optional_pooled_metric(
                cell, "avg_time_to_goal_s", "successful_time_to_goal_s",
            )
            for gamma, cell in summary["per_gamma"].items()
        }
        metrics = {
            "pooled_SR": _finite(pooled["SR"]),
            "minimum_gamma_SR": min(gamma_sr.values()),
            "collision_rate": _finite(pooled["CR"]),
            "OOB_plus_timeout": (
                _finite(pooled["OOB"]) + _finite(pooled["timeout"])
            ),
            "raw_validity": _pooled_metric(
                pooled, "raw_validity", "window_validity",
            ),
            "mean_min_clearance_m": _optional_pooled_metric(
                pooled, "avg_min_clearance_m", "successful_min_clearance_m",
            ),
            "mean_time_to_goal_s": _optional_pooled_metric(
                pooled, "avg_time_to_goal_s", "successful_time_to_goal_s",
            ),
            "route_coverage": _pooled_metric(
                pooled, "coverage", "route_coverage",
            ),
            "minimum_mode_share": min(shares.values()),
        }
        intermediate_checks = {
            "pooled_SR_ge_0.70": metrics["pooled_SR"] >= 0.70,
            "each_gamma_SR_ge_0.50": metrics["minimum_gamma_SR"] >= 0.50,
            "collision_le_0.05": metrics["collision_rate"] <= 0.05,
            "OOB_plus_timeout_le_0.10": metrics["OOB_plus_timeout"] <= 0.10,
            "validity_ge_0.90": metrics["raw_validity"] >= 0.90,
            "clearance_ge_0.10m": (
                metrics["mean_min_clearance_m"] is not None
                and metrics["mean_min_clearance_m"] >= 0.10
            ),
            "coverage_4_of_4": metrics["route_coverage"] == 1.0,
            "each_mode_share_ge_0.10": metrics["minimum_mode_share"] >= 0.10,
        }
        final_checks = {
            "pooled_SR_ge_0.95": metrics["pooled_SR"] >= 0.95,
            "each_gamma_SR_ge_0.93": metrics["minimum_gamma_SR"] >= 0.93,
            "collision_le_0.03": metrics["collision_rate"] <= 0.03,
            "OOB_plus_timeout_le_0.03": metrics["OOB_plus_timeout"] <= 0.03,
            "validity_ge_0.95": metrics["raw_validity"] >= 0.95,
            "coverage_4_of_4": metrics["route_coverage"] == 1.0,
        }
        intermediate_passed = sum(intermediate_checks.values())
        final_passed = sum(final_checks.values())
        uniform_l1 = sum(abs(share - 0.25) for share in shares.values())
        low_gamma = min(gamma_time_to_goal, key=float)
        high_gamma = max(gamma_time_to_goal, key=float)
        low_time = gamma_time_to_goal[low_gamma]
        high_time = gamma_time_to_goal[high_gamma]
        ranked.append({
            "label": label,
            "raw_eval": str(path.resolve()),
            "round": int(args.round),
            # Compatibility fields intentionally mean the true final stop rule.
            "qualified": all(final_checks.values()),
            "close": all(intermediate_checks.values()),
            "intermediate_qualified": all(intermediate_checks.values()),
            "final_qualified": all(final_checks.values()),
            "checks_passed": final_passed,
            "checks_total": len(final_checks),
            "checks": final_checks,
            "intermediate_checks_passed": intermediate_passed,
            "intermediate_checks_total": len(intermediate_checks),
            "intermediate_checks": intermediate_checks,
            "final_checks_passed": final_passed,
            "final_checks_total": len(final_checks),
            "final_checks": final_checks,
            "metrics": metrics,
            "gamma_SR": gamma_sr,
            "gamma_time_to_goal_s": gamma_time_to_goal,
            "gamma_time_trend": {
                "lowest_gamma": low_gamma,
                "highest_gamma": high_gamma,
                "highest_gamma_faster_than_lowest": (
                    high_time <= low_time
                    if high_time is not None and low_time is not None else None
                ),
            },
            "status_counts": statuses,
            "route_counts": counts,
            "route_shares": shares,
            "normalized_route_entropy": entropy,
            "route_uniformity_l1_from_quarter": uniform_l1,
        })

    ranked.sort(key=lambda row: (
        row["final_qualified"],
        row["intermediate_qualified"],
        row["final_checks_passed"],
        row["intermediate_checks_passed"],
        row["metrics"]["route_coverage"],
        row["normalized_route_entropy"],
        row["metrics"]["minimum_mode_share"],
        row["metrics"]["pooled_SR"],
        row["metrics"]["raw_validity"],
        -row["metrics"]["collision_rate"],
        -row["metrics"]["OOB_plus_timeout"],
        row["gamma_time_trend"]["highest_gamma_faster_than_lowest"] is True,
    ), reverse=True)
    output = {
        "status": "PRE2_PAPER_ARM_RANKING",
        "round": int(args.round),
        "intermediate_winner_rule": {
            "pooled_SR_min": 0.70,
            "each_gamma_SR_min": 0.50,
            "collision_max": 0.05,
            "OOB_plus_timeout_max": 0.10,
            "raw_validity_min": 0.90,
            "mean_min_clearance_m_min": 0.10,
            "route_coverage": "4/4",
            "each_mode_share_min": 0.10,
            "stops_expansion": False,
        },
        "approved_stop_rule": {
            "pooled_SR_min": 0.95,
            "each_gamma_SR_min": 0.93,
            "collision_max": 0.03,
            "OOB_plus_timeout_max": 0.03,
            "raw_validity_min": 0.95,
            "route_coverage": "4/4",
            "absolute_time_to_goal_constraint": None,
            "stops_expansion": True,
        },
        "ranking_preferences_not_stop_conditions": {
            "route_balance": "maximize entropy and minimize distance from 25% per mode",
            "time": "report gamma-wise trend; prefer higher gamma to be faster",
        },
        "intermediate_winners": [
            row["label"] for row in ranked if row["intermediate_qualified"]
        ],
        "winner": (
            ranked[0]["label"]
            if ranked and ranked[0]["final_qualified"] else None
        ),
        "best": ranked[0]["label"] if ranked else None,
        "ranking": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
