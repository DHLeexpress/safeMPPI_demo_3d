#!/usr/bin/env python3
"""Convert a complete native lab raw evaluation to paper-curve JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_se(values: Iterable[float]) -> tuple[float | None, float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    count = len(finite)
    if not count:
        return None, 0.0, 0
    mean = sum(finite) / count
    if count == 1:
        return mean, 0.0, count
    variance = sum((value - mean) ** 2 for value in finite) / (count - 1)
    return mean, math.sqrt(variance / count), count


def _cell(round_i: int, gamma: float, rows: list[dict]) -> dict:
    selected = [
        row for row in rows
        if math.isclose(float(row["gamma"]), gamma, rel_tol=0.0, abs_tol=1e-12)
    ]
    if not selected:
        raise ValueError(f"round {round_i} gamma {gamma:g} has no rows")
    episodes = len(selected)
    successes = [row for row in selected if row["status"] == "SUCCESS"]
    collision = _mean_se(row["status"] == "COLLISION" for row in selected)
    validity = _mean_se(row["window_validity"] for row in selected)
    clearance = _mean_se(
        row["min_clearance_m"]
        for row in successes
        if row.get("min_clearance_m") is not None
    )
    time_to_goal = _mean_se(
        row["time_to_goal_s"]
        for row in successes
        if row.get("time_to_goal_s") is not None
    )
    modes = {
        name: sum(row.get("mode") == name for row in successes)
        for name in ("below", "above", "left", "right")
    }

    def metric(statistic: tuple[float | None, float, int]) -> dict:
        mean, se, count = statistic
        return {"mean": mean, "se": se, "n": count}

    return {
        "round": round_i,
        "gamma": gamma,
        "m": episodes,
        "temp": 1.0,
        "SR": len(successes) / episodes,
        "OOB": sum(row["status"] == "OOB" for row in selected) / episodes,
        "timeout": sum(row["status"] == "TIMEOUT" for row in selected) / episodes,
        "CR": metric(collision),
        "v_safe": metric(validity),
        "clearance": metric(clearance),
        "time": metric(time_to_goal),
        "successful_route_counts": modes,
    }


def convert(raw_eval: Path, output: Path) -> list[dict]:
    payload = json.loads(raw_eval.read_text())
    if payload.get("status") != "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE":
        raise ValueError("input is not a completed lab native-raw evaluation")
    if payload.get("sampling_temperature") != 1.0:
        raise ValueError("input sampling temperature is not 1.0")
    if payload.get("sigma_tilt_used") is not False:
        raise ValueError("evaluation must not use acquisition tilting")
    rows_by_round = payload.get("rows")
    if not isinstance(rows_by_round, dict) or not rows_by_round:
        raise ValueError("input contains no per-episode rows")
    rounds = sorted(map(int, rows_by_round))
    if rounds != list(range(rounds[-1] + 1)):
        raise ValueError(f"evaluation rounds are not contiguous from r0: {rounds}")
    gammas = sorted({
        float(row["gamma"])
        for rows in rows_by_round.values()
        for row in rows
    })
    if not gammas:
        raise ValueError("input contains no gamma values")
    cells = [
        _cell(round_i, gamma, rows_by_round[str(round_i)])
        for round_i in rounds
        for gamma in gammas
    ]
    episode_counts = {cell["m"] for cell in cells}
    if len(episode_counts) != 1:
        raise ValueError(f"round-gamma episode counts differ: {episode_counts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(
        json.dumps(cell, sort_keys=True, allow_nan=False) + "\n"
        for cell in cells
    ))
    provenance = {
        "status": "LAB_NATIVE_RAW_PAPER_JSONL_COMPLETE",
        "input": str(raw_eval.resolve()),
        "input_sha256": _sha256(raw_eval),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "rounds": rounds,
        "gammas": gammas,
        "episodes_per_gamma_round": next(iter(episode_counts)),
        "selection": "all preregistered rows; no cherry-picking or smoothing",
        "metrics": {
            "CR": "physical COLLISION status fraction",
            "v_safe": "mean executed-window validity across all episodes",
            "clearance": "successful-trajectory minimum clearance mean and SE",
            "time": "successful-trajectory time-to-goal mean and SE",
        },
        "renderer_intervals": (
            "95% Wilson for CR and validity; mean +/- 1.96 SE for clearance/TtG"
        ),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = convert(args.raw_eval.resolve(), args.output.resolve())
    print(
        f"wrote {args.output.resolve()} with {len(cells)} cells "
        f"({len({cell['round'] for cell in cells})} rounds)"
    )


if __name__ == "__main__":
    main()
