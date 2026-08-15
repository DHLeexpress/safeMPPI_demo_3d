#!/usr/bin/env python3
"""Score completed 0812 margin-hybrid arms and render the best raw curves.

The scorer is deliberately read-only with respect to expansion/evaluation
artifacts.  It discovers local mirrors using ``QUEUE.json`` provenance (with a
directory scan fallback), reads only complete r0..rN ``raw_eval.json`` files,
and atomically publishes stage-level summary artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / (
    "results/stage1_single_ball_t128/"
    "0812_pre2_margin_hybrid_spooler_sweep"
)
MODES = ("below", "above", "left", "right")
RATE_METRICS = ("SR", "CR", "OOB", "validity")
PLOT_METRICS = (
    ("SR", "Success rate", None),
    ("CR", "Collision rate", None),
    ("OOB", "Out-of-bounds rate", None),
    ("validity", "Raw validity", None),
    ("clearance_m", "Successful min clearance", "m"),
    ("time_to_goal_s", "Successful time to goal", "s"),
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite metric {value!r}")
    return number


def _optional_metric(cell: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = cell.get(name)
        if value is not None:
            return _finite(value)
    return None


def _required_metric(cell: dict[str, Any], *names: str) -> float:
    value = _optional_metric(cell, *names)
    if value is None:
        raise KeyError(f"missing required metric aliases {names}")
    return value


def _cell_metrics(cell: dict[str, Any]) -> dict[str, float | None]:
    coverage = _required_metric(cell, "route_coverage", "coverage")
    if coverage > 1.0:
        coverage /= len(MODES)
    return {
        "SR": _required_metric(cell, "SR"),
        "CR": _required_metric(cell, "CR"),
        "OOB": _required_metric(cell, "OOB"),
        "timeout": _optional_metric(cell, "timeout") or 0.0,
        "validity": _required_metric(cell, "raw_validity", "window_validity"),
        "clearance_m": _optional_metric(
            cell, "avg_min_clearance_m", "successful_min_clearance_m",
        ),
        "time_to_goal_s": _optional_metric(
            cell, "avg_time_to_goal_s", "successful_time_to_goal_s",
        ),
        "coverage": coverage,
    }


def _route_diagnostics(
    summary_cell: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_counts = summary_cell.get("route_counts")
    if isinstance(summary_counts, dict):
        counts = {mode: int(summary_counts.get(mode, 0)) for mode in MODES}
    else:
        counter = Counter(
            str(row.get("mode"))
            for row in rows if row.get("status") == "SUCCESS"
        )
        counts = {mode: int(counter[mode]) for mode in MODES}
    total = sum(counts.values())
    shares = {
        mode: counts[mode] / total if total else 0.0 for mode in MODES
    }
    entropy = -sum(
        share * math.log(share)
        for share in shares.values() if share > 0.0
    ) / math.log(len(MODES))
    return {
        "counts": counts,
        "shares": shares,
        "successful_trajectories": total,
        "covered_modes": sum(count > 0 for count in counts.values()),
        "coverage_fraction": sum(count > 0 for count in counts.values()) / 4.0,
        "normalized_entropy": entropy,
        "minimum_mode_share": min(shares.values()),
        "L1_from_uniform": sum(abs(share - 0.25) for share in shares.values()),
    }


def _trend(
    rounds: list[int], values: list[float | None], *, absolute_tolerance: float,
) -> dict[str, Any]:
    points = [
        (round_i, float(value))
        for round_i, value in zip(rounds, values) if value is not None
    ]
    if len(points) < 2:
        return {
            "rounds": [point[0] for point in points],
            "values": [point[1] for point in points],
            "net_change": None,
            "upward_step_fraction": None,
            "net_nonnegative": None,
            "tail_plateau": None,
            "tail_slope_per_round": None,
            "quality": 0.0,
        }
    x = np.asarray([point[0] for point in points], np.float64)
    y = np.asarray([point[1] for point in points], np.float64)
    deltas = np.diff(y)
    data_range = float(np.ptp(y))
    tolerance = max(float(absolute_tolerance), 0.05 * data_range)
    upward_fraction = float(np.mean(deltas >= -tolerance))
    net_change = float(y[-1] - y[0])
    net_nonnegative = bool(net_change >= -tolerance)
    tail_count = min(3, len(points))
    tail_x = x[-tail_count:]
    tail_y = y[-tail_count:]
    tail_slope = float(np.polyfit(tail_x, tail_y, 1)[0])
    plateau_tolerance = max(
        float(absolute_tolerance),
        0.20 * abs(net_change) / max(float(x[-1] - x[0]), 1.0),
    )
    tail_plateau = bool(abs(tail_slope) <= plateau_tolerance)
    quality = (
        0.50 * upward_fraction
        + 0.25 * float(net_nonnegative)
        + 0.25 * float(tail_plateau)
    )
    return {
        "rounds": [int(value) for value in x],
        "values": [float(value) for value in y],
        "absolute_tolerance": tolerance,
        "net_change": net_change,
        "upward_step_fraction": upward_fraction,
        "net_nonnegative": net_nonnegative,
        "tail_plateau": tail_plateau,
        "tail_slope_per_round": tail_slope,
        "tail_plateau_tolerance_per_round": plateau_tolerance,
        "quality": quality,
    }


def _nonincreasing_fraction(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if len(finite) < 2:
        return None
    return sum(
        right <= left + 1.0e-12
        for left, right in zip(finite, finite[1:])
    ) / (len(finite) - 1)


def _gamma_diagnostics(
    summary: dict[str, Any], round_key: str,
) -> dict[str, Any]:
    cells = summary[round_key]["per_gamma"]
    ordered = sorted(cells, key=float)
    metrics = {gamma: _cell_metrics(cells[gamma]) for gamma in ordered}
    clearance = [metrics[gamma]["clearance_m"] for gamma in ordered]
    time_to_goal = [metrics[gamma]["time_to_goal_s"] for gamma in ordered]
    clearance_fraction = _nonincreasing_fraction(clearance)
    time_fraction = _nonincreasing_fraction(time_to_goal)
    low, high = ordered[0], ordered[-1]
    clearance_endpoint = (
        metrics[high]["clearance_m"] <= metrics[low]["clearance_m"]
        if metrics[high]["clearance_m"] is not None
        and metrics[low]["clearance_m"] is not None else None
    )
    time_endpoint = (
        metrics[high]["time_to_goal_s"] <= metrics[low]["time_to_goal_s"]
        if metrics[high]["time_to_goal_s"] is not None
        and metrics[low]["time_to_goal_s"] is not None else None
    )
    available = [
        float(value) for value in (
            clearance_fraction, time_fraction,
            clearance_endpoint, time_endpoint,
        ) if value is not None
    ]
    return {
        "gamma_order": ordered,
        "per_gamma": metrics,
        "minimum_gamma_SR": min(
            float(metrics[gamma]["SR"]) for gamma in ordered
        ),
        "SR_range": max(float(metrics[gamma]["SR"]) for gamma in ordered)
        - min(float(metrics[gamma]["SR"]) for gamma in ordered),
        "clearance_nonincreasing_pair_fraction": clearance_fraction,
        "time_to_goal_nonincreasing_pair_fraction": time_fraction,
        "high_gamma_clearance_le_low_gamma": clearance_endpoint,
        "high_gamma_faster_or_equal": time_endpoint,
        "trend_quality": sum(available) / len(available) if available else 0.0,
    }


def _roundwise_gamma_diagnostics(
    summary: dict[str, Any], rounds: list[int],
) -> dict[str, Any]:
    by_round = {
        str(round_i): _gamma_diagnostics(summary, str(round_i))
        for round_i in rounds
    }
    qualities = [cell["trend_quality"] for cell in by_round.values()]
    return {
        "by_round": by_round,
        "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
        "minimum_quality": min(qualities) if qualities else 0.0,
        "fully_preserved_round_fraction": (
            sum(quality >= 1.0 - 1.0e-12 for quality in qualities)
            / len(qualities) if qualities else 0.0
        ),
    }


def _mode_counts(values: Any) -> dict[str, int]:
    counter = Counter(int(value) for value in values or [])
    return {str(mode): int(counter[mode]) for mode in range(4)}


def _acquisition_diagnostics(arm_directory: str, target_round: int) -> dict[str, Any]:
    metrics_path = Path(arm_directory) / "metrics.jsonl"
    if not metrics_path.is_file():
        return {"available": False, "reason": "metrics.jsonl not mirrored"}
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        cell = json.loads(line)
        if int(cell.get("round", -1)) <= target_round:
            rows.append(cell)
    by_round: dict[str, Any] = {}
    for cell in rows:
        per_gamma: dict[str, Any] = {}
        commits = cell.get("successful_executed_commit_by_gamma", {})
        for gamma, commit in commits.items():
            candidate_modes = commit.get("success_episode_sample_update_modes", {})
            per_gamma[str(gamma)] = {
                "retry_batches_used": int(commit.get("retry_batches_used", 0)),
                "attempted_episode_count": int(commit.get("attempted_episode_count", 0)),
                "terminal_success_count": int(commit.get("success_episode_count", 0)),
                "terminal_success_mode_counts": _mode_counts(candidate_modes.values()),
                "committed_trajectory_count": int(
                    commit.get("committed_trajectory_count", 0)
                ),
                "committed_mode_counts": _mode_counts(
                    commit.get("committed_sample_update_modes", [])
                ),
            }
        by_round[str(cell["round"])] = {
            "positive_loss": cell.get("positive_loss"),
            "optimizer_step": cell.get("optimizer_step"),
            "requested_optimizer_steps": cell.get("requested_optimizer_steps"),
            "round_total_s": cell.get("round_total_s"),
            "retry_batches_by_gamma": cell.get("retry_batches_by_gamma", {}),
            "available_trajectories_by_mode_gamma": cell.get(
                "available_trajectories_by_mode_gamma", {}
            ),
            "sampled_rows_by_mode_gamma": cell.get(
                "sampled_rows_by_mode_gamma", {}
            ),
            "per_gamma": per_gamma,
        }
    return {
        "available": True,
        "metrics_jsonl": str(metrics_path.resolve()),
        "round_count": len(by_round),
        "by_round": by_round,
    }


def _round_diagnostics(payload: dict[str, Any], target_round: int) -> dict[str, Any]:
    summary = payload["summary"]
    rows_by_round = payload["rows"]
    rounds = sorted(
        int(key) for key in summary if int(key) <= int(target_round)
    )
    curves: dict[str, list[float | None]] = {
        name: [] for name in (
            "SR", "CR", "OOB", "timeout", "validity", "clearance_m",
            "time_to_goal_s", "coverage",
        )
    }
    route_by_round: dict[str, Any] = {}
    for round_i in rounds:
        key = str(round_i)
        metric = _cell_metrics(summary[key]["pooled"])
        for name in curves:
            curves[name].append(metric[name])
        route_by_round[key] = _route_diagnostics(
            summary[key]["pooled"], rows_by_round.get(key, []),
        )
    expansion_indices = [i for i, round_i in enumerate(rounds) if round_i >= 1]
    expansion_rounds = [rounds[i] for i in expansion_indices]
    clearance = [curves["clearance_m"][i] for i in expansion_indices]
    time_to_goal = [curves["time_to_goal_s"][i] for i in expansion_indices]
    return {
        "rounds": rounds,
        "pooled_curves": curves,
        "route_by_round": route_by_round,
        "clearance_trend_r1_to_target": _trend(
            expansion_rounds, clearance, absolute_tolerance=0.002,
        ),
        "time_to_goal_trend_r1_to_target": _trend(
            expansion_rounds, time_to_goal, absolute_tolerance=0.10,
        ),
    }


def _raw_eval_for_arm(arm: Path, target_round: int) -> Path | None:
    exact = arm / f"fixed_eval_r000_r{target_round:03d}" / "raw_eval.json"
    if exact.is_file():
        return exact
    for candidate in sorted(arm.glob("fixed_eval_r000_r*/raw_eval.json")):
        try:
            payload = _read_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(target_round) in payload.get("summary", {}):
            return candidate
    return None


def _discover(stage: Path, target_round: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queued: dict[str, dict[str, Any]] = {}
    queue_path = stage / "QUEUE.json"
    if queue_path.is_file():
        for record in _read_json(queue_path).get("records", []):
            if isinstance(record, dict) and record.get("remote_output"):
                queued[Path(record["remote_output"]).name] = record
    arms_root = stage / "arms"
    basenames = set(queued)
    # Once the authoritative queue exists, exclude smoke/calibration mirrors
    # that share the stage's arms directory.  Directory discovery is only a
    # fallback for older stages without QUEUE.json provenance.
    if not queue_path.is_file() and arms_root.is_dir():
        basenames.update(path.name for path in arms_root.iterdir() if path.is_dir())
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for basename in sorted(basenames):
        arm = arms_root / basename
        record = queued.get(basename, {})
        provenance_path = arm / "SPOOL_PROVENANCE.json"
        provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
        name = str(provenance.get("name") or record.get("name") or basename)
        config_hash = provenance.get("config_hash") or record.get("config_hash")
        raw_eval = _raw_eval_for_arm(arm, target_round) if arm.is_dir() else None
        discovery = {
            "name": name,
            "basename": basename,
            "config_hash": config_hash,
            "physical_gpu": provenance.get(
                "physical_gpu", record.get("physical_gpu"),
            ),
            "arm_directory": str(arm.resolve()),
            "remote_output": record.get("remote_output"),
        }
        if raw_eval is None:
            pending.append({**discovery, "reason": "target raw_eval not mirrored"})
            continue
        try:
            payload = _read_json(raw_eval)
            summary = payload.get("summary", {})
            required = {str(index) for index in range(target_round + 1)}
            if not required.issubset(summary):
                raise ValueError("raw evaluation does not contain every r0..target round")
            if not isinstance(payload.get("rows"), dict):
                raise ValueError("raw evaluation rows are missing")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            pending.append({**discovery, "reason": f"invalid raw_eval: {error}"})
            continue
        completed.append({
            **discovery,
            "raw_eval": str(raw_eval.resolve()),
            "raw_eval_sha256": _sha256(raw_eval),
            "payload": payload,
        })
    return completed, pending


def _score_arm(discovery: dict[str, Any], target_round: int) -> dict[str, Any]:
    payload = discovery["payload"]
    key = str(target_round)
    pooled_cell = payload["summary"][key]["pooled"]
    metrics = _cell_metrics(pooled_cell)
    routes = _route_diagnostics(pooled_cell, payload["rows"].get(key, []))
    rounds = _round_diagnostics(payload, target_round)
    gamma = _gamma_diagnostics(payload["summary"], key)
    gamma_roundwise = _roundwise_gamma_diagnostics(
        payload["summary"], rounds["rounds"],
    )
    acquisition = _acquisition_diagnostics(
        discovery["arm_directory"], target_round,
    )
    status_counts = dict(Counter(
        str(row.get("status")) for row in payload["rows"].get(key, [])
    ))
    clearance_quality = rounds["clearance_trend_r1_to_target"]["quality"]
    time_quality = rounds["time_to_goal_trend_r1_to_target"]["quality"]
    oob_timeout = float(metrics["OOB"]) + float(metrics["timeout"])
    unsafe_rate = float(metrics["CR"]) + oob_timeout
    ranking_key = [
        # Joint safety comes first: eliminating collisions by turning them
        # into OOB/timeout is never considered an improvement.
        unsafe_rate,
        max(float(metrics["CR"]), oob_timeout),
        float(metrics["CR"]),
        1.0 - float(metrics["validity"]),
        oob_timeout,
        1.0 - float(metrics["SR"]),
        1.0 - float(routes["coverage_fraction"]),
        1.0 - float(routes["minimum_mode_share"]),
        1.0 - float(routes["normalized_entropy"]),
        1.0 - float(clearance_quality),
        1.0 - float(time_quality),
        1.0 - float(gamma_roundwise["minimum_quality"]),
        1.0 - float(gamma_roundwise["mean_quality"]),
    ]
    final_checks = {
        "pooled_SR_ge_0.95": float(metrics["SR"]) >= 0.95,
        "each_gamma_SR_ge_0.93": gamma["minimum_gamma_SR"] >= 0.93,
        "collision_le_0.03": float(metrics["CR"]) <= 0.03,
        "OOB_plus_timeout_le_0.03": (
            float(metrics["OOB"]) + float(metrics["timeout"]) <= 0.03
        ),
        "validity_ge_0.95": float(metrics["validity"]) >= 0.95,
        "coverage_4_of_4": routes["covered_modes"] == 4,
    }
    return {
        key: value for key, value in discovery.items() if key != "payload"
    } | {
        "round": target_round,
        "victory_priority": [
            "joint_CR_OOB_timeout_to_zero",
            "worst_safety_component_to_zero",
            "collision_rate_to_zero",
            "validity_to_one",
            "OOB_to_zero",
            "timeout_to_zero",
            "success_rate_to_one",
            "coverage_to_four_of_four",
            "route_balance",
            "clearance_upward_then_plateau",
            "time_to_goal_upward_then_plateau",
            "gamma_trends_preserved",
        ],
        "victory_distance_vector_lower_is_better": ranking_key,
        "metrics": metrics,
        "status_counts": status_counts,
        "coverage": routes,
        "round_diagnostics": rounds,
        "gamma_diagnostics": gamma,
        "roundwise_gamma_diagnostics": gamma_roundwise,
        "acquisition_diagnostics": acquisition,
        "final_checks": final_checks,
        "final_qualified": all(final_checks.values()),
    }


def _plot_best(best: dict[str, Any], output_stem: Path) -> dict[str, str]:
    rounds = best["round_diagnostics"]["rounds"]
    pooled = best["round_diagnostics"]["pooled_curves"]
    payload = _read_json(Path(best["raw_eval"]))
    gamma_order = best["gamma_diagnostics"]["gamma_order"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = plt.get_cmap("viridis")(np.linspace(0.10, 0.90, len(gamma_order)))
    for axis, (name, label, unit) in zip(axes.flat, PLOT_METRICS):
        axis.plot(
            rounds, pooled[name], color="black", linewidth=2.5,
            marker="o", markersize=4, label="pooled",
        )
        for gamma, color in zip(gamma_order, colors):
            values = []
            for round_i in rounds:
                cell = _cell_metrics(
                    payload["summary"][str(round_i)]["per_gamma"][gamma]
                )
                values.append(cell[name])
            axis.plot(
                rounds, values, color=color, linewidth=1.2,
                marker=".", alpha=0.72, label=rf"$\gamma={gamma}$",
            )
        axis.set_xlabel("Expansion round")
        axis.set_ylabel(f"{label} ({unit})" if unit else label)
        axis.set_xticks(rounds)
        axis.grid(alpha=0.22, linewidth=0.6)
        if name in RATE_METRICS:
            axis.set_ylim(-0.02, 1.02)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        f"Best margin-hybrid arm: {best['name']} (r0–r{best['round']})",
        y=0.995,
    )
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=len(labels), frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def _markdown(payload: dict[str, Any]) -> str:
    def formatted(value: float | None, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# PRE2 margin-hybrid sweep leaderboard",
        "",
        "Ranking is lexicographic: joint unsafe rate (CR+OOB+timeout) → 0, "
        "worst safety component → 0, CR → 0, validity → 1, then "
        "coverage/balance, clearance and TtG upward-to-plateau behavior, "
        "and roundwise gamma-trend preservation.",
        "",
    ]
    if not payload["leaderboard"]:
        lines.extend(["No complete r0–r8 fixed evaluation has been mirrored yet.", ""])
    else:
        lines.extend([
            f"Best completed arm: **{payload['best_arm']}**",
            "",
            "| Rank | Arm | CR | Validity | OOB | Timeout | SR | Coverage | "
            "Min share | Clearance (m) | TtG (s) | C trend | T trend | Gamma |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in payload["leaderboard"]:
            metric = row["metrics"]
            coverage = row["coverage"]
            rounds = row["round_diagnostics"]
            lines.append(
                f"| {row['rank']} | {row['name'].replace('|', '/')} | "
                f"{metric['CR']:.4f} | {metric['validity']:.4f} | "
                f"{metric['OOB']:.4f} | {metric['timeout']:.4f} | "
                f"{metric['SR']:.4f} | {coverage['covered_modes']}/4 | "
                f"{coverage['minimum_mode_share']:.3f} | "
                f"{formatted(metric['clearance_m'], 4)} | "
                f"{formatted(metric['time_to_goal_s'], 3)} | "
                f"{rounds['clearance_trend_r1_to_target']['quality']:.3f} | "
                f"{rounds['time_to_goal_trend_r1_to_target']['quality']:.3f} | "
                f"{row['gamma_diagnostics']['trend_quality']:.3f} |"
            )
        lines.append("")
    if payload["pending"]:
        lines.extend(["## Pending or incomplete", ""])
        for row in payload["pending"]:
            lines.append(f"- `{row['name']}`: {row['reason']}")
        lines.append("")
    return "\n".join(lines)


def score_stage(stage: Path, target_round: int) -> dict[str, Any]:
    completed, pending = _discover(stage, target_round)
    leaderboard = [_score_arm(arm, target_round) for arm in completed]
    leaderboard.sort(
        key=lambda row: tuple(row["victory_distance_vector_lower_is_better"])
    )
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank
    return {
        "schema_version": 1,
        "status": "PRE2_MARGIN_HYBRID_LEADERBOARD",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage.resolve()),
        "target_round": target_round,
        "ranking_contract": {
            "type": "lexicographic; no weighted scalar can trade a higher-priority safety metric for a lower-priority metric",
            "lower_is_better": True,
            "priority": [
                "CR+OOB+timeout", "max(CR,OOB+timeout)", "CR",
                "1-validity", "OOB+timeout", "1-SR",
                "1-coverage", "1-minimum-mode-share", "1-route-entropy",
                "1-clearance-trend-quality", "1-TtG-trend-quality",
                "1-min-roundwise-gamma-trend-quality",
                "1-mean-roundwise-gamma-trend-quality",
            ],
            "trend_window": "r1 through target; r0 is retained in raw curves as the PRE2 baseline",
        },
        "completed_count": len(leaderboard),
        "pending_count": len(pending),
        "best_arm": leaderboard[0]["name"] if leaderboard else None,
        "final_winners": [row["name"] for row in leaderboard if row["final_qualified"]],
        "leaderboard": leaderboard,
        "pending": pending,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--target-round", type=int, default=8)
    args = parser.parse_args()
    if args.target_round < 0:
        parser.error("--target-round must be nonnegative")
    stage = args.stage.resolve()
    result = score_stage(stage, args.target_round)
    if result["leaderboard"]:
        best = result["leaderboard"][0]
        result["best_raw_curve_artifacts"] = _plot_best(
            best,
            stage / f"BEST_ARM_RAW_CURVES_R000_R{args.target_round:03d}",
        )
        diagnostics = {
            "schema_version": 1,
            "best_arm": best["name"],
            "raw_eval": best["raw_eval"],
            "raw_eval_sha256": best["raw_eval_sha256"],
            "round": best["round"],
            "metrics": best["metrics"],
            "coverage": best["coverage"],
            "round_diagnostics": best["round_diagnostics"],
            "gamma_diagnostics": best["gamma_diagnostics"],
            "roundwise_gamma_diagnostics": best[
                "roundwise_gamma_diagnostics"
            ],
            "acquisition_diagnostics": best["acquisition_diagnostics"],
            "final_checks": best["final_checks"],
        }
        diagnostics_path = stage / "BEST_ARM_DIAGNOSTICS.json"
        _atomic_json(diagnostics_path, diagnostics)
        result["best_arm_diagnostics"] = str(diagnostics_path.resolve())
    _atomic_json(stage / "LEADERBOARD.json", result)
    _atomic_text(stage / "LEADERBOARD.md", _markdown(result))
    print(json.dumps({
        "completed_count": result["completed_count"],
        "pending_count": result["pending_count"],
        "best_arm": result["best_arm"],
    }, indent=2))


if __name__ == "__main__":
    main()
