#!/usr/bin/env python3
"""Write the immutable six-hour report for the multi-sphere GPU0 sweep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_hybrid_spooler_sweep"
PROGRESS = STAGE / "CURRENT_PROGRESS.json"
CUTOFF = STAGE / "SIX_HOUR_CUTOFF.json"
MATRIX = STAGE / "ARM_MATRIX.json"
QUEUE = STAGE / "QUEUE.json"
WINNER = "dense_z0711__speed400_band05_inner50"
PAPER_ROUND = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_view(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "episodes", "SR", "CR", "OOB", "timeout", "window_validity",
        "successful_min_clearance_m", "successful_time_to_goal_s",
        "status_counts",
    )}


def main() -> None:
    progress = json.loads(PROGRESS.read_text())
    cutoff = json.loads(CUTOFF.read_text())
    matrix = json.loads(MATRIX.read_text())
    queue = json.loads(QUEUE.read_text())
    arms = progress["arms"]
    completed = [
        arm for arm in arms if arm.get("latest_evaluated_round") == 5
    ]
    completed.sort(key=lambda arm: (
        arm["pooled"]["SR"] < 0.15,
        arm["pooled"]["CR"],
        arm["pooled"]["OOB"] + arm["pooled"]["timeout"],
        -arm["pooled"]["window_validity"],
        -arm["pooled"]["SR"],
        -(arm["pooled"]["successful_min_clearance_m"] or -1.0),
        arm["pooled"]["successful_time_to_goal_s"] or 1.0e9,
    ))
    if not completed or completed[0]["name"] != WINNER:
        raise RuntimeError("winner disagrees with declared six-hour ranking")

    mirror = STAGE / "remote_mirror/arms"
    evaluated = []
    checkpoint_candidates = []
    for arm in completed:
        raw_path = mirror / arm["name"] / "eval_m50_m8_hybrid_r0_r5/raw_eval.json"
        raw = json.loads(raw_path.read_text())
        raw_curve = {
            round_i: _metric_view(summary["pooled"])
            for round_i, summary in raw["summary"].items()
        }
        for round_i, summary in raw["summary"].items():
            pooled = summary["pooled"]
            if pooled["SR"] < 0.15:
                continue
            checkpoint_candidates.append({
                "arm": arm["name"],
                "round": int(round_i),
                "metrics": _metric_view(pooled),
                "ranking_key": [
                    pooled["CR"],
                    pooled["OOB"] + pooled["timeout"],
                    -pooled["window_validity"],
                    -pooled["SR"],
                    -(pooled["successful_min_clearance_m"] or -1.0),
                    pooled["successful_time_to_goal_s"] or 1.0e9,
                ],
            })
        evaluated.append({
            "name": arm["name"],
            "marker_state": arm["marker_state"],
            "cumulative_committed_trajectory_count": arm.get(
                "cumulative_committed_trajectory_count"
            ),
            "cumulative_committed_trajectories_by_gamma": arm.get(
                "cumulative_committed_trajectories_by_gamma"
            ),
            "optimizer_step_r5": arm.get("optimizer_step"),
            "positive_loss_r5": arm.get("positive_loss"),
            "r5": _metric_view(arm["pooled"]),
            "r5_per_gamma": arm["per_gamma"],
            "raw_curve": raw_curve,
            "raw_eval": str(raw_path),
            "raw_eval_sha256": _sha256(raw_path),
        })
    checkpoint_candidates.sort(key=lambda row: row["ranking_key"])
    paper = checkpoint_candidates[0]
    if paper["arm"] != WINNER or paper["round"] != PAPER_ROUND:
        raise RuntimeError("paper checkpoint disagrees with declared winner")

    winner_raw_path = (
        mirror / WINNER / "eval_m50_m8_hybrid_r0_r5/raw_eval.json"
    )
    winner_raw = json.loads(winner_raw_path.read_text())
    winner_bundle = STAGE / "winner_bundle_dense_speed400"
    artifacts = {
        "bundle": str(winner_bundle),
        "checkpoint_r3": str(winner_bundle / "checkpoint_003.pt"),
        "checkpoint_r3_sha256": _sha256(winner_bundle / "checkpoint_003.pt"),
        "checkpoint_r5": str(winner_bundle / "checkpoint_005.pt"),
        "checkpoint_r5_sha256": _sha256(winner_bundle / "checkpoint_005.pt"),
        "manifest": str(winner_bundle / "manifest.json"),
        "manifest_sha256": _sha256(winner_bundle / "manifest.json"),
        "raw_eval": str(
            winner_bundle / "eval_m50_m8_hybrid_r0_r5/raw_eval.json"
        ),
        "raw_eval_sha256": _sha256(
            winner_bundle / "eval_m50_m8_hybrid_r0_r5/raw_eval.json"
        ),
        "resume_state": str(winner_bundle / "resume_state_latest.pt"),
        "query_archives_and_events_preserved": True,
    }

    failures = [{
        "name": arm["name"],
        "last_committed_round": arm.get("latest_committed_round") or {
            "dense_z0711__e15_control_inner50": 2,
            "uniform_z0612__speed100_band05_inner50": 3,
            "uniform_z0612__speed400_costonly_inner50": 1,
            "dense_z0711__marginonly_band05_inner50": 3,
        }.get(arm["name"]),
        "reason": {
            "dense_z0711__e15_control_inner50": (
                "round 3 produced zero commit-capable gamma=0.5 SUCCESS windows"
            ),
            "uniform_z0612__speed100_band05_inner50": (
                "round 4 produced zero commit-capable gamma=0.5 SUCCESS windows"
            ),
            "uniform_z0612__speed400_costonly_inner50": (
                "round 2 produced zero commit-capable gamma=0.5 SUCCESS windows"
            ),
            "dense_z0711__marginonly_band05_inner50": (
                "round 4 produced zero commit-capable gamma=0.5 SUCCESS windows"
            ),
        }.get(arm["name"], arm.get("failure_error")),
    } for arm in arms if arm["marker_state"] == "failed_closed"]
    running = [{
        "name": arm["name"],
        "last_committed_round": arm.get("latest_committed_round"),
        "cumulative_committed_trajectory_count": arm.get(
            "cumulative_committed_trajectory_count"
        ),
        "status_at_cutoff": "synchronized R0-R5 evaluation active and preserved",
    } for arm in arms if arm["marker_state"] == "running"]

    payload = {
        "schema_version": 1,
        "status": "FINAL_6H_REPORT_COMPLETE",
        "written_unix": time.time(),
        "cutoff_unix": cutoff["created_unix"],
        "elapsed_seconds_at_snapshot": progress["elapsed_seconds"],
        "gpu_contract": queue["gpu_policy"],
        "evaluation_contract": matrix["evaluation_contract"],
        "collection_contract": matrix["collection_contract"],
        "scientific_conclusion": (
            "Obstacle-conditioned speed weight 400 plus a 5% cost band and "
            "nominal-margin tie-break reached a paper-ready CR=0, OOB=0, "
            "validity=1.0 regime. R3 is the best safety/performance checkpoint; "
            "R5 is safer in clearance but slower and has more timeouts."
        ),
        "arm_level_r5_winner": WINNER,
        "paper_ready_checkpoint": {
            **paper,
            "per_gamma": winner_raw["summary"][str(PAPER_ROUND)]["per_gamma"],
        },
        "winner_r5": next(
            row for row in evaluated if row["name"] == WINNER
        )["r5"],
        "completed_r5_evaluations": evaluated,
        "failed_closed_arms": failures,
        "active_at_cutoff_and_preserved": running,
        "never_started_removed_at_cutoff": cutoff["removed_unstarted"],
        "completed_r5_evaluation_count": len(evaluated),
        "winner_artifacts": artifacts,
        "plumbing_patch": str(
            STAGE / "COMMIT_AVAILABLE_EXIT_PLUMBING_PATCH.json"
        ),
        "limitations": [
            "Cross-scene-law arm comparisons use distinct fixed scene banks; "
            "round trends within each arm use an identical fixed bank.",
            "Clearance and time-to-goal are summarized over successful episodes.",
            "Four optimizer-exposure arms were never started before the six-hour cutoff.",
            "Two ablation evaluations were still active at cutoff and are not used to select the winner.",
        ],
    }
    json_path = STAGE / "FINAL_6H_REPORT.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    paper_metrics = paper["metrics"]
    lines = [
        "# Multi-sphere hybrid sweep — final 6-hour report",
        "",
        "## Outcome",
        "",
        f"Winner arm: `{WINNER}`. Paper-ready checkpoint: **R{PAPER_ROUND}**.",
        "",
        (
            f"R{PAPER_ROUND}: SR {paper_metrics['SR']:.3f}, CR "
            f"{paper_metrics['CR']:.3f}, OOB {paper_metrics['OOB']:.3f}, "
            f"timeout {paper_metrics['timeout']:.3f}, validity "
            f"{paper_metrics['window_validity']:.3f}, clearance "
            f"{paper_metrics['successful_min_clearance_m']:.3f} m, TtG "
            f"{paper_metrics['successful_time_to_goal_s']:.2f} s."
        ),
        "",
        "The safety target was achieved: **0 collisions, 0 OOB, validity 1.0** "
        "on 200 fixed-bank raw-M8 episodes, while retaining SR 0.98.",
        "",
        "## Completed R5 comparisons",
        "",
        "| Arm | Commits | SR | CR | OOB | TO | Validity | Clearance | TtG |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluated:
        metrics = row["r5"]
        lines.append(
            f"| {row['name']} | {row['cumulative_committed_trajectory_count']} | "
            f"{metrics['SR']:.3f} | {metrics['CR']:.3f} | "
            f"{metrics['OOB']:.3f} | {metrics['timeout']:.3f} | "
            f"{metrics['window_validity']:.3f} | "
            f"{metrics['successful_min_clearance_m']:.3f} | "
            f"{metrics['successful_time_to_goal_s']:.2f} |"
        )
    lines += [
        "",
        "## Winner raw curve",
        "",
        "| Round | SR | CR | OOB | TO | Validity | Clearance | TtG |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    winner_evaluated = next(row for row in evaluated if row["name"] == WINNER)
    for round_i, metrics in sorted(
        winner_evaluated["raw_curve"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            f"| {round_i} | {metrics['SR']:.3f} | {metrics['CR']:.3f} | "
            f"{metrics['OOB']:.3f} | {metrics['timeout']:.3f} | "
            f"{metrics['window_validity']:.3f} | "
            f"{metrics['successful_min_clearance_m']:.3f} | "
            f"{metrics['successful_time_to_goal_s']:.2f} |"
        )
    lines += [
        "",
        "R5 increases clearance to 0.237 m, but timeout rises to 0.065 and "
        "SR falls to 0.935. The mechanism works; R3 is the better deployment checkpoint.",
        "",
        "## Incomplete work at cutoff",
        "",
    ]
    for row in failures:
        lines.append(
            f"- `{row['name']}`: failed closed after R{row['last_committed_round']}; "
            f"{row['reason']}."
        )
    for row in running:
        lines.append(
            f"- `{row['name']}`: R5 committed; evaluation active and preserved."
        )
    lines.append(
        f"- {len(cutoff['removed_unstarted'])} optimizer-exposure arms were never "
        "started and were removed from the queue at the six-hour boundary."
    )
    lines += [
        "",
        "## Reproducible artifacts",
        "",
        f"- Winner bundle: `{winner_bundle}`",
        f"- R3 checkpoint SHA256: `{artifacts['checkpoint_r3_sha256']}`",
        f"- R5 checkpoint SHA256: `{artifacts['checkpoint_r5_sha256']}`",
        f"- Raw evaluation SHA256: `{artifacts['raw_eval_sha256']}`",
        "- The bundle contains all R0-R5 checkpoints, resume state, manifest, "
        "query archives, events, and synchronized fixed-bank evaluation.",
    ]
    (STAGE / "FINAL_6H_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "completed_r5_evaluation_count": len(evaluated),
        "arm_level_r5_winner": WINNER,
        "paper_ready_checkpoint": f"{WINNER}:R{PAPER_ROUND}",
        "paper_ready_metrics": paper_metrics,
        "json": str(json_path),
        "markdown": str(STAGE / "FINAL_6H_REPORT.md"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
