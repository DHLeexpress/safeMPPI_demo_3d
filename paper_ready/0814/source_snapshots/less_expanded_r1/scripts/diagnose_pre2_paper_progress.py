#!/usr/bin/env python3
"""Write a compact, repeatable snapshot of the PRE2 paper expansion arms."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


DEFAULT_ROOT = Path(
    "/Users/dhl/Documents/safeMPPI_demo_3d/results/"
    "stage1_single_ball_t128/0810_pre2_paper_closed_loop"
)
SAMPLE_RE = re.compile(
    r"\[sample-update\] r(?P<round>\d+) gamma (?P<gamma>[0-9.]+) "
    r"retry batch (?P<retry>\d+)/(?P<cap>\d+).*?"
    r"unguided b(?P<below>\d+)/a(?P<above>\d+)/"
    r"l(?P<left>\d+)/r(?P<right>\d+) \| need mode \"(?P<need>[^\"]*)\""
)


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _active_commands() -> str:
    result = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    )
    return result.stdout


def _cancelled_arm_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.glob("*CANCELLED.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entries = []
        if isinstance(payload.get("arm"), str):
            entries.append(payload["arm"])
        for key in ("arms", "cancelled_arms"):
            if isinstance(payload.get(key), list):
                entries.extend(
                    item.get("arm") for item in payload[key]
                    if isinstance(item, dict) and isinstance(item.get("arm"), str)
                )
        names.update(entry.rsplit("/", 1)[-1] for entry in entries)
    return names


def _expansion_active(commands: str, arm: Path) -> bool:
    pattern = re.compile(
        rf"(?:^|\s)--output\s+{re.escape(str(arm.resolve()))}(?:\s|$)"
    )
    return any(
        "research_ball_expansion_optimization.py" in line and pattern.search(line)
        for line in commands.splitlines()
    )


def _latest_sample(log: Path) -> dict | None:
    if not log.is_file():
        return None
    latest = None
    for line in log.read_text(errors="replace").splitlines():
        match = SAMPLE_RE.search(line)
        if match:
            latest = match.groupdict()
    if latest is None:
        return None
    for key in ("round", "retry", "cap", "below", "above", "left", "right"):
        latest[key] = int(latest[key])
    latest["missing_modes"] = [
        int(value) for value in latest.pop("need").split(",") if value
    ]
    return latest


def _round_summary(row: dict) -> dict:
    commits = row.get("successful_executed_commit_by_gamma", {})
    per_gamma = {}
    for gamma, details in commits.items():
        mode_counts = {str(mode): 0 for mode in range(4)}
        for mode in details.get("success_episode_sample_update_modes", {}).values():
            mode_counts[str(mode)] += 1
        attempted = int(details.get("attempted_episode_count", 0))
        successes = int(details.get("success_episode_count", 0))
        per_gamma[str(gamma)] = {
            "retry_batches": details.get("retry_batches_used"),
            "attempted_episodes": attempted,
            "terminal_successes": successes,
            "terminal_success_rate": successes / attempted if attempted else None,
            "terminal_success_modes": mode_counts,
        }
    gp_buffer = int(row.get("gp_buffer", 0))
    fast_contexts = int(row.get("retry_verify_all_fast_path_contexts", 0))
    if gp_buffer == 0:
        beta_interpretation = "inactive_no_gp_reference"
    elif fast_contexts:
        beta_interpretation = "active_on_ranked_B_but_bypassed_on_retry_all_K"
    else:
        beta_interpretation = "active"
    available_trajectories = list(
        row.get("available_trajectories_by_mode_gamma", {}).values()
    )
    available_rows = list(
        row.get("available_rows_by_mode_gamma", {}).values()
    )
    sampled_rows = list(
        row.get("sampled_rows_by_mode_gamma", {}).values()
    )

    def span(values: list) -> dict | None:
        if not values:
            return None
        return {
            "strata": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "total": sum(values),
        }

    return {
        "round": row.get("round"),
        "per_gamma": per_gamma,
        "gp_buffer": gp_buffer,
        "beta": row.get("beta"),
        "uncertainty_uplift": row.get("uncertainty_uplift"),
        "marginal_ess_mean": row.get("marginal_ess_mean"),
        "retry_verify_all_fast_path_contexts": fast_contexts,
        "beta_interpretation": beta_interpretation,
        "gather_s": row.get("gather_s"),
        "update_s": row.get("update_s"),
        "round_total_s": row.get("round_total_s"),
        "positive_loss": row.get("positive_loss"),
        "optimizer_step": row.get("optimizer_step"),
        "optimizer_steps_this_round": row.get("steps"),
        "replay_scope": row.get("replay_scope"),
        "replay_positive_total": row.get("replay_positive_total"),
        "available_trajectories_by_stratum": span(available_trajectories),
        "available_rows_by_stratum": span(available_rows),
        "sampled_rows_by_stratum": span(sampled_rows),
    }


def _arm_snapshot(
    stage: str, arm: Path, root: Path, commands: str, cancelled: bool,
) -> dict:
    logs = [
        *(root / "logs").glob(f"{stage}_{arm.name}_r*.log"),
        *(root / "logs/continuation").glob(f"{arm.name}_r*.log"),
        *(root / "logs/adaptive_continuation").glob(f"{arm.name}_r*.log"),
    ]
    log = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
    metrics = _json_lines(arm / "metrics.jsonl")
    checkpoints = sorted(arm.glob("checkpoint_*.pt"))
    active = _expansion_active(commands, arm)
    continuation_markers = sorted(
        (root / "continuation_launches" / arm.name).glob("r*.json")
    )
    continuation_markers.extend(sorted(
        (root / "adaptive_continuation_launches" / arm.name).glob("r*.json")
    ))
    continuation_targets = []
    for marker in continuation_markers:
        try:
            continuation_targets.append(
                int(json.loads(marker.read_text())["target_round"])
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    target_round = max(continuation_targets, default=5)
    traceback = bool(log and "Traceback" in log.read_text(errors="replace"))
    if cancelled:
        state = "cancelled"
    elif active:
        state = "active"
    elif (
        metrics and checkpoints
        and int(checkpoints[-1].stem.rsplit("_", 1)[1]) % 5 == 0
    ):
        state = "full_boundary_complete"
    elif traceback:
        state = "failed_closed"
    else:
        state = "inactive"
    durations = [
        float(row["round_total_s"]) for row in metrics
        if row.get("round_total_s") is not None
    ]
    eta_s = None
    if active and durations:
        eta_s = statistics.median(durations) * max(target_round - len(metrics), 0)
    return {
        "arm": arm.name,
        "stage": stage,
        "state": state,
        "process_active": active,
        "committed_rounds": len(metrics),
        "target_round": target_round,
        "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
        "resume_state_present": (arm / "resume_state_latest.pt").is_file(),
        "latest_sample": _latest_sample(log) if log else None,
        "latest_round": _round_summary(metrics[-1]) if metrics else None,
        "estimated_remaining_expansion_s": eta_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    commands = _active_commands()
    cancelled_arms = _cancelled_arm_names(root)
    arms = []
    for stage in ("initial", "adaptive"):
        stage_root = root / stage
        if stage_root.is_dir():
            arms.extend(
                _arm_snapshot(
                    stage, arm, root, commands, arm.name in cancelled_arms,
                )
                for arm in sorted(stage_root.iterdir())
                if arm.is_dir() and not arm.name.endswith(".paper_arm_process.lock")
            )
    active_etas = [
        row["estimated_remaining_expansion_s"] for row in arms
        if row["estimated_remaining_expansion_s"] is not None
    ]
    payload = {
        "generated_unix": time.time(),
        "root": str(root),
        "active_count": sum(row["state"] == "active" for row in arms),
        "failed_closed_count": sum(
            row["state"] == "failed_closed" for row in arms
        ),
        "cancelled_count": sum(row["state"] == "cancelled" for row in arms),
        "estimated_slowest_active_expansion_s": max(active_etas, default=None),
        "uncertainty_audit_rule": (
            "gp_buffer=0 means beta had no GP information; retry_B=K with "
            "retry_verify_all_fast_path bypasses acquisition ranking on retries"
        ),
        "arms": arms,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.write:
        destination = root / "CURRENT_DIAGNOSIS.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(rendered)
        temporary.replace(destination)
    print(rendered, end="")


if __name__ == "__main__":
    main()
