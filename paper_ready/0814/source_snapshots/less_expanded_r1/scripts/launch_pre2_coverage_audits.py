#!/usr/bin/env python3
"""Launch bounded odd-round PRE2 coverage audits without blocking expansion."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
RESULT = ROOT / "results/stage1_single_ball_t128/0810_pre2_paper_closed_loop"
PRE = (
    ROOT / "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
GPUS = (1, 3)
TASK_CONFIG = RESULT / "initial/quad_high_sr_trunk3/task_config_resolved.json"
REMOTE_RE = re.compile(
    r"dohyun@helios\.robotics\.caltech\.edu:"
    r"(?P<path>/data3/research1/safeMPPI_remote_cli/runs/[^/'\"]+)/"
)


def _cancelled_arm_names() -> set[str]:
    names: set[str] = set()
    for path in RESULT.glob("*CANCELLED.json"):
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _pre2_gpu_loads() -> dict[int, int]:
    """Count live PRE2 expansion/evaluation wrappers by requested Helios GPU."""
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    loads = {gpu: 0 for gpu in GPUS}
    result_text = str(RESULT.resolve())
    for command in commands.splitlines():
        if result_text not in command or "scripts/research_" not in command:
            continue
        match = re.search(r"--helios-gpu\s+(\d+)", command)
        if match and int(match.group(1)) in loads:
            loads[int(match.group(1))] += 1
    return loads


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _ensure_task_config(arm: Path) -> None:
    destination = arm / "task_config_resolved.json"
    if destination.is_file():
        return
    if not TASK_CONFIG.is_file():
        raise FileNotFoundError(TASK_CONFIG)
    shutil.copy2(TASK_CONFIG, destination)


def _failed_attempt_count(marker: Path) -> int:
    count = len(list(marker.parent.glob(f"{marker.stem}.failed_attempt_*.json")))
    if marker.with_suffix(".failed_once.json").is_file():
        count += 1
    return count


def _ensure_remote_metadata(stage: str, arm: Path) -> bool:
    metadata = arm / ".helios_remote.json"
    if metadata.is_file():
        return True
    logs = sorted((RESULT / "logs").glob(f"{stage}_{arm.name}_r*.log"))
    if not logs:
        return False
    matches = list(REMOTE_RE.finditer(logs[-1].read_text(errors="replace")))
    if not matches:
        return False
    _write(metadata, {
        "status": "HELIOS_REMOTE_EXPANSION_IN_PROGRESS",
        "remote_output": matches[-1].group("path"),
        "recovered_for_checkpoint_evaluation": True,
    })
    return True


def _manifest_override(stage: str, arm: Path, round_i: int) -> Path:
    checkpoint = arm / f"checkpoint_{round_i:03d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = {
        "status": "LIVE_CHECKPOINT_COVERAGE_MANIFEST",
        "config": payload["config"],
        "checkpoint_round": int(round_i),
        "source_checkpoint": str(checkpoint.resolve()),
        "stage": stage,
        "arm": arm.name,
    }
    output = (
        RESULT / "coverage_inputs" / stage / arm.name
        / f"manifest_r{round_i:03d}.json"
    )
    _write(output, manifest)
    return output


def _is_promoted(arm: Path) -> bool:
    marker_root = RESULT / "continuation_launches" / arm.name
    for marker in marker_root.glob("r*.json"):
        try:
            if int(json.loads(marker.read_text()).get("target_round", 0)) > 5:
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=80)
    args = parser.parse_args()
    if args.max_new < 0 or args.episodes < 1:
        parser.error("--max-new must be nonnegative and --episodes positive")

    candidates = []
    cancelled_arms = _cancelled_arm_names()
    for stage in ("initial", "adaptive"):
        stage_root = RESULT / stage
        if not stage_root.is_dir():
            continue
        for arm in sorted(path for path in stage_root.iterdir() if path.is_dir()):
            if arm.name in cancelled_arms or arm.name.endswith(".paper_arm_process.lock"):
                continue
            if not _ensure_remote_metadata(stage, arm):
                continue
            _ensure_task_config(arm)
            published = sorted(arm.glob("checkpoint_*.pt"))
            coverage_rounds = [
                int(path.stem.rsplit("_", 1)[1])
                for path in published
                if int(path.stem.rsplit("_", 1)[1]) > 0
                and int(path.stem.rsplit("_", 1)[1]) % 2 == 1
                and int(path.stem.rsplit("_", 1)[1]) % 5 != 0
            ]
            for round_i in coverage_rounds:
                checkpoint = arm / f"checkpoint_{round_i:03d}.pt"
                evaluation = (
                    RESULT / "coverage_evaluations" / stage / arm.name
                    / f"r{round_i:03d}_e{args.episodes}"
                )
                marker = (
                    RESULT / "coverage_launches" / stage / arm.name
                    / f"r{round_i:03d}_e{args.episodes}.json"
                )
                if (evaluation / "raw_eval.json").is_file() or not checkpoint.is_file():
                    continue
                if not marker.is_file() and _failed_attempt_count(marker) >= 4:
                    continue
                if marker.is_file():
                    state = json.loads(marker.read_text())
                    if _pid_alive(int(state.get("pid", -1))):
                        continue
                    failure_count = _failed_attempt_count(marker) + 1
                    marker.replace(marker.with_name(
                        f"{marker.stem}.failed_attempt_{failure_count}.json"
                    ))
                    if failure_count >= 4:
                        # The first three attempts predate exact-round staging;
                        # preserve one bounded retry with the repaired evaluator.
                        continue
                candidates.append((stage, arm, round_i, evaluation, marker))

    # A continued arm is already promoted by full-boundary evidence. Audit its
    # newest odd checkpoint before spending GPU time on stale initial arms.
    candidates.sort(key=lambda item: (
        not _is_promoted(item[1]),
        -item[2] if _is_promoted(item[1]) else item[2],
        item[0],
        item[1].name,
    ))
    one_per_arm = []
    repeated_arm = []
    seen_arms = set()
    for candidate in candidates:
        arm_key = (candidate[0], candidate[1].name)
        if arm_key in seen_arms:
            repeated_arm.append(candidate)
        else:
            one_per_arm.append(candidate)
            seen_arms.add(arm_key)
    candidates = one_per_arm + repeated_arm

    launched = []
    gpu_loads = _pre2_gpu_loads()
    for stage, arm, round_i, evaluation, marker in candidates[:args.max_new]:
        gpu = min(GPUS, key=lambda candidate: (gpu_loads[candidate], candidate))
        gpu_loads[gpu] += 1
        log = (
            RESULT / "logs" / "coverage"
            / f"{stage}_{arm.name}_r{round_i:03d}_e{args.episodes}.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
        manifest_override = _manifest_override(stage, arm, round_i)
        env = os.environ.copy()
        env.update({
            "PRE": str(PRE),
            "EXPANSION": str(arm),
            "EXPANSION_MANIFEST": str(manifest_override),
            "EVAL_OUT": str(evaluation),
            "HELIOS_GPU": str(gpu),
            "CHECKPOINT_ROUND": str(round_i),
            "COVERAGE_EPISODES": str(args.episodes),
        })
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_pre2_coverage_eval.sh"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        state = {
            "status": "LAUNCHED",
            "attempt": _failed_attempt_count(marker) + 1,
            "pid": process.pid,
            "launched_unix": time.time(),
            "stage": stage,
            "arm": arm.name,
            "round": round_i,
            "episodes_per_gamma": args.episodes,
            "gpu": gpu,
            "evaluation": str(evaluation),
            "manifest_override": str(manifest_override),
            "log": str(log),
        }
        _write(marker, state)
        launched.append(state)

    print(json.dumps({
        "status": "PRE2_COVERAGE_AUDIT_LAUNCH",
        "launched": launched,
        "pending_eligible_next_tick": max(len(candidates) - len(launched), 0),
    }, indent=2))


if __name__ == "__main__":
    main()
