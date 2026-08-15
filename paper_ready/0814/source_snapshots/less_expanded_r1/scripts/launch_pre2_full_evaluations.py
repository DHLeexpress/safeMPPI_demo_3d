#!/usr/bin/env python3
"""Launch each completed PRE2 five-round boundary evaluation independently."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
RESULT = ROOT / "results/stage1_single_ball_t128/0810_pre2_paper_closed_loop"
PRE = (
    ROOT / "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
GPUS = (1, 3)
TASK_CONFIG = RESULT / "initial/quad_high_sr_trunk3/task_config_resolved.json"
GPU_BY_ARM = {
    "exp_balanced_head": 0,
    "exp_balanced_trunk1": 1,
    "exp_balanced_trunk3": 3,
    "quad_fast_head": 0,
    "quad_fast_trunk1": 1,
    "quad_fast_trunk3": 3,
    "quad_high_sr_head": 0,
    "quad_high_sr_trunk1": 1,
    "quad_high_sr_trunk3": 3,
}


def _pre2_gpu_loads(commands: str) -> dict[int, int]:
    loads = {gpu: 0 for gpu in GPUS}
    result_text = str(RESULT.resolve())
    for command in commands.splitlines():
        if result_text not in command or "scripts/research_" not in command:
            continue
        match = re.search(r"--helios-gpu\s+(\d+)", command)
        if match and int(match.group(1)) in loads:
            loads[int(match.group(1))] += 1
    return loads


def _ensure_task_config(arm: Path) -> None:
    destination = arm / "task_config_resolved.json"
    if destination.is_file():
        return
    if not TASK_CONFIG.is_file():
        raise FileNotFoundError(TASK_CONFIG)
    shutil.copy2(TASK_CONFIG, destination)


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


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _expansion_active(commands: str, output: Path) -> bool:
    pattern = re.compile(
        rf"(?:^|\s)--output\s+{re.escape(str(output.resolve()))}(?:\s|$)"
    )
    return any(
        "research_ball_expansion_optimization.py" in line and pattern.search(line)
        for line in commands.splitlines()
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new", type=int, default=3)
    args = parser.parse_args()
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    candidates = []
    cancelled_arms = _cancelled_arm_names()
    baseline_evaluation = RESULT / "evaluations/baseline/pre2/r000"
    baseline_marker = (
        RESULT / "full_evaluation_launches/baseline/pre2/r000.json"
    )
    baseline_source = RESULT / "initial/quad_high_sr_trunk3"
    baseline_available = (baseline_evaluation / "raw_eval.json").is_file()
    if not baseline_available:
        for raw_eval in (RESULT / "evaluations").glob("*/*/r*/raw_eval.json"):
            try:
                if "0" in json.loads(raw_eval.read_text()).get("summary", {}):
                    baseline_available = True
                    break
            except (OSError, json.JSONDecodeError):
                continue
    if not baseline_available and baseline_source.is_dir():
        baseline_pending = False
        if baseline_marker.is_file():
            state = json.loads(baseline_marker.read_text())
            if _pid_alive(int(state.get("pid", -1))):
                baseline_pending = True
            else:
                attempt = 1
                while baseline_marker.with_name(
                    f"{baseline_marker.stem}.failed_attempt_{attempt}.json"
                ).exists():
                    attempt += 1
                baseline_marker.replace(baseline_marker.with_name(
                    f"{baseline_marker.stem}.failed_attempt_{attempt}.json"
                ))
                baseline_pending = attempt >= 3
        if not baseline_pending:
            candidates.append((
                "baseline", baseline_source, 0,
                baseline_evaluation, baseline_marker,
            ))
    for stage in ("initial", "adaptive"):
        stage_root = RESULT / stage
        if not stage_root.is_dir():
            continue
        for arm in sorted(path for path in stage_root.iterdir() if path.is_dir()):
            if arm.name in cancelled_arms or arm.name.endswith(".paper_arm_process.lock"):
                continue
            _ensure_task_config(arm)
            if _expansion_active(commands, arm):
                continue
            boundaries = sorted(
                int(path.stem.rsplit("_", 1)[1])
                for path in arm.glob("checkpoint_*.pt")
                if int(path.stem.rsplit("_", 1)[1]) > 0
                and int(path.stem.rsplit("_", 1)[1]) % 5 == 0
            )
            for round_i in boundaries:
                evaluation = (
                    RESULT / "evaluations" / stage / arm.name
                    / f"r{round_i:03d}"
                )
                marker = (
                    RESULT / "full_evaluation_launches" / stage / arm.name
                    / f"r{round_i:03d}.json"
                )
                if (evaluation / "raw_eval.json").is_file():
                    continue
                if marker.is_file():
                    state = json.loads(marker.read_text())
                    if _pid_alive(int(state.get("pid", -1))):
                        continue
                    attempt = 1
                    while marker.with_name(
                        f"{marker.stem}.failed_attempt_{attempt}.json"
                    ).exists():
                        attempt += 1
                    marker.replace(marker.with_name(
                        f"{marker.stem}.failed_attempt_{attempt}.json"
                    ))
                    if attempt >= 3:
                        continue
                candidates.append((stage, arm, round_i, evaluation, marker))

    launched = []
    gpu_loads = _pre2_gpu_loads(commands)
    for stage, arm, round_i, evaluation, marker in candidates[:args.max_new]:
        if stage == "baseline":
            gpu = min(GPUS, key=lambda candidate: (gpu_loads[candidate], candidate))
        elif arm.name in GPU_BY_ARM and GPU_BY_ARM[arm.name] in GPUS:
            gpu = GPU_BY_ARM[arm.name]
        else:
            gpu = min(GPUS, key=lambda candidate: (gpu_loads[candidate], candidate))
        gpu_loads[gpu] += 1
        log = (
            RESULT / "logs" / "full_evaluation"
            / f"{stage}_{arm.name}_r{round_i:03d}.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
        env = os.environ.copy()
        env.update({
            "PRE": str(PRE),
            "EXPANSION": str(arm),
            "EVAL_OUT": str(evaluation),
            "HELIOS_GPU": str(gpu),
            "TARGET_ROUND": str(round_i),
        })
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_pre2_paper_eval.sh"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        state = {
            "status": "FULL_EVALUATION_LAUNCHED",
            "pid": process.pid,
            "launched_unix": time.time(),
            "stage": stage,
            "arm": "pre2" if stage == "baseline" else arm.name,
            "round": round_i,
            "gpu": gpu,
            "evaluation": str(evaluation),
            "log": str(log),
        }
        _write(marker, state)
        launched.append(state)
    print(json.dumps({
        "status": "PRE2_FULL_EVALUATION_LAUNCH",
        "launched": launched,
        "queued_unlaunched": max(len(candidates) - len(launched), 0),
    }, indent=2))


if __name__ == "__main__":
    main()
