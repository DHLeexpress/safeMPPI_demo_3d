#!/usr/bin/env python3
"""Launch selected PRE2 arms to the next full-evaluation boundary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
RESULT = ROOT / "results/stage1_single_ball_t128/0810_pre2_paper_closed_loop"
PRE = (
    ROOT / "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)


@dataclass(frozen=True)
class Arm:
    execution: str
    scope: str
    gpu: int
    seed: int


ARMS = {
    "exp_balanced_head": Arm("exp_balanced", "head", 3, 82100),
    "exp_balanced_trunk1": Arm("exp_balanced", "trunk1", 1, 82101),
    "exp_balanced_trunk3": Arm("exp_balanced", "trunk3", 3, 82102),
    "quad_fast_head": Arm("quad_fast", "head", 3, 82103),
    "quad_fast_trunk1": Arm("quad_fast", "trunk1", 1, 82104),
    "quad_fast_trunk3": Arm("quad_fast", "trunk3", 3, 82105),
    "quad_high_sr_head": Arm("quad_high_sr", "head", 3, 82106),
    "quad_high_sr_trunk1": Arm("quad_high_sr", "trunk1", 1, 82107),
    "quad_high_sr_trunk3": Arm("quad_high_sr", "trunk3", 3, 82108),
}


def _active_commands() -> str:
    return subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout


def _expansion_active(commands: str, output: Path) -> bool:
    return any(
        "research_ball_expansion_optimization.py" in line
        and str(output.resolve()) in line
        for line in commands.splitlines()
    )


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, choices=ARMS)
    parser.add_argument("--target-round", type=int, required=True)
    parser.add_argument("--attempt", default="base")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--flow-base-std", type=float, default=1.0)
    parser.add_argument("--flow-base-std-final", type=float, default=1.0)
    parser.add_argument("--flow-base-std-schedule", default="none")
    parser.add_argument("--retry-cap", type=int, default=64)
    args = parser.parse_args()
    if args.target_round < 10 or args.target_round % 5:
        parser.error("--target-round must be a multiple of five at least 10")

    commands = _active_commands()
    launched = []
    skipped = []
    for label in args.arm:
        arm = ARMS[label]
        output = RESULT / "initial" / label
        previous = args.target_round - 5
        if not (output / f"checkpoint_{previous:03d}.pt").is_file():
            skipped.append({"arm": label, "reason": f"missing checkpoint {previous}"})
            continue
        if (output / f"checkpoint_{args.target_round:03d}.pt").is_file():
            skipped.append({"arm": label, "reason": "target already complete"})
            continue
        if _expansion_active(commands, output):
            skipped.append({"arm": label, "reason": "expansion already active"})
            continue
        marker = (
            RESULT / "continuation_launches" / label
            / f"r{args.target_round:03d}_{args.attempt}.json"
        )
        legacy_marker = marker.with_name(f"r{args.target_round:03d}.json")
        if marker.is_file() or (args.attempt == "base" and legacy_marker.is_file()):
            skipped.append({"arm": label, "reason": "target already launched"})
            continue
        log = (
            RESULT / "logs" / "continuation"
            / f"{label}_r{args.target_round:03d}.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
        env = os.environ.copy()
        env.update({
            "PRE": str(PRE),
            "OUT": str(output),
            "HELIOS_GPU": str(arm.gpu),
            "EXECUTION_ARM": arm.execution,
            "UPDATE_SCOPE": arm.scope,
            "ROUNDS": str(args.target_round),
            "SEED": str(arm.seed),
            "BETA": str(args.beta),
            "FLOW_BASE_STD": str(args.flow_base_std),
            "FLOW_BASE_STD_FINAL": str(args.flow_base_std_final),
            "FLOW_BASE_STD_SCHEDULE": args.flow_base_std_schedule,
            "RETRY_RESAMPLE_BATCH_CAP": str(args.retry_cap),
        })
        process = subprocess.Popen(
            ["bash", "scripts/RECIPE_pre2_paper_arm.sh"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        state = {
            "status": "CONTINUATION_LAUNCHED",
            "arm": label,
            "pid": process.pid,
            "launched_unix": time.time(),
            "previous_round": previous,
            "target_round": args.target_round,
            "attempt": args.attempt,
            "beta": args.beta,
            "flow_base_std": args.flow_base_std,
            "flow_base_std_final": args.flow_base_std_final,
            "flow_base_std_schedule": args.flow_base_std_schedule,
            "retry_cap": args.retry_cap,
            "gpu": arm.gpu,
            "log": str(log),
        }
        _write(marker, state)
        launched.append(state)
    print(json.dumps({"launched": launched, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
