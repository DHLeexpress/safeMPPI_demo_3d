#!/usr/bin/env python3
"""Continue one paired PRE2 weight/replay arm from its exact saved state."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
RESULT = ROOT / "results/stage1_single_ball_t128/0810_pre2_paper_closed_loop"
PRE = (
    ROOT / "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
WEIGHT_RE = re.compile(
    r"^quad_high_sr_trunk3_w(?P<weight>\d+)"
    r"(?P<replay>_sliding3_s(?P<steps>\d+))?_paired_s82300$"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--target-round", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int, choices=(1, 3))
    parser.add_argument("--attempt", default="base")
    parser.add_argument("--retry-cap", type=int, default=4096)
    args = parser.parse_args()
    match = WEIGHT_RE.fullmatch(args.arm)
    if match is None:
        parser.error("--arm is not an approved paired weight/replay arm")
    if args.target_round < 10 or args.target_round % 5:
        parser.error("--target-round must be a multiple of five at least 10")

    output = RESULT / "adaptive" / args.arm
    previous = args.target_round - 5
    if not (output / f"checkpoint_{previous:03d}.pt").is_file():
        parser.error(f"missing checkpoint {previous} for {args.arm}")
    if (output / f"checkpoint_{args.target_round:03d}.pt").is_file():
        print(json.dumps({"status": "SKIPPED", "reason": "target complete"}))
        return
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    if any(
        "research_ball_expansion_optimization.py" in line
        and str(output.resolve()) in line
        for line in commands.splitlines()
    ):
        print(json.dumps({"status": "SKIPPED", "reason": "already active"}))
        return

    marker = (
        RESULT / "adaptive_continuation_launches" / args.arm
        / f"r{args.target_round:03d}_{args.attempt}.json"
    )
    if marker.is_file():
        print(json.dumps({"status": "SKIPPED", "reason": "already launched"}))
        return

    replay_scope = "sliding" if match.group("replay") else "cumulative"
    replay_rounds = 3 if replay_scope == "sliding" else 100
    optimizer_steps = int(match.group("steps") or 2500)
    weight = int(match.group("weight"))
    log = (
        RESULT / "logs" / "adaptive_continuation"
        / f"{args.arm}_r{args.target_round:03d}_{args.attempt}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a")
    env = os.environ.copy()
    env.update({
        "PRE": str(PRE),
        "OUT": str(output),
        "HELIOS_GPU": str(args.gpu),
        "EXECUTION_ARM": "quad_high_sr",
        "UPDATE_SCOPE": "trunk3",
        "ROUNDS": str(args.target_round),
        "SEED": "82300",
        "BETA": "0.1",
        "FLOW_BASE_STD": "1.0",
        "FLOW_BASE_STD_FINAL": "1.0",
        "FLOW_BASE_STD_SCHEDULE": "none",
        "RETRY_RESAMPLE_BATCH_CAP": str(args.retry_cap),
        "QUAD_HIGH_SR_WEIGHT": str(weight),
        "REPLAY_SCOPE": replay_scope,
        "REPLAY_ROUNDS": str(replay_rounds),
        "OPTIMIZER_STEPS_PER_ROUND": str(optimizer_steps),
    })
    process = subprocess.Popen(
        ["bash", "scripts/RECIPE_pre2_paper_arm.sh"],
        cwd=ROOT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    state = {
        "status": "ADAPTIVE_CONTINUATION_LAUNCHED",
        "arm": args.arm,
        "pid": process.pid,
        "launched_unix": time.time(),
        "previous_round": previous,
        "target_round": args.target_round,
        "attempt": args.attempt,
        "gpu": args.gpu,
        "retry_cap": args.retry_cap,
        "weight": weight,
        "replay_scope": replay_scope,
        "replay_rounds": replay_rounds,
        "optimizer_steps_per_round": optimizer_steps,
        "log": str(log),
    }
    _write(marker, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
