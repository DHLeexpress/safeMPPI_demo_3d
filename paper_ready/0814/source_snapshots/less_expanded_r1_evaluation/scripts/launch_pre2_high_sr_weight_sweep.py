#!/usr/bin/env python3
"""Launch a paired quadratic-weight sweep of the leading PRE2 arm."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
RESULT = ROOT / "results/stage1_single_ball_t128/0810_pre2_paper_closed_loop"
PRE = (
    ROOT / "results/stage1_single_ball_t128/"
    "0810_pretrain_mirrored1000_hp100_t128_d3_balanced_ema_recovered"
)
VARIANTS = (
    (2250, 1, 82300),
    (2500, 0, 82300),
    (2750, 3, 82300),
    (3000, 0, 82300),
)
EXPERIMENT_SUFFIX = "_paired_s82300"
TARGET_ROUND = 5
RETRY_CAP = 4096


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _active(output: Path, commands: str) -> bool:
    return any(
        "research_ball_expansion_optimization.py" in line
        and str(output.resolve()) in line
        for line in commands.splitlines()
    )


def main() -> None:
    commands = subprocess.run(
        ["ps", "-ax", "-o", "command="], check=True,
        capture_output=True, text=True,
    ).stdout
    launched = []
    skipped = []
    for weight, gpu, seed in VARIANTS:
        label = f"quad_high_sr_trunk3_w{weight}{EXPERIMENT_SUFFIX}"
        output = RESULT / "adaptive" / label
        marker = RESULT / "weight_sweep_launches" / f"{label}_r005.json"
        if (output / "checkpoint_005.pt").is_file():
            skipped.append({"arm": label, "reason": "r5 already complete"})
            continue
        if _active(output, commands):
            skipped.append({"arm": label, "reason": "already active"})
            continue
        if marker.is_file():
            skipped.append({"arm": label, "reason": "already launched"})
            continue
        if output.exists():
            skipped.append({"arm": label, "reason": "non-fresh output exists"})
            continue

        log = RESULT / "logs" / f"adaptive_{label}_r005.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
        env = os.environ.copy()
        env.update({
            "PRE": str(PRE),
            "OUT": str(output),
            "HELIOS_GPU": str(gpu),
            "EXECUTION_ARM": "quad_high_sr",
            "UPDATE_SCOPE": "trunk3",
            "ROUNDS": str(TARGET_ROUND),
            "SEED": str(seed),
            "BETA": "0.1",
            "FLOW_BASE_STD": "1.0",
            "FLOW_BASE_STD_FINAL": "1.0",
            "FLOW_BASE_STD_SCHEDULE": "none",
            "RETRY_RESAMPLE_BATCH_CAP": str(RETRY_CAP),
            "QUAD_HIGH_SR_WEIGHT": str(weight),
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
            "status": "WEIGHT_SWEEP_LAUNCHED",
            "arm": label,
            "pid": process.pid,
            "launched_unix": time.time(),
            "source": "fresh recovered PRE2",
            "paired_seed_design": True,
            "target_round": TARGET_ROUND,
            "execution_rule": "quadratic_cost",
            "execution_clearance_quadratic_weight": weight,
            "execution_clearance_quadratic_target_m": 0.6,
            "retry_cap": RETRY_CAP,
            "gpu": gpu,
            "seed": seed,
            "log": str(log),
        }
        _write(marker, state)
        launched.append(state)

    print(json.dumps({"launched": launched, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
