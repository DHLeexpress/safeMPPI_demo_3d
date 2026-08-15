#!/usr/bin/env python3
"""Capture exact-CUDA faithful-M1 trajectories after the GPU1 worker exits.

This remote worker is intentionally separate from the expansion process.  It
re-runs the canonical fixed random bank and the fixed bowling scene for
R1/R3/R5/R7/R9, preserving full states/controls for 3-D visualization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _run(command: list[str]) -> None:
    print("[trajectory-capture] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    if args.output_base.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_base}")
    scripts = Path(__file__).resolve().parent
    common = [
        "--pretrain-dir", spec["source_pretrain"],
        "--expansion", spec["extension_output"],
        "--device", "cuda:0",
        "--checkpoint-rounds", "1,3,5,7,9",
        "--episodes", "50",
        "--samples-per-step", "1",
        "--sampling-temperature", "1.0",
        "--execution-clearance-exp-weight", "15",
        "--execution-clearance-target-m", "0.6",
        "--execution-clearance-exp-temperature", "0.15",
        "--execution-taskspace-quadratic-weight", "250",
        "--execution-taskspace-quadratic-target-m", "0.15",
        "--execution-axis-cylinder-quadratic-weight", "5",
        "--execution-axis-cylinder-radius-m", "1.1",
        "--execution-control-weight", "0.05",
        "--execution-obstacle-speed-weight", "400",
        "--execution-cost-band-fraction", "0.05",
        "--save-raw-trajectories",
    ]
    _run([
        sys.executable,
        str(scripts / "evaluate_multisphere_min_cost_deployment.py"),
        *common,
        "--scene-bank-json", spec["scene_bank"],
        "--evaluation-output", str(args.output_base / "random_fixed_bank"),
        "--seed", "91000",
    ])
    _run([
        sys.executable,
        str(scripts / "evaluate_multisphere_bowling_hybrid.py"),
        *common,
        "--output", str(args.output_base / "bowling_fixed"),
        "--seed", "92000",
    ])
    (args.output_base / "COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE",
        "device": "cuda:0 (physical GPU1 via CUDA_VISIBLE_DEVICES=1)",
        "rounds": [1, 3, 5, 7, 9],
        "attempts_per_checkpoint": 200,
        "raw_trajectory_capture": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
