#!/usr/bin/env python3
"""Run the fixed HP100 token-width sweep on one demonstration archive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOKEN_DIMS = (64, 128, 256)
AUDIT_EPISODES_PER_GAMMA = 100


def validate_sources(demo_dir: Path, ood_config: Path) -> None:
    required = (
        demo_dir,
        demo_dir / "manifest.json",
        ood_config,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing sweep inputs: {missing}")


def build_arms(
    demo_dir: Path,
    ood_config: Path,
    devices: list[str],
) -> list[dict[str, Any]]:
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--device values must be nonempty and unique")
    return [
        {
            "name": f"hp100_t{token_dim}_d3",
            "demo_dir": demo_dir,
            "ood_config": ood_config,
            "grid_token_dim": token_dim,
            "trunk_depth": 3,
            "device": devices[index % len(devices)],
        }
        for index, token_dim in enumerate(TOKEN_DIMS)
    ]


def trainer_command(
    args: argparse.Namespace,
    arm: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/pretrain_lab_reference_flow.py"),
        "--demo-dir", str(arm["demo_dir"]),
        "--output", str(args.output_root / arm["name"]),
        "--context-model", "uniform_hp100",
        "--grid-token-dim", str(arm["grid_token_dim"]),
        "--trunk-depth", "3",
        "--hidden", "48",
        "--representation-dim", "32",
        "--nfe", "16",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-windows-per-trajectory",
        str(args.max_windows_per_trajectory),
        "--cuda-amp",
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--min-epochs", str(args.min_epochs),
        "--device", str(arm["device"]),
        "--audit-episodes", str(AUDIT_EPISODES_PER_GAMMA),
        "--audit-seed", str(args.audit_seed),
        "--ood-config", str(arm["ood_config"]),
        "--ood-audit-episodes", str(AUDIT_EPISODES_PER_GAMMA),
        "--ood-audit-seed", str(args.ood_audit_seed),
        "--seed", str(args.seed),
    ]
    return command


def _pooled(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summary_row(arm: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "pretrain_manifest.json").read_text())
    row = {
        "arm": arm["name"],
        "device": arm["device"],
        "grid_token_dim": int(arm["grid_token_dim"]),
        "trunk_depth": 3,
        "parameter_count": int(manifest["trainable_parameter_count"]),
        "actual_epochs": int(manifest["actual_epochs"]),
        "selected_epoch": int(manifest["selected_epoch"]),
        "selected_valid_loss": float(manifest["selected_valid_loss"]),
    }
    for prefix, field in (
        ("id", "raw_audit_summary"),
        ("ood", "ood_raw_audit_summary"),
    ):
        summaries = manifest.get(field)
        if not summaries:
            raise ValueError(f"{arm['name']}: missing {field}")
        for metric in ("SR", "CR", "OOB", "window_validity"):
            row[f"{prefix}_{metric}"] = _pooled(summaries, metric)
    return row


def run_arm(args: argparse.Namespace, arm: dict[str, Any]) -> dict[str, Any]:
    output = args.output_root / arm["name"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite arm {output}")
    environment = os.environ.copy()
    thread_count = str(args.cpu_threads_per_arm)
    environment.update({
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
    })
    log_path = args.output_root / f"{arm['name']}.log"
    with log_path.open("x") as stream:
        subprocess.run(
            trainer_command(args, arm),
            check=True,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return summary_row(arm, output)


def write_table(rows: list[dict[str, Any]], output_root: Path) -> None:
    (output_root / "architecture_table.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    with (output_root / "architecture_table.csv").open(
        "x", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--ood-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--cpu-threads-per-arm", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-windows-per-trajectory", type=int, default=32,
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--audit-seed", type=int, default=91000)
    parser.add_argument("--ood-audit-seed", type=int, default=191000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    validate_sources(args.demo_dir, args.ood_config)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.cpu_threads_per_arm < 1:
        raise ValueError("--cpu-threads-per-arm must be positive")
    if args.max_windows_per_trajectory < 1:
        raise ValueError("--max-windows-per-trajectory must be positive")
    if args.patience < 1 or args.min_epochs < 1:
        raise ValueError("this sweep requires enabled early stopping")
    if args.min_delta < 0.0 or args.min_epochs > args.epochs:
        raise ValueError("invalid early-stopping contract")

    arms = build_arms(args.demo_dir, args.ood_config, args.device)
    args.output_root.mkdir(parents=True)
    recipe = {
        "status": "LAB_HP100_TOKEN_SWEEP_RUNNING",
        "demo_dir": str(args.demo_dir.resolve()),
        "ood_config": str(args.ood_config.resolve()),
        "arms": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in arm.items()
            }
            for arm in arms
        ],
        "context_model": "uniform_hp100",
        "trunk_depth": 3,
        "token_dims": list(TOKEN_DIMS),
        "cuda_amp": {"enabled": True, "dtype": "bfloat16"},
        "audit": {
            "episodes_per_gamma": AUDIT_EPISODES_PER_GAMMA,
            "id_seed": args.audit_seed,
            "ood_seed": args.ood_audit_seed,
            "sampling_temperature": 1.0,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_windows_per_trajectory": args.max_windows_per_trajectory,
        "learning_rate": args.learning_rate,
        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta,
            "min_epochs": args.min_epochs,
        },
        "seed": args.seed,
    }
    (args.output_root / "sweep_recipe.json").write_text(
        json.dumps(recipe, indent=2) + "\n"
    )

    rows_by_name: dict[str, dict[str, Any]] = {}
    per_device = {
        device: [arm for arm in arms if arm["device"] == device]
        for device in args.device
    }

    def run_queue(device_arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [run_arm(args, arm) for arm in device_arms]

    try:
        with ThreadPoolExecutor(max_workers=len(per_device)) as executor:
            futures = {
                executor.submit(run_queue, device_arms): device
                for device, device_arms in per_device.items()
            }
            for future in as_completed(futures):
                for row in future.result():
                    rows_by_name[row["arm"]] = row
    except BaseException as exc:
        (args.output_root / "SWEEP_FAILED.json").write_text(json.dumps({
            **recipe,
            "status": "LAB_HP100_TOKEN_SWEEP_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_arms": sorted(rows_by_name),
        }, indent=2) + "\n")
        raise

    rows = [rows_by_name[arm["name"]] for arm in arms]
    write_table(rows, args.output_root)
    (args.output_root / "SWEEP_COMPLETE.json").write_text(json.dumps({
        **recipe,
        "status": "LAB_HP100_TOKEN_SWEEP_COMPLETE",
        "architecture_table": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
