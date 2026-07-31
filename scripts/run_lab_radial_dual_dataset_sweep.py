#!/usr/bin/env python3
"""Run the fixed depth-3 radial-token sweep on two explicit datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOKEN_DIMS = (64, 128, 256)
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_datasets(values: list[list[str]]) -> list[dict[str, Any]]:
    datasets = []
    labels = set()
    for label, demo_dir, context_cache, ood_config in values:
        if not LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"invalid dataset label {label!r}")
        if label in labels:
            raise ValueError(f"duplicate dataset label {label!r}")
        labels.add(label)
        paths = {
            "demo_dir": Path(demo_dir),
            "context_cache": Path(context_cache),
            "ood_config": Path(ood_config),
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"dataset {label!r} has missing paths: {missing}"
            )
        datasets.append({"label": label, **paths})
    if len(datasets) != 2:
        raise ValueError(
            f"exactly two --dataset entries required, got {len(datasets)}"
        )
    return datasets


def build_arms(
    datasets: list[dict[str, Any]],
    devices: list[str],
) -> list[dict[str, Any]]:
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--device values must be nonempty and unique")
    arms = []
    for dataset in datasets:
        for token_dim in TOKEN_DIMS:
            index = len(arms)
            arms.append({
                **dataset,
                "name": f"{dataset['label']}_radial_t{token_dim}_d3",
                "grid_token_dim": token_dim,
                "trunk_depth": 3,
                "device": devices[index % len(devices)],
            })
    return arms


def gpu_uuid(device: str) -> str:
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise ValueError(
            "exclusive-device waiting requires devices formatted as cuda:N"
        )
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    index = int(match.group(1))
    for row in result.stdout.splitlines():
        row_index, row_uuid = (value.strip() for value in row.split(",", 1))
        if int(row_index) == index:
            return row_uuid
    raise ValueError(f"CUDA device index {index} is not visible")


def wait_for_exclusive_device(device: str, poll_seconds: float) -> None:
    expected_uuid = gpu_uuid(device)
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        occupied = any(
            row.split(",", 1)[0].strip() == expected_uuid
            for row in result.stdout.splitlines()
            if row.strip()
        )
        if not occupied:
            return
        time.sleep(poll_seconds)


def trainer_command(args: argparse.Namespace, arm: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/pretrain_lab_reference_flow.py"),
        "--demo-dir", str(arm["demo_dir"]),
        "--context-cache", str(arm["context_cache"]),
        "--output", str(args.output_root / arm["name"]),
        "--context-model", "radial_hp3d",
        "--grid-token-dim", str(arm["grid_token_dim"]),
        "--trunk-depth", "3",
        "--history-token-dim", "32",
        "--hidden", "48",
        "--representation-dim", "32",
        "--nfe", "16",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--min-epochs", str(args.min_epochs),
        "--device", str(arm["device"]),
        "--audit-episodes", str(args.audit_episodes),
        "--audit-seed", str(args.audit_seed),
        "--ood-config", str(arm["ood_config"]),
        "--ood-audit-episodes", str(args.audit_episodes),
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
        "dataset": arm["label"],
        "device": arm["device"],
        "grid_token_dim": int(arm["grid_token_dim"]),
        "trunk_depth": 3,
        "parameter_count": int(manifest["trainable_parameter_count"]),
        "requested_epochs": int(
            manifest.get("requested_epochs", manifest["epochs"])
        ),
        "actual_epochs": int(
            manifest.get("actual_epochs", manifest["epochs"])
        ),
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
    log_path = args.output_root / f"{arm['name']}.log"
    environment = os.environ.copy()
    thread_count = str(args.cpu_threads_per_arm)
    environment.update({
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
    })
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
    with (output_root / "architecture_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=4,
        metavar=("LABEL", "DEMO_DIR", "CONTEXT_CACHE", "OOD_CONFIG"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--max-parallel-per-device", type=int, default=1)
    parser.add_argument("--cpu-threads-per-arm", type=int, default=8)
    parser.add_argument(
        "--wait-for-exclusive-devices",
        action="store_true",
        help="start each device queue only after that physical GPU is idle",
    )
    parser.add_argument("--device-poll-seconds", type=float, default=15.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--audit-episodes", type=int, default=100)
    parser.add_argument("--audit-seed", type=int, default=91000)
    parser.add_argument("--ood-audit-seed", type=int, default=191000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.max_parallel_per_device < 1:
        raise ValueError("--max-parallel-per-device must be positive")
    if args.cpu_threads_per_arm < 1:
        raise ValueError("--cpu-threads-per-arm must be positive")
    if args.device_poll_seconds <= 0.0:
        raise ValueError("--device-poll-seconds must be positive")
    if min(args.patience, args.min_delta, args.min_epochs) < 0:
        raise ValueError("early-stop values must be nonnegative")
    if args.min_epochs > args.epochs:
        raise ValueError("--min-epochs cannot exceed --epochs")

    datasets = parse_datasets(args.dataset)
    arms = build_arms(datasets, args.device)
    args.output_root.mkdir(parents=True)
    recipe = {
        "status": "LAB_RADIAL_DUAL_DATASET_SWEEP_RUNNING",
        "datasets": [{
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in dataset.items()
        } for dataset in datasets],
        "arms": [{
            key: str(value) if isinstance(value, Path) else value
            for key, value in arm.items()
        } for arm in arms],
        "depth": 3,
        "token_dims": list(TOKEN_DIMS),
        "devices": list(args.device),
        "max_parallel_per_device": args.max_parallel_per_device,
        "cpu_threads_per_arm": args.cpu_threads_per_arm,
        "wait_for_exclusive_devices": args.wait_for_exclusive_devices,
        "device_poll_seconds": args.device_poll_seconds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta,
            "min_epochs": args.min_epochs,
        },
        "audit_episodes_per_gamma": args.audit_episodes,
        "audit_seed": args.audit_seed,
        "ood_audit_seed": args.ood_audit_seed,
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

    def run_device_queue(device_arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if args.wait_for_exclusive_devices:
            wait_for_exclusive_device(
                device_arms[0]["device"],
                args.device_poll_seconds,
            )
        if args.max_parallel_per_device == 1:
            return [run_arm(args, arm) for arm in device_arms]
        with ThreadPoolExecutor(
            max_workers=args.max_parallel_per_device
        ) as executor:
            futures = {
                executor.submit(run_arm, args, arm): arm
                for arm in device_arms
            }
            try:
                return [
                    future.result() for future in as_completed(futures)
                ]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    try:
        with ThreadPoolExecutor(max_workers=len(per_device)) as executor:
            futures = {
                executor.submit(run_device_queue, device_arms): device
                for device, device_arms in per_device.items()
            }
            for future in as_completed(futures):
                for row in future.result():
                    rows_by_name[row["arm"]] = row
    except BaseException as exc:
        (args.output_root / "SWEEP_FAILED.json").write_text(json.dumps({
            **recipe,
            "status": "LAB_RADIAL_DUAL_DATASET_SWEEP_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_arms": sorted(rows_by_name),
        }, indent=2) + "\n")
        raise

    rows = [rows_by_name[arm["name"]] for arm in arms]
    write_table(rows, args.output_root)
    (args.output_root / "SWEEP_COMPLETE.json").write_text(json.dumps({
        **recipe,
        "status": "LAB_RADIAL_DUAL_DATASET_SWEEP_COMPLETE",
        "architecture_table": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
