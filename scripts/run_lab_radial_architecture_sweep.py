"""Run the five fixed nonuniform-radial architecture arms sequentially."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    {
        "name": "radial_t64_d2",
        "context_model": "radial_hp3d",
        "grid_token_dim": 64,
        "trunk_depth": 2,
    },
    {
        "name": "radial_t128_d2",
        "context_model": "radial_hp3d",
        "grid_token_dim": 128,
        "trunk_depth": 2,
    },
    {
        "name": "radial_t256_d2",
        "context_model": "radial_hp3d",
        "grid_token_dim": 256,
        "trunk_depth": 2,
    },
    {
        "name": "radial_t64_d3",
        "context_model": "radial_hp3d",
        "grid_token_dim": 64,
        "trunk_depth": 3,
    },
    {
        "name": "radial_t64_d3_gru32",
        "context_model": "radial_hp3d_gru",
        "grid_token_dim": 64,
        "trunk_depth": 3,
    },
)


def _pooled(summary: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in summary]))


def _summary_row(arm: dict, output: Path) -> dict:
    manifest = json.loads((output / "pretrain_manifest.json").read_text())
    row = {
        **arm,
        "history_token_dim": (
            32 if arm["context_model"] == "radial_hp3d_gru" else 0
        ),
        "parameter_count": int(manifest["trainable_parameter_count"]),
        "selected_epoch": int(manifest["selected_epoch"]),
        "selected_valid_loss": float(manifest["selected_valid_loss"]),
    }
    for prefix, key in (
        ("id", "raw_audit_summary"),
        ("ood", "ood_raw_audit_summary"),
    ):
        values = manifest[key]
        if values is None:
            continue
        for metric in ("SR", "CR", "OOB", "window_validity"):
            row[f"{prefix}_{metric}"] = _pooled(values, metric)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ood-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--audit-episodes", type=int, default=100)
    parser.add_argument("--audit-seed", type=int, default=91000)
    parser.add_argument("--ood-audit-seed", type=int, default=191000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-parallel-arms",
        type=int,
        choices=(1, 2),
        default=1,
        help="run at most two independent arms on the declared device",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite sweep {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    recipe = {
        "status": "LAB_RADIAL_ARCHITECTURE_SWEEP_RUNNING",
        "arms": list(ARMS),
        "demo_dir": str(args.demo_dir.resolve()),
        "context_cache": str(args.context_cache.resolve()),
        "ood_config": str(args.ood_config.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "audit_episodes_per_gamma": args.audit_episodes,
        "audit_seed": args.audit_seed,
        "ood_audit_seed": args.ood_audit_seed,
        "seed": args.seed,
        "max_parallel_arms": args.max_parallel_arms,
    }
    (args.output_root / "sweep_recipe.json").write_text(
        json.dumps(recipe, indent=2) + "\n"
    )
    def run_arm(arm: dict) -> dict:
        output = args.output_root / arm["name"]
        command = [
            sys.executable,
            str(ROOT / "scripts/pretrain_lab_reference_flow.py"),
            "--demo-dir", str(args.demo_dir),
            "--context-cache", str(args.context_cache),
            "--output", str(output),
            "--context-model", arm["context_model"],
            "--grid-token-dim", str(arm["grid_token_dim"]),
            "--trunk-depth", str(arm["trunk_depth"]),
            "--history-token-dim", "32",
            "--hidden", "48",
            "--representation-dim", "32",
            "--nfe", "16",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--device", args.device,
            "--audit-episodes", str(args.audit_episodes),
            "--audit-seed", str(args.audit_seed),
            "--ood-config", str(args.ood_config),
            "--ood-audit-episodes", str(args.audit_episodes),
            "--ood-audit-seed", str(args.ood_audit_seed),
            "--seed", str(args.seed),
        ]
        subprocess.run(command, check=True, env=os.environ.copy())
        return _summary_row(arm, output)

    rows_by_name = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel_arms) as executor:
        future_to_arm = {
            executor.submit(run_arm, arm): arm
            for arm in ARMS
        }
        try:
            for future in as_completed(future_to_arm):
                arm = future_to_arm[future]
                rows_by_name[arm["name"]] = future.result()
        except BaseException:
            for future in future_to_arm:
                future.cancel()
            raise
    rows = [rows_by_name[arm["name"]] for arm in ARMS]

    (args.output_root / "architecture_table.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    fieldnames = list(rows[0])
    with (args.output_root / "architecture_table.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    complete = {
        **recipe,
        "status": "LAB_RADIAL_ARCHITECTURE_SWEEP_COMPLETE",
        "architecture_table": rows,
    }
    (args.output_root / "SWEEP_COMPLETE.json").write_text(
        json.dumps(complete, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
