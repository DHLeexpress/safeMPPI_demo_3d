#!/usr/bin/env python3
"""Compare same-data pretraining arms in learning and raw deployment space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_arm(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--arm must be label=pretrain_directory")
    label, value = raw.split("=", 1)
    return label, Path(value)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _deployment_summary(rows: list[dict]) -> dict:
    counts = {mode: 0 for mode in ("below", "above", "left", "right")}
    for row in rows:
        for mode, count in row["route_counts"].items():
            counts[mode] += int(count)
    lateral_total = counts["left"] + counts["right"]
    return {
        "episodes": sum(int(row["episodes"]) for row in rows),
        "mean_gamma_SR": _mean(rows, "SR"),
        "mean_gamma_collision_rate": _mean(rows, "CR"),
        "mean_gamma_OOB_rate": _mean(rows, "OOB"),
        "mean_gamma_timeout_rate": _mean(rows, "timeout"),
        "mean_gamma_window_validity": _mean(rows, "window_validity"),
        "mean_successful_min_clearance_m": _mean(
            rows, "successful_min_clearance_m"
        ),
        "mean_successful_time_to_goal_s": _mean(
            rows, "successful_time_to_goal_s"
        ),
        "route_counts": counts,
        "left_right_balance": (
            1.0 - abs(counts["left"] - counts["right"]) / lateral_total
            if lateral_total else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument(
        "--deployment-reference",
        action="append",
        default=[],
        help="optional label=pretrain_directory; excluded from same-data proof",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = {}
    for raw in args.arm:
        label, directory = _parse_arm(raw)
        manifest_path = directory / "pretrain_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        method = manifest.get("training_method", {})
        arms[label] = {
            "directory": str(directory.resolve()),
            "source_archive_sha256": manifest["source_archive_digest"]["sha256"],
            "training_windows": int(manifest["training_windows"]),
            "validation_windows": int(manifest["validation_windows"]),
            "selected_epoch": int(manifest["selected_epoch"]),
            "selected_validation_loss": float(manifest["selected_valid_loss"]),
            "deterministic_validation": method.get(
                "selected_checkpoint_validation_audit"
            ),
            "training_method": {
                key: method.get(key) for key in (
                    "batch_sampler", "optimizer", "weight_decay",
                    "warmup_epochs", "gradient_clip_norm", "ema_decay",
                    "selected_checkpoint_parameter_source",
                )
            },
            "batch_balance_audit": method.get("batch_balance_audit"),
            "parameter_update_audit": method.get("parameter_update_audit"),
            "single_sphere_pre_expansion": _deployment_summary(
                manifest["ood_raw_audit_summary"]
            ),
            "single_sphere_by_gamma": manifest["ood_raw_audit_summary"],
        }

    digests = {row["source_archive_sha256"] for row in arms.values()}
    train_counts = {row["training_windows"] for row in arms.values()}
    validation_counts = {row["validation_windows"] for row in arms.values()}
    deployment_references = {}
    for raw in args.deployment_reference:
        label, directory = _parse_arm(raw)
        manifest = json.loads((directory / "pretrain_manifest.json").read_text())
        deployment_references[label] = {
            "directory": str(directory.resolve()),
            "different_training_data": True,
            "single_sphere_pre_expansion": _deployment_summary(
                manifest["ood_raw_audit_summary"]
            ),
            "single_sphere_by_gamma": manifest["ood_raw_audit_summary"],
        }

    payload = {
        "contract": {
            "same_source_archive": len(digests) == 1,
            "same_training_window_count": len(train_counts) == 1,
            "same_validation_window_count": len(validation_counts) == 1,
            "proof_axes": [
                "deterministic held-out CFM loss by gamma",
                "full-network encoder/trunk/head parameter movement",
                "fixed-seed pre-expansion single-sphere deployment",
            ],
        },
        "arms": arms,
        "deployment_references": deployment_references,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )

    lines = [
        "# PRE2 same-data training proof",
        "",
        "| arm | fixed CFM loss | SR | collision | validity | clearance | time-to-goal | L/R balance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in arms.items():
        deployment = row["single_sphere_pre_expansion"]
        validation = row["deterministic_validation"] or {}
        lines.append(
            f"| {label} | {validation.get('mean_cfm_loss', float('nan')):.6f} "
            f"| {deployment['mean_gamma_SR']:.3f} "
            f"| {deployment['mean_gamma_collision_rate']:.3f} "
            f"| {deployment['mean_gamma_window_validity']:.3f} "
            f"| {deployment['mean_successful_min_clearance_m']:.3f} "
            f"| {deployment['mean_successful_time_to_goal_s']:.3f} "
            f"| {deployment['left_right_balance']:.3f} |"
        )
    if deployment_references:
        lines.extend([
            "",
            "Deployment-only reference (different training data; excluded "
            "from the neural optimization proof):",
            "",
            "| reference | SR | collision | validity | clearance | time-to-goal | L/R balance |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, row in deployment_references.items():
            deployment = row["single_sphere_pre_expansion"]
            lines.append(
                f"| {label} | {deployment['mean_gamma_SR']:.3f} "
                f"| {deployment['mean_gamma_collision_rate']:.3f} "
                f"| {deployment['mean_gamma_window_validity']:.3f} "
                f"| {deployment['mean_successful_min_clearance_m']:.3f} "
                f"| {deployment['mean_successful_time_to_goal_s']:.3f} "
                f"| {deployment['left_right_balance']:.3f} |"
            )
    lines.extend([
        "",
        "The deployment table is the fixed-seed single-sphere PRE baseline "
        "before any expansion or guidance.",
    ])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
