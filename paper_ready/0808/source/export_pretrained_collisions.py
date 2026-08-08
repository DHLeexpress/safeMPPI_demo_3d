#!/usr/bin/env python3
"""Export deterministic P0806-pretrained collision examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gamma_tag(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def bit_identical(first: dict, second: dict) -> bool:
    keys = ("states", "controls", "applied_controls", "dense_steps")
    return (
        first["status"] == second["status"]
        and all(np.array_equal(first[key], second[key]) for key in keys)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-nfe", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_sha = sha256_file(args.checkpoint)
    config_sha = sha256_file(args.config)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {checkpoint_sha}")
    if config_sha != args.expected_config_sha256:
        raise SystemExit(f"config SHA mismatch: {config_sha}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    packaged_nfe = int(payload["arch"]["nfe"])
    if packaged_nfe != args.expected_nfe:
        raise SystemExit(f"packaged NFE mismatch: {packaged_nfe}")

    from safe_mppi.config import load_config
    from safe_mppi.lab_reference_flow_task import raw_reference_rollout
    from safe_mppi.lab_visual_flow import load_lab_reference_policy

    config = load_config(args.config)
    policy = load_lab_reference_policy(args.checkpoint).to(args.device).eval()
    selections = json.loads(args.selections.read_text())
    selected = [
        row for row in selections["trajectories"]
        if int(row["physical_gpu"]) == args.physical_gpu
    ]
    if not selected:
        raise SystemExit(f"no selections for physical GPU {args.physical_gpu}")

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for selection in selected:
        gamma = float(selection["gamma"])
        seed = int(selection["seed"])
        expected_status = str(selection["expected_status"])
        runs = []
        for _ in range(args.repeat):
            with torch.no_grad():
                runs.append(raw_reference_rollout(
                    policy,
                    config,
                    gamma,
                    seed,
                    device=args.device,
                    sampling_temperature=args.sampling_temperature,
                ))
        if any(not bit_identical(runs[0], run) for run in runs[1:]):
            raise SystemExit(f"non-bit-identical repeat: gamma={gamma:g}, seed={seed}")
        result = runs[0]
        if result["status"] != expected_status:
            raise SystemExit(
                f"status mismatch: gamma={gamma:g}, seed={seed}, "
                f"{result['status']} != {expected_status}"
            )
        if result["min_clearance_m"] >= 0.0:
            raise SystemExit(f"collision without penetration: gamma={gamma:g}, seed={seed}")

        filename = f"gamma_{gamma_tag(gamma)}_collision_seed_{seed}.npz"
        path = args.output / filename
        np.savez_compressed(
            path,
            states=result["states"],
            controls=result["controls"],
            applied_controls=result["applied_controls"],
            dense_steps=result["dense_steps"],
            status=np.str_(result["status"]),
            gamma=np.float32(gamma),
            seed=np.int64(seed),
            sampling_temperature=np.float32(args.sampling_temperature),
            min_clearance_m=np.float64(result["min_clearance_m"]),
            checkpoint_sha256=np.str_(checkpoint_sha),
            config_sha256=np.str_(config_sha),
            source_id=np.str_(args.source_id),
            physical_gpu=np.int64(args.physical_gpu),
        )
        records.append({
            "file": filename,
            "sha256": sha256_file(path),
            "gamma": gamma,
            "seed": seed,
            "status": result["status"],
            "min_clearance_m": result["min_clearance_m"],
            "steps": int(len(result["controls"])),
            "physical_gpu": args.physical_gpu,
            "repeat_count": args.repeat,
            "repeat_verification": "BIT_IDENTICAL",
        })

    manifest = {
        "schema": "paper_ready_pretrained_collision_rollouts_v1",
        "policy_label": "pretrained_p0806",
        "checkpoint_sha256": checkpoint_sha,
        "packaged_nfe": packaged_nfe,
        "config_sha256": config_sha,
        "source_id": args.source_id,
        "physical_gpu": args.physical_gpu,
        "sampling_temperature": args.sampling_temperature,
        "records": records,
    }
    manifest_path = args.output / f"manifest_gpu{args.physical_gpu}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
