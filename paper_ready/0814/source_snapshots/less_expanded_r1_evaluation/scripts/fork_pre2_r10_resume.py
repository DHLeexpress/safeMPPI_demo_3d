#!/usr/bin/env python3
"""Create a non-destructive r10 continuation fork from committed artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.expansion import (  # noqa: E402
    _record_identity_hash,
    _sliding_success_gp_rows,
    _trunk_suffix_parameters,
)
from safe_mppi.lab_flow_expansion import load_lab_expansion_policy  # noqa: E402


def _copy(source: Path, output: Path, name: str) -> None:
    path = source / name
    if not path.is_file():
        raise FileNotFoundError(path)
    shutil.copy2(path, output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=10)
    parser.add_argument("--trainable-trunk-layers", type=int, default=3)
    args = parser.parse_args()

    if args.round != 10:
        parser.error("this audited fork currently supports only round 10")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    latest = torch.load(
        args.source / "resume_state_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = torch.load(
        args.source / f"checkpoint_{args.round:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = dict(checkpoint["config"])
    if config["optimizer_steps_total"] is not None:
        raise ValueError("fork requires per-round optimizer steps")
    if float(config["round_learning_rate_warmup_power"]) != 0.0:
        raise ValueError("fork requires no outer-round LR schedule")

    for round_i in range(args.round + 1):
        _copy(args.source, args.output, f"checkpoint_{round_i:03d}.pt")
    for round_i in range(1, args.round + 1):
        _copy(args.source, args.output, f"query_archive_round_{round_i:03d}.pt")
        _copy(args.source, args.output, f"events_round_{round_i:03d}.pt")
    _copy(args.source, args.output, "task_config_resolved.json")

    manifest_source = args.source / f"manifest_before_resume_round_{args.round:03d}.json"
    if not manifest_source.is_file():
        raise FileNotFoundError(manifest_source)
    shutil.copy2(manifest_source, args.output / "manifest.json")
    manifest = json.loads(manifest_source.read_text())
    if len(manifest["rounds"]) != args.round:
        raise ValueError("historical r10 manifest does not end at round 10")

    metrics = (args.source / "metrics.jsonl").read_text().splitlines()
    (args.output / "metrics.jsonl").write_text(
        "\n".join(metrics[:args.round]) + "\n"
    )
    first_action = json.loads((args.source / "first_action_stats.json").read_text())
    first_action["rows"] = [
        row for row in first_action.get("rows", [])
        if int(row.get("round", -1)) <= args.round
    ]
    (args.output / "first_action_stats.json").write_text(
        json.dumps(first_action, indent=2) + "\n"
    )
    fa = json.loads((args.source / "fa_alloc_log.json").read_text())
    fa["retry_progress"] = [
        row for row in fa.get("retry_progress", [])
        if int(row.get("round", -1)) <= args.round
    ]
    fa["rounds"] = {
        key: value for key, value in fa.get("rounds", {}).items()
        if int(key) <= args.round
    }
    (args.output / "fa_alloc_log.json").write_text(
        json.dumps(fa, indent=2) + "\n"
    )

    archive = [row for row in latest["archive"] if int(row.round) <= args.round]
    gp_evidence = [
        row for row in latest["gp_evidence"] if int(row.round) <= args.round
    ]
    expected_rows = sum(
        len(torch.load(
            args.source / f"query_archive_round_{round_i:03d}.pt",
            map_location="cpu",
            weights_only=False,
        ))
        for round_i in range(1, args.round + 1)
    )
    if len(archive) != expected_rows or len(gp_evidence) != expected_rows:
        raise ValueError(
            f"r10 replay mismatch: archive={len(archive)} gp={len(gp_evidence)} "
            f"query_rows={expected_rows}"
        )

    policy = load_lab_expansion_policy(
        args.pretrain_dir / "pretrained.pt"
    ).to("cpu")
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.freeze_visual_encoder_for_expansion()
    parameters = _trunk_suffix_parameters(
        policy, args.trainable_trunk_layers,
    )
    names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    if names != latest["trainable_parameter_names"]:
        raise ValueError("r10 fork trainable parameter order differs from source")
    optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))
    optimizer_step = int(manifest["rounds"][-1]["optimizer_step"])
    active_gp = _sliding_success_gp_rows(
        gp_evidence,
        config["gammas"],
        int(config["gp_buffer_cap"]),
        through_round=args.round - 1,
        selector=config["gp_sliding_row_selector"],
    )
    resume = {
        "version": latest["version"],
        "status": "COMMITTED_ROUND_RESUME",
        "completed_round": args.round,
        "config": config,
        "model": checkpoint["model"],
        "optimizer": optimizer.state_dict(),
        "optimizer_metadata": {
            "_safe_mppi_schedule_step": optimizer_step,
            "_safe_mppi_base_lrs": (float(config["learning_rate"]),),
            "_safe_mppi_round_lr_scale": 1.0,
        },
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "beta": float(config["beta"]),
        "round_rows": list(manifest["rounds"]),
        "archive": archive,
        "frozen_gp_rows": None,
        "frozen_gp_hash": _record_identity_hash(active_gp),
        "round1_gp_candidates": [],
        "gp_evidence": gp_evidence,
        "cumulative_anchors": {
            float(gamma): [] for gamma in config["gammas"]
        },
        "cumulative_adaptive": {
            float(gamma): [] for gamma in config["gammas"]
        },
        # The source retained no r10 RNG snapshot. Reusing the valid CUDA
        # Philox states from its later committed snapshot lets paired forks
        # share an identical future stream without inventing a device state.
        "numpy_rng_state": latest["numpy_rng_state"],
        "torch_rng_state": latest["torch_rng_state"],
        "torch_cpu_rng_state": latest["torch_cpu_rng_state"],
        "torch_device_rng_state": latest["torch_device_rng_state"],
    }
    torch.save(resume, args.output / "resume_state_latest.pt")
    (args.output / "resume_state.json").write_text(json.dumps({
        "status": "COMMITTED_ROUND_RESUME",
        "version": latest["version"],
        "completed_round": args.round,
        "next_round": args.round + 1,
        "optimizer_step": optimizer_step,
        "flow_base_std_next": float(config["flow_base_std"]),
        "resume_state": "resume_state_latest.pt",
    }, indent=2) + "\n")
    provenance = {
        "status": "R10_FORK_READY",
        "source": str(args.source.resolve()),
        "source_checkpoint": str(
            (args.source / f"checkpoint_{args.round:03d}.pt").resolve()
        ),
        "completed_round": args.round,
        "replay_rows": len(archive),
        "optimizer_step": optimizer_step,
        "optimizer_state": (
            "cold-start Adam: the in-place source retained only its r15 "
            "optimizer snapshot"
        ),
        "rng_state": (
            "paired borrowed CUDA Philox/NumPy state from the source r15 "
            "snapshot: the r10 RNG snapshot was not retained"
        ),
        "causal_use": (
            "compare paired forks made from this script; absolute r10->r11 "
            "change also includes the disclosed cold-start Adam"
        ),
    }
    (args.output / "FORK_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
