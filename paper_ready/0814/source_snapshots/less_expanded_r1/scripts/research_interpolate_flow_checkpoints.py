"""Build evaluator-compatible linear interpolants between two flow checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch


def _interpolate(early: dict, late: dict, alpha: float) -> dict:
    if early.keys() != late.keys():
        raise ValueError("checkpoint state dictionaries have different keys")
    result = {}
    for name in late:
        left, right = late[name], early[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"incompatible checkpoint tensor {name}")
        result[name] = (
            left.lerp(right, alpha)
            if left.is_floating_point()
            else left.clone()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-expansion", type=Path, required=True)
    parser.add_argument("--early-round", type=int, required=True)
    parser.add_argument("--late-round", type=int, required=True)
    parser.add_argument("--alpha", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    alphas = [float(value) for value in args.alpha]
    if not alphas or any(not 0.0 < value < 1.0 for value in alphas):
        parser.error("every --alpha must lie strictly between zero and one")

    early_path = args.source_expansion / f"checkpoint_{args.early_round:03d}.pt"
    late_path = args.source_expansion / f"checkpoint_{args.late_round:03d}.pt"
    early_payload = torch.load(early_path, map_location="cpu", weights_only=False)
    late_payload = torch.load(late_path, map_location="cpu", weights_only=False)
    args.output.mkdir(parents=True)
    torch.save(
        {
            **late_payload,
            "round": 0,
            "interpolation": {
                "alpha_toward_early": 0.0,
                "early_checkpoint": str(early_path.resolve()),
                "late_checkpoint": str(late_path.resolve()),
            },
        },
        args.output / "checkpoint_000.pt",
    )
    rows = []
    for round_i, alpha in enumerate(alphas, start=1):
        payload = {
            **late_payload,
            "round": round_i,
            "model": _interpolate(
                early_payload["model"], late_payload["model"], alpha,
            ),
            "interpolation": {
                "alpha_toward_early": alpha,
                "early_checkpoint": str(early_path.resolve()),
                "late_checkpoint": str(late_path.resolve()),
            },
        }
        torch.save(payload, args.output / f"checkpoint_{round_i:03d}.pt")
        rows.append({"round": round_i, "alpha_toward_early": alpha})

    shutil.copy2(
        args.source_expansion / "task_config_resolved.json",
        args.output / "task_config_resolved.json",
    )
    manifest = {
        "kind": "flow checkpoint interpolation screen",
        "config": {"rounds": len(alphas)},
        "source_expansion": str(args.source_expansion.resolve()),
        "early_round": int(args.early_round),
        "late_round": int(args.late_round),
        "checkpoint_000": "unmodified late checkpoint",
        "interpolants": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
