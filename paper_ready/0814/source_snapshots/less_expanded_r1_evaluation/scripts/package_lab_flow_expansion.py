#!/usr/bin/env python3
"""Package one expansion state as a self-contained lab deployment checkpoint."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expansion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256")
    parser.add_argument("--expected-expansion-sha256")
    parser.add_argument(
        "--flow-nfe", type=int,
        help="optional deployment Euler-step override recorded in arch.nfe",
    )
    args = parser.parse_args()
    if args.flow_nfe is not None and args.flow_nfe < 1:
        parser.error("--flow-nfe must be positive")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    pretrained_sha = sha256_file(args.pretrained)
    expansion_sha = sha256_file(args.expansion_checkpoint)
    if (
        args.expected_pretrained_sha256 is not None
        and pretrained_sha != args.expected_pretrained_sha256
    ):
        raise ValueError("pretrained checkpoint SHA-256 mismatch")
    if (
        args.expected_expansion_sha256 is not None
        and expansion_sha != args.expected_expansion_sha256
    ):
        raise ValueError("expansion checkpoint SHA-256 mismatch")

    pretrained = torch.load(
        args.pretrained, map_location="cpu", weights_only=False,
    )
    expansion = torch.load(
        args.expansion_checkpoint, map_location="cpu", weights_only=False,
    )
    for key in ("arch", "contract", "model"):
        if key not in pretrained:
            raise KeyError(f"pretrained checkpoint is missing {key!r}")
    for key in ("round", "config", "model"):
        if key not in expansion:
            raise KeyError(f"expansion checkpoint is missing {key!r}")

    base_state = pretrained["model"]
    expanded_state = expansion["model"]
    if base_state.keys() != expanded_state.keys():
        raise ValueError("pretrained and expanded state keys differ")
    for key in base_state:
        base = base_state[key]
        expanded = expanded_state[key]
        if base.shape != expanded.shape or base.dtype != expanded.dtype:
            raise ValueError(f"incompatible expanded tensor {key!r}")
        if not bool(torch.isfinite(expanded).all()):
            raise ValueError(f"non-finite expanded tensor {key!r}")

    contract = dict(pretrained["contract"])
    contract.update({
        "scope": "experimental_safe_flow_expansion_checkpoint",
        "deployment_safety_qualified": False,
    })
    arch = dict(pretrained["arch"])
    if args.flow_nfe is not None:
        arch["nfe"] = int(args.flow_nfe)
    payload = {
        "model": expanded_state,
        "arch": arch,
        "contract": contract,
        "provenance": {
            "pretrained_checkpoint_sha256": pretrained_sha,
            "expansion_checkpoint_sha256": expansion_sha,
            "expansion_round": int(expansion["round"]),
            "expansion_config": expansion["config"],
            "packaged_flow_nfe": int(arch["nfe"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"{sha256_file(args.output)}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
