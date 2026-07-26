#!/usr/bin/env python3
"""One-time fixed-beta calibration for a declared candidate perturbation law."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import demo_windows, load_policy
from safe_mppi.expansion import (RBFPosterior, calibrate_fixed_beta,
                                 perturb_plan_candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--perturb-std", type=float, required=True)
    parser.add_argument("--perturb-scope", choices=("first_action", "coherent_horizon"),
                        default="coherent_horizon")
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    policy = load_policy(args.pretrain_dir / "pretrained.pt")
    contexts_np, plans_np, _, _ = demo_windows(args.pretrain_dir / "demos")
    contexts, plans = torch.from_numpy(contexts_np), torch.from_numpy(plans_np)
    manifest = json.loads((args.pretrain_dir / "pretrain_manifest.json").read_text())
    gp = RBFPosterior(float(manifest["rbf_lengthscale"]), 1.0e-2)
    generator = torch.Generator().manual_seed(args.seed)
    buffer_ids = torch.randperm(len(contexts), generator=generator)[:128]
    gp.set_buffer(policy.embed(contexts[buffer_ids], plans[buffer_ids]))
    pools = []
    for index in torch.randperm(len(contexts), generator=generator)[:24].tolist():
        candidates = policy.sample(contexts[index], 16, generator)
        candidates = perturb_plan_candidates(
            policy, candidates, args.perturb_std, generator, args.perturb_scope)
        pools.append(gp.sigma(policy.embed(contexts[index], candidates)))
    beta = calibrate_fixed_beta(pools, args.ess_target)
    payload = {
        "status": "FIXED_BETA_CALIBRATED",
        "pretrain_dir": str(args.pretrain_dir.resolve()),
        "perturb_std": args.perturb_std,
        "perturb_scope": args.perturb_scope,
        "ess_target": args.ess_target,
        "beta": beta,
        "pools": len(pools),
    }
    output = args.output or args.pretrain_dir / (
        f"beta_{args.perturb_scope}_p{args.perturb_std:g}.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
