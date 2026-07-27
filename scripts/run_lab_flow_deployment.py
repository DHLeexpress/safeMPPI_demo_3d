#!/usr/bin/env python3
"""Run the frozen lab flow policy in Minhyuk's unchanged deployment harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "deploy_sim"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deploy_sim import run_offline as native  # noqa: E402
from flow_deployment.bridge import verify_deploy_sim_lock  # noqa: E402
from flow_deployment.lab_pretrained import (  # noqa: E402
    DEFAULT_CHECKPOINT_SHA256,
    DEFAULT_CONFIG_SHA256,
    load_lab_deployment_controller,
    sha256_file,
)
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_ball_pretrain.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=DEFAULT_CONFIG_SHA256,
    )
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    deploy_sim_lock_before = verify_deploy_sim_lock(ROOT)
    config_sha256 = sha256_file(args.config)
    if config_sha256 != args.expected_config_sha256:
        raise ValueError(
            "config SHA-256 mismatch: "
            f"expected {args.expected_config_sha256}, got {config_sha256}"
        )
    config = load_config(args.config)
    env = TaskEnvironment(config)
    controller, policy_contract = load_lab_deployment_controller(
        args.checkpoint,
        env,
        sampling_temperature=args.sampling_temperature,
        device=args.device,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    swarm = native.Swarm(
        np.array([env.start[0], env.start[1], 0.0]),
        seed=args.seed,
    )
    result = native.harness.run(
        controller,
        env,
        config,
        swarm,
        gamma=args.gamma,
        seed=args.seed,
        verbose=True,
    )
    metrics = native.harness.summarize(result, env, verbose=True)
    native_base = native.save(
        result,
        env,
        config,
        args.output,
        args.gamma,
        args.gif,
        True,
    )
    contract = {
        "status": "LAB_FLOW_DEPLOYMENT_COMPLETE",
        "scope": (
            "Online state-feedback inference in the unchanged calibrated "
            "deploy_sim harness; not a flight safety certificate."
        ),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "policy": policy_contract,
        "state_seen_by_policy": (
            "current measured plant position p_meas concatenated with the "
            "harness reference velocity v_ref"
        ),
        "gamma": float(args.gamma),
        "seed": int(args.seed),
        "metrics": metrics,
        "deploy_sim_lock_before": deploy_sim_lock_before,
        "deploy_sim_lock_after": verify_deploy_sim_lock(ROOT),
        "native_output_base": (
            None if native_base is None else str(Path(native_base).resolve())
        ),
    }
    (args.output / "deployment_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    print(json.dumps(contract, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
