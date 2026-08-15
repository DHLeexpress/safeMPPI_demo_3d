#!/usr/bin/env python3
"""Generate deterministic governed references from one frozen lab flow policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_deployment.lab_pretrained import (  # noqa: E402
    DEFAULT_CHECKPOINT_SHA256,
    DEFAULT_CONFIG_SHA256,
    load_lab_deployment_controller,
    sha256_file,
)
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402


def gamma_name(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def save_overlay(rows: list[dict], env: TaskEnvironment, output: Path) -> None:
    fig = plt.figure(figsize=(8.1, 6.3))
    axis = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("plasma")(
        np.linspace(0.08, 0.92, len(rows))
    )
    for color, row in zip(colors, rows):
        path = row["dense_positions"]
        axis.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linewidth=2.2,
            label=rf"$\gamma={row['gamma']:g}$ ({row['status']})",
        )
    for sphere in env.spheres:
        center, radius = sphere[:3], float(sphere[3])
        u = np.linspace(0.0, 2.0 * np.pi, 40)
        v = np.linspace(0.0, np.pi, 24)
        axis.plot_surface(
            center[0] + radius * np.outer(np.cos(u), np.sin(v)),
            center[1] + radius * np.outer(np.sin(u), np.sin(v)),
            center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
            color="0.65",
            alpha=0.28,
            linewidth=0,
        )
    axis.scatter(*env.start[:3], marker="s", s=55, color="black")
    axis.scatter(*env.goal, marker="*", s=130, color="gold", edgecolor="black")
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    axis.legend(loc="best")
    axis.set_title("Frozen pretrained-flow references")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=(0.1, 0.3, 0.5, 1.0),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(91000,),
        help="one seed broadcasts to every gamma; otherwise one seed per gamma",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.seeds) not in {1, len(args.gammas)}:
        parser.error("--seeds must contain one value or one value per gamma")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

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
    policy = controller.policy
    seeds = (
        args.seeds * len(args.gammas)
        if len(args.seeds) == 1
        else args.seeds
    )
    rows = []
    for gamma, seed in zip(args.gammas, seeds):
        result = raw_reference_rollout(
            policy,
            config,
            gamma,
            seed,
            device=args.device,
            sampling_temperature=args.sampling_temperature,
        )
        dense_positions = np.concatenate([
            result["states"][:1, :3],
            result["dense_steps"].reshape(-1, 3),
        ])
        name = f"gamma_{gamma_name(gamma)}_seed_{seed}"
        np.savez_compressed(
            args.output / f"{name}.npz",
            dense_positions=dense_positions,
            states=result["states"],
            controls=result["controls"],
            executed_controls=result["applied_controls"],
            dense_steps=result["dense_steps"],
            gamma=np.float32(gamma),
            seed=np.int64(seed),
            sampling_temperature=np.float32(args.sampling_temperature),
            status=np.asarray(result["status"]),
        )
        rows.append({
            "gamma": float(gamma),
            "seed": int(seed),
            "status": result["status"],
            "mode": result["mode"],
            "window_validity": float(result["window_validity"]),
            "min_clearance_m": result["min_clearance_m"],
            "time_to_goal_s": result["time_to_goal_s"],
            "file": f"{name}.npz",
            "dense_positions": dense_positions,
        })

    save_overlay(rows, env, args.output / "frozen_references.png")
    manifest = {
        "status": "LAB_FLOW_FROZEN_REFERENCES_COMPLETE",
        "scope": (
            "Open-loop governed reference export. The policy is replanned "
            "against its simulated reference state while generating the file; "
            "the exported file itself contains no later state feedback."
        ),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "policy": policy_contract,
        "runs": [
            {key: value for key, value in row.items() if key != "dense_positions"}
            for row in rows
        ],
        "all_successful": all(row["status"] == "SUCCESS" for row in rows),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
