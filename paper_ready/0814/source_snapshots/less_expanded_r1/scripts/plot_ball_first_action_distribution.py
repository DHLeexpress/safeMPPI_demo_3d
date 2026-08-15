"""Plot canonical raw first-action distributions across expansion checkpoints.

The policy is sampled directly at the canonical start with standard flow base noise.  Candidate
perturbation, RBF acquisition, verification, and execution ranking are never called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import build_context, load_policy
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


def sliced_wasserstein_1(samples: np.ndarray, reference: np.ndarray,
                         directions: np.ndarray) -> float:
    """Deterministic sliced-W1 distance in action units."""
    left = np.sort(np.asarray(samples, float) @ directions.T, axis=0)
    right = np.sort(np.asarray(reference, float) @ directions.T, axis=0)
    return float(np.mean(np.abs(left - right)))


def checkpoint_policy(pretrain_dir: Path, expansion: Path, round_i: int):
    policy = load_policy(pretrain_dir / "pretrained.pt")
    payload = torch.load(expansion / f"checkpoint_{round_i:03d}.pt", weights_only=False)
    policy.load_state_dict(payload["model"])
    policy.eval()
    return policy


@torch.no_grad()
def sample_first_actions(policy, context: torch.Tensor, count: int, seed: int) -> np.ndarray:
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    return policy.sample(context, count, generator)[:, 0, :].cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=(0, 2, 4, 6, 8, 10))
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=24017)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 32:
        raise ValueError("--samples must be at least 32")

    config = load_config(args.pretrain_dir / "demo_config.json")
    env = TaskEnvironment(config)
    gammas = tuple(config.data.gammas)
    rounds = tuple(args.rounds)
    rng = np.random.default_rng(args.seed + 991)
    directions = rng.normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    colors = {
        round_i: plt.get_cmap("viridis")(index / max(len(rounds) - 1, 1))
        for index, round_i in enumerate(rounds)
    }

    samples: dict[tuple[int, float], np.ndarray] = {}
    for gamma_index, gamma in enumerate(gammas):
        context = torch.from_numpy(build_context(env, env.start, float(gamma)))
        for round_i in rounds:
            policy = checkpoint_policy(args.pretrain_dir, args.expansion, round_i)
            # Resetting the same generator per checkpoint gives a round-independent CRN bank.
            samples[(round_i, gamma)] = sample_first_actions(
                policy, context, args.samples, args.seed + 1009 * gamma_index)

    rows = []
    fig = plt.figure(figsize=(15.4, 12.2))
    for panel, gamma in enumerate(gammas, start=1):
        axis = fig.add_subplot(2, 2, panel, projection="3d")
        reference = samples[(rounds[0], gamma)]
        previous = None
        for round_i in rounds:
            values = samples[(round_i, gamma)]
            distance_r0 = sliced_wasserstein_1(values, reference, directions)
            distance_previous = (
                0.0 if previous is None
                else sliced_wasserstein_1(values, previous, directions)
            )
            mean = values.mean(axis=0)
            rows.append({
                "round": int(round_i), "gamma": float(gamma),
                "mean_first_action": mean.tolist(),
                "p_a_z_positive": float(np.mean(values[:, 2] > 0.0)),
                "sliced_w1_vs_round0": distance_r0,
                "sliced_w1_vs_previous_plotted_round": distance_previous,
            })
            indices = np.linspace(0, len(values) - 1, min(180, len(values))).round().astype(int)
            axis.scatter(
                values[indices, 0], values[indices, 1], values[indices, 2],
                color=colors[round_i], s=7, alpha=0.10,
            )
            axis.scatter(
                *mean, color=colors[round_i], edgecolor="black", linewidth=0.45, s=62,
                label=(rf"$r={round_i}$: $P(a_z>0)={np.mean(values[:, 2] > 0):.2f}$, "
                       rf"$SW_1(p_r,p_0)={distance_r0:.3f}$"),
            )
            previous = values
        axis.set_title(rf"$\gamma={gamma:g}$", fontsize=17)
        axis.set_xlabel(r"$a_x$ [$\mathrm{m/s^2}$]")
        axis.set_ylabel(r"$a_y$ [$\mathrm{m/s^2}$]")
        axis.set_zlabel(r"$a_z$ [$\mathrm{m/s^2}$]")
        axis.set_xlim(-1.02, 1.02)
        axis.set_ylim(-1.02, 1.02)
        axis.set_zlim(-1.02, 1.02)
        axis.view_init(elev=22, azim=-58)
        axis.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(
        "Canonical temperature-1 first-action distributions\n"
        "(fixed start, common base noise; no 0.2 candidate perturbation)",
        fontsize=20, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=210, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    arrays = {
        f"r{round_i}_g{gamma:g}".replace(".", "p"): values
        for (round_i, gamma), values in samples.items()
    }
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(json.dumps({
        "status": "CANONICAL_RAW_FIRST_ACTION_DISTRIBUTION_COMPLETE",
        "rounds": list(rounds), "gammas": list(gammas),
        "samples_per_cell": args.samples, "seed": args.seed,
        "sampling_temperature": 1.0,
        "common_random_numbers_across_rounds": True,
        "candidate_perturbation_applied": False,
        "distance": "256-direction deterministic sliced Wasserstein-1",
        "rows": rows,
    }, indent=2) + "\n")
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
