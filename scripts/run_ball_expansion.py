"""Run B1 Safe Flow Expansion on the ball task with full per-step event logging.

Loads the pretrained 10-D-context flow policy, wires the BallFlowTask (GREEN rebuilt-polytope
verifier + untilted native SafeMPPI execution cost) into ``run_safe_expansion``, and stores every
acquisition event (all K candidates, marginal sigma, selected B, verifier verdicts, executed
index, robot state) for the mechanism/representation analyses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import BallFlowTask, load_policy
from safe_mppi.config import load_config
from safe_mppi.expansion import ExpansionConfig, run_safe_expansion

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, default=ROOT / "outputs" / "ball_flow")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "ball_flow" / "expansion")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--beta", type=float, default=0.003,
                        help="really small = near-greedy top-sigma acquisition; each round "
                             "incrementally queries the newest region of feature space")
    parser.add_argument("--start-diversity", action="store_true", default=False)
    parser.add_argument("--no-start-diversity", dest="start_diversity", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = load_policy(args.pretrain_dir / "pretrained.pt")
    pretrain = json.loads((args.pretrain_dir / "pretrain_manifest.json").read_text())
    calibration = torch.load(args.pretrain_dir / "calibration_features.pt",
                             weights_only=False)
    task_config = load_config(args.pretrain_dir / "demo_config.json")
    task = BallFlowTask(task_config, start_diversity=args.start_diversity)

    config = ExpansionConfig(
        rounds=args.rounds, gammas=tuple(task_config.data.gammas), parallel_episodes=2,
        max_steps=task_config.taskspace.max_steps, K=16, B=4, batch_size=32,
        inner_steps=None, learning_rate=args.learning_rate, replay_rounds=2,
        gp_buffer_cap=256, gp_noise=1.0e-2, rbf_lengthscale=None,
        beta=float(args.beta),
        adaptive_beta=False, negative_alpha=0.0, seed=args.seed,
    )

    events = []

    def callback(event):
        events.append({
            "round": event["round"], "step": event["step"], "gamma": event["gamma"],
            "episode": event["episode"], "context_id": event["context_id"],
            "robot": np.asarray(event["state_before"]["x"], np.float32),
            "context": event["context"].numpy(),
            "candidates": event["candidates"].numpy().astype(np.float32),
            "sigma_K": event["sigma_K"].numpy().astype(np.float32),
            "selected": list(event["selected"]),
            "selected_sigma": list(event["selected_sigma"]),
            "verification": event["verification"],
            "chosen_local": event["chosen_local"],
            "status": event["status"],
        })

    started = time.perf_counter()
    manifest = run_safe_expansion(policy, task, args.output, config=config,
                                  calibration_features=calibration,
                                  event_callback=callback)
    torch.save(events, args.output / "events.pt")
    print(f"[expansion] rounds={args.rounds} events={len(events)} "
          f"D={manifest['D']} D+={manifest['D_plus']} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    for row in manifest["rounds"]:
        print(f"  round {row['round']:2d}: positives {row['positives']:3d}/{row['queries']:3d} "
              f"success {row['success']}/8 NVP {row['NVP']:3d} "
              f"loss {row['positive_loss']}", flush=True)


if __name__ == "__main__":
    main()
