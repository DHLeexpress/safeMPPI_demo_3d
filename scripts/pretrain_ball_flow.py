"""Pretrain the 10-D-context conditional flow policy on biased below-ball demonstrations.

Steps: (1) collect 10 SafeMPPI demonstrations per gamma with configs/ball_biased_demo.json,
(2) fit the CFM policy on sliding H=10 windows, (3) audit the raw pretrained policy per gamma,
(4) calibrate the RBF length scale (50 embeddings) and the fixed acquisition beta (ESS 0.5).
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

from safe_mppi.acquire import acquire
from safe_mppi.ball_flow_task import PLAN_H, build_context, demo_windows, raw_rollout
from safe_mppi.environment import TaskEnvironment
from safe_mppi.expansion import RBFPosterior, calibrate_fixed_beta, mean_pairwise_lengthscale
from safe_mppi.flow_model import ConditionalFlowMLP

ROOT = Path(__file__).resolve().parents[1]


def collect_demos(output: Path, episodes_per_gamma: int) -> Path:
    config = json.loads((ROOT / "configs" / "ball_biased_demo.json").read_text())
    config["data"]["episodes_per_gamma"] = int(episodes_per_gamma)
    config_path = output / "demo_config.json"
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    demo_dir = output / "demos"
    if not (demo_dir / "manifest.json").exists():
        acquire(config_path, demo_dir, device="cpu")
    return demo_dir


def augment_windows(env, contexts_np, plans_np, meta, copies: int, jitter: float, seed: int):
    """Geometry-consistent context jitter: perturb [p, v], rebuild the 10-D context exactly."""
    rng = np.random.default_rng(seed)
    rows_c, rows_p = [contexts_np], [plans_np]
    states = np.stack([np.concatenate([env.goal - c[:3], c[3:6]]) for c in contexts_np])
    gammas = contexts_np[:, 9]
    for _ in range(copies):
        noisy = states + rng.normal(0.0, jitter, size=states.shape).astype(np.float32)
        rows_c.append(np.stack([build_context(env, noisy[i], float(gammas[i]))
                                for i in range(len(noisy))]))
        rows_p.append(plans_np)
    return np.concatenate(rows_c).astype(np.float32), np.concatenate(rows_p)


def train(policy: ConditionalFlowMLP, contexts: torch.Tensor, plans: torch.Tensor,
          epochs: int, batch_size: int, learning_rate: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    holdout = max(1, int(0.1 * len(contexts)))
    order = torch.randperm(len(contexts), generator=generator)
    train_ids, valid_ids = order[holdout:], order[:holdout]
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                          eta_min=0.03 * learning_rate)
    history = []
    for epoch in range(epochs):
        epoch_order = train_ids[torch.randperm(len(train_ids), generator=generator)]
        losses = []
        for start in range(0, len(epoch_order), batch_size):
            ids = epoch_order[start:start + batch_size]
            loss = policy.cfm_loss(contexts[ids], plans[ids], reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        with torch.no_grad():
            validation = float(policy.cfm_loss(contexts[valid_ids], plans[valid_ids],
                                               reduction="mean"))
        history.append({"epoch": epoch, "train": float(np.mean(losses)), "valid": validation})
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} train {history[-1]['train']:.4f} valid {validation:.4f}",
                  flush=True)
    return history


def audit_raw(policy, config, gammas, episodes: int, seed0: int):
    rows = []
    for gamma in gammas:
        for episode in range(episodes):
            result = raw_rollout(policy, config, gamma, seed0 + episode)
            rows.append({"gamma": gamma, "seed": seed0 + episode, "status": result["status"],
                         "mode": result["mode"], "time": result["time_to_goal_s"],
                         "clearance": result["min_clearance_m"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ball_flow")
    parser.add_argument("--episodes-per-gamma", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--representation-dim", type=int, default=64)
    parser.add_argument("--augment-copies", type=int, default=2)
    parser.add_argument("--augment-jitter", type=float, default=0.02)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    demo_dir = collect_demos(args.output, args.episodes_per_gamma)
    contexts_np, plans_np, meta, config = demo_windows(demo_dir)
    env = TaskEnvironment(config)
    aug_c, aug_p = augment_windows(env, contexts_np, plans_np, meta,
                                   args.augment_copies, args.augment_jitter, args.seed)
    contexts = torch.from_numpy(aug_c)
    plans = torch.from_numpy(aug_p)
    print(f"[dataset] {len(contexts_np)} windows (+{args.augment_copies}x jitter copies -> "
          f"{len(contexts)}) from {args.episodes_per_gamma * len(config.data.gammas)} demos",
          flush=True)

    torch.manual_seed(args.seed)
    policy = ConditionalFlowMLP(
        context_dim=contexts.shape[1], plan_shape=(PLAN_H, 3), hidden=args.hidden,
        representation_dim=args.representation_dim,
        control_limit=config.safemppi.demo_u_max, nfe=args.nfe)
    history = train(policy, contexts, plans, args.epochs, args.batch_size,
                    args.learning_rate, args.seed)

    audit = audit_raw(policy, config, config.data.gammas, episodes=8, seed0=5000)
    success = float(np.mean([row["status"] == "SUCCESS" for row in audit]))
    modes = sorted({row["mode"] for row in audit if row["status"] == "SUCCESS"})
    print(f"[pretrained raw audit] SR={success:.2f} modes={modes}", flush=True)

    generator = torch.Generator().manual_seed(args.seed + 1)
    calibration_ids = torch.randperm(len(contexts), generator=generator)[:50]
    calibration = policy.embed(contexts[calibration_ids], plans[calibration_ids])
    lengthscale = mean_pairwise_lengthscale(calibration)

    gp = RBFPosterior(lengthscale, 1.0e-2)
    buffer_ids = torch.randperm(len(contexts), generator=generator)[:128]
    gp.set_buffer(policy.embed(contexts[buffer_ids], plans[buffer_ids]))
    pool_ids = torch.randperm(len(contexts), generator=generator)[:24]
    pools = []
    for index in pool_ids.tolist():
        candidates = policy.sample(contexts[index], 16, generator)
        pools.append(gp.sigma(policy.embed(contexts[index], candidates)))
    beta = calibrate_fixed_beta(pools, target=0.5)
    print(f"[calibration] lengthscale={lengthscale:.4f} beta={beta:.5f}", flush=True)

    torch.save({"model": policy.state_dict(),
                "arch": {"context_dim": int(contexts.shape[1]), "plan_shape": [PLAN_H, 3],
                         "hidden": args.hidden, "representation_dim": args.representation_dim,
                         "control_limit": config.safemppi.demo_u_max, "nfe": args.nfe}},
               args.output / "pretrained.pt")
    torch.save(calibration, args.output / "calibration_features.pt")
    manifest = {
        "windows": len(contexts), "epochs": args.epochs,
        "final_train_loss": history[-1]["train"], "final_valid_loss": history[-1]["valid"],
        "raw_audit_success_rate": success, "raw_audit_modes": modes,
        "raw_audit": audit, "rbf_lengthscale": lengthscale, "beta": beta,
        "demo_dir": str(demo_dir),
    }
    (args.output / "pretrain_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([row["epoch"] for row in history], [row["train"] for row in history], label="train")
    ax.plot([row["epoch"] for row in history], [row["valid"] for row in history], label="valid")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CFM loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "pretrain_loss.png", dpi=150)
    plt.close(fig)
    print("[outputs]", args.output / "pretrained.pt", flush=True)


if __name__ == "__main__":
    main()
