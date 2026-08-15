"""Automated (data size x model) sweep: when does expansion find the novel mode reliably?

Demonstrations come from configs/ball_fan_demo.json — angularly well-distributed strictly below
the ball's z=2 plane — so `above` is the one novel route mode. Every cell pretrains a flow on a
demo subset with a fixed optimization budget, runs a short canonical-start B1 expansion, and
measures (a) above-mode candidates the policy samples near the ball, (b) above-mode verifier
positives entering D+, and (c) the above share of raw temperature-1 successes. The winner is
re-run across extra seeds for the reliability claim. Everything lands in
outputs/ball_flow_sweep/: per-cell JSON, sweep_summary.json, sweep_grid.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safe_mppi.acquire import acquire
from safe_mppi.ball_flow_task import (BallFlowTask, PLAN_H, ROUTE_MODES, demo_windows,
                                      plan_states, raw_rollout, route_mode)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.expansion import (ExpansionConfig, RBFPosterior, calibrate_fixed_beta,
                                 mean_pairwise_lengthscale, run_safe_expansion)
from safe_mppi.flow_model import ConditionalFlowMLP
from pretrain_ball_flow import augment_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "d2h64": {"trunk_depth": 2, "hidden": 64, "representation_dim": 64},
    "d3h64": {"trunk_depth": 3, "hidden": 64, "representation_dim": 64},
    "d2h96": {"trunk_depth": 2, "hidden": 96, "representation_dim": 96},
}


def ensure_demos(sweep_dir: Path) -> Path:
    demo_dir = sweep_dir / "demos"
    if not (demo_dir / "manifest.json").exists():
        acquire(ROOT / "configs" / "ball_fan_demo.json", demo_dir, device="cpu")
    return demo_dir


def pretrain_cell(demo_dir: Path, data_size: int, model_key: str, seed: int,
                  epochs: int, batch_size: int = 64, learning_rate: float = 3.0e-4):
    contexts_np, plans_np, meta, config = demo_windows(demo_dir)
    keep = np.asarray([row["seed"] < data_size for row in meta])
    contexts_np, plans_np = contexts_np[keep], plans_np[keep]
    kept_meta = [row for row, flag in zip(meta, keep) if flag]
    env = TaskEnvironment(config)
    aug_c, aug_p = augment_windows(env, contexts_np, plans_np, kept_meta, 2, 0.02, seed)
    contexts, plans = torch.from_numpy(aug_c), torch.from_numpy(aug_p)

    torch.manual_seed(seed)
    policy = ConditionalFlowMLP(context_dim=contexts.shape[1], plan_shape=(PLAN_H, 3),
                                control_limit=config.safemppi.demo_u_max, nfe=16,
                                **MODELS[model_key])
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.03 * learning_rate)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(contexts), generator=generator)
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            loss = policy.cfm_loss(contexts[ids], plans[ids], reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
    generator2 = torch.Generator().manual_seed(seed + 1)
    ids = torch.randperm(len(contexts), generator=generator2)[:50]
    calibration = policy.embed(contexts[ids], plans[ids])
    lengthscale = mean_pairwise_lengthscale(calibration)
    gp = RBFPosterior(lengthscale, 1.0e-2)
    buffer_ids = torch.randperm(len(contexts), generator=generator2)[:128]
    gp.set_buffer(policy.embed(contexts[buffer_ids], plans[buffer_ids]))
    pools = []
    for index in torch.randperm(len(contexts), generator=generator2)[:16].tolist():
        candidates = policy.sample(contexts[index], 16, generator2)
        pools.append(gp.sigma(policy.embed(contexts[index], candidates)))
    beta = calibrate_fixed_beta(pools, target=0.5)
    return policy, config, calibration, lengthscale, beta, float(np.mean(losses[-100:]))


class DiscoveryCounter:
    """Counts route modes of near-ball candidates and selected verifier positives per round."""

    def __init__(self, env: TaskEnvironment):
        self.env = env
        self.candidates: dict[tuple[int, str], int] = {}
        self.positives: dict[tuple[int, str], int] = {}

    def __call__(self, event):
        state6 = np.concatenate([event["state_before"]["x"][:3],
                                 event["state_before"]["x"][3:6]])
        if not 0.7 <= state6[0] <= 1.6:
            return
        modes = [route_mode(self.env, plan_states(self.env, state6, plan.numpy())[:, :3])
                 for plan in event["candidates"]]
        for mode in modes:
            key = (event["round"], mode)
            self.candidates[key] = self.candidates.get(key, 0) + 1
        for local, k in enumerate(event["selected"]):
            if event["verification"][local]["valid"]:
                key = (event["round"], modes[k])
                self.positives[key] = self.positives.get(key, 0) + 1

    def totals(self, mode: str):
        cand = sum(v for (r, m), v in self.candidates.items() if m == mode)
        pos = sum(v for (r, m), v in self.positives.items() if m == mode)
        first = min((r for (r, m), v in self.positives.items() if m == mode and v), default=None)
        return cand, pos, first


def raw_mode_eval(policy, config, gammas, episodes: int, seed0: int):
    rows = [raw_rollout(policy, config, gamma, seed0 + 37 * episode)
            for gamma in gammas for episode in range(episodes)]
    successes = [row for row in rows if row["status"] == "SUCCESS"]
    shares = {mode: (np.mean([row["mode"] == mode for row in successes])
                     if successes else 0.0) for mode in ROUTE_MODES}
    return {
        "SR": float(np.mean([row["status"] == "SUCCESS" for row in rows])),
        "CR": float(np.mean([row["status"] == "COLLISION" for row in rows])),
        "mode_shares": {mode: float(share) for mode, share in shares.items()},
        "coverage": sum(share > 0 for share in shares.values()) / len(ROUTE_MODES),
    }


def run_cell(sweep_dir: Path, demo_dir: Path, data_size: int, model_key: str, seed: int,
             rounds: int, epochs: int, beta: float = 0.003):
    name = f"d{data_size:02d}_{model_key}_s{seed}"
    cell_dir = sweep_dir / name
    result_path = cell_dir / "cell.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    started = time.perf_counter()
    policy, config, calibration, lengthscale, beta_ess, train_loss = pretrain_cell(
        demo_dir, data_size, model_key, seed, epochs)
    task = BallFlowTask(config, start_diversity=False)
    counter = DiscoveryCounter(task.env)
    expansion_dir = cell_dir / "expansion"
    if expansion_dir.exists():
        import shutil
        shutil.rmtree(expansion_dir)
    expansion_config = ExpansionConfig(
        rounds=rounds, gammas=tuple(config.data.gammas), parallel_episodes=2,
        max_steps=config.taskspace.max_steps, K=16, B=4, batch_size=32, inner_steps=None,
        learning_rate=1.0e-5, replay_rounds=2, gp_buffer_cap=256, gp_noise=1.0e-2,
        beta=beta, adaptive_beta=False, negative_alpha=0.0, seed=seed)
    manifest = run_safe_expansion(policy, task, expansion_dir, config=expansion_config,
                                  calibration_features=calibration, event_callback=counter)
    gammas = list(config.data.gammas)
    raw0_policy = ConditionalFlowMLP(context_dim=10, plan_shape=(PLAN_H, 3),
                                     control_limit=config.safemppi.demo_u_max, nfe=16,
                                     **MODELS[model_key])
    raw0_policy.load_state_dict(torch.load(expansion_dir / "checkpoint_000.pt",
                                           weights_only=False)["model"])
    raw0 = raw_mode_eval(raw0_policy, config, gammas, 16, 91000 + seed)
    raw_final = raw_mode_eval(policy, config, gammas, 16, 91000 + seed)
    above_cand, above_pos, first_above = counter.totals("above")
    row = {
        "cell": name, "data_per_gamma": data_size, "model": model_key, "seed": seed,
        "train_loss": train_loss, "beta": beta, "beta_ess_reference": beta_ess,
        "rbf_lengthscale": lengthscale,
        "expansion_success": sum(r["success"] for r in manifest["rounds"]),
        "raw_round0": raw0, "raw_final": raw_final,
        "above_candidates_near_ball": above_cand, "above_positives": above_pos,
        "first_above_positive_round": first_above,
        "above_raw_share_final": raw_final["mode_shares"]["above"],
        "discovered_in_acquisition": above_pos >= 5,
        "discovered_in_raw": raw_final["mode_shares"]["above"] > 0.0,
        "minutes": round((time.perf_counter() - started) / 60.0, 1),
    }
    result_path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"[cell {name}] SR {raw0['SR']:.2f}->{raw_final['SR']:.2f} "
          f"above cand/pos {above_cand}/{above_pos} raw-above "
          f"{row['above_raw_share_final']:.2f} ({row['minutes']}min)", flush=True)
    return row


def grid_figure(rows, data_sizes, model_keys, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    metrics = [("above_positives", "above verifier positives (D+)"),
               ("above_raw_share_final", "above share of raw successes"),
               (("raw_final", "SR"), "raw SR (final)")]
    lookup = {(r["data_per_gamma"], r["model"]): r for r in rows if r["seed"] == 0}
    for ax, (key, title) in zip(axes, metrics):
        grid = np.zeros((len(model_keys), len(data_sizes)))
        for i, model in enumerate(model_keys):
            for j, size in enumerate(data_sizes):
                row = lookup.get((size, model))
                if row is None:
                    grid[i, j] = np.nan
                elif isinstance(key, tuple):
                    grid[i, j] = row[key[0]][key[1]]
                else:
                    grid[i, j] = row[key]
        image = ax.imshow(grid, cmap="viridis", aspect="auto")
        for i in range(len(model_keys)):
            for j in range(len(data_sizes)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}" if grid[i, j] < 10
                            else f"{int(grid[i, j])}", ha="center", va="center",
                            color="white", fontsize=9)
        ax.set_xticks(range(len(data_sizes)), [f"{s}/γ" for s in data_sizes])
        ax.set_yticks(range(len(model_keys)), model_keys)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.045)
    fig.suptitle("Novel-mode (above) discovery vs demonstrations x model "
                 "(fan demos, canonical starts, 12-round expansion)", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=ROOT / "outputs" / "ball_flow_sweep")
    parser.add_argument("--data-sizes", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--models", nargs="+", default=["d2h64", "d3h64"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.003,
                        help="really small = near-greedy top-sigma acquisition, so each "
                             "round incrementally queries the newest feature space")
    parser.add_argument("--reliability-seeds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    sweep_dir = args.sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = ensure_demos(sweep_dir)

    rows = []
    for data_size in args.data_sizes:
        for model_key in args.models:
            rows.append(run_cell(sweep_dir, demo_dir, data_size, model_key, 0,
                                 args.rounds, args.epochs, args.beta))
    grid_figure(rows, args.data_sizes, args.models, sweep_dir / "sweep_grid.png")

    def score(row):
        return (row["above_positives"], row["above_raw_share_final"],
                row["raw_final"]["SR"])
    winner = max(rows, key=score)
    print(f"[winner] {winner['cell']}", flush=True)
    reliability = [winner]
    for seed in args.reliability_seeds:
        reliability.append(run_cell(sweep_dir, demo_dir, winner["data_per_gamma"],
                                    winner["model"], seed, args.rounds,
                                    args.epochs, args.beta))
    summary = {
        "grid": rows,
        "winner": {"data_per_gamma": winner["data_per_gamma"], "model": winner["model"]},
        "reliability": [{
            "seed": row["seed"],
            "above_positives": row["above_positives"],
            "discovered_in_acquisition": row["discovered_in_acquisition"],
            "above_raw_share_final": row["above_raw_share_final"],
            "discovered_in_raw": row["discovered_in_raw"],
        } for row in reliability],
        "reliability_rate_acquisition": float(np.mean(
            [row["discovered_in_acquisition"] for row in reliability])),
        "reliability_rate_raw": float(np.mean(
            [row["discovered_in_raw"] for row in reliability])),
    }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["reliability"], indent=2), flush=True)
    print(f"[reliability] acquisition {summary['reliability_rate_acquisition']:.2f} "
          f"raw {summary['reliability_rate_raw']:.2f}", flush=True)


if __name__ == "__main__":
    main()
