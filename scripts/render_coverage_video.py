"""Final coverage video: expansion iterations progressively cover the ball with trajectories.

For each saved checkpoint the script runs a fresh untilted raw seed bank, then renders an MP4:
the left panel accumulates every generated trajectory in 3D around the ball (color = route mode,
newer rounds brighter) while the camera orbits; the right panels grow the metric curves
(raw SR / CR / route coverage / untilted verifier validity) and the head-on crossing fan. The
video ends holding on the fully covered ball next to the achieved metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safe_mppi.ball_flow_task import (BallFlowTask, ROUTE_MODES, load_policy, raw_rollout,
                                      route_mode)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from evaluate_ball_expansion import validity_probe  # noqa: E402

MODE_COLORS = {"below": "#1468b3", "above": "#c8321b", "left": "#17964b",
               "right": "#8a3ffc", "none": "#9aa0a6"}


def executed_episodes(expansion: Path, env: TaskEnvironment):
    """Self-generated verifier-positive episode paths per round, from the event log."""
    events_path = expansion / "events.pt"
    if not events_path.exists():
        return {}
    events = torch.load(events_path, weights_only=False)
    grouped: dict[tuple[int, int], list] = {}
    for event in events:
        grouped.setdefault((event["round"], event["episode"]), []).append(event)
    per_round: dict[int, list] = {}
    for (round_i, _), rows in grouped.items():
        rows.sort(key=lambda event: event["step"])
        path = np.asarray([row["robot"][:3] for row in rows], float)
        if len(path) < 3:
            continue
        per_round.setdefault(round_i, []).append(
            (path, route_mode(env, path)))
    return per_round


def collect(expansion: Path, pretrain_dir: Path, stride: int, episodes: int, seed: int):
    config = load_config(pretrain_dir / "demo_config.json")
    task = BallFlowTask(config)
    env = task.env
    gammas = list(config.data.gammas)
    manifest_rounds = sorted(int(p.stem.split("_")[1]) for p in expansion.glob("checkpoint_*.pt"))
    rounds = sorted({manifest_rounds[0], *manifest_rounds[::stride], manifest_rounds[-1]})
    executed = executed_episodes(expansion, env)
    per_round = []
    previous_round = -1
    for round_i in rounds:
        policy = load_policy(pretrain_dir / "pretrained.pt")
        policy.load_state_dict(torch.load(expansion / f"checkpoint_{round_i:03d}.pt",
                                          weights_only=False)["model"])
        rows = [raw_rollout(policy, config, gamma, seed + 37 * episode)
                for gamma in gammas for episode in range(episodes)]
        probes = validity_probe(policy, task, gammas, 12, seed + 7)
        successes = [row for row in rows if row["status"] == "SUCCESS"]
        modes = {mode for row in successes if (mode := row["mode"]) in ROUTE_MODES}
        window = [r for r in executed if previous_round < r <= round_i]
        generated = [item for r in window for item in executed[r]]
        previous_round = round_i
        per_round.append({
            "round": round_i,
            "trajectories": [(row["states"][:, :3], row["mode"], row["status"]) for row in rows],
            "generated": generated,
            "SR": float(np.mean([row["status"] == "SUCCESS" for row in rows])),
            "CR": float(np.mean([row["status"] == "COLLISION" for row in rows])),
            "coverage": len(modes) / len(ROUTE_MODES),
            "validity": float(np.mean([row["valid"] for row in probes])),
            "modes": sorted(modes),
        })
        print(f"[collect] round {round_i}: SR {per_round[-1]['SR']:.2f} "
              f"coverage {per_round[-1]['coverage']:.2f} modes {per_round[-1]['modes']}",
              flush=True)
    return env, per_round


def render(env: TaskEnvironment, per_round, output: Path, fps: int = 8,
           frames_per_round: int = 6, hold: int = 18):
    sphere = np.asarray(env.spheres[0], float)
    total = len(per_round) * frames_per_round + hold
    fig = plt.figure(figsize=(12.8, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.42, wspace=0.16)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2600)
    frame_index = 0
    with writer.saving(fig, str(output), dpi=120):
        for stage in range(len(per_round)):
            for _ in range(frames_per_round):
                _draw(fig, grid, env, sphere, per_round, stage,
                      frame_index / max(total - 1, 1), final=False)
                writer.grab_frame()
                frame_index += 1
        for _ in range(hold):
            _draw(fig, grid, env, sphere, per_round, len(per_round) - 1,
                  frame_index / max(total - 1, 1), final=True)
            writer.grab_frame()
            frame_index += 1
    plt.close(fig)
    return output


def _draw(fig, grid, env, sphere, per_round, stage, progress, final):
    fig.clear()
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 14)
    ax.plot_surface(sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
                    sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
                    sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
                    color="#70757d", alpha=0.55, linewidth=0)
    for past in range(stage + 1):
        age = stage - past
        alpha = max(0.85 - 0.12 * age, 0.20)
        for xyz, mode in per_round[past].get("generated", []):
            ax.plot(*np.asarray(xyz).T, color=MODE_COLORS.get(mode, "#9aa0a6"),
                    lw=0.55, alpha=min(alpha, 0.4), ls=":")
        for xyz, mode, status in per_round[past]["trajectories"]:
            if status != "SUCCESS":
                continue
            ax.plot(*np.asarray(xyz).T, color=MODE_COLORS.get(mode, "#9aa0a6"),
                    lw=1.0, alpha=alpha)
    ax.scatter(*env.start[:3], marker="s", color="#111111", s=40)
    ax.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=170)
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(1.1, 2.9)
    ax.set_box_aspect((3.6, 2.0, 1.8))
    ax.view_init(elev=20.0 + 8.0 * np.sin(2.0 * np.pi * progress),
                 azim=-120.0 + 360.0 * progress)
    ax.set_axis_off()
    row = per_round[stage]
    ax.set_title(f"round {row['round']} — raw trajectories (solid) + self-generated "
                 f"verifier-positive episodes (dotted)\n"
                 f"SR {row['SR']:.2f}, coverage {row['coverage']:.2f}",
                 fontsize=10.5, weight="bold")

    ax_metrics = fig.add_subplot(grid[0, 1])
    rounds = [r["round"] for r in per_round[:stage + 1]]
    for key, color, label in (("SR", "#174f92", "raw SR"), ("CR", "#c8321b", "raw CR"),
                              ("coverage", "#17964b", "route coverage"),
                              ("validity", "#8a3ffc", "verifier validity")):
        ax_metrics.plot(rounds, [r[key] for r in per_round[:stage + 1]], "-o", ms=3,
                        color=color, label=label)
    ax_metrics.set_xlim(per_round[0]["round"] - 0.5, per_round[-1]["round"] + 0.5)
    ax_metrics.set_ylim(-0.03, 1.03)
    ax_metrics.grid(alpha=0.25)
    ax_metrics.legend(fontsize=7, loc="center right")
    ax_metrics.set_xlabel("expansion round")
    ax_metrics.set_title("metrics", fontsize=10)

    ax_fan = fig.add_subplot(grid[1, 1])
    theta = np.linspace(0, 2 * np.pi, 100)
    ax_fan.fill(sphere[1] + sphere[3] * np.cos(theta), sphere[2] + sphere[3] * np.sin(theta),
                color="#8f969f", alpha=0.5, zorder=2)
    ax_fan.axhline(sphere[2], color="#cc3311", lw=0.9, ls="--")
    for past in range(stage + 1):
        for xyz, mode, status in per_round[past]["trajectories"]:
            if status != "SUCCESS":
                continue
            xyz = np.asarray(xyz)
            ax_fan.plot(xyz[:, 1], xyz[:, 2], color=MODE_COLORS.get(mode, "#9aa0a6"),
                        lw=0.7, alpha=0.4)
    ax_fan.set_xlim(0.95, -0.95)
    ax_fan.set_ylim(sphere[2] - 0.95, sphere[2] + 0.95)
    ax_fan.set_aspect("equal")
    ax_fan.grid(alpha=0.2)
    ax_fan.set_title("head-on view (from start)", fontsize=10)
    if final:
        row = per_round[-1]
        fig.text(0.30, 0.10,
                 f"final: SR {row['SR']:.2f}   CR {row['CR']:.2f}   "
                 f"coverage {row['coverage']:.2f}   validity {row['validity']:.2f}   "
                 f"modes: {', '.join(row['modes'])}",
                 fontsize=12, weight="bold", ha="center",
                 bbox={"facecolor": "#fff8dc", "edgecolor": "#6a4e00", "pad": 8})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=97000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    expansion = args.expansion.resolve()
    pretrain_dir = (args.pretrain_dir or expansion.parent).resolve()
    output = args.output or expansion / "coverage_video.mp4"
    env, per_round = collect(expansion, pretrain_dir, args.stride, args.episodes, args.seed)
    render(env, per_round, output)
    print("[video]", output, flush=True)


if __name__ == "__main__":
    main()
