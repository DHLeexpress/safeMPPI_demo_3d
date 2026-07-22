"""Evaluate saved expansion checkpoints with an untilted raw-policy seed bank.

Per checkpoint (round): closed-loop temperature-1 rollouts (no verifier, no tilt) give raw
success/collision rates, route coverage, clearance, and time-to-goal; a raw open-loop probe at
fixed contexts gives untilted GREEN-verifier validity. Also renders: result curves, mode-colored
rollout galleries, the crossing fan per round, sigma-tilt mechanism figures, and the mechanism
video built from logged events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_task import (BallFlowTask, ROUTE_MODES, build_context, load_policy,
                                      plan_states, raw_rollout, route_mode)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.expansion_visualize import (MechanismFrame, plot_expansion_results,
                                           plot_rollout_gallery, render_expansion_mechanism)
from safe_mppi.visualize import PLASMA

ROOT = Path(__file__).resolve().parents[1]
MODE_COLORS = {"below": "#1468b3", "above": "#c8321b", "left": "#17964b",
               "right": "#8a3ffc", "none": "#9aa0a6"}


def checkpoint_policy(expansion_dir: Path, pretrain_dir: Path, round_i: int):
    policy = load_policy(pretrain_dir / "pretrained.pt")
    payload = torch.load(expansion_dir / f"checkpoint_{round_i:03d}.pt", weights_only=False)
    policy.load_state_dict(payload["model"])
    return policy


def raw_eval(policy, config, gammas, episodes: int, seed0: int):
    rows = []
    for gamma in gammas:
        for episode in range(episodes):
            result = raw_rollout(policy, config, gamma, seed0 + 37 * episode)
            rows.append({
                "gamma": float(gamma), "episode": episode, "status": result["status"],
                "mode": result["mode"], "min_clearance_m": result["min_clearance_m"],
                "time_to_goal_s": result["time_to_goal_s"],
                "states": result["states"],
            })
    return rows


def validity_probe(policy, task: BallFlowTask, gammas, samples: int, seed: int):
    """Raw open-loop plans at fixed probe states, judged by the untilted GREEN verifier."""
    env = task.env
    probes = [np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], np.float32),
              np.array([1.0, 0.0, 2.0, 0.55, 0.0, 0.0], np.float32),
              np.array([1.35, 0.0, 1.9, 0.5, 0.0, -0.1], np.float32)]
    generator = torch.Generator().manual_seed(seed)
    out = []
    for gamma in gammas:
        for index, state in enumerate(probes):
            context = torch.from_numpy(build_context(env, state, float(gamma)))
            candidates = policy.sample(context, samples, generator)
            results = task.verify(context, candidates, float(gamma))
            for candidate, result in zip(candidates, results):
                states = plan_states(env, state, candidate.numpy())
                out.append({"gamma": float(gamma), "probe": index,
                            "valid": bool(result.valid), "margin": float(result.margin),
                            "mode": route_mode(env, states[:, :3])})
    return out


def summarize(rows, probe_rows):
    def rate(rows, key, value):
        return float(np.mean([row[key] == value for row in rows])) if rows else None
    modes = sorted({row["mode"] for row in rows
                    if row["status"] == "SUCCESS" and row["mode"] in ROUTE_MODES})
    clearances = [row["min_clearance_m"] for row in rows if row["min_clearance_m"] is not None]
    times = [row["time_to_goal_s"] for row in rows if row["time_to_goal_s"] is not None]
    return {
        "SR": rate(rows, "status", "SUCCESS"), "CR": rate(rows, "status", "COLLISION"),
        "OOB": rate(rows, "status", "OOB"), "timeout": rate(rows, "status", "TIMEOUT"),
        "modes": modes, "coverage": len(modes) / len(ROUTE_MODES),
        "avg_min_clearance_m": (float(np.mean(clearances)) if clearances else None),
        "avg_time_to_goal_s": (float(np.mean(times)) if times else None),
        "raw_validity": (float(np.mean([row["valid"] for row in probe_rows]))
                         if probe_rows else None),
    }


def above_start_probe(policy, config, gammas, episodes: int, seed0: int):
    """Raw closed-loop rollouts from a fixed above/ahead start: is the above corridor learned?"""
    start = np.array([1.05, 0.0, 2.42, 0.4, 0.0, 0.0], np.float32)
    rows = []
    for gamma in gammas:
        for episode in range(episodes):
            result = raw_rollout(policy, config, gamma, seed0 + 61 * episode, start=start)
            rows.append({"gamma": float(gamma), "status": result["status"],
                         "mode": result["mode"]})
    above = [row for row in rows if row["mode"] == "above" and row["status"] == "SUCCESS"]
    return {"SR": float(np.mean([row["status"] == "SUCCESS" for row in rows])),
            "above_success_share": len(above) / len(rows)}


def draw_scene_factory(env: TaskEnvironment):
    def draw_scene(ax):
        sphere = np.asarray(env.spheres[0], float)
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 10)
        ax.plot_surface(sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v)),
                        sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v)),
                        sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v)),
                        color="#8f969f", alpha=0.30, linewidth=0)
        ax.scatter(*env.start[:3], marker="s", color="#111111", s=25)
        ax.scatter(*env.goal, marker="*", color="#ffca28", edgecolor="#6a4e00", s=110)
        ax.set_xlim(*env.bounds[0])
        ax.set_ylim(*env.bounds[1])
        ax.set_zlim(*env.bounds[2])
        ax.set_box_aspect(tuple(env.bounds[:, 1] - env.bounds[:, 0]))
    return draw_scene


def coverage_figure(summary, gammas, output):
    rounds = sorted(int(k) for k in summary)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex=True)
    panels = [("SR", "raw success rate"), ("CR", "raw collision rate"),
              ("coverage", "route coverage (of 4 modes)"),
              ("raw_validity", "untilted verifier validity")]
    for ax, (key, label) in zip(axes.flat, panels):
        pooled = [summary[str(r)]["pooled"][key] for r in rounds]
        ax.plot(rounds, pooled, "-o", color="#174f92", lw=2.0, label="pooled")
        for gamma in gammas:
            values = [summary[str(r)]["per_gamma"][f"{gamma:g}"][key] for r in rounds]
            ax.plot(rounds, values, alpha=0.45, lw=1.1,
                    color=PLASMA(Normalize(0, 1)(gamma)), label=rf"$\gamma={gamma:g}$")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.set_xlabel("expansion round")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Raw temperature-1 evaluation across expansion rounds", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def mode_share_figure(per_round_rows, output):
    rounds = sorted(per_round_rows)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bottom = np.zeros(len(rounds))
    for mode in (*ROUTE_MODES, "none"):
        shares = []
        for round_i in rounds:
            rows = [row for row in per_round_rows[round_i] if row["status"] == "SUCCESS"]
            shares.append(np.mean([row["mode"] == mode for row in rows]) if rows else 0.0)
        shares = np.asarray(shares)
        ax.bar(rounds, shares, bottom=bottom, color=MODE_COLORS[mode], label=mode, width=0.8)
        bottom += shares
    ax.set_xlabel("expansion round")
    ax.set_ylabel("share of raw successful episodes")
    ax.set_title("Route-mode composition of raw successes", weight="bold")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def crossing_fan_figure(env, per_round_rows, rounds, output):
    sphere = np.asarray(env.spheres[0], float)
    fig, axes = plt.subplots(1, len(rounds), figsize=(4.4 * len(rounds), 4.4),
                             sharey=True)
    theta = np.linspace(0, 2 * np.pi, 120)
    for ax, round_i in zip(np.atleast_1d(axes), rounds):
        ax.fill(sphere[1] + sphere[3] * np.cos(theta), sphere[2] + sphere[3] * np.sin(theta),
                color="#8f969f", alpha=0.45, zorder=2)
        ax.axhline(sphere[2], color="#cc3311", lw=0.9, ls="--")
        for row in per_round_rows[round_i]:
            if row["status"] != "SUCCESS":
                continue
            states = row["states"]
            ax.plot(states[:, 1], states[:, 2], color=MODE_COLORS[row["mode"]],
                    lw=1.0, alpha=0.55)
        ax.set_xlim(0.9, -0.9)
        ax.set_ylim(sphere[2] - 0.9, sphere[2] + 0.9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"round {round_i}")
        ax.set_xlabel("y [m] (+y left)")
    np.atleast_1d(axes)[0].set_ylabel("z [m]")
    fig.suptitle("Head-on raw successful trajectories by round (color = route mode)",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def green_verifier_figure(env, config, events, output):
    """The GREEN verifier: rebuilt polytope chain along one executed verifier-positive plan."""
    from safe_mppi.ball_analysis import draw_polytope_soft
    from safe_mppi.ball_flow_task import verifier_chain_margins
    from safe_mppi.geometry import NominalPolytope, build_nominal_polytope
    candidates_events = [event for event in events
                         if event["chosen_local"] is not None
                         and 1.05 <= event["robot"][0] <= 1.35
                         and abs(event["gamma"] - 0.3) < 1e-9]
    event = candidates_events[len(candidates_events) // 2]
    state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
    plan = event["candidates"][event["selected"][event["chosen_local"]]]
    states = plan_states(env, state6, plan)
    margins = verifier_chain_margins(env, states[:, :3], float(event["gamma"]),
                                     config.safemppi.sensing_range)
    fig = plt.figure(figsize=(12.8, 5.2))
    ax = fig.add_subplot(121, projection="3d")
    draw_scene_factory(env)(ax)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from safe_mppi.geometry import hull_edges, polytope_vertices
    for h in (0, 3, 6, 9):
        polytope = build_nominal_polytope(states[h, :3], env.spheres, env.cylinders,
                                          env.bounds,
                                          sensing_range=config.safemppi.sensing_range,
                                          obstacle_margin=0.0)
        if h == 3:
            draw_polytope_soft(ax, polytope, float(event["gamma"]), color="#1d7a3e",
                               cmap=plt.get_cmap("Greens"))
        else:
            vertices = polytope_vertices(polytope)
            if vertices is not None:
                _, edges = hull_edges(vertices)
                ax.add_collection3d(Line3DCollection(vertices[edges], colors="#1d7a3e",
                                                     linewidths=0.35, alpha=0.30))
    ax.plot(*states[:, :3].T, color="#111111", lw=2.4, marker="o", ms=3)
    robot = states[0, :3]
    ax.set_xlim(robot[0] - 0.6, robot[0] + 1.2)
    ax.set_ylim(robot[1] - 0.8, robot[1] + 0.8)
    ax.set_zlim(max(robot[2] - 0.8, 1.0), robot[2] + 0.8)
    ax.set_box_aspect((1.8, 1.6, 1.6))
    ax.set_title("GREEN verifier: rebuilt polytope chain along the executed plan\n"
                 rf"($\gamma={event['gamma']:g}$, round {event['round']}; soft fill at h=3, "
                 "outlines at h=0,6,9)", fontsize=10)
    ax2 = fig.add_subplot(122)
    ax2.bar(range(1, len(margins) + 1), margins, color="#1d7a3e", alpha=0.8)
    ax2.axhline(0.0, color="#c8321b", lw=1.2)
    ax2.set_xlabel("plan step h")
    ax2.set_ylabel(r"$H_P(q_{h+1}) - (1-\gamma)$ under the rebuilt polytope at $q_h$")
    ax2.set_title("per-knot verifier margins (all must stay above the red line)",
                  fontsize=10)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def mode_timeline_figure(env, events, per_round_rows, output):
    """First appearance of each route mode: tilted acquisition (D+) vs raw sampling."""
    acquisition = {}
    for event in events:
        state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
        if not (0.7 <= state6[0] <= 1.5):
            continue
        for local, k in enumerate(event["selected"]):
            if event["verification"][local]["valid"]:
                mode = route_mode(env, plan_states(env, state6,
                                                   event["candidates"][k])[:, :3])
                acquisition.setdefault(event["round"], set()).add(mode)
    raw = {round_i: {row["mode"] for row in rows if row["status"] == "SUCCESS"}
           for round_i, rows in per_round_rows.items()}
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    for row_i, mode in enumerate(ROUTE_MODES):
        acq_rounds = [r for r in sorted(acquisition) if mode in acquisition[r]]
        raw_rounds = [r for r in sorted(raw) if mode in raw[r]]
        ax.scatter(acq_rounds, [row_i + 0.16] * len(acq_rounds), marker="o", s=26,
                   facecolor="none", edgecolor=MODE_COLORS[mode],
                   label="tilted acquisition D+" if row_i == 0 else None)
        ax.scatter(raw_rounds, [row_i - 0.16] * len(raw_rounds), marker="s", s=26,
                   color=MODE_COLORS[mode],
                   label="raw temperature-1" if row_i == 0 else None)
    ax.set_yticks(range(len(ROUTE_MODES)), ROUTE_MODES)
    ax.set_xlabel("expansion round")
    ax.set_title("Mode presence: verifier-positive tilted queries (open) vs raw sampling "
                 "(filled)", weight="bold", fontsize=10)
    ax.grid(alpha=0.2, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gamma_trend_figure(rows_final, demo_metrics_csv, gammas, output):
    """Final-round raw clearance / time-to-goal vs gamma, next to the SafeMPPI demo trend."""
    import csv as csv_module
    with open(demo_metrics_csv) as stream:
        demo = {float(row["gamma"]): row for row in csv_module.DictReader(stream)}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key, demo_key, label in (
            (axes[0], "min_clearance_m", "avg_min_clearance_m", "avg min clearance [m]"),
            (axes[1], "time_to_goal_s", "avg_time_to_goal_s", "avg time to goal [s]")):
        ours, reference = [], []
        for gamma in gammas:
            values = [row[key] for row in rows_final
                      if row["gamma"] == gamma and row[key] is not None
                      and row["status"] == "SUCCESS"]
            ours.append(float(np.mean(values)) if values else np.nan)
            reference.append(float(demo[gamma][demo_key]))
        ax.plot(gammas, ours, "-o", color="#174f92", label="expansion raw (final round)")
        ax.plot(gammas, reference, "--s", color="#9aa0a6", label="SafeMPPI demonstrations")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(label)
        ax.set_xticks(gammas)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Gamma trends: raw expansion policy vs the SafeMPPI demonstrator",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sigma_tilt_figures(env, events, output_dir, beta):
    """(1) anatomy of one acquisition step; (2) per-mode sigma decay across rounds."""
    interesting = [event for event in events
                   if 1.0 <= event["robot"][0] <= 1.45 and len(set(event["selected"])) > 1]
    event = max(interesting, key=lambda e: float(np.std(e["sigma_K"])))
    paths = np.stack([plan_states(env, np.concatenate([event["robot"][:3],
                                                       event["robot"][3:6]]),
                                  plan)[:, :3] for plan in event["candidates"]])
    sigma = event["sigma_K"]
    norm = Normalize(float(sigma.min()), float(sigma.max()) + 1e-12)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(12.6, 4.8))
    ax = fig.add_subplot(131, projection="3d")
    draw_scene_factory(env)(ax)
    valid_flags = [bool(v["valid"]) for v in event["verification"]]
    for k, path in enumerate(paths):
        ax.plot(*path.T, color=cmap(norm(sigma[k])), lw=2.0 if k in event["selected"] else 0.9,
                ls="--" if k in event["selected"] else "-",
                alpha=0.95 if k in event["selected"] else 0.45)
    for local, k in enumerate(event["selected"]):
        ax.plot(*paths[k].T, color=("#17964b" if valid_flags[local] else "#c8321b"),
                lw=2.6, alpha=0.9)
    robot = event["robot"][:3]
    ax.set_xlim(robot[0] - 0.45, robot[0] + 1.05)
    ax.set_ylim(robot[1] - 0.7, robot[1] + 0.7)
    ax.set_zlim(max(robot[2] - 0.7, 1.0), robot[2] + 0.7)
    ax.set_box_aspect((1.5, 1.4, 1.4))
    ax.set_title(rf"K=16 plans colored by $\sigma(\phi_s)$; selected B dashed"
                 f"\n(green/red = verifier verdict), round {event['round']}", fontsize=9)
    ax2 = fig.add_subplot(132)
    order = np.argsort(sigma)[::-1]
    weights = np.exp((sigma - sigma.max()) / beta)
    weights = weights / weights.sum()
    ax2.bar(range(len(sigma)), sigma[order], color=[cmap(norm(s)) for s in sigma[order]])
    marks = [int(np.where(order == k)[0][0]) for k in event["selected"]]
    ax2.scatter(marks, sigma[order][marks] + 0.002, marker="v", color="#c8321b", s=40,
                label="selected B")
    ax2.set_xlabel("candidate (sorted)")
    ax2.set_ylabel(r"$\sigma(\phi_s)$")
    ax2.legend(fontsize=8)
    ax2.set_title("marginal uncertainty + selection", fontsize=9)
    ax3 = fig.add_subplot(133)
    ax3.bar(range(len(sigma)), weights[order], color="#5b8fd6")
    ax3.set_xlabel("candidate (sorted by sigma)")
    ax3.set_ylabel(r"$\pi(j)\propto\exp((\sigma_j-\max\sigma)/\beta)$")
    ax3.set_title(rf"acquisition softmax, $\beta$ fixed", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "sigma_tilt_anatomy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    by_round_mode = {}
    for event in events:
        state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
        if not (0.8 <= state6[0] <= 1.5):
            continue
        for k, plan in enumerate(event["candidates"]):
            mode = route_mode(env, plan_states(env, state6, plan)[:, :3])
            by_round_mode.setdefault((event["round"], mode), []).append(
                float(event["sigma_K"][k]))
    rounds = sorted({key[0] for key in by_round_mode})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for mode in (*ROUTE_MODES, "none"):
        values = [np.mean(by_round_mode.get((r, mode), [np.nan])) for r in rounds]
        ax.plot(rounds, values, "-o", ms=4, color=MODE_COLORS[mode], label=mode)
    ax.set_xlabel("expansion round")
    ax.set_ylabel(r"mean $\sigma(\phi_s)$ of near-ball candidates")
    ax.set_title(r"$\sigma$ per route mode: novel modes start high, decay as queried",
                 weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "sigma_mode_decay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def mechanism_video(env, events, output, rounds):
    positives, negatives = [], []
    frames = []
    chosen_rounds = set(rounds)
    for event in events:
        robot = event["robot"][:3]
        if event["status"] == "NVP":
            negatives.append(robot)
        else:
            positives.append(robot)
        if event["round"] not in chosen_rounds or event["episode"] % 4 or event["step"] % 2:
            continue
        state6 = np.concatenate([event["robot"][:3], event["robot"][3:6]])
        paths = np.stack([plan_states(env, state6, plan)[:, :3]
                          for plan in event["candidates"]])
        valid_selected = tuple(event["selected"][local]
                               for local, verdict in enumerate(event["verification"])
                               if verdict["valid"])
        executed = (None if event["chosen_local"] is None
                    else event["selected"][event["chosen_local"]])
        frames.append(MechanismFrame(
            round=event["round"], gamma=event["gamma"], robot=robot,
            candidate_paths=paths, sigma=event["sigma_K"],
            selected=tuple(event["selected"]), positive=valid_selected,
            executed=executed, nvp=event["status"] == "NVP",
            positive_states=np.asarray(positives[-400:], float).reshape(-1, 3),
            negative_states=np.asarray(negatives[-400:], float).reshape(-1, 3),
        ))
    return render_expansion_mechanism(frames, output,
                                      draw_scene=draw_scene_factory(env), fps=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, default=ROOT / "outputs" / "ball_flow")
    parser.add_argument("--expansion", type=Path,
                        default=ROOT / "outputs" / "ball_flow" / "expansion")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--probe-samples", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=91000)
    args = parser.parse_args()

    config = load_config(args.pretrain_dir / "demo_config.json")
    env = TaskEnvironment(config)
    task = BallFlowTask(config)
    gammas = list(config.data.gammas)
    manifest = json.loads((args.expansion / "manifest.json").read_text())
    total_rounds = manifest["config"]["rounds"]
    eval_rounds = sorted({0, *range(0, total_rounds + 1, args.stride), total_rounds})

    summary, per_round_rows = {}, {}
    for round_i in eval_rounds:
        policy = checkpoint_policy(args.expansion, args.pretrain_dir, round_i)
        rows = raw_eval(policy, config, gammas, args.episodes, args.seed)
        probes = validity_probe(policy, task, gammas, args.probe_samples, args.seed + 7)
        per_round_rows[round_i] = rows
        per_gamma = {}
        for gamma in gammas:
            gamma_rows = [row for row in rows if row["gamma"] == gamma]
            gamma_probe = [row for row in probes if row["gamma"] == gamma]
            per_gamma[f"{gamma:g}"] = summarize(gamma_rows, gamma_probe)
        summary[str(round_i)] = {"pooled": summarize(rows, probes), "per_gamma": per_gamma}
        if round_i in (eval_rounds[0], eval_rounds[-1]):
            summary[str(round_i)]["above_start_probe"] = above_start_probe(
                policy, config, gammas, episodes=6, seed0=args.seed + 500)
        pooled = summary[str(round_i)]["pooled"]
        print(f"round {round_i:2d}: SR {pooled['SR']:.2f} CR {pooled['CR']:.2f} "
              f"coverage {pooled['coverage']:.2f} modes {pooled['modes']} "
              f"validity {pooled['raw_validity']:.2f}"
              + (f" | above-start probe {summary[str(round_i)]['above_start_probe']}"
                 if "above_start_probe" in summary[str(round_i)] else ""), flush=True)

    eval_dir = args.expansion / "eval"
    eval_dir.mkdir(exist_ok=True)
    slim = {round_i: [{k: v for k, v in row.items() if k != "states"} for row in rows]
            for round_i, rows in per_round_rows.items()}
    (eval_dir / "raw_eval.json").write_text(json.dumps(
        {"summary": summary, "rows": slim}, indent=2) + "\n")

    plot_expansion_results(args.expansion / "metrics.jsonl", eval_dir / "expansion_curves.png")
    coverage_figure(summary, gammas, eval_dir / "raw_curves.png")
    mode_share_figure(per_round_rows, eval_dir / "mode_share.png")
    gallery_rounds = sorted({eval_rounds[0], eval_rounds[len(eval_rounds) // 2],
                             eval_rounds[-1]})
    crossing_fan_figure(env, per_round_rows, gallery_rounds, eval_dir / "raw_crossing_fan.png")
    rollouts = {(round_i, float(gamma)): [row["states"][:, :3]
                                          for row in per_round_rows[round_i]
                                          if row["gamma"] == gamma]
                for round_i in gallery_rounds for gamma in gammas}
    plot_rollout_gallery(rollouts, gallery_rounds, gammas, eval_dir / "raw_gallery.png",
                         draw_scene=draw_scene_factory(env))

    events = torch.load(args.expansion / "events.pt", weights_only=False)
    sigma_tilt_figures(env, events, eval_dir, float(manifest["config"]["beta"]))
    green_verifier_figure(env, config, events, eval_dir / "green_verifier_chain.png")
    mode_timeline_figure(env, events, per_round_rows, eval_dir / "mode_timeline.png")
    gamma_trend_figure(per_round_rows[total_rounds],
                       args.pretrain_dir / "demos" / "metrics.csv", gammas,
                       eval_dir / "gamma_trend_vs_safemppi.png")
    video_rounds = sorted({1, max(1, total_rounds // 2), total_rounds})
    mechanism_video(env, events, eval_dir / "mechanism.mp4", video_rounds)
    print("[outputs]", eval_dir, flush=True)


if __name__ == "__main__":
    main()
