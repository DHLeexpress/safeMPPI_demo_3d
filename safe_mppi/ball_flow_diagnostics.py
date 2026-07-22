"""Automated audit of the penultimate noised flow representation phi_s on the ball task.

Protocol (fixed-bank): one frozen candidate bank (policy samples + handcrafted mode arcs +
uniform random plans at fixed approach contexts), identical noise seeds and flow times across
rounds, several base noises averaged per plan. For each checkpoint the module produces

  - t-SNE panels on one shared embedding: color = validity / route mode / clearance / GP sigma,
    marker = gamma, transparency = round;
  - local probes: kNN validity & route-mode accuracy, linear + RBF validity AUROC, route-mode
    probe accuracy, control-magnitude shortcut check;
  - geometry checks: |phi_i - phi_j| vs trajectory distance, sigma vs nearest-queried-feature
    distance;
  - the decisive acquisition metric  P(new valid route mode | high-sigma B) versus
    P(new valid route mode | uniform B);
  - a flow-time ablation s in {0.5, 0.8, 0.9, 0.95}.

Everything lands in <expansion>/representation: figures, diagnostics.json, and
REPRESENTATION_REPORT.md. t-SNE is treated as visualization only; claims rest on the probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import torch

from .ball_flow_task import (BallFlowTask, PLAN_H, ROUTE_MODES, build_context, context_state,
                             load_policy, plan_states, route_mode)
from .config import load_config
from .environment import TaskEnvironment
from .expansion import RBFPosterior, _normalize

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError("ball_flow_diagnostics requires scikit-learn") from error

FLOW_TIMES = (0.5, 0.8, 0.9, 0.95)
DEFAULT_S = 0.9
NOISE_DRAWS = 4
MODE_COLORS = {"below": "#1468b3", "above": "#c8321b", "left": "#17964b",
               "right": "#8a3ffc", "none": "#9aa0a6"}
GAMMA_MARKERS = {0.1: "o", 0.3: "s", 0.5: "^", 1.0: "D"}


def _arc_plan(rng, direction: np.ndarray) -> np.ndarray:
    amplitude = rng.uniform(0.35, 1.0)
    forward = rng.uniform(0.1, 0.5)
    plan = np.zeros((PLAN_H, 3), np.float32)
    plan[:, 0] = forward
    plan[:3, 1:] = amplitude * direction[None]
    plan[3:6, 1:] = -amplitude * direction[None]
    plan += rng.normal(0.0, 0.12, size=plan.shape)
    return np.clip(plan, -1.0, 1.0).astype(np.float32)


BANK_STATES = (np.array([0.8, 0.0, 2.0, 0.6, 0.0, 0.0], np.float32),
               np.array([1.0, 0.0, 2.0, 0.55, 0.0, 0.0], np.float32))


def build_bank(task: BallFlowTask, policy0, gammas, seed: int = 20260722):
    """Fixed audit bank at two approach states (easy + hard); labeled by the GREEN verifier."""
    env = task.env
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    directions = {"below": np.array([0.0, -1.0]), "above": np.array([0.0, 1.0]),
                  "left": np.array([1.0, 0.0]), "right": np.array([-1.0, 0.0])}
    contexts, plans, meta = [], [], []
    for state_id, state6 in enumerate(BANK_STATES):
        for gamma in gammas:
            context = build_context(env, state6, float(gamma))
            pool, sources = [], []
            samples = policy0.sample(torch.from_numpy(context), 20, generator).numpy()
            pool.extend(samples)
            sources.extend(["policy"] * len(samples))
            for mode, direction in directions.items():
                for _ in range(10):
                    pool.append(_arc_plan(rng, direction))
                    sources.append(f"arc:{mode}")
            for _ in range(15):
                pool.append(rng.uniform(-1.0, 1.0, size=(PLAN_H, 3)).astype(np.float32))
                sources.append("random")
            results = task.verify(torch.from_numpy(context),
                                  torch.from_numpy(np.asarray(pool, np.float32)),
                                  float(gamma))
            for plan, source, result in zip(pool, sources, results):
                states = plan_states(env, state6, plan)
                dense = env.dense_positions(states, plan.reshape(PLAN_H, 3))
                clearance = env.obstacle_clearance(dense)
                finite = clearance[np.isfinite(clearance)]
                du = np.diff(plan.reshape(PLAN_H, 3), axis=0)
                contexts.append(context)
                plans.append(np.asarray(plan, np.float32))
                meta.append({
                    "gamma": float(gamma), "source": source, "bank_state": state_id,
                    "valid": bool(result.valid), "margin": float(result.margin),
                    "mode": route_mode(env, states[:, :3]),
                    "min_clearance_m": float(finite.min()) if len(finite) else 2.0,
                    "progress_m": float(np.linalg.norm(state6[:3] - env.goal)
                                        - np.linalg.norm(states[-1, :3] - env.goal)),
                    "mean_du": float(np.linalg.norm(du, axis=1).mean()),
                    "plan_norm": float(np.abs(plan).mean()),
                    "path": states[:, :3],
                })
    return (np.asarray(contexts, np.float32), np.asarray(plans, np.float32), meta)


def bank_embeddings(policy, contexts: np.ndarray, plans: np.ndarray, flow_time: float,
                    noise_seed: int = 777) -> np.ndarray:
    """Averaged noised representation: identical base-noise draws for every checkpoint."""
    contexts_t = torch.from_numpy(contexts)
    plans_t = torch.from_numpy(plans)
    generator = torch.Generator().manual_seed(noise_seed)
    total = None
    for _ in range(NOISE_DRAWS):
        base = torch.randn(len(plans_t), PLAN_H * 3, generator=generator)
        phi = policy.embed(contexts_t, plans_t, flow_time=flow_time, base=base)
        total = phi if total is None else total + phi
    return (total / NOISE_DRAWS).numpy()


def archive_modes(env: TaskEnvironment, archive) -> list[dict]:
    rows = []
    for record in archive:
        state6 = context_state(env, record.context.numpy())
        states = plan_states(env, state6, record.candidate.numpy())
        position = state6[0]
        phase = ("approach" if position < 1.1
                 else "interaction" if position <= 1.9 else "post")
        rows.append({"round": record.round, "gamma": record.gamma,
                     "valid": bool(record.verification.valid),
                     "mode": route_mode(env, states[:, :3]), "phase": phase,
                     "context": record.context.numpy(), "candidate": record.candidate.numpy()})
    return rows


def gp_for_round(policy, env, archive_rows, round_i: int, lengthscale: float,
                 replay_rounds: int = 2, cap: int = 256, noise: float = 1.0e-2,
                 seed: int = 0) -> RBFPosterior:
    gp = RBFPosterior(lengthscale, noise)
    eligible = [row for row in archive_rows
                if row["valid"] and round_i - replay_rounds < row["round"] <= round_i]
    if eligible:
        rng = np.random.default_rng(seed + round_i)
        if len(eligible) > cap:
            eligible = [eligible[i] for i in rng.choice(len(eligible), cap, replace=False)]
        contexts = torch.from_numpy(np.stack([row["context"] for row in eligible]))
        plans = torch.from_numpy(np.stack([row["candidate"] for row in eligible]))
        gp.set_buffer(policy.embed(contexts, plans))
    return gp


def knn_accuracy(features: np.ndarray, labels, k: int = 10) -> float:
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features, labels)
    neighbors = model.kneighbors(features, n_neighbors=k + 1, return_distance=False)[:, 1:]

    def majority(values):
        unique, counts = np.unique(values, return_counts=True)
        return unique[int(np.argmax(counts))]

    predictions = [majority(labels[row]) for row in neighbors]
    return float(np.mean(np.asarray(predictions) == labels))


def probe_metrics(features: np.ndarray, meta, seed: int = 0) -> dict:
    scaler = StandardScaler().fit(features)
    X = scaler.transform(features)
    valid = np.asarray([row["valid"] for row in meta])
    modes = np.asarray([row["mode"] for row in meta])
    magnitude = np.asarray([row["plan_norm"] for row in meta])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    split = int(0.7 * len(X))
    train, test = order[:split], order[split:]
    out = {"knn_validity_acc": knn_accuracy(X, valid),
           "knn_mode_acc": knn_accuracy(X, modes),
           "knn_control_magnitude_acc": knn_accuracy(X, magnitude > np.median(magnitude))}
    if len(set(valid[train].tolist())) == 2 and len(set(valid[test].tolist())) == 2:
        linear = LogisticRegression(max_iter=4000).fit(X[train], valid[train])
        out["linear_validity_auroc"] = float(roc_auc_score(
            valid[test], linear.predict_proba(X[test])[:, 1]))
        rbf = SVC(kernel="rbf", random_state=seed).fit(X[train], valid[train])
        out["rbf_validity_auroc"] = float(roc_auc_score(
            valid[test], rbf.decision_function(X[test])))
    mode_model = LogisticRegression(max_iter=4000).fit(X[train], modes[train])
    out["mode_probe_acc"] = float(np.mean(mode_model.predict(X[test]) == modes[test]))
    routed = np.isin(modes, ROUTE_MODES)
    out["mode_silhouette"] = (float(silhouette_score(X[routed], modes[routed]))
                              if routed.sum() > 10 and len(set(modes[routed].tolist())) > 1
                              else float("nan"))
    return out


def geometry_metrics(features: np.ndarray, meta, gp_sigma: np.ndarray,
                     buffer_features: torch.Tensor | None, pairs: int = 3000,
                     seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    normalized = _normalize(torch.from_numpy(features)).numpy()
    i = rng.integers(0, len(features), pairs)
    j = rng.integers(0, len(features), pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    feature_distance = np.linalg.norm(normalized[i] - normalized[j], axis=1)
    trajectory_distance = np.asarray([
        float(np.linalg.norm(meta[a]["path"] - meta[b]["path"], axis=1).mean())
        for a, b in zip(i, j)])
    out = {
        "phi_vs_traj_pearson": float(stats.pearsonr(feature_distance,
                                                    trajectory_distance)[0]),
        "phi_vs_traj_spearman": float(stats.spearmanr(feature_distance,
                                                      trajectory_distance)[0]),
        "sigma_vs_plan_norm_pearson": float(stats.pearsonr(
            gp_sigma, [row["plan_norm"] for row in meta])[0]),
    }
    if buffer_features is not None and len(buffer_features):
        buffer = _normalize(buffer_features).numpy()
        nearest = np.min(np.linalg.norm(normalized[:, None] - buffer[None], axis=2), axis=1)
        out["sigma_vs_nearest_buffer_pearson"] = float(stats.pearsonr(gp_sigma, nearest)[0])
    else:
        out["sigma_vs_nearest_buffer_pearson"] = None
    return out


def discovery_metric(policy, task, gp, known_modes: set[str], gammas, beta: float,
                     seed: int, draws: int = 400, K: int = 16, B: int = 4) -> dict | None:
    """P(selected B contains a verifier-valid plan of an unknown route mode)."""
    missing = [mode for mode in ROUTE_MODES if mode not in known_modes]
    if not missing:
        return None
    env = task.env
    state6 = np.array([1.0, 0.0, 2.0, 0.55, 0.0, 0.0], np.float32)
    rng = np.random.default_rng(seed)
    tilted_hits, uniform_hits, trials = 0, 0, 0
    for gamma in gammas:
        context = torch.from_numpy(build_context(env, state6, float(gamma)))
        generator = torch.Generator().manual_seed(seed + int(gamma * 1000))
        candidates = policy.sample(context, K, generator)
        results = task.verify(context, candidates, float(gamma))
        new_valid = np.asarray([
            result.valid and route_mode(env, plan_states(env, state6,
                                                         candidate.numpy())[:, :3]) in missing
            for candidate, result in zip(candidates, results)])
        sigma = gp.sigma(policy.embed(context, candidates)).numpy()
        weights = np.exp((sigma - sigma.max()) / beta)
        weights = weights / weights.sum()
        for _ in range(draws):
            tilted = rng.choice(K, size=B, replace=False, p=weights)
            uniform = rng.choice(K, size=B, replace=False)
            tilted_hits += bool(new_valid[tilted].any())
            uniform_hits += bool(new_valid[uniform].any())
            trials += 1
    return {"p_new_mode_high_sigma": tilted_hits / trials,
            "p_new_mode_uniform": uniform_hits / trials, "trials": trials,
            "missing_modes": missing}


def tsne_panels(embeddings_by_round: dict[int, np.ndarray], meta, sigma_by_round,
                output: Path, seed: int = 0):
    rounds = sorted(embeddings_by_round)
    stacked = np.concatenate([embeddings_by_round[r] for r in rounds])
    planar = TSNE(n_components=2, perplexity=30, random_state=seed,
                  init="pca").fit_transform(StandardScaler().fit_transform(stacked))
    per_round = {r: planar[k * len(meta):(k + 1) * len(meta)] for k, r in enumerate(rounds)}
    alphas = {r: a for r, a in zip(rounds, np.linspace(0.35, 0.95, len(rounds)))}
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 11.6))
    specs = [("verifier validity", None), ("route mode", None),
             ("min clearance [m]", "clearance"), (r"GP $\sigma(\phi_s)$", "sigma")]
    for ax, (title, kind) in zip(axes.flat, specs):
        for r in rounds:
            xy = per_round[r]
            for g, marker in GAMMA_MARKERS.items():
                rows = [k for k, row in enumerate(meta) if abs(row["gamma"] - g) < 1e-9]
                if kind is None and title.startswith("verifier"):
                    colors = ["#17964b" if meta[k]["valid"] else "#c8321b" for k in rows]
                elif kind is None:
                    colors = [MODE_COLORS[meta[k]["mode"]] for k in rows]
                elif kind == "clearance":
                    norm = Normalize(0.0, 0.5)
                    colors = plt.get_cmap("magma")(
                        norm([min(meta[k]["min_clearance_m"], 0.5) for k in rows]))
                else:
                    sigma = sigma_by_round[r]
                    norm = Normalize(0.0, max(float(np.max(list(map(np.max,
                                     sigma_by_round.values())))), 1e-9))
                    colors = plt.get_cmap("viridis")(norm([sigma[k] for k in rows]))
                ax.scatter(xy[rows, 0], xy[rows, 1], c=colors, marker=marker, s=13,
                           alpha=alphas[r], linewidths=0.0)
        ax.set_title(f"color = {title}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    legend_modes = [Line2D([], [], marker="o", ls="", color=MODE_COLORS[m], label=m)
                    for m in (*ROUTE_MODES, "none")]
    legend_gamma = [Line2D([], [], marker=m, ls="", color="#555555", label=rf"$\gamma={g:g}$")
                    for g, m in GAMMA_MARKERS.items()]
    axes[0, 1].legend(handles=legend_modes, fontsize=8, loc="upper right")
    axes[0, 0].legend(handles=legend_gamma, fontsize=8, loc="upper right")
    fig.suptitle("Fixed audit bank, one shared t-SNE embedding — rounds "
                 f"{rounds} (opacity = round), marker = gamma, s = {DEFAULT_S}",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tsne_by_flow_time(policy, contexts, plans, meta, output: Path, seed: int = 0):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.0))
    for ax, flow_time in zip(axes.flat, FLOW_TIMES):
        phi = bank_embeddings(policy, contexts, plans, flow_time)
        xy = TSNE(n_components=2, perplexity=30, random_state=seed,
                  init="pca").fit_transform(StandardScaler().fit_transform(phi))
        colors = [MODE_COLORS[row["mode"]] for row in meta]
        edge = ["#111111" if row["valid"] else "none" for row in meta]
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=14, alpha=0.8, linewidths=0.4,
                   edgecolors=edge)
        ax.set_title(rf"$s={flow_time}$ (color = route mode, edged = verifier-valid)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Final checkpoint: noised-representation flow-time ablation",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, default=None)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    expansion = args.expansion.resolve()
    pretrain_dir = (args.pretrain_dir or expansion.parent).resolve()
    output = expansion / "representation"
    output.mkdir(exist_ok=True)
    manifest = json.loads((expansion / "manifest.json").read_text())
    total_rounds = manifest["config"]["rounds"]
    beta = float(manifest["config"]["beta"])
    lengthscale = float(manifest["rbf_lengthscale"])
    config = load_config(pretrain_dir / "demo_config.json")
    task = BallFlowTask(config)
    env = task.env
    gammas = list(config.data.gammas)

    def policy_at(round_i: int):
        policy = load_policy(pretrain_dir / "pretrained.pt")
        payload = torch.load(expansion / f"checkpoint_{round_i:03d}.pt", weights_only=False)
        policy.load_state_dict(payload["model"])
        return policy

    policy0 = policy_at(0)
    contexts, plans, meta = build_bank(task, policy0, gammas)
    print(f"[bank] {len(meta)} items; valid {sum(row['valid'] for row in meta)}; "
          f"modes {sorted({row['mode'] for row in meta})}", flush=True)

    archive = torch.load(expansion / "query_archive.pt", weights_only=False)
    arch_rows = archive_modes(env, archive)

    eval_rounds = sorted({0, *range(0, total_rounds + 1, args.stride), total_rounds})
    per_round, embeddings_by_round, sigma_by_round = {}, {}, {}
    for round_i in eval_rounds:
        policy = policy_at(round_i)
        gp = gp_for_round(policy, env, arch_rows, round_i, lengthscale)
        phi = bank_embeddings(policy, contexts, plans, DEFAULT_S)
        gp_sigma = gp.sigma(torch.from_numpy(phi)).numpy()
        embeddings_by_round[round_i] = phi
        sigma_by_round[round_i] = gp_sigma
        row = {"round": round_i, **probe_metrics(phi, meta, args.seed)}
        row.update(geometry_metrics(phi, meta, gp_sigma, gp.X, seed=args.seed))
        counts: dict[str, int] = {}
        for r in arch_rows:
            if (r["valid"] and r["mode"] in ROUTE_MODES
                    and round_i - 2 < r["round"] <= round_i):
                counts[r["mode"]] = counts.get(r["mode"], 0) + 1
        known = {mode for mode, count in counts.items() if count >= 3}
        row["known_modes"] = sorted(known)
        row["discovery"] = discovery_metric(policy, task, gp, known, gammas, beta,
                                            seed=args.seed + 100 + round_i)
        per_round[round_i] = row
        d = row["discovery"]
        print(f"round {round_i:2d}: knn_mode {row['knn_mode_acc']:.2f} "
              f"knn_valid {row['knn_validity_acc']:.2f} "
              f"sil {row['mode_silhouette']:.2f} "
              + (f"discovery {d['p_new_mode_high_sigma']:.2f} vs {d['p_new_mode_uniform']:.2f}"
                 if d else "discovery n/a (all modes known)"), flush=True)

    final = eval_rounds[-1]
    ablation = {}
    for flow_time in FLOW_TIMES:
        phi = bank_embeddings(policy_at(final), contexts, plans, flow_time)
        ablation[str(flow_time)] = probe_metrics(phi, meta, args.seed)

    tsne_rounds = sorted({0, total_rounds // 2, final})
    for r in tsne_rounds:
        if r not in embeddings_by_round:
            policy = policy_at(r)
            embeddings_by_round[r] = bank_embeddings(policy, contexts, plans, DEFAULT_S)
            gp = gp_for_round(policy, env, arch_rows, r, lengthscale)
            sigma_by_round[r] = gp.sigma(torch.from_numpy(embeddings_by_round[r])).numpy()
    tsne_panels({r: embeddings_by_round[r] for r in tsne_rounds}, meta,
                sigma_by_round, output / "tsne_panels.png", args.seed)
    tsne_by_flow_time(policy_at(final), contexts, plans, meta,
                      output / "tsne_by_flow_time.png", args.seed)

    rounds = sorted(per_round)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    axes[0].plot(rounds, [per_round[r]["knn_mode_acc"] for r in rounds], "-o",
                 label="kNN route mode")
    axes[0].plot(rounds, [per_round[r]["knn_validity_acc"] for r in rounds], "-s",
                 label="kNN validity")
    axes[0].plot(rounds, [per_round[r]["knn_control_magnitude_acc"] for r in rounds], "--",
                 color="#9aa0a6", label="kNN |u| (shortcut check)")
    axes[0].set_title("local probe accuracy")
    axes[0].legend(fontsize=8)
    axes[1].plot(rounds, [per_round[r]["sigma_vs_nearest_buffer_pearson"] or np.nan
                          for r in rounds], "-o", color="#174f92")
    axes[1].set_title(r"corr($\sigma$, nearest queried-feature distance)")
    discovery_rounds = [r for r in rounds if per_round[r]["discovery"]]
    axes[2].bar([r - 0.45 for r in discovery_rounds],
                [per_round[r]["discovery"]["p_new_mode_high_sigma"]
                 for r in discovery_rounds], width=0.85, color="#5b2d8f",
                label=r"high-$\sigma$ B")
    axes[2].bar([r + 0.45 for r in discovery_rounds],
                [per_round[r]["discovery"]["p_new_mode_uniform"]
                 for r in discovery_rounds], width=0.85, color="#9aa0a6", label="uniform B")
    axes[2].set_title("P(new valid route mode in selected B)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("round")
    fig.tight_layout()
    fig.savefig(output / "representation_probes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    eval_json = expansion / "eval" / "raw_eval.json"
    coverage = None
    if eval_json.exists():
        raw = json.loads(eval_json.read_text())["summary"]
        coverage = {int(k): v["pooled"]["coverage"] for k, v in raw.items()}
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(rounds, [per_round[r]["mode_silhouette"] for r in rounds], "-o",
                color="#174f92", label="mode silhouette on $\\phi_s$")
        shared = [r for r in rounds if r in coverage]
        ax2 = ax.twinx()
        ax2.plot(shared, [coverage[r] for r in shared], "-s", color="#c8321b",
                 label="raw route coverage")
        ax.set_xlabel("round")
        ax.set_ylabel("silhouette")
        ax2.set_ylabel("coverage")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_title("Feature clustering vs task-space route coverage", weight="bold")
        fig.tight_layout()
        fig.savefig(output / "cluster_vs_coverage.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    pooled_trials = [per_round[r]["discovery"] for r in rounds if per_round[r]["discovery"]]
    decisive = {
        "p_new_mode_high_sigma": (float(np.mean([d["p_new_mode_high_sigma"]
                                                 for d in pooled_trials]))
                                  if pooled_trials else None),
        "p_new_mode_uniform": (float(np.mean([d["p_new_mode_uniform"]
                                              for d in pooled_trials]))
                               if pooled_trials else None),
        "rounds_with_missing_modes": len(pooled_trials),
    }
    payload = {
        "bank_size": len(meta), "flow_time_default": DEFAULT_S,
        "noise_draws": NOISE_DRAWS, "per_round": per_round,
        "flow_time_ablation_final_round": ablation, "decisive_metric": decisive,
        "coverage_by_round": coverage,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n")

    verdict = (decisive["p_new_mode_high_sigma"] is not None
               and decisive["p_new_mode_high_sigma"] > decisive["p_new_mode_uniform"])
    lines = ["# Noised-representation audit (ball task)", "",
             f"Fixed bank of {len(meta)} candidates at the approach state; "
             f"s={DEFAULT_S}, {NOISE_DRAWS} shared base-noise draws averaged per plan.", "",
             "## Decisive acquisition metric", "",
             f"- P(new valid route mode | high-sigma B) = "
             f"**{decisive['p_new_mode_high_sigma']}**",
             f"- P(new valid route mode | uniform B) = **{decisive['p_new_mode_uniform']}**",
             f"- verdict: **{'PASS' if verdict else 'FAIL'}** — sigma-tilted acquisition "
             f"{'does' if verdict else 'does NOT'} preferentially reach unseen valid route "
             "modes (averaged over rounds that still had missing modes).", "",
             "## Per-round probes (s=0.9)", "",
             "| round | kNN mode | kNN valid | lin AUROC | rbf AUROC | mode probe | "
             "silhouette | corr(sigma,novelty) | kNN |u| shortcut |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rounds:
        row = per_round[r]
        lines.append(
            f"| {r} | {row['knn_mode_acc']:.2f} | {row['knn_validity_acc']:.2f} | "
            f"{row.get('linear_validity_auroc', float('nan')):.2f} | "
            f"{row.get('rbf_validity_auroc', float('nan')):.2f} | "
            f"{row['mode_probe_acc']:.2f} | {row['mode_silhouette']:.2f} | "
            f"{(row['sigma_vs_nearest_buffer_pearson'] if row['sigma_vs_nearest_buffer_pearson'] is not None else float('nan')):.2f} | "
            f"{row['knn_control_magnitude_acc']:.2f} |")
    lines += ["", "## Flow-time ablation (final round)", "",
              "| s | kNN mode | kNN valid | mode probe | silhouette |",
              "|---:|---:|---:|---:|---:|"]
    for flow_time in FLOW_TIMES:
        row = ablation[str(flow_time)]
        lines.append(f"| {flow_time} | {row['knn_mode_acc']:.2f} | "
                     f"{row['knn_validity_acc']:.2f} | {row['mode_probe_acc']:.2f} | "
                     f"{row['mode_silhouette']:.2f} |")
    lines += ["", "t-SNE is used for visualization only; the claims above rest on the "
              "fixed-bank local probes and the acquisition discovery rate.", ""]
    (output / "REPRESENTATION_REPORT.md").write_text("\n".join(lines))
    print("[outputs]", output, flush=True)


if __name__ == "__main__":
    main()
