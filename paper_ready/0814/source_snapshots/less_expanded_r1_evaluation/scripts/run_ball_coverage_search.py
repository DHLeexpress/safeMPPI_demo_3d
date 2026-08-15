#!/usr/bin/env python3
"""Run the first causal four-route coverage sweep and rank raw M=50 checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


MODES = ("below", "above", "left", "right")
GAMMAS = (0.1, 0.3, 0.5, 1.0)
def arm(name, archive, perturb, K, repeats, alpha, beta, *,
        B=4, learning_rate=1e-4, replay_rounds=3,
        top_fraction=0.2, cap=1800, adaptive=False, ess_target=0.5,
        optimizer_steps=None, replay_selector="sigma_top",
        context_quota=4, action_weight=0.5, cluster_count=32,
        execution_rule="min_cost", execution_ess_target=0.25,
        paired_noised_representation=False,
        acquisition_feature="learned_phi", coverage_replay="none",
        rbf_lengthscale=None):
    return {
        "name": name, "archive": archive, "perturb": perturb, "K": K, "B": B,
        "repeats": repeats, "alpha": alpha, "beta": beta,
        "learning_rate": learning_rate, "replay_rounds": replay_rounds,
        "top_fraction": top_fraction, "cap": cap,
        "adaptive": adaptive, "ess_target": ess_target,
        "optimizer_steps": optimizer_steps,
        "replay_selector": replay_selector,
        "context_quota": context_quota,
        "action_weight": action_weight,
        "cluster_count": cluster_count,
        "execution_rule": execution_rule,
        "execution_ess_target": execution_ess_target,
        "paired_noised_representation": paired_noised_representation,
        "acquisition_feature": acquisition_feature,
        "coverage_replay": coverage_replay,
        "rbf_lengthscale": rbf_lengthscale,
    }


STAGE1_ARMS = (
    arm("exec_p20_k16_rep04_a01_b1e3", "executed_plus_nvp_negative", .20, 16, 4, .01, 1e-3),
    arm("allq_p20_k16_rep04_a01_b1e3", "all_queries", .20, 16, 4, .01, 1e-3),
    arm("allq_p20_k16_rep16_a01_b1e3", "all_queries", .20, 16, 16, .01, 1e-3),
    arm("allq_p35_k16_rep04_a01_b1e3", "all_queries", .35, 16, 4, .01, 1e-3),
    arm("allq_p35_k16_rep16_a01_b1e3", "all_queries", .35, 16, 16, .01, 1e-3),
    arm("allq_p35_k16_rep16_a00_b1e3", "all_queries", .35, 16, 16, 0., 1e-3),
)
STAGE2_ARMS = (
    arm("allq_p50_k16_rep04_a00_b1e3", "all_queries", .50, 16, 4, 0., 1e-3),
    arm("allq_p50_k32_rep04_a00_b1e3", "all_queries", .50, 32, 4, 0., 1e-3),
    arm("allq_p50_k32_rep16_a00_b1e3", "all_queries", .50, 32, 16, 0., 1e-3),
    arm("allq_p70_k32_rep04_a00_b1e3", "all_queries", .70, 32, 4, 0., 1e-3),
    arm("allq_p70_k32_rep16_a00_b1e3", "all_queries", .70, 32, 16, 0., 1e-3),
    arm("allq_p50_k32_rep16_a00_b1e4", "all_queries", .50, 32, 16, 0., 1e-4),
)
STAGE3_ARMS = (
    arm("p050_s016_lr1em6_b1em2", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-6, replay_rounds=1, cap=512,
        optimizer_steps=16),
    arm("p050_s032_lr1em6_b1em2", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-6, replay_rounds=1, cap=512,
        optimizer_steps=32),
    arm("p050_s064_lr1em6_b1em2", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-6, replay_rounds=1, cap=512,
        optimizer_steps=64),
    arm("p070_s032_lr1em6_b1em2", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-6, replay_rounds=1, cap=512,
        optimizer_steps=32),
    arm("p050_s032_lr1em6_b1em3", "all_queries", .5, 64, 1, 0., 1e-3,
        B=16, learning_rate=1e-6, replay_rounds=1, cap=512,
        optimizer_steps=32),
    arm("p050_s032_lr1em6_adapt50", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-6, replay_rounds=1, cap=512,
        adaptive=True, ess_target=.5, optimizer_steps=32),
)
STAGE4_ARMS = (
    arm("sigma_min_cost_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=8e-8, replay_rounds=3, cap=512,
        optimizer_steps=192),
    arm("uniform_min_cost_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=8e-8, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="uniform"),
    arm("kcenter_min_cost_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=8e-8, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="context_kcenter"),
    arm("kcenter_max_margin_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=8e-8, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="context_kcenter",
        execution_rule="max_margin"),
    arm("kcenter_uniform_positive_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=8e-8, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="context_kcenter",
        execution_rule="uniform_positive"),
    arm("kcenter_uniform_positive_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=8e-8, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="context_kcenter",
        execution_rule="uniform_positive"),
)
STAGE5_ARMS = (
    arm("cluster_softmin_e010_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.10),
    arm("cluster_softmin_e025_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.25),
    arm("cluster_softmin_e050_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.50),
    arm("cluster_softmin_e025_eta100_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        action_weight=1.0, execution_rule="softmin_cost",
        execution_ess_target=.25),
    arm("cluster_softmin_e025_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.25),
    arm("cluster_softmin_e050_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.50),
)
STAGE6_ARMS = (
    arm("paired_cluster_softmin_e025_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.25,
        paired_noised_representation=True),
    arm("paired_cluster_softmin_e025_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.25,
        paired_noised_representation=True),
    arm("paired_cluster_softmin_e050_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="cluster_balanced",
        execution_rule="softmin_cost", execution_ess_target=.50,
        paired_noised_representation=True),
    arm("paired_cluster_adapt025_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        adaptive=True, ess_target=.25, optimizer_steps=192,
        replay_selector="cluster_balanced", execution_rule="softmin_cost",
        execution_ess_target=.25, paired_noised_representation=True),
    arm("paired_kcenter_softmin_e025_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="context_kcenter",
        execution_rule="softmin_cost", execution_ess_target=.25,
        paired_noised_representation=True),
    arm("paired_sigma_softmin_e025_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, cap=512,
        optimizer_steps=192, replay_selector="sigma_top",
        execution_rule="softmin_cost", execution_ess_target=.25,
        paired_noised_representation=True),
)
STAGE7_ARMS = (
    arm("angular_only_l080_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, top_fraction=1.0, cap=512,
        optimizer_steps=192, replay_selector="uniform",
        acquisition_feature="task_angular", rbf_lengthscale=.8),
    arm("voronoi_only_phi_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, top_fraction=1.0, cap=512,
        replay_selector="uniform", coverage_replay="circular_voronoi"),
    arm("angular_voronoi_l040_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, top_fraction=1.0, cap=512,
        replay_selector="uniform", acquisition_feature="task_angular",
        coverage_replay="circular_voronoi", rbf_lengthscale=.4),
    arm("angular_voronoi_l080_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, top_fraction=1.0, cap=512,
        replay_selector="uniform", acquisition_feature="task_angular",
        coverage_replay="circular_voronoi", rbf_lengthscale=.8),
    arm("angular_voronoi_l160_p070", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, top_fraction=1.0, cap=512,
        replay_selector="uniform", acquisition_feature="task_angular",
        coverage_replay="circular_voronoi", rbf_lengthscale=1.6),
    arm("angular_voronoi_l080_p050", "all_queries", .5, 64, 1, 0., 1e-2,
        B=16, learning_rate=1e-7, replay_rounds=3, top_fraction=1.0, cap=512,
        replay_selector="uniform", acquisition_feature="task_angular",
        coverage_replay="circular_voronoi", rbf_lengthscale=.8),
    arm("angular_voronoi_l080_p070_lr3em8", "all_queries", .7, 64, 1, 0., 1e-2,
        B=16, learning_rate=3e-8, replay_rounds=3, top_fraction=1.0, cap=512,
        replay_selector="uniform", acquisition_feature="task_angular",
        coverage_replay="circular_voronoi", rbf_lengthscale=.8),
)


def run_checked(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout


def score_evaluation(path: Path, episodes: int) -> list[dict]:
    payload = json.loads(path.read_text())
    scored = []
    for round_text, rows in payload["rows"].items():
        counts = {
            gamma: {mode: 0 for mode in MODES}
            for gamma in GAMMAS
        }
        successes = {gamma: 0 for gamma in GAMMAS}
        for row in rows:
            gamma = min(GAMMAS, key=lambda value: abs(value - float(row["gamma"])))
            if row["status"] == "SUCCESS":
                successes[gamma] += 1
                if row["mode"] in MODES:
                    counts[gamma][row["mode"]] += 1
        j_gamma = {
            gamma: 4.0 * min(counts[gamma].values()) / episodes
            for gamma in GAMMAS
        }
        tv_gamma = {
            gamma: 0.5 * sum(
                abs(counts[gamma][mode] / episodes - 0.25)
                for mode in MODES
            )
            for gamma in GAMMAS
        }
        cell_count = sum(
            counts[gamma][mode] > 0 for gamma in GAMMAS for mode in MODES
        )
        min_cell = min(
            counts[gamma][mode] for gamma in GAMMAS for mode in MODES
        )
        min_sr = min(successes[gamma] / episodes for gamma in GAMMAS)
        mean_sr = sum(successes.values()) / (episodes * len(GAMMAS))
        scored.append({
            "round": int(round_text),
            "counts": {str(gamma): counts[gamma] for gamma in GAMMAS},
            "successes": {str(gamma): successes[gamma] for gamma in GAMMAS},
            "nonzero_cells": int(cell_count),
            "min_cell": int(min_cell),
            "J_worst": min(j_gamma.values()),
            "J_mean": sum(j_gamma.values()) / len(GAMMAS),
            "min_SR": min_sr,
            "mean_SR": mean_sr,
            "mean_TV": sum(tv_gamma.values()) / len(GAMMAS),
            "accepted": bool(
                min_cell >= 6
                and min_sr >= 0.8
                and max(tv_gamma.values()) <= 0.2
            ),
        })
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=93100)
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "5", "6", "7"),
                        default="1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pretrain = args.pretrain_dir.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    arms = {
        "1": STAGE1_ARMS, "2": STAGE2_ARMS,
        "3": STAGE3_ARMS, "4": STAGE4_ARMS, "5": STAGE5_ARMS,
        "6": STAGE6_ARMS, "7": STAGE7_ARMS,
    }[args.stage]

    def run_arm(spec: dict) -> dict:
        name = spec["name"]
        arm_dir = output_root / name
        train = [
            sys.executable, "scripts/run_ball_expansion.py",
            "--pretrain-dir", str(pretrain),
            "--output", str(arm_dir),
            "--rounds", str(args.rounds),
            "--parallel-episodes", "20",
            "--K", str(spec["K"]), "--B", str(spec["B"]),
            "--inner-steps", str(spec["repeats"]),
            "--learning-rate", str(spec["learning_rate"]),
            "--beta", str(spec["beta"]),
            "--replay-rounds", str(spec["replay_rounds"]),
            "--replay-top-fraction", str(spec["top_fraction"]),
            "--gp-buffer-cap", str(spec["cap"]),
            "--candidate-perturb-std", str(spec["perturb"]),
            "--candidate-perturb-scope", "coherent_horizon",
            "--negative-alpha", str(spec["alpha"]),
            "--archive-rule", spec["archive"],
            "--execution-rule", spec["execution_rule"],
            "--replay-selector", spec["replay_selector"],
            "--replay-context-quota", str(spec["context_quota"]),
            "--replay-action-weight", str(spec["action_weight"]),
            "--replay-cluster-count", str(spec["cluster_count"]),
            "--execution-ess-target", str(spec["execution_ess_target"]),
            "--acquisition-feature", spec["acquisition_feature"],
            "--coverage-replay", spec["coverage_replay"],
            "--tight-corridor",
            "--event-log", "none",
            "--seed", "1200",
        ]
        if spec["adaptive"]:
            train.extend(["--adaptive-beta", "--ess-target", str(spec["ess_target"])])
        if spec["paired_noised_representation"]:
            train.append("--paired-noised-representation")
        if spec["rbf_lengthscale"] is not None:
            train.extend(["--rbf-lengthscale", str(spec["rbf_lengthscale"])])
        if spec["optimizer_steps"] is not None:
            train.extend([
                "--optimizer-steps-per-round", str(spec["optimizer_steps"]),
            ])
        train_log = run_checked(train, root, environment)
        (arm_dir / "train.log").write_text(train_log)
        evaluate = [
            sys.executable, "scripts/evaluate_ball_expansion.py",
            "--pretrain-dir", str(pretrain),
            "--expansion", str(arm_dir),
            "--episodes", str(args.episodes),
            "--stride", "1",
            "--seed", str(args.eval_seed),
            "--metrics-only",
        ]
        eval_log = run_checked(evaluate, root, environment)
        (arm_dir / "eval.log").write_text(eval_log)
        scores = score_evaluation(arm_dir / "eval" / "raw_eval.json", args.episodes)
        return {
            "arm": name,
            **{key: value for key, value in spec.items() if key != "name"},
            "scores": scores,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_arm, spec): spec["name"]
            for spec in arms
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            best = max(
                result["scores"],
                key=lambda row: (
                    row["nonzero_cells"], row["J_worst"], row["J_mean"],
                    row["mean_SR"], -row["mean_TV"],
                ),
            )
            print(
                f"[{result['arm']}] best r{best['round']} cells={best['nonzero_cells']}/16 "
                f"Jworst={best['J_worst']:.3f} Jmean={best['J_mean']:.3f} "
                f"SR={best['mean_SR']:.3f} mincell={best['min_cell']}",
                flush=True,
            )

    candidates = [
        {**score, "arm": result["arm"]}
        for result in results for score in result["scores"]
    ]
    candidates.sort(
        key=lambda row: (
            row["nonzero_cells"], row["J_worst"], row["J_mean"],
            row["mean_SR"], -row["mean_TV"],
        ),
        reverse=True,
    )
    summary = {
        "status": "BALL_COVERAGE_STAGE_COMPLETE",
        "episodes_per_gamma": args.episodes,
        "eval_seed": args.eval_seed,
        "results": sorted(results, key=lambda row: row["arm"]),
        "ranking": candidates,
        "winner": candidates[0],
    }
    (output_root / "stage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    winner = candidates[0]
    print(
        f"[winner] {winner['arm']}@r{winner['round']} "
        f"cells={winner['nonzero_cells']}/16 Jworst={winner['J_worst']:.3f} "
        f"Jmean={winner['J_mean']:.3f} SR={winner['mean_SR']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
