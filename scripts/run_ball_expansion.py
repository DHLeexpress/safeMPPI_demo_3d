"""Run B1 Safe Flow Expansion on the ball task with full per-step event logging.

Loads the pretrained 10-D-context flow policy, wires the BallFlowTask (bounded GREEN
trajectory-fitted variable-face verifier + untilted native SafeMPPI execution cost) into
``run_safe_expansion``, and stores every acquisition event (all K candidates, marginal sigma,
selected B, verifier verdicts, executed index, robot state) for the
mechanism/representation analyses. The BLUE one-step nominal check is logged only as a diagnostic.
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

from safe_mppi.ball_flow_task import BallFlowTask, demo_windows, load_policy
from safe_mppi.ball_flow_theta import ThetaBallFlowTask, demo_windows_theta
from safe_mppi.config import load_config
from safe_mppi.expansion import ExpansionConfig, run_safe_expansion
from safe_mppi.lab_flow_expansion import (
    LabFlowExpansionTask,
    load_lab_expansion_policy,
)
from safe_mppi.lab_clutter_expansion import (
    LabClutterSphereExpansionTask,
    load_lab_clutter_expansion_policy,
)
from safe_mppi.lab_reference_flow_task import lab_reference_demo_windows

ROOT = Path(__file__).resolve().parents[1]


def _is_lab_pretrain(manifest: dict) -> bool:
    return (
        manifest.get("kind") == "lab raw-command reference-flow pretraining"
        or str(manifest.get("context_schema", "")).startswith("lab_")
    )


def _lab_source_demo_dir(pretrain_dir: Path, manifest: dict) -> Path:
    source = Path(manifest["source_demo_dir"])
    if not source.is_absolute():
        source = pretrain_dir / source
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(
            "lab pretraining manifest source_demo_dir is unavailable: "
            f"{source}"
        )
    return source


def _lab_task_config(pretrain_dir: Path, manifest: dict):
    source = _lab_source_demo_dir(pretrain_dir, manifest)
    return load_config(source / "resolved_config.json")


def _lab_clutter_profile(task_config) -> bool:
    """Classify an optional lab randomization without unsafe fall-through."""
    randomization = task_config.raw.get("scene_randomization", {})
    if not randomization.get("enabled", False):
        return False
    if (
        randomization.get("obstacle_family") != "spheres"
        or int(randomization.get("count", -1)) != 3
    ):
        raise ValueError(
            "--lab-task-config with enabled scene_randomization requires "
            "exactly three spheres"
        )
    return True


def _record_lab_setup_failure(
    output: Path,
    *,
    output_was_unsafe: bool,
    stage: str,
    error: Exception,
) -> None:
    """Persist lab setup failures without overwriting an existing run."""
    if output_was_unsafe:
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "FAILED.json").write_text(json.dumps({
        "status": "EXPANSION_FAILED_CLOSED",
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error": str(error),
    }, indent=2) + "\n")


@torch.no_grad()
def _pretrained_phi_calibration(
    policy,
    pretrain_dir: Path,
    task_config,
    seed: int,
    *,
    flow_base_std: float,
    paired_noised_representation: bool,
    count: int = 50,
) -> torch.Tensor:
    """Build endpoint or paired features from scale-matched pretrained proposals."""
    demo_dir = pretrain_dir / "demos"
    if not (demo_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            "--paired-noised-representation requires the pretrained demo archive at "
            f"{demo_dir}"
        )
    loader = (
        demo_windows_theta
        if int(policy.context_dim) == 12 else demo_windows
    )
    contexts_np, _, meta, demo_config = loader(demo_dir)
    if contexts_np.ndim != 2 or len(contexts_np) < count:
        raise ValueError(
            "paired calibration requires at least 50 two-dimensional demo contexts"
        )
    if contexts_np.shape[1] != int(policy.context_dim):
        raise ValueError(
            "paired calibration context dimension does not match the pretrained policy: "
            f"{contexts_np.shape[1]} != {policy.context_dim}"
        )

    task_gammas = np.asarray(task_config.data.gammas, dtype=np.float64)
    demo_gammas = np.asarray(demo_config.data.gammas, dtype=np.float64)
    if (task_gammas.shape != demo_gammas.shape
            or not np.allclose(task_gammas, demo_gammas, rtol=0.0, atol=1.0e-7)):
        raise ValueError(
            "paired calibration demo gammas do not match demo_config.json"
        )

    metadata_gammas = np.asarray([row["gamma"] for row in meta], dtype=np.float64)
    selection_generator = torch.Generator().manual_seed(int(seed) + 17001)
    selected: list[int] = []
    base_count, remainder = divmod(count, len(task_gammas))
    for gamma_index, gamma in enumerate(task_gammas):
        target = base_count + int(gamma_index < remainder)
        available = np.flatnonzero(
            np.isclose(metadata_gammas, gamma, rtol=0.0, atol=1.0e-7)
        )
        if len(available) < target:
            raise ValueError(
                f"paired calibration needs {target} contexts for gamma={gamma:g}, "
                f"but the demo archive contains {len(available)}"
            )
        order = torch.randperm(len(available), generator=selection_generator)[:target]
        selected.extend(available[order.numpy()].tolist())

    contexts = torch.from_numpy(contexts_np[np.asarray(selected)]).to(
        next(policy.parameters()).device
    )
    sample_generator = torch.Generator(device=contexts.device)
    sample_generator.manual_seed(int(seed) + 17002)
    plans, bases = [], []
    sampler = getattr(policy, "sample_with_base", None)
    if not callable(sampler):
        raise TypeError(
            "scale-matched phi calibration requires policy.sample_with_base"
        )
    for context in contexts:
        if flow_base_std == 1.0:
            plan, base = sampler(context, 1, sample_generator)
        else:
            plan, base = sampler(
                context, 1, sample_generator, base_std=flow_base_std,
            )
        if plan.shape != base.shape or len(plan) != 1:
            raise ValueError(
                "sample_with_base must return one equally shaped plan/base pair"
            )
        plans.append(plan[0])
        bases.append(base[0])
    paired_plans = torch.stack(plans)
    paired_bases = torch.stack(bases)
    features = policy.embed(
        contexts,
        paired_plans,
        base=(paired_bases if paired_noised_representation else None),
    )
    if features.ndim != 2 or len(features) != count or not torch.isfinite(features).all():
        raise ValueError(
            "scale-matched phi calibration did not produce 50 finite feature rows"
        )
    return features


@torch.no_grad()
def _lab_pretrained_phi_calibration(
    policy,
    pretrain_dir: Path,
    pretrain_manifest: dict,
    task_config,
    seed: int,
    *,
    flow_base_std: float,
    paired_noised_representation: bool,
    count: int = 50,
) -> torch.Tensor:
    """Scale-matched calibration for raw10 or visual lab contexts."""
    source = _lab_source_demo_dir(pretrain_dir, pretrain_manifest)
    contexts_np, _, metadata, demo_config = lab_reference_demo_windows(
        source,
        context_schema=policy.context_schema,
    )
    if len(contexts_np) < count:
        raise ValueError("lab calibration requires at least 50 demo contexts")
    task_gammas = np.asarray(task_config.data.gammas, dtype=np.float64)
    demo_gammas = np.asarray(demo_config.data.gammas, dtype=np.float64)
    if (
        task_gammas.shape != demo_gammas.shape
        or not np.allclose(
            task_gammas, demo_gammas, rtol=0.0, atol=1.0e-7,
        )
    ):
        raise ValueError("lab calibration demo gammas do not match task config")
    metadata_gammas = np.asarray(
        [row["gamma"] for row in metadata], dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed) + 17001)
    selected: list[int] = []
    base_count, remainder = divmod(count, len(task_gammas))
    for gamma_index, gamma in enumerate(task_gammas):
        target = base_count + int(gamma_index < remainder)
        available = np.flatnonzero(
            np.isclose(metadata_gammas, gamma, rtol=0.0, atol=1.0e-7)
        )
        if len(available) < target:
            raise ValueError(
                f"lab calibration needs {target} contexts for gamma={gamma:g}"
            )
        selected.extend(
            rng.choice(available, size=target, replace=False).tolist()
        )
    device = next(policy.parameters()).device
    contexts = torch.from_numpy(
        contexts_np[np.asarray(selected)]
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 17002)
    plans, bases = [], []
    for context in contexts:
        plan, base = policy.sample_with_base(
            context, 1, generator, base_std=flow_base_std,
        )
        plans.append(plan[0])
        bases.append(base[0])
    plans_tensor = torch.stack(plans)
    bases_tensor = torch.stack(bases)
    features = policy.embed(
        contexts,
        plans_tensor,
        base=(bases_tensor if paired_noised_representation else None),
    )
    if (
        features.ndim != 2
        or len(features) != count
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError(
            "lab scale-matched calibration did not produce finite features"
        )
    return features


def _learned_phi_calibration(
    policy,
    pretrain_dir: Path,
    pretrain_manifest: dict,
    task_config,
    seed: int,
    *,
    lab_profile: bool,
    flow_base_std: float,
    paired_noised_representation: bool,
) -> torch.Tensor:
    """Use generator samples for every lab RBF calibration."""
    if lab_profile:
        return _lab_pretrained_phi_calibration(
            policy,
            pretrain_dir,
            pretrain_manifest,
            task_config,
            seed,
            flow_base_std=flow_base_std,
            paired_noised_representation=paired_noised_representation,
        )
    if paired_noised_representation or flow_base_std != 1.0:
        return _pretrained_phi_calibration(
            policy,
            pretrain_dir,
            task_config,
            seed,
            flow_base_std=flow_base_std,
            paired_noised_representation=paired_noised_representation,
        )
    return torch.load(
        pretrain_dir / "calibration_features.pt", weights_only=False,
    )


@torch.no_grad()
def _angular_pretrained_calibration(
    policy,
    task: BallFlowTask,
    pretrain_dir: Path,
    seed: int,
    flow_base_std: float,
    count: int = 50,
) -> torch.Tensor:
    """Build continuous angular features from pretrained policy samples."""
    loader = (
        demo_windows_theta
        if int(policy.context_dim) == 12 else demo_windows
    )
    contexts_np, _, meta, _ = loader(pretrain_dir / "demos")
    metadata_gammas = np.asarray([row["gamma"] for row in meta], dtype=np.float64)
    gammas = np.asarray(task.config.data.gammas, dtype=np.float64)
    rng = np.random.default_rng(int(seed) + 18001)
    selected: list[int] = []
    base_count, remainder = divmod(count, len(gammas))
    for gamma_index, gamma in enumerate(gammas):
        target = base_count + int(gamma_index < remainder)
        available = np.flatnonzero(
            np.isclose(metadata_gammas, gamma, rtol=0.0, atol=1.0e-7)
        )
        if len(available) < target:
            raise ValueError(
                f"angular calibration needs {target} contexts for gamma={gamma:g}"
            )
        selected.extend(rng.choice(available, size=target, replace=False).tolist())
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed) + 18002)
    values = []
    for row in contexts_np[np.asarray(selected)]:
        context = torch.from_numpy(row).to(device)
        if flow_base_std == 1.0:
            plan = policy.sample(context, 1, generator)
        else:
            plan = policy.sample(
                context, 1, generator, base_std=flow_base_std,
            )
        values.append(task.angular_descriptors(context, plan)[0])
    features = torch.stack(values)
    if (features.shape != (count, 2) or not bool(torch.isfinite(features).all())
            or bool((features.norm(dim=1) <= 1.0e-12).any())):
        raise ValueError("angular calibration did not produce 50 finite unit directions")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, default=ROOT / "outputs" / "ball_flow")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "ball_flow" / "expansion")
    parser.add_argument(
        "--lab-task-config",
        type=Path,
        help=(
            "optional lab expansion task override; use the randomized-sphere "
            "config to transfer a cylinder-pretrained visual policy OOD"
        ),
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--first-layer-lr-scale", type=float, default=1.0,
        help=(
            "expansion-only multiplier for the first flow Linear layer and, "
            "for lab visual policies, the visual encoder; the representation "
            "layer and head retain --learning-rate"
        ),
    )
    parser.add_argument("--beta", type=float, default=None,
                        help="fixed beta; default uses the one-time pretrained calibration")
    parser.add_argument("--adaptive-beta", action="store_true",
                        help="recalibrate beta after each round to the requested ESS target")
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument("--parallel-episodes", type=int, default=10)
    parser.add_argument(
        "--verifier-workers",
        type=int,
        default=1,
        help=(
            "persistent CPU worker processes used to verify independent active "
            "episode contexts; 1 preserves serial verification"
        ),
    )
    parser.add_argument(
        "--max-retry-batches", type=int, default=4,
        help=(
            "whole-episode batches attempted per gamma by "
            "successful_executed_windows; never retries an individual step"
        ),
    )
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=4,
                        help="optimizer repeats on each microbatch during one exact replay pass")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--optimizer-steps-per-round", type=int, default=None,
        help=("optional cap on distinct replay microbatches per round; unlike "
              "--inner-steps, this controls total sampled rows before repeats"),
    )
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--replay-top-fraction", type=float, default=0.2)
    parser.add_argument(
        "--replay-selector",
        choices=("sigma_top", "uniform", "context_kcenter", "cluster_balanced"),
        default="sigma_top",
    )
    parser.add_argument("--replay-context-quota", type=int, default=4)
    parser.add_argument("--replay-action-weight", type=float, default=0.5)
    parser.add_argument("--replay-cluster-count", type=int, default=32)
    parser.add_argument(
        "--flow-base-std",
        type=float,
        default=1.0,
        help=(
            "expansion-only standard deviation of the initial flow latent "
            "x0~N(0,std^2 I); 1 preserves canonical unit-normal sampling. "
            "This is distinct from post-flow --candidate-perturb-std"
        ),
    )
    parser.add_argument("--replay-rounds", type=int, default=3)
    parser.add_argument(
        "--gp-buffer-cap", type=int, default=1800,
        help=(
            "total cap for bounded GP modes (split equally across gamma for "
            "sliding successful-window modes); serialized but ignored by "
            "cumulative_success_per_gamma_frozen_phi_exact"
        ),
    )
    parser.add_argument(
        "--gp-exact-max-rows-per-gamma", type=int, default=1024,
        help=(
            "hard fail-closed support limit for the exact cumulative "
            "successful-window GP; rows are never thinned"
        ),
    )
    parser.add_argument("--rbf-lengthscale", type=float, default=None)
    parser.add_argument(
        "--gp-reference-mode",
        choices=(
            "recent_current_phi",
            "round1_fixed_frozen_phi",
            "cumulative_accepted_frozen_phi",
            "cumulative_success_per_gamma_frozen_phi_exact",
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
        ),
        default="recent_current_phi",
        help=(
            "rolling recent-current representation (canonical), or one deterministic "
            "round-1 accepted-positive reference embedded forever by frozen pretrained phi, "
            "a bounded cumulative frozen-phi anchor/adaptive coreset, or exact "
            "per-gamma cumulative successful executed-window support, or a "
            "bounded per-gamma successful-window support embedded by either "
            "frozen pretrained phi or the current round phi"
        ),
    )
    parser.add_argument(
        "--gp-sliding-row-selector",
        choices=("trajectory_uniform", "fifo_tail"),
        default="trajectory_uniform",
        help=(
            "trajectory_uniform allocates equal successful-trajectory mass and "
            "evenly samples departure-to-tail windows; fifo_tail exactly restores "
            "the legacy high-window-start selector"
        ),
    )
    parser.add_argument(
        "--target-gate-start-round", type=int, default=None,
        help=(
            "optional task target gate for execution/replay only; it never changes "
            "the safety label"
        ),
    )
    parser.add_argument(
        "--target-region",
        choices=("above_wedge", "above_halfspace"),
        default="above_wedge",
        help=(
            "q1-only target used when the target gate is active: z-2 >= |y| "
            "(default), or the less restrictive z >= 2"
        ),
    )
    parser.add_argument("--candidate-perturb-std", type=float, default=0.5)
    parser.add_argument("--candidate-perturb-scope",
                        choices=("first_action", "coherent_horizon"),
                        default="coherent_horizon")
    parser.add_argument("--negative-alpha", type=float, default=0.01)
    parser.add_argument("--archive-rule",
                        choices=("all_queries", "executed_only",
                                 "executed_plus_nvp_negative",
                                 "successful_executed_windows"),
                        default="all_queries")
    parser.add_argument(
        "--successful-trajectory-selector",
        choices=("lowest_episode_id", "max_above_fraction", "max_mean_z"),
        default="lowest_episode_id",
        help=(
            "admission ranking used only by successful_executed_windows; "
            "max_above_fraction ranks terminal SUCCESS trajectories by the "
            "fraction of post-action states with z>=2, while max_mean_z ranks "
            "their mean post-action height; both break ties by episode id"
        ),
    )
    parser.add_argument(
        "--successful-trajectories-per-gamma",
        type=int,
        default=1,
        help=(
            "number of distinct ranked commit-capable terminal SUCCESS "
            "trajectories required and archived per gamma/round by "
            "successful_executed_windows"
        ),
    )
    parser.add_argument(
        "--replay-acceptance",
        choices=("execution_eligible", "safety_valid"),
        default="execution_eligible",
        help=(
            "legacy controller-coupled replay acceptance (default), or replay "
            "every full-H verifier-positive selected-B query independently of "
            "progress/target gates; safety_valid requires --archive-rule all_queries"
        ),
    )
    parser.add_argument("--execution-rule",
                        choices=("max_margin", "min_cost", "uniform_positive",
                                 "softmin_cost", "max_uncertainty",
                                 "uncertainty_cost", "max_step_margin"),
                        default="max_margin")
    parser.add_argument("--execution-ess-target", type=float, default=0.25)
    parser.add_argument("--execution-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--acquisition-feature",
                        choices=("learned_phi", "task_angular"),
                        default="learned_phi")
    parser.add_argument("--coverage-replay",
                        choices=("none", "circular_voronoi"),
                        default="none")
    parser.add_argument("--replay-augmentation",
                        choices=("none", "task_d4"), default="none")
    parser.add_argument("--execution-z-bias-mode", choices=("none", "favor_above"),
                        default="none",
                        help=("diagnostic execution-ranking term only; favor_above mirrors the "
                              "demo z exponential and penalizes plans below its plane"))
    parser.add_argument("--tight-corridor", action="store_true",
                        help="require only the dense first executed segment to satisfy "
                             "x in [0,3] and z in [1.5,2.5]")
    parser.add_argument(
        "--verifier-mode",
        choices=("full_polytope", "single_sphere_affine"),
        default="full_polytope",
        help=(
            "bounded GREEN verifier with 80 artificial sensing faces, or the "
            "same fitted real-sphere affine barrier without artificial faces"
        ),
    )
    parser.add_argument(
        "--verifier-solver",
        choices=("analytic", "cvxpy"),
        default="analytic",
        help=(
            "variable-face solver backend; analytic is the unchanged legacy "
            "active-set implementation and default rollback path, while cvxpy "
            "uses the equivalent explicit SOCP with CLARABEL"
        ),
    )
    parser.add_argument("--event-log", choices=("none", "full"), default="full",
                        help="omit all-K visualization events during metric-only sweeps")
    parser.add_argument(
        "--paired-noised-representation", action="store_true",
        help=("use phi_s((1-s)x_0+sU,c) with each sampled plan's paired Gaussian "
              "flow base for RBF calibration, acquisition, and replay"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output_was_unsafe = (
        args.output.exists()
        and (
            not args.output.is_dir()
            or any(args.output.iterdir())
        )
    )

    pretrain = json.loads((args.pretrain_dir / "pretrain_manifest.json").read_text())
    lab_profile = _is_lab_pretrain(pretrain)
    if lab_profile:
        try:
            task_config = (
                load_config(args.lab_task_config)
                if args.lab_task_config is not None
                else _lab_task_config(args.pretrain_dir, pretrain)
            )
            randomization = task_config.raw.get("scene_randomization", {})
            clutter_profile = _lab_clutter_profile(task_config)
            if clutter_profile:
                policy = load_lab_clutter_expansion_policy(
                    args.pretrain_dir / "pretrained.pt"
                )
                task = LabClutterSphereExpansionTask(
                    task_config,
                    context_schema=policy.context_schema,
                    execution_z_bias_mode=args.execution_z_bias_mode,
                    tight_corridor=args.tight_corridor,
                    verifier_mode=args.verifier_mode,
                    verifier_solver=args.verifier_solver,
                )
            else:
                policy = load_lab_expansion_policy(
                    args.pretrain_dir / "pretrained.pt"
                )
                task = LabFlowExpansionTask(
                    task_config,
                    context_schema=policy.context_schema,
                    execution_z_bias_mode=args.execution_z_bias_mode,
                    tight_corridor=args.tight_corridor,
                    verifier_mode=args.verifier_mode,
                    verifier_solver=args.verifier_solver,
                )
        except Exception as error:
            _record_lab_setup_failure(
                args.output,
                output_was_unsafe=output_was_unsafe,
                stage="lab_task_setup",
                error=error,
            )
            raise
        context_contract = policy.context_schema
    else:
        clutter_profile = False
        policy = load_policy(args.pretrain_dir / "pretrained.pt")
        task_config = load_config(args.pretrain_dir / "demo_config.json")
        context_contract = pretrain.get("context_contract", "legacy10")
        task_type = (
            ThetaBallFlowTask
            if context_contract == "local_theta12" else BallFlowTask
        )
        task = task_type(
            task_config,
            execution_z_bias_mode=args.execution_z_bias_mode,
            tight_corridor=args.tight_corridor,
            target_region=args.target_region,
            verifier_mode=args.verifier_mode,
            verifier_solver=args.verifier_solver,
        )
    if args.acquisition_feature == "task_angular":
        if lab_profile:
            error = ValueError(
                "lab profile currently supports learned_phi acquisition only"
            )
            _record_lab_setup_failure(
                args.output,
                output_was_unsafe=output_was_unsafe,
                stage="lab_rbf_calibration",
                error=error,
            )
            raise error
        calibration = _angular_pretrained_calibration(
            policy, task, args.pretrain_dir, args.seed, args.flow_base_std,
        )
        if args.beta is None:
            raise ValueError("task_angular acquisition requires an explicit --beta")
    else:
        try:
            calibration = _learned_phi_calibration(
                policy,
                args.pretrain_dir,
                pretrain,
                task_config,
                args.seed,
                lab_profile=lab_profile,
                flow_base_std=args.flow_base_std,
                paired_noised_representation=(
                    args.paired_noised_representation
                ),
            )
        except Exception as error:
            if lab_profile:
                _record_lab_setup_failure(
                    args.output,
                    output_was_unsafe=output_was_unsafe,
                    stage="lab_rbf_calibration",
                    error=error,
                )
            raise
    beta = float(
        pretrain.get("beta", 5.0e-4)
        if args.beta is None else args.beta
    )

    config = ExpansionConfig(
        rounds=args.rounds, gammas=tuple(task_config.data.gammas),
        parallel_episodes=args.parallel_episodes,
        verifier_workers=args.verifier_workers,
        max_retry_batches=args.max_retry_batches,
        max_steps=task_config.taskspace.max_steps, K=args.K, B=args.B,
        batch_size=args.batch_size,
        inner_steps=args.optimizer_steps_per_round, microbatch_repeats=args.inner_steps,
        learning_rate=args.learning_rate,
        first_layer_lr_scale=args.first_layer_lr_scale,
        replay_rounds=args.replay_rounds,
        gp_buffer_cap=args.gp_buffer_cap, gp_noise=1.0e-2,
        rbf_lengthscale=args.rbf_lengthscale,
        beta=beta, adaptive_beta=args.adaptive_beta, ess_target=args.ess_target,
        negative_alpha=args.negative_alpha,
        replay_top_fraction=args.replay_top_fraction,
        replay_selector=args.replay_selector,
        replay_context_quota=args.replay_context_quota,
        replay_action_weight=args.replay_action_weight,
        replay_cluster_count=args.replay_cluster_count,
        flow_base_std=args.flow_base_std,
        candidate_perturb_std=args.candidate_perturb_std,
        candidate_perturb_scope=args.candidate_perturb_scope,
        execution_rule=args.execution_rule,
        execution_ess_target=args.execution_ess_target,
        execution_uncertainty_weight=args.execution_uncertainty_weight,
        archive_rule=args.archive_rule,
        successful_trajectory_selector=args.successful_trajectory_selector,
        successful_trajectories_per_gamma=(
            args.successful_trajectories_per_gamma
        ),
        replay_acceptance=args.replay_acceptance,
        paired_noised_representation=args.paired_noised_representation,
        acquisition_feature=args.acquisition_feature,
        coverage_replay=args.coverage_replay,
        replay_augmentation=args.replay_augmentation,
        target_gate_start_round=args.target_gate_start_round,
        gp_reference_mode=args.gp_reference_mode,
        gp_sliding_row_selector=args.gp_sliding_row_selector,
        gp_exact_max_rows_per_gamma=args.gp_exact_max_rows_per_gamma,
        seed=args.seed,
    )

    events = []
    def callback(event):
        event_context = event["context"].numpy()
        if lab_profile:
            # The 6,919-D visual volume is reproducible from robot state and
            # scene geometry; storing it at every event would add many GB to a
            # 50-round log without helping the mechanism visualization.
            event_context = np.concatenate([
                event_context[:7],
                event_context[
                    -(
                        18 if clutter_profile else 6
                    ):
                ],
            ]).astype(np.float32)
        events.append({
            "round": event["round"], "step": event["step"], "gamma": event["gamma"],
            "episode": event["episode"], "context_id": event["context_id"],
            "retry_batch": event["retry_batch"], "replica": event["replica"],
            "robot": np.asarray(event["state_before"]["x"], np.float32),
            "robot_after": np.asarray(event["state_after"]["x"], np.float32),
            "context": event_context,
            "context_compacted": bool(lab_profile),
            "base_candidates": event["base_candidates"].numpy().astype(np.float32),
            "flow_bases": (
                event["flow_bases"].numpy().astype(np.float32)
                if event.get("flow_bases") is not None else None
            ),
            "candidates": event["candidates"].numpy().astype(np.float32),
            "sigma_K": event["sigma_K"].numpy().astype(np.float32),
            "selected": list(event["selected"]),
            "selected_sigma": list(event["selected_sigma"]),
            "verification": event["verification"],
            "chosen_local": event["chosen_local"],
            "archived_negative_local": event["archived_negative_local"],
            "status": event["status"],
            "nvp_reason": event.get("nvp_reason"),
            "target_gate_active": bool(event.get("target_gate_active", False)),
        })

    started = time.perf_counter()
    try:
        manifest = run_safe_expansion(
            policy, task, args.output, config=config,
            calibration_features=calibration,
            event_callback=(callback if args.event_log == "full" else None),
        )
    except Exception as error:
        if not output_was_unsafe:
            args.output.mkdir(parents=True, exist_ok=True)
            if args.event_log == "full":
                torch.save(events, args.output / "FAILED_events.pt")
            (args.output / "FAILED.json").write_text(json.dumps({
                "status": "EXPANSION_FAILED_CLOSED",
                "error_type": type(error).__name__,
                "error": str(error),
                "event_count": len(events),
                "elapsed_s": time.perf_counter() - started,
            }, indent=2) + "\n")
        raise
    # Make the task-specific cost contract machine-checkable.  The demo configuration retains
    # a nonzero below-equator z bias, but BallFlowTask.native_cost intentionally never applies it
    # during self-expansion execution ranking.
    manifest["ball_execution_cost"] = {
        "variant": ("native_state_control_smoothness_terminal_without_demo_z_bias"
                    if args.execution_z_bias_mode == "none"
                    else "native_plus_diagnostic_favor_above_exponential"),
        "demo_z_bias_weight": float(task_config.safemppi.z_bias_weight),
        "execution_z_bias_mode": args.execution_z_bias_mode,
        "expansion_z_bias_weight": (
            float(task_config.safemppi.z_bias_weight)
            if args.execution_z_bias_mode == "favor_above" else 0.0
        ),
        "expansion_z_bias_plane": float(task_config.safemppi.z_bias_plane),
        "expansion_z_bias_temperature": float(task_config.safemppi.z_bias_temperature),
    }
    manifest["ball_conditioning"] = {
        "context_contract": context_contract,
        "context_dim": int(policy.context_dim),
        "plan_coordinates": (
            "start_goal_local_frame"
            if context_contract == "local_theta12" else "world_frame"
        ),
        "theta_prior": pretrain.get("theta_prior"),
        "first_layer_lr_scale": float(args.first_layer_lr_scale),
    }
    manifest["ball_verifier_bounds"] = {
        "x_min": float(task.env.bounds[0, 0]),
        "x_max": float(task.env.bounds[0, 1]),
        "contract": ("only the dense first executed segment must remain inside taskspace; "
                     "the unexecuted H-1 tail is not taskspace-gated"),
    }
    manifest["ball_full_h_verifier"] = {
        "variant": args.verifier_mode,
        "face_solver": args.verifier_solver,
        "face_solver_contract": (
            "legacy_exact_active_set"
            if args.verifier_solver == "analytic"
            else "radial_shortcut_then_cvxpy_clarabel_with_analytic_fallback"
        ),
        "legacy_rollback_cli": "--verifier-solver analytic",
        "source_contract": (
            "the dimension-independent counterpart of the cloned 2-D "
            "ieee_compact_polytope_verifier_package max-margin face block"
        ),
        "beta_t": "1-(1-gamma)^t",
        "artificial_faces": args.verifier_mode == "full_polytope",
        "artificial_face_count": (
            80 if args.verifier_mode == "full_polytope" else 0
        ),
        "artificial_faces_note": (
            "fitted artificial-sphere faces use the nominal icosphere directions "
            "and bound the GREEN polytope at the effective sensing radius"
        ),
        "one_step_nominal": "diagnostic only; never used for execution eligibility",
        "progress": {
            "definition": "min_h (q_{h+1,x}-q_{h,x})",
            "execution_eligible": "strictly positive at every plan knot",
            "safety_label": (
                "underlying Verification.valid is unchanged and denotes only "
                "the full-H safety test"
            ),
            "D_plus_admission": (
                "only reverified valid positive-progress executed <=H suffixes "
                "from a committed terminal SUCCESS; short terminal suffixes are "
                "zero-padded only for fixed-shape phi/CFM and use masked CFM loss"
                if args.archive_rule == "successful_executed_windows"
                else "full-H safety positives"
            ),
        },
    }
    manifest["ball_verifier_corridor"] = {
        "enabled": bool(args.tight_corridor),
        "x": [0.0, 3.0],
        "z": [1.5, 2.5],
        "contract": ("corridor exit along the dense first executed segment makes a query "
                     "invalid; the unexecuted H-1 tail is not corridor-gated; if selected B "
                     "contains no valid forward-monotone candidate, "
                     "terminate NVP"),
    }
    target_geometry = {
        "above_wedge": {
            "boundary_planes": ["z-2=+y", "z-2=-y"],
            "accepted_region": "z-2 >= |y|",
        },
        "above_halfspace": {
            "boundary_planes": ["z=2"],
            "accepted_region": "z >= 2",
        },
    }[args.target_region]
    manifest["ball_target_region_gate"] = {
        "enabled": args.target_gate_start_round is not None,
        "start_round": args.target_gate_start_round,
        "target_region": args.target_region,
        **target_geometry,
        "checked_state": "q1 only (the next executed state); never the unexecuted H-tail",
        "safety_label": (
            "underlying GREEN/Verification.valid remains the full-H safety "
            "predicate and is independent of this target gate"
        ),
        "D_plus_admission": (
            "only reverified valid positive-progress executed <=H suffixes from "
            "a committed terminal SUCCESS"
            if args.archive_rule == "successful_executed_windows"
            else "determined by replay_acceptance below"
        ),
        "execution": (
            "from start_round, require full-H safe AND per-knot positive x progress "
            f"AND q1 in {args.target_region}"
        ),
        "replay_acceptance": (
            "successful-window archive ignores per-query gathering admission; "
            "only committed reverified executed <=H suffixes enter replay"
            if args.archive_rule == "successful_executed_windows"
            else (
                "all full-H safe selected-B queries, independent of progress and target"
                if args.replay_acceptance == "safety_valid"
                else (
                    "legacy controller-coupled rule; from start_round require full-H "
                    f"safe AND per-knot positive x progress AND q1 in {args.target_region}"
                )
            )
        ),
    }
    if lab_profile:
        for key in tuple(manifest):
            if key.startswith("ball_"):
                del manifest[key]
        manifest["task_profile"] = (
            "minhyuk_lab_random_three_sphere_visual_expansion"
            if clutter_profile else "minhyuk_lab_ball_visual_expansion"
        )
        manifest["lab_conditioning"] = {
            "context_schema": context_contract,
            "policy_context_dim": int(policy.policy_context_dim),
            "verifier_only_context_suffix": (
                [
                    "previous_applied_acceleration_3d",
                    "previous_raw_acceleration_3d",
                    "three_spheres_flattened_12d",
                ]
                if clutter_profile else [
                    "previous_applied_acceleration_3d",
                    "previous_raw_acceleration_3d",
                ]
            ),
            "visual_encoder_and_first_flow_layer_lr_scale": float(
                args.first_layer_lr_scale
            ),
            "event_context": (
                "7-D low state plus verifier-only dynamics/scene suffix; "
                "visual grid omitted because it is reproducible from robot "
                "state and scene"
            ),
        }
        manifest["lab_reference_dynamics"] = {
            "policy_output": "pre_smoothing_raw_acceleration_command",
            "governor": "ReferenceGovernor applied exactly once",
            "deployment_accel_smooth": float(
                task_config.safemppi.deployment_accel_smooth
            ),
            "max_speed": float(task_config.safemppi.max_speed),
            "max_vertical_speed": float(
                task_config.safemppi.max_vertical_speed
            ),
        }
        manifest["lab_execution_cost"] = {
            "variant": (
                "configured running/terminal/control/smoothness/soft-clearance/"
                "progress/taskspace cost on governed states"
            ),
            "excluded_term": "demonstration-only below-plane z bias",
            "execution_rule": args.execution_rule,
        }
        manifest["lab_verifier"] = {
            "variant": args.verifier_mode,
            "face_solver": args.verifier_solver,
            "artificial_face_count": (
                80 if args.verifier_mode == "full_polytope" else 0
            ),
            "taskspace_bounds": task.env.bounds.tolist(),
            "tight_corridor_flag": bool(args.tight_corridor),
            "tight_corridor_contract": (
                "lab configured taskspace/geofence only; canonical ball "
                "[0,3] x [1.5,2.5] bounds are not used"
            ),
            "progress": (
                "strictly positive at every plan knot along the fixed "
                "start-goal unit axis"
            ),
            "unexecuted_tail_taskspace_gate": False,
            "full_h_collision_and_green": True,
        }
        manifest["lab_coverage_objective"] = {
            "successful_trajectory_selector": (
                args.successful_trajectory_selector
            ),
            "above_plane_z": (
                float(task.env.spheres[0, 2])
                if len(task.env.spheres) == 1 else None
            ),
            "selector_changes_safety_label": False,
        }
        if clutter_profile:
            manifest["lab_scene_randomization"] = dict(randomization)
            manifest["lab_scene_ledger"] = list(task.scene_ledger)
            (args.output / "task_config_resolved.json").write_text(
                json.dumps(task_config.raw, indent=2) + "\n"
            )
    manifest["event_log"] = args.event_log
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if args.event_log == "full":
        torch.save(events, args.output / "events.pt")
    print(f"[expansion] rounds={args.rounds} events={len(events)} "
          f"D={manifest['D']} D+={manifest['D_plus']} "
          f"D_accept={manifest['D_replay_accepted']} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    for row in manifest["rounds"]:
        print(f"  round {row['round']:2d}: verified "
              f"{row['verifier_positives']:3d}/{row['verifier_queries']:3d} -> stored "
              f"{row['positives']:3d}/{row['queries']:3d} "
              f"success {row['success']}/{row['attempted_episode_count']} "
              f"NVP {row['NVP']:3d} "
              f"accepted {row['replay_accepted_positives']:3d} "
              f"replay {row['replay_positives']}/{row['replay_positive_total']} "
              f"GP {row['gp_buffer']:4d} "
              f"(anchor/adapt/evidence {row['gp_anchor_count']}/"
              f"{row['gp_adaptive_count']}/{row['gp_evidence_count']}) "
              f"mESS {row['marginal_ESS_over_K']:.3f} "
              f"sigma {row['sigma_pool_mean']:.3f}->{row['sigma_selected_mean']:.3f} "
              f"uplift {row['uncertainty_uplift']:+.3f} "
              f"steps {row['steps']} time {row['round_total_s']:.1f}s "
              f"loss {row['positive_loss']}", flush=True)


if __name__ == "__main__":
    main()
