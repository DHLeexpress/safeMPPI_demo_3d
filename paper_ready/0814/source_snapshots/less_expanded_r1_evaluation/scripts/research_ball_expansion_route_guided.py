"""Run exact-quota expansion with route-persistent prototype guidance.

This independent arm keeps the Reserve-G-scale optimizer and safety verifier,
but guided episodes match complete candidate action windows to progress-aligned
successful trajectory suffixes instead of repeatedly swapping by first-action
quadrant direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
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
    LAB_CLUTTER_GOVERNOR_DIM,
    LAB_CLUTTER_SCENE_SCHEMA,
    LabClutterSphereExpansionTask,
    load_lab_clutter_expansion_policy,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_reference_flow_task import lab_reference_demo_windows
from safe_mppi.helios_remote import (
    add_helios_arguments,
    run_expansion_on_helios,
)
from safe_mppi.lab_visual_flow import (
    LAB_HP100_EXACT_MEMORY_PACKED_DIM,
    LAB_HP100_EXACT_MEMORY_SCHEMA,
    LAB_HP100_HISTORY_SCHEMA,
    LAB_HP100_PACKED_DIM,
    LAB_HP100_SCHEMA,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_SCHEMA,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
)
from safe_mppi.path_focused_clutter import PATH_FOCUSED_DISTRIBUTIONS
from safe_mppi.progress import format_round_summary, show_progress
from safe_mppi.route_prototype_guidance import (
    RoutePrototypeGuide,
    blend_route_proposals,
)

ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_COMMAND = "RECIPE.sh"
LAB_HISTORY_CONTEXT_SCHEMAS = frozenset({
    LAB_HP100_HISTORY_SCHEMA,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
})
LAB_CONTEXT_BASE_PACKED_DIMS = {
    LAB_HP100_EXACT_MEMORY_SCHEMA: LAB_HP100_EXACT_MEMORY_PACKED_DIM,
    LAB_HP100_HISTORY_SCHEMA: LAB_HP100_PACKED_DIM,
    LAB_HP100_SCHEMA: LAB_HP100_PACKED_DIM,
    LAB_VISUAL_SCHEMA: LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA: LAB_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_SCHEMA: LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA: LAB_RADIAL_VISUAL_PACKED_DIM,
}


def _round_schedule_fraction(round_index: int, total_rounds: int) -> float:
    """Map real expansion rounds to [0, 1], including both endpoints."""
    if total_rounds <= 1:
        return 0.0
    return float(np.clip(
        (int(round_index) - 1) / (int(total_rounds) - 1), 0.0, 1.0,
    ))


def _write_reproduction_command(
    output: Path,
    argv: list[str] | tuple[str, ...] | None = None,
    *,
    cwd: Path | None = None,
) -> Path:
    """Write the shell-expanded invocation as a copy/paste-ready script."""
    tokens = list(
        getattr(sys, "orig_argv", sys.argv) if argv is None else argv
    )
    if not tokens:
        raise ValueError("cannot record an empty process invocation")
    output.mkdir(parents=True, exist_ok=True)
    invocation = " \\\n  ".join(shlex.quote(token) for token in tokens)
    command_path = output / REPRODUCTION_COMMAND
    command_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Reconstructed from process argv. Shell variables and the original\n"
        "# line wrapping are already expanded before Python starts.\n"
        f"cd {shlex.quote(str(Path.cwd() if cwd is None else cwd))}\n\n"
        f"{invocation}\n"
    )
    command_path.chmod(0o755)
    return command_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_flow_nfe(policy: torch.nn.Module, flow_nfe: int | None) -> None:
    """Override rollout integration steps on adapters and their core flow."""
    if flow_nfe is None:
        return
    core = getattr(policy, "policy", policy)
    flow = getattr(core, "flow", None)
    if flow is None or not hasattr(flow, "nfe"):
        raise TypeError("--flow-nfe requires a policy with a flow.nfe field")
    flow.nfe = int(flow_nfe)
    if hasattr(core, "nfe"):
        core.nfe = int(flow_nfe)
    if hasattr(policy, "nfe"):
        policy.nfe = int(flow_nfe)


def _resolved_success_episode_keys(
    round_row: dict,
) -> tuple[set[tuple[int, float, int]], set[tuple[int, float, int]]]:
    """Return all resolved SUCCESS keys and their authoritative committed subset."""
    round_i = int(round_row["round"])
    details = round_row["successful_executed_commit_by_gamma"]
    success_keys = {
        (round_i, float(gamma_key), int(episode_id))
        for gamma_key, detail in details.items()
        for episode_id in detail["success_episode_above_fractions"]
    }
    declared_successes = sum(
        int(detail["success_episode_count"])
        for detail in details.values()
    )
    if len(success_keys) != declared_successes:
        raise RuntimeError(
            "resolved SUCCESS event keys do not match the declared successful "
            "episode count"
        )
    committed_keys = {
        (round_i, float(gamma_key), int(episode_id))
        for gamma_key, detail in details.items()
        for episode_id in detail["committed_episode_ids"]
    }
    declared = sum(
        int(detail["committed_trajectory_count"])
        for detail in details.values()
    )
    if len(committed_keys) != declared:
        raise RuntimeError(
            "committed-success event keys do not match the declared plural "
            "trajectory count"
        )
    if not committed_keys.issubset(success_keys):
        raise RuntimeError(
            "authoritative committed episodes are absent from the resolved "
            "terminal-SUCCESS set"
        )
    return success_keys, committed_keys


def _retain_committed_round_events(
    pending_events: list[dict],
    round_row: dict,
) -> list[dict]:
    """Keep all resolved SUCCESS traces needed to audit the committed subset."""
    round_i = int(round_row["round"])
    if any(int(event["round"]) != round_i for event in pending_events):
        raise RuntimeError(
            "committed-success event buffer crossed a round boundary before "
            "authoritative resolution"
        )
    success_keys, _ = _resolved_success_episode_keys(round_row)
    grouped: dict[tuple[int, float, int], list[dict]] = {}
    for event in pending_events:
        key = (
            int(event["round"]),
            float(event["gamma"]),
            int(event["episode"]),
        )
        if key in success_keys:
            grouped.setdefault(key, []).append(event)
    missing = success_keys.difference(grouped)
    if missing:
        raise RuntimeError(
            "resolved terminal-SUCCESS episodes have no event trace: "
            f"{sorted(missing)}"
        )
    for key, rows in grouped.items():
        rows.sort(key=lambda event: int(event["step"]))
        steps = [int(event["step"]) for event in rows]
        if steps != list(range(len(rows))):
            raise RuntimeError(
                f"committed SUCCESS episode {key} has a non-contiguous event trace"
            )
        if any(event.get("status") is not None for event in rows[:-1]):
            raise RuntimeError(
                f"committed SUCCESS episode {key} continues after a terminal event"
            )
        if rows[-1].get("status") != "SUCCESS":
            raise RuntimeError(
                f"resolved successful episode {key} does not terminate SUCCESS"
            )
    return [
        event for event in pending_events
        if (
            int(event["round"]),
            float(event["gamma"]),
            int(event["episode"]),
        ) in success_keys
    ]


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
    if randomization.get("obstacle_family") != "spheres":
        raise ValueError(
            "--lab-task-config with enabled scene_randomization requires "
            "exactly three spheres or path-focused variable-count spheres"
        )
    if randomization.get("distribution") in PATH_FOCUSED_DISTRIBUTIONS:
        sphere_scene_spec_from_config(task_config)
        return True
    if int(randomization.get("count", -1)) != 3:
        raise ValueError(
            "--lab-task-config with enabled scene_randomization requires "
            "exactly three spheres or path-focused variable-count spheres"
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
    _write_reproduction_command(output)


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
    context_artifact = pretrain_manifest.get("rbf_calibration", {}).get(
        "context_artifact"
    )
    cached_context_path = (
        pretrain_dir / str(context_artifact)
        if context_artifact is not None else None
    )
    if cached_context_path is not None and cached_context_path.is_file():
        expected_sha256 = pretrain_manifest["rbf_calibration"].get(
            "context_artifact_sha256"
        )
        if (
            expected_sha256 is not None
            and _sha256_file(cached_context_path) != expected_sha256
        ):
            raise ValueError("lab calibration context artifact hash mismatch")
        cached_contexts = torch.load(
            cached_context_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_context_dim = int(
            getattr(policy, "policy_context_dim", policy.context_dim)
        )
        if (
            cached_contexts.ndim != 2
            or cached_contexts.shape[1] != expected_context_dim
            or not bool(torch.isfinite(cached_contexts).all())
        ):
            raise ValueError(
                "lab calibration context artifact violates its policy contract"
            )
        contexts_np = cached_contexts.numpy()
        metadata_gammas = np.asarray(contexts_np[:, 6], dtype=np.float64)
        demo_gammas = np.unique(metadata_gammas)
    else:
        source = _lab_source_demo_dir(pretrain_dir, pretrain_manifest)
        contexts_np, _, metadata, demo_config = lab_reference_demo_windows(
            source,
            context_schema=policy.context_schema,
        )
        metadata_gammas = np.asarray(
            [row["gamma"] for row in metadata], dtype=np.float64,
        )
        demo_gammas = np.asarray(demo_config.data.gammas, dtype=np.float64)
    if len(contexts_np) < count:
        raise ValueError("lab calibration requires at least 50 demo contexts")
    task_gammas = np.asarray(task_config.data.gammas, dtype=np.float64)
    if (
        task_gammas.shape != demo_gammas.shape
        or not np.allclose(
            task_gammas, demo_gammas, rtol=0.0, atol=1.0e-7,
        )
    ):
        raise ValueError("lab calibration demo gammas do not match task config")
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
    add_helios_arguments(parser)
    parser.add_argument("--pretrain-dir", type=Path, default=ROOT / "outputs" / "ball_flow")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "ball_flow" / "expansion")
    parser.add_argument(
        "--resume-from", type=Path,
        help=(
            "continue this same output directory from its atomic last-committed "
            "round state, restoring Adam momentum, LR step, RNG, replay, and GP "
            "evidence; the interrupted uncommitted round is sampled again"
        ),
    )
    parser.add_argument(
        "--lab-task-config",
        type=Path,
        help=(
            "optional lab expansion task override; use the randomized-sphere "
            "config to transfer a cylinder-pretrained visual policy OOD"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="policy/CFM device, e.g. cpu, cuda, or cuda:0",
    )
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument(
        "--learning-rate-final", type=float, default=1.0e-6,
        help=(
            "optional final learning rate for optimizer-step cosine decay; "
            "requires --learning-rate-decay-steps and remains fixed after it"
        ),
    )
    parser.add_argument(
        "--learning-rate-decay-steps", type=int, default=50_000,
        help=(
            "number of Adam steps over which to cosine-decay from "
            "--learning-rate to --learning-rate-final"
        ),
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=5.0,
        help="optional global gradient-norm clipping before every Adam step",
    )
    parser.add_argument(
        "--round-learning-rate-warmup-power", type=float, default=2.0,
        help=("outer-round LR scale (round/rounds)^power, clamped at the "
              "final LR; 0 disables the stability warmup"),
    )
    parser.add_argument(
        "--first-layer-lr-scale", type=float, default=1.0,
        help=(
            "expansion-only multiplier for the first flow Linear layer and, "
            "for unfrozen lab visual policies, the visual encoder; the "
            "representation layer and head retain --learning-rate"
        ),
    )
    parser.add_argument(
        "--freeze-visual-encoder-during-expansion",
        action="store_true",
        help=(
            "lab visual policies only: disable visual-encoder gradients and "
            "exclude its parameters from Adam while keeping the flow trunk, "
            "representation layer, head, and current-phi acquisition trainable"
        ),
    )
    parser.add_argument(
        "--head-only-expansion",
        action="store_true",
        help=(
            "freeze the complete policy except the final flow output head; "
            "for the T128 checkpoint this updates only flow.head (32->30)"
        ),
    )
    parser.add_argument(
        "--trainable-trunk-layers",
        type=int,
        default=3,
        help=(
            "partial-depth adaptation: train the final flow output head plus "
            "the last N parameterised trunk blocks and freeze everything else; "
            "omitted keeps the configured optimizer scope, and N may not exceed "
            "the checkpoint trunk depth"
        ),
    )
    parser.add_argument(
        "--train-gru-during-expansion",
        action="store_true",
        help=(
            "explicitly update a visual-history GRU during expansion; by "
            "default a GRU checkpoint freezes only its history encoder, and "
            "this flag is rejected for checkpoints without a GRU"
        ),
    )
    parser.add_argument("--beta", type=float, default=0.1,
                        help="fixed beta; default uses the one-time pretrained calibration")
    parser.add_argument("--adaptive-beta", action="store_true",
                        help="recalibrate beta after each round to the requested ESS target")
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument("--parallel-episodes", type=int, default=8)
    parser.add_argument(
        "--verifier-workers",
        type=int,
        default=32,
        help=(
            "persistent CPU worker processes used to verify independent active "
            "episode contexts; 1 preserves serial verification"
        ),
    )
    parser.add_argument(
        "--retry-exhaustion-policy",
        choices=("abort", "resample_scene"),
        default="resample_scene",
        help=(
            "RESEARCH: on retry exhaustion, keep drawing fresh episode "
            "batches (resample_scene) instead of aborting the round"
        ),
    )
    parser.add_argument(
        "--retry-resample-batch-cap", type=int, default=512,
        help="RESEARCH: total retry-batch cap under resample_scene",
    )
    parser.add_argument(
        "--max-retry-batches", type=int, default=32,
        help=(
            "whole-episode batches attempted per gamma by "
            "successful_executed_windows; never retries an individual step"
        ),
    )
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=1,
                        help="optimizer repeats on each microbatch during one exact replay pass")
    parser.add_argument(
        "--replay-passes-per-round", type=int, default=1,
        help=(
            "freshly reshuffled exact replay passes per round; unlike "
            "--inner-steps, this moves through the full replay pool before "
            "revisiting it"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--optimizer-steps-per-round", type=int, default=None,
        help=("optional cap on distinct replay microbatches per round; unlike "
              "--inner-steps, this controls total sampled rows before repeats; "
              "cannot be combined with --replay-passes-per-round > 1"),
    )
    parser.add_argument(
        "--optimizer-steps-total", type=int, default=50_000,
        help=("exact optimizer-step budget distributed across all rounds; "
              "the final cosine endpoint is reached on the final round"),
    )
    parser.add_argument(
        "--optimizer-step-allocation",
        choices=("uniform", "linear_round", "quadratic_round"),
        default="quadratic_round",
        help=("uniform steps, steps proportional to cumulative pool size, or "
              "quadratic late-weighting to minimize early small-pool distortion"),
    )
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument(
        "--retry-B", type=int, default=16,
        help=(
            "queries verified per context after retry batch 0; defaults to all "
            "K=16 candidates so missing-mode guidance is not restricted to "
            "the initial acquisition subset"
        ),
    )
    parser.add_argument(
        "--retry-verify-all-fast-path", action="store_true",
        help=(
            "when retry-B equals K, verify every retry candidate without "
            "recomputing GP acquisition; requires fixed beta, uniform replay, "
            "and min_cost execution"
        ),
    )
    parser.add_argument("--replay-top-fraction", type=float, default=1.0)
    parser.add_argument(
        "--replay-selector",
        choices=("sigma_top", "uniform", "context_kcenter", "cluster_balanced"),
        default="uniform",
    )
    parser.add_argument(
        "--replay-scope",
        choices=("sliding", "cumulative"),
        default="cumulative",
        help="use every committed round or only --replay-rounds recent rounds",
    )
    parser.add_argument(
        "--replay-batch-sampler",
        choices=("row_permutation", "mode_gamma_stratified"),
        default="mode_gamma_stratified",
        help=("Reserve G sampler: equal mode x gamma batch mass, then uniform "
              "trajectory lineage and uniform window"),
    )
    parser.add_argument(
        "--flow-nfe", type=int, default=12,
        help="ODE Euler steps used for expansion rollout sampling",
    )
    parser.add_argument(
        "--batched-rollout-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("integrate every active episode's K candidates in one GPU ODE "
              "batch; use --no-batched-rollout-sampling for the legacy path"),
    )
    parser.add_argument("--replay-context-quota", type=int, default=4)
    parser.add_argument("--replay-action-weight", type=float, default=0.5)
    parser.add_argument("--replay-cluster-count", type=int, default=32)
    parser.add_argument(
        "--flow-base-std",
        type=float,
        default=1.1,
        help=(
            "expansion-only standard deviation of the initial flow latent "
            "x0~N(0,std^2 I); 1 preserves canonical unit-normal sampling. "
            "This is distinct from post-flow --candidate-perturb-std"
        ),
    )
    parser.add_argument(
        "--flow-base-std-schedule",
        choices=("none", "linear", "cosine", "adaptive"),
        default="cosine",
        help=(
            "RESEARCH: per-round schedule for --flow-base-std. linear/cosine "
            "anneal from --flow-base-std to --flow-base-std-final across the "
            "rounds; adaptive sets round r+1 std to "
            "final + (init-final) * deficit^gain where deficit is 1 - "
            "normalized entropy of the measured first-action angular "
            "distribution at the start state (wide while collapsed, narrow "
            "once symmetric)"
        ),
    )
    parser.add_argument(
        "--flow-base-std-final", type=float, default=1.0,
        help=(
            "RESEARCH: terminal std for the schedule (defaults to "
            "--flow-base-std, making every schedule a constant)"
        ),
    )
    parser.add_argument(
        "--adaptive-std-gain", type=float, default=1.0,
        help="RESEARCH: exponent on the coverage deficit for adaptive std",
    )
    parser.add_argument(
        "--beta-coverage-gain", type=float, default=0.0,
        help=(
            "RESEARCH: scale acquisition beta by 1 + gain * deficit using the "
            "same first-action coverage deficit (0 disables)"
        ),
    )
    parser.add_argument(
        "--first-action-samples", type=int, default=256,
        help=(
            "RESEARCH: flow samples drawn at the start state each round to "
            "measure the first-action angular distribution"
        ),
    )
    parser.add_argument(
        "--fa-alloc",
        choices=("none", "step0", "steps"),
        default="steps",
        help=(
            "RESEARCH: allocate parallel episodes across the 4 route "
            "quadrants at execution time. Each episode slot (replica mod 4) "
            "owns a quadrant; among the ALREADY-VERIFIED candidates the "
            "episode executes the one whose first action best aligns with "
            "its quadrant. step0 applies only to the first step (the mode "
            "decision); steps applies for --fa-alloc-steps steps with a "
            "min_cost band guard. none (default) is untouched min_cost"
        ),
    )
    parser.add_argument(
        "--fa-alloc-steps", type=int, default=30,
        help="RESEARCH: how many initial steps keep quadrant guidance in "
             "steps mode",
    )
    parser.add_argument(
        "--fa-alloc-band", type=float, default=0.5,
        help=(
            "RESEARCH: at steps>0, only candidates whose min_cost ranking key "
            "is within best + band*(worst-best) may be re-picked (0=never "
            "override, 1=any verified candidate)"
        ),
    )
    parser.add_argument(
        "--fa-alloc-step0-band", type=float, default=1.0,
        help="RESEARCH: same band at step 0 (default 1.0: any verified "
             "candidate may carry the mode decision)",
    )
    parser.add_argument(
        "--fa-alloc-map", type=str, default="0,0,0,0,1,1,1,1",
        help=(
            "RESEARCH: comma list of quadrant ids (0=below,1=above,2=left,"
            "3=right) assigned to episode slots replica 0..N-1; slots beyond "
            "the list wrap. Default round-robin 0,1,2,3. Skewing the map "
            "gives weak modes more episode slots so their successes enter "
            "the random_success commit pool more often"
        ),
    )
    parser.add_argument(
        "--fa-alloc-retry-map",
        choices=("fixed", "missing_quota"),
        default="missing_quota",
        help=(
            "RESEARCH: keep --fa-alloc-map fixed, or after each logged retry "
            "milestone redistribute the same guided cohort across the route "
            "modes still missing for that gamma; unguided episodes are "
            "unchanged"
        ),
    )
    parser.add_argument(
        "--fa-alloc-retry-band",
        type=float,
        default=1.0,
        help=(
            "RESEARCH: optional post-step-0 min_cost band used only while "
            "--fa-alloc-retry-map=missing_quota is actively retargeting a "
            "guided retry; default keeps the ordinary per-mode bands"
        ),
    )
    parser.add_argument(
        "--fa-alloc-band-below", type=float, default=1.0,
        help=(
            "RESEARCH: override --fa-alloc-band for below-assigned episodes "
            "only (faP evidence: below crossings need ~0.8 to beat min_cost "
            "near the sphere, but a global 0.8 band degrades SR/CR)"
        ),
    )
    parser.add_argument(
        "--route-guide-prototypes",
        type=Path,
        required=True,
        help=(
            "route/action-window prototype bank; relative paths are resolved "
            "inside --pretrain-dir so Helios staging remains self-contained"
        ),
    )
    parser.add_argument(
        "--route-guide-band", type=float, default=1.0,
        help=(
            "fraction of the verified min-cost span admitted to prototype "
            "matching; 1 permits every verifier-positive progress candidate"
        ),
    )
    parser.add_argument(
        "--route-guide-velocity-weight", type=float, default=0.5,
        help="velocity weight when matching the current state to a prototype",
    )
    parser.add_argument(
        "--route-guide-cosine-weight", type=float, default=0.25,
        help="full-window direction penalty added to weighted action MSE",
    )
    parser.add_argument(
        "--route-guide-time-decay", type=float, default=0.9,
        help="per-step geometric weight for prototype action-window matching",
    )
    parser.add_argument(
        "--route-guide-proposal-strengths",
        type=str,
        default=None,
        help=(
            "optional comma-separated convex blend strengths for replacing "
            "the final K candidate slots with progress-matched route-window "
            "proposals before acquisition and verification; omitted keeps "
            "the selection-only route guide"
        ),
    )
    parser.add_argument(
        "--geofence-floor-z", type=float, default=None,
        help=(
            "RESEARCH: relax the survival floor (OOB termination + verifier "
            "walls) to this z during expansion WITHOUT touching the context: "
            "the planepack/visual encoding still sees the original taskspace "
            "box, so the policy conditioning is unchanged. Only where the "
            "robot is ALLOWED to fly changes"
        ),
    )
    parser.add_argument(
        "--fa-below-diagonal", type=float, default=0.0,
        help=(
            "RESEARCH: lateral (left) component mixed into the below target "
            "direction; the feasible below crossing is diagonal-down, not "
            "straight under the sphere"
        ),
    )
    parser.add_argument("--replay-rounds", type=int, default=21)
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
    parser.add_argument("--candidate-perturb-std", type=float, default=0.0)
    parser.add_argument("--candidate-perturb-scope",
                        choices=("first_action", "coherent_horizon"),
                        default="coherent_horizon")
    parser.add_argument("--negative-alpha", type=float, default=0.0)
    parser.add_argument("--archive-rule",
                        choices=("all_queries", "executed_only",
                                 "executed_plus_nvp_negative",
                                 "successful_executed_windows"),
                        default="successful_executed_windows")
    parser.add_argument(
        "--successful-trajectory-selector",
        choices=("lowest_episode_id", "random_success",
                 "max_above_fraction", "max_mean_z"),
        default="random_success",
        help=(
            "admission ranking used only by successful_executed_windows; "
            "random_success uniformly orders terminal successes using a "
            "seeded round/gamma/episode key; "
            "max_above_fraction ranks terminal SUCCESS trajectories by the "
            "fraction of post-action states with z>=2, while max_mean_z ranks "
            "their mean post-action height; both break ties by episode id"
        ),
    )
    parser.add_argument(
        "--successful-trajectories-per-gamma",
        type=int,
        default=12,
        help=(
            "number of distinct ranked commit-capable terminal SUCCESS "
            "trajectories required and archived per gamma/round by "
            "successful_executed_windows"
        ),
    )
    parser.add_argument(
        "--sample-update-mode",
        type=str,
        default="0,0,0,1,1,1,2,2,2,3,3,3",
        help=(
            "RESEARCH: comma-separated exact route quota for committed "
            "successful trajectories (0=below,1=above,2=left,3=right). "
            "Every gamma/batch pairs --parallel-episodes guided and unguided "
            "rollouts at the same current round-scheduled flow-base std until "
            "this shared quota is filled; successes "
            "in already-full modes are not committed"
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
    parser.add_argument(
        "--execution-step-margin-weight",
        type=float,
        default=0.0,
        help=(
            "min_cost-only lambda in J_native - lambda * first-step nominal "
            "H_P margin; zero preserves native cost ranking"
        ),
    )
    parser.add_argument(
        "--execution-clearance-exp-weight",
        type=float,
        default=0.0,
        help=(
            "lab-clutter min_cost-only weight on mean_h "
            "exp((target-clearance_h)/temperature); 0 exactly restores the "
            "existing native execution score"
        ),
    )
    parser.add_argument(
        "--execution-clearance-exp-temperature",
        type=float,
        default=0.10,
        help="positive temperature for --execution-clearance-exp-weight",
    )
    parser.add_argument(
        "--execution-clearance-target-m",
        type=float,
        default=0.20,
        help="clearance target in meters for the optional exponential score",
    )
    parser.add_argument(
        "--execution-soft-clearance-weight",
        type=float,
        default=None,
        help=(
            "single-ball min_cost-only override of the native SafeMPPI "
            "quadratic soft-clearance weight; omitted preserves the task "
            "config exactly"
        ),
    )
    parser.add_argument(
        "--execution-soft-clearance-target-m",
        type=float,
        default=None,
        help=(
            "single-ball min_cost-only override of the native SafeMPPI "
            "clearance target in meters; omitted preserves the task config"
        ),
    )
    parser.add_argument(
        "--execution-taskspace-weight",
        type=float,
        default=None,
        help=(
            "single-ball min_cost-only override of the native taskspace "
            "exponential weight; pass 0 to remove J_bound from execution "
            "ranking while preserving verifier and rollout OOB checks"
        ),
    )
    parser.add_argument(
        "--execution-plane-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "min_cost-only penalty on nominal states near the z_bias_plane, "
            "matching the sphere-clutter runner so the same calibration "
            "transfers; unlike --execution-plane-escape-weight this is not "
            "gated by a pass-by disc and pairs with "
            "--execution-above-penalty-weight to break the up/down symmetry. "
            "0 disables"
        ),
    )
    parser.add_argument(
        "--execution-plane-penalty-sigma",
        type=float,
        default=0.15,
        help="width in metres of the plane penalty kernel",
    )
    parser.add_argument(
        "--execution-plane-penalty-shape",
        choices=("gaussian", "laplacian"),
        default="gaussian",
        help=(
            "in-plane penalty kernel: gaussian exp(-0.5 (dz/sigma)^2) "
            "(default, vanishes past ~2 sigma) or laplacian "
            "exp(-0.5 |dz|/sigma), whose slope persists across the whole z "
            "box so a broad heavy in-plane penalty still separates off-plane "
            "candidates"
        ),
    )
    parser.add_argument(
        "--execution-above-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "min_cost-only linear penalty on nominal height above the "
            "z_bias_plane. Pair with --execution-plane-penalty-weight: the "
            "above branch is monotonically taxed (so escaped routes go BELOW "
            "rather than above) only when this weight exceeds "
            "plane_weight / (2 * sigma). 0 disables"
        ),
    )
    parser.add_argument(
        "--execution-plane-escape-weight",
        type=float,
        default=0.0,
        help=(
            "single-sphere min_cost-only weight on mean_h "
            "exp(-(z_h-z_c)^2/(2 sigma^2)) over the nominal plan knots inside "
            "the horizontal pass-by disc, where z_c is the obstacle centre "
            "height; the term is symmetric in z, so leaving that plane upward "
            "or downward relieves it equally, and 0 exactly restores the "
            "existing native execution score"
        ),
    )
    parser.add_argument(
        "--execution-plane-escape-sigma",
        type=float,
        default=0.15,
        help=(
            "positive width in meters of the obstacle-plane escape band used "
            "by --execution-plane-escape-weight"
        ),
    )
    parser.add_argument(
        "--execution-plane-escape-gate-radius",
        type=float,
        default=1.0,
        help=(
            "horizontal radius in meters around the obstacle centre inside "
            "which the plane-escape term applies; knots farther away in the xy "
            "plane, including the start and goal, are never penalized"
        ),
    )
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
    parser.add_argument(
        "--event-log",
        choices=("none", "full", "committed_success"),
        default="committed_success",
        help=(
            "full stores every all-K gathering event; committed_success keeps the "
            "same complete event records for all terminal-SUCCESS episodes needed "
            "to audit the authoritative plural committed subset; none is for "
            "metric-only sweeps"
        ),
    )
    parser.add_argument(
        "--paired-noised-representation", action="store_true",
        help=("use phi_s((1-s)x_0+sU,c) with each sampled plan's paired Gaussian "
              "flow base for RBF calibration, acquisition, and replay"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
        if args.output.resolve() != args.resume_from.resolve():
            parser.error("--resume-from must equal --output")
    if args.helios:
        status = run_expansion_on_helios(
            args, sys.argv[1:], ROOT,
            script_name="research_ball_expansion_route_guided.py",
        )
        _write_reproduction_command(args.output)
        return status
    if not args.route_guide_prototypes.is_absolute():
        args.route_guide_prototypes = (
            args.pretrain_dir / args.route_guide_prototypes
        )
    if not args.route_guide_prototypes.is_file():
        parser.error(
            f"route prototype bank not found: {args.route_guide_prototypes}"
        )
    if not 0.0 <= args.route_guide_band <= 1.0:
        parser.error("--route-guide-band must lie in [0,1]")
    proposal_strengths: tuple[float, ...] = ()
    if args.route_guide_proposal_strengths is not None:
        try:
            proposal_strengths = tuple(
                float(value)
                for value in args.route_guide_proposal_strengths.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error(
                "--route-guide-proposal-strengths must be comma-separated floats"
            )
        if (
            not proposal_strengths
            or len(proposal_strengths) > args.K
            or any(
                not np.isfinite(value) or not 0.0 <= value <= 1.0
                for value in proposal_strengths
            )
        ):
            parser.error(
                "--route-guide-proposal-strengths must contain 1..K values "
                "in [0,1]"
            )
        if args.paired_noised_representation:
            parser.error(
                "route proposal injection cannot use "
                "--paired-noised-representation: an injected endpoint has no "
                "corresponding sampled flow base"
            )
        if args.fa_alloc == "none":
            parser.error("route proposal injection requires --fa-alloc")
    if (
        args.event_log == "committed_success"
        and args.archive_rule != "successful_executed_windows"
    ):
        parser.error(
            "--event-log committed_success requires "
            "--archive-rule successful_executed_windows"
        )
    if args.head_only_expansion and args.train_gru_during_expansion:
        parser.error(
            "--head-only-expansion cannot be combined with "
            "--train-gru-during-expansion"
        )
    if args.trainable_trunk_layers is not None:
        if args.head_only_expansion:
            parser.error(
                "--trainable-trunk-layers cannot be combined with "
                "--head-only-expansion"
            )
        if args.train_gru_during_expansion:
            parser.error(
                "--trainable-trunk-layers cannot be combined with "
                "--train-gru-during-expansion"
            )
        if args.trainable_trunk_layers < 1:
            parser.error("--trainable-trunk-layers must be positive")
    if (
        not np.isfinite(args.execution_clearance_exp_weight)
        or args.execution_clearance_exp_weight < 0.0
    ):
        parser.error(
            "--execution-clearance-exp-weight must be finite and nonnegative"
        )
    if (
        not np.isfinite(args.execution_clearance_exp_temperature)
        or args.execution_clearance_exp_temperature <= 0.0
    ):
        parser.error(
            "--execution-clearance-exp-temperature must be finite and positive"
        )
    if (
        not np.isfinite(args.execution_clearance_target_m)
        or args.execution_clearance_target_m < 0.0
    ):
        parser.error(
            "--execution-clearance-target-m must be finite and nonnegative"
        )
    for option, value in (
        ("--execution-soft-clearance-weight",
         args.execution_soft_clearance_weight),
        ("--execution-soft-clearance-target-m",
         args.execution_soft_clearance_target_m),
        ("--execution-taskspace-weight",
         args.execution_taskspace_weight),
    ):
        if value is not None and (
            not np.isfinite(value) or value < 0.0
        ):
            parser.error(f"{option} must be finite and nonnegative")
    if (
        args.execution_rule != "min_cost"
        and (
            args.execution_soft_clearance_weight is not None
            or args.execution_soft_clearance_target_m is not None
            or args.execution_taskspace_weight is not None
        )
    ):
        parser.error(
            "execution-cost overrides require "
            "--execution-rule min_cost"
        )
    for option, value in (
        ("--execution-plane-escape-weight",
         args.execution_plane_escape_weight),
        ("--execution-plane-escape-gate-radius",
         args.execution_plane_escape_gate_radius),
        ("--execution-plane-penalty-weight",
         args.execution_plane_penalty_weight),
        ("--execution-above-penalty-weight",
         args.execution_above_penalty_weight),
    ):
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"{option} must be finite and nonnegative")
    if (
        not np.isfinite(args.execution_plane_penalty_sigma)
        or args.execution_plane_penalty_sigma <= 0.0
    ):
        parser.error(
            "--execution-plane-penalty-sigma must be finite and positive"
        )
    if (
        args.execution_rule != "min_cost"
        and (
            args.execution_plane_penalty_weight != 0.0
            or args.execution_above_penalty_weight != 0.0
        )
    ):
        parser.error(
            "--execution-plane-penalty-weight/--execution-above-penalty-weight "
            "require --execution-rule min_cost"
        )
    if (
        not np.isfinite(args.execution_plane_escape_sigma)
        or args.execution_plane_escape_sigma <= 0.0
    ):
        parser.error(
            "--execution-plane-escape-sigma must be finite and positive"
        )
    if (
        args.execution_plane_escape_weight != 0.0
        and args.execution_rule != "min_cost"
    ):
        parser.error(
            "--execution-plane-escape-weight requires --execution-rule min_cost"
        )
    if (
        not np.isfinite(args.execution_step_margin_weight)
        or args.execution_step_margin_weight < 0.0
    ):
        parser.error(
            "--execution-step-margin-weight must be finite and nonnegative"
        )
    if (
        args.execution_step_margin_weight > 0.0
        and args.execution_rule != "min_cost"
    ):
        parser.error(
            "--execution-step-margin-weight requires --execution-rule min_cost"
        )
    if (
        args.execution_step_margin_weight > 0.0
        and (
            args.execution_clearance_exp_weight > 0.0
            or args.execution_soft_clearance_weight is not None
            or args.execution_soft_clearance_target_m is not None
        )
    ):
        parser.error(
            "--execution-step-margin-weight cannot be combined with proximity "
            "execution overrides"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"--device {args.device!r} requires CUDA")
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
            if (
                args.execution_step_margin_weight > 0.0
                and float(task_config.safemppi.soft_clearance_weight) != 0.0
            ):
                parser.error(
                    "--execution-step-margin-weight requires the task config "
                    "soft_clearance_weight to be zero"
                )
            randomization = task_config.raw.get("scene_randomization", {})
            clutter_profile = _lab_clutter_profile(task_config)
            if clutter_profile:
                scene_spec = sphere_scene_spec_from_config(task_config)
                policy = load_lab_clutter_expansion_policy(
                    args.pretrain_dir / "pretrained.pt",
                    verifier_suffix_dim=(
                        LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim
                    ),
                    train_history_encoder=(
                        args.train_gru_during_expansion
                    ),
                ).to(device)
                effective_clearance_weight = (
                    args.execution_clearance_exp_weight
                    if args.execution_rule == "min_cost" else 0.0
                )
                task = LabClutterSphereExpansionTask(
                    task_config,
                    context_schema=policy.context_schema,
                    device=device,
                    execution_z_bias_mode=args.execution_z_bias_mode,
                    tight_corridor=args.tight_corridor,
                    verifier_mode=args.verifier_mode,
                    verifier_solver=args.verifier_solver,
                    execution_clearance_exp_weight=(
                        effective_clearance_weight
                    ),
                    execution_clearance_exp_temperature=(
                        args.execution_clearance_exp_temperature
                    ),
                    execution_clearance_target_m=(
                        args.execution_clearance_target_m
                    ),
                    execution_taskspace_weight=(
                        args.execution_taskspace_weight
                    ),
                    scene_spec=scene_spec,
                )
            else:
                policy = load_lab_expansion_policy(
                    args.pretrain_dir / "pretrained.pt",
                    train_history_encoder=(
                        args.train_gru_during_expansion
                    ),
                ).to(device)
                task = LabFlowExpansionTask(
                    task_config,
                    context_schema=policy.context_schema,
                    device=device,
                    execution_z_bias_mode=args.execution_z_bias_mode,
                    tight_corridor=args.tight_corridor,
                    verifier_mode=args.verifier_mode,
                    verifier_solver=args.verifier_solver,
                    execution_soft_clearance_weight=(
                        args.execution_soft_clearance_weight
                    ),
                    execution_soft_clearance_target_m=(
                        args.execution_soft_clearance_target_m
                    ),
                    execution_taskspace_weight=(
                        args.execution_taskspace_weight
                    ),
                    execution_plane_escape_weight=(
                        args.execution_plane_escape_weight
                    ),
                    execution_plane_escape_sigma=(
                        args.execution_plane_escape_sigma
                    ),
                    execution_plane_escape_gate_radius=(
                        args.execution_plane_escape_gate_radius
                    ),
                    execution_plane_penalty_weight=(
                        args.execution_plane_penalty_weight
                    ),
                    execution_plane_penalty_sigma=(
                        args.execution_plane_penalty_sigma
                    ),
                    execution_plane_penalty_shape=(
                        args.execution_plane_penalty_shape
                    ),
                    execution_above_penalty_weight=(
                        args.execution_above_penalty_weight
                    ),
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
        if args.freeze_visual_encoder_during_expansion:
            parser.error(
                "--freeze-visual-encoder-during-expansion requires a lab "
                "visual checkpoint"
            )
        if args.train_gru_during_expansion:
            parser.error(
                "--train-gru-during-expansion requires a lab visual-history "
                "checkpoint"
            )
        clutter_profile = False
        policy = load_policy(args.pretrain_dir / "pretrained.pt").to(device)
        task_config = load_config(args.pretrain_dir / "demo_config.json")
        context_contract = pretrain.get("context_contract", "legacy10")
        task_type = (
            ThetaBallFlowTask
            if context_contract == "local_theta12" else BallFlowTask
        )
        task = task_type(
            task_config,
            device=device,
            execution_z_bias_mode=args.execution_z_bias_mode,
            tight_corridor=args.tight_corridor,
            target_region=args.target_region,
            verifier_mode=args.verifier_mode,
            verifier_solver=args.verifier_solver,
        )
    _set_flow_nfe(policy, args.flow_nfe)
    if (
        args.execution_clearance_exp_weight > 0.0
        and not (lab_profile and clutter_profile)
    ):
        parser.error(
            "--execution-clearance-exp-weight is supported only by the "
            "randomized lab-sphere task"
        )
    if (
        args.execution_plane_escape_weight != 0.0
        and not (lab_profile and not clutter_profile)
    ):
        parser.error(
            "--execution-plane-escape-weight is supported only by the "
            "single-sphere lab task"
        )
    sample_update_mode = None
    if args.sample_update_mode is not None:
        try:
            sample_update_mode = tuple(
                int(value.strip())
                for value in args.sample_update_mode.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error(
                "--sample-update-mode must be a comma-separated list of "
                "integer route IDs"
            )
        if not sample_update_mode or any(
            mode < 0 or mode > 3 for mode in sample_update_mode
        ):
            parser.error("--sample-update-mode entries must lie in [0,3]")
        if len(sample_update_mode) != args.successful_trajectories_per_gamma:
            parser.error(
                "--sample-update-mode must contain exactly "
                "--successful-trajectories-per-gamma entries"
            )
        if args.archive_rule != "successful_executed_windows":
            parser.error(
                "--sample-update-mode requires "
                "--archive-rule successful_executed_windows"
            )
        if not (lab_profile and not clutter_profile):
            parser.error(
                "--sample-update-mode requires the single-sphere lab task"
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
        retry_exhaustion_policy=args.retry_exhaustion_policy,
        retry_resample_batch_cap=args.retry_resample_batch_cap,
        max_steps=task_config.taskspace.max_steps, K=args.K, B=args.B,
        retry_B=args.retry_B,
        retry_verify_all_fast_path=args.retry_verify_all_fast_path,
        batch_size=args.batch_size,
        inner_steps=args.optimizer_steps_per_round,
        optimizer_steps_total=args.optimizer_steps_total,
        optimizer_step_allocation=args.optimizer_step_allocation,
        replay_passes=args.replay_passes_per_round,
        microbatch_repeats=args.inner_steps,
        learning_rate=args.learning_rate,
        learning_rate_final=args.learning_rate_final,
        learning_rate_decay_steps=args.learning_rate_decay_steps,
        round_learning_rate_warmup_power=(
            args.round_learning_rate_warmup_power
        ),
        gradient_clip_norm=args.gradient_clip_norm,
        first_layer_lr_scale=args.first_layer_lr_scale,
        freeze_visual_encoder=(
            args.freeze_visual_encoder_during_expansion
        ),
        head_only_update=args.head_only_expansion,
        replay_rounds=args.replay_rounds,
        replay_scope=args.replay_scope,
        replay_batch_sampler=args.replay_batch_sampler,
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
        execution_step_margin_weight=args.execution_step_margin_weight,
        execution_ess_target=args.execution_ess_target,
        execution_uncertainty_weight=args.execution_uncertainty_weight,
        archive_rule=args.archive_rule,
        successful_trajectory_selector=args.successful_trajectory_selector,
        successful_trajectories_per_gamma=(
            args.successful_trajectories_per_gamma
        ),
        sample_update_mode=sample_update_mode,
        replay_acceptance=args.replay_acceptance,
        paired_noised_representation=args.paired_noised_representation,
        batched_rollout_sampling=args.batched_rollout_sampling,
        flow_nfe=args.flow_nfe,
        acquisition_feature=args.acquisition_feature,
        coverage_replay=args.coverage_replay,
        replay_augmentation=args.replay_augmentation,
        target_gate_start_round=args.target_gate_start_round,
        gp_reference_mode=args.gp_reference_mode,
        gp_sliding_row_selector=args.gp_sliding_row_selector,
        gp_exact_max_rows_per_gamma=args.gp_exact_max_rows_per_gamma,
        seed=args.seed,
    )

    resume_completed_round = 0
    events = []
    if args.resume_from is not None:
        resume_metadata_path = args.resume_from / "resume_state.json"
        if not resume_metadata_path.is_file():
            raise FileNotFoundError(
                f"missing committed resume metadata: {resume_metadata_path}"
            )
        resume_metadata = json.loads(resume_metadata_path.read_text())
        resume_completed_round = int(resume_metadata["completed_round"])
        if args.event_log in {"full", "committed_success"}:
            for round_index in range(1, resume_completed_round + 1):
                event_path = args.resume_from / f"events_round_{round_index:03d}.pt"
                if not event_path.is_file():
                    raise FileNotFoundError(
                        "exact visual/event resume is missing committed events: "
                        f"{event_path}"
                    )
                events.extend(torch.load(
                    event_path, map_location="cpu", weights_only=False,
                ))
    pending_events = []
    source_event_count = len(events)
    pruned_event_count = 0

    def callback(event):
        nonlocal source_event_count
        event_context = event["context"].numpy()
        if lab_profile:
            # The visual grid volume is reproducible from robot state and
            # scene geometry; storing it at every event would add many GB to a
            # 50-round log.  GRU history is not reconstructible and is kept.
            suffix_dim = (
                task.verifier_suffix_dim if clutter_profile else 6
            )
            base_packed_dim = LAB_CONTEXT_BASE_PACKED_DIMS.get(
                context_contract,
                int(policy.policy_context_dim),
            )
            history = (
                event_context[base_packed_dim:policy.policy_context_dim]
                if context_contract in LAB_HISTORY_CONTEXT_SCHEMAS
                else np.empty(0, np.float32)
            )
            event_context = np.concatenate([
                event_context[:7],
                history,
                event_context[-suffix_dim:],
            ]).astype(np.float32)
        compact_event = {
            "round": event["round"], "step": event["step"], "gamma": event["gamma"],
            "episode": event["episode"], "context_id": event["context_id"],
            "retry_batch": event["retry_batch"], "replica": event["replica"],
            "sample_update_cohort": event["sample_update_cohort"],
            "flow_base_std": float(event["flow_base_std"]),
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
        }
        source_event_count += 1
        if args.event_log == "full":
            events.append(compact_event)
        else:
            pending_events.append(compact_event)

    def committed_round_callback(round_row):
        nonlocal pruned_event_count
        retained = _retain_committed_round_events(
            pending_events, round_row,
        )
        events.extend(retained)
        pruned_event_count += len(pending_events) - len(retained)
        pending_events.clear()

    retry_needed_modes: dict[tuple[int, float], list[int]] = {}

    def retry_progress(retry_row):
        retry_needed_modes[
            (int(retry_row["round"]), float(retry_row["gamma"]))
        ] = list(map(int, retry_row["needed_modes"]))
        needed = ",".join(map(str, retry_row["needed_modes"]))
        counts = retry_row["cohort_mode_counts"]
        guided = counts["guided"]
        unguided = counts["unguided"]
        fa_stats["retry_progress"].append({
            "round": int(retry_row["round"]),
            "gamma": float(retry_row["gamma"]),
            "retry_batch": int(retry_row["retry_batch"]),
            "retry_batch_cap": int(retry_row["retry_batch_cap"]),
            "needed_modes": list(map(int, retry_row["needed_modes"])),
            "guided_mode_counts": list(map(int, guided)),
            "unguided_mode_counts": list(map(int, unguided)),
            "candidate_count_K": int(args.K),
            "query_count_B": int(
                args.retry_B
                if int(retry_row["retry_batch"]) > 0 else args.B
            ),
            "flow_base_std": float(config.flow_base_std),
        })
        _write_fa_stats()
        print(
            f"[sample-update] r{int(retry_row['round'])} "
            f"gamma {float(retry_row['gamma']):g} "
            f"retry batch {int(retry_row['retry_batch'])}/"
            f"{int(retry_row['retry_batch_cap']) - 1} | "
            f"guided b{guided[0]}/a{guided[1]}/l{guided[2]}/r{guided[3]} | "
            f"unguided b{unguided[0]}/a{unguided[1]}"
            f"/l{unguided[2]}/r{unguided[3]} | "
            f'need mode "{needed}"',
            flush=True,
        )

    started = time.perf_counter()

    # ------------------------------------------------------------------
    # RESEARCH instrumentation: first-action distribution tracker + std/beta
    # scheduling.  Everything lives in this script; the library is untouched.
    # The tracker measures, at the episode start state, the angular spread of
    # the first plan action in the head-on (lateral/vertical) plane; the
    # coverage deficit (1 - normalized entropy over the 4 route quadrants)
    # optionally drives the next round's flow-base-std and an acquisition
    # beta scale.
    # ------------------------------------------------------------------
    from safe_mppi.ball_flow_theta import start_goal_frame as _sg_frame
    from safe_mppi import expansion as _expansion_mod

    if args.flow_base_std_schedule != "none" and not lab_profile:
        parser.error("--flow-base-std-schedule requires the lab task profile")
    std_init = float(args.flow_base_std)
    std_final = (
        std_init if args.flow_base_std_final is None
        else float(args.flow_base_std_final)
    )
    if not (std_final > 0.0) or not np.isfinite(std_final):
        parser.error("--flow-base-std-final must be finite and positive")
    if args.first_action_samples < 16:
        parser.error("--first-action-samples must be >= 16")
    research_state = {
        "applied_std": std_init,
        "beta_scale": 1.0,
        "deficit": None,
        "rows": [],
    }
    if args.resume_from is not None:
        first_action_path = args.resume_from / "first_action_stats.json"
        if first_action_path.is_file():
            previous_first_action = json.loads(first_action_path.read_text())
            research_state["rows"] = list(previous_first_action.get("rows", []))
            if research_state["rows"]:
                last_first_action = research_state["rows"][-1]
                research_state["deficit"] = float(
                    last_first_action["coverage_deficit"]
                )
                research_state["beta_scale"] = float(
                    last_first_action.get("beta_scale_used", 1.0)
                )
        research_state["applied_std"] = float(
            resume_metadata["flow_base_std_next"]
        )
    _frame = _sg_frame(task.env) if lab_profile else None

    def _angular_occupancy(round_index: int, base_std: float) -> np.ndarray:
        sample_count = int(args.first_action_samples)
        pooled = np.zeros(4)
        per_gamma = {}
        with torch.no_grad():
            for g_idx, gamma in enumerate(config.gammas):
                probe_state = task.reset(
                    float(gamma), 0, 770000 + 97 * round_index + g_idx
                )
                probe_context = task.context(probe_state, float(gamma))
                probe_context = probe_context.detach().to(device)
                gen = torch.Generator(device=device).manual_seed(
                    880000 + 97 * round_index + g_idx
                )
                if base_std == 1.0:
                    plans = policy.sample(probe_context, sample_count, gen)
                else:
                    plans = policy.sample(
                        probe_context, sample_count, gen,
                        base_std=base_std,
                    )
                first = (
                    plans.detach().reshape(sample_count, -1, 3)[:, 0, :]
                    .cpu().numpy()
                )
                local = first @ np.asarray(_frame, np.float64)
                angles = np.degrees(np.arctan2(local[:, 2], local[:, 1]))
                counts = np.array([
                    np.sum((angles >= -135) & (angles < -45)),   # below
                    np.sum((angles >= 45) & (angles < 135)),     # above
                    np.sum((angles >= -45) & (angles < 45)),     # left
                    np.sum((angles >= 135) | (angles < -135)),   # right
                ], float)
                pooled += counts
                per_gamma[f"{float(gamma):g}"] = (counts / counts.sum()).tolist()
        return pooled / pooled.sum(), per_gamma

    def _entropy_norm(occupancy: np.ndarray) -> float:
        nonzero = occupancy[occupancy > 0]
        return float(-(nonzero * np.log(nonzero)).sum() / np.log(4.0))

    def _first_action_stats(round_index: int) -> dict:
        current_std = float(config.flow_base_std)
        occupancy, per_gamma = _angular_occupancy(round_index, current_std)
        # Fixed-noise reference series: the same measurement at base_std=1.0
        # every round, so policy-head collapse is separable from the std knob.
        if current_std == 1.0:
            occupancy_ref = occupancy
        else:
            occupancy_ref, _ = _angular_occupancy(round_index, 1.0)
        h_norm = _entropy_norm(occupancy)
        deficit = float(np.clip(1.0 - h_norm, 0.0, 1.0))
        row = {
            "round": round_index,
            "flow_base_std_used": current_std,
            "beta_scale_used": research_state["beta_scale"],
            "occupancy": {
                "below": occupancy[0], "above": occupancy[1],
                "left": occupancy[2], "right": occupancy[3],
            },
            "occupancy_ref_std1": {
                "below": occupancy_ref[0], "above": occupancy_ref[1],
                "left": occupancy_ref[2], "right": occupancy_ref[3],
            },
            "entropy_normalized_ref_std1": _entropy_norm(occupancy_ref),
            "per_gamma_occupancy": per_gamma,
            "entropy_normalized": h_norm,
            "coverage_deficit": deficit,
            "samples_per_gamma": int(args.first_action_samples),
        }
        research_state["rows"].append(row)
        research_state["deficit"] = deficit
        # run_safe_expansion demands an empty output dir at start, so the
        # round-0 row stays buffered until the dir exists (round 1 onward).
        if args.output.is_dir():
            (args.output / "first_action_stats.json").write_text(json.dumps({
                "schedule": args.flow_base_std_schedule,
                "std_init": std_init,
                "std_final": std_final,
                "adaptive_std_gain": args.adaptive_std_gain,
                "beta_coverage_gain": args.beta_coverage_gain,
                "rows": research_state["rows"],
            }, indent=2))
        print(
            f"[first-action] r{round_index:>2} std {current_std:.3f} "
            f"beta_scale {research_state['beta_scale']:.2f} "
            f"occ b{occupancy[0]:.2f}/a{occupancy[1]:.2f}"
            f"/l{occupancy[2]:.2f}/r{occupancy[3]:.2f} "
            f"ref1 b{occupancy_ref[0]:.2f}/a{occupancy_ref[1]:.2f}"
            f"/l{occupancy_ref[2]:.2f}/r{occupancy_ref[3]:.2f} "
            f"Hn {h_norm:.3f} deficit {deficit:.3f}",
            flush=True,
        )
        return row

    def _schedule_std(next_round: int) -> float:
        total = max(int(args.rounds), 1)
        if args.flow_base_std_schedule == "none":
            return std_init
        if args.flow_base_std_schedule == "linear":
            frac = _round_schedule_fraction(next_round, total)
            return std_init + (std_final - std_init) * frac
        if args.flow_base_std_schedule == "cosine":
            frac = _round_schedule_fraction(next_round, total)
            return std_final + (std_init - std_final) * 0.5 * (
                1.0 + np.cos(np.pi * frac)
            )
        deficit = research_state["deficit"]
        if deficit is None:
            return std_init
        return std_final + (std_init - std_final) * (
            deficit ** float(args.adaptive_std_gain)
        )

    # ------------------------------------------------------------------
    # RESEARCH: repulsive episode ensemble -- quadrant allocation of the
    # parallel episodes at execution-selection time (EXECUTION_SELECTION_HOOK
    # in safe_mppi.expansion; hook only ever re-picks among candidates that
    # already passed the full verifier + progress gate).
    # ------------------------------------------------------------------
    QUADRANT_NAMES = {0: "below", 1: "above", 2: "left", 3: "right"}
    if args.fa_alloc_map is not None:
        fa_map = [int(x) % 4 for x in args.fa_alloc_map.split(",") if x.strip()]
        if not fa_map:
            parser.error("--fa-alloc-map must list at least one quadrant id")
    else:
        fa_map = [0, 1, 2, 3]
    route_guide = RoutePrototypeGuide.load(
        args.route_guide_prototypes,
        device=device,
        velocity_weight=args.route_guide_velocity_weight,
        cosine_weight=args.route_guide_cosine_weight,
        time_decay=args.route_guide_time_decay,
    )
    fa_stats = {
        "mode": "route_action_prototype",
        "steps": "until_terminal",
        "band": float(args.fa_alloc_band),
        "step0_band": float(args.fa_alloc_step0_band),
        "retry_map": args.fa_alloc_retry_map,
        "retry_band": args.fa_alloc_retry_band,
        "sample_update_retry_guidance": (
            "paired_guided_and_unguided_cohorts"
            if sample_update_mode is not None else "unchanged"
        ),
        "retry_progress": [],
        "rounds": {},
        "route_guide": {
            "prototype_bank": str(args.route_guide_prototypes),
            "band": float(args.route_guide_band),
            "velocity_weight": float(args.route_guide_velocity_weight),
            "cosine_weight": float(args.route_guide_cosine_weight),
            "time_decay": float(args.route_guide_time_decay),
            "contract": (
                "nearest position+velocity success state followed by "
                "full verified action-window matching"
            ),
            "proposal_strengths": list(proposal_strengths),
            "proposal_contract": (
                "retain ordinary policy candidates, then convex-blend only "
                "the final candidate slots before acquisition and full "
                "verification"
                if proposal_strengths else "selection_only"
            ),
        },
    }
    if args.resume_from is not None:
        fa_log_path = args.resume_from / "fa_alloc_log.json"
        if fa_log_path.is_file():
            previous_fa_stats = json.loads(fa_log_path.read_text())
            fa_stats["retry_progress"] = list(
                previous_fa_stats.get("retry_progress", [])
            )
            fa_stats["rounds"] = dict(previous_fa_stats.get("rounds", {}))

    def _write_fa_stats():
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "fa_alloc_log.json").write_text(
            json.dumps(fa_stats, indent=2) + "\n"
        )

    def _fa_round_bucket(round_i):
        return fa_stats["rounds"].setdefault(str(round_i), {
            "offers": 0, "overrides": 0,
            "per_quadrant_offers": {name: 0 for name in QUADRANT_NAMES.values()},
            "per_quadrant_hits": {name: 0 for name in QUADRANT_NAMES.values()},
            "route_guide_offers": 0,
            "route_guide_overrides": 0,
            "route_guide_missing_groups": 0,
            "route_guide_score_sum": 0.0,
            "route_guide_cosine_sum": 0.0,
            "route_proposal_contexts": 0,
            "route_proposal_candidates": 0,
            "route_proposal_missing_groups": 0,
        })

    def _active_quadrant(*, episode, gamma, round_i):
        retry_modes = retry_needed_modes.get(
            (int(round_i), float(gamma)),
        )
        active_map = (
            retry_modes
            if (
                args.fa_alloc_retry_map == "missing_quota"
                and int(episode.get("retry_batch", 0)) > 0
                and retry_modes
            )
            else fa_map
        )
        return active_map[int(episode["replica"]) % len(active_map)]

    def _route_proposal_hook(*, episode, step, gamma, round_i, context,
                             state_before, candidates, base_candidates,
                             flow_bases):
        if not proposal_strengths:
            return None
        if (
            sample_update_mode is not None
            and episode["sample_update_cohort"] == "unguided"
        ):
            return None
        if flow_bases is not None:
            raise ValueError(
                "route proposal injection requires endpoint-only acquisition"
            )
        quadrant = _active_quadrant(
            episode=episode, gamma=gamma, round_i=round_i,
        )
        target, valid_horizon, _ = route_guide.target_window(
            episode=episode, gamma=float(gamma), route=quadrant,
        )
        bucket = _fa_round_bucket(round_i)
        if target is None:
            bucket["route_proposal_missing_groups"] += 1
            return None
        bucket["route_proposal_contexts"] += 1
        bucket["route_proposal_candidates"] += len(proposal_strengths)
        return blend_route_proposals(
            candidates, target, valid_horizon, proposal_strengths,
        )

    def _fa_hook(*, episode, step, gamma, round_i, eligible, queried,
                 execution_key, chosen):
        if (
            sample_update_mode is not None
            and episode["sample_update_cohort"] == "unguided"
        ):
            return None
        quadrant = _active_quadrant(
            episode=episode, gamma=gamma, round_i=round_i,
        )
        keys = {index: float(execution_key(index)) for index in eligible}
        best = min(keys.values())
        worst = max(keys.values())
        cutoff = best + float(args.route_guide_band) * (worst - best)
        admitted = [i for i in eligible if keys[i] <= cutoff + 1e-12]
        route_choice, route_diagnostics = route_guide.choose(
            episode=episode,
            gamma=float(gamma),
            route=quadrant,
            candidates=queried,
            eligible=admitted,
        )
        bucket = _fa_round_bucket(round_i)
        bucket["route_guide_offers"] += 1
        name = QUADRANT_NAMES[quadrant]
        bucket["per_quadrant_offers"][name] += 1
        if route_choice is None:
            bucket["route_guide_missing_groups"] += 1
            return None
        bucket["route_guide_score_sum"] += float(route_diagnostics["score"])
        bucket["route_guide_cosine_sum"] += float(route_diagnostics["cosine"])
        if route_choice == chosen:
            return None
        bucket["route_guide_overrides"] += 1
        bucket["overrides"] += 1
        return route_choice

    if args.fa_alloc != "none":
        if not lab_profile:
            parser.error("--fa-alloc requires the lab task profile")
        _expansion_mod.EXECUTION_SELECTION_HOOK = _fa_hook
        if proposal_strengths:
            _expansion_mod.CANDIDATE_PROPOSAL_HOOK = _route_proposal_hook

    # ------------------------------------------------------------------
    # RESEARCH: survival-floor relaxation. env.bounds feeds three consumers:
    # (1) planepack context (policy conditioning) -- MUST keep the original
    #     box, else the policy goes off-distribution (measured -0.2 SR);
    # (2) OOB termination in advance()/raw rollouts;
    # (3) verifier validity + H_P wall margins (workers inherit env at spawn).
    # We relax the stored bounds (2+3) and flip the original floor back only
    # for the duration of each task.context() call (1).
    # ------------------------------------------------------------------
    if args.geofence_floor_z is not None:
        if not lab_profile:
            parser.error("--geofence-floor-z requires the lab task profile")
        original_floor = float(task.env.bounds[2, 0])
        relaxed_floor = float(args.geofence_floor_z)
        if relaxed_floor >= original_floor:
            parser.error(
                "--geofence-floor-z must be below the configured floor "
                f"({original_floor})"
            )
        task.env.bounds = np.array(task.env.bounds, float).copy()
        task.env.bounds[2, 0] = relaxed_floor
        # Class-level patch (NOT an instance attribute: the verifier workers
        # pickle the task instance and a closure in __dict__ would break
        # that; workers never call context()).
        _context_unpatched = type(task).context

        def _context_with_original_walls(self, state, gamma):
            self.env.bounds[2, 0] = original_floor
            try:
                return _context_unpatched(self, state, gamma)
            finally:
                self.env.bounds[2, 0] = relaxed_floor

        type(task).context = _context_with_original_walls
        print(
            f"[geofence] survival floor relaxed {original_floor} -> "
            f"{relaxed_floor} (context/planepack keeps {original_floor})",
            flush=True,
        )

    if args.beta_coverage_gain > 0.0:
        _original_acquire = _expansion_mod.RBFPosterior.acquire

        def _scaled_acquire(self, features, B, beta, *acq_args, **acq_kwargs):
            return _original_acquire(
                self, features, B,
                beta * research_state["beta_scale"],
                *acq_args, **acq_kwargs,
            )

        _expansion_mod.RBFPosterior.acquire = _scaled_acquire

    if lab_profile and args.resume_from is None:
        _first_action_stats(0)
        if args.flow_base_std_schedule == "adaptive":
            # Round 1 already reacts to the measured pretrained collapse.
            object.__setattr__(config, "flow_base_std", _schedule_std(1))
            research_state["applied_std"] = float(config.flow_base_std)
        if args.beta_coverage_gain > 0.0:
            research_state["beta_scale"] = (
                1.0 + args.beta_coverage_gain * research_state["deficit"]
            )
    elif lab_profile:
        object.__setattr__(
            config, "flow_base_std", float(resume_metadata["flow_base_std_next"]),
        )
        print(
            f"[resume] committed r{resume_completed_round}; next r"
            f"{resume_completed_round + 1} std {config.flow_base_std:.3f}",
            flush=True,
        )

    def _committed_modes_line(round_row: dict) -> str:
        """Crossing-quadrant counts of THIS round's committed successes.

        With sample-update quotas, the authoritative IDs were classified from
        the same dense path and crossing function as raw evaluation.
        """
        if not lab_profile:
            return ""
        round_index = int(round_row["round"])
        counts = {"below": 0, "above": 0, "left": 0, "right": 0}
        if sample_update_mode is not None:
            names = ("below", "above", "left", "right")
            for detail in round_row[
                "successful_executed_commit_by_gamma"
            ].values():
                for mode in detail["committed_sample_update_modes"]:
                    counts[names[int(mode)]] += 1
            return (
                f" modes b{counts['below']}/a{counts['above']}"
                f"/l{counts['left']}/r{counts['right']}"
            )
        committed = {
            (float(gamma), int(episode_id))
            for gamma, detail in round_row[
                "successful_executed_commit_by_gamma"
            ].items()
            for episode_id in detail["committed_episode_ids"]
        }
        sphere = np.asarray(task.env.spheres[0], float)
        frame = np.asarray(_frame, float)
        start = np.asarray(task.env.start[:3], float)
        centre_local = (sphere[:3] - start) @ frame
        grouped: dict[tuple, list] = {}
        for ev in events:
            if int(ev["round"]) != round_index:
                continue
            key = (ev["gamma"], ev["episode"], ev["retry_batch"], ev["replica"])
            grouped.setdefault(key, []).append(ev)
        for evs in grouped.values():
            evs.sort(key=lambda e: e["step"])
            if (
                evs[-1]["status"] != "SUCCESS"
                or (float(evs[-1]["gamma"]), int(evs[-1]["episode"]))
                not in committed
            ):
                continue
            states = np.array(
                [e["robot"][:3] for e in evs] + [evs[-1]["robot_after"][:3]],
                float,
            )
            local = (states - start) @ frame
            idx = int(np.argmin(np.abs(local[:, 0] - centre_local[0])))
            d = local[idx, 1:3] - centre_local[1:3]
            ang = np.degrees(np.arctan2(d[1], d[0]))
            if -135 <= ang < -45:
                counts["below"] += 1
            elif 45 <= ang < 135:
                counts["above"] += 1
            elif -45 <= ang < 45:
                counts["left"] += 1
            else:
                counts["right"] += 1
        return (
            f" modes b{counts['below']}/a{counts['above']}"
            f"/l{counts['left']}/r{counts['right']}"
        )

    def round_progress(round_row):
        if args.event_log == "committed_success":
            committed_round_callback(round_row)
        if args.event_log in {"full", "committed_success"}:
            committed_round = int(round_row["round"])
            _atomic_torch_save(
                [
                    event for event in events
                    if int(event["round"]) == committed_round
                ],
                args.output / f"events_round_{committed_round:03d}.pt",
            )
        show_progress(
            "Stage-1 expansion",
            int(round_row["round"]),
            int(args.rounds),
        )
        # Stream the same per-round record the end-of-run recap prints, so a
        # long expansion is observable while it runs instead of only after.
        # The trailing "modes b/a/l/r" counts are this round's COMMITTED
        # successes classified at the ball-plane crossing (same rule as the
        # evaluation's route modes).
        print(
            format_round_summary(round_row)
            + _committed_modes_line(round_row),
            flush=True,
        )
        if args.fa_alloc != "none":
            _write_fa_stats()
            bucket = fa_stats["rounds"].get(str(int(round_row["round"])))
            if bucket:
                offers = int(bucket["route_guide_offers"])
                mean_score = (
                    float(bucket["route_guide_score_sum"]) / offers
                    if offers else 0.0
                )
                mean_cosine = (
                    float(bucket["route_guide_cosine_sum"]) / offers
                    if offers else 0.0
                )
                print(
                    f"[route-guide] r{int(round_row['round']):>2} offers "
                    f"{offers} overrides {bucket['route_guide_overrides']} "
                    f"missing_groups {bucket['route_guide_missing_groups']} "
                    f"mean_score {mean_score:.5f} "
                    f"mean_cosine {mean_cosine:.4f}",
                    flush=True,
                )
                if proposal_strengths:
                    print(
                        f"[route-proposal] r{int(round_row['round']):>2} "
                        f"contexts {bucket['route_proposal_contexts']} "
                        f"candidates {bucket['route_proposal_candidates']} "
                        f"missing_groups "
                        f"{bucket['route_proposal_missing_groups']}",
                        flush=True,
                    )
        if lab_profile:
            finished = int(round_row["round"])
            _first_action_stats(finished)
            next_std = _schedule_std(finished + 1)
            object.__setattr__(config, "flow_base_std", float(next_std))
            research_state["applied_std"] = float(next_std)
            if args.beta_coverage_gain > 0.0:
                research_state["beta_scale"] = (
                    1.0 + args.beta_coverage_gain * research_state["deficit"]
                )

    try:
        manifest = run_safe_expansion(
            policy, task, args.output, config=config,
            calibration_features=calibration,
            event_callback=(
                callback if args.event_log in {"full", "committed_success"}
                else None
            ),
            round_callback=round_progress,
            retry_callback=(
                retry_progress if sample_update_mode is not None else None
            ),
            trainable_trunk_layers=args.trainable_trunk_layers,
            resume_from=args.resume_from,
        )
    except Exception as error:
        if not output_was_unsafe:
            args.output.mkdir(parents=True, exist_ok=True)
            if args.event_log in {"full", "committed_success"}:
                failure_events = (
                    events + pending_events
                    if args.event_log == "committed_success" else events
                )
                torch.save(failure_events, args.output / "FAILED_events.pt")
            failure = {
                "status": "EXPANSION_FAILED_CLOSED",
                "error_type": type(error).__name__,
                "error": str(error),
                "event_count": (
                    len(events) + len(pending_events)
                    if args.event_log == "committed_success" else len(events)
                ),
                "elapsed_s": time.perf_counter() - started,
                "fa_alloc_diagnostics": fa_stats,
            }
            if args.event_log == "committed_success":
                failure["event_log_contract"] = {
                    "mode": "committed_success",
                    "source_event_count": source_event_count,
                    "retained_resolved_event_count": len(events),
                    "pending_unresolved_event_count": len(pending_events),
                    "pruned_resolved_event_count": pruned_event_count,
                }
            (args.output / "FAILED.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            _write_reproduction_command(args.output)
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
    manifest["runtime_device"] = str(device)
    if lab_profile:
        for key in tuple(manifest):
            if key.startswith("ball_"):
                del manifest[key]
        history_present = (
            context_contract in LAB_HISTORY_CONTEXT_SCHEMAS
        )
        history_frozen = (
            not any(
                parameter.requires_grad
                for parameter in policy.policy.history_encoder.parameters()
            )
            if history_present
            else None
        )
        visual_encoder = getattr(policy.policy, "grid_encoder", None)
        visual_encoder_present = isinstance(
            visual_encoder, torch.nn.Module,
        )
        visual_encoder_frozen = (
            not any(
                parameter.requires_grad
                for parameter in visual_encoder.parameters()
            )
            if visual_encoder_present
            else None
        )
        if (
            visual_encoder_present
            and visual_encoder_frozen
            != (
                args.freeze_visual_encoder_during_expansion
                or args.head_only_expansion
                or args.trainable_trunk_layers is not None
            )
        ):
            raise RuntimeError(
                "actual visual-encoder freeze state disagrees with the "
                "explicit expansion contract"
            )
        if (
            history_present
            and history_frozen == args.train_gru_during_expansion
        ):
            raise RuntimeError(
                "actual GRU freeze state disagrees with the explicit "
                "expansion contract"
            )
        manifest["task_profile"] = (
            (
                "minhyuk_lab_random_three_sphere_visual_expansion"
                if task.scene_schema == LAB_CLUTTER_SCENE_SCHEMA
                else (
                    "minhyuk_lab_path_focused_variable_sphere_"
                    "visual_expansion"
                )
            )
            if clutter_profile
            else "minhyuk_lab_ball_visual_expansion"
        )
        manifest["lab_conditioning"] = {
            "context_schema": context_contract,
            "policy_context_dim": int(policy.policy_context_dim),
            "exact_previous_raw_and_applied_in_policy": (
                context_contract == LAB_HP100_EXACT_MEMORY_SCHEMA
            ),
            "history_encoder": {
                "present": history_present,
                "frozen_during_expansion": history_frozen,
                "explicit_unfreeze_flag": bool(
                    args.train_gru_during_expansion
                ),
                "history_source": (
                    "prior_10_executed_pre_smoothing_raw_commands_with_"
                    "left_padding_validity_bits"
                    if context_contract in LAB_HISTORY_CONTEXT_SCHEMAS
                    else None
                ),
            },
            "verifier_only_context_suffix": (
                (
                    [
                        "previous_applied_acceleration_3d",
                        "previous_raw_acceleration_3d",
                        "three_spheres_flattened_12d",
                    ]
                    if task.scene_schema == LAB_CLUTTER_SCENE_SCHEMA
                    else [
                        "previous_applied_acceleration_3d",
                        "previous_raw_acceleration_3d",
                        "sphere_count_scalar_1d",
                        (
                            "sphere_rows_zero_padded_to_"
                            f"{task.scene_spec.max_count}x4"
                        ),
                    ]
                )
                if clutter_profile else [
                    "previous_applied_acceleration_3d",
                    "previous_raw_acceleration_3d",
                ]
            ),
            "device": str(device),
            "visual_encoder_and_first_flow_layer_lr_scale": float(
                args.first_layer_lr_scale
            ),
            "visual_encoder": {
                "present": visual_encoder_present,
                "frozen_during_expansion": visual_encoder_frozen,
                "explicit_freeze_flag": bool(
                    args.freeze_visual_encoder_during_expansion
                ),
            },
            "event_context": (
                "7-D low state plus prior raw-command history when present "
                "plus verifier-only dynamics/scene suffix; visual grid "
                "omitted because it is reproducible from robot state and scene"
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
            "step_margin_blend": {
                "weight": float(args.execution_step_margin_weight),
                "formula": "J_native - weight * first_step_nominal_H_P_margin",
                "proximity_cost": "required zero when weight is positive",
            },
            "native_soft_clearance": {
                "configured_weight": float(
                    task_config.safemppi.soft_clearance_weight
                ),
                "configured_target_m": float(
                    task_config.safemppi.soft_clearance_target
                ),
                "effective_weight": float(
                    getattr(
                        task,
                        "execution_soft_clearance_weight",
                        task_config.safemppi.soft_clearance_weight,
                    )
                ),
                "effective_target_m": float(
                    getattr(
                        task,
                        "execution_soft_clearance_target_m",
                        task_config.safemppi.soft_clearance_target,
                    )
                ),
                "formula": "weight * sum_h max(target_m-clearance_h, 0)^2",
                "scope": "execution ranking only; verifier labels unchanged",
            },
            "native_taskspace_exponential": {
                "configured_weight": float(
                    task_config.safemppi.taskspace_exponential_weight
                ),
                "effective_weight": float(
                    getattr(
                        task,
                        "execution_taskspace_weight",
                        task_config.safemppi.taskspace_exponential_weight,
                    )
                ),
                "temperature": float(
                    task_config.safemppi.taskspace_exponential_temperature
                ),
                "scope": (
                    "execution ranking only; verifier first-step taskspace "
                    "gate and rollout OOB termination unchanged"
                ),
            },
            "execution_clearance_exponential": {
                "configured_weight": float(
                    args.execution_clearance_exp_weight
                ),
                "effective_weight": float(
                    args.execution_clearance_exp_weight
                    if clutter_profile and args.execution_rule == "min_cost"
                    else 0.0
                ),
                "temperature": float(
                    args.execution_clearance_exp_temperature
                ),
                "target_m": float(args.execution_clearance_target_m),
                "formula": (
                    "weight * mean_h exp((target_m-clearance_h)/temperature)"
                ),
                "predicted_states": "H post-action plan knots",
                "applies_only_to": "min_cost",
            },
            "obstacle_plane_escape": {
                "configured_weight": float(
                    args.execution_plane_escape_weight
                ),
                "effective_weight": float(
                    getattr(task, "execution_plane_escape_weight", 0.0)
                ),
                "sigma_m": float(args.execution_plane_escape_sigma),
                "gate_radius_m": float(
                    args.execution_plane_escape_gate_radius
                ),
                "plane_z": (
                    float(task.obstacle_plane_z)
                    if hasattr(task, "obstacle_plane_z") else None
                ),
                "plane_center_xy": (
                    [float(value) for value in task.obstacle_plane_center_xy]
                    if hasattr(task, "obstacle_plane_center_xy") else None
                ),
                "formula": (
                    "weight * mean_h exp(-(z_h-z_c)^2/(2 sigma_m^2)) * "
                    "1[||xy_h-xy_c|| <= gate_radius_m]"
                ),
                "predicted_states": "H post-action plan knots",
                "symmetry": (
                    "symmetric in z about the obstacle centre plane; upward "
                    "and downward escape are relieved equally"
                ),
                "gate": (
                    "horizontal pass-by disc around the obstacle centre, so "
                    "the start and goal on the same plane are not penalized"
                ),
                "applies_only_to": "min_cost",
                "scope": (
                    "execution ranking only; verifier labels, replay "
                    "admission, episode termination, and CFM targets unchanged"
                ),
            },
        }
        if args.execution_plane_escape_weight != 0.0:
            manifest["semantics"]["execution_plane_escape"] = (
                "min_cost execution ranking adds a symmetric obstacle-plane "
                "escape term over the nominal plan knots inside the horizontal "
                "pass-by disc; it never changes the safety label, replay "
                "admission, termination, or CFM targets"
            )
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
            "sample_update_mode": (
                list(sample_update_mode)
                if sample_update_mode is not None else None
            ),
            "sample_update_mode_names": {
                "0": "below", "1": "above", "2": "left", "3": "right",
            },
            "sample_update_mode_classifier": (
                "evaluation-identical dense executed path: "
                "trajectory_crossing_theta -> theta_name"
                if sample_update_mode is not None else None
            ),
            "sample_update_retry_guidance": (
                "every batch pairs parallel_episodes guided and unguided "
                "rollouts at the same round-scheduled flow_base_std; only "
                "the guided cohort re-picks within verifier-positive candidates "
                "by progress-aligned successful action-window prototypes; both "
                "cohorts share one route quota and stop together"
                if sample_update_mode is not None else None
            ),
            "above_plane_z": (
                float(task.env.spheres[0, 2])
                if len(task.env.spheres) == 1 else None
            ),
            "selector_changes_safety_label": False,
        }
        (args.output / "task_config_resolved.json").write_text(
            json.dumps(task_config.raw, indent=2) + "\n"
        )
        if clutter_profile:
            manifest["lab_scene_randomization"] = dict(randomization)
            manifest["lab_scene_schema"] = task.scene_schema
            manifest["lab_scene_ledger"] = list(task.scene_ledger)
    manifest["event_log"] = args.event_log
    if args.event_log == "committed_success":
        if pending_events:
            raise RuntimeError(
                "successful expansion returned with unresolved pending event traces"
            )
        if source_event_count != len(events) + pruned_event_count:
            raise RuntimeError(
                "committed-success event accounting does not conserve source events"
            )
        retained_success_episode_count = len({
            (
                int(event["round"]),
                float(event["gamma"]),
                int(event["episode"]),
            )
            for event in events
        })
        committed_episode_count = sum(
            int(detail["committed_trajectory_count"])
            for row in manifest["rounds"]
            for detail in row[
                "successful_executed_commit_by_gamma"
            ].values()
        )
        manifest["event_log_contract"] = {
            "mode": "committed_success",
            "source_event_count": source_event_count,
            "retained_event_count": len(events),
            "pruned_event_count": pruned_event_count,
            "retained_success_episode_count": (
                retained_success_episode_count
            ),
            "committed_episode_count": committed_episode_count,
            "pending_event_count": 0,
            "event_record": (
                "the unchanged complete compact event dict for every K/B/sigma/"
                "verifier/execution step of every resolved terminal-SUCCESS "
                "episode; manifest IDs identify the authoritative plural "
                "committed subset"
            ),
            "sigma_color_normalization": (
                "within-round successful-trace-only"
            ),
        }
    committed_trajectories = sum(
        int(detail["committed_trajectory_count"])
        for row in manifest["rounds"]
        for detail in row["successful_executed_commit_by_gamma"].values()
    )
    expected_trajectories = (
        args.rounds
        * len(task_config.data.gammas)
        * args.successful_trajectories_per_gamma
    )
    if committed_trajectories != expected_trajectories:
        raise RuntimeError(
            "completed expansion trajectory count disagrees with the exact "
            f"quota: {committed_trajectories} != {expected_trajectories}"
        )
    if args.optimizer_steps_total is not None:
        final_optimizer_step = int(manifest["rounds"][-1]["optimizer_step"])
        if final_optimizer_step != args.optimizer_steps_total:
            raise RuntimeError(
                "global optimizer budget did not end exactly on the final "
                f"round: {final_optimizer_step} != {args.optimizer_steps_total}"
            )
    manifest["reserve_g_online_optimization"] = {
        "committed_trajectories": committed_trajectories,
        "expected_trajectories": expected_trajectories,
        "replay_scope": args.replay_scope,
        "replay_batch_sampler": args.replay_batch_sampler,
        "batch_semantics": (
            "equal mode x gamma mass, then uniform trajectory lineage and "
            "uniform committed window"
        ),
        "optimizer_steps_total": args.optimizer_steps_total,
        "optimizer_step_allocation": args.optimizer_step_allocation,
        "round_learning_rate_warmup_power": (
            args.round_learning_rate_warmup_power
        ),
        "optimizer_steps_final": int(
            manifest["rounds"][-1]["optimizer_step"]
        ),
        "flow_nfe": args.flow_nfe,
        "batched_rollout_sampling": args.batched_rollout_sampling,
        "retry_query_count_B": args.retry_B,
        "retry_exhaustion_policy": args.retry_exhaustion_policy,
        "retry_resample_batch_cap": args.retry_resample_batch_cap,
        "sample_archive_contract": (
            "query_archive_round_NNN.pt is written atomically after every "
            "committed round; query_archive.pt merges all committed rounds"
        ),
        "trainable_trunk_layers": args.trainable_trunk_layers,
        "trainable_parameter_count": manifest["optimizer_scope"][
            "trainable_parameter_count"
        ],
    }
    manifest["fa_alloc_diagnostics"] = fa_stats
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if args.event_log in {"full", "committed_success"}:
        torch.save(events, args.output / "events.pt")
    _write_reproduction_command(args.output)
    print(f"[expansion] rounds={args.rounds} events={len(events)} "
          f"D={manifest['D']} D+={manifest['D_plus']} "
          f"D_accept={manifest['D_replay_accepted']} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    for row in manifest["rounds"]:
        print(format_round_summary(row), flush=True)


if __name__ == "__main__":
    main()
