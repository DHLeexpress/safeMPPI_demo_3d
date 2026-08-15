"""Task-agnostic, single-arm Safe Flow Expansion reference.

The task adapter owns dynamics, context construction, nominal ``H_P``, the full
verifier, and the execution cost.  This file owns only the B1 expansion loop:
RBF uncertainty, budgeted acquisition, fail-closed execution, recent positive
replay, and optional signed negative gradients.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch

# RESEARCH hook: when a callable is installed here, the min_cost execution
# branch offers it the chance to re-pick among the verified,
# progress-eligible candidates of the current step (see call site). None --
# the permanent default -- preserves the original behavior exactly.
EXECUTION_SELECTION_HOOK: Callable[..., int | None] | None = None

# RESEARCH hook: an experiment may replace some freshly sampled candidates
# before acquisition and verification.  The hook must return a finite tensor
# with exactly the same shape, dtype, and device.  None -- the permanent
# default -- leaves candidate generation byte-for-byte unchanged.
CANDIDATE_PROPOSAL_HOOK: Callable[..., torch.Tensor | None] | None = None

_RESUME_STATE_VERSION = 1
_RESUME_STATE_NAME = "resume_state_latest.pt"
_RESUME_METADATA_NAME = "resume_state.json"
_RETRY_FAST_PATH_EXECUTION_RULES = frozenset({
    "min_cost", "exponential_cost", "quadratic_cost", "max_margin",
    "max_step_margin",
})
_RESUME_MUTABLE_CONFIG_FIELDS = frozenset({
    # These affect only future rollout gathering.  Keeping them mutable is
    # intentional: a stuck quota search can resume from the last committed
    # round with a different sampling/guidance strategy without discarding
    # the learned model or Adam momentum.
    "parallel_episodes", "verifier_workers", "max_retry_batches",
    "K", "B", "retry_B", "retry_verify_all_fast_path", "flow_base_std",
    "beta",
    "candidate_perturb_std",
    "candidate_perturb_scope", "batched_rollout_sampling", "flow_nfe",
    "retry_exhaustion_policy", "retry_resample_batch_cap",
    # Scene replacement only changes which future, not-yet-committed rollout
    # pair is sampled.  It does not alter replay rows, optimizer state, or any
    # already committed round artifact.
    "paired_scene_replace_after_retry_batches",
    "paired_scene_max_replacements_per_slot",
    "paired_scene_replace_interval_batches",
})


def bounded_regret_step_margin_choice(
    costs: Sequence[float],
    step_margins: Sequence[float],
    fraction: float,
) -> int:
    """Choose a nominal-margin leader inside a bounded execution-cost band."""
    if len(costs) < 1 or len(costs) != len(step_margins):
        raise ValueError("costs and step_margins must have equal nonzero length")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0,1]")
    cost_values = [float(value) for value in costs]
    margin_values = [float(value) for value in step_margins]
    if not all(map(math.isfinite, cost_values + margin_values)):
        raise ValueError("costs and step_margins must be finite")
    minimum = min(cost_values)
    threshold = minimum + fraction * (max(cost_values) - minimum)
    shortlist = [
        index for index, value in enumerate(cost_values)
        if value <= threshold
    ]
    return max(
        shortlist,
        key=lambda index: (margin_values[index], -cost_values[index]),
    )


def _successful_trajectory_quota_for(
    config: "ExpansionConfig", round_i: int, gamma: float,
) -> int:
    """Return the exact even mirrored quota for one round/gamma cell."""
    if config.successful_trajectories_per_round is None:
        return int(config.successful_trajectories_per_gamma)
    if int(round_i) < 1:
        raise ValueError("scheduled trajectory quota requires round >= 1")
    gamma_indices = [
        index for index, value in enumerate(config.gammas)
        if math.isclose(float(value), float(gamma), rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(gamma_indices) != 1:
        raise ValueError(f"gamma {float(gamma):g} is outside the configured grid")
    gamma_count = len(config.gammas)
    total_pairs = int(config.successful_trajectories_per_round) // 2
    base_pairs, extra_pairs = divmod(total_pairs, gamma_count)
    first_extra = ((int(round_i) - 1) * extra_pairs) % gamma_count
    high_indices = {
        (first_extra + offset) % gamma_count
        for offset in range(extra_pairs)
    }
    return 2 * (base_pairs + int(gamma_indices[0] in high_indices))


def _sample_update_modes_for(
    config: "ExpansionConfig", round_i: int, gamma: float,
) -> tuple[int, ...] | None:
    """Return the active prefix of exact pair labels for a scheduled cell."""
    if config.sample_update_mode is None:
        return None
    quota = _successful_trajectory_quota_for(config, round_i, gamma)
    return tuple(config.sample_update_mode[:quota])


@dataclass(frozen=True)
class ExpansionConfig:
    rounds: int = 10
    gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    parallel_episodes: int = 2
    verifier_workers: int = 1
    max_retry_batches: int = 4
    max_steps: int = 100
    K: int = 16
    B: int = 4
    retry_B: int | None = None
    retry_verify_all_fast_path: bool = False
    batch_size: int = 32
    inner_steps: int | None = None  # None = one exact pass over every eligible D+ row.
    optimizer_steps_total: int | None = None
    optimizer_step_allocation: str = "uniform"
    replay_passes: int = 1
    microbatch_repeats: int = 1
    learning_rate: float = 3.0e-5
    learning_rate_final: float | None = None
    learning_rate_decay_steps: int | None = None
    round_learning_rate_warmup_power: float = 0.0
    gradient_clip_norm: float | None = None
    first_layer_lr_scale: float = 1.0
    freeze_visual_encoder: bool = False
    head_only_update: bool = False
    replay_rounds: int = 2
    replay_scope: str = "sliding"
    replay_batch_sampler: str = "row_permutation"
    gp_buffer_cap: int = 256
    gp_noise: float = 1.0e-2
    rbf_lengthscale: float | None = None
    beta: float = 0.05
    adaptive_beta: bool = False
    ess_target: float = 0.5
    negative_alpha: float = 0.0
    replay_top_fraction: float = 1.0
    replay_selector: str = "sigma_top"
    replay_context_quota: int = 4
    replay_action_weight: float = 0.5
    replay_cluster_count: int = 32
    flow_base_std: float = 1.0
    candidate_perturb_std: float = 0.0
    candidate_perturb_scope: str = "coherent_horizon"
    execution_rule: str = "min_cost"
    execution_step_margin_weight: float = 0.0
    execution_cost_band_fraction: float = 0.0
    execution_ess_target: float = 0.25
    execution_uncertainty_weight: float = 1.0
    archive_rule: str = "all_queries"
    successful_trajectory_selector: str = "lowest_episode_id"
    successful_trajectories_per_gamma: int = 1
    successful_trajectories_per_round: int | None = None
    sample_update_mode: tuple[int, ...] | None = None
    sample_update_cohorts: str = "paired"
    paired_scene_replace_after_retry_batches: int | None = None
    paired_scene_replace_interval_batches: int = 1
    paired_scene_max_replacements_per_slot: int = 0
    replay_acceptance: str = "execution_eligible"
    paired_noised_representation: bool = False
    batched_rollout_sampling: bool = False
    flow_nfe: int | None = None
    seed: int = 0
    acquisition_feature: str = "learned_phi"
    coverage_replay: str = "none"
    replay_augmentation: str = "none"
    target_gate_start_round: int | None = None
    gp_reference_mode: str = "recent_current_phi"
    gp_sliding_row_selector: str = "trajectory_uniform"
    gp_exact_max_rows_per_gamma: int = 1024
    retry_exhaustion_policy: str = "abort"
    retry_resample_batch_cap: int = 512

    def validate(self) -> None:
        if (self.rounds < 1 or self.parallel_episodes < 1
                or self.verifier_workers < 1
                or self.max_retry_batches < 1 or self.max_steps < 1):
            raise ValueError(
                "rounds, parallel_episodes, verifier_workers, "
                "max_retry_batches, and max_steps must be positive"
            )
        if not (1 <= self.B <= self.K):
            raise ValueError("require 1 <= B <= K")
        if self.retry_B is not None and not (1 <= self.retry_B <= self.K):
            raise ValueError("require 1 <= retry_B <= K")
        if self.retry_verify_all_fast_path and self.retry_B != self.K:
            raise ValueError(
                "retry_verify_all_fast_path requires retry_B == K"
            )
        if self.retry_verify_all_fast_path and (
            self.adaptive_beta
            or self.replay_selector != "uniform"
            or self.execution_rule not in _RETRY_FAST_PATH_EXECUTION_RULES
        ):
            raise ValueError(
                "retry_verify_all_fast_path requires fixed beta, uniform "
                "replay selection, and a deterministic execution rule that "
                "does not consume acquisition uncertainty"
            )
        if (self.batch_size < 1 or self.replay_passes < 1
                or self.microbatch_repeats < 1
                or (self.inner_steps is not None and self.inner_steps < 1)):
            raise ValueError(
                "batch_size, replay_passes, microbatch_repeats, and an "
                "explicit inner_steps must be positive"
            )
        if self.optimizer_steps_total is not None and (
            self.optimizer_steps_total < self.rounds
            or self.inner_steps is not None
        ):
            raise ValueError(
                "optimizer_steps_total must provide at least one step per "
                "round and cannot be combined with a per-round inner_steps cap"
            )
        if self.optimizer_step_allocation not in {
            "uniform", "linear_round", "quadratic_round",
        }:
            raise ValueError(
                "optimizer_step_allocation must be uniform, linear_round, "
                "or quadratic_round"
            )
        if self.replay_passes > 1 and self.inner_steps is not None:
            raise ValueError(
                "replay_passes>1 cannot be combined with an inner_steps cap; "
                "use uncapped exact replay passes"
            )
        if self.replay_rounds < 1 or self.gp_buffer_cap < 2:
            raise ValueError("replay_rounds must be positive and gp_buffer_cap at least two")
        if self.replay_scope not in {"sliding", "cumulative"}:
            raise ValueError("replay_scope must be sliding or cumulative")
        if self.replay_batch_sampler not in {
            "row_permutation", "mode_gamma_stratified",
        }:
            raise ValueError(
                "replay_batch_sampler must be row_permutation or "
                "mode_gamma_stratified"
            )
        if self.replay_batch_sampler == "mode_gamma_stratified":
            if (
                self.archive_rule != "successful_executed_windows"
                or self.sample_update_mode is None
            ):
                raise ValueError(
                    "mode_gamma_stratified replay requires exact "
                    "sample_update_mode successful-window archives"
                )
            stratum_count = len(set(self.sample_update_mode)) * len(
                self.gammas
            )
            if self.batch_size < stratum_count:
                raise ValueError(
                    "mode_gamma_stratified batch_size must cover all "
                    "requested label x gamma strata"
                )
            if self.replay_passes != 1 or self.microbatch_repeats != 1:
                raise ValueError(
                    "mode_gamma_stratified replay requires one replay pass "
                    "and one update per sampled batch"
                )
            if self.inner_steps is None and self.optimizer_steps_total is None:
                raise ValueError(
                    "mode_gamma_stratified replay requires an explicit "
                    "per-round or total optimizer-step budget"
                )
        if (
            self.optimizer_steps_total is not None
            and self.replay_batch_sampler != "mode_gamma_stratified"
        ):
            raise ValueError(
                "optimizer_steps_total requires mode_gamma_stratified replay"
            )
        if self.flow_nfe is not None and self.flow_nfe < 1:
            raise ValueError("flow_nfe must be positive")
        if self.gp_noise <= 0.0 or self.beta <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("gp_noise, beta, and learning_rate must be positive")
        if (self.learning_rate_final is None) != (
            self.learning_rate_decay_steps is None
        ):
            raise ValueError(
                "learning_rate_final and learning_rate_decay_steps must be set together"
            )
        if self.learning_rate_final is not None and not (
            0.0 <= self.learning_rate_final <= self.learning_rate
            and self.learning_rate_decay_steps is not None
            and self.learning_rate_decay_steps >= 1
        ):
            raise ValueError(
                "learning_rate_final must lie in [0, learning_rate] and "
                "learning_rate_decay_steps must be positive"
            )
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if (
            not math.isfinite(self.round_learning_rate_warmup_power)
            or self.round_learning_rate_warmup_power < 0.0
        ):
            raise ValueError(
                "round_learning_rate_warmup_power must be finite and nonnegative"
            )
        if not 0.0 < self.first_layer_lr_scale <= 1.0:
            raise ValueError("first_layer_lr_scale must lie in (0,1]")
        if not 0.0 < self.ess_target <= 1.0 or self.negative_alpha < 0.0:
            raise ValueError("ess_target must lie in (0,1] and negative_alpha be nonnegative")
        if self.adaptive_beta and self.ess_target < 1.0 / self.K:
            raise ValueError(
                "adaptive ESS target is infeasible: normalized ESS is at "
                f"least 1/K={1.0 / self.K:g} for K={self.K}"
            )
        if not 0.0 < self.replay_top_fraction <= 1.0:
            raise ValueError("replay_top_fraction must lie in (0,1]")
        if self.replay_selector not in {
                "sigma_top", "uniform", "context_kcenter", "cluster_balanced"}:
            raise ValueError(
                "replay_selector must be sigma_top, uniform, context_kcenter, "
                "or cluster_balanced"
            )
        if (self.replay_context_quota < 1 or self.replay_cluster_count < 1
                or self.replay_action_weight < 0.0):
            raise ValueError(
                "replay quotas/counts must be positive and replay_action_weight nonnegative"
            )
        if not math.isfinite(self.flow_base_std) or self.flow_base_std < 0.0:
            raise ValueError("flow_base_std must be finite and nonnegative")
        if self.candidate_perturb_std < 0.0:
            raise ValueError("candidate_perturb_std must be nonnegative")
        if self.candidate_perturb_scope not in {"first_action", "coherent_horizon"}:
            raise ValueError("candidate_perturb_scope must be first_action or coherent_horizon")
        if self.execution_rule not in {
                "min_cost", "exponential_cost", "quadratic_cost",
                "max_margin", "uniform_positive", "softmin_cost",
                "max_uncertainty", "uncertainty_cost", "max_step_margin"}:
            raise ValueError(
                "execution_rule must be min_cost, exponential_cost, "
                "quadratic_cost, max_margin, uniform_positive, softmin_cost, "
                "max_uncertainty, uncertainty_cost, or max_step_margin"
            )
        if (
            not math.isfinite(self.execution_step_margin_weight)
            or self.execution_step_margin_weight < 0.0
        ):
            raise ValueError(
                "execution_step_margin_weight must be finite and nonnegative"
            )
        if (
            self.execution_step_margin_weight > 0.0
            and self.execution_rule != "min_cost"
        ):
            raise ValueError(
                "execution_step_margin_weight requires execution_rule=min_cost"
            )
        if (
            not math.isfinite(self.execution_cost_band_fraction)
            or not 0.0 <= self.execution_cost_band_fraction <= 1.0
        ):
            raise ValueError(
                "execution_cost_band_fraction must lie in [0,1]"
            )
        if (
            self.execution_cost_band_fraction > 0.0
            and self.execution_rule != "min_cost"
        ):
            raise ValueError(
                "execution_cost_band_fraction requires execution_rule=min_cost"
            )
        if (
            self.execution_cost_band_fraction > 0.0
            and self.execution_step_margin_weight > 0.0
        ):
            raise ValueError(
                "execution cost-band and linear step-margin blend are "
                "mutually exclusive"
            )
        if not 0.0 < self.execution_ess_target <= 1.0:
            raise ValueError("execution_ess_target must lie in (0,1]")
        if self.execution_uncertainty_weight < 0.0:
            raise ValueError("execution_uncertainty_weight must be nonnegative")
        if self.archive_rule not in {
                "all_queries", "executed_only", "executed_plus_nvp_negative",
                "successful_executed_windows"}:
            raise ValueError("archive_rule must be all_queries, executed_only, or "
                             "executed_plus_nvp_negative, or "
                             "successful_executed_windows")
        if self.archive_rule == "successful_executed_windows":
            if self.negative_alpha != 0.0:
                raise ValueError(
                    "successful_executed_windows requires negative_alpha=0"
                )
            if self.target_gate_start_round is not None:
                raise ValueError(
                    "successful_executed_windows requires an inactive target gate"
                )
        if self.successful_trajectory_selector not in {
                "lowest_episode_id", "random_success",
                "max_above_fraction", "max_mean_z"}:
            raise ValueError(
                "successful_trajectory_selector must be lowest_episode_id, "
                "random_success, max_above_fraction, or max_mean_z"
            )
        if self.successful_trajectories_per_gamma < 1:
            raise ValueError(
                "successful_trajectories_per_gamma must be positive"
            )
        if self.successful_trajectories_per_round is not None:
            if (
                self.sample_update_mode is None
                or self.successful_trajectories_per_round < 2 * len(self.gammas)
                or self.successful_trajectories_per_round % 2
            ):
                raise ValueError(
                    "successful_trajectories_per_round requires exact pair "
                    "labels and an even total of at least two per gamma"
                )
            maximum = max(
                _successful_trajectory_quota_for(self, round_i, gamma)
                for round_i in range(1, len(self.gammas) + 1)
                for gamma in self.gammas
            )
            if maximum > self.successful_trajectories_per_gamma:
                raise ValueError(
                    "successful_trajectories_per_gamma must cover the maximum "
                    "scheduled per-cell quota"
                )
        cohort_multiplier = (
            2
            if self.sample_update_mode is not None
            and self.sample_update_cohorts == "paired"
            else 1
        )
        if (
            self.successful_trajectories_per_gamma
            > self.parallel_episodes * self.max_retry_batches
            * cohort_multiplier
        ):
            raise ValueError(
                "successful_trajectories_per_gamma cannot exceed "
                "the available episodes across retry batches"
            )
        if (
            self.archive_rule != "successful_executed_windows"
            and self.successful_trajectories_per_gamma != 1
        ):
            raise ValueError(
                "successful_trajectories_per_gamma only applies to "
                "archive_rule=successful_executed_windows"
            )
        if self.sample_update_mode is not None:
            if self.archive_rule != "successful_executed_windows":
                raise ValueError(
                    "sample_update_mode requires "
                    "archive_rule=successful_executed_windows"
                )
            if len(self.sample_update_mode) != self.successful_trajectories_per_gamma:
                raise ValueError(
                    "sample_update_mode must contain exactly "
                    "successful_trajectories_per_gamma entries"
                )
            if any(
                not isinstance(mode, int) or mode < 0
                for mode in self.sample_update_mode
            ):
                raise ValueError(
                    "sample_update_mode entries must be nonnegative integers"
                )
        if self.sample_update_cohorts not in {"paired", "unguided_only"}:
            raise ValueError(
                "sample_update_cohorts must be paired or unguided_only"
            )
        if (
            self.sample_update_mode is None
            and self.sample_update_cohorts != "paired"
        ):
            raise ValueError(
                "sample_update_cohorts=unguided_only requires "
                "sample_update_mode"
            )
        replacement_enabled = (
            self.paired_scene_replace_after_retry_batches is not None
            or self.paired_scene_max_replacements_per_slot > 0
        )
        if replacement_enabled:
            if (
                self.paired_scene_replace_after_retry_batches is None
                or self.paired_scene_replace_after_retry_batches < 1
                or self.paired_scene_replace_interval_batches < 1
                or self.paired_scene_max_replacements_per_slot < 1
            ):
                raise ValueError(
                    "paired scene replacement requires a positive retry-batch "
                    "threshold/interval and positive per-slot replacement limit"
                )
            expected_labels = tuple(range(len(self.sample_update_mode or ())))
            if (
                len(expected_labels) < 2
                or len(expected_labels) % 2
                or self.sample_update_mode != expected_labels
            ):
                raise ValueError(
                    "paired scene replacement requires contiguous even exact "
                    "labels 0..Q-1"
                )
        elif self.paired_scene_max_replacements_per_slot < 0:
            raise ValueError(
                "paired_scene_max_replacements_per_slot must be nonnegative"
            )
        if self.retry_exhaustion_policy not in {
            "abort", "resample_scene", "commit_available",
        }:
            raise ValueError(
                "retry_exhaustion_policy must be abort, resample_scene, or "
                "commit_available"
            )
        if (
            self.retry_exhaustion_policy == "resample_scene"
            and self.retry_resample_batch_cap < self.max_retry_batches
        ):
            raise ValueError(
                "retry_resample_batch_cap must be at least max_retry_batches"
            )
        if self.replay_acceptance not in {"execution_eligible", "safety_valid"}:
            raise ValueError(
                "replay_acceptance must be execution_eligible or safety_valid"
            )
        if (self.replay_acceptance == "safety_valid"
                and self.archive_rule != "all_queries"):
            raise ValueError(
                "safety_valid replay_acceptance requires archive_rule=all_queries "
                "so every successfully evaluated selected-B query is archived"
            )
        if self.acquisition_feature not in {"learned_phi", "task_angular"}:
            raise ValueError("acquisition_feature must be learned_phi or task_angular")
        if self.coverage_replay not in {"none", "circular_voronoi"}:
            raise ValueError("coverage_replay must be none or circular_voronoi")
        if self.replay_augmentation not in {"none", "task_d4"}:
            raise ValueError("replay_augmentation must be none or task_d4")
        if (self.target_gate_start_round is not None
                and self.target_gate_start_round < 1):
            raise ValueError("target_gate_start_round must be positive")
        if self.gp_reference_mode not in {
                "recent_current_phi", "round1_fixed_frozen_phi",
                "cumulative_accepted_frozen_phi",
                "cumulative_success_per_gamma_frozen_phi_exact",
                "sliding_success_per_gamma_frozen_phi",
                "sliding_success_per_gamma_current_phi"}:
            raise ValueError(
                "gp_reference_mode must be recent_current_phi or "
                "round1_fixed_frozen_phi or cumulative_accepted_frozen_phi or "
                "cumulative_success_per_gamma_frozen_phi_exact or "
                "sliding_success_per_gamma_frozen_phi or "
                "sliding_success_per_gamma_current_phi"
            )
        if self.gp_reference_mode == "cumulative_accepted_frozen_phi":
            gamma_count = len(self.gammas)
            if (self.acquisition_feature != "learned_phi"
                    or self.gp_buffer_cap % (2 * gamma_count) != 0
                    or self.gp_buffer_cap < 4 * gamma_count):
                raise ValueError(
                    "cumulative_accepted_frozen_phi requires learned_phi and a "
                    "gp_buffer_cap divisible by twice the gamma count, with at "
                    "least four rows per gamma"
                )
        if self.gp_reference_mode in {
            "cumulative_success_per_gamma_frozen_phi_exact",
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
        }:
            if (
                self.archive_rule != "successful_executed_windows"
                or self.acquisition_feature != "learned_phi"
            ):
                raise ValueError(
                    "successful per-gamma learned-phi GP modes require "
                    "successful_executed_windows and learned_phi"
                )
        if self.gp_reference_mode in {
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
        }:
            gamma_count = len(self.gammas)
            if (
                self.gp_buffer_cap % gamma_count != 0
                or self.gp_buffer_cap // gamma_count < 2
            ):
                raise ValueError(
                    "sliding successful-window GP modes require a total "
                    "gp_buffer_cap divisible by the gamma count and at least "
                    "two rows per gamma"
                )
        if self.gp_sliding_row_selector not in {
            "trajectory_uniform", "fifo_tail",
        }:
            raise ValueError(
                "gp_sliding_row_selector must be trajectory_uniform or fifo_tail"
            )
        if self.gp_exact_max_rows_per_gamma < 1:
            raise ValueError("gp_exact_max_rows_per_gamma must be positive")


@dataclass(frozen=True)
class Verification:
    """One successfully evaluated full-verifier result.

    ``valid`` is the full-H label stored in D+. ``hp_eligible`` is retained as
    a nominal one-step diagnostic but is not an execution gate.
    ``progress_eligible`` is a separate performance gate. In the ball task it
    requires positive x progress at every plan knot, without using the
    unexecuted tail's distance to the goal.
    ``execution_cost`` ranks candidates satisfying ``valid`` and
    ``progress_eligible``.
    """

    valid: bool
    hp_eligible: bool
    margin: float
    execution_cost: float
    progress: float = 0.0
    progress_eligible: bool = True
    error: bool = False
    target_eligible: bool = True
    step_margin: float | None = None


@dataclass
class QueryRecord:
    round: int
    gamma: float
    episode: int
    context_id: int
    context: torch.Tensor
    candidate: torch.Tensor
    verification: Verification
    acquisition_sigma: float = 0.0
    flow_base: torch.Tensor | None = None
    nvp_context: bool = False
    acquisition_feature: torch.Tensor | None = None
    replay_eligible: bool = True
    retry_batch: int | None = None
    replica: int | None = None
    trajectory_id: str | None = None
    window_id: str | None = None
    window_start: int | None = None
    valid_horizon: int | None = None
    loss_mask: torch.Tensor | None = None
    sample_update_mode: int | None = None


def _incomplete_pair_slots(
    label_commit_capable_success_counts: Sequence[int],
    replacement_counts: dict[int, int],
    max_replacements_per_slot: int,
) -> list[int]:
    """Return bounded pair slots missing either commit-capable member."""
    counts = [int(value) for value in label_commit_capable_success_counts]
    if len(counts) < 2 or len(counts) % 2 or any(value < 0 for value in counts):
        raise ValueError(
            "paired commit-capable counts must be even/nonnegative"
        )
    if int(max_replacements_per_slot) < 0:
        raise ValueError("paired replacement limit must be nonnegative")
    return [
        slot
        for slot in range(len(counts) // 2)
        if not (
            counts[2 * slot] > 0
            and counts[2 * slot + 1] > 0
        )
        and int(replacement_counts.get(slot, 0))
        < int(max_replacements_per_slot)
    ]


def _pair_replacement_interval_elapsed(
    previous_completed_batch: int | None,
    completed_batch: int,
    interval_batches: int,
) -> bool:
    """Gate repeated scene replacement without delaying the first one."""
    if int(completed_batch) < 0 or int(interval_batches) < 1:
        raise ValueError("replacement batch/interval values are invalid")
    return (
        previous_completed_batch is None
        or int(completed_batch) - int(previous_completed_batch)
        >= int(interval_batches)
    )


def _invalidate_pair_slot_success_candidates(
    episodes: Sequence[dict[str, Any]],
    pair_slot: int,
    mode_getter: Callable[[dict[str, Any]], int | None],
    replacement_record: dict[str, Any],
) -> tuple[int, int]:
    """Discard both members' accumulated terminal/commit candidates."""
    labels = {2 * int(pair_slot), 2 * int(pair_slot) + 1}
    terminal = commit_capable = 0
    for episode in episodes:
        if (
            episode.get("status") != "SUCCESS"
            or episode.get("paired_scene_quota_invalidated") is not None
        ):
            continue
        mode = mode_getter(episode)
        if mode not in labels:
            continue
        terminal += 1
        was_commit_capable = bool(episode.get("commit_capable", False))
        commit_capable += int(was_commit_capable)
        episode["paired_scene_quota_invalidated"] = {
            "pair_slot": int(pair_slot),
            "old_version": int(replacement_record["old_version"]),
            "new_version": int(replacement_record["new_version"]),
        }
        episode["commit_capable_before_pair_replacement"] = (
            was_commit_capable
        )
        episode["commit_capable"] = False
    return terminal, commit_capable


def _successful_trajectory_id(round_i: int, gamma: float, episode: int) -> str:
    return f"r{round_i:04d}:g{gamma:.9g}:e{episode:06d}"


def _successful_window_id(trajectory_id: str, start: int) -> str:
    return f"{trajectory_id}:w{start:06d}"


def _counter_seed(master_seed: int, *coordinates: int | str) -> int:
    """Stable order-independent seed for success-commit gathering CRNs."""
    digest = hashlib.sha256()
    digest.update(str(int(master_seed)).encode())
    for coordinate in coordinates:
        digest.update(b":")
        digest.update(str(coordinate).encode())
    return int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)


class ExpansionPolicy(Protocol):
    def parameters(self): ...
    def sample(self, context: torch.Tensor, count: int,
               generator: torch.Generator,
               base_std: float = 1.0) -> torch.Tensor: ...
    def sample_with_base(self, context: torch.Tensor, count: int,
                         generator: torch.Generator,
                         base_std: float = 1.0,
                         ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def embed(self, context: torch.Tensor, candidates: torch.Tensor,
              flow_time: float = 0.9,
              base: torch.Tensor | None = None) -> torch.Tensor: ...
    def cfm_loss(self, contexts: torch.Tensor, candidates: torch.Tensor,
                 reduction: str = "none",
                 loss_mask: torch.Tensor | None = None) -> torch.Tensor: ...
    def state_dict(self): ...


class ExpansionTask(Protocol):
    def reset(self, gamma: float, episode: int, seed: int) -> Any: ...
    def context(self, state: Any, gamma: float) -> torch.Tensor: ...
    def verify(self, context: torch.Tensor, candidates: torch.Tensor,
               gamma: float) -> Sequence[Verification]: ...
    def advance(self, state: Any, candidate: torch.Tensor) -> Any: ...
    def terminal(self, state: Any) -> str | None: ...
    def successful_trajectory_above_fraction(
        self, executed_states: Sequence[Any]
    ) -> float: ...
    def successful_trajectory_mean_z(
        self, executed_states: Sequence[Any]
    ) -> float: ...
    def successful_trajectory_mode(
        self, executed_states: Sequence[Any]
    ) -> int | None: ...


_VERIFY_WORKER_TASK: ExpansionTask | None = None


def _initialize_verify_worker(task: ExpansionTask) -> None:
    """Install one immutable task per persistent CPU verifier process."""
    global _VERIFY_WORKER_TASK
    _VERIFY_WORKER_TASK = task
    torch.set_num_threads(1)


def _verify_worker(
    payload: tuple[np.ndarray, np.ndarray, float],
) -> tuple[Verification, ...]:
    """Evaluate one active context in a worker without policy/GP state."""
    if _VERIFY_WORKER_TASK is None:
        raise RuntimeError("verifier worker task was not initialized")
    context, candidates, gamma = payload
    return tuple(_VERIFY_WORKER_TASK.verify(
        torch.from_numpy(context),
        torch.from_numpy(candidates),
        float(gamma),
    ))


class _OrderedVerifier:
    """Serial or spawn-process verifier with deterministic input ordering."""

    def __init__(self, task: ExpansionTask, workers: int):
        self.task = task
        self.workers = int(workers)
        self.pool: ProcessPoolExecutor | None = None
        if self.workers > 1:
            try:
                pickle.dumps(task)
            except Exception as error:
                raise TypeError(
                    "verifier_workers>1 requires a pickleable, CPU-safe task"
                ) from error

    def __enter__(self) -> "_OrderedVerifier":
        return self

    def __exit__(self, *_exc_info) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def verify_many(
        self,
        blocks: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
    ) -> list[tuple[Verification, ...]]:
        if self.workers == 1:
            return [
                tuple(self.task.verify(context, candidates, gamma))
                for context, candidates, gamma in blocks
            ]
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_verify_worker,
                initargs=(self.task,),
            )
        payloads = [
            (
                np.ascontiguousarray(context.detach().cpu().numpy()),
                np.ascontiguousarray(candidates.detach().cpu().numpy()),
                float(gamma),
            )
            for context, candidates, gamma in blocks
        ]
        return list(self.pool.map(_verify_worker, payloads, chunksize=1))


def _normalize(values: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def _stable_accumulation_dtype(values: torch.Tensor) -> torch.dtype:
    """Use float64 where supported and float32 on Apple MPS."""
    return torch.float32 if values.device.type == "mps" else torch.float64


def mean_pairwise_lengthscale(features: torch.Tensor) -> float:
    # Calibration uses only 50 rows; keeping this reduction on CPU preserves
    # float64 numerics and avoids the unsupported MPS pdist operator.
    values = _normalize(features.detach().cpu().to(torch.float64))
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("RBF calibration needs at least two embedding rows")
    lengthscale = float(torch.pdist(values).mean())
    if not math.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("pretrained embeddings do not define a positive length scale")
    return lengthscale


class RBFPosterior:
    """Exact RBF posterior variance on a finite previous-round support set."""

    def __init__(self, lengthscale: float, noise: float):
        if lengthscale <= 0.0 or noise <= 0.0:
            raise ValueError("RBF lengthscale and observation noise must be positive")
        self.lengthscale, self.noise = float(lengthscale), float(noise)
        self.X: torch.Tensor | None = None
        self.L: torch.Tensor | None = None

    @staticmethod
    def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a * a).sum(1, keepdim=True) + (b * b).sum(1)[None]
                - 2.0 * a @ b.T).clamp_min(0.0)

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sqdist(a, b) / (2.0 * self.lengthscale ** 2))

    @torch.no_grad()
    def set_buffer(self, features: torch.Tensor | None) -> None:
        if features is None or len(features) == 0:
            self.X = self.L = None
            return
        stored = (
            features.detach().cpu()
            if features.device.type == "mps" else features.detach()
        )
        self.X = _normalize(stored)
        factor_dtype = torch.float64
        kernel = self.kernel(self.X, self.X).to(factor_dtype)
        eye = torch.eye(
            len(kernel), dtype=factor_dtype, device=kernel.device,
        )
        jitter = self.noise
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(kernel + jitter * eye)
            if int(info.max()) == 0:
                self.L = factor
                return
            jitter *= 10.0
        raise RuntimeError("RBF posterior Cholesky failed")

    @torch.no_grad()
    def covariance(self, features: torch.Tensor) -> torch.Tensor:
        query = _normalize(
            features.detach().cpu()
            if features.device.type == "mps" else features.detach()
        )
        covariance = self.kernel(query, query)
        if self.X is not None:
            cross = self.kernel(query, self.X)
            solved = torch.cholesky_solve(cross.T.to(self.L.dtype), self.L)
            covariance -= cross @ solved.to(cross.dtype)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += self.noise * torch.eye(
            len(query), dtype=query.dtype, device=query.device)
        return covariance

    @torch.no_grad()
    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        diagonal = torch.diagonal(self.covariance(features)) - self.noise
        return diagonal.clamp_min(0.0).sqrt()

    @torch.no_grad()
    def _acquire_from_covariance(
        self,
        covariance: torch.Tensor,
        features: torch.Tensor,
        B: int,
        beta: float,
        generator: torch.Generator,
    ) -> tuple[list[int], list[float], list[float], torch.Tensor]:
        remaining = torch.arange(len(features), device=covariance.device)
        selected, selected_sigma, ess = [], [], []
        initial_sigma = (
            torch.diagonal(covariance) - self.noise
        ).clamp_min(0.0).sqrt()
        for _ in range(B):
            # The Gibbs tilt is defined on posterior standard deviation, not variance.
            # Subtract the observation-noise diagonal added by ``covariance`` so the
            # first draw agrees with ``sigma(features)`` and beta calibration.
            scores = (
                torch.diagonal(covariance) - self.noise
            ).clamp_min(0.0).sqrt()
            weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
            probability = weights / weights.sum()
            if features.device.type == "mps":
                uniform = float(torch.rand(
                    (), device=features.device, generator=generator,
                ).cpu())
                local = min(
                    int(torch.searchsorted(
                        probability.cumsum(0),
                        torch.tensor(uniform, dtype=probability.dtype),
                    )),
                    len(probability) - 1,
                )
            else:
                local = int(torch.multinomial(
                    probability, 1, generator=generator,
                ))
            selected.append(int(remaining[local]))
            selected_sigma.append(float(scores[local]))
            ess.append(float(1.0 / (probability.to(
                _stable_accumulation_dtype(probability)
            ).square().sum()
                                    * probability.numel())))
            keep = torch.ones(len(remaining), dtype=torch.bool, device=remaining.device)
            keep[local] = False
            if not bool(keep.any()):
                break
            cross = covariance[keep, local]
            denominator = covariance[local, local].clamp_min(1.0e-12)
            covariance = covariance[keep][:, keep] - torch.outer(cross, cross) / denominator
            covariance = 0.5 * (covariance + covariance.T)
            remaining = remaining[keep]
        return selected, selected_sigma, ess, initial_sigma

    @torch.no_grad()
    def acquire(self, features: torch.Tensor, B: int, beta: float,
        generator: torch.Generator) -> tuple[list[int], list[float], list[float]]:
        selected, selected_sigma, ess, _ = self._acquire_from_covariance(
            self.covariance(features), features, B, beta, generator,
        )
        return selected, selected_sigma, ess

    @torch.no_grad()
    def acquire_with_sigma(
        self,
        features: torch.Tensor,
        B: int,
        beta: float,
        generator: torch.Generator,
    ) -> tuple[list[int], list[float], list[float], torch.Tensor]:
        """Acquire B points and return the already-computed initial sigma."""
        return self._acquire_from_covariance(
            self.covariance(features), features, B, beta, generator,
        )


def normalized_ess(scores: torch.Tensor, beta: float) -> float:
    weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
    probability = weights / weights.sum()
    return float(1.0 / (probability.to(
        _stable_accumulation_dtype(probability)
    ).square().sum()
                        * probability.numel()))


def calibrate_fixed_beta(score_pools: Sequence[torch.Tensor], target: float,
                         lower: float = 1.0e-5, upper: float = 10.0) -> float:
    """Choose beta once from representative pools; expansion keeps it fixed."""
    if not score_pools or not 0.0 < target <= 1.0:
        raise ValueError("provide score pools and an ESS target in (0,1]")
    for _ in range(60):
        middle = math.sqrt(lower * upper)
        value = float(np.median([normalized_ess(pool, middle) for pool in score_pools]))
        if value < target:
            lower = middle
        else:
            upper = middle
    return math.sqrt(lower * upper)


def _recent(records: Sequence[QueryRecord], round_i: int, width: int,
            *, positive: bool | None = None) -> list[QueryRecord]:
    first = max(0, round_i - width + 1)
    output = [row for row in records if first <= row.round <= round_i]
    if positive is not None:
        output = [row for row in output if row.verification.valid is positive]
    return output


def _top_uncertainty_by_round(
    records: Sequence[QueryRecord],
    fraction: float,
    *,
    group_by_gamma: bool = False,
) -> list[QueryRecord]:
    """Keep causal sigma tails per round, optionally preserving every gamma."""
    if fraction >= 1.0:
        return list(records)
    groups: dict[tuple[int, float | None], list[QueryRecord]] = {}
    for row in records:
        key = (row.round, row.gamma if group_by_gamma else None)
        groups.setdefault(key, []).append(row)
    kept = []
    for key in sorted(groups):
        rows = sorted(
            groups[key], key=lambda row: row.acquisition_sigma, reverse=True
        )
        count = max(1, int(math.ceil(fraction * len(rows))))
        threshold = rows[count - 1].acquisition_sigma
        # A top-fraction ranking is undefined when uncertainties tie.  Include the
        # complete boundary tie rather than turning stable archive order into a
        # hidden gamma/context/acquisition-slot filter.
        kept.extend(row for row in rows if row.acquisition_sigma >= threshold)
    return kept


def _balanced_cap(records: Sequence[QueryRecord], cap: int,
                  rng: np.random.Generator) -> list[QueryRecord]:
    """Round/gamma-balanced sampling without replacement for the GP only."""
    cells: dict[tuple[int, float], list[QueryRecord]] = {}
    for row in records:
        cells.setdefault((row.round, row.gamma), []).append(row)
    chosen: list[QueryRecord] = []
    while len(chosen) < cap and any(cells.values()):
        for key in sorted(cells):
            if len(chosen) == cap:
                break
            cell = cells[key]
            if cell:
                chosen.append(cell.pop(int(rng.integers(len(cell)))))
    return chosen


def _equal_mass_weights(records: Sequence[QueryRecord]) -> torch.Tensor:
    """Equal mass over gamma -> (round, episode) -> context -> positive query."""
    gammas = sorted({row.gamma for row in records})
    lineages: dict[float, set[tuple[int, int]]] = {}
    contexts: dict[tuple[float, int, int], set[int]] = {}
    queries: dict[tuple[float, int, int, int], int] = {}
    for row in records:
        lineage = (row.round, row.episode)
        lineages.setdefault(row.gamma, set()).add(lineage)
        contexts.setdefault((row.gamma, *lineage), set()).add(row.context_id)
        query = (row.gamma, *lineage, row.context_id)
        queries[query] = queries.get(query, 0) + 1
    weights = []
    for row in records:
        lineage = (row.gamma, row.round, row.episode)
        query = (*lineage, row.context_id)
        weights.append(1.0 / (
            len(gammas) * len(lineages[row.gamma]) * len(contexts[lineage]) * queries[query]
        ))
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean()


def _circular_voronoi_weights(
    records: Sequence[QueryRecord],
    lineage_weights: torch.Tensor,
) -> torch.Tensor:
    """Multiply lineage mass by conditional periodic angular Voronoi mass.

    The task supplies a two-dimensional direction descriptor.  Exact duplicate
    directions share one Voronoi cell, so archive multiplicity cannot create
    extra angular mass.  Density correction is performed independently for each
    conditional query context; pooling contexts would let a common direction at
    one state erase uncertainty at another state.  Per-context rescaling keeps
    the original gamma/lineage/context mass unchanged.
    """
    if len(records) != len(lineage_weights):
        raise ValueError("records and lineage_weights must have the same length")
    multipliers = torch.zeros(len(records), dtype=lineage_weights.dtype)
    context_groups: dict[tuple[float, int, int, int], list[int]] = {}
    for index, row in enumerate(records):
        key = (row.gamma, row.round, row.episode, row.context_id)
        context_groups.setdefault(key, []).append(index)
    for key in sorted(context_groups):
        indices = context_groups[key]
        descriptors = []
        directed_indices = []
        for index in indices:
            value = getattr(records[index], "acquisition_feature", None)
            if value is None:
                raise ValueError(
                    "circular_voronoi replay requires a stored 2-D task "
                    "acquisition feature for every replay row"
                )
            descriptor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
            if descriptor.numel() != 2 or not bool(torch.isfinite(descriptor).all()):
                raise ValueError(
                    "circular_voronoi replay requires finite 2-D descriptors"
                )
            if float(descriptor.norm()) > 1.0e-12:
                directed_indices.append(index)
                descriptors.append(descriptor / descriptor.norm())
            else:
                # An exactly axial plan has no defined angle. Keep it at neutral
                # lineage mass instead of assigning an arbitrary route direction.
                multipliers[index] = 1.0
        if not descriptors:
            continue
        directions = torch.stack(descriptors)
        angles = torch.remainder(
            torch.atan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi
        ).cpu().numpy()
        unique_angles, inverse, counts = np.unique(
            angles, return_inverse=True, return_counts=True
        )
        if len(unique_angles) == 1:
            cell_mass = np.asarray([2.0 * math.pi], dtype=np.float64)
        else:
            right_gaps = np.remainder(
                np.roll(unique_angles, -1) - unique_angles, 2.0 * math.pi
            )
            cell_mass = 0.5 * (np.roll(right_gaps, 1) + right_gaps)
        per_row_mass = cell_mass[inverse] / counts[inverse]
        # Unit mean within gamma makes this a density correction rather than a
        # hidden learning-rate change.
        per_row_mass *= len(directed_indices) / per_row_mass.sum()
        multipliers[directed_indices] = torch.as_tensor(
            per_row_mass, dtype=multipliers.dtype
        )

    combined = lineage_weights * multipliers
    for key in sorted(context_groups):
        indices = context_groups[key]
        original_mass = lineage_weights[indices].sum()
        combined_mass = combined[indices].sum()
        if float(combined_mass) <= 0.0:
            raise ValueError("circular Voronoi replay produced zero context mass")
        combined[indices] *= original_mass / combined_mass
    return combined


def _stack(records: Sequence[QueryRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.stack([row.context for row in records]),
            torch.stack([row.candidate for row in records]))


def _stack_loss_masks(records: Sequence[QueryRecord]) -> torch.Tensor | None:
    masks = [getattr(row, "loss_mask", None) for row in records]
    if all(mask is None for mask in masks):
        return None
    values = []
    for row, mask in zip(records, masks):
        if mask is None:
            values.append(torch.ones_like(row.candidate))
        else:
            if tuple(mask.shape) != tuple(row.candidate.shape):
                raise ValueError("replay loss mask must match the stored candidate shape")
            values.append(mask)
    return torch.stack(values)


@torch.no_grad()
def _embed_records(
    policy: ExpansionPolicy,
    records: Sequence[QueryRecord],
    device: torch.device,
) -> torch.Tensor:
    """Re-embed archive rows with their originally paired CFM base noise.

    ``getattr`` keeps archives serialized before ``flow_base`` was introduced
    readable.  Mixed legacy/paired rows are embedded in two batches and then
    restored to archive order.
    """
    if not records:
        raise ValueError("cannot embed an empty record sequence")
    contexts, candidates = (value.to(device) for value in _stack(records))
    bases = [getattr(row, "flow_base", None) for row in records]
    if all(base is None for base in bases):
        return policy.embed(contexts, candidates)
    if all(base is not None for base in bases):
        return policy.embed(
            contexts, candidates,
            base=torch.stack([base for base in bases if base is not None]).to(device),
        )
    output: list[torch.Tensor | None] = [None] * len(records)
    for has_base in (False, True):
        indices = [
            index for index, base in enumerate(bases)
            if (base is not None) is has_base
        ]
        if not indices:
            continue
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        block_contexts = contexts.index_select(0, index_tensor)
        block_candidates = candidates.index_select(0, index_tensor)
        if has_base:
            block = policy.embed(
                block_contexts,
                block_candidates,
                base=torch.stack([
                    bases[index] for index in indices
                    if bases[index] is not None
                ]).to(device),
            )
        else:
            block = policy.embed(block_contexts, block_candidates)
        for local, index in enumerate(indices):
            output[index] = block[local]
    return torch.stack([value for value in output if value is not None])


@torch.no_grad()
def _task_angular_features(
    task: ExpansionTask,
    context: torch.Tensor,
    candidates: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Validate and normalize a task-supplied two-dimensional direction."""
    feature_fn = getattr(task, "acquisition_features", None)
    if callable(feature_fn):
        raw_features = feature_fn(context, candidates, gamma)
    else:
        feature_fn = getattr(task, "angular_descriptors", None)
        if callable(feature_fn):
            raw_features = feature_fn(context, candidates)
        else:
            raw_features = None
    if raw_features is None:
        raise TypeError(
            "task_angular acquisition or circular_voronoi replay requires "
            "task.acquisition_features(context, candidates, gamma) or "
            "task.angular_descriptors(context, candidates)"
        )
    features = torch.as_tensor(
        raw_features,
        dtype=candidates.dtype,
        device=candidates.device,
    ).detach()
    if features.shape != (len(candidates), 2):
        raise ValueError(
            "task.acquisition_features must return one 2-D row per candidate"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("task acquisition features must be finite")
    return _normalize(features)


def _stack_acquisition_features(
    records: Sequence[QueryRecord],
    device: torch.device,
) -> torch.Tensor:
    values = []
    for row in records:
        value = getattr(row, "acquisition_feature", None)
        if value is None:
            raise ValueError(
                "task_angular GP refit requires a stored 2-D acquisition "
                "feature for every positive buffer row"
            )
        values.append(torch.as_tensor(value))
    features = torch.stack(values).to(device)
    if features.shape != (len(records), 2):
        raise ValueError("stored task acquisition features must be two-dimensional")
    return features


def _record_identity_hash(records: Sequence[QueryRecord]) -> str:
    """Stable provenance hash for a frozen GP reference row set."""
    digest = hashlib.sha256()
    for row in records:
        digest.update(
            f"{row.round}:{row.gamma:.9g}:{row.episode}:{row.context_id}\n"
            .encode("utf-8")
        )
        candidate = row.candidate.detach().cpu().contiguous().numpy()
        digest.update(candidate.tobytes(order="C"))
        loss_mask = getattr(row, "loss_mask", None)
        if loss_mask is not None:
            digest.update(loss_mask.detach().cpu().contiguous().numpy().tobytes(order="C"))
        digest.update(f":valid_horizon={getattr(row, 'valid_horizon', None)}\n".encode())
    return digest.hexdigest()


def _tensor_identity_hash(values: torch.Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _policy_state_hash(policy: ExpansionPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_identity_hash(value).encode())
    return digest.hexdigest()


def _gp_window_start_diagnostics(
    records: Sequence[QueryRecord],
    gammas: Sequence[float],
) -> dict[str, dict[str, int | float | None]]:
    output: dict[str, dict[str, int | float | None]] = {}
    for gamma in map(float, gammas):
        rows = [row for row in records if row.gamma == gamma]
        starts = np.asarray([
            int(row.window_start)
            for row in rows if row.window_start is not None
        ], dtype=int)
        trajectories = {
            str(row.trajectory_id)
            for row in rows if row.trajectory_id is not None
        }
        if len(starts):
            quantiles = np.quantile(starts, [0.0, 0.25, 0.5, 0.75, 1.0])
            stats = {
                "rows": int(len(rows)),
                "trajectories": int(len(trajectories)),
                "start_min": int(quantiles[0]),
                "start_q25": float(quantiles[1]),
                "start_median": float(quantiles[2]),
                "start_q75": float(quantiles[3]),
                "start_max": int(quantiles[4]),
                "start_le_4": int(np.sum(starts <= 4)),
                "start_le_9": int(np.sum(starts <= 9)),
            }
        else:
            stats = {
                "rows": int(len(rows)),
                "trajectories": int(len(trajectories)),
                "start_min": None,
                "start_q25": None,
                "start_median": None,
                "start_q75": None,
                "start_max": None,
                "start_le_4": 0,
                "start_le_9": 0,
            }
        output[f"{gamma:.9g}"] = stats
    return output


def _record_identity_key(row: QueryRecord) -> tuple[int, float, int, int, str]:
    """Order-independent deterministic key for frozen-feature coreset ties."""
    candidate = row.candidate.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(candidate.tobytes(order="C"))
    loss_mask = getattr(row, "loss_mask", None)
    if loss_mask is not None:
        digest.update(loss_mask.detach().cpu().contiguous().numpy().tobytes(order="C"))
    digest.update(f":valid_horizon={getattr(row, 'valid_horizon', None)}".encode())
    candidate_hash = digest.hexdigest()
    return (row.round, row.gamma, row.episode, row.context_id, candidate_hash)


def _sliding_success_gp_rows(
    records: Sequence[QueryRecord],
    gammas: Sequence[float],
    total_cap: int,
    *,
    through_round: int,
    selector: str = "trajectory_uniform",
) -> list[QueryRecord]:
    """Bounded deterministic support from committed successes, per gamma.

    ``records`` remains the immutable cumulative audit log.  Only the active
    posterior support changes. ``trajectory_uniform`` gives each successful
    trajectory nearly equal capacity and samples its window starts uniformly
    from departure through tail. ``fifo_tail`` exactly preserves the legacy
    high-window-start behavior for rollback. Both selectors are independent of
    archive/list order and preserve strict previous-round causality.
    """
    gamma_values = tuple(map(float, gammas))
    if total_cap % len(gamma_values) != 0:
        raise ValueError("sliding GP total cap must be divisible by gamma count")
    if selector not in {"trajectory_uniform", "fifo_tail"}:
        raise ValueError(
            "sliding GP selector must be trajectory_uniform or fifo_tail"
        )
    per_gamma_cap = total_cap // len(gamma_values)

    def fifo_key(row: QueryRecord) -> tuple[int, int, int, int, str, str]:
        identity = _record_identity_key(row)[-1]
        return (
            int(row.round),
            int(row.window_start if row.window_start is not None else -1),
            int(row.episode),
            int(row.context_id),
            str(row.window_id or ""),
            identity,
        )

    active: list[QueryRecord] = []
    for gamma in sorted(gamma_values):
        rows = sorted(
            (
                row for row in records
                if row.gamma == gamma and row.round <= through_round
            ),
            key=fifo_key,
        )
        if selector == "fifo_tail":
            active.extend(rows[-per_gamma_cap:])
            continue

        trajectories: dict[str, list[QueryRecord]] = {}
        for row in rows:
            if row.trajectory_id is None or row.window_start is None:
                raise ValueError(
                    "trajectory_uniform GP support requires trajectory_id and "
                    "window_start on every successful-window row"
                )
            trajectories.setdefault(str(row.trajectory_id), []).append(row)
        trajectory_rows = [
            (
                max(int(row.round) for row in block),
                trajectory_id,
                sorted(block, key=fifo_key),
            )
            for trajectory_id, block in trajectories.items()
        ]
        trajectory_rows.sort(key=lambda item: (item[0], item[1]))
        if len(trajectory_rows) > per_gamma_cap:
            trajectory_rows = trajectory_rows[-per_gamma_cap:]

        quotas = [0] * len(trajectory_rows)
        remaining = min(
            per_gamma_cap,
            sum(len(block) for _, _, block in trajectory_rows),
        )
        while remaining:
            progressed = False
            # Newer trajectories receive any deterministic remainder.
            for index in range(len(trajectory_rows) - 1, -1, -1):
                block = trajectory_rows[index][2]
                if quotas[index] >= len(block):
                    continue
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break

        trajectory_count = len(trajectory_rows)
        for trajectory_index, ((_, _, block), quota) in enumerate(
            zip(trajectory_rows, quotas)
        ):
            if quota == 0:
                continue
            if quota == 1:
                # When the number of successful trajectories reaches the
                # per-gamma cap, selecting block[0] for every trajectory
                # creates the mirror-image failure of the legacy tail bias:
                # an all-departure GP. Stratify the single retained row from
                # departure through tail across the retained trajectories.
                denominator = max(trajectory_count - 1, 1)
                indices = [
                    (trajectory_index * (len(block) - 1)) // denominator
                ]
            else:
                last = len(block) - 1
                indices = [
                    (position * last) // (quota - 1)
                    for position in range(quota)
                ]
            active.extend(block[index] for index in indices)
    return active


@torch.no_grad()
def _frozen_phi_farthest_first(
    policy: ExpansionPolicy,
    records: Sequence[QueryRecord],
    count: int,
    device: torch.device,
    *,
    reference: Sequence[QueryRecord] = (),
) -> list[QueryRecord]:
    """Deterministic route-label-free coreset in a frozen normalized ``phi``.

    With a reference set, candidates farthest from the reference and from
    already selected candidates enter first. Without one, the deterministic
    first center is farthest from the candidate centroid. Stable identity
    sorting resolves exact geometric ties without depending on archive order.
    """
    if count < 1 or not records:
        return []
    rows = sorted(records, key=_record_identity_key)
    if len(rows) <= count:
        return rows
    features = _normalize(_embed_records(policy, rows, device))
    if reference:
        reference_rows = sorted(reference, key=_record_identity_key)
        reference_features = _normalize(
            _embed_records(policy, reference_rows, device)
        )
        squared = (
            (features * features).sum(1, keepdim=True)
            + (reference_features * reference_features).sum(1)[None]
            - 2.0 * features @ reference_features.T
        ).clamp_min(0.0)
        minimum_distance = squared.min(dim=1).values
    else:
        centroid = features.mean(dim=0, keepdim=True)
        first_distance = (features - centroid).square().sum(dim=1)
        first = int(torch.argmax(first_distance))
        minimum_distance = (features - features[first]).square().sum(dim=1)
        selected = [first]
        minimum_distance[first] = -1.0

    if reference:
        selected = []
    while len(selected) < min(count, len(rows)):
        local = int(torch.argmax(minimum_distance))
        selected.append(local)
        distance = (features - features[local]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
    return [rows[index] for index in selected]


def _cumulative_gp_quotas(config: ExpansionConfig) -> tuple[int, int, int]:
    """Return per-gamma total, immutable-anchor, and ingress quotas."""
    per_gamma = config.gp_buffer_cap // len(config.gammas)
    anchor = per_gamma // 2
    adaptive = per_gamma - anchor
    ingress = max(1, adaptive // 2)
    return per_gamma, anchor, ingress


@torch.no_grad()
def _context_kcenter_replay(
    records: Sequence[QueryRecord],
    policy: ExpansionPolicy,
    quota: int,
    action_weight: float,
    *,
    embedding_batch_size: int = 4096,
) -> list[QueryRecord]:
    """Select a route-label-free geometric coreset independently per query context.

    The current model supplies normalized ``phi`` and the normalized raw plan supplies
    complementary geometry.  Farthest-first selection acts only on verified positives;
    it does not alter GP fitting, acquisition, labels, or execution.
    """
    if not records:
        return []
    device = next(policy.parameters()).device
    joint_blocks = []
    for start in range(0, len(records), embedding_batch_size):
        block = records[start:start + embedding_batch_size]
        _, candidates = (value.to(device) for value in _stack(block))
        phi = _normalize(_embed_records(policy, block, device))
        action = _normalize(candidates.reshape(len(candidates), -1))
        joint_blocks.append(
            torch.cat([phi, action_weight * action], dim=1)
            .div(math.sqrt(1.0 + action_weight ** 2))
            .cpu()
        )
    joint = torch.cat(joint_blocks)
    groups: dict[tuple[int, float, int, int], list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(
            (row.round, row.gamma, row.episode, row.context_id), []
        ).append(index)

    selected_global: list[int] = []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) <= quota:
            selected_global.extend(indices)
            continue
        features = joint[indices]
        distances = (
            (features * features).sum(1, keepdim=True)
            + (features * features).sum(1)[None]
            - 2.0 * features @ features.T
        ).clamp_min(0.0)
        flat = int(torch.argmax(distances))
        first, second = divmod(flat, len(indices))
        chosen = [first]
        if second != first:
            chosen.append(second)
        min_distance = distances[:, chosen].min(dim=1).values
        while len(chosen) < quota:
            min_distance[chosen] = -1.0
            next_index = int(torch.argmax(min_distance))
            if next_index in chosen:
                # All features are identical; stable archive order is the only
                # label-free deterministic tie break.
                next_index = next(i for i in range(len(indices)) if i not in chosen)
            chosen.append(next_index)
            min_distance = torch.minimum(min_distance, distances[:, next_index])
        selected_global.extend(indices[index] for index in chosen)
    return [records[index] for index in sorted(selected_global)]


@torch.no_grad()
def _cluster_balanced_replay(
    records: Sequence[QueryRecord],
    policy: ExpansionPolicy,
    cluster_count: int,
    action_weight: float,
    budget: int,
    generator: torch.Generator,
    *,
    iterations: int = 8,
    embedding_batch_size: int = 4096,
) -> list[QueryRecord]:
    """Flatten current-feature density without using route labels.

    Mini-batch-sized replay mass is divided equally across gamma and then across
    k-means cells in normalized ``[phi, U]``.  This is a training selector only;
    it does not alter the GP posterior or verifier.
    """
    if not records or budget >= len(records):
        return list(records)
    device = next(policy.parameters()).device
    blocks = []
    for start in range(0, len(records), embedding_batch_size):
        block = records[start:start + embedding_batch_size]
        _, candidates = (value.to(device) for value in _stack(block))
        phi = _normalize(_embed_records(policy, block, device))
        action = _normalize(candidates.reshape(len(candidates), -1))
        blocks.append(
            torch.cat([phi, action_weight * action], dim=1)
            .div(math.sqrt(1.0 + action_weight ** 2))
            .cpu()
        )
    features = torch.cat(blocks)
    gamma_groups: dict[float, list[int]] = {}
    for index, row in enumerate(records):
        gamma_groups.setdefault(row.gamma, []).append(index)
    gammas = sorted(gamma_groups)
    base, remainder = divmod(budget, len(gammas))
    selected_global: list[int] = []
    for gamma_index, gamma in enumerate(gammas):
        indices = gamma_groups[gamma]
        gamma_budget = min(len(indices), base + int(gamma_index < remainder))
        if gamma_budget >= len(indices):
            selected_global.extend(indices)
            continue
        x = features[indices]
        k = min(cluster_count, len(indices), gamma_budget)
        first = int(torch.randint(
            len(indices), (1,), generator=generator, device=x.device
        ))
        seeds = [first]
        min_seed_distance = (
            (x - x[first]).square().sum(dim=1)
        )
        while len(seeds) < k:
            min_seed_distance[seeds] = -1.0
            next_seed = int(torch.argmax(min_seed_distance))
            seeds.append(next_seed)
            min_seed_distance = torch.minimum(
                min_seed_distance,
                (x - x[next_seed]).square().sum(dim=1),
            )
        centers = x[seeds].clone()
        assignments = torch.zeros(len(x), dtype=torch.long)
        distances = torch.empty(len(x))
        for _ in range(iterations):
            squared = (
                (x * x).sum(1, keepdim=True)
                + (centers * centers).sum(1)[None]
                - 2.0 * x @ centers.T
            ).clamp_min(0.0)
            distances, assignments = squared.min(dim=1)
            sums = torch.zeros_like(centers)
            counts = torch.zeros(k, dtype=x.dtype)
            sums.index_add_(0, assignments, x)
            counts.index_add_(0, assignments, torch.ones(len(x), dtype=x.dtype))
            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty, None]
            centers = _normalize(centers)

        cells = [
            torch.nonzero(assignments == cluster, as_tuple=False).flatten().tolist()
            for cluster in range(k)
        ]
        for cluster, cell in enumerate(cells):
            cell.sort(key=lambda local: (float(distances[local]), local))
        chosen_local: list[int] = []
        cursor = 0
        while len(chosen_local) < gamma_budget:
            progressed = False
            for cell in cells:
                if cursor < len(cell):
                    chosen_local.append(cell[cursor])
                    progressed = True
                    if len(chosen_local) == gamma_budget:
                        break
            if not progressed:
                break
            cursor += 1
        selected_global.extend(indices[local] for local in chosen_local)
    return [records[index] for index in sorted(selected_global)]


def _softmin_choice(costs: Sequence[float], target: float,
                    generator: torch.Generator, device: torch.device) -> tuple[int, float]:
    """Sample one cost with a per-context normalized-ESS target."""
    values = torch.as_tensor(costs, dtype=torch.float32, device=device)
    if len(values) == 1 or float(values.max() - values.min()) <= 1.0e-12:
        probability = torch.full_like(values, 1.0 / len(values))
    else:
        scores = -values
        temperature = calibrate_fixed_beta(
            [scores.detach().cpu()], target, lower=1.0e-8, upper=1.0e8
        )
        probability = torch.softmax((scores - scores.max()) / temperature, dim=0)
    choice = int(torch.multinomial(probability, 1, generator=generator))
    ess = float(
        1.0 / (
            probability.to(
                _stable_accumulation_dtype(probability)
            ).square().sum()
            * len(probability)
        )
    )
    return choice, ess


def perturb_plan_candidates(policy: ExpansionPolicy, candidates: torch.Tensor, std: float,
                            generator: torch.Generator,
                            scope: str = "coherent_horizon") -> torch.Tensor:
    """Add one action-space Gaussian vector per candidate plan.

    The canonical ``coherent_horizon`` law broadcasts the same 3-D vector over H.
    ``first_action`` is retained only as an explicit ablation.  The returned plans are
    embedded, verified, stored, executed, and used as CFM targets.
    """
    if std == 0.0:
        return candidates
    if candidates.ndim < 2:
        raise ValueError("candidate plans must include batch and action dimensions")
    if scope not in {"first_action", "coherent_horizon"}:
        raise ValueError("scope must be first_action or coherent_horizon")
    noise_shape = (len(candidates), candidates.shape[-1])
    noise = torch.randn(noise_shape, dtype=candidates.dtype, device=candidates.device,
                        generator=generator) * std
    perturbed = candidates.clone()
    if candidates.ndim == 2:
        perturbed += noise
    elif scope == "first_action":
        perturbed[:, 0] += noise
    else:
        broadcast = noise.reshape(len(candidates), *([1] * (candidates.ndim - 2)),
                                  candidates.shape[-1])
        perturbed += broadcast
    limit = getattr(policy, "control_limit", None)
    return perturbed.clamp(-float(limit), float(limit)) if limit is not None else perturbed


def _gradient_norm(gradients: Sequence[torch.Tensor | None], device) -> torch.Tensor:
    return torch.sqrt(sum((gradient.square().sum() for gradient in gradients
                           if gradient is not None),
                          torch.zeros((), device=device)).clamp_min(0.0))


def _apply_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    cfg: ExpansionConfig,
    step: int,
) -> float:
    """Apply global cosine decay and the current outer-round LR scale."""
    base_lrs = getattr(optimizer, "_safe_mppi_base_lrs", None)
    if base_lrs is None:
        base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        setattr(optimizer, "_safe_mppi_base_lrs", base_lrs)
    round_scale = float(
        getattr(optimizer, "_safe_mppi_round_lr_scale", 1.0)
    )
    if cfg.learning_rate_final is None:
        factor = round_scale
    else:
        progress = min(step / int(cfg.learning_rate_decay_steps), 1.0)
        final_ratio = float(cfg.learning_rate_final / cfg.learning_rate)
        cosine_factor = final_ratio + 0.5 * (1.0 - final_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
        factor = max(final_ratio, cosine_factor * round_scale)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr * factor)
    return float(optimizer.param_groups[-1]["lr"])


def _step_cosine_learning_rate(
    optimizer: torch.optim.Optimizer,
    cfg: ExpansionConfig,
) -> tuple[int, float]:
    """Advance the optional optimizer-step cosine schedule, clamped at eta_min."""
    step = int(getattr(optimizer, "_safe_mppi_schedule_step", 0)) + 1
    setattr(optimizer, "_safe_mppi_schedule_step", step)
    return step, _apply_optimizer_learning_rate(optimizer, cfg, step)


def _set_round_learning_rate(
    optimizer: torch.optim.Optimizer,
    cfg: ExpansionConfig,
    round_i: int,
) -> tuple[float, float]:
    scale = (
        1.0
        if cfg.round_learning_rate_warmup_power == 0.0
        else (round_i / cfg.rounds) ** cfg.round_learning_rate_warmup_power
    )
    setattr(optimizer, "_safe_mppi_round_lr_scale", float(scale))
    step = int(getattr(optimizer, "_safe_mppi_schedule_step", 0))
    return float(scale), _apply_optimizer_learning_rate(
        optimizer, cfg, step,
    )


def _optimizer_steps_for_round(
    cfg: ExpansionConfig,
    round_i: int,
) -> int | None:
    """Allocate an exact global optimizer budget across expansion rounds."""
    if cfg.optimizer_steps_total is None:
        return cfg.inner_steps
    if cfg.optimizer_step_allocation == "quadratic_round":
        denominator = cfg.rounds * (cfg.rounds + 1) * (
            2 * cfg.rounds + 1
        ) // 6
        before_count = round_i - 1
        before_weight = before_count * (before_count + 1) * (
            2 * before_count + 1
        ) // 6
        through_weight = round_i * (round_i + 1) * (
            2 * round_i + 1
        ) // 6
        before = cfg.optimizer_steps_total * before_weight // denominator
        through = cfg.optimizer_steps_total * through_weight // denominator
    elif cfg.optimizer_step_allocation == "linear_round":
        denominator = cfg.rounds * (cfg.rounds + 1) // 2
        before_weight = (round_i - 1) * round_i // 2
        through_weight = round_i * (round_i + 1) // 2
        before = cfg.optimizer_steps_total * before_weight // denominator
        through = cfg.optimizer_steps_total * through_weight // denominator
    else:
        before = cfg.optimizer_steps_total * (round_i - 1) // cfg.rounds
        through = cfg.optimizer_steps_total * round_i // cfg.rounds
    return through - before


def _mode_gamma_stratified_batches(
    positives: Sequence[QueryRecord],
    *,
    steps: int,
    batch_size: int,
    gammas: Sequence[float],
    modes: Sequence[int],
    rng: np.random.Generator,
) -> tuple[Any, dict[str, dict[str, int]]]:
    """Sample mode/gamma -> trajectory -> window, matching Reserve G."""
    trajectory_rows: dict[
        tuple[int, float], dict[str, list[int]]
    ] = {}
    for index, row in enumerate(positives):
        mode = getattr(row, "sample_update_mode", None)
        if mode is None or row.trajectory_id is None:
            raise ValueError(
                "mode_gamma_stratified replay requires mode-labelled "
                "trajectory lineages on every positive row"
            )
        key = (int(mode), float(row.gamma))
        trajectory_rows.setdefault(key, {}).setdefault(
            str(row.trajectory_id), []
        ).append(index)
    expected = {
        (mode, float(gamma))
        for mode in sorted(set(map(int, modes)))
        for gamma in gammas
    }
    missing = expected - set(trajectory_rows)
    unexpected = set(trajectory_rows) - expected
    if missing or unexpected:
        raise ValueError(
            "mode_gamma_stratified replay requires all and only requested "
            "label x gamma "
            f"strata; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    keys = sorted(expected)
    counts = [batch_size // len(keys)] * len(keys)
    for index in range(batch_size % len(keys)):
        counts[index] += 1
    def batches():
        for _ in range(steps):
            batch: list[int] = []
            for key, count in zip(keys, counts):
                lineages = sorted(trajectory_rows[key])
                lineage_indices = rng.integers(
                    len(lineages), size=count,
                )
                for lineage_index in lineage_indices:
                    rows = trajectory_rows[key][
                        lineages[int(lineage_index)]
                    ]
                    batch.append(rows[int(rng.integers(len(rows)))])
            rng.shuffle(batch)
            yield batch

    def label(key: tuple[int, float]) -> str:
        return f"mode={key[0]},gamma={key[1]:.9g}"

    return batches(), {
        "available_trajectories_by_mode_gamma": {
            label(key): len(trajectory_rows[key]) for key in keys
        },
        "available_rows_by_mode_gamma": {
            label(key): sum(map(len, trajectory_rows[key].values()))
            for key in keys
        },
        "sampled_rows_by_mode_gamma": {
            label(key): counts[index] * steps
            for index, key in enumerate(keys)
        },
    }


def _update(policy: ExpansionPolicy, task: ExpansionTask,
            optimizer: torch.optim.Optimizer,
            positives: Sequence[QueryRecord], negatives: Sequence[QueryRecord],
            cfg: ExpansionConfig, generator: torch.Generator) -> dict[str, Any]:
    optimizer_steps = cfg.inner_steps
    if not positives:
        return {
            "steps": 0,
            "optimizer_step": int(
                getattr(optimizer, "_safe_mppi_schedule_step", 0)
            ),
            "learning_rate": float(optimizer.param_groups[-1]["lr"]),
            "positive_loss": None,
            "negative_loss": None,
            "replay_batch_sampler": cfg.replay_batch_sampler,
            "requested_optimizer_steps": optimizer_steps,
        }
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    device = parameters[0].device
    replay_diagnostics: dict[str, dict[str, int]] = {}
    if cfg.replay_batch_sampler == "mode_gamma_stratified":
        if optimizer_steps is None:
            raise ValueError(
                "mode_gamma_stratified replay requires optimizer steps"
            )
        sampler_seed = int(torch.randint(
            2 ** 31, (), generator=generator, device=device,
        ).item())
        batches, replay_diagnostics = _mode_gamma_stratified_batches(
            positives,
            steps=optimizer_steps,
            batch_size=cfg.batch_size,
            gammas=cfg.gammas,
            modes=cfg.sample_update_mode,
            rng=np.random.default_rng(sampler_seed),
        )
        weights = torch.ones(len(positives), dtype=torch.float32)
    else:
        order = []
        for _ in range(cfg.replay_passes):
            pass_order = torch.randperm(
                len(positives), generator=generator, device=device
            ).cpu().tolist()
            if optimizer_steps is not None:
                pass_order = pass_order[:optimizer_steps * cfg.batch_size]
            order.extend(pass_order)
        batches = (
            order[start:start + cfg.batch_size]
            for start in range(0, len(order), cfg.batch_size)
        )
        weights = _equal_mass_weights(positives)
    if cfg.coverage_replay == "circular_voronoi":
        weights = _circular_voronoi_weights(positives, weights)
    positive_values, negative_values = [], []
    optimizer_step = int(getattr(optimizer, "_safe_mppi_schedule_step", 0))
    learning_rate = float(optimizer.param_groups[-1]["lr"])
    for ids in batches:
        batch = [positives[index] for index in ids]
        contexts, candidates = (value.to(device) for value in _stack(batch))
        loss_mask = _stack_loss_masks(batch)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        batch_weights = weights[ids].to(device)
        if cfg.replay_augmentation == "task_d4":
            augment = getattr(task, "d4_replay_batch", None)
            if not callable(augment):
                raise TypeError(
                    "task_d4 replay requires task.d4_replay_batch(contexts, candidates)"
                )
            contexts, candidates = augment(contexts, candidates)
            if len(contexts) != 4 * len(batch) or len(candidates) != len(contexts):
                raise ValueError("task_d4 replay must return four transforms per row")
            batch_weights = batch_weights.repeat(4)
            if loss_mask is not None:
                loss_mask = loss_mask.repeat(4, *([1] * (loss_mask.ndim - 1)))
        for _ in range(cfg.microbatch_repeats):
            if loss_mask is None:
                per_sample = policy.cfm_loss(
                    contexts, candidates, reduction="none",
                )
            else:
                per_sample = policy.cfm_loss(
                    contexts, candidates, reduction="none",
                    loss_mask=loss_mask,
                )
            positive_loss = (per_sample * batch_weights).mean()
            optimizer.zero_grad()
            if cfg.negative_alpha and negatives:
                negative_ids = torch.randint(
                    len(negatives), (len(batch),), generator=generator, device=device
                ).cpu().tolist()
                neg_context, neg_candidate = _stack([negatives[index] for index in negative_ids])
                negative_loss = policy.cfm_loss(
                    neg_context.to(device), neg_candidate.to(device), reduction="mean"
                )
                positive_grad = torch.autograd.grad(
                    positive_loss, parameters, allow_unused=True)
                negative_grad = torch.autograd.grad(
                    negative_loss, parameters, allow_unused=True)
                pos_norm = _gradient_norm(positive_grad, device)
                neg_norm = _gradient_norm(negative_grad, device)
                rho = cfg.negative_alpha * pos_norm / (neg_norm + 1.0e-12)
                for parameter, pos, neg in zip(parameters, positive_grad, negative_grad):
                    if pos is None and neg is None:
                        parameter.grad = None
                    elif pos is None:
                        parameter.grad = -rho * neg.detach()
                    elif neg is None:
                        parameter.grad = pos.detach()
                    else:
                        parameter.grad = pos.detach() - rho * neg.detach()
                negative_values.append(float(negative_loss.detach()))
            else:
                positive_loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters, cfg.gradient_clip_norm,
                )
            optimizer.step()
            optimizer_step, learning_rate = _step_cosine_learning_rate(
                optimizer, cfg,
            )
            positive_values.append(float(positive_loss.detach()))
    return {
        "steps": len(positive_values),
        "optimizer_step": optimizer_step,
        "learning_rate": learning_rate,
        "positive_loss": float(np.mean(positive_values)),
        "negative_loss": (float(np.mean(negative_values)) if negative_values else None),
        "replay_batch_sampler": cfg.replay_batch_sampler,
        "requested_optimizer_steps": optimizer_steps,
        **replay_diagnostics,
    }


def _head_only_parameters(policy: ExpansionPolicy) -> list[torch.nn.Parameter]:
    """Freeze the policy backbone and return only its final-head parameters."""
    head = getattr(policy, "head", None)
    if not isinstance(head, torch.nn.Module):
        raise TypeError(
            "head_only_update requires a policy with a torch Module head"
        )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    parameters = list(head.parameters())
    if not parameters:
        raise ValueError("policy head has no parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _trunk_suffix_parameters(
        policy: ExpansionPolicy, trainable_trunk_layers: int,
) -> list[torch.nn.Parameter]:
    """Freeze the policy except its head and the last N trunk blocks.

    ``trainable_trunk_layers`` counts parameterised blocks of the flow trunk
    from its output side, so N=1 adapts the representation layer feeding the
    head and N=trunk depth adapts the whole trunk.  Every remaining parameter
    -- earlier trunk blocks, conditioning/input encoders, and any history or
    visual encoder -- stays frozen.
    """
    head = getattr(policy, "head", None)
    if not isinstance(head, torch.nn.Module):
        raise TypeError(
            "trainable_trunk_layers requires a policy with a torch Module head"
        )
    trunk = getattr(policy, "trunk", None)
    if not isinstance(trunk, torch.nn.Module):
        raise TypeError(
            "trainable_trunk_layers requires a policy with a torch Module trunk"
        )
    blocks = [
        module for module in trunk.children()
        if any(True for _ in module.parameters())
    ]
    if trainable_trunk_layers < 1:
        raise ValueError("trainable_trunk_layers must be positive")
    if trainable_trunk_layers > len(blocks):
        raise ValueError(
            f"trainable_trunk_layers={trainable_trunk_layers} exceeds the "
            f"policy trunk depth {len(blocks)}"
        )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    parameters = list(head.parameters())
    for block in blocks[-trainable_trunk_layers:]:
        parameters.extend(block.parameters())
    if not parameters:
        raise ValueError(
            "policy head and selected trunk blocks have no parameters"
        )
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _resolve_resume_state_path(resume_from: str | Path) -> Path:
    path = Path(resume_from)
    return path / _RESUME_STATE_NAME if path.is_dir() else path


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Publish a torch artifact only after its complete payload is durable."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _validate_resume_config(
    saved: dict[str, Any], current: ExpansionConfig, completed_round: int,
) -> None:
    saved = dict(saved)
    # Checkpoints written before the bounded replacement-cooldown addition
    # performed the legacy every-batch replacement after the first trigger.
    saved.setdefault("paired_scene_replace_interval_batches", 1)
    saved.setdefault("successful_trajectories_per_round", None)
    current_values = asdict(current)
    if set(saved) != set(current_values):
        raise ValueError(
            "resume state was written by an incompatible ExpansionConfig schema"
        )
    changed = {
        key: (saved[key], current_values[key])
        for key in saved
        if saved[key] != current_values[key]
        and key not in _RESUME_MUTABLE_CONFIG_FIELDS
        and key != "rounds"
    }
    if changed:
        details = ", ".join(
            f"{key}={before!r}->{after!r}"
            for key, (before, after) in sorted(changed.items())
        )
        raise ValueError(
            "resume changed optimizer/replay semantics; only future rollout "
            f"sampling options may change ({details})"
        )
    saved_rounds = int(saved["rounds"])
    if current.rounds < completed_round:
        raise ValueError(
            f"resume target rounds={current.rounds} precedes committed round "
            f"{completed_round}"
        )
    if current.rounds != saved_rounds and (
        current.optimizer_steps_total is not None
        or current.round_learning_rate_warmup_power != 0.0
    ):
        raise ValueError(
            "changing --rounds on resume would alter the global optimizer/LR "
            "schedule; keep the original round target"
        )


def _load_resume_state(
    resume_from: str | Path, device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_resume_state_path(resume_from)
    if not path.is_file():
        raise FileNotFoundError(f"missing exact resume state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("version") != _RESUME_STATE_VERSION:
        raise ValueError(f"unsupported expansion resume state: {path}")
    if state.get("status") != "COMMITTED_ROUND_RESUME":
        raise ValueError(f"resume state is not a committed-round snapshot: {path}")
    return path, state


def _run_safe_expansion_impl(
    policy: ExpansionPolicy,
    task: ExpansionTask,
    output_dir: str | Path,
    *,
    config: ExpansionConfig = ExpansionConfig(),
    calibration_features: torch.Tensor | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    round_callback: Callable[[dict[str, Any]], None] | None = None,
    retry_callback: Callable[[dict[str, Any]], None] | None = None,
    pair_replacement_callback: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    trainable_trunk_layers: int | None = None,
    resume_from: str | Path | None = None,
    verifier: _OrderedVerifier,
) -> dict[str, Any]:
    """Run expert-free B1 expansion; task callbacks supply every task-specific fact."""
    config.validate()
    sample_update_mode_fn = None
    pair_identity_fn = None
    pair_current_version_fn = None
    pair_replace_fn = None
    if config.sample_update_mode is not None:
        sample_update_mode_fn = getattr(
            task, "successful_trajectory_mode", None,
        )
        if not callable(sample_update_mode_fn):
            raise TypeError(
                "sample_update_mode requires "
                "task.successful_trajectory_mode(executed_states)"
            )
    if config.paired_scene_max_replacements_per_slot > 0:
        pair_identity_fn = getattr(
            task, "successful_trajectory_pair_identity", None,
        )
        pair_current_version_fn = getattr(
            task, "paired_scene_current_version", None,
        )
        pair_replace_fn = getattr(task, "replace_paired_scene_slot", None)
        if not all(callable(value) for value in (
            pair_identity_fn, pair_current_version_fn, pair_replace_fn,
        )):
            raise TypeError(
                "paired scene replacement requires task pair-identity, "
                "current-version, and replacement callbacks"
            )
    if trainable_trunk_layers is not None and config.head_only_update:
        raise ValueError(
            "trainable_trunk_layers cannot be combined with head_only_update"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None and any(output_dir.iterdir()):
        raise FileExistsError("expansion output_dir must be empty")
    resume_path: Path | None = None
    resume_state: dict[str, Any] | None = None
    if resume_from is not None:
        resume_path, resume_state = _load_resume_state(
            resume_from, next(policy.parameters()).device,
        )
        if resume_path.parent.resolve() != output_dir.resolve():
            raise ValueError(
                "exact resume must continue in the directory containing "
                f"{_RESUME_STATE_NAME}"
            )
    if config.rbf_lengthscale is None:
        if calibration_features is None:
            raise ValueError("provide 50 pretrained embeddings or set rbf_lengthscale explicitly")
        if (config.acquisition_feature == "task_angular"
                and (calibration_features.ndim != 2
                     or calibration_features.shape[1] != 2)):
            raise ValueError(
                "task_angular RBF calibration requires two-dimensional "
                "task acquisition features"
            )
        lengthscale = mean_pairwise_lengthscale(calibration_features)
    else:
        lengthscale = float(config.rbf_lengthscale)
    archive: list[QueryRecord] = []
    if config.freeze_visual_encoder:
        freeze_visual_encoder = getattr(
            policy, "freeze_visual_encoder_for_expansion", None,
        )
        if not callable(freeze_visual_encoder):
            raise TypeError(
                "freeze_visual_encoder requires a policy with "
                "freeze_visual_encoder_for_expansion()"
            )
        freeze_visual_encoder()
    if config.head_only_update:
        optimizer_parameters = _head_only_parameters(policy)
    elif trainable_trunk_layers is not None:
        optimizer_parameters = _trunk_suffix_parameters(
            policy, int(trainable_trunk_layers),
        )
    elif config.first_layer_lr_scale < 1.0:
        parameter_groups = getattr(policy, "expansion_parameter_groups", None)
        if not callable(parameter_groups):
            raise TypeError(
                "first_layer_lr_scale<1 requires "
                "policy.expansion_parameter_groups(base_lr, scale)"
            )
        optimizer_parameters = parameter_groups(
            config.learning_rate, config.first_layer_lr_scale,
        )
    else:
        optimizer_parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
    trainable_parameter_names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in policy.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(optimizer_parameters, lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    device = next(policy.parameters()).device
    acquisition_policy = policy
    if config.gp_reference_mode in {
            "round1_fixed_frozen_phi", "cumulative_accepted_frozen_phi",
            "cumulative_success_per_gamma_frozen_phi_exact",
            "sliding_success_per_gamma_frozen_phi"}:
        if config.acquisition_feature != "learned_phi":
            raise ValueError(
                f"{config.gp_reference_mode} requires learned_phi acquisition"
            )
        acquisition_policy = copy.deepcopy(policy)
        for parameter in acquisition_policy.parameters():
            parameter.requires_grad_(False)
        eval_fn = getattr(acquisition_policy, "eval", None)
        if callable(eval_fn):
            eval_fn()
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    beta = float(config.beta)
    round_rows = []
    frozen_gp_rows: list[QueryRecord] | None = None
    frozen_gp_hash: str | None = None
    round1_gp_candidates: list[QueryRecord] = []
    gp_evidence: list[QueryRecord] = []
    cumulative_anchors = {float(gamma): [] for gamma in config.gammas}
    cumulative_adaptive = {float(gamma): [] for gamma in config.gammas}
    start_round = 1
    resumed_from_round: int | None = None
    if resume_state is None:
        _atomic_torch_save(
            {"round": 0, "model": policy.state_dict(),
             "config": asdict(config), "pretrained": True},
            output_dir / "checkpoint_000.pt",
        )
    else:
        resumed_from_round = int(resume_state["completed_round"])
        _validate_resume_config(
            resume_state["config"], config, resumed_from_round,
        )
        expected_names = list(resume_state["trainable_parameter_names"])
        if expected_names != trainable_parameter_names:
            raise ValueError(
                "resume optimizer parameter order/scope differs from the "
                "committed run"
            )
        policy.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        _move_optimizer_state(optimizer, device)
        optimizer_meta = resume_state["optimizer_metadata"]
        for name in (
            "_safe_mppi_schedule_step", "_safe_mppi_base_lrs",
            "_safe_mppi_round_lr_scale",
        ):
            if name in optimizer_meta:
                setattr(optimizer, name, optimizer_meta[name])
        rng.bit_generator.state = resume_state["numpy_rng_state"]
        torch_rng.set_state(resume_state["torch_rng_state"])
        torch.set_rng_state(resume_state["torch_cpu_rng_state"])
        if device.type == "cuda" and resume_state.get("torch_device_rng_state") is not None:
            torch.cuda.set_rng_state(
                resume_state["torch_device_rng_state"], device=device,
            )
        beta = float(resume_state["beta"])
        if float(config.beta) != float(resume_state["config"]["beta"]):
            # Beta changes only future acquisition ranking.  Preserve an
            # adaptively calibrated value on an unchanged resume, but honor
            # an explicit recovery-time beta override for a stuck quota.
            beta = float(config.beta)
        round_rows = list(resume_state["round_rows"])
        archive = list(resume_state["archive"])
        frozen_gp_rows = resume_state["frozen_gp_rows"]
        frozen_gp_hash = resume_state["frozen_gp_hash"]
        round1_gp_candidates = list(resume_state["round1_gp_candidates"])
        gp_evidence = list(resume_state["gp_evidence"])
        cumulative_anchors = resume_state["cumulative_anchors"]
        cumulative_adaptive = resume_state["cumulative_adaptive"]
        saved_flow_std = float(resume_state["config"]["flow_base_std"])
        object.__setattr__(config, "flow_base_std", saved_flow_std)
        if len(round_rows) != resumed_from_round:
            raise ValueError(
                "resume round history does not match completed_round"
            )
        required = [
            output_dir / f"checkpoint_{resumed_from_round:03d}.pt",
            output_dir / f"query_archive_round_{resumed_from_round:03d}.pt",
            output_dir / "metrics.jsonl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "resume directory is missing committed artifacts: "
                + ", ".join(missing)
            )
        start_round = resumed_from_round + 1
        if start_round > config.rounds:
            raise ValueError(
                f"run already committed round {resumed_from_round}; increase "
                "--rounds only when no global optimizer/round LR schedule is used"
            )
        previous_manifest = output_dir / "manifest.json"
        if previous_manifest.is_file():
            previous_manifest.replace(
                output_dir
                / f"manifest_before_resume_round_{resumed_from_round:03d}.json"
            )
        (output_dir / "RESUME_IN_PROGRESS.json").write_text(json.dumps({
            "status": "RESUME_IN_PROGRESS",
            "resumed_from_round": resumed_from_round,
            "next_round": start_round,
        }, indent=2) + "\n")
    for round_i in range(start_round, config.rounds + 1):
        round_started = time.perf_counter()
        beta_used = beta
        gp_anchor_count = 0
        gp_adaptive_count = 0
        acquisition_representation_hash = _policy_state_hash(
            acquisition_policy
        )
        if config.gp_reference_mode == "round1_fixed_frozen_phi":
            if round_i == 1:
                gp_rows = []
            else:
                if frozen_gp_rows is None:
                    round1_positive = list(round1_gp_candidates)
                    if not round1_positive:
                        raise RuntimeError(
                            "round1_fixed_frozen_phi found no eligible round-1 "
                            "positive reference rows"
                        )
                    frozen_rng = np.random.default_rng(config.seed + 7919)
                    frozen_gp_rows = _balanced_cap(
                        round1_positive, config.gp_buffer_cap, frozen_rng
                    )
                    frozen_gp_hash = _record_identity_hash(frozen_gp_rows)
                gp_rows = frozen_gp_rows
        elif config.gp_reference_mode == "cumulative_accepted_frozen_phi":
            if round_i == 1:
                gp_rows = []
            else:
                per_gamma_cap, anchor_cap, ingress_cap = (
                    _cumulative_gp_quotas(config)
                )
                adaptive_cap = per_gamma_cap - anchor_cap
                if round_i == 2:
                    for gamma in sorted(cumulative_anchors):
                        candidates = [
                            row for row in gp_evidence
                            if row.round == 1 and row.gamma == gamma
                        ]
                        cumulative_anchors[gamma] = _frozen_phi_farthest_first(
                            acquisition_policy, candidates, anchor_cap, device
                        )
                else:
                    for gamma in sorted(cumulative_adaptive):
                        incoming = [
                            row for row in gp_evidence
                            if row.round == round_i - 1 and row.gamma == gamma
                        ]
                        reference = (
                            cumulative_anchors[gamma]
                            + cumulative_adaptive[gamma]
                        )
                        admitted = _frozen_phi_farthest_first(
                            acquisition_policy, incoming, ingress_cap, device,
                            reference=reference,
                        )
                        cumulative_adaptive[gamma] = (
                            _frozen_phi_farthest_first(
                                acquisition_policy,
                                cumulative_adaptive[gamma] + admitted,
                                adaptive_cap,
                                device,
                                reference=cumulative_anchors[gamma],
                            )
                        )
                gp_rows = [
                    row
                    for gamma in sorted(cumulative_anchors)
                    for row in (
                        cumulative_anchors[gamma]
                        + cumulative_adaptive[gamma]
                    )
                ]
                frozen_gp_hash = (
                    _record_identity_hash(gp_rows) if gp_rows else None
                )
                gp_anchor_count = sum(map(len, cumulative_anchors.values()))
                gp_adaptive_count = sum(map(len, cumulative_adaptive.values()))
        elif (
            config.gp_reference_mode
            == "cumulative_success_per_gamma_frozen_phi_exact"
        ):
            # Every row was reconstructed from a committed terminal-SUCCESS
            # trajectory in an earlier round. Exact means no cap-based thinning:
            # the explicit support limit is checked atomically before commit.
            gp_rows = list(gp_evidence)
            frozen_gp_hash = (
                _record_identity_hash(gp_rows) if gp_rows else None
            )
        elif config.gp_reference_mode in {
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
        }:
            gp_rows = _sliding_success_gp_rows(
                gp_evidence,
                config.gammas,
                config.gp_buffer_cap,
                through_round=round_i - 1,
                selector=config.gp_sliding_row_selector,
            )
            if any(row.round >= round_i for row in gp_rows):
                raise RuntimeError(
                    "sliding GP support must contain only previous-round evidence"
                )
            frozen_gp_hash = (
                _record_identity_hash(gp_rows) if gp_rows else None
            )
        else:
            previous_positive = _recent(
                archive, round_i - 1, config.replay_rounds, positive=True
            )
            previous_positive = _top_uncertainty_by_round(
                previous_positive, config.replay_top_fraction,
                group_by_gamma=(
                    config.archive_rule == "successful_executed_windows"
                ),
            )
            gp_rows = _balanced_cap(previous_positive, config.gp_buffer_cap, rng)
        shared_gp: RBFPosterior | None = None
        gp_by_gamma: dict[float, RBFPosterior] = {}
        gp_feature_hash_by_gamma: dict[str, str | None] = {
            f"{float(gamma):.9g}": None for gamma in config.gammas
        }
        if (
            config.acquisition_feature == "task_angular"
            or config.gp_reference_mode in {
                "cumulative_success_per_gamma_frozen_phi_exact",
                "sliding_success_per_gamma_frozen_phi",
                "sliding_success_per_gamma_current_phi",
            }
        ):
            gp_by_gamma = {
                float(gamma): RBFPosterior(lengthscale, config.gp_noise)
                for gamma in config.gammas
            }
            for gamma, gp in gp_by_gamma.items():
                gamma_rows = [row for row in gp_rows if row.gamma == gamma]
                if gamma_rows:
                    if config.acquisition_feature == "task_angular":
                        features = _stack_acquisition_features(
                            gamma_rows, device
                        )
                    else:
                        features = _embed_records(
                            acquisition_policy, gamma_rows, device
                        )
                    gp_feature_hash_by_gamma[f"{gamma:.9g}"] = (
                        _tensor_identity_hash(_normalize(features))
                    )
                    gp.set_buffer(features)
        else:
            shared_gp = RBFPosterior(lengthscale, config.gp_noise)
            if gp_rows:
                shared_gp.set_buffer(
                    _embed_records(acquisition_policy, gp_rows, device)
                )
        gp_refit_s = time.perf_counter() - round_started
        gather_started = time.perf_counter()
        episodes: list[dict[str, Any]] = []
        retry_capability_reverify_queries = 0
        retry_batches_by_gamma = {
            float(gamma): 0 for gamma in config.gammas
        }
        completed_retry_gammas: set[float] = set()
        exhausted_retry_gammas: set[float] = set()
        paired_scene_replacements: list[dict[str, Any]] = []
        pair_replacement_counts: dict[tuple[float, int], int] = {}
        pair_replacement_last_batch: dict[tuple[float, int], int] = {}

        def launch_episode_batch(gamma_value: float) -> None:
            retry_batch = retry_batches_by_gamma[gamma_value]
            cohorts = (
                (
                    ("guided", "unguided")
                    if config.sample_update_cohorts == "paired"
                    else ("unguided",)
                )
                if config.sample_update_mode is not None else (None,)
            )
            for cohort in cohorts:
                for replica in range(config.parallel_episodes):
                    episode = len(episodes)
                    if config.archive_rule == "successful_executed_windows":
                        reset_seed = _counter_seed(
                            config.seed, "reset", round_i, retry_batch, replica
                        )
                    else:
                        reset_seed = config.seed + 10000 * round_i + episode
                    episodes.append({
                        "gamma": gamma_value, "episode": episode,
                        "replica": replica, "retry_batch": retry_batch,
                        "sample_update_cohort": cohort,
                        "step": 0, "status": None,
                        "state": task.reset(
                            gamma_value, episode, reset_seed
                        ),
                        "executed_steps": [],
                    })
            retry_batches_by_gamma[gamma_value] += 1

        def reverify_executed_suffix(
            executed_steps: Sequence[dict[str, Any]],
            start: int,
            horizon: int,
            gamma_value: float,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Verification
        ]:
            """Verify the available executed suffix and pad only for fixed-shape CFM."""
            valid_horizon = min(horizon, len(executed_steps) - start)
            if valid_horizon < 1:
                raise ValueError("executed suffix must contain at least one action")
            context_window = executed_steps[start]["context"].to(device)
            actual_window = torch.stack([
                executed_steps[offset]["action"]
                for offset in range(start, start + valid_horizon)
            ]).to(device)
            padded_window = torch.zeros(
                (horizon, *actual_window.shape[1:]),
                dtype=actual_window.dtype,
                device=device,
            )
            padded_window[:valid_horizon] = actual_window
            loss_mask = torch.zeros_like(padded_window)
            loss_mask[:valid_horizon] = 1.0
            verification = list(task.verify(
                context_window, actual_window.unsqueeze(0), gamma_value
            ))
            if len(verification) != 1:
                raise RuntimeError(
                    "reconstructed executed-window verification must return "
                    "exactly one result"
                )
            return (
                context_window, actual_window, padded_window, loss_mask,
                verification[0],
            )

        def paired_executed_suffix_base(
            executed_steps: Sequence[dict[str, Any]],
            start: int,
            horizon: int,
        ) -> torch.Tensor | None:
            """Compose a successful window from its actual action/base pairs."""
            if not config.paired_noised_representation:
                return None
            valid_horizon = min(horizon, len(executed_steps) - start)
            paired_actions = []
            for step_record in executed_steps[start:start + valid_horizon]:
                flow_base_action = step_record.get("flow_base_action")
                if flow_base_action is None:
                    raise RuntimeError(
                        "paired successful-window replay is missing an executed "
                        "first-action flow base"
                    )
                paired_actions.append(flow_base_action.to(device))
            actual_base = torch.stack(paired_actions)
            if valid_horizon == horizon:
                return actual_base
            padding = torch.zeros(
                (horizon - valid_horizon, *actual_base.shape[1:]),
                dtype=actual_base.dtype,
                device=device,
            )
            return torch.cat([actual_base, padding], dim=0)

        def has_commit_capable_window(
            episode_record: dict[str, Any],
            gamma_value: float,
        ) -> bool:
            """Reverify executed sliding windows before accepting a retry batch."""
            nonlocal retry_capability_reverify_queries
            executed_steps = episode_record["executed_steps"]
            if not executed_steps:
                return False
            horizons = {
                int(step_record["plan_horizon"])
                for step_record in executed_steps
            }
            if len(horizons) != 1:
                raise RuntimeError(
                    "executed candidates changed horizon within one "
                    "successful trajectory"
                )
            horizon = horizons.pop()
            action_shapes = {
                tuple(step_record["action"].shape)
                for step_record in executed_steps
            }
            if len(action_shapes) != 1:
                raise RuntimeError(
                    "executed action shape changed within one successful trajectory"
                )
            for start in range(len(executed_steps)):
                *_, result = reverify_executed_suffix(
                    executed_steps, start, horizon, gamma_value,
                )
                retry_capability_reverify_queries += 1
                if (
                    not result.error
                    and result.valid
                    and result.progress_eligible
                ):
                    return True
            return False

        def episode_sample_update_mode(
            episode_record: dict[str, Any],
        ) -> int | None:
            if config.sample_update_mode is None:
                return None
            if "sample_update_mode" not in episode_record:
                executed_states = [
                    step_record["state_after"]
                    for step_record in episode_record["executed_steps"]
                ]
                mode = sample_update_mode_fn(executed_states)
                if mode is not None:
                    mode = int(mode)
                    if mode < 0:
                        raise ValueError(
                            "successful_trajectory_mode must return None or "
                            "a nonnegative integer"
                        )
                episode_record["sample_update_mode"] = mode
            return episode_record["sample_update_mode"]

        def episode_pair_identity(
            episode_record: dict[str, Any],
        ) -> dict[str, int] | None:
            if pair_identity_fn is None:
                return None
            if "paired_scene_identity" not in episode_record:
                executed_states = [
                    step_record["state_after"]
                    for step_record in episode_record["executed_steps"]
                ]
                identity = pair_identity_fn(executed_states)
                if identity is not None:
                    identity = {
                        "pair_slot": int(identity["pair_slot"]),
                        "label": int(identity["label"]),
                        "version": int(identity["version"]),
                    }
                    if identity["label"] // 2 != identity["pair_slot"]:
                        raise RuntimeError(
                            "successful trajectory pair identity is inconsistent"
                        )
                episode_record["paired_scene_identity"] = identity
            return episode_record["paired_scene_identity"]

        def is_current_pair_success(
            episode_record: dict[str, Any],
            gamma_value: float,
        ) -> bool:
            if episode_record.get("paired_scene_quota_invalidated") is not None:
                return False
            if pair_current_version_fn is None:
                return True
            identity = episode_pair_identity(episode_record)
            if identity is None:
                return False
            current_version = int(pair_current_version_fn(
                gamma_value, identity["pair_slot"],
            ))
            return identity["version"] == current_version

        def missing_sample_update_modes(
            gamma_episodes: Sequence[dict[str, Any]],
        ) -> list[int]:
            if config.sample_update_mode is None:
                return []
            available: dict[int, int] = {}
            for episode_record in gamma_episodes:
                if not (
                    episode_record["status"] == "SUCCESS"
                    and bool(episode_record.get("commit_capable", False))
                    and is_current_pair_success(
                        episode_record, float(episode_record["gamma"]),
                    )
                ):
                    continue
                mode = episode_sample_update_mode(episode_record)
                if mode is not None:
                    available[mode] = available.get(mode, 0) + 1
            missing = []
            requested_modes = _sample_update_modes_for(
                config, round_i, float(gamma_episodes[0]["gamma"]),
            ) if gamma_episodes else ()
            for requested_mode in requested_modes or ():
                if available.get(requested_mode, 0) > 0:
                    available[requested_mode] -= 1
                else:
                    missing.append(requested_mode)
            return missing

        def sample_update_cohort_mode_counts(
            gamma_episodes: Sequence[dict[str, Any]],
        ) -> dict[str, list[int]]:
            label_count = max(config.sample_update_mode) + 1
            counts = {
                "guided": [0] * label_count,
                "unguided": [0] * label_count,
            }
            for episode_record in gamma_episodes:
                cohort = episode_record.get("sample_update_cohort")
                if cohort not in counts or not (
                    episode_record["status"] == "SUCCESS"
                    and bool(episode_record.get("commit_capable", False))
                    and is_current_pair_success(
                        episode_record, float(episode_record["gamma"]),
                    )
                ):
                    continue
                mode = episode_sample_update_mode(episode_record)
                if mode is not None:
                    counts[cohort][mode] += 1
            return counts

        def maybe_replace_incomplete_pair_slots(
            gamma_value: float,
            gamma_episodes: Sequence[dict[str, Any]],
            completed_batch: int,
        ) -> list[dict[str, Any]]:
            threshold = config.paired_scene_replace_after_retry_batches
            if (
                threshold is None
                or pair_replace_fn is None
                or completed_batch < threshold
            ):
                return []
            active_modes = _sample_update_modes_for(
                config, round_i, gamma_value,
            )
            label_count = len(active_modes or ())
            terminal_counts = [0] * label_count
            commit_capable_counts = [0] * label_count
            current_successes = []
            for episode_record in gamma_episodes:
                if (
                    episode_record["status"] != "SUCCESS"
                    or not is_current_pair_success(
                        episode_record, gamma_value,
                    )
                ):
                    continue
                mode = episode_sample_update_mode(episode_record)
                if mode is None:
                    raise RuntimeError(
                        "paired replacement found an out-of-range success label"
                    )
                if not 0 <= mode < label_count:
                    global_label_count = max(config.sample_update_mode) + 1
                    if 0 <= mode < global_label_count:
                        # A scheduled per-round quota can use only an even
                        # prefix of the globally configured pair labels.  The
                        # fixed-size parallel cohort still rolls out the
                        # remaining labels; those successes are deliberately
                        # outside this round/gamma's quota and replacement
                        # bookkeeping.
                        continue
                    raise RuntimeError(
                        "paired replacement found an out-of-range success label"
                    )
                terminal_counts[mode] += 1
                commit_capable_counts[mode] += int(bool(
                    episode_record.get("commit_capable", False)
                ))
                current_successes.append(episode_record)
            per_slot_counts = {
                slot: pair_replacement_counts.get(
                    (float(gamma_value), slot), 0,
                )
                for slot in range(label_count // 2)
            }
            eligible_slots = _incomplete_pair_slots(
                commit_capable_counts,
                per_slot_counts,
                config.paired_scene_max_replacements_per_slot,
            )
            eligible_slots = [
                slot for slot in eligible_slots
                if _pair_replacement_interval_elapsed(
                    pair_replacement_last_batch.get(
                        (float(gamma_value), slot)
                    ),
                    completed_batch,
                    config.paired_scene_replace_interval_batches,
                )
            ]
            replacement_events = []
            for pair_slot in eligible_slots:
                before_terminal = terminal_counts[
                    2 * pair_slot:2 * pair_slot + 2
                ]
                before_commit_capable = commit_capable_counts[
                    2 * pair_slot:2 * pair_slot + 2
                ]
                replacement = dict(pair_replace_fn(
                    gamma_value, pair_slot,
                ))
                invalidated_terminal, invalidated_commit_capable = (
                    _invalidate_pair_slot_success_candidates(
                        current_successes,
                        pair_slot,
                        episode_sample_update_mode,
                        replacement,
                    )
                )
                pair_replacement_counts[(
                    float(gamma_value), pair_slot,
                )] = per_slot_counts[pair_slot] + 1
                pair_replacement_last_batch[(
                    float(gamma_value), pair_slot,
                )] = int(completed_batch)
                replacement.update({
                    "trigger_completed_retry_batch": int(completed_batch),
                    "trigger_retry_batches_launched": int(
                        retry_batches_by_gamma[float(gamma_value)]
                    ),
                    "trigger_terminal_success_counts": list(map(
                        int, before_terminal,
                    )),
                    "trigger_commit_capable_success_counts": list(map(
                        int, before_commit_capable,
                    )),
                    "invalidated_terminal_success_count": int(
                        invalidated_terminal
                    ),
                    "invalidated_commit_capable_success_count": int(
                        invalidated_commit_capable
                    ),
                    "replacement_ordinal_for_slot": int(
                        pair_replacement_counts[
                            (float(gamma_value), pair_slot)
                        ]
                    ),
                    "quota_after_replacement": [
                        2 * pair_slot, 2 * pair_slot + 1,
                    ],
                })
                paired_scene_replacements.append(replacement)
                replacement_events.append(replacement)
                if pair_replacement_callback is not None:
                    pair_replacement_callback(dict(replacement))
            return replacement_events

        for gamma in config.gammas:
            launch_episode_batch(float(gamma))
        ess_values, marginal_ess_values, score_pools, nvp = [], [], [], 0
        sigma_pool_means: list[float] = []
        sigma_selected_means: list[float] = []
        execution_ess_values = []
        verifier_queries = verifier_positives = 0
        verifier_dispatch_s = 0.0
        flow_sampling_s = 0.0
        acquisition_s = 0.0
        retry_verify_all_fast_path_contexts = 0
        verifier_jobs = 0
        verifier_candidates = 0
        context_id = 0

        def consume_verified(
            prepared: dict[str, Any],
            result_block: Sequence[Verification],
        ) -> None:
            """Apply ordered verifier results without changing rollout semantics."""
            nonlocal nvp, verifier_queries, verifier_positives
            episode = prepared["episode"]
            step = prepared["step"]
            gamma = prepared["gamma"]
            gather_rng = prepared["gather_rng"]
            state_before = prepared["state_before"]
            context = prepared["context"]
            flow_bases = prepared["flow_bases"]
            base_candidates = prepared["base_candidates"]
            candidates = prepared["candidates"]
            task_features = prepared["task_features"]
            sigma_k = prepared["sigma_k"]
            selected = prepared["selected"]
            selected_sigma = prepared["selected_sigma"]
            queried = prepared["queried"]
            current_context_id = prepared["context_id"]
            results = list(result_block)
            if len(results) != len(selected):
                raise RuntimeError(
                    "task.verify must return exactly one result per selected query"
                )
            gate_active = (
                config.target_gate_start_round is not None
                and round_i >= config.target_gate_start_round
            )
            current_records = {}
            for local, (candidate, result, acquisition_sigma) in enumerate(zip(
                    queried, results, selected_sigma)):
                if result.error:
                    continue
                verifier_queries += 1
                verifier_positives += int(result.valid)
                if config.replay_acceptance == "safety_valid":
                    replay_eligible = bool(result.valid)
                else:
                    replay_eligible = bool(
                        result.valid
                        and (
                            config.target_gate_start_round is None
                            or (
                                result.progress_eligible
                                and (not gate_active or result.target_eligible)
                            )
                        )
                    )
                record = QueryRecord(
                    round_i, gamma, int(episode["episode"]),
                    current_context_id,
                    context.detach().cpu(), candidate.detach().cpu(), result,
                    acquisition_sigma=float(sigma_k[selected[local]]),
                    flow_base=(
                        flow_bases[selected[local]].detach().cpu()
                        if flow_bases is not None else None
                    ),
                    acquisition_feature=(
                        task_features[selected[local]].detach().cpu()
                        if task_features is not None else None
                    ),
                    replay_eligible=replay_eligible,
                )
                current_records[local] = record
                if (
                    config.archive_rule != "successful_executed_windows"
                    and config.gp_reference_mode == "round1_fixed_frozen_phi"
                    and round_i == 1
                    and replay_eligible
                ):
                    round1_gp_candidates.append(record)
                if (
                    config.archive_rule != "successful_executed_windows"
                    and config.gp_reference_mode
                    == "cumulative_accepted_frozen_phi"
                    and replay_eligible
                ):
                    gp_evidence.append(record)
            eligible = [
                index for index, result in enumerate(results)
                if (
                    not result.error
                    and result.valid
                    and result.progress_eligible
                    and (not gate_active or result.target_eligible)
                )
            ]
            nvp_reason = None
            if not eligible:
                episode["status"] = "NVP"
                nvp += 1
                safe = [
                    result for result in results
                    if not result.error and result.valid
                ]
                forward = [
                    result for result in safe if result.progress_eligible
                ]
                if gate_active and forward and not any(
                        result.target_eligible for result in forward):
                    nvp_reason = "TARGET"
                elif safe and not forward:
                    nvp_reason = "PROGRESS"
                else:
                    nvp_reason = "VERIFIER"
                archived_local = None
                if config.archive_rule == "all_queries":
                    for record in current_records.values():
                        record.nvp_context = True
                        archive.append(record)
                elif config.archive_rule == "executed_plus_nvp_negative":
                    rejected = [
                        local for local, record in current_records.items()
                        if not record.verification.valid
                    ]
                    if rejected:
                        archived_local = min(
                            rejected,
                            key=lambda local: results[local].execution_cost,
                        )
                        current_records[archived_local].nvp_context = True
                        archive.append(current_records[archived_local])
                chosen = None
            else:
                archived_local = None
                if config.execution_rule == "max_margin":
                    chosen = max(
                        eligible, key=lambda index: results[index].margin
                    )
                elif config.execution_rule == "max_step_margin":
                    if any(
                        results[index].step_margin is None
                        for index in eligible
                    ):
                        raise ValueError(
                            "max_step_margin requires verifier step_margin values"
                        )
                    chosen = max(
                        eligible,
                        key=lambda index: float(results[index].step_margin),
                    )
                elif config.execution_rule == "max_uncertainty":
                    chosen = max(
                        eligible,
                        key=lambda index: float(sigma_k[selected[index]]),
                    )
                elif config.execution_rule == "uncertainty_cost":
                    costs = torch.as_tensor(
                        [results[index].execution_cost for index in eligible],
                        dtype=torch.float64,
                    )
                    uncertainties = torch.as_tensor(
                        [
                            float(sigma_k[selected[index]])
                            for index in eligible
                        ],
                        dtype=torch.float64,
                    )
                    cost_span = (
                        costs.max() - costs.min()
                    ).clamp_min(1.0e-12)
                    sigma_span = (
                        uncertainties.max() - uncertainties.min()
                    ).clamp_min(1.0e-12)
                    score = (
                        config.execution_uncertainty_weight
                        * (uncertainties - uncertainties.min()) / sigma_span
                        - (costs - costs.min()) / cost_span
                    )
                    chosen = eligible[int(torch.argmax(score))]
                elif config.execution_rule == "uniform_positive":
                    draw = int(torch.randint(
                        len(eligible), (1,), generator=gather_rng,
                        device=device,
                    ))
                    chosen = eligible[draw]
                elif config.execution_rule == "softmin_cost":
                    draw, execution_ess = _softmin_choice(
                        [
                            results[index].execution_cost
                            for index in eligible
                        ],
                        config.execution_ess_target, gather_rng, device,
                    )
                    execution_ess_values.append(execution_ess)
                    chosen = eligible[draw]
                else:
                    if (
                        config.execution_step_margin_weight > 0.0
                        and any(
                            results[index].step_margin is None
                            for index in eligible
                        )
                    ):
                        raise ValueError(
                            "execution_step_margin_weight requires verifier "
                            "step_margin values"
                        )
                    if config.execution_cost_band_fraction > 0.0:
                        eligible_costs = {
                            index: float(results[index].execution_cost)
                            for index in eligible
                        }
                        local_choice = bounded_regret_step_margin_choice(
                            [eligible_costs[index] for index in eligible],
                            [
                                float(results[index].step_margin or 0.0)
                                for index in eligible
                            ],
                            config.execution_cost_band_fraction,
                        )
                        chosen = eligible[local_choice]
                        execution_key = eligible_costs.__getitem__
                    else:
                        execution_key = lambda index: (  # noqa: E731
                            results[index].execution_cost
                            - config.execution_step_margin_weight
                            * float(results[index].step_margin or 0.0)
                        )
                        chosen = min(eligible, key=execution_key)
                    # RESEARCH hook (default None -> behavior above is exact):
                    # lets an experiment re-pick among the SAME verified,
                    # progress-eligible candidates, e.g. to allocate parallel
                    # episodes across route modes. Safety is untouched -- the
                    # hook can only choose candidates that already passed the
                    # full verifier.
                    if EXECUTION_SELECTION_HOOK is not None:
                        override = EXECUTION_SELECTION_HOOK(
                            episode=episode,
                            step=step,
                            gamma=gamma,
                            round_i=round_i,
                            eligible=eligible,
                            queried=queried,
                            execution_key=execution_key,
                            chosen=chosen,
                        )
                        if override is not None:
                            override = int(override)
                            if override not in eligible:
                                raise ValueError(
                                    "EXECUTION_SELECTION_HOOK must return an "
                                    "eligible candidate index or None"
                                )
                            chosen = override
                if config.archive_rule == "all_queries":
                    archive.extend(current_records.values())
                elif config.archive_rule != "successful_executed_windows":
                    archive.append(current_records[chosen])
                staged_executed_step = None
                if config.archive_rule == "successful_executed_windows":
                    executed_candidate = queried[chosen].detach()
                    if (
                        executed_candidate.ndim < 2
                        or len(executed_candidate) < 1
                    ):
                        raise ValueError(
                            "successful_executed_windows requires candidates "
                            "with an explicit leading horizon dimension"
                        )
                    staged_executed_step = {
                        "context_id": current_context_id,
                        "context": context.detach().cpu(),
                        "action": executed_candidate[0].detach().cpu(),
                        "plan_horizon": int(len(executed_candidate)),
                    }
                    if config.paired_noised_representation:
                        if flow_bases is None:
                            raise RuntimeError(
                                "paired successful-window replay requires sampled "
                                "flow bases"
                            )
                        staged_executed_step["flow_base_action"] = (
                            flow_bases[selected[chosen]][0].detach().cpu()
                        )
                episode["state"] = task.advance(
                    episode["state"], queried[chosen]
                )
                if staged_executed_step is not None:
                    staged_executed_step["state_after"] = copy.deepcopy(
                        episode["state"]
                    )
                    episode["executed_steps"].append(staged_executed_step)
                episode["status"] = task.terminal(episode["state"])
            if event_callback is not None:
                event = {
                    "round": round_i, "step": step, "gamma": gamma,
                    "episode": int(episode["episode"]),
                    "context_id": current_context_id,
                    "retry_batch": int(episode["retry_batch"]),
                    "replica": int(episode["replica"]),
                    "sample_update_cohort": episode["sample_update_cohort"],
                    "flow_base_std": float(prepared["flow_base_std"]),
                    "state_before": state_before,
                    "state_after": episode["state"],
                    "context": context.detach().cpu(),
                    "base_candidates": base_candidates.detach().cpu(),
                    "flow_bases": (
                        flow_bases.detach().cpu()
                        if flow_bases is not None else None
                    ),
                    "candidates": candidates.detach().cpu(),
                    "sigma_K": sigma_k,
                    "selected": selected,
                    "selected_sigma": selected_sigma,
                    "verification": [asdict(result) for result in results],
                    "chosen_local": chosen,
                    "archived_negative_local": archived_local,
                    "status": episode["status"],
                    "nvp_reason": nvp_reason,
                    "target_gate_active": gate_active,
                }
                if task_features is not None:
                    event["task_acquisition_features_K"] = (
                        task_features.detach().cpu()
                    )
                event_callback(event)
            episode["step"] = step + 1
            if (
                episode["status"] is None
                and int(episode["step"]) >= config.max_steps
            ):
                episode["status"] = "TIMEOUT"

        while True:
            active = [episode for episode in episodes if episode["status"] is None]
            if not active:
                if config.archive_rule != "successful_executed_windows":
                    break
                launched = False
                for gamma in map(float, config.gammas):
                    if gamma in completed_retry_gammas:
                        continue
                    gamma_episodes = [
                        episode for episode in episodes
                        if float(episode["gamma"]) == gamma
                    ]
                    for episode in gamma_episodes:
                        if (
                            episode["status"] == "SUCCESS"
                            and "commit_capable" not in episode
                        ):
                            episode["commit_capable"] = has_commit_capable_window(
                                episode, gamma
                            )
                    completed_batch = retry_batches_by_gamma[gamma] - 1
                    replacement_events = maybe_replace_incomplete_pair_slots(
                        gamma, gamma_episodes, completed_batch,
                    )
                    commit_capable_successes = sum(
                        episode["status"] == "SUCCESS"
                        and bool(episode.get("commit_capable", False))
                        and is_current_pair_success(episode, gamma)
                        for episode in gamma_episodes
                    )
                    missing_modes = missing_sample_update_modes(gamma_episodes)
                    quota_complete = (
                        not missing_modes
                        if config.sample_update_mode is not None
                        else commit_capable_successes
                        >= _successful_trajectory_quota_for(
                            config, round_i, gamma,
                        )
                    )
                    if (
                        config.sample_update_mode is not None
                        and retry_callback is not None
                        and (
                            completed_batch % 4 == 0
                            or bool(replacement_events)
                        )
                    ):
                        retry_callback({
                            "round": round_i,
                            "gamma": gamma,
                            "retry_batch": completed_batch,
                            "max_retry_batches": config.max_retry_batches,
                            "retry_batch_cap": (
                                config.retry_resample_batch_cap
                                if config.retry_exhaustion_policy
                                == "resample_scene"
                                else config.max_retry_batches
                            ),
                            "needed_modes": list(missing_modes),
                            "cohort_mode_counts": (
                                sample_update_cohort_mode_counts(gamma_episodes)
                            ),
                            "paired_scene_replacements": list(
                                replacement_events
                            ),
                        })
                    if quota_complete:
                        completed_retry_gammas.add(gamma)
                    elif retry_batches_by_gamma[gamma] < config.max_retry_batches:
                        launch_episode_batch(gamma)
                        launched = True
                    elif (
                        config.retry_exhaustion_policy == "resample_scene"
                        and retry_batches_by_gamma[gamma]
                        < config.retry_resample_batch_cap
                    ):
                        # Instead of aborting the round, keep launching whole
                        # episode batches; the reset seed stream is keyed on the
                        # retry-batch counter, so every extra batch draws fresh
                        # independently sampled scenes for this gamma.
                        launch_episode_batch(gamma)
                        launched = True
                    else:
                        completed_retry_gammas.add(gamma)
                        exhausted_retry_gammas.add(gamma)
                if not launched:
                    break
                active = [
                    episode for episode in episodes
                    if episode["status"] is None
                ]
            episode_inputs: list[dict[str, Any]] = []
            for episode in active:
                step = int(episode["step"])
                gamma = float(episode["gamma"])
                if config.archive_rule == "successful_executed_windows":
                    gather_rng = torch.Generator(device=device).manual_seed(
                        _counter_seed(
                            config.seed, "gather", round_i,
                            int(episode["retry_batch"]),
                            int(episode["replica"]), step,
                        )
                    )
                else:
                    gather_rng = torch_rng
                state_before = episode["state"]
                context = task.context(episode["state"], gamma).detach().to(device)
                episode_inputs.append({
                    "episode": episode,
                    "step": step,
                    "gamma": gamma,
                    "gather_rng": gather_rng,
                    "state_before": state_before,
                    "context": context,
                })
            batched_candidates = batched_bases = None
            if config.batched_rollout_sampling:
                sampler = getattr(policy, "sample_many_with_base", None)
                if not callable(sampler):
                    raise TypeError(
                        "batched_rollout_sampling requires policy."
                        "sample_many_with_base(contexts, count, generators)"
                    )
                sample_started = time.perf_counter()
                batched_candidates, batched_bases = sampler(
                    torch.stack([
                        prepared["context"] for prepared in episode_inputs
                    ]),
                    config.K,
                    [prepared["gather_rng"] for prepared in episode_inputs],
                    base_std=config.flow_base_std,
                )
                flow_sampling_s += time.perf_counter() - sample_started
            candidate_blocks: list[dict[str, Any]] = []
            for input_index, prepared_input in enumerate(episode_inputs):
                episode = prepared_input["episode"]
                step = prepared_input["step"]
                gamma = prepared_input["gamma"]
                gather_rng = prepared_input["gather_rng"]
                state_before = prepared_input["state_before"]
                context = prepared_input["context"]
                episode_flow_base_std = config.flow_base_std
                flow_bases = None
                if batched_candidates is not None:
                    base_candidates = batched_candidates[input_index]
                    if config.paired_noised_representation:
                        flow_bases = batched_bases[input_index]
                elif config.paired_noised_representation:
                    sampler = getattr(policy, "sample_with_base", None)
                    if not callable(sampler):
                        raise TypeError(
                            "paired_noised_representation requires "
                            "policy.sample_with_base(context, count, generator) "
                            "returning (plans, initial_bases)"
                        )
                    sample_started = time.perf_counter()
                    if episode_flow_base_std == 1.0:
                        base_candidates, flow_bases = sampler(
                            context, config.K, gather_rng
                        )
                    else:
                        base_candidates, flow_bases = sampler(
                            context, config.K, gather_rng,
                            base_std=episode_flow_base_std,
                        )
                    flow_sampling_s += time.perf_counter() - sample_started
                    base_candidates = base_candidates.detach()
                    flow_bases = flow_bases.detach()
                    if (len(flow_bases) != config.K
                            or flow_bases.reshape(config.K, -1).shape[1]
                            != base_candidates.reshape(config.K, -1).shape[1]):
                        raise ValueError(
                            "sample_with_base must return one plan-shaped initial "
                            "base per sampled candidate"
                        )
                else:
                    sample_started = time.perf_counter()
                    if episode_flow_base_std == 1.0:
                        base_candidates = policy.sample(
                            context, config.K, gather_rng
                        ).detach()
                    else:
                        base_candidates = policy.sample(
                            context, config.K, gather_rng,
                            base_std=episode_flow_base_std,
                        ).detach()
                    flow_sampling_s += time.perf_counter() - sample_started
                candidates = perturb_plan_candidates(
                    policy, base_candidates, config.candidate_perturb_std, gather_rng,
                    config.candidate_perturb_scope,
                ).detach()
                if CANDIDATE_PROPOSAL_HOOK is not None:
                    proposed = CANDIDATE_PROPOSAL_HOOK(
                        episode=episode,
                        step=step,
                        gamma=gamma,
                        round_i=round_i,
                        context=context,
                        state_before=state_before,
                        candidates=candidates,
                        base_candidates=base_candidates,
                        flow_bases=flow_bases,
                    )
                    if proposed is not None:
                        if not isinstance(proposed, torch.Tensor):
                            raise TypeError(
                                "CANDIDATE_PROPOSAL_HOOK must return a "
                                "torch.Tensor or None"
                            )
                        if (
                            proposed.shape != candidates.shape
                            or proposed.dtype != candidates.dtype
                            or proposed.device != candidates.device
                        ):
                            raise ValueError(
                                "CANDIDATE_PROPOSAL_HOOK must preserve candidate "
                                "shape, dtype, and device"
                            )
                        if not torch.isfinite(proposed).all():
                            raise ValueError(
                                "CANDIDATE_PROPOSAL_HOOK returned nonfinite values"
                            )
                        candidates = proposed.detach()
                candidate_blocks.append({
                    **prepared_input,
                    "flow_base_std": episode_flow_base_std,
                    "flow_bases": flow_bases,
                    "base_candidates": base_candidates,
                    "candidates": candidates,
                })
            batched_features = None
            if (
                config.batched_rollout_sampling
                and config.acquisition_feature == "learned_phi"
            ):
                embedder = getattr(acquisition_policy, "embed_many", None)
                if not callable(embedder):
                    raise TypeError(
                        "batched_rollout_sampling requires acquisition policy."
                        "embed_many(contexts, candidates, bases)"
                    )
                embed_started = time.perf_counter()
                block_bases = (
                    torch.stack([
                        prepared["flow_bases"]
                        for prepared in candidate_blocks
                    ])
                    if candidate_blocks[0]["flow_bases"] is not None
                    else None
                )
                batched_features = embedder(
                    torch.stack([
                        prepared["context"] for prepared in candidate_blocks
                    ]),
                    torch.stack([
                        prepared["candidates"] for prepared in candidate_blocks
                    ]),
                    bases=block_bases,
                )
                acquisition_s += time.perf_counter() - embed_started
            prepared_blocks: list[dict[str, Any]] = []
            for block_index, candidate_block in enumerate(candidate_blocks):
                used_retry_fast_path = False
                episode = candidate_block["episode"]
                step = candidate_block["step"]
                gamma = candidate_block["gamma"]
                gather_rng = candidate_block["gather_rng"]
                state_before = candidate_block["state_before"]
                context = candidate_block["context"]
                flow_bases = candidate_block["flow_bases"]
                base_candidates = candidate_block["base_candidates"]
                candidates = candidate_block["candidates"]
                episode_flow_base_std = candidate_block["flow_base_std"]
                acquisition_started = time.perf_counter()
                task_features = None
                if (config.acquisition_feature == "task_angular"
                        or config.coverage_replay == "circular_voronoi"):
                    task_features = _task_angular_features(
                        task, context, candidates, gamma
                    )
                if config.acquisition_feature == "task_angular":
                    features = task_features
                    acquisition_gp = gp_by_gamma[gamma]
                else:
                    features = (
                        batched_features[block_index]
                        if batched_features is not None
                        else (
                            acquisition_policy.embed(
                                context, candidates, base=flow_bases
                            )
                            if flow_bases is not None
                            else acquisition_policy.embed(
                                context, candidates
                            )
                        )
                    )
                    if (
                        config.gp_reference_mode in {
                            "cumulative_success_per_gamma_frozen_phi_exact",
                            "sliding_success_per_gamma_frozen_phi",
                            "sliding_success_per_gamma_current_phi",
                        }
                    ):
                        acquisition_gp = gp_by_gamma[gamma]
                    else:
                        if shared_gp is None:
                            raise RuntimeError(
                                "learned_phi acquisition GP was not initialized"
                            )
                        acquisition_gp = shared_gp
                query_count = (
                    config.retry_B
                    if (
                        config.retry_B is not None
                        and int(episode["retry_batch"]) > 0
                    )
                    else config.B
                )
                if (
                    config.retry_verify_all_fast_path
                    and int(episode["retry_batch"]) > 0
                    and query_count == len(features)
                ):
                    # Every candidate is queried, deterministic execution is
                    # acquisition-order invariant, and the validated config
                    # forbids consumers that need retry sigma values.  Avoid
                    # the otherwise dominant 1,536-row GP posterior solve.
                    selected = list(range(len(features)))
                    sigma_k_device = torch.zeros(
                        len(features), dtype=features.dtype,
                        device=features.device,
                    )
                    selected_sigma = [0.0] * len(selected)
                    ess = [1.0] * len(selected)
                    used_retry_fast_path = True
                else:
                    (
                        selected,
                        selected_sigma,
                        ess,
                        sigma_k_device,
                    ) = acquisition_gp.acquire_with_sigma(
                        features, query_count, beta_used, gather_rng
                    )
                if used_retry_fast_path:
                    retry_verify_all_fast_path_contexts += 1
                sigma_k = sigma_k_device.detach().cpu()
                score_pools.append(sigma_k)
                marginal_ess_values.append(normalized_ess(sigma_k, beta_used))
                ess_values.extend(ess)
                sigma_pool_means.append(float(sigma_k.mean()))
                sigma_selected_means.append(
                    float(sigma_k[torch.as_tensor(selected, dtype=torch.long)].mean())
                )
                acquisition_s += time.perf_counter() - acquisition_started
                queried = candidates[selected]
                prepared_blocks.append({
                    "episode": episode,
                    "step": step,
                    "gamma": gamma,
                    "flow_base_std": episode_flow_base_std,
                    "gather_rng": gather_rng,
                    "state_before": state_before,
                    "context": context,
                    "flow_bases": flow_bases,
                    "base_candidates": base_candidates,
                    "candidates": candidates,
                    "task_features": task_features,
                    "sigma_k": sigma_k,
                    "selected": selected,
                    "selected_sigma": selected_sigma,
                    "queried": queried,
                    "context_id": context_id,
                })
                context_id += 1
            verify_started = time.perf_counter()
            verified_blocks = verifier.verify_many([
                (prepared["context"], prepared["queried"], prepared["gamma"])
                for prepared in prepared_blocks
            ])
            verifier_dispatch_s += time.perf_counter() - verify_started
            verifier_jobs += len(prepared_blocks)
            verifier_candidates += sum(
                len(prepared["queried"]) for prepared in prepared_blocks
            )
            for prepared, results in zip(prepared_blocks, verified_blocks):
                consume_verified(prepared, results)
        gather_s = time.perf_counter() - gather_started
        statuses = [episode["status"] or "TIMEOUT" for episode in episodes]
        if (
            config.archive_rule == "successful_executed_windows"
            and exhausted_retry_gammas
            and config.retry_exhaustion_policy != "commit_available"
        ):
            missing = ", ".join(
                f"{gamma:g}" for gamma in sorted(exhausted_retry_gammas)
            )
            missing_quota = ""
            if config.sample_update_mode is not None:
                parts = []
                for gamma in sorted(exhausted_retry_gammas):
                    gamma_episodes = [
                        episode for episode in episodes
                        if float(episode["gamma"]) == gamma
                    ]
                    needed = ",".join(map(
                        str, missing_sample_update_modes(gamma_episodes),
                    ))
                    requested = _successful_trajectory_quota_for(
                        config, round_i, gamma,
                    )
                    parts.append(
                        f'gamma={gamma:g} quota={requested} need mode "{needed}"'
                    )
                missing_quota = "; " + "; ".join(parts)
            if config.retry_exhaustion_policy == "resample_scene":
                raise RuntimeError(
                    f"round {round_i} exhausted retry_resample_batch_cap="
                    f"{config.retry_resample_batch_cap} fresh-scene batches "
                    f"(max_retry_batches={config.max_retry_batches}) before "
                    "obtaining the scheduled "
                    "commit-capable terminal SUCCESS trajectories for "
                    f"gamma(s): {missing}{missing_quota}; round not committed"
                )
            raise RuntimeError(
                f"round {round_i} exhausted max_retry_batches="
                f"{config.max_retry_batches} before obtaining "
                "the scheduled commit-capable "
                "terminal SUCCESS trajectories for "
                f"gamma(s): {missing}{missing_quota}; round not committed"
            )
        successful_executed_commit_by_gamma: dict[str, dict[str, Any]] = {}
        committed_trajectory_ids: list[str] = []
        committed_window_ids: list[str] = []
        commit_reverify_queries = 0
        commit_reverify_valid = 0
        commit_reverify_progress = 0
        if config.archive_rule == "successful_executed_windows":
            committed_records: list[QueryRecord] = []
            for gamma in config.gammas:
                gamma = float(gamma)
                requested_trajectory_count = _successful_trajectory_quota_for(
                    config, round_i, gamma,
                )
                requested_sample_update_modes = _sample_update_modes_for(
                    config, round_i, gamma,
                )
                successful = sorted(
                    (
                        episode for episode in episodes
                        if float(episode["gamma"]) == gamma
                        and episode["status"] == "SUCCESS"
                        and bool(episode.get("commit_capable", False))
                        and is_current_pair_success(episode, gamma)
                    ),
                    key=lambda episode: int(episode["episode"]),
                )
                above_fraction_fn = getattr(
                    task, "successful_trajectory_above_fraction", None
                )
                mean_z_fn = getattr(
                    task, "successful_trajectory_mean_z", None
                )
                above_fractions: dict[int, float | None] = {}
                mean_z_values: dict[int, float | None] = {}
                for successful_episode in successful:
                    episode_id = int(successful_episode["episode"])
                    if callable(above_fraction_fn):
                        above_fraction = float(above_fraction_fn([
                            step_record["state_after"]
                            for step_record in successful_episode["executed_steps"]
                        ]))
                        if not math.isfinite(above_fraction):
                            raise ValueError(
                                "successful trajectory above fraction must be finite"
                            )
                        above_fractions[episode_id] = above_fraction
                    else:
                        above_fractions[episode_id] = None
                    if callable(mean_z_fn):
                        mean_z = float(mean_z_fn([
                            step_record["state_after"]
                            for step_record in successful_episode["executed_steps"]
                        ]))
                        if not math.isfinite(mean_z):
                            raise ValueError(
                                "successful trajectory mean z must be finite"
                            )
                        mean_z_values[episode_id] = mean_z
                    else:
                        mean_z_values[episode_id] = None
                sample_update_modes = {
                    int(successful_episode["episode"]): (
                        episode_sample_update_mode(successful_episode)
                        if config.sample_update_mode is not None else None
                    )
                    for successful_episode in successful
                }
                sample_update_cohorts = {
                    int(successful_episode["episode"]): successful_episode.get(
                        "sample_update_cohort"
                    )
                    for successful_episode in successful
                }
                paired_scene_identities = {
                    int(successful_episode["episode"]): (
                        episode_pair_identity(successful_episode)
                        if pair_identity_fn is not None else None
                    )
                    for successful_episode in successful
                }
                if (
                    successful
                    and config.successful_trajectory_selector
                    == "max_above_fraction"
                ):
                    if not callable(above_fraction_fn):
                        raise TypeError(
                            "max_above_fraction requires task."
                            "successful_trajectory_above_fraction"
                        )
                    successful.sort(key=lambda episode: (
                        -float(above_fractions[int(episode["episode"])]),
                        int(episode["episode"]),
                    ))
                elif (
                    successful
                    and config.successful_trajectory_selector == "max_mean_z"
                ):
                    if not callable(mean_z_fn):
                        raise TypeError(
                            "max_mean_z requires task."
                            "successful_trajectory_mean_z"
                        )
                    successful.sort(key=lambda episode: (
                        -float(mean_z_values[int(episode["episode"])]),
                        int(episode["episode"]),
                    ))
                elif (
                    successful
                    and config.successful_trajectory_selector == "random_success"
                ):
                    successful.sort(key=lambda episode: (
                        _counter_seed(
                            config.seed, "successful_trajectory_selector",
                            round_i, f"{gamma:.9g}",
                            int(episode["episode"]),
                        ),
                        int(episode["episode"]),
                    ))
                committed_successful = successful[
                    :requested_trajectory_count
                ]
                if config.sample_update_mode is not None:
                    remaining: dict[int, int] = {}
                    for requested_mode in requested_sample_update_modes or ():
                        remaining[requested_mode] = (
                            remaining.get(requested_mode, 0) + 1
                        )
                    committed_successful = []
                    for successful_episode in successful:
                        mode = sample_update_modes[
                            int(successful_episode["episode"])
                        ]
                        if mode is not None and remaining.get(mode, 0) > 0:
                            committed_successful.append(successful_episode)
                            remaining[mode] -= 1
                detail: dict[str, Any] = {
                    "selector": config.successful_trajectory_selector,
                    "requested_trajectory_count": (
                        requested_trajectory_count
                    ),
                    "quota_complete": (
                        len(committed_successful)
                        >= requested_trajectory_count
                    ),
                    "commit_available_policy": (
                        config.retry_exhaustion_policy == "commit_available"
                    ),
                    "retry_batches_used": retry_batches_by_gamma[gamma],
                    "attempted_episode_count": sum(
                        float(episode["gamma"]) == gamma for episode in episodes
                    ),
                    "success_episode_count": len(successful),
                    "success_episode_above_fractions": {
                        str(episode_id): above_fractions[episode_id]
                        for episode_id in sorted(above_fractions)
                    },
                    "success_episode_mean_z": {
                        str(episode_id): mean_z_values[episode_id]
                        for episode_id in sorted(mean_z_values)
                    },
                    "committed_trajectory_id": None,
                    "committed_episode_id": None,
                    "successful_retry_batch": None,
                    "above_fraction": None,
                    "mean_z": None,
                    "executed_steps": 0,
                    "candidate_window_count": 0,
                    "full_h_valid_window_count": 0,
                    "committed_window_count": 0,
                    "committed_window_ids": [],
                    "committed_trajectory_count": 0,
                    "committed_trajectory_ids": [],
                    "committed_episode_ids": [],
                    "committed_trajectories": [],
                    "all_committed_window_count": 0,
                    "all_committed_window_ids": [],
                }
                if config.sample_update_mode is not None:
                    detail.update({
                        "sample_update_mode": list(
                            requested_sample_update_modes or ()
                        ),
                        "success_episode_sample_update_modes": {
                            str(episode_id): sample_update_modes[episode_id]
                            for episode_id in sorted(sample_update_modes)
                        },
                        "committed_sample_update_modes": [],
                        "success_episode_sample_update_cohorts": {
                            str(episode_id): sample_update_cohorts[episode_id]
                            for episode_id in sorted(sample_update_cohorts)
                        },
                        "committed_sample_update_cohorts": [],
                        "success_episode_paired_scene_identities": {
                            str(episode_id): paired_scene_identities[episode_id]
                            for episode_id in sorted(paired_scene_identities)
                        },
                        "committed_paired_scene_identities": [],
                    })
                if (
                    len(committed_successful)
                    < requested_trajectory_count
                    and config.retry_exhaustion_policy != "commit_available"
                ):
                    raise RuntimeError(
                        f"round {round_i} has only {len(committed_successful)} "
                        "quota-eligible ranked commit-capable terminal SUCCESS "
                        "trajectories for "
                        f"gamma={gamma:g}, fewer than requested "
                        f"{requested_trajectory_count}; "
                        "round not committed"
                    )
                for rank, committed_episode in enumerate(
                    committed_successful
                ):
                    episode_id = int(committed_episode["episode"])
                    trajectory_id = _successful_trajectory_id(
                        round_i, gamma, episode_id
                    )
                    committed_trajectory_ids.append(trajectory_id)
                    detail["committed_trajectory_ids"].append(trajectory_id)
                    detail["committed_episode_ids"].append(episode_id)
                    if config.sample_update_mode is not None:
                        detail["committed_sample_update_modes"].append(
                            sample_update_modes[episode_id]
                        )
                        detail["committed_sample_update_cohorts"].append(
                            sample_update_cohorts[episode_id]
                        )
                        detail[
                            "committed_paired_scene_identities"
                        ].append(paired_scene_identities[episode_id])
                    trajectory_detail: dict[str, Any] = {
                        "rank": rank,
                        "trajectory_id": trajectory_id,
                        "episode_id": episode_id,
                        "successful_retry_batch": int(
                            committed_episode["retry_batch"]
                        ),
                        "above_fraction": above_fractions[episode_id],
                        "mean_z": mean_z_values[episode_id],
                        "executed_steps": 0,
                        "candidate_window_count": 0,
                        "full_h_valid_window_count": 0,
                        "committed_window_count": 0,
                        "committed_window_ids": [],
                    }
                    if config.sample_update_mode is not None:
                        trajectory_detail["sample_update_mode"] = (
                            sample_update_modes[episode_id]
                        )
                        trajectory_detail["sample_update_cohort"] = (
                            sample_update_cohorts[episode_id]
                        )
                        trajectory_detail["paired_scene_identity"] = (
                            paired_scene_identities[episode_id]
                        )
                    executed_steps = committed_episode["executed_steps"]
                    trajectory_detail["executed_steps"] = len(executed_steps)
                    if executed_steps:
                        horizons = {
                            int(step_record["plan_horizon"])
                            for step_record in executed_steps
                        }
                        if len(horizons) != 1:
                            raise RuntimeError(
                                "executed candidates changed horizon within one "
                                "successful trajectory"
                            )
                        horizon = horizons.pop()
                        action_shapes = {
                            tuple(step_record["action"].shape)
                            for step_record in executed_steps
                        }
                        if len(action_shapes) != 1:
                            raise RuntimeError(
                                "executed action shape changed within one "
                                "successful trajectory"
                            )
                        window_count = len(executed_steps)
                        trajectory_detail["candidate_window_count"] = (
                            window_count
                        )
                        for start in range(window_count):
                            context_record = executed_steps[start]
                            (
                                context_window,
                                actual_window,
                                candidate_window,
                                loss_mask,
                                result,
                            ) = reverify_executed_suffix(
                                executed_steps, start, horizon, gamma,
                            )
                            valid_horizon = len(actual_window)
                            paired_window_base = paired_executed_suffix_base(
                                executed_steps, start, horizon,
                            )
                            commit_reverify_queries += 1
                            if result.error:
                                continue
                            if result.valid:
                                trajectory_detail[
                                    "full_h_valid_window_count"
                                ] += 1
                                commit_reverify_valid += 1
                            if not (result.valid and result.progress_eligible):
                                continue
                            commit_reverify_progress += 1
                            with torch.no_grad():
                                task_feature = None
                                if (
                                    config.acquisition_feature == "task_angular"
                                    or config.coverage_replay == "circular_voronoi"
                                ):
                                    task_feature = _task_angular_features(
                                        task, context_window,
                                        actual_window.unsqueeze(0), gamma,
                                    )
                                if config.acquisition_feature == "task_angular":
                                    reconstructed_feature = task_feature
                                    reconstructed_gp = gp_by_gamma[gamma]
                                else:
                                    reconstructed_feature = acquisition_policy.embed(
                                        context_window.unsqueeze(0),
                                        candidate_window.unsqueeze(0),
                                        base=(
                                            paired_window_base.unsqueeze(0)
                                            if paired_window_base is not None
                                            else None
                                        ),
                                    )
                                    if (
                                        config.gp_reference_mode in {
                                            "cumulative_success_per_gamma_frozen_phi_exact",
                                            "sliding_success_per_gamma_frozen_phi",
                                            "sliding_success_per_gamma_current_phi",
                                        }
                                    ):
                                        reconstructed_gp = gp_by_gamma[gamma]
                                    else:
                                        if shared_gp is None:
                                            raise RuntimeError(
                                                "learned_phi acquisition GP was "
                                                "not initialized"
                                            )
                                        reconstructed_gp = shared_gp
                                acquisition_sigma = float(
                                    reconstructed_gp.sigma(
                                        reconstructed_feature
                                    )[0].detach().cpu()
                                )
                            window_id = _successful_window_id(
                                trajectory_id, start
                            )
                            committed_window_ids.append(window_id)
                            trajectory_detail["committed_window_ids"].append(
                                window_id
                            )
                            detail["all_committed_window_ids"].append(window_id)
                            committed_records.append(QueryRecord(
                                round_i, gamma, episode_id,
                                int(context_record["context_id"]),
                                context_window.detach().cpu(),
                                candidate_window.detach().cpu(),
                                result,
                                acquisition_sigma=acquisition_sigma,
                                flow_base=(
                                    paired_window_base.detach().cpu()
                                    if paired_window_base is not None else None
                                ),
                                acquisition_feature=(
                                    task_feature[0].detach().cpu()
                                    if task_feature is not None else None
                                ),
                                replay_eligible=True,
                                retry_batch=int(
                                    committed_episode["retry_batch"]
                                ),
                                replica=int(committed_episode["replica"]),
                                trajectory_id=trajectory_id,
                                window_id=window_id,
                                window_start=start,
                                valid_horizon=valid_horizon,
                                loss_mask=loss_mask.detach().cpu(),
                                sample_update_mode=(
                                    sample_update_modes[episode_id]
                                    if config.sample_update_mode is not None
                                    else None
                                ),
                            ))
                    trajectory_detail["committed_window_count"] = len(
                        trajectory_detail["committed_window_ids"]
                    )
                    detail["committed_trajectories"].append(
                        trajectory_detail
                    )
                    if rank == 0:
                        detail["committed_trajectory_id"] = trajectory_id
                        detail["committed_episode_id"] = episode_id
                        detail["successful_retry_batch"] = trajectory_detail[
                            "successful_retry_batch"
                        ]
                        detail["above_fraction"] = trajectory_detail[
                            "above_fraction"
                        ]
                        detail["mean_z"] = trajectory_detail["mean_z"]
                        detail["executed_steps"] = trajectory_detail[
                            "executed_steps"
                        ]
                        detail["candidate_window_count"] = trajectory_detail[
                            "candidate_window_count"
                        ]
                        detail["full_h_valid_window_count"] = trajectory_detail[
                            "full_h_valid_window_count"
                        ]
                        detail["committed_window_count"] = trajectory_detail[
                            "committed_window_count"
                        ]
                        detail["committed_window_ids"] = list(
                            trajectory_detail["committed_window_ids"]
                        )
                detail["committed_trajectory_count"] = len(
                    detail["committed_trajectory_ids"]
                )
                detail["all_committed_window_count"] = len(
                    detail["all_committed_window_ids"]
                )
                successful_executed_commit_by_gamma[f"{gamma:.9g}"] = detail
            if (
                config.gp_reference_mode in {
                    "cumulative_success_per_gamma_frozen_phi_exact",
                    "sliding_success_per_gamma_frozen_phi",
                    "sliding_success_per_gamma_current_phi",
                }
            ):
                for gamma in map(float, config.gammas):
                    new_count = sum(
                        row.gamma == gamma for row in committed_records
                    )
                    if new_count < 1:
                        raise RuntimeError(
                            f"round {round_i} produced no commit-capable "
                            f"SUCCESS window for gamma={gamma:g}; round not committed"
                        )
                    if (
                        config.gp_reference_mode
                        == "cumulative_success_per_gamma_frozen_phi_exact"
                    ):
                        total_count = sum(
                            row.gamma == gamma for row in gp_evidence
                        ) + new_count
                        if total_count > config.gp_exact_max_rows_per_gamma:
                            raise RuntimeError(
                                f"round {round_i} exact GP support for gamma={gamma:g} "
                                f"would exceed gp_exact_max_rows_per_gamma="
                                f"{config.gp_exact_max_rows_per_gamma}; round not committed"
                            )
            archive.extend(committed_records)
            if (
                config.gp_reference_mode == "round1_fixed_frozen_phi"
                and round_i == 1
            ):
                round1_gp_candidates.extend(committed_records)
            if config.gp_reference_mode in {
                "cumulative_accepted_frozen_phi",
                "cumulative_success_per_gamma_frozen_phi_exact",
                "sliding_success_per_gamma_frozen_phi",
                "sliding_success_per_gamma_current_phi",
            }:
                gp_evidence.extend(committed_records)
        replay_archive = (
            archive
            if config.replay_scope == "cumulative"
            else _recent(
                archive, round_i, config.replay_rounds, positive=True
            )
        )
        eligible_positive = [
            row for row in replay_archive
            if row.verification.valid and row.replay_eligible
        ]
        eligible_positive_total = len(eligible_positive)
        if config.replay_selector == "sigma_top":
            eligible_positive = _top_uncertainty_by_round(
                eligible_positive, config.replay_top_fraction,
                group_by_gamma=(
                    config.archive_rule == "successful_executed_windows"
                ),
            )
        elif config.replay_selector == "context_kcenter":
            eligible_positive = _context_kcenter_replay(
                eligible_positive, policy, config.replay_context_quota,
                config.replay_action_weight,
            )
        elif config.replay_selector == "cluster_balanced":
            replay_budget = (
                config.inner_steps * config.batch_size
                if config.inner_steps is not None else len(eligible_positive)
            )
            eligible_positive = _cluster_balanced_replay(
                eligible_positive, policy, config.replay_cluster_count,
                config.replay_action_weight, replay_budget, torch_rng,
            )
        negative = (
            [row for row in archive if not row.verification.valid]
            if config.replay_scope == "cumulative"
            else _recent(
                archive, round_i, config.replay_rounds, positive=False,
            )
        )
        update_started = time.perf_counter()
        round_optimizer_steps = _optimizer_steps_for_round(config, round_i)
        round_lr_scale, learning_rate_at_round_start = (
            _set_round_learning_rate(optimizer, config, round_i)
        )
        update_config = (
            config
            if config.optimizer_steps_total is None
            else replace(
                config,
                inner_steps=round_optimizer_steps,
                optimizer_steps_total=None,
            )
        )
        update = _update(policy, task, optimizer, eligible_positive, negative,
                         update_config, torch_rng)
        update_s = time.perf_counter() - update_started
        if (config.adaptive_beta and score_pools
                and any(float(pool.max() - pool.min()) > 1.0e-4
                        for pool in score_pools)):
            beta = calibrate_fixed_beta(score_pools, config.ess_target)
        row = {
            "round": round_i,
            "queries": sum(record.round == round_i for record in archive),
            "positives": sum(record.round == round_i and record.verification.valid
                             for record in archive),
            "replay_accepted_positives": sum(
                record.round == round_i and record.replay_eligible
                for record in archive
            ),
            "verifier_queries": verifier_queries,
            "verifier_positives": verifier_positives,
            "gp_buffer": len(gp_rows),
            "gp_buffer_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in gp_rows
                )
                for gamma in config.gammas
            },
            "gp_sliding_row_selector": (
                config.gp_sliding_row_selector
                if config.gp_reference_mode in {
                    "sliding_success_per_gamma_frozen_phi",
                    "sliding_success_per_gamma_current_phi",
                }
                else None
            ),
            "gp_window_start_by_gamma": _gp_window_start_diagnostics(
                gp_rows, config.gammas,
            ),
            "gp_reference_hash_by_gamma": {
                f"{float(gamma):.9g}": (
                    _record_identity_hash([
                        record for record in gp_rows
                        if record.gamma == float(gamma)
                    ])
                    if any(
                        record.gamma == float(gamma) for record in gp_rows
                    )
                    else None
                )
                for gamma in config.gammas
            },
            "gp_feature_hash_by_gamma": gp_feature_hash_by_gamma,
            "acquisition_representation_hash": (
                acquisition_representation_hash
            ),
            "gp_reference_hash": frozen_gp_hash,
            "gp_anchor_count": gp_anchor_count,
            "gp_adaptive_count": gp_adaptive_count,
            "gp_evidence_count": len(gp_evidence),
            "successful_executed_commit_by_gamma": (
                successful_executed_commit_by_gamma
            ),
            "committed_trajectory_ids": committed_trajectory_ids,
            "committed_window_ids": committed_window_ids,
            "commit_reverify_queries": commit_reverify_queries,
            "retry_capability_reverify_queries": (
                retry_capability_reverify_queries
            ),
            "commit_reverify_valid": commit_reverify_valid,
            "commit_reverify_progress": commit_reverify_progress,
            "replay_positive_total": eligible_positive_total,
            "replay_positives": len(eligible_positive),
            "replay_selector": config.replay_selector,
            "replay_scope": config.replay_scope,
            "round_learning_rate_scale": round_lr_scale,
            "learning_rate_at_round_start": learning_rate_at_round_start,
            "beta": beta_used,
            "beta_next": beta,
            "ESS_over_K": float(np.mean(ess_values)) if ess_values else 1.0,
            "marginal_ESS_over_K": (
                float(np.mean(marginal_ess_values))
                if marginal_ess_values else 1.0
            ),
            "sigma_pool_mean": (
                float(np.mean(sigma_pool_means)) if sigma_pool_means else None
            ),
            "sigma_selected_mean": (
                float(np.mean(sigma_selected_means))
                if sigma_selected_means else None
            ),
            "uncertainty_uplift": (
                float(np.mean(sigma_selected_means) - np.mean(sigma_pool_means))
                if sigma_pool_means else None
            ),
            "execution_ESS": (
                float(np.mean(execution_ess_values))
                if execution_ess_values else None
            ),
            "NVP": nvp,
            "success": statuses.count("SUCCESS"),
            "timeout": statuses.count("TIMEOUT"),
            "retry_batches_by_gamma": {
                f"{gamma:.9g}": count
                for gamma, count in retry_batches_by_gamma.items()
            },
            "attempted_episode_count": len(episodes),
            "paired_scene_replacements": list(paired_scene_replacements),
            "paired_scene_replacement_count": len(
                paired_scene_replacements
            ),
            "round_success_commit_complete": (
                config.archive_rule == "successful_executed_windows"
            ),
            "gp_refit_s": gp_refit_s,
            "gather_s": gather_s,
            "flow_sampling_s": flow_sampling_s,
            "acquisition_s": acquisition_s,
            "retry_verify_all_fast_path_contexts": (
                retry_verify_all_fast_path_contexts
            ),
            "verifier_dispatch_s": verifier_dispatch_s,
            "verifier_jobs": verifier_jobs,
            "verifier_candidates": verifier_candidates,
            "update_s": update_s,
            "round_total_s": time.perf_counter() - round_started,
            **update,
        }
        round_rows.append(row)
        if round_callback is not None:
            round_callback(row)
        _atomic_torch_save(
            {"round": round_i, "model": policy.state_dict(),
             "config": asdict(config), "pretrained": False},
            output_dir / f"checkpoint_{round_i:03d}.pt",
        )
        # Keep each committed round's replay rows independently recoverable.
        # A later quota hard-cap aborts before the final cumulative archive is
        # written, but must not discard samples from completed overnight rounds.
        round_archive_path = output_dir / f"query_archive_round_{round_i:03d}.pt"
        round_archive_tmp = round_archive_path.with_suffix(".pt.tmp")
        torch.save(
            [record for record in archive if record.round == round_i],
            round_archive_tmp,
        )
        round_archive_tmp.replace(round_archive_path)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        resume_payload = {
            "version": _RESUME_STATE_VERSION,
            "status": "COMMITTED_ROUND_RESUME",
            "completed_round": round_i,
            "config": asdict(config),
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_metadata": {
                name: getattr(optimizer, name)
                for name in (
                    "_safe_mppi_schedule_step", "_safe_mppi_base_lrs",
                    "_safe_mppi_round_lr_scale",
                )
                if hasattr(optimizer, name)
            },
            "trainable_parameter_names": trainable_parameter_names,
            "trainable_parameter_count": trainable_parameter_count,
            "beta": beta,
            "round_rows": round_rows,
            "archive": archive,
            "frozen_gp_rows": frozen_gp_rows,
            "frozen_gp_hash": frozen_gp_hash,
            "round1_gp_candidates": round1_gp_candidates,
            "gp_evidence": gp_evidence,
            "cumulative_anchors": cumulative_anchors,
            "cumulative_adaptive": cumulative_adaptive,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch_rng.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_device_rng_state": (
                torch.cuda.get_rng_state(device)
                if device.type == "cuda" else None
            ),
        }
        _atomic_torch_save(
            resume_payload, output_dir / _RESUME_STATE_NAME,
        )
        resume_metadata = {
            "status": "COMMITTED_ROUND_RESUME",
            "version": _RESUME_STATE_VERSION,
            "completed_round": round_i,
            "next_round": round_i + 1,
            "optimizer_step": int(
                getattr(optimizer, "_safe_mppi_schedule_step", 0)
            ),
            "flow_base_std_next": float(config.flow_base_std),
            "resume_state": _RESUME_STATE_NAME,
        }
        metadata_path = output_dir / _RESUME_METADATA_NAME
        metadata_tmp = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.tmp"
        )
        metadata_tmp.write_text(
            json.dumps(resume_metadata, indent=2, allow_nan=False) + "\n"
        )
        metadata_tmp.replace(metadata_path)
    manifest = {
        "status": "SAFE_FLOW_EXPANSION_COMPLETE",
        "config": asdict(config),
        "rbf_lengthscale": lengthscale,
        "final_beta": beta,
        "D": len(archive),
        "D_plus": sum(row.verification.valid for row in archive),
        "D_replay_accepted": sum(row.replay_eligible for row in archive),
        "rounds": round_rows,
        "resume": {
            "resumed_from_round": resumed_from_round,
            "latest_committed_round": int(round_rows[-1]["round"]),
            "state_file": _RESUME_STATE_NAME,
            "boundary": "after_optimizer_update_and_committed_archive",
        },
        "optimizer_scope": {
            "mode": (
                "head_only" if config.head_only_update
                else (
                    "last_n_trunk_layers"
                    if trainable_trunk_layers is not None
                    else "configured_trainable"
                )
            ),
            "trainable_trunk_layers": (
                None if trainable_trunk_layers is None
                else int(trainable_trunk_layers)
            ),
            "trainable_parameter_names": trainable_parameter_names,
            "trainable_parameter_count": trainable_parameter_count,
        },
        "semantics": {
            "GP": (
                "separate per-gamma RBF posteriors over recent full-H verifier-"
                "positive task angular descriptors; no cross-gamma uncertainty"
                if config.acquisition_feature == "task_angular"
                else {
                    "recent_current_phi": (
                        "recent full-H verifier positives embedded by current phi"
                    ),
                    "round1_fixed_frozen_phi": (
                        "fixed round-1 accepted selected-B evidence embedded by "
                        "frozen pretrained phi"
                    ),
                    "cumulative_accepted_frozen_phi": (
                        "bounded cumulative accepted selected-B evidence coreset "
                        "embedded by frozen pretrained phi"
                    ),
                    "cumulative_success_per_gamma_frozen_phi_exact": (
                        "separate exact per-gamma posteriors over every previously "
                        "committed successful executed window, embedded by frozen "
                        "pretrained endpoint phi; no thinning"
                    ),
                    "sliding_success_per_gamma_frozen_phi": (
                        "separate per-gamma bounded posteriors over committed "
                        "successful executed windows, embedded by frozen "
                        "pretrained endpoint phi"
                    ),
                    "sliding_success_per_gamma_current_phi": (
                        "separate per-gamma bounded posteriors over committed "
                        "successful executed windows, re-embedded at each round "
                        "by the current model endpoint phi"
                    ),
                }[config.gp_reference_mode]
            ),
            "acquisition_feature": config.acquisition_feature,
            "replay_augmentation": config.replay_augmentation,
            "D": ({
                "all_queries": "all successful selected-B verifier queries",
                "executed_only": "only the verifier-positive candidate selected for execution",
                "executed_plus_nvp_negative": (
                    "each executed positive plus at most one lowest-native-cost "
                    "full-verifier reject at terminal NVP"
                ),
                "successful_executed_windows": (
                    "all executed suffix starts reconstructed from "
                    "the scheduled number of distinct commit-capable successful "
                    "trajectories per gamma and round, "
                    "selected "
                    f"by {config.successful_trajectory_selector}; each available "
                    "suffix of length <=H is reverified and must be valid with "
                    "positive progress, then zero-padded only for fixed-shape "
                    "embedding with CFM loss masked to its valid horizon"
                ),
            }[config.archive_rule]),
            "D_plus": (
                "reverified valid positive-progress executed <=H suffixes from "
                "commit-capable terminal successes"
                if config.archive_rule == "successful_executed_windows"
                else "full-H verifier positives"
            ),
            "replay_acceptance": (
                "successful-window archive admission overrides per-query replay "
                "acceptance: only reverified valid positive-progress executed "
                "<=H suffixes from committed terminal successes are replayed"
                if config.archive_rule == "successful_executed_windows"
                else (
                    "every full-H verifier-positive selected-B query; independent "
                    "of progress and task target gates"
                    if config.replay_acceptance == "safety_valid"
                    else (
                        "legacy controller-coupled acceptance: validity alone when "
                        "no task target gate is configured; otherwise validity, "
                        "progress, and the active target gate"
                    )
                )
            ),
            "execution": ("positive-per-knot-x verifier positives ranked by "
                          f"{config.execution_rule}; nominal one-step H_P is diagnostic only"),
            "failure": "NVP terminates that episode; no expert fallback",
            "replay": (
                (
                    "cumulative D+"
                    if config.replay_scope == "cumulative"
                    else f"most recent {config.replay_rounds} rounds of D+"
                )
                + " selected by "
                f"{config.replay_selector}"
                + (
                    f" (top fraction={config.replay_top_fraction})"
                    if config.replay_selector == "sigma_top" else ""
                )
                + (
                    " (per-context quota="
                    f"{config.replay_context_quota}, action weight="
                    f"{config.replay_action_weight})"
                    if config.replay_selector == "context_kcenter" else ""
                )
                + (
                    " (clusters/gamma="
                    f"{config.replay_cluster_count}, action weight="
                    f"{config.replay_action_weight})"
                    if config.replay_selector == "cluster_balanced" else ""
                )
                + (
                    "; then equal mode x gamma mass, uniform trajectory "
                    "lineage, and uniform window"
                    if config.replay_batch_sampler
                    == "mode_gamma_stratified"
                    else "; then gamma-lineage-context-query equal mass"
                )
                + (
                    " multiplied by within-gamma periodic circular Voronoi "
                    "arc mass without removing replay rows"
                    if config.coverage_replay == "circular_voronoi" else ""
                )
            ),
            "negative_replay": ({
                "all_queries": "recent selected-B full-verifier rejects",
                "executed_only": "none: temporary non-executed queries never enter D",
                "executed_plus_nvp_negative": (
                    "at most one lowest-native-cost full-verifier reject per NVP rollout"
                ),
                "successful_executed_windows": (
                    "none: NVP, collision, OOB, and timeout trajectories are "
                    "discarded in full"
                ),
            }[config.archive_rule]),
            "candidate_perturbation": ("one Gaussian action vector per candidate applied "
                                       f"with scope={config.candidate_perturb_scope} before "
                                       "embedding and verification"),
            "flow_base_sampling": (
                "expansion-only initial flow latent "
                rf"x0~N(0,{config.flow_base_std ** 2:g}I), "
                "applied before flow integration"
                + (
                    "; sample-update pairs guided and unguided cohorts at the "
                    "same round-scheduled std in every batch"
                    if config.sample_update_mode is not None else ""
                )
                + "; raw evaluation remains unit normal"
            ),
            "representation": (
                "paired noised phi_s using each sampled plan's stored initial CFM base"
                if config.paired_noised_representation
                else "endpoint-only phi_s without a paired CFM base"
            ),
            "target_gate": (
                "inactive"
                if config.target_gate_start_round is None
                else (
                    (
                        "task target_eligible gates execution from round "
                        f"{config.target_gate_start_round}; safety_valid replay "
                        "remains independent; safety label unchanged"
                    )
                    if config.replay_acceptance == "safety_valid"
                    else (
                        "task target_eligible gates execution and replay acceptance "
                        f"from round {config.target_gate_start_round}; "
                        "safety label unchanged"
                    )
                )
            ),
            "gp_reference": (
                "rolling recent positives embedded by current phi"
                if config.gp_reference_mode == "recent_current_phi"
                else (
                    "one deterministic gamma-balanced cap of round-1 accepted "
                    "positives, reused with frozen pretrained phi"
                    if config.gp_reference_mode == "round1_fixed_frozen_phi"
                    else (
                        (
                            "exact per-gamma cumulative successful-window support "
                            "through the previous round, with frozen pretrained "
                            "endpoint phi and no thinning"
                        )
                        if config.gp_reference_mode
                        == "cumulative_success_per_gamma_frozen_phi_exact"
                        else (
                            (
                                "per-gamma bounded cumulative successful-window "
                                "evidence through the previous round, re-embedded "
                                "by current endpoint phi"
                                if config.gp_reference_mode
                                == "sliding_success_per_gamma_current_phi"
                                else (
                                    "per-gamma bounded cumulative successful-window "
                                    "evidence through the previous round, with "
                                    "frozen pretrained endpoint phi"
                                    if config.gp_reference_mode
                                    == "sliding_success_per_gamma_frozen_phi"
                                    else (
                                        "bounded cumulative frozen-phi coreset "
                                        "with equal gamma capacity, immutable "
                                        "round-1 anchors, and deterministic "
                                        "farthest-first adaptive discoveries"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "acquisition_diagnostics": (
                "ESS_over_K is the mean sequential conditional normalized ESS "
                "over B without-replacement draws; marginal_ESS_over_K is the "
                "first-draw Gibbs ESS/K computed from marginal sigma_K; uplift "
                "compares selected marginal sigma with the all-K marginal mean"
            ),
            "successful_executed_windows": (
                "inactive"
                if config.archive_rule != "successful_executed_windows"
                else (
                    "after gathering, choose "
                    "the scheduled number of distinct terminal "
                    "SUCCESS trajectories per gamma, ranked by "
                    "successful_trajectory_selector"
                    + (
                        f", filtered to exact mode quota "
                        "for that round/gamma cell"
                        if config.sample_update_mode is not None else ""
                    )
                    + "; reconstruct all L suffix starts "
                    "from consecutive actually executed first actions; reverify "
                    "only the available <=H states and commit valid "
                    "positive-progress suffixes; zero-pad to H for frozen-phi "
                    "embedding and mask CFM loss to the available actions; "
                    "recompute their sigma "
                    "against the round-start GP; selection is admission ranking "
                    "only and never changes validity or NVP; retry whole batches "
                    "until every gamma has the requested number of commit-capable "
                    "terminal SUCCESS trajectories, with "
                    "top-fraction replay grouped by (round,gamma) and complete "
                    "sigma-boundary ties retained"
                )
            ),
            "episode_retry": (
                "inactive"
                if config.archive_rule != "successful_executed_windows"
                else (
                    f"up to {config.max_retry_batches} whole-episode batches per "
                    f"gamma, {config.parallel_episodes} episodes per cohort"
                    + (
                        " in paired guided/unguided cohorts"
                        if config.sample_update_mode is not None else ""
                    )
                    + "; stop "
                    "a gamma after accumulating "
                    "its scheduled commit-capable "
                    "terminal SUCCESS trajectories"
                    + (
                        f" satisfying exact mode quota "
                        "for that round/gamma cell; later successes in "
                        "already-full modes are not committed"
                        if config.sample_update_mode is not None else ""
                    )
                    + "; no step retry; exhaustion "
                    "aborts the round "
                    "before archive, GP, optimizer, or checkpoint mutation; "
                    "counter-keyed gather RNG coordinates are "
                    "(seed,round,retry_batch,replica,step) and deliberately "
                    "exclude gamma"
                    + (
                        " and cohort so guided/unguided replicas use paired bases"
                        if config.sample_update_mode is not None else ""
                    )
                )
            ),
        },
    }
    if config.gp_reference_mode == "cumulative_accepted_frozen_phi":
        per_gamma_cap, anchor_cap, ingress_cap = _cumulative_gp_quotas(config)
        final_active = [
            row
            for gamma in sorted(cumulative_anchors)
            for row in cumulative_anchors[gamma] + cumulative_adaptive[gamma]
        ]
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "round": None,
            "through_round": max(0, config.rounds - 1),
            "count": len(final_active),
            "anchor_count": sum(map(len, cumulative_anchors.values())),
            "adaptive_count": sum(map(len, cumulative_adaptive.values())),
            "evidence_count": len(gp_evidence),
            "identity_sha256": frozen_gp_hash,
            "frozen_phi": True,
            "active_cap": config.gp_buffer_cap,
            "per_gamma_total_cap": per_gamma_cap,
            "per_gamma_anchor_cap": anchor_cap,
            "per_gamma_adaptive_cap": per_gamma_cap - anchor_cap,
            "per_gamma_ingress_cap": ingress_cap,
        }
    elif (
        config.gp_reference_mode
        == "cumulative_success_per_gamma_frozen_phi_exact"
    ):
        active_rows = list(gp_rows)
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "posterior_through_round": max(0, config.rounds - 1),
            "evidence_through_round": config.rounds,
            "count": len(active_rows),
            "count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in active_rows
                )
                for gamma in config.gammas
            },
            "evidence_count": len(gp_evidence),
            "evidence_count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in gp_evidence
                )
                for gamma in config.gammas
            },
            "identity_sha256": frozen_gp_hash,
            "identity_sha256_by_gamma": {
                f"{float(gamma):.9g}": (
                    _record_identity_hash([
                        row for row in active_rows
                        if row.gamma == float(gamma)
                    ])
                    if any(
                        row.gamma == float(gamma) for row in active_rows
                    )
                    else None
                )
                for gamma in config.gammas
            },
            "frozen_phi": True,
            "paired_noised_representation": (
                config.paired_noised_representation
            ),
            "exact": True,
            "thinning": "none",
            "hard_max_rows_per_gamma": config.gp_exact_max_rows_per_gamma,
            "legacy_gp_buffer_cap_used": False,
            "legacy_gp_buffer_cap_ignored_value": config.gp_buffer_cap,
            "window_source": (
                "every reverified available <=H suffix of actually executed "
                "controls from "
                + (
                    "up to "
                    if config.retry_exhaustion_policy == "commit_available"
                    else ""
                )
                + f"{config.successful_trajectories_per_gamma} distinct "
                "commit-capable terminal SUCCESS trajectories per gamma/round; "
                "zero-padded for fixed-shape phi with valid-horizon CFM masks"
            ),
        }
    elif config.gp_reference_mode in {
        "sliding_success_per_gamma_frozen_phi",
        "sliding_success_per_gamma_current_phi",
    }:
        active_rows = list(gp_rows)
        per_gamma_cap = config.gp_buffer_cap // len(config.gammas)
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "posterior_through_round": max(0, config.rounds - 1),
            "evidence_through_round": config.rounds,
            "count": len(active_rows),
            "count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in active_rows
                )
                for gamma in config.gammas
            },
            "evidence_count": len(gp_evidence),
            "evidence_count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in gp_evidence
                )
                for gamma in config.gammas
            },
            "identity_sha256": frozen_gp_hash,
            "identity_sha256_by_gamma": {
                f"{float(gamma):.9g}": (
                    _record_identity_hash([
                        row for row in active_rows
                        if row.gamma == float(gamma)
                    ])
                    if any(
                        row.gamma == float(gamma) for row in active_rows
                    )
                    else None
                )
                for gamma in config.gammas
            },
            "frozen_phi": (
                config.gp_reference_mode
                == "sliding_success_per_gamma_frozen_phi"
            ),
            "representation": (
                "pretrained_phi0"
                if config.gp_reference_mode
                == "sliding_success_per_gamma_frozen_phi"
                else "current_round_phi_reembedded_before_gather"
            ),
            "paired_noised_representation": (
                config.paired_noised_representation
            ),
            "paired_success_window_base": (
                "concatenated actual first-action generator latents from "
                "successive closed-loop decisions; terminal padding is zero"
                if config.paired_noised_representation else None
            ),
            "exact": False,
            "active_cap": config.gp_buffer_cap,
            "cap_scope": "equal_per_gamma",
            "per_gamma_cap": per_gamma_cap,
            "row_selector": config.gp_sliding_row_selector,
            "eviction": (
                (
                    "legacy highest-window-start FIFO tail"
                    if config.gp_sliding_row_selector == "fifo_tail"
                    else (
                        "equal successful-trajectory quotas with uniformly "
                        "spaced departure-to-tail window starts"
                    )
                )
                + "; cumulative evidence log retained"
            ),
            "active_window_start_by_gamma": _gp_window_start_diagnostics(
                active_rows, config.gammas,
            ),
            "window_source": (
                "every reverified available <=H suffix of actually executed "
                "controls from "
                + (
                    "up to "
                    if config.retry_exhaustion_policy == "commit_available"
                    else ""
                )
                + f"{config.successful_trajectories_per_gamma} distinct "
                "commit-capable terminal SUCCESS trajectories per gamma/round; "
                "zero-padded for fixed-shape phi with valid-horizon CFM masks"
            ),
        }
    else:
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "round": (
                1 if config.gp_reference_mode == "round1_fixed_frozen_phi" else None
            ),
            "count": len(frozen_gp_rows or []),
            (
                "source_window_count"
                if config.archive_rule == "successful_executed_windows"
                else "source_query_count"
            ): len(round1_gp_candidates),
            "identity_sha256": frozen_gp_hash,
            "frozen_phi": config.gp_reference_mode == "round1_fixed_frozen_phi",
        }
    torch.save(archive, output_dir / "query_archive.pt")
    if config.gp_reference_mode in {
        "cumulative_accepted_frozen_phi",
        "cumulative_success_per_gamma_frozen_phi_exact",
        "sliding_success_per_gamma_frozen_phi",
        "sliding_success_per_gamma_current_phi",
    }:
        torch.save(gp_evidence, output_dir / "gp_evidence.pt")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                          allow_nan=False) + "\n")
    (output_dir / "RESUME_IN_PROGRESS.json").unlink(missing_ok=True)
    return manifest


def run_safe_expansion(
    policy: ExpansionPolicy,
    task: ExpansionTask,
    output_dir: str | Path,
    *,
    config: ExpansionConfig = ExpansionConfig(),
    calibration_features: torch.Tensor | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    round_callback: Callable[[dict[str, Any]], None] | None = None,
    retry_callback: Callable[[dict[str, Any]], None] | None = None,
    pair_replacement_callback: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    trainable_trunk_layers: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run expansion with ordered CPU-parallel verification when requested.

    ``trainable_trunk_layers`` is the optional partial-depth adaptation scope:
    ``None`` keeps the configured optimizer scope, while N>=1 trains only the
    flow head plus the last N trunk blocks.  It is mutually exclusive with
    ``ExpansionConfig.head_only_update``.  ``resume_from`` continues from an
    atomic committed-round state in the same output directory; an interrupted
    in-progress round is deliberately sampled again.
    """
    config.validate()
    with _OrderedVerifier(task, config.verifier_workers) as verifier:
        return _run_safe_expansion_impl(
            policy,
            task,
            output_dir,
            config=config,
            calibration_features=calibration_features,
            event_callback=event_callback,
            round_callback=round_callback,
            retry_callback=retry_callback,
            pair_replacement_callback=pair_replacement_callback,
            trainable_trunk_layers=trainable_trunk_layers,
            resume_from=resume_from,
            verifier=verifier,
        )
