#!/usr/bin/env python3
"""Offline PRE2 post-training on successful multi-sphere axis-180 experts.

This file is deliberately isolated from the online expansion trainer.  It
consumes the standard lab demonstration archive contract:

* ``manifest.json`` declares only admitted successful trajectories in ``runs``;
* ``resolved_config.json`` is the exact randomized-sphere task configuration;
* every run NPZ stores pre-smoothing raw ``controls[T,3]``, ``states[T+1,6]``,
  and the per-run obstacle arrays used to rebuild the PRE2 H_P context;
* every run declares ``gamma``, a stable axis-180 ``pair_id``, and pair member
  original/source/0 or axis_180/mirror/1.

The existing lab window loader remains authoritative for state/control replay,
scene hashes, H=10 plans, and PRE2 context construction.  This trainer changes
only all three parameterised flow-trunk blocks plus the head (12,926 parameters),
never the visual encoder.  Batches are balanced over gamma x pair member, then
sample a trajectory and a window uniformly.  Flow-matching bases and times are
sampled independently; stored/paired flow bases are intentionally unsupported.

The five cumulative prefixes 40/80/120/160/200 become evaluator rounds 1..5.
One Adam optimizer and one cosine schedule persist across all five stages.
``checkpoint_000.pt`` is bitwise the PRE2 model, and checkpoints 001..005 use
the normal evaluator payload shape.  This is a backup post-trainer, not an
online collector and not a claim that training loss implies closed-loop safety.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import numpy as np
import torch

from safe_mppi.expansion import _trunk_suffix_parameters
from safe_mppi.lab_clutter_evaluation import (
    PATH_FOCUSED_SPHERE_SCENE_SCHEMAS,
    PATH_FOCUSED_SPHERE_TASK_PROFILE,
)
from safe_mppi.lab_clutter_expansion import sphere_scene_spec_from_config
from safe_mppi.lab_reference_flow_task import lab_reference_demo_windows
from safe_mppi.lab_visual_flow import (
    LAB_HP100_SCHEMA,
    load_lab_reference_policy,
)

if __package__:
    from scripts.research_overfit_committed_modes import _cfm_loss
else:  # ``python scripts/posttrain_multisphere_axis180_expert.py``
    from research_overfit_committed_modes import _cfm_loss


DEFAULT_PREFIXES = (40, 80, 120, 160, 200)
EXPECTED_GAMMA_COUNT = 4
EXPECTED_PAIR_MEMBER_COUNT = 2
EXPECTED_TRAINABLE_PARAMETER_COUNT = 12_926


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    file: str
    gamma: float
    pair_id: str
    pair_member: int
    collection_round: int
    pair_slot: int
    manifest_index: int
    row: dict

    @property
    def stratum(self) -> tuple[float, int]:
        return self.gamma, self.pair_member

    @property
    def pair_order(self) -> tuple[int, int, int, str]:
        return (
            self.collection_round,
            self.pair_slot,
            self.manifest_index,
            self.pair_id,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_present(row: dict, names: Iterable[str], default=None):
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return default


def _pair_member(row: dict) -> int:
    raw = _first_present(row, (
        "pair_member_index",
        "paired_scene_member",
        "pair_member",
        "paired_scene_member_name",
    ))
    if isinstance(raw, (int, np.integer)) and int(raw) in (0, 1):
        return int(raw)
    normalized = str(raw).strip().lower()
    names = {
        "0": 0,
        "original": 0,
        "source": 0,
        "1": 1,
        "axis_180": 1,
        "axis-180": 1,
        "rotated": 1,
        "mirror": 1,
        "mirrored": 1,
    }
    if normalized not in names:
        raise ValueError(
            "axis-180 collector run requires pair member 0/original/source "
            f"or 1/axis_180/mirror; got {raw!r}"
        )
    return names[normalized]


def _is_admitted(row: dict) -> bool:
    value = _first_present(row, (
        "accepted", "pair_admitted", "trajectory_accepted",
    ))
    return bool(value)


def trajectory_catalog(manifest: dict) -> list[Trajectory]:
    """Parse and validate the collector's admitted trajectory identities."""
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("collector manifest requires a nonempty runs list")
    catalog = []
    observed_files = set()
    observed_ids = set()
    for index, row in enumerate(runs):
        if not isinstance(row, dict) or not _is_admitted(row):
            raise ValueError(
                "collector manifest runs must contain admitted trajectories only"
            )
        filename = str(row.get("file", ""))
        if not filename or filename in observed_files:
            raise ValueError(f"missing or duplicate collector NPZ file {filename!r}")
        gamma = float(row["gamma"])
        if not math.isfinite(gamma):
            raise ValueError("collector gamma must be finite")
        pair_id = str(_first_present(row, (
            "pair_id", "paired_scene_id",
        ), ""))
        if not pair_id:
            raise ValueError(f"collector run {filename} lacks an axis-180 pair id")
        trajectory_id = str(row.get("trajectory_id", filename))
        if trajectory_id in observed_ids:
            raise ValueError(f"duplicate trajectory id {trajectory_id!r}")
        collection_round = int(_first_present(row, (
            "collection_round", "round", "expansion_round",
        ), 0))
        pair_slot = int(_first_present(row, (
            "pair_index", "paired_scene_pair_slot", "pair_slot",
        ), index))
        catalog.append(Trajectory(
            trajectory_id=trajectory_id,
            file=filename,
            gamma=gamma,
            pair_id=pair_id,
            pair_member=_pair_member(row),
            collection_round=collection_round,
            pair_slot=pair_slot,
            manifest_index=index,
            row=row,
        ))
        observed_files.add(filename)
        observed_ids.add(trajectory_id)
    return catalog


def cumulative_prefixes(
    catalog: Sequence[Trajectory],
    gammas: Sequence[float],
    prefixes: Sequence[int],
) -> dict[int, tuple[str, ...]]:
    """Choose complete pair-balanced cumulative prefixes deterministically."""
    gamma_values = tuple(sorted(map(float, gammas)))
    if len(gamma_values) != EXPECTED_GAMMA_COUNT:
        raise ValueError(
            f"PRE2 post-training requires four gammas, got {gamma_values}"
        )
    prefix_values = tuple(map(int, prefixes))
    if (
        not prefix_values
        or any(value <= 0 for value in prefix_values)
        or tuple(sorted(set(prefix_values))) != prefix_values
    ):
        raise ValueError("trajectory prefixes must be positive, unique, and increasing")
    stratum_count = len(gamma_values) * EXPECTED_PAIR_MEMBER_COUNT
    if any(value % stratum_count for value in prefix_values):
        raise ValueError(
            f"every trajectory prefix must be divisible by {stratum_count}"
        )

    by_pair: dict[tuple[float, str], dict[int, Trajectory]] = defaultdict(dict)
    for trajectory in catalog:
        matches = [
            gamma for gamma in gamma_values
            if np.isclose(trajectory.gamma, gamma, rtol=0.0, atol=1.0e-9)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"trajectory gamma {trajectory.gamma:g} is not uniquely configured"
            )
        key = (matches[0], trajectory.pair_id)
        if trajectory.pair_member in by_pair[key]:
            raise ValueError(
                f"pair {key} repeats member {trajectory.pair_member}"
            )
        by_pair[key][trajectory.pair_member] = trajectory
    incomplete = {
        key: sorted(members)
        for key, members in by_pair.items()
        if set(members) != {0, 1}
    }
    if incomplete:
        raise ValueError(f"collector contains incomplete axis-180 pairs: {incomplete}")

    pairs_by_gamma: dict[float, list[dict[int, Trajectory]]] = defaultdict(list)
    for (gamma, _), members in by_pair.items():
        pairs_by_gamma[gamma].append(members)
    for gamma in gamma_values:
        pairs_by_gamma[gamma].sort(
            key=lambda members: min(
                trajectory.pair_order for trajectory in members.values()
            )
        )

    output = {}
    previous: set[str] = set()
    for prefix in prefix_values:
        pairs_per_gamma = prefix // stratum_count
        selected = []
        for gamma in gamma_values:
            available = pairs_by_gamma[gamma]
            if len(available) < pairs_per_gamma:
                raise ValueError(
                    f"prefix {prefix} needs {pairs_per_gamma} pairs for "
                    f"gamma={gamma:g}, found {len(available)}"
                )
            for members in available[:pairs_per_gamma]:
                selected.extend(
                    members[member].trajectory_id for member in (0, 1)
                )
        selected_set = set(selected)
        if len(selected) != prefix or len(selected_set) != prefix:
            raise RuntimeError("balanced prefix construction lost a trajectory")
        if not previous.issubset(selected_set):
            raise RuntimeError("trajectory prefixes are not cumulative")
        counts = Counter(
            trajectory.stratum for trajectory in catalog
            if trajectory.trajectory_id in selected_set
        )
        if set(counts.values()) != {prefix // stratum_count}:
            raise RuntimeError(f"prefix {prefix} is not gamma/member balanced")
        output[prefix] = tuple(sorted(selected_set))
        previous = selected_set
    return output


def window_groups(
    metadata: Sequence[dict],
    catalog: Sequence[Trajectory],
) -> dict[str, np.ndarray]:
    """Bind reconstructed windows back to exactly one collector trajectory."""
    trajectory_by_file = {trajectory.file: trajectory for trajectory in catalog}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        filename = str(row["file"])
        trajectory = trajectory_by_file.get(filename)
        if trajectory is None:
            raise ValueError(f"window references undeclared run file {filename!r}")
        if not np.isclose(
            float(row["gamma"]), trajectory.gamma, rtol=0.0, atol=1.0e-9,
        ):
            raise ValueError(f"window gamma disagrees for {filename}")
        groups[trajectory.trajectory_id].append(index)
    missing = {
        trajectory.trajectory_id for trajectory in catalog
    } - set(groups)
    if missing:
        raise ValueError(
            "every admitted trajectory must yield at least one full H=10 window; "
            f"missing={sorted(missing)}"
        )
    return {
        trajectory_id: np.asarray(indices, np.int64)
        for trajectory_id, indices in groups.items()
    }


def balanced_batch_row_ids(
    trajectory_ids: Sequence[str],
    catalog_by_id: dict[str, Trajectory],
    groups: dict[str, np.ndarray],
    gammas: Sequence[float],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample gamma/member -> trajectory -> window with exact batch balance."""
    strata: dict[tuple[float, int], list[str]] = defaultdict(list)
    for trajectory_id in trajectory_ids:
        trajectory = catalog_by_id[trajectory_id]
        strata[trajectory.stratum].append(trajectory_id)
    expected = {
        (float(gamma), member)
        for gamma in gammas for member in range(EXPECTED_PAIR_MEMBER_COUNT)
    }
    if set(strata) != expected:
        raise ValueError(
            "training prefix does not contain all gamma x pair-member strata"
        )
    if int(batch_size) < len(expected) or int(batch_size) % len(expected):
        raise ValueError(
            f"batch size must be a positive multiple of {len(expected)}"
        )
    per_stratum = int(batch_size) // len(expected)
    rows = []
    diagnostics = {}
    for gamma, member in sorted(expected):
        lineages = sorted(strata[(gamma, member)])
        selected = rng.choice(lineages, size=per_stratum, replace=True)
        for trajectory_id in selected:
            choices = groups[str(trajectory_id)]
            rows.append(int(choices[int(rng.integers(len(choices)))]))
        diagnostics[f"gamma={gamma:.9g},member={member}"] = per_stratum
    rng.shuffle(rows)
    return np.asarray(rows, np.int64), diagnostics


@torch.no_grad()
def _encode_contexts(policy, contexts: np.ndarray, device, batch_size: int):
    values = []
    for start in range(0, len(contexts), int(batch_size)):
        external = torch.from_numpy(
            np.asarray(contexts[start:start + batch_size], np.float32)
        ).to(device)
        values.append(policy.encode_context(external).detach())
    return torch.cat(values)


def _cpu_state_dict(policy) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    output: Path,
    stage: int,
    optimizer_step: int,
    model: dict[str, torch.Tensor],
    *,
    prefix: int,
    args,
    pretrained: bool,
) -> None:
    _atomic_torch_save({
        "round": int(stage),
        "model": model,
        "config": {
            "diagnostic": "multisphere_axis180_expert_posttrain",
            "optimizer_step": int(optimizer_step),
            "trajectory_prefix": int(prefix),
            "trainable_trunk_layers": 3,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "flow_coupling": "independent",
            "learning_rate": float(args.learning_rate),
            "final_learning_rate": float(args.final_learning_rate),
            "gradient_clip_norm": float(args.gradient_clip_norm),
        },
        "pretrained": bool(pretrained),
    }, output / f"checkpoint_{stage:03d}.pt")


def _scene_ledger(
    collector: Path,
    catalog: Sequence[Trajectory],
    schema: str,
) -> list[dict]:
    ledger = []
    seen = set()
    for trajectory in catalog:
        row = trajectory.row
        scene_hash = str(row.get("scene_hash", ""))
        if not scene_hash:
            raise ValueError(
                f"collector run {trajectory.file} requires scene_hash"
            )
        if scene_hash in seen:
            continue
        spheres = row.get("spheres")
        if spheres is None:
            with np.load(collector / trajectory.file, allow_pickle=False) as data:
                if "spheres" not in data.files:
                    raise ValueError(
                        f"collector run {trajectory.file} lacks spheres"
                    )
                spheres = np.asarray(data["spheres"], np.float32).tolist()
        ledger.append({
            "schema": schema,
            "scene_hash": scene_hash,
            "spheres": spheres,
            "gamma": trajectory.gamma,
            "pair_id": trajectory.pair_id,
            "paired_scene_member": trajectory.pair_member,
            "source_file": trajectory.file,
        })
        seen.add(scene_hash)
    return ledger


def _validate_arguments(parser, args) -> tuple[int, ...]:
    prefixes = tuple(map(int, args.prefix_trajectories))
    if prefixes != DEFAULT_PREFIXES:
        parser.error(
            "this preregistered backup trainer requires prefixes "
            "40 80 120 160 200"
        )
    if args.steps_per_prefix < 1:
        parser.error("--steps-per-prefix must be positive")
    if args.batch_size < 8 or args.batch_size % 8:
        parser.error("--batch-size must be a positive multiple of 8")
    if args.context_batch_size < 1:
        parser.error("--context-batch-size must be positive")
    if not 0.0 < args.final_learning_rate <= args.learning_rate:
        parser.error(
            "require 0 < --final-learning-rate <= --learning-rate"
        )
    if args.gradient_clip_norm <= 0.0:
        parser.error("--gradient-clip-norm must be positive")
    return prefixes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prefix-trajectories", type=int, nargs="+",
        default=DEFAULT_PREFIXES,
    )
    parser.add_argument("--steps-per-prefix", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the full collector/context/prefix contract without writes",
    )
    args = parser.parse_args(argv)
    args.prefix_trajectories = _validate_arguments(parser, args)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    collector = args.collector.resolve()
    pretrain_dir = args.pretrain_dir.resolve()
    manifest_path = collector / "manifest.json"
    resolved_config = collector / "resolved_config.json"
    if not manifest_path.is_file() or not resolved_config.is_file():
        raise FileNotFoundError(
            "collector requires manifest.json and resolved_config.json"
        )
    collector_manifest = json.loads(manifest_path.read_text())
    catalog = trajectory_catalog(collector_manifest)
    if len(catalog) != args.prefix_trajectories[-1]:
        raise ValueError(
            "preregistered collector must contain exactly "
            f"{args.prefix_trajectories[-1]} admitted trajectories; "
            f"found {len(catalog)}"
        )

    pretrained_path = pretrain_dir / "pretrained.pt"
    pretrain_manifest_path = pretrain_dir / "pretrain_manifest.json"
    pretrained_payload = torch.load(
        pretrained_path, map_location="cpu", weights_only=False,
    )
    pretrain_manifest = json.loads(pretrain_manifest_path.read_text())
    policy = load_lab_reference_policy(pretrained_path)
    if (
        getattr(policy, "context_schema", None) != LAB_HP100_SCHEMA
        or pretrain_manifest.get("context_schema") != LAB_HP100_SCHEMA
    ):
        raise ValueError(
            "backup post-training requires the PRE2 uniform-H_P context schema "
            f"{LAB_HP100_SCHEMA!r}"
        )

    contexts, plans, metadata, config = lab_reference_demo_windows(
        collector,
        validate_archive=True,
        context_schema=policy.context_schema,
        max_windows_per_trajectory=None,
    )
    gammas = tuple(sorted(map(float, config.data.gammas)))
    prefixes = cumulative_prefixes(
        catalog, gammas, args.prefix_trajectories,
    )
    groups = window_groups(metadata, catalog)
    if plans.ndim != 3 or tuple(plans.shape[1:]) != (10, 3):
        raise ValueError(
            f"collector must reconstruct H10x3 plans, got {plans.shape}"
        )
    if contexts.ndim != 2 or contexts.shape[1] != policy.context_dim:
        raise ValueError(
            "reconstructed PRE2 external context dimension mismatch: "
            f"{contexts.shape} vs {policy.context_dim}"
        )

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    trainable = _trunk_suffix_parameters(policy.flow, 3)
    trainable_names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise ValueError(
            "PRE2 trunk3+head parameter contract changed: "
            f"{trainable_count} != {EXPECTED_TRAINABLE_PARAMETER_COUNT}"
        )
    if any(not name.startswith("flow.") for name in trainable_names):
        raise RuntimeError("visual/context encoder unexpectedly became trainable")

    schema = sphere_scene_spec_from_config(config).scene_schema
    if schema not in PATH_FOCUSED_SPHERE_SCENE_SCHEMAS:
        raise ValueError(
            f"expected a path-focused sphere collector, got schema {schema!r}"
        )
    prefix_summary = {
        str(prefix): {
            "trajectories": len(ids),
            "windows": int(sum(len(groups[value]) for value in ids)),
            "trajectories_by_gamma_member": dict(sorted(Counter(
                f"gamma={next(t.gamma for t in catalog if t.trajectory_id == value):.9g},"
                f"member={next(t.pair_member for t in catalog if t.trajectory_id == value)}"
                for value in ids
            ).items())),
        }
        for prefix, ids in prefixes.items()
    }
    dry_summary = {
        "status": "DRY_RUN_VALID" if args.dry_run else "READY_TO_TRAIN",
        "collector_trajectories": len(catalog),
        "collector_windows": len(plans),
        "context_schema": policy.context_schema,
        "context_dim": int(policy.context_dim),
        "plan_shape": list(plans.shape[1:]),
        "gammas": list(gammas),
        "prefixes": prefix_summary,
        "trainable_parameter_count": trainable_count,
        "trainable_parameter_names": trainable_names,
        "visual_encoder_frozen": True,
        "flow_coupling": "independent",
    }
    if args.dry_run:
        print(json.dumps(dry_summary, indent=2), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    device = torch.device(args.device)
    policy = policy.to(device).eval()
    encoded = _encode_contexts(
        policy, contexts, device, args.context_batch_size,
    )
    candidates = torch.from_numpy(np.asarray(plans, np.float32)).to(device)
    loss_masks = torch.ones_like(candidates)
    policy.flow.train()
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    total_steps = args.steps_per_prefix * len(prefixes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.final_learning_rate,
    )
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output = args.output.resolve()
    output.mkdir(parents=True)
    _save_checkpoint(
        output, 0, 0, pretrained_payload["model"],
        prefix=0, args=args, pretrained=True,
    )
    catalog_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in catalog
    }
    losses = []
    round_rows = []
    optimizer_step = 0
    for stage, (prefix, trajectory_ids) in enumerate(prefixes.items(), start=1):
        stage_losses = []
        batch_balance = None
        for _ in range(args.steps_per_prefix):
            chosen, batch_balance = balanced_batch_row_ids(
                trajectory_ids,
                catalog_by_id,
                groups,
                gammas,
                args.batch_size,
                rng,
            )
            ids = torch.as_tensor(chosen, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            # Reuse the proven committed-mode CFM implementation, but force
            # fresh independent bases.  Collector NPZ flow bases are ignored.
            loss = _cfm_loss(
                policy.flow,
                encoded[ids],
                candidates[ids],
                loss_masks[ids],
                "independent",
                paired_bases=None,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, args.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            value = float(loss.detach().cpu())
            losses.append(value)
            stage_losses.append(value)
        _save_checkpoint(
            output,
            stage,
            optimizer_step,
            _cpu_state_dict(policy),
            prefix=prefix,
            args=args,
            pretrained=False,
        )
        round_row = {
            "round": stage,
            "trajectory_prefix": prefix,
            "replay_windows": prefix_summary[str(prefix)]["windows"],
            "optimizer_steps_this_stage": args.steps_per_prefix,
            "optimizer_step": optimizer_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "loss_first_100": float(np.mean(stage_losses[:100])),
            "loss_last_100": float(np.mean(stage_losses[-100:])),
            "batch_rows_by_gamma_member": batch_balance,
            "gradient_norm_last": float(gradient_norm.detach().cpu()),
        }
        round_rows.append(round_row)
        print(
            f"[posttrain] stage={stage}/5 prefix={prefix} "
            f"step={optimizer_step}/{total_steps} "
            f"loss100={round_row['loss_last_100']:.6f} "
            f"lr={round_row['learning_rate']:.3g}",
            flush=True,
        )

    shutil.copy2(resolved_config, output / "task_config_resolved.json")
    ledger = _scene_ledger(collector, catalog, schema)
    manifest = {
        "status": "COMPLETE",
        "kind": "multisphere axis-180 expert cumulative-prefix post-training",
        "config": {
            "rounds": len(prefixes),
            "gammas": list(gammas),
            "K": 0,
            "B": 0,
        },
        "rounds": round_rows,
        "checkpoint_stage_map": [
            {"round": 0, "trajectory_prefix": 0, "optimizer_step": 0},
            *round_rows,
        ],
        "D": int(len(plans)),
        "D_plus": int(len(plans)),
        "task_profile": PATH_FOCUSED_SPHERE_TASK_PROFILE,
        "lab_conditioning": {
            "context_schema": policy.context_schema,
            "policy_context_dim": int(policy.context_dim),
            "visual_encoder": {
                "present": True,
                "frozen_during_expansion": True,
                "explicit_freeze_flag": True,
            },
        },
        "lab_scene_schema": schema,
        "lab_scene_ledger": ledger,
        "lab_scene_randomization": config.raw.get("scene_randomization"),
        "lab_scene_rng_contract": {
            "source": "offline collector manifest/NPZ",
            "evaluation_scene_reuse": False,
        },
        "lab_paired_scene_quota": {
            "enabled": True,
            "rotation": "start_goal_axis_180",
            "pair_admission": "both successful collector members required",
        },
        "event_log": "none",
        "optimizer_scope": {
            "mode": "all_three_flow_trunk_blocks_plus_head",
            "trainable_trunk_layers": 3,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "visual_encoder_frozen": True,
        },
        "optimizer": {
            "name": "Adam",
            "persistent_across_prefixes": True,
            "learning_rate": args.learning_rate,
            "final_learning_rate": args.final_learning_rate,
            "cosine_total_steps": total_steps,
            "steps_per_prefix": args.steps_per_prefix,
            "batch_size": args.batch_size,
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "sampling": {
            "hierarchy": "gamma x pair_member -> trajectory -> window",
            "exact_batch_stratum_balance": True,
            "flow_coupling": "independent",
            "stored_paired_bases_used": False,
            "prefixes": prefix_summary,
        },
        "collector": {
            "path": str(collector),
            "manifest_sha256": _sha256(manifest_path),
            "resolved_config_sha256": _sha256(resolved_config),
            "trajectory_count": len(catalog),
            "window_count": len(plans),
        },
        "pretrained": {
            "path": str(pretrained_path),
            "sha256": _sha256(pretrained_path),
            "round_zero_bitwise_source": True,
        },
        "source_assumptions": [
            "manifest.runs contains admitted successful trajectories only",
            "NPZ controls are pre-smoothing raw acceleration commands",
            "states and controls replay exactly under resolved_config dynamics",
            "per-run obstacle arrays reconstruct the PRE2 H_P context",
            "every pair_id has one original and one axis_180 member",
            "full H10 windows only; no terminal zero-padding is synthesized",
            "collector safety labels are trusted and are not reverified here",
        ],
        "seed": args.seed,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "training_loss.json").write_text(
        json.dumps({"cfm_loss": losses}, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "output": str(output),
        "rounds": len(round_rows),
        "optimizer_steps": optimizer_step,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
