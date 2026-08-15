"""Repair PRE2 from the exact round-1 multi-sphere committed trajectories.

This is deliberately separate from online expansion.  It tunes the stopping
step on complete held-out original/axis-180 scene pairs, then restarts from
PRE2 and trains on all 40 trajectory lineages for exactly that many steps.
The output is an evaluator-compatible one-round package and never mutates the
source expansion or pretrained checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.lab_visual_flow import load_lab_reference_policy
from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import sphere_scene_spec_from_config
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)


def _load_rows(expansion: Path):
    archive_path = expansion / "query_archive_round_001.pt"
    if not archive_path.is_file():
        archive_path = expansion / "query_archive.pt"
    rows = torch.load(archive_path, map_location="cpu", weights_only=False)
    rows = [
        row for row in rows
        if row.replay_eligible and row.verification.valid
    ]
    if not rows:
        raise ValueError("round-1 archive has no valid replay-eligible rows")
    return rows, archive_path


def _lineages(rows):
    groups: dict[str, list[int]] = defaultdict(list)
    metadata = {}
    for index, row in enumerate(rows):
        trajectory_id = str(row.trajectory_id)
        groups[trajectory_id].append(index)
        value = (float(row.gamma), int(row.sample_update_mode))
        previous = metadata.setdefault(trajectory_id, value)
        if previous != value:
            raise ValueError(f"trajectory {trajectory_id} changes gamma/label")
    by_cell = defaultdict(list)
    for trajectory_id, (gamma, label) in metadata.items():
        by_cell[(gamma, label)].append(trajectory_id)
    if len(groups) != 40 or set(map(len, by_cell.values())) != {1}:
        raise ValueError(
            "repair requires exactly one lineage in each of 4 gamma x 10 "
            f"paired-label cells; got {len(groups)} trajectories and "
            f"{len(by_cell)} cells"
        )
    gammas = sorted({value[0] for value in metadata.values()})
    labels = sorted({value[1] for value in metadata.values()})
    if len(gammas) != 4 or labels != list(range(10)):
        raise ValueError(f"unexpected gamma/label support: {gammas}, {labels}")
    return dict(groups), metadata, gammas


def _load_success_event_executed_rows(
    expansion: Path, policy, device,
):
    """Rebuild masked executed-action windows from every saved success."""
    event_path = expansion / "events_round_001.pt"
    events = torch.load(event_path, map_location="cpu", weights_only=False)
    config = load_config(expansion / "task_config_resolved.json")
    scene_spec = sphere_scene_spec_from_config(config)
    task = LabClutterPre2MultiPairExpansionTask(
        config,
        context_schema=policy.context_schema,
        device=device,
        tight_corridor=False,
        verifier_mode="full_polytope",
        verifier_solver="analytic",
        execution_taskspace_quadratic_weight=250.0,
        execution_taskspace_quadratic_target_m=0.15,
        execution_axis_cylinder_quadratic_weight=5.0,
        execution_axis_cylinder_radius_m=1.1,
        execution_control_weight=0.05,
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=81411,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        fixed_scene_layout="none",
        scene_spec=scene_spec,
    )
    rows = []
    groups: dict[str, list[int]] = defaultdict(list)
    pair_slot_by_trajectory = {}
    stratum_by_trajectory = {}
    gammas = set()
    event_groups = defaultdict(list)
    for event in events:
        trajectory_id = (
            f"g{float(event['gamma']):g}:e{int(event['episode'])}:"
            f"b{int(event['retry_batch'])}:r{int(event['replica'])}:"
            f"{event['scene_hash']}"
        )
        event_groups[trajectory_id].append(event)

    for trajectory_id, trajectory_events in sorted(event_groups.items()):
        trajectory_events.sort(key=lambda event: int(event["step"]))
        if [int(event["step"]) for event in trajectory_events] != list(
            range(len(trajectory_events))
        ):
            raise ValueError("success-event trajectory steps are not contiguous")
        if trajectory_events[-1]["status"] != "SUCCESS":
            raise ValueError("event archive contains a non-success trajectory")
        executed_actions = []
        for event in trajectory_events:
            chosen_local = event.get("chosen_local")
            if chosen_local is None:
                raise ValueError("successful trajectory has an unexecuted event")
            selected = list(map(int, event["selected"]))
            verification = event["verification"][int(chosen_local)]
            if not (
                verification["valid"] and verification["progress_eligible"]
            ):
                raise ValueError("executed success-event action is not eligible")
            plan = torch.as_tensor(
                event["candidates"][selected[int(chosen_local)]],
                dtype=torch.float32,
            )
            executed_actions.append(plan[0])

        for start, event in enumerate(trajectory_events):
            compact = np.asarray(event["context"], np.float32)
            if compact.shape != (38,):
                raise ValueError(
                    "success-event compact context must have 38 values"
                )
            spheres = scene_spec.unpack(task.env, compact[13:])
            state = {
                "x": np.asarray(event["robot"], np.float32),
                "previous_applied": compact[7:10].copy(),
                "previous_raw": compact[10:13].copy(),
                "spheres": spheres,
                "scene_seed": 0,
                "scene_hash": str(event["scene_hash"]),
                "steps": int(event["step"]),
                "collided": False,
                "oob": False,
            }
            full_context = task.context(
                state, float(event["gamma"]),
            ).cpu()
            recovered_compact = torch.cat([
                full_context[:7], full_context[policy.context_dim:],
            ]).numpy()
            if not np.allclose(
                recovered_compact, compact, rtol=0.0, atol=2.0e-5,
            ):
                raise RuntimeError(
                    "could not faithfully rebuild event H_P context"
                )
            stop = min(start + 10, len(executed_actions))
            horizon = stop - start
            candidate = torch.zeros((10, 3), dtype=torch.float32)
            candidate[:horizon] = torch.stack(
                executed_actions[start:stop]
            )
            loss_mask = torch.zeros_like(candidate)
            loss_mask[:horizon] = 1.0
            rows.append(SimpleNamespace(
                context=full_context,
                candidate=candidate,
                loss_mask=loss_mask,
                gamma=float(event["gamma"]),
                trajectory_id=trajectory_id,
            ))
            groups[trajectory_id].append(len(rows) - 1)

        first = trajectory_events[0]
        slot = int(first["paired_scene_pair_slot"])
        previous = pair_slot_by_trajectory.setdefault(trajectory_id, slot)
        if previous != slot:
            raise ValueError("success episode changes paired-scene slot")
        gamma = float(first["gamma"])
        member = int(first["paired_scene_member"])
        stratum_by_trajectory[trajectory_id] = (gamma, member)
        gammas.add(gamma)
    if len(groups) < 40 or not rows:
        raise ValueError("success-event distillation found too little evidence")
    return (
        rows,
        dict(groups),
        pair_slot_by_trajectory,
        stratum_by_trajectory,
        sorted(gammas),
        event_path,
    )


def _stack_rows(rows, context_dim: int):
    contexts = torch.stack([
        row.context[:context_dim].float() for row in rows
    ])
    candidates = torch.stack([row.candidate.float() for row in rows])
    masks = torch.stack([
        row.loss_mask.float()
        if row.loss_mask is not None else torch.ones_like(row.candidate)
        for row in rows
    ])
    return contexts, candidates, masks


def _balanced_batch(
    rng: np.random.Generator,
    trajectory_ids: list[str],
    groups: dict[str, list[int]],
    batch_size: int,
) -> np.ndarray:
    repeats, remainder = divmod(batch_size, len(trajectory_ids))
    selected = []
    for trajectory_id in trajectory_ids:
        indices = groups[trajectory_id]
        selected.extend(
            int(indices[int(rng.integers(len(indices)))])
            for _ in range(repeats)
        )
    if remainder:
        extra = rng.choice(
            trajectory_ids,
            size=remainder,
            replace=remainder > len(trajectory_ids),
        )
        for trajectory_id in extra:
            indices = groups[str(trajectory_id)]
            selected.append(indices[int(rng.integers(len(indices)))])
    rng.shuffle(selected)
    return np.asarray(selected, np.int64)


def _stratified_batch(
    rng: np.random.Generator,
    trajectory_ids: list[str],
    groups: dict[str, list[int]],
    strata_by_trajectory: dict[str, tuple[float, int]],
    batch_size: int,
) -> np.ndarray:
    strata = defaultdict(list)
    for trajectory_id in trajectory_ids:
        strata[strata_by_trajectory[trajectory_id]].append(trajectory_id)
    keys = sorted(strata)
    if batch_size % len(keys):
        raise ValueError(
            "batch size must be divisible by gamma x mirror-member strata"
        )
    count = batch_size // len(keys)
    selected = []
    for key in keys:
        trajectories = rng.choice(
            strata[key], size=count, replace=count > len(strata[key]),
        )
        for trajectory_id in trajectories:
            indices = groups[str(trajectory_id)]
            selected.append(indices[int(rng.integers(len(indices)))])
    rng.shuffle(selected)
    return np.asarray(selected, np.int64)


def _cfm_loss(policy, contexts, candidates, masks):
    encoded = policy.encode_context(contexts)
    target = candidates.reshape(len(candidates), policy.flow.plan_dim)
    mask = masks.reshape_as(target)
    base = torch.randn_like(target) * mask
    time = torch.rand(len(target), device=target.device)
    point = (1.0 - time[:, None]) * base + time[:, None] * target
    error = policy.flow(point, time, encoded) - (target - base)
    return error.square().mul(mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def _fixed_cfm_loss(
    policy, contexts, candidates, masks, bases, times, batch_size: int,
) -> float:
    policy.eval()
    squared_error = 0.0
    coordinate_count = 0.0
    for start in range(0, len(contexts), batch_size):
        stop = start + batch_size
        context = contexts[start:stop]
        target = candidates[start:stop].reshape(
            len(context), policy.flow.plan_dim,
        )
        mask = masks[start:stop].reshape_as(target)
        base = bases[start:stop].reshape_as(target) * mask
        time = times[start:stop]
        point = (1.0 - time[:, None]) * base + time[:, None] * target
        prediction = policy.flow(
            point, time, policy.encode_context(context),
        )
        squared_error += float(
            (prediction - (target - base)).square().mul(mask).sum().cpu()
        )
        coordinate_count += float(mask.sum().cpu())
    policy.train()
    return squared_error / coordinate_count


def _trainable_parameters(policy, encoder_scope: str):
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    flow = list(policy.flow.parameters())
    for parameter in flow:
        parameter.requires_grad_(True)
    if encoder_scope == "frozen":
        encoder = []
    elif encoder_scope == "last_projection":
        encoder = list(policy.grid_encoder.radial_mixer[5].parameters())
    elif encoder_scope == "full":
        encoder = list(policy.grid_encoder.parameters())
    else:
        raise ValueError(f"unknown encoder scope {encoder_scope!r}")
    for parameter in encoder:
        parameter.requires_grad_(True)
    return flow, encoder


def _trainable_names(policy):
    return [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]


def _ema_update(ema, policy, names, decay: float):
    current = dict(policy.named_parameters())
    for name in names:
        ema[name].mul_(decay).add_(current[name].detach(), alpha=1.0 - decay)


@contextmanager
def _ema_parameters(policy, ema, names):
    parameters = dict(policy.named_parameters())
    raw = {name: parameters[name].detach().clone() for name in names}
    try:
        for name in names:
            parameters[name].data.copy_(ema[name])
        yield
    finally:
        for name in names:
            parameters[name].data.copy_(raw[name])


def _cosine_learning_rates(
    optimizer, step: int, total_steps: int, final_ratios: list[float],
):
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group, ratio in zip(optimizer.param_groups, final_ratios):
        base = float(group["initial_lr"])
        group["lr"] = base * (ratio + (1.0 - ratio) * cosine)


def _teacher_regularization(
    policy,
    teacher,
    calibration_contexts,
    batch_size: int,
    rng: np.random.Generator,
    vector_weight: float,
    feature_weight: float,
):
    if vector_weight == 0.0 and feature_weight == 0.0:
        zero = next(policy.parameters()).new_zeros(())
        return zero, zero
    ids = torch.as_tensor(
        rng.integers(len(calibration_contexts), size=batch_size),
        device=calibration_contexts.device,
        dtype=torch.long,
    )
    context = calibration_contexts[ids]
    point = torch.randn(
        len(context), policy.flow.plan_dim, device=context.device,
    )
    time = torch.rand(len(context), device=context.device)
    with torch.no_grad():
        teacher_encoded = teacher.encode_context(context)
        teacher_vector = teacher.flow(point, time, teacher_encoded)
    student_encoded = policy.encode_context(context)
    student_vector = policy.flow(point, time, student_encoded)
    vector_loss = (student_vector - teacher_vector).square().mean()
    feature_loss = (student_encoded - teacher_encoded).square().mean()
    return vector_loss, feature_loss


def _new_optimizer(policy, args):
    flow, encoder = _trainable_parameters(policy, args.encoder_scope)
    groups = [{
        "params": flow,
        "lr": float(args.flow_learning_rate),
        "initial_lr": float(args.flow_learning_rate),
        "name": "flow",
    }]
    ratios = [float(args.final_flow_learning_rate / args.flow_learning_rate)]
    if encoder:
        groups.append({
            "params": encoder,
            "lr": float(args.encoder_learning_rate),
            "initial_lr": float(args.encoder_learning_rate),
            "name": "encoder",
        })
        ratios.append(float(
            args.final_encoder_learning_rate / args.encoder_learning_rate
        ))
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return optimizer, ratios


def _training_run(
    *,
    args,
    pretrain_checkpoint: Path,
    contexts,
    candidates,
    masks,
    groups,
    trajectory_ids,
    strata_by_trajectory,
    calibration_contexts,
    steps: int,
    schedule_steps: int,
    device,
    seed: int,
    validation=None,
):
    policy = load_lab_reference_policy(pretrain_checkpoint).to(device).train()
    teacher = load_lab_reference_policy(pretrain_checkpoint).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer, final_ratios = _new_optimizer(policy, args)
    names = _trainable_names(policy)
    ema = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if name in set(names)
    }
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    curve = []
    best = None
    stale = 0

    def audit(step: int):
        nonlocal best, stale
        if validation is None:
            return False
        with _ema_parameters(policy, ema, names):
            value = _fixed_cfm_loss(policy, *validation, args.audit_batch_size)
            state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in policy.state_dict().items()
            }
        record = {
            "step": int(step),
            "heldout_fixed_cfm": float(value),
            "train_cfm_50": (
                float(np.mean([row["cfm"] for row in curve[-50:]]))
                if curve else None
            ),
            "flow_lr": float(optimizer.param_groups[0]["lr"]),
            "encoder_lr": (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1 else None
            ),
        }
        selectable = step >= args.minimum_steps
        if selectable and (
            best is None or value < best["value"] - args.min_delta
        ):
            best = {"value": float(value), "step": int(step), "model": state}
            stale = 0
            record["selected"] = True
        elif selectable:
            stale += 1
            record["selected"] = False
        else:
            record["selected"] = False
        print(
            f"[repair:{args.encoder_scope}] step={step} "
            f"heldout={value:.6f} train50={record['train_cfm_50']} "
            f"stale={stale}/{args.patience}", flush=True,
        )
        audits.append(record)
        return step >= args.minimum_steps and stale >= args.patience

    audits = []
    audit(0)
    completed_steps = 0
    for step in range(1, steps + 1):
        ids_np = (
            _stratified_batch(
                rng, trajectory_ids, groups, strata_by_trajectory,
                args.batch_size,
            )
            if strata_by_trajectory is not None
            else _balanced_batch(
                rng, trajectory_ids, groups, args.batch_size,
            )
        )
        ids = torch.as_tensor(ids_np, device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        cfm = _cfm_loss(
            policy, contexts[ids], candidates[ids], masks[ids],
        )
        vector_loss, feature_loss = _teacher_regularization(
            policy,
            teacher,
            calibration_contexts,
            args.distillation_batch_size,
            rng,
            args.vector_distillation_weight,
            args.feature_distillation_weight,
        )
        objective = (
            cfm
            + args.vector_distillation_weight * vector_loss
            + args.feature_distillation_weight * feature_loss
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in policy.parameters()
             if parameter.requires_grad],
            args.gradient_clip_norm,
        )
        optimizer.step()
        _ema_update(ema, policy, names, args.ema_decay)
        _cosine_learning_rates(
            optimizer, step, schedule_steps, final_ratios,
        )
        curve.append({
            "step": int(step),
            "cfm": float(cfm.detach().cpu()),
            "vector_distillation": float(vector_loss.detach().cpu()),
            "feature_distillation": float(feature_loss.detach().cpu()),
            "objective": float(objective.detach().cpu()),
        })
        completed_steps = step
        if validation is not None and step % args.audit_every == 0:
            if audit(step):
                break

    if validation is not None:
        return policy, {
            "curve": curve,
            "audits": audits,
            "best": best,
            "completed_steps": int(completed_steps),
            "trainable_names": names,
        }
    with _ema_parameters(policy, ema, names):
        final_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in policy.state_dict().items()
        }
    return policy, {
        "curve": curve,
        "final_model": final_state,
        "completed_steps": int(completed_steps),
        "trainable_names": names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--data-expansion", type=Path, required=True)
    parser.add_argument(
        "--training-source",
        choices=("committed_40", "successful_event_executed_windows"),
        default="committed_40",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoder-scope",
        choices=("frozen", "last_projection", "full"),
        required=True,
    )
    parser.add_argument("--maximum-steps", type=int, default=480)
    parser.add_argument("--minimum-steps", type=int, default=72)
    parser.add_argument("--audit-every", type=int, default=24)
    parser.add_argument("--audit-draws", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--holdout-pair-slot", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--audit-batch-size", type=int, default=256)
    parser.add_argument("--distillation-batch-size", type=int, default=16)
    parser.add_argument("--flow-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--final-flow-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--encoder-learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--final-encoder-learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--vector-distillation-weight", type=float, default=0.2)
    parser.add_argument("--feature-distillation-weight", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0 <= args.holdout_pair_slot < 5:
        parser.error("--holdout-pair-slot must lie in [0,4]")
    positive = (
        "maximum_steps", "minimum_steps", "audit_every", "audit_draws", "patience",
        "batch_size", "audit_batch_size", "distillation_batch_size",
        "flow_learning_rate", "encoder_learning_rate",
        "gradient_clip_norm", "ema_decay",
    )
    if any(float(getattr(args, name)) <= 0.0 for name in positive):
        parser.error("step, batch, LR, clipping, and EMA values must be positive")
    if args.minimum_steps > args.maximum_steps:
        parser.error("--minimum-steps cannot exceed --maximum-steps")
    if not 0.0 < args.ema_decay < 1.0:
        parser.error("--ema-decay must lie in (0,1)")
    if not 0.0 <= args.final_flow_learning_rate <= args.flow_learning_rate:
        parser.error("invalid final flow learning rate")
    if not 0.0 <= args.final_encoder_learning_rate <= args.encoder_learning_rate:
        parser.error("invalid final encoder learning rate")
    if min(
        args.weight_decay,
        args.vector_distillation_weight,
        args.feature_distillation_weight,
        args.min_delta,
    ) < 0.0:
        parser.error("regularization and min-delta values must be nonnegative")
    device = torch.device(args.device)
    pretrained = args.pretrain_dir / "pretrained.pt"
    probe_policy = load_lab_reference_policy(pretrained)
    if args.training_source == "committed_40":
        rows, archive_path = _load_rows(args.data_expansion)
        groups, metadata, gammas = _lineages(rows)
        pair_slot_by_trajectory = {
            trajectory_id: label // 2
            for trajectory_id, (_, label) in metadata.items()
        }
        strata_by_trajectory = None
    else:
        (
            rows, groups, pair_slot_by_trajectory, strata_by_trajectory,
            gammas, archive_path,
        ) = _load_success_event_executed_rows(
            args.data_expansion, probe_policy, torch.device("cpu"),
        )
    contexts_cpu, candidates_cpu, masks_cpu = _stack_rows(
        rows, probe_policy.context_dim,
    )
    train_ids = sorted([
        trajectory_id for trajectory_id in groups
        if pair_slot_by_trajectory[trajectory_id] != args.holdout_pair_slot
    ])
    validation_ids = sorted(set(groups) - set(train_ids))
    if not train_ids or not validation_ids:
        raise RuntimeError("held-out complete-pair split is empty")
    validation_rows = np.asarray([
        index for trajectory_id in validation_ids for index in groups[trajectory_id]
    ], np.int64)
    audit_generator = torch.Generator().manual_seed(args.seed + 101)
    validation_rows = np.repeat(validation_rows, int(args.audit_draws))
    validation_bases = torch.randn(
        (len(validation_rows), *candidates_cpu.shape[1:]),
        generator=audit_generator,
    )
    validation_times = torch.rand(
        len(validation_rows), generator=audit_generator,
    )
    calibration_contexts = torch.load(
        args.pretrain_dir / "calibration_contexts.pt",
        map_location="cpu", weights_only=False,
    ).float()
    if calibration_contexts.shape[1] != probe_policy.context_dim:
        raise ValueError("PRE calibration context schema does not match policy")

    contexts = contexts_cpu.to(device)
    candidates = candidates_cpu.to(device)
    masks = masks_cpu.to(device)
    calibration_contexts = calibration_contexts.to(device)
    validation_index = torch.as_tensor(
        validation_rows, device=device, dtype=torch.long,
    )
    validation = (
        contexts[validation_index],
        candidates[validation_index],
        masks[validation_index],
        validation_bases.to(device),
        validation_times.to(device),
    )
    _, selection = _training_run(
        args=args,
        pretrain_checkpoint=pretrained,
        contexts=contexts,
        candidates=candidates,
        masks=masks,
        groups=groups,
        trajectory_ids=train_ids,
        strata_by_trajectory=strata_by_trajectory,
        calibration_contexts=calibration_contexts,
        steps=args.maximum_steps,
        schedule_steps=args.maximum_steps,
        device=device,
        seed=args.seed,
        validation=validation,
    )
    selected_steps = int(selection["best"]["step"])
    _, final = _training_run(
        args=args,
        pretrain_checkpoint=pretrained,
        contexts=contexts,
        candidates=candidates,
        masks=masks,
        groups=groups,
        trajectory_ids=sorted(groups),
        strata_by_trajectory=strata_by_trajectory,
        calibration_contexts=calibration_contexts,
        steps=selected_steps,
        schedule_steps=args.maximum_steps,
        device=device,
        seed=args.seed + 1_000_003,
        validation=None,
    )

    args.output.mkdir(parents=True)
    shutil.copy2(
        args.data_expansion / "task_config_resolved.json",
        args.output / "task_config_resolved.json",
    )
    shutil.copy2(
        args.data_expansion / "checkpoint_000.pt",
        args.output / "checkpoint_000.pt",
    )
    source_checkpoint = torch.load(
        args.data_expansion / "checkpoint_001.pt",
        map_location="cpu", weights_only=False,
    )
    repaired_checkpoint = copy.deepcopy(source_checkpoint)
    repaired_checkpoint["model"] = final["final_model"]
    repaired_checkpoint["config"] = {
        **dict(source_checkpoint.get("config", {})),
        "optimization_repair": True,
        "encoder_scope": args.encoder_scope,
        "selected_optimizer_steps": selected_steps,
        "selection_rule": "minimum heldout complete-pair fixed CFM loss",
        "candidate_source": "round-1 committed executed windows",
    }
    repaired_checkpoint["pretrained"] = False
    torch.save(repaired_checkpoint, args.output / "checkpoint_001.pt")

    source_manifest = json.loads(
        (args.data_expansion / "manifest.json").read_text()
    )
    evaluator_manifest = copy.deepcopy(source_manifest)
    evaluator_manifest["status"] = "completed"
    evaluator_manifest["config"]["rounds"] = 1
    evaluator_manifest["optimization_repair"] = {
        "kind": "heldout-pair-selected PRE2 post-training",
        "encoder_scope": args.encoder_scope,
        "selected_optimizer_steps": selected_steps,
        "source_archive": str(archive_path.resolve()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(evaluator_manifest, indent=2, allow_nan=False) + "\n"
    )
    repair_manifest = {
        "kind": "round-1 multi-sphere optimization repair",
        "training_source": args.training_source,
        "source_pretrained": str(pretrained.resolve()),
        "source_expansion": str(args.data_expansion.resolve()),
        "source_archive": str(archive_path.resolve()),
        "encoder_scope": args.encoder_scope,
        "trajectory_count": len(groups),
        "row_count": len(rows),
        "gammas": gammas,
        "holdout_pair_slot": int(args.holdout_pair_slot),
        "training_trajectory_count": len(train_ids),
        "validation_trajectory_count": len(validation_ids),
        "validation_trajectory_ids": validation_ids,
        "selection_completed_steps": selection["completed_steps"],
        "selected_optimizer_steps": selected_steps,
        "selected_heldout_fixed_cfm": selection["best"]["value"],
        "final_training_steps": final["completed_steps"],
        "trainable_parameter_names": final["trainable_names"],
        "trainable_parameter_count": int(sum(
            tensor.numel() for name, tensor in final["final_model"].items()
            if name in set(final["trainable_names"])
        )),
        "batch_sampler": "exact trajectory-lineage balanced",
        "batch_size": int(args.batch_size),
        "flow_learning_rate": float(args.flow_learning_rate),
        "final_flow_learning_rate": float(args.final_flow_learning_rate),
        "encoder_learning_rate": (
            None if args.encoder_scope == "frozen"
            else float(args.encoder_learning_rate)
        ),
        "final_encoder_learning_rate": (
            None if args.encoder_scope == "frozen"
            else float(args.final_encoder_learning_rate)
        ),
        "weight_decay": float(args.weight_decay),
        "vector_distillation_weight": float(args.vector_distillation_weight),
        "feature_distillation_weight": float(args.feature_distillation_weight),
        "distillation_context_count": int(len(calibration_contexts)),
        "ema_decay": float(args.ema_decay),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "seed": int(args.seed),
        "selection_audits": selection["audits"],
    }
    (args.output / "repair_manifest.json").write_text(
        json.dumps(repair_manifest, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "training_curve.json").write_text(json.dumps({
        "selection_train": selection["curve"],
        "final_all_40_train": final["curve"],
    }, indent=2, allow_nan=False) + "\n")
    print(json.dumps(repair_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
