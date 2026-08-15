"""Memorization diagnostic on exact mode-quota committed trajectories.

This intentionally performs no rollout collection.  It loads one or more
successful-executed-window archives, samples trajectory lineages uniformly,
and repeatedly applies the CFM objective to only those committed rows.  The
output checkpoints are evaluator-compatible so low training loss and raw
closed-loop route coverage can be tested as separate claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from safe_mppi.expansion import _trunk_suffix_parameters
from safe_mppi.lab_visual_flow import load_lab_reference_policy


MODES = ("below", "above", "left", "right")


def _cfm_loss(
    flow, contexts, candidates, loss_masks, coupling: str,
    paired_bases=None,
):
    target = candidates.reshape(len(candidates), flow.plan_dim)
    mask = loss_masks.reshape_as(target)
    if coupling == "paired_base":
        if paired_bases is None:
            raise ValueError("paired_base coupling requires stored flow bases")
        base = paired_bases.reshape_as(target)
    else:
        base = torch.randn_like(target)
    if coupling == "batch_ot":
        pairwise = target[:, None, :] - base[None, :, :]
        cost = (
            (pairwise.detach().square() * mask[:, None, :]).sum(dim=-1)
            / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ).cpu().numpy()
        target_ids, base_ids = linear_sum_assignment(cost)
        if not np.array_equal(target_ids, np.arange(len(target))):
            raise RuntimeError("OT assignment did not cover target rows in order")
        base = base[torch.as_tensor(base_ids, device=base.device)]
    # Match ConditionalFlowMLP.cfm_loss exactly for short terminal suffixes:
    # padded target, interpolation point, and vector target are all zero there.
    base = base * mask
    time = torch.rand(len(target), device=target.device)
    point = (1.0 - time[:, None]) * base + time[:, None] * target
    squared_error = (
        flow(point, time, contexts) - (target - base)
    ).square()
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def _trajectory_modes(expansion: Path) -> dict[str, int]:
    manifest = json.loads((expansion / "manifest.json").read_text())
    mapping = {}
    for round_row in manifest["rounds"]:
        for detail in round_row[
            "successful_executed_commit_by_gamma"
        ].values():
            trajectory_ids = detail.get("committed_trajectory_ids", [])
            modes = detail.get("committed_sample_update_modes", [])
            if len(trajectory_ids) != len(modes):
                raise ValueError(
                    f"trajectory/mode count mismatch in {expansion}"
                )
            for trajectory_id, mode in zip(trajectory_ids, modes):
                if trajectory_id in mapping:
                    raise ValueError(
                        f"duplicate committed trajectory id {trajectory_id!r}"
                    )
                mapping[str(trajectory_id)] = int(mode)
    if not mapping:
        raise ValueError(f"no exact mode-quota trajectories in {expansion}")
    return mapping


def _trajectory_gammas(expansion: Path) -> dict[str, float]:
    manifest = json.loads((expansion / "manifest.json").read_text())
    mapping = {}
    for round_row in manifest["rounds"]:
        for gamma, detail in round_row[
            "successful_executed_commit_by_gamma"
        ].items():
            for trajectory_id in detail.get("committed_trajectory_ids", []):
                if trajectory_id in mapping:
                    raise ValueError(
                        f"duplicate committed trajectory id {trajectory_id!r}"
                    )
                mapping[str(trajectory_id)] = float(gamma)
    return mapping


def _load_rows(expansions: list[Path]):
    rows = []
    row_modes = []
    lineage_sources = {}
    for source_index, expansion in enumerate(expansions):
        mode_by_trajectory = _trajectory_modes(expansion)
        gamma_by_trajectory = _trajectory_gammas(expansion)
        archive = torch.load(
            expansion / "query_archive.pt",
            map_location="cpu",
            weights_only=False,
        )
        seen = set()
        for row in archive:
            trajectory_id = str(row.trajectory_id)
            if trajectory_id not in mode_by_trajectory:
                continue
            if not row.verification.valid or not row.replay_eligible:
                continue
            global_id = f"source{source_index}:{trajectory_id}"
            rows.append(row)
            row_modes.append(mode_by_trajectory[trajectory_id])
            lineage_sources[global_id] = (
                source_index, trajectory_id,
                mode_by_trajectory[trajectory_id],
                gamma_by_trajectory[trajectory_id],
            )
            seen.add(trajectory_id)
        missing = set(mode_by_trajectory) - seen
        if missing:
            raise ValueError(
                f"archive {expansion} is missing committed trajectories: "
                f"{sorted(missing)}"
            )
    return rows, row_modes, lineage_sources


def _lineage_indices(rows, expansions: list[Path]):
    # Rows from separate sources can reuse the same trajectory id.  Preserve
    # source identity by detecting source blocks in the order loaded above.
    groups = defaultdict(list)
    offset = 0
    for source_index, expansion in enumerate(expansions):
        source_mapping = _trajectory_modes(expansion)
        archive = torch.load(
            expansion / "query_archive.pt",
            map_location="cpu",
            weights_only=False,
        )
        accepted = [
            row for row in archive
            if str(row.trajectory_id) in source_mapping
            and row.verification.valid and row.replay_eligible
        ]
        for local_index, row in enumerate(accepted):
            groups[f"source{source_index}:{row.trajectory_id}"].append(
                offset + local_index
            )
        offset += len(accepted)
    if offset != len(rows):
        raise RuntimeError("lineage indexing does not match loaded replay rows")
    return dict(groups)


@torch.no_grad()
def _encode_contexts(policy, rows, device, batch_size: int):
    values = []
    for start in range(0, len(rows), batch_size):
        external = torch.stack([
            row.context[:policy.context_dim]
            for row in rows[start:start + batch_size]
        ]).to(device)
        values.append(policy.encode_context(external).detach())
    return torch.cat(values)


@torch.no_grad()
def _fixed_audit(
    policy, encoded, candidates, loss_masks, bases, audit_times, batch_size,
):
    target = candidates.reshape(len(candidates), policy.flow.plan_dim)
    mask = loss_masks.reshape_as(target)
    base = bases.reshape_as(target) * mask
    by_time = {}
    for audit_time in audit_times:
        squared_error = 0.0
        value_count = 0
        for start in range(0, len(target), batch_size):
            stop = start + batch_size
            target_batch = target[start:stop]
            base_batch = base[start:stop]
            time = torch.full(
                (len(target_batch),), float(audit_time),
                device=target.device, dtype=target.dtype,
            )
            point = (
                (1.0 - time[:, None]) * base_batch
                + time[:, None] * target_batch
            )
            prediction = policy.flow(
                point, time, encoded[start:stop],
            )
            mask_batch = mask[start:stop]
            squared_error += float(
                (prediction - (target_batch - base_batch))
                .square().mul(mask_batch).sum().detach().cpu()
            )
            value_count += float(mask_batch.sum().detach().cpu())
        by_time[f"{float(audit_time):g}"] = squared_error / value_count
    return {
        "fixed_noise_time_mse_mean": float(np.mean(list(by_time.values()))),
        "fixed_noise_mse_by_time": by_time,
    }


def _save_checkpoint(output, stage, step, policy, args, losses, audit):
    torch.save({
        "round": int(stage),
        "model": policy.state_dict(),
        "config": {
            "diagnostic": "committed_mode_memorization",
            "optimizer_step": int(step),
            "learning_rate": float(args.learning_rate),
            "final_learning_rate": args.final_learning_rate,
            "weight_decay": float(args.weight_decay),
            "anchor_lambda": float(args.anchor_lambda),
            "gradient_clip_norm": args.gradient_clip_norm,
            "flow_coupling": args.flow_coupling,
            "mode_sampling_weights": args.mode_sampling_weights,
            "train_scope": args.train_scope,
            "trainable_trunk_layers": args.trainable_trunk_layers,
        },
        "pretrained": bool(stage == 0),
    }, output / f"checkpoint_{stage:03d}.pt")
    return {
        "round": int(stage),
        "optimizer_step": int(step),
        "mean_recent_loss": (
            float(np.mean(losses[-100:])) if losses else None
        ),
        "audit": audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-expansion", type=Path, action="append", required=True,
        help="repeat to combine independently collected exact-quota archives",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-steps", type=int, nargs="+", default=(0, 100, 300, 1000, 3000))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--final-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--anchor-lambda", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument(
        "--batch-sampler",
        choices=("trajectory_uniform", "mode_gamma_stratified"),
        default="trajectory_uniform",
    )
    parser.add_argument(
        "--mode-sampling-weights", type=float, nargs=4,
        metavar=("BELOW", "ABOVE", "LEFT", "RIGHT"),
        help=(
            "optional positive mode weights for mode_gamma_stratified batches; "
            "gamma remains uniform within every mode"
        ),
    )
    parser.add_argument(
        "--flow-coupling",
        choices=("independent", "batch_ot", "paired_base"),
        default="independent",
    )
    parser.add_argument("--train-scope", choices=("head", "flow"), default="flow")
    parser.add_argument(
        "--trainable-trunk-layers", type=int,
        help=("train the head plus the last N parameterised flow-trunk "
              "layers; overrides --train-scope flow"),
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--audit-times", type=float, nargs="+",
        default=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99),
    )
    parser.add_argument("--audit-batch-size", type=int, default=1024)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    save_steps = sorted(set(map(int, args.save_steps)))
    if save_steps[0] != 0 or save_steps[-1] != args.steps:
        parser.error("--save-steps must start at 0 and end at --steps")
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0.0:
        parser.error("require positive steps, batch size, and learning rate")
    if args.audit_batch_size < 1 or any(
        not 0.0 <= value <= 1.0 for value in args.audit_times
    ):
        parser.error("audit batch size must be positive and times lie in [0,1]")
    if args.final_learning_rate is not None and not (
        0.0 <= args.final_learning_rate <= args.learning_rate
    ):
        parser.error(
            "--final-learning-rate must lie in [0, --learning-rate]"
        )
    if args.weight_decay < 0.0 or args.anchor_lambda < 0.0:
        parser.error("weight decay and anchor lambda must be nonnegative")
    if args.gradient_clip_norm is not None and args.gradient_clip_norm <= 0.0:
        parser.error("--gradient-clip-norm must be positive")
    if args.trainable_trunk_layers is not None:
        if args.trainable_trunk_layers < 1:
            parser.error("--trainable-trunk-layers must be positive")
        if args.train_scope == "head":
            parser.error(
                "--trainable-trunk-layers cannot be combined with "
                "--train-scope head"
            )
    if args.mode_sampling_weights is not None:
        if args.batch_sampler != "mode_gamma_stratified":
            parser.error(
                "--mode-sampling-weights requires mode_gamma_stratified"
            )
        if any(value <= 0.0 for value in args.mode_sampling_weights):
            parser.error("--mode-sampling-weights must all be positive")

    device = torch.device(args.device)
    policy = load_lab_reference_policy(
        args.pretrain_dir / "pretrained.pt"
    ).to(device)
    base = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False,
    )
    policy.load_state_dict(base["model"], strict=True)
    policy.train()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if args.trainable_trunk_layers is not None:
        trainable = _trunk_suffix_parameters(
            policy.flow, args.trainable_trunk_layers,
        )
    else:
        trainable = list(
            policy.flow.head.parameters()
            if args.train_scope == "head"
            else policy.flow.parameters()
        )
        for parameter in trainable:
            parameter.requires_grad_(True)

    rows, row_modes, lineage_sources = _load_rows(args.data_expansion)
    groups = _lineage_indices(rows, args.data_expansion)
    trajectory_ids = sorted(groups)
    trajectory_mode_counts = Counter(
        lineage_sources[trajectory_id][2]
        for trajectory_id in trajectory_ids
    )
    if len(set(trajectory_mode_counts.values())) != 1:
        raise ValueError(
            "memorization diagnostic requires equal trajectory quota per mode; "
            f"got {dict(trajectory_mode_counts)}"
        )
    strata = defaultdict(list)
    for trajectory_id in trajectory_ids:
        metadata = lineage_sources[trajectory_id]
        strata[(metadata[2], metadata[3])].append(trajectory_id)
    if args.batch_sampler == "mode_gamma_stratified" and len(strata) != 16:
        raise ValueError(
            "mode_gamma_stratified sampling requires all 4 modes x 4 gammas; "
            f"got {sorted(strata)}"
        )

    encoded = _encode_contexts(policy, rows, device, args.batch_size)
    candidates = torch.stack([row.candidate for row in rows]).to(device)
    loss_masks = torch.stack([
        (
            row.loss_mask
            if row.loss_mask is not None
            else torch.ones_like(row.candidate)
        )
        for row in rows
    ]).to(device)
    paired_bases = None
    if args.flow_coupling == "paired_base":
        if any(row.flow_base is None for row in rows):
            raise ValueError(
                "paired_base coupling requires flow_base on every replay row"
            )
        paired_bases = torch.stack([row.flow_base for row in rows]).to(device)
    audit_generator = torch.Generator().manual_seed(args.seed + 9049)
    audit_bases = torch.randn(
        candidates.shape, generator=audit_generator,
        dtype=candidates.dtype,
    ).to(device)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.final_learning_rate,
        )
        if args.final_learning_rate is not None else None
    )
    anchor_parameters = [parameter.detach().clone() for parameter in trainable]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable
    )
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    losses = []
    anchor_losses = []
    objective_losses = []
    checkpoints = []
    args.output.mkdir(parents=True)
    audit = _fixed_audit(
        policy, encoded, candidates, loss_masks, audit_bases,
        args.audit_times, args.audit_batch_size,
    )
    checkpoints.append(_save_checkpoint(
        args.output, 0, 0, policy, args, losses, audit,
    ))
    next_stage = 1
    for step in range(1, args.steps + 1):
        if args.batch_sampler == "trajectory_uniform":
            chosen_trajectories = rng.choice(
                trajectory_ids, size=args.batch_size, replace=True,
            )
        else:
            keys = sorted(strata)
            if args.mode_sampling_weights is None:
                probabilities = np.full(len(keys), 1.0 / len(keys))
            else:
                mode_weights = np.asarray(args.mode_sampling_weights, float)
                probabilities = np.asarray([
                    mode_weights[int(key[0])] for key in keys
                ], float)
                probabilities /= probabilities.sum()
            expected = args.batch_size * probabilities
            counts = np.floor(expected).astype(int)
            remainder = args.batch_size - int(counts.sum())
            if remainder:
                order = np.argsort(-(expected - counts), kind="stable")
                counts[order[:remainder]] += 1
            chosen_trajectories = []
            for key, count in zip(keys, counts):
                chosen_trajectories.extend(
                    rng.choice(strata[key], size=int(count), replace=True)
                )
            rng.shuffle(chosen_trajectories)
        chosen_rows = [
            groups[str(trajectory_id)][
                int(rng.integers(len(groups[str(trajectory_id)])))
            ]
            for trajectory_id in chosen_trajectories
        ]
        ids = torch.as_tensor(chosen_rows, device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        cfm_loss = _cfm_loss(
            policy.flow, encoded[ids], candidates[ids], loss_masks[ids],
            args.flow_coupling,
            None if paired_bases is None else paired_bases[ids],
        )
        anchor_loss = sum(
            (parameter - reference).square().sum()
            for parameter, reference in zip(trainable, anchor_parameters)
        ) / trainable_parameter_count
        objective = cfm_loss + args.anchor_lambda * anchor_loss
        objective.backward()
        if args.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                trainable, args.gradient_clip_norm,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(cfm_loss.detach().cpu()))
        anchor_losses.append(float(anchor_loss.detach().cpu()))
        objective_losses.append(float(objective.detach().cpu()))
        if next_stage < len(save_steps) and step == save_steps[next_stage]:
            audit = _fixed_audit(
                policy, encoded, candidates, loss_masks, audit_bases,
                args.audit_times, args.audit_batch_size,
            )
            checkpoints.append(_save_checkpoint(
                args.output, next_stage, step, policy, args, losses, audit,
            ))
            print(
                f"[memorize] step {step}/{args.steps} "
                f"loss100={np.mean(losses[-100:]):.6f} "
                f"audit={audit['fixed_noise_time_mse_mean']:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3g}",
                flush=True,
            )
            next_stage += 1

    source_task_config = args.data_expansion[0] / "task_config_resolved.json"
    shutil.copy2(source_task_config, args.output / "task_config_resolved.json")
    manifest = {
        "kind": "committed-mode memorization diagnostic",
        "config": {"rounds": len(checkpoints) - 1},
        "rounds": checkpoints[1:],
        "checkpoint_stage_map": checkpoints,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "data_expansions": [str(path.resolve()) for path in args.data_expansion],
        "trajectory_counts_by_mode": {
            MODES[mode]: int(trajectory_mode_counts[mode])
            for mode in range(4)
        },
        "replay_row_counts_by_mode": {
            MODES[mode]: int(sum(value == mode for value in row_modes))
            for mode in range(4)
        },
        "sampling": (
            "uniform trajectory lineage, then uniform window"
            if args.batch_sampler == "trajectory_uniform"
            else "mode-gamma-stratified trajectory lineage, then uniform window"
        ),
        "batch_sampler": args.batch_sampler,
        "mode_sampling_weights": args.mode_sampling_weights,
        "flow_coupling": args.flow_coupling,
        "train_scope": args.train_scope,
        "trainable_trunk_layers": args.trainable_trunk_layers,
        "trainable_parameter_count": int(sum(
            parameter.numel() for parameter in trainable
        )),
        "learning_rate": float(args.learning_rate),
        "final_learning_rate": args.final_learning_rate,
        "weight_decay": float(args.weight_decay),
        "anchor_lambda": float(args.anchor_lambda),
        "gradient_clip_norm": args.gradient_clip_norm,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "audit_times": [float(value) for value in args.audit_times],
        "audit_noise_seed": int(args.seed + 9049),
        "valid_replay_coordinate_fraction": float(loss_masks.mean().cpu()),
        "terminal_padding_semantics": (
            "zero candidate/base/interpolation/vector-target coordinates and "
            "exclude padded output coordinates from loss"
        ),
        "seed": int(args.seed),
        "loss_first_100": float(np.mean(losses[:100])),
        "loss_last_100": float(np.mean(losses[-100:])),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output / "training_loss.json").write_text(
        json.dumps({
            "cfm_loss": losses,
            "anchor_loss": anchor_losses,
            "objective_loss": objective_losses,
        }, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
