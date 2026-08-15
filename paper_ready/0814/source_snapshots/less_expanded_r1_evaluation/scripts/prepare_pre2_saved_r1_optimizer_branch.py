#!/usr/bin/env python3
"""Rebuild a committed r1 boundary with more optimizer exposure.

The source r1 trajectories are reused byte-for-byte.  No acquisition or
verification is rerun.  Before producing the branch, the utility replays the
historical optimizer budget from PRE2 and requires its model to equal the
published source ``checkpoint_001.pt`` bit-for-bit.  It then restarts from the
same PRE2 state and runs one uninterrupted, identically seeded replay stream
to the requested larger budget.

The resulting directory is an exact-state r1 resume package.  Continue it in
place with ``research_ball_expansion_optimization.py --resume-from OUTPUT``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(os.environ.get(
    "SAFE_MPPI_SOURCE_ROOT", Path(__file__).resolve().parents[1],
)).resolve()
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return deepcopy(value)


def _same_model(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor],
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def _model_displacement(
    base: dict[str, torch.Tensor], current: dict[str, torch.Tensor],
) -> dict[str, float]:
    if base.keys() != current.keys():
        raise ValueError("model state keys differ")
    squared_delta = 0.0
    squared_base = 0.0
    max_abs_delta = 0.0
    for name in base:
        left = base[name].detach().cpu().to(torch.float64)
        right = current[name].detach().cpu().to(torch.float64)
        delta = right - left
        squared_delta += float(delta.square().sum())
        squared_base += float(left.square().sum())
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max()))
    l2 = math.sqrt(squared_delta)
    return {
        "l2": l2,
        "relative_l2": l2 / max(math.sqrt(squared_base), 1.0e-30),
        "max_abs": max_abs_delta,
    }


def _read_round_one(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if int(row.get("round", -1)) == 1]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one r1 metric row in {path}")
    return selected[0]


def _branch_config(
    source: dict[str, Any], *, optimizer_steps: int, target_rounds: int,
) -> dict[str, Any]:
    if int(source.get("inner_steps", -1)) != 2500:
        raise ValueError("source r1 must use 2500 optimizer steps")
    if source.get("optimizer_steps_total") is not None:
        raise ValueError("source must use a per-round optimizer budget")
    if int(source.get("microbatch_repeats", -1)) != 1:
        raise ValueError("source must use one update per sampled microbatch")
    if int(source.get("replay_passes", -1)) != 1:
        raise ValueError("source must use one replay pass")
    if source.get("replay_batch_sampler") != "mode_gamma_stratified":
        raise ValueError("source must use mode_gamma_stratified replay")
    if source.get("replay_selector") != "uniform":
        raise ValueError("source must use uniform replay selection")
    if source.get("archive_rule") != "successful_executed_windows":
        raise ValueError("source must use successful executed windows")
    if not bool(source.get("freeze_visual_encoder", False)):
        raise ValueError("source must freeze the visual encoder")
    if bool(source.get("head_only_update", False)):
        raise ValueError("source must use the approved trunk3 update scope")
    if source.get("replay_augmentation") != "none":
        raise ValueError("source replay augmentation must be disabled")
    if float(source.get("negative_alpha", math.nan)) != 0.0:
        raise ValueError("source negative replay must be disabled")
    if float(source.get("round_learning_rate_warmup_power", math.nan)) != 0.0:
        raise ValueError("target-round changes require no outer-round LR schedule")
    if float(source.get("learning_rate", math.nan)) != float(
        source.get("learning_rate_final", math.nan)
    ):
        raise ValueError("exact branch requires the source constant learning rate")
    if optimizer_steps <= 2500:
        raise ValueError("target optimizer steps must exceed the 2500-step source")
    if target_rounds < 2:
        raise ValueError("target rounds must extend beyond r1")
    result = deepcopy(source)
    result["inner_steps"] = int(optimizer_steps)
    result["rounds"] = int(target_rounds)
    changed = {
        key for key in source if source[key] != result[key]
    }
    if changed != {"inner_steps", "rounds"}:
        raise RuntimeError(f"unexpected branch config changes: {sorted(changed)}")
    return result


def _validate_archive(
    rows: list[Any], config: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("r1 archive is empty")
    if any(int(row.round) != 1 for row in rows):
        raise ValueError("r1 archive contains another round")
    if any(
        not bool(row.verification.valid) or not bool(row.replay_eligible)
        for row in rows
    ):
        raise ValueError("r1 archive contains a non-replay-positive row")
    requested_modes = tuple(int(value) for value in config["sample_update_mode"])
    requested = Counter(requested_modes)
    gammas = tuple(float(value) for value in config["gammas"])
    lineage_cell: dict[str, tuple[float, int]] = {}
    lineages_by_cell: dict[tuple[float, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.trajectory_id is None or row.sample_update_mode is None:
            raise ValueError("every archived row must carry trajectory and mode IDs")
        lineage = str(row.trajectory_id)
        cell = (float(row.gamma), int(row.sample_update_mode))
        previous = lineage_cell.setdefault(lineage, cell)
        if previous != cell:
            raise ValueError(f"trajectory {lineage} changes mode/gamma cell")
        lineages_by_cell[cell].add(lineage)
    expected_cells = {
        (gamma, mode): count
        for gamma in gammas for mode, count in requested.items()
    }
    observed_cells = {
        cell: len(lineages) for cell, lineages in lineages_by_cell.items()
    }
    if observed_cells != expected_cells:
        raise ValueError(
            "r1 archive does not contain the exact requested trajectory quota: "
            f"observed={observed_cells}, expected={expected_cells}"
        )
    return {
        "row_count": len(rows),
        "trajectory_count": len(lineage_cell),
        "lineages_by_mode_gamma": {
            f"mode={mode},gamma={gamma:g}": observed_cells[(gamma, mode)]
            for gamma in gammas for mode in sorted(requested)
        },
    }


def _replay_once(
    *,
    pretrain: Path,
    checkpoint_zero: dict[str, Any],
    rows: list[Any],
    config_dict: dict[str, Any],
    optimizer_steps: int,
    trainable_trunk_layers: int,
    device: torch.device,
) -> dict[str, Any]:
    # Private imports are deliberate: this utility must exercise the exact
    # online replay and optimizer implementation, not a lookalike loop.
    from safe_mppi.expansion import (
        ExpansionConfig,
        _set_round_learning_rate,
        _trunk_suffix_parameters,
        _update,
    )
    from safe_mppi.lab_flow_expansion import load_lab_expansion_policy

    replay_config = deepcopy(config_dict)
    replay_config["inner_steps"] = int(optimizer_steps)
    config = ExpansionConfig(**replay_config)
    config.validate()
    policy = load_lab_expansion_policy(pretrain).to(device)
    policy.load_state_dict(checkpoint_zero["model"], strict=True)
    if config.freeze_visual_encoder:
        policy.freeze_visual_encoder_for_expansion()
    parameters = _trunk_suffix_parameters(policy, trainable_trunk_layers)
    names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    # This is the ordering in run_safe_expansion: optimizer construction,
    # dedicated replay generator seed, then global CFM seed.
    numpy_rng = np.random.default_rng(config.seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    round_scale, start_lr = _set_round_learning_rate(optimizer, config, 1)
    started = time.perf_counter()
    update = _update(
        policy, None, optimizer, rows, [], config, torch_rng,
    )
    elapsed = time.perf_counter() - started
    if int(update["steps"]) != optimizer_steps:
        raise RuntimeError("replay did not execute the exact requested step count")
    model = _cpu_clone(policy.state_dict())
    result = {
        "model": model,
        "optimizer": _cpu_clone(optimizer.state_dict()),
        "optimizer_metadata": {
            name: _cpu_clone(getattr(optimizer, name))
            for name in (
                "_safe_mppi_schedule_step", "_safe_mppi_base_lrs",
                "_safe_mppi_round_lr_scale",
            )
            if hasattr(optimizer, name)
        },
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters
        ),
        "numpy_rng_state": deepcopy(numpy_rng.bit_generator.state),
        "torch_rng_state": torch_rng.get_state().cpu().clone(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().clone(),
        "torch_device_rng_state": (
            torch.cuda.get_rng_state(device).cpu().clone()
            if device.type == "cuda" else None
        ),
        "round_learning_rate_scale": float(round_scale),
        "learning_rate_at_round_start": float(start_lr),
        "update": update,
        "update_s": float(elapsed),
    }
    del optimizer, policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _truncate_first_action(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    payload["rows"] = [
        row for row in payload.get("rows", [])
        if int(row.get("round", -1)) <= 1
    ]
    if {int(row["round"]) for row in payload["rows"]} != {0, 1}:
        raise ValueError("source first-action log must contain r0 and r1")
    output.write_text(json.dumps(payload, indent=2) + "\n")


def _truncate_fa_log(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    payload["retry_progress"] = [
        row for row in payload.get("retry_progress", [])
        if int(row.get("round", -1)) <= 1
    ]
    payload["rounds"] = {
        key: value for key, value in payload.get("rounds", {}).items()
        if int(key) <= 1
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def prepare_branch(
    *,
    source: Path,
    pretrain_dir: Path,
    output: Path,
    device: str = "cpu",
    reference_optimizer_steps: int = 2500,
    target_optimizer_steps: int = 5000,
    target_rounds: int = 7,
    trainable_trunk_layers: int = 3,
    replay_once: Callable[..., dict[str, Any]] = _replay_once,
) -> dict[str, Any]:
    source = source.resolve()
    pretrain_dir = pretrain_dir.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if reference_optimizer_steps != 2500:
        raise ValueError("historical reference budget must remain 2500")
    if trainable_trunk_layers != 3:
        raise ValueError("this audited speed branch requires trunk3")
    required = (
        "checkpoint_000.pt", "checkpoint_001.pt",
        "query_archive_round_001.pt", "events_round_001.pt",
        "metrics.jsonl", "first_action_stats.json", "fa_alloc_log.json",
    )
    immutable_source = [
        "checkpoint_000.pt", "checkpoint_001.pt",
        "query_archive_round_001.pt", "events_round_001.pt",
    ]
    optional_task_config = source / "task_config_resolved.json"
    if optional_task_config.is_file():
        immutable_source.append("task_config_resolved.json")
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    pretrained = pretrain_dir / "pretrained.pt"
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    source_hashes = {name: _sha256(source / name) for name in required}
    if optional_task_config.is_file():
        source_hashes["task_config_resolved.json"] = _sha256(
            optional_task_config
        )
    source_hashes["pretrained.pt"] = _sha256(pretrained)
    checkpoint_zero = torch.load(
        source / "checkpoint_000.pt", map_location="cpu", weights_only=False,
    )
    checkpoint_one = torch.load(
        source / "checkpoint_001.pt", map_location="cpu", weights_only=False,
    )
    if int(checkpoint_zero.get("round", -1)) != 0:
        raise ValueError("source checkpoint_000 is not PRE2/r0")
    if int(checkpoint_one.get("round", -1)) != 1:
        raise ValueError("source checkpoint_001 is not r1")
    if not isinstance(checkpoint_zero.get("model"), dict):
        raise ValueError("checkpoint_000 has no model state")
    if not isinstance(checkpoint_one.get("model"), dict):
        raise ValueError("checkpoint_001 has no model state")
    source_config = deepcopy(checkpoint_zero["config"])
    if checkpoint_one.get("config") != source_config:
        raise ValueError("source r0/r1 checkpoint configs differ")
    branch_config = _branch_config(
        source_config,
        optimizer_steps=target_optimizer_steps,
        target_rounds=target_rounds,
    )
    rows = torch.load(
        source / "query_archive_round_001.pt",
        map_location="cpu", weights_only=False,
    )
    archive = _validate_archive(rows, source_config)
    source_row = _read_round_one(source / "metrics.jsonl")
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requires CUDA")

    reference = replay_once(
        pretrain=pretrained,
        checkpoint_zero=checkpoint_zero,
        rows=rows,
        config_dict=source_config,
        optimizer_steps=reference_optimizer_steps,
        trainable_trunk_layers=trainable_trunk_layers,
        device=device_value,
    )
    if not _same_model(reference["model"], checkpoint_one["model"]):
        raise RuntimeError(
            "historical r1/2500 model was not reproduced bit-for-bit; "
            "refusing to create a mislabeled 5000-step branch"
        )
    target = replay_once(
        pretrain=pretrained,
        checkpoint_zero=checkpoint_zero,
        rows=rows,
        config_dict=branch_config,
        optimizer_steps=target_optimizer_steps,
        trainable_trunk_layers=trainable_trunk_layers,
        device=device_value,
    )
    target_update = target["update"]
    reference_displacement = _model_displacement(
        checkpoint_zero["model"], reference["model"]
    )
    target_displacement = _model_displacement(
        checkpoint_zero["model"], target["model"]
    )
    target_row = deepcopy(source_row)
    target_row.update(target_update)
    target_row.update({
        "round_learning_rate_scale": target["round_learning_rate_scale"],
        "learning_rate_at_round_start": target[
            "learning_rate_at_round_start"
        ],
        "update_s": target["update_s"],
        "round_total_s": (
            float(source_row["round_total_s"])
            - float(source_row["update_s"])
            + float(target["update_s"])
        ),
        "optimizer_replay_bootstrap": True,
    })

    branch_checkpoint = {
        "round": 1,
        "model": target["model"],
        "config": branch_config,
        "pretrained": False,
    }
    resume = {
        "version": 1,
        "status": "COMMITTED_ROUND_RESUME",
        "completed_round": 1,
        "config": branch_config,
        "model": target["model"],
        "optimizer": target["optimizer"],
        "optimizer_metadata": target["optimizer_metadata"],
        "trainable_parameter_names": target["trainable_parameter_names"],
        "trainable_parameter_count": target["trainable_parameter_count"],
        "beta": float(branch_config["beta"]),
        "round_rows": [target_row],
        "archive": rows,
        "frozen_gp_rows": None,
        "frozen_gp_hash": None,
        "round1_gp_candidates": [],
        "gp_evidence": rows,
        "cumulative_anchors": {
            float(gamma): [] for gamma in branch_config["gammas"]
        },
        "cumulative_adaptive": {
            float(gamma): [] for gamma in branch_config["gammas"]
        },
        "numpy_rng_state": target["numpy_rng_state"],
        "torch_rng_state": target["torch_rng_state"],
        "torch_cpu_rng_state": target["torch_cpu_rng_state"],
        "torch_device_rng_state": target["torch_device_rng_state"],
    }
    provenance = {
        "status": "EXACT_SAVED_R1_OPTIMIZER_BRANCH_READY",
        "source": str(source),
        "pretrain_dir": str(pretrain_dir),
        "output": str(output),
        "source_artifact_sha256": source_hashes,
        "source_checkpoint_001_model_sha256": _state_sha256(
            checkpoint_one["model"]
        ),
        "replayed_reference_model_sha256": _state_sha256(reference["model"]),
        "target_model_sha256": _state_sha256(target["model"]),
        "historical_2500_model_bitwise_reproduced": True,
        "reference_optimizer_steps": reference_optimizer_steps,
        "target_optimizer_steps": target_optimizer_steps,
        "target_rounds": target_rounds,
        "trainable_trunk_layers": trainable_trunk_layers,
        "device": str(device_value),
        "archive": archive,
        "optimizer_diagnostics": {
            "reference_positive_loss": reference["update"].get(
                "positive_loss"
            ),
            "target_positive_loss": target_update.get("positive_loss"),
            "reference_update_seconds": reference["update_s"],
            "target_update_seconds": target["update_s"],
            "reference_model_displacement_from_pre2": reference_displacement,
            "target_model_displacement_from_pre2": target_displacement,
        },
        "optimizer_stream_contract": (
            "fresh PRE2 Adam and seed are reset identically; the target is one "
            "uninterrupted 5000-step mode-gamma-stratified replay call, so its "
            "first 2500 batches/CFM draws are the bitwise-verified historical "
            "r1 prefix"
        ),
        "scientific_delta": {
            "optimizer_steps_per_round": [
                reference_optimizer_steps, target_optimizer_steps,
            ],
            "target_rounds_metadata": [
                int(source_config["rounds"]), target_rounds,
            ],
            "acquisition_reused_without_rerun": True,
        },
        "resume_contract": (
            "checkpoint/model, Adam moments, optimizer schedule, global and "
            "dedicated torch RNG, NumPy RNG, exact r1 replay/GP evidence, and "
            "committed events are present for in-place r2->r7 continuation"
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.prepare-", dir=output.parent,
    ))
    try:
        for name in (
            "checkpoint_000.pt", "query_archive_round_001.pt",
            "events_round_001.pt",
        ):
            shutil.copy2(source / name, temporary / name)
        if optional_task_config.is_file():
            shutil.copy2(
                optional_task_config, temporary / "task_config_resolved.json",
            )
        torch.save(branch_checkpoint, temporary / "checkpoint_001.pt")
        torch.save(resume, temporary / "resume_state_latest.pt")
        (temporary / "resume_state.json").write_text(json.dumps({
            "status": "COMMITTED_ROUND_RESUME",
            "version": 1,
            "completed_round": 1,
            "next_round": 2,
            "optimizer_step": target_optimizer_steps,
            "flow_base_std_next": float(branch_config["flow_base_std"]),
            "resume_state": "resume_state_latest.pt",
        }, indent=2) + "\n")
        (temporary / "metrics.jsonl").write_text(
            json.dumps(target_row, allow_nan=False) + "\n"
        )
        _truncate_first_action(
            source / "first_action_stats.json",
            temporary / "first_action_stats.json",
        )
        _truncate_fa_log(
            source / "fa_alloc_log.json", temporary / "fa_alloc_log.json",
        )
        (temporary / "BRANCH_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n"
        )
        (temporary / "BRANCH_SCREENING_MANIFEST.json").write_text(
            json.dumps({
                "status": "SAVED_R1_OPTIMIZER_BRANCH_BOUNDARY",
                "kind": "saved r1 optimizer exposure screening boundary",
                "config": branch_config,
                "rounds": [target_row],
                "optimizer_branch_provenance": "BRANCH_PROVENANCE.json",
            }, indent=2, allow_nan=False) + "\n"
        )
        # The source may continue to append r2+ metrics while this branch is
        # prepared.  Only committed r1 inputs are immutable and guarded here;
        # mutable logs were filtered to their r0/r1 prefix above.
        for name in immutable_source:
            if _sha256(source / name) != source_hashes[name]:
                raise RuntimeError(f"source artifact changed during replay: {name}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-optimizer-steps", type=int, default=2500)
    parser.add_argument("--target-optimizer-steps", type=int, default=5000)
    parser.add_argument("--target-rounds", type=int, default=7)
    parser.add_argument("--trainable-trunk-layers", type=int, default=3)
    args = parser.parse_args()
    report = prepare_branch(
        source=args.source,
        pretrain_dir=args.pretrain_dir,
        output=args.output,
        device=args.device,
        reference_optimizer_steps=args.reference_optimizer_steps,
        target_optimizer_steps=args.target_optimizer_steps,
        target_rounds=args.target_rounds,
        trainable_trunk_layers=args.trainable_trunk_layers,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
