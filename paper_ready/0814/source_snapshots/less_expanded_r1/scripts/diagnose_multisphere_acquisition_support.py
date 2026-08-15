#!/usr/bin/env python3
"""Reverify all K proposals at failed paired-clutter NVP contexts.

The committed-success event format deliberately drops the large learned
visual grid.  This diagnostic regenerates that grid from the exact paired
scene and robot/governor state, checks the regenerated scene and compact
context against the event, and only then runs the unchanged full-H verifier on
all K saved candidates.  It is CPU-only and never samples or updates a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import sphere_scene_spec_from_config
from safe_mppi.lab_clutter_pre2_expansion import LabClutterPre2ExpansionTask
from safe_mppi.lab_visual_flow import LAB_HP100_SCHEMA


DENSE_EXECUTION_CONTRACT = {
    "context_schema": LAB_HP100_SCHEMA,
    "verifier_mode": "full_polytope",
    "verifier_solver": "analytic",
    "execution_clearance_exp_weight": 0.0,
    "execution_clearance_exp_temperature": 0.10,
    "execution_clearance_target_m": 0.20,
    "execution_taskspace_weight": None,
    "execution_taskspace_quadratic_weight": 250.0,
    "execution_taskspace_quadratic_target_m": 0.15,
    "execution_axis_cylinder_quadratic_weight": 5.0,
    "execution_axis_cylinder_radius_m": 1.10,
    "execution_control_weight": 0.05,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_arrays(rows: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = np.ascontiguousarray(row)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_paired_member(value: str | int) -> int:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "0": 0,
        "original": 0,
        "source": 0,
        "1": 1,
        "axis180": 1,
        "axis_180": 1,
        "rotated": 1,
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "paired member must be 0/original or 1/axis_180"
        )
    return aliases[normalized]


def build_dense_task(task_config: Path, paired_seed: int):
    config = load_config(task_config)
    scene_spec = sphere_scene_spec_from_config(config)
    task = LabClutterPre2ExpansionTask(
        config,
        context_schema=LAB_HP100_SCHEMA,
        device="cpu",
        execution_z_bias_mode="none",
        tight_corridor=False,
        verifier_mode=DENSE_EXECUTION_CONTRACT["verifier_mode"],
        verifier_solver=DENSE_EXECUTION_CONTRACT["verifier_solver"],
        execution_clearance_exp_weight=DENSE_EXECUTION_CONTRACT[
            "execution_clearance_exp_weight"
        ],
        execution_clearance_exp_temperature=DENSE_EXECUTION_CONTRACT[
            "execution_clearance_exp_temperature"
        ],
        execution_clearance_target_m=DENSE_EXECUTION_CONTRACT[
            "execution_clearance_target_m"
        ],
        execution_taskspace_weight=DENSE_EXECUTION_CONTRACT[
            "execution_taskspace_weight"
        ],
        execution_taskspace_quadratic_weight=DENSE_EXECUTION_CONTRACT[
            "execution_taskspace_quadratic_weight"
        ],
        execution_taskspace_quadratic_target_m=DENSE_EXECUTION_CONTRACT[
            "execution_taskspace_quadratic_target_m"
        ],
        execution_axis_cylinder_quadratic_weight=DENSE_EXECUTION_CONTRACT[
            "execution_axis_cylinder_quadratic_weight"
        ],
        execution_axis_cylinder_radius_m=DENSE_EXECUTION_CONTRACT[
            "execution_axis_cylinder_radius_m"
        ],
        execution_control_weight=DENSE_EXECUTION_CONTRACT[
            "execution_control_weight"
        ],
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=int(paired_seed),
        fixed_scene_layout="none",
        scene_spec=scene_spec,
    )
    return config, task


def _member_name(member: int) -> str:
    return "original" if int(member) == 0 else "axis_180"


def exact_member_state(task, round_i: int, gamma: float, member: int) -> dict:
    task.begin_expansion_round(int(round_i), clear_scene_ledger=True)
    # Paired geometry is a deterministic function of (paired seed, round,
    # gamma).  Episode parity selects the requested member; the base reset seed
    # is metadata only for this task.
    state = task.reset(float(gamma), int(member), seed=0)
    if int(state["paired_scene_member"]) != int(member):
        raise RuntimeError("paired task returned the wrong scene member")
    return state


def validate_event_scene(
    event: dict,
    state: dict,
    *,
    round_i: int,
    gamma: float,
    member: int,
) -> None:
    expected_id = f"r{int(round_i):03d}_g{float(gamma):.9g}"
    checks = {
        "round": (int(event.get("round", -1)), int(round_i)),
        "gamma": (float(event.get("gamma", np.nan)), float(gamma)),
        "paired_scene_member": (
            int(event.get("paired_scene_member", -1)), int(member),
        ),
        "paired_scene_member_name": (
            event.get("paired_scene_member_name"), _member_name(member),
        ),
        "paired_scene_id": (event.get("paired_scene_id"), expected_id),
        "scene_hash": (event.get("scene_hash"), state["scene_hash"]),
    }
    mismatches = []
    for name, (observed, expected) in checks.items():
        equal = (
            bool(np.isclose(observed, expected, rtol=0.0, atol=1.0e-9))
            if name == "gamma" else observed == expected
        )
        if not equal:
            mismatches.append(f"{name}={observed!r}, expected {expected!r}")
    if int(event.get("episode", -1)) % 2 != int(member):
        mismatches.append(
            f"episode parity={int(event.get('episode', -1)) % 2}, "
            f"expected {member}"
        )
    if mismatches:
        raise RuntimeError("paired scene mismatch: " + "; ".join(mismatches))


def rebuild_full_verifier_context(
    task,
    event: dict,
    exact_state: dict,
    gamma: float,
) -> torch.Tensor:
    if event.get("context_compacted") is not True:
        raise RuntimeError("event context is not marked compacted")
    compact = np.asarray(event["context"], np.float32).reshape(-1)
    expected_compact = 7 + int(task.verifier_suffix_dim)
    if compact.shape != (expected_compact,):
        raise RuntimeError(
            f"compact context shape {compact.shape} != ({expected_compact},)"
        )
    robot = np.asarray(event["robot"], np.float32).reshape(-1)
    if robot.shape != (6,):
        raise RuntimeError(f"event robot shape {robot.shape} != (6,)")
    state = dict(exact_state)
    state.update({
        "x": robot.copy(),
        "previous_applied": compact[7:10].copy(),
        "previous_raw": compact[10:13].copy(),
        "steps": int(event.get("step", 0)),
        "collided": False,
        "oob": False,
    })
    full = task.context(state, float(gamma)).detach().cpu()
    values = full.numpy()
    if not np.allclose(values[:7], compact[:7], rtol=0.0, atol=2.0e-6):
        delta = float(np.max(np.abs(values[:7] - compact[:7])))
        raise RuntimeError(
            f"reconstructed learned state/gamma prefix differs by {delta:.3g}"
        )
    if not np.array_equal(values[-task.verifier_suffix_dim:], compact[7:]):
        delta = float(np.max(np.abs(
            values[-task.verifier_suffix_dim:] - compact[7:]
        )))
        raise RuntimeError(
            f"reconstructed verifier suffix differs by {delta:.3g}"
        )
    decoded_spheres = task.scene_from_context(full)
    if not np.array_equal(
        np.asarray(decoded_spheres, np.float32),
        np.asarray(exact_state["spheres"], np.float32),
    ):
        raise RuntimeError("reconstructed full context contains another scene")
    return full


def _initial_sigma_ranks(sigma: np.ndarray) -> np.ndarray:
    values = np.asarray(sigma, np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise RuntimeError("saved initial sigma contains a non-finite value")
    # Rank 1 is the largest saved pre-fantasy marginal sigma.  Candidate index
    # is an explicit deterministic tiebreaker; the actual sequential GP batch
    # can differ because later draws condition on earlier selected points.
    order = np.lexsort((np.arange(len(values)), -values))
    ranks = np.empty(len(values), np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _histogram(indices: list[int], ranks: np.ndarray, K: int) -> list[int]:
    output = np.zeros(K, np.int64)
    for index in indices:
        output[int(ranks[int(index)]) - 1] += 1
    return output.tolist()


def run_diagnostic(
    *,
    failed_events: Path,
    task_config: Path,
    round_i: int,
    gamma: float,
    paired_member: int,
    paired_seed: int,
) -> dict[str, Any]:
    failed_events = failed_events.resolve()
    task_config = task_config.resolve()
    if not failed_events.is_file():
        raise FileNotFoundError(f"missing failed event artifact: {failed_events}")
    if not task_config.is_file():
        raise FileNotFoundError(f"missing task config: {task_config}")
    config, task = build_dense_task(task_config, paired_seed)
    exact_state = exact_member_state(
        task, round_i, gamma, paired_member,
    )
    events = torch.load(
        failed_events, map_location="cpu", weights_only=False,
    )
    if not isinstance(events, list):
        raise TypeError("failed-events artifact must contain a list")
    matching = [
        event for event in events
        if int(event.get("round", -1)) == int(round_i)
        and np.isclose(
            float(event.get("gamma", np.nan)), float(gamma),
            rtol=0.0, atol=1.0e-9,
        )
        and int(event.get("paired_scene_member", -1)) == int(paired_member)
    ]
    if not matching:
        raise RuntimeError("no events match the requested round/gamma/member")
    for event in matching:
        validate_event_scene(
            event, exact_state, round_i=round_i, gamma=gamma,
            member=paired_member,
        )
    terminal = [
        event for event in matching
        if event.get("status") == "NVP"
    ]
    if not terminal:
        raise RuntimeError("matching events contain no terminal NVP context")
    terminal.sort(key=lambda event: (
        int(event["retry_batch"]), int(event["episode"]),
        int(event["replica"]), int(event["step"]),
    ))

    K_values = {int(np.asarray(event["candidates"]).shape[0]) for event in terminal}
    if len(K_values) != 1:
        raise RuntimeError(f"terminal contexts disagree on K: {K_values}")
    K = K_values.pop()
    if K < 1:
        raise RuntimeError("K must be positive")
    rank_histograms = {
        name: np.zeros(K, np.int64)
        for name in (
            "selected_all", "unselected_all", "selected_green",
            "unselected_green", "selected_execution_eligible",
            "unselected_execution_eligible",
        )
    }
    totals = {
        "selected_candidates": 0,
        "unselected_candidates": 0,
        "selected_green": 0,
        "unselected_green": 0,
        "selected_execution_eligible": 0,
        "unselected_execution_eligible": 0,
        "contexts_with_unselected_green": 0,
        "contexts_with_unselected_execution_eligible": 0,
        "contexts_rescued_by_verify_all_K": 0,
        "saved_selected_validity_mismatches": 0,
    }
    compact_contexts: list[np.ndarray] = []
    full_contexts: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    verdict_rows: list[np.ndarray] = []
    per_context = []
    event_identities = []
    for event in terminal:
        full_context = rebuild_full_verifier_context(
            task, event, exact_state, gamma,
        )
        candidates = np.asarray(event["candidates"], np.float32)
        if candidates.shape != (K, 10, 3):
            raise RuntimeError(
                f"candidate shape {candidates.shape} != ({K}, 10, 3)"
            )
        selected = [int(index) for index in event["selected"]]
        if len(selected) != len(set(selected)) or any(
            index < 0 or index >= K for index in selected
        ):
            raise RuntimeError("saved selected indices are invalid")
        unselected = [index for index in range(K) if index not in set(selected)]
        results = task.verify(
            full_context, torch.from_numpy(candidates), float(gamma),
        )
        if len(results) != K:
            raise RuntimeError("full-K verifier returned the wrong result count")
        saved_results = event.get("verification", [])
        if len(saved_results) != len(selected):
            raise RuntimeError("saved B verification count disagrees with selection")
        saved_mismatches = sum(
            bool(saved_results[local]["valid"]) != bool(results[index].valid)
            for local, index in enumerate(selected)
        )
        totals["saved_selected_validity_mismatches"] += saved_mismatches
        if saved_mismatches:
            raise RuntimeError(
                "reverified selected GREEN labels differ from the saved verifier"
            )

        green = [index for index, result in enumerate(results) if result.valid]
        execution_eligible = [
            index for index, result in enumerate(results)
            if result.valid and result.progress_eligible
            and (
                not bool(event.get("target_gate_active", False))
                or result.target_eligible
            )
        ]
        selected_green = sorted(set(selected).intersection(green))
        unselected_green = sorted(set(unselected).intersection(green))
        selected_eligible = sorted(
            set(selected).intersection(execution_eligible)
        )
        unselected_eligible = sorted(
            set(unselected).intersection(execution_eligible)
        )
        if selected_eligible:
            raise RuntimeError(
                "saved terminal NVP context has a reverified selected "
                "execution-eligible candidate"
            )
        sigma = np.asarray(event["sigma_K"], np.float64).reshape(-1)
        if sigma.shape != (K,):
            raise RuntimeError(f"saved sigma shape {sigma.shape} != ({K},)")
        ranks = _initial_sigma_ranks(sigma)
        groups = {
            "selected_all": selected,
            "unselected_all": unselected,
            "selected_green": selected_green,
            "unselected_green": unselected_green,
            "selected_execution_eligible": selected_eligible,
            "unselected_execution_eligible": unselected_eligible,
        }
        for name, indices in groups.items():
            rank_histograms[name] += np.asarray(
                _histogram(indices, ranks, K), np.int64,
            )
        totals["selected_candidates"] += len(selected)
        totals["unselected_candidates"] += len(unselected)
        totals["selected_green"] += len(selected_green)
        totals["unselected_green"] += len(unselected_green)
        totals["selected_execution_eligible"] += len(selected_eligible)
        totals["unselected_execution_eligible"] += len(unselected_eligible)
        totals["contexts_with_unselected_green"] += int(bool(unselected_green))
        totals["contexts_with_unselected_execution_eligible"] += int(
            bool(unselected_eligible)
        )
        totals["contexts_rescued_by_verify_all_K"] += int(
            not selected_eligible and bool(unselected_eligible)
        )
        identity = {
            "context_id": int(event["context_id"]),
            "episode": int(event["episode"]),
            "retry_batch": int(event["retry_batch"]),
            "replica": int(event["replica"]),
            "step": int(event["step"]),
        }
        event_identities.append(identity)
        per_context.append({
            **identity,
            "nvp_reason": event.get("nvp_reason"),
            "selected_indices": selected,
            "selected_green_indices": selected_green,
            "unselected_green_indices": unselected_green,
            "unselected_execution_eligible_indices": unselected_eligible,
            "unselected_green_initial_sigma_ranks": [
                int(ranks[index]) for index in unselected_green
            ],
            "unselected_execution_eligible_initial_sigma_ranks": [
                int(ranks[index]) for index in unselected_eligible
            ],
        })
        compact_contexts.append(np.asarray(event["context"], np.float32))
        full_contexts.append(full_context.numpy().astype(np.float32))
        candidate_rows.append(candidates)
        sigma_rows.append(sigma.astype(np.float32))
        selected_rows.append(np.asarray(selected, np.int64))
        verdict_rows.append(np.asarray([
            [
                int(result.valid), int(result.progress_eligible),
                int(result.target_eligible), int(result.hp_eligible),
            ]
            for result in results
        ], np.int8))

    nvp_reasons: dict[str, int] = {}
    for event in terminal:
        reason = str(event.get("nvp_reason"))
        nvp_reasons[reason] = nvp_reasons.get(reason, 0) + 1
    member_metadata = {
        key: exact_state[key]
        for key in (
            "paired_scene_id", "paired_scene_seed",
            "paired_scene_proposal_index", "paired_source_scene_hash",
            "paired_rotated_scene_hash", "paired_rotation",
            "paired_scene_member", "paired_scene_member_name", "scene_hash",
        )
    }
    return {
        "kind": "paired multi-sphere acquisition-support diagnostic",
        "contract_version": 1,
        "scope": {
            "round": int(round_i),
            "gamma": float(gamma),
            "paired_member": int(paired_member),
            "paired_member_name": _member_name(paired_member),
            "paired_seed": int(paired_seed),
            "matching_event_count": len(matching),
            "terminal_nvp_context_count": len(terminal),
            "nvp_reason_counts": nvp_reasons,
            "K": K,
            "B": len(terminal[0]["selected"]),
        },
        "scene": {
            **member_metadata,
            "spheres": np.asarray(exact_state["spheres"], float).tolist(),
            "scene_schema": task.scene_schema,
        },
        "task": {
            "task_config": str(task_config),
            "taskspace_bounds": np.asarray(task.env.bounds, float).tolist(),
            "execution_contract": DENSE_EXECUTION_CONTRACT,
            "configured_taskspace_exponential_weight": float(
                config.safemppi.taskspace_exponential_weight
            ),
        },
        "counts": totals,
        "initial_sigma_rank_histogram": {
            "rank_convention": (
                "1 is highest saved pre-fantasy marginal sigma_K; stable "
                "candidate-index tiebreak; bins enumerate ranks 1..K"
            ),
            **{
                name: values.tolist()
                for name, values in rank_histograms.items()
            },
        },
        "interpretation": {
            "verify_all_K_would_rescue_context_fraction": (
                totals["contexts_rescued_by_verify_all_K"] / len(terminal)
            ),
            "unselected_green_fraction": (
                totals["unselected_green"]
                / max(1, totals["unselected_candidates"])
            ),
            "unselected_execution_eligible_fraction": (
                totals["unselected_execution_eligible"]
                / max(1, totals["unselected_candidates"])
            ),
            "diagnosis": (
                "acquisition_excludes_execution_eligible_support"
                if totals["contexts_rescued_by_verify_all_K"] > 0
                else "no_execution_eligible_support_outside_selected_B"
            ),
        },
        "hashes": {
            "failed_events_sha256": _sha256_file(failed_events),
            "task_config_sha256": _sha256_file(task_config),
            "diagnostic_script_sha256": _sha256_file(Path(__file__)),
            "event_identity_sha256": _sha256_json(event_identities),
            "sphere_rows_sha256": _sha256_arrays([
                np.asarray(exact_state["spheres"], np.float32)
            ]),
            "compact_contexts_sha256": _sha256_arrays(compact_contexts),
            "reconstructed_full_contexts_sha256": _sha256_arrays(
                full_contexts
            ),
            "candidate_tensors_sha256": _sha256_arrays(candidate_rows),
            "initial_sigma_tensors_sha256": _sha256_arrays(sigma_rows),
            "selected_indices_sha256": _sha256_arrays(selected_rows),
            "full_K_verdicts_sha256": _sha256_arrays(verdict_rows),
        },
        "per_context": per_context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-events", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--paired-member", type=parse_paired_member, required=True,
    )
    parser.add_argument("--paired-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round must be positive")
    if not np.isfinite(args.gamma) or args.gamma <= 0.0:
        parser.error("--gamma must be finite and positive")
    result = run_diagnostic(
        failed_events=args.failed_events,
        task_config=args.task_config,
        round_i=args.round,
        gamma=args.gamma,
        paired_member=args.paired_member,
        paired_seed=args.paired_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    command = " ".join(shlex.quote(token) for token in sys.argv)
    print(
        f"[acquisition-support] wrote {args.output} | "
        f"NVP={result['scope']['terminal_nvp_context_count']} "
        f"selected GREEN={result['counts']['selected_green']} "
        f"unselected GREEN={result['counts']['unselected_green']} "
        f"verify-all rescue={result['counts']['contexts_rescued_by_verify_all_K']}"
    )
    print(f"[reproduce] {command}")


if __name__ == "__main__":
    main()
