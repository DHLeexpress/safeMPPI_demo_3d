"""Reconstruct the trunk-3 r10 raw/OOB and committed-replay story.

The diagnostic is deliberately read-only.  It reruns the fixed raw seed bank
with trajectory retention, reconstructs committed successful trajectories from
the per-round query archives, and summarizes whether GP/ranked-B or retry-fast-K
collection could have introduced a route-count bias.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.expansion import _sliding_success_gp_rows
from safe_mppi.lab_flow_evaluation import _checkpoint_policy
from safe_mppi.lab_flow_expansion import LabFlowExpansionTask
from safe_mppi.lab_reference_flow_task import raw_reference_rollout


MODE_NAMES = {0: "below", 1: "above", 2: "left", 3: "right"}
FACE_NAMES = (
    "x-min", "x-max", "y-min", "y-max", "z-min", "z-max",
)


def _downsample(path: np.ndarray, maximum: int) -> list[list[float]]:
    path = np.asarray(path, np.float64).reshape(-1, 3)
    if len(path) > maximum:
        indices = np.unique(np.linspace(0, len(path) - 1, maximum).round().astype(int))
        path = path[indices]
    return np.round(path, 4).tolist()


def _signed_face_margins(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 3)
    bounds = np.asarray(bounds, np.float64)
    return np.column_stack([
        points[:, 0] - bounds[0, 0], bounds[0, 1] - points[:, 0],
        points[:, 1] - bounds[1, 0], bounds[1, 1] - points[:, 1],
        points[:, 2] - bounds[2, 0], bounds[2, 1] - points[:, 2],
    ])


def _boundary_summary(points: np.ndarray, bounds: np.ndarray) -> dict:
    margins = _signed_face_margins(points, bounds)
    flat_index = int(np.argmin(margins))
    point_index, face_index = np.unravel_index(flat_index, margins.shape)
    return {
        "minimum_margin_m": float(margins[point_index, face_index]),
        "nearest_face": FACE_NAMES[int(face_index)],
        "nearest_point_index": int(point_index),
        "nearest_point": np.round(points[point_index], 5).tolist(),
    }


def _summarize_raw_results(results, config, *, round_i: int) -> tuple[list[dict], dict]:
    env = TaskEnvironment(config)
    rows: list[dict] = []
    for gamma, episode, result in results:
            states = np.asarray(result["states"], np.float64)
            dense = np.asarray(result["dense_steps"], np.float64).reshape(-1, 3)
            dense_path = np.concatenate([states[:1, :3], dense], axis=0)
            boundary = _boundary_summary(dense_path, env.bounds)
            row = {
                "kind": "raw",
                "round": int(round_i),
                "gamma": float(gamma),
                "episode": int(episode),
                "status": str(result["status"]),
                "mode": str(result["mode"]),
                "steps": int(len(result["controls"])),
                "min_clearance_m": result["min_clearance_m"],
                "window_validity": float(result["window_validity"]),
                "path": _downsample(dense_path, 48),
                **boundary,
            }
            if result["status"] == "OOB":
                margins = _signed_face_margins(dense_path, env.bounds)
                outside = np.flatnonzero(np.any(margins < 0.0, axis=1))
                first = int(outside[0])
                face = int(np.argmin(margins[first]))
                state_index = min(max(first // config.safemppi.integration_substeps, 0),
                                  len(states) - 1)
                row["oob"] = {
                    "face": FACE_NAMES[face],
                    "first_dense_index": first,
                    "point": np.round(dense_path[first], 5).tolist(),
                    "previous_point": np.round(dense_path[max(0, first - 1)], 5).tolist(),
                    "velocity": np.round(states[state_index, 3:6], 5).tolist(),
                }
            rows.append(row)

    status_counts = Counter(row["status"] for row in rows)
    oob_faces = Counter(
        row["oob"]["face"] for row in rows if row["status"] == "OOB"
    )
    success_modes = Counter(
        row["mode"] for row in rows if row["status"] == "SUCCESS"
    )
    summary = {
        "episodes": len(rows),
        "status_counts": dict(status_counts),
        "rates": {
            key: float(status_counts.get(key, 0) / len(rows))
            for key in ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
        },
        "success_modes": dict(success_modes),
        "oob_faces": dict(oob_faces),
        "oob_min_margin_quantiles_m": _quantiles([
            row["minimum_margin_m"] for row in rows if row["status"] == "OOB"
        ]),
    }
    return rows, summary


def _raw_diagnostic(
    policy, config, *, gammas, episodes: int, seed: int, device: str,
    round_i: int,
) -> tuple[list[dict], dict]:
    results = []
    for gamma in gammas:
        for episode in range(episodes):
            results.append((float(gamma), int(episode), raw_reference_rollout(
                policy,
                config,
                float(gamma),
                int(seed) + 37 * episode,
                device=device,
                sampling_temperature=1.0,
            )))
    return _summarize_raw_results(
        results,
        config,
        round_i=round_i,
    )


def _saved_raw_diagnostic(path: Path, config, round_i: int):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    source_rows = saved.get(round_i, saved.get(str(round_i)))
    if source_rows is None:
        raise KeyError(f"round {round_i} missing from {path}")
    return _summarize_raw_results(
        (
            (float(row["gamma"]), int(row["episode"]), row)
            for row in source_rows
        ),
        config,
        round_i=round_i,
    )


def _quantiles(values) -> dict:
    values = np.asarray(list(values), np.float64)
    if not len(values):
        return {}
    return {
        label: float(value)
        for label, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0)),
        )
    }


def _training_trajectories(
    arm: Path, config, *, through_round: int,
) -> tuple[list[dict], list[dict]]:
    task = LabFlowExpansionTask(
        config,
        context_schema="lab_hp100_history3",
        device="cpu",
        tight_corridor=True,
    )
    env = task.env
    trajectories: list[dict] = []
    per_round: list[dict] = []
    for round_i in range(1, through_round + 1):
        archive_path = arm / f"query_archive_round_{round_i:03d}.pt"
        records = torch.load(archive_path, map_location="cpu", weights_only=False)
        grouped = defaultdict(list)
        for record in records:
            grouped[str(record.trajectory_id)].append(record)
        stratum_rows = Counter()
        stratum_trajectories = Counter()
        near_wall = Counter()
        round_mode_margin = defaultdict(list)
        for trajectory_id, group in sorted(grouped.items()):
            group.sort(key=lambda record: int(record.window_start))
            first = group[0]
            mode_i = int(first.sample_update_mode)
            mode = MODE_NAMES[mode_i]
            gamma = float(first.gamma)
            positions = []
            for record in group:
                state6, _, _ = task._decode_context(record.context)
                positions.append(state6[:3])
            positions = np.asarray(positions, np.float64)
            boundary = _boundary_summary(positions, env.bounds)
            obstacle_clearance = env.obstacle_clearance(positions)
            key = f"g{gamma:g}:{mode}"
            stratum_rows[key] += len(group)
            stratum_trajectories[key] += 1
            round_mode_margin[mode].append(boundary["minimum_margin_m"])
            if boundary["minimum_margin_m"] < 0.10:
                near_wall[mode] += 1
            trajectories.append({
                "kind": "training",
                "round": round_i,
                "gamma": gamma,
                "mode": mode,
                "trajectory_id": trajectory_id,
                "windows": len(group),
                "min_clearance_m": float(np.min(obstacle_clearance)),
                "path": _downsample(positions, 32),
                **boundary,
            })
        per_round.append({
            "round": round_i,
            "trajectory_count": len(grouped),
            "row_count": len(records),
            "stratum_trajectory_counts": dict(sorted(stratum_trajectories.items())),
            "stratum_row_counts": dict(sorted(stratum_rows.items())),
            "near_wall_trajectory_counts_lt_0.10m": dict(near_wall),
            "mode_min_boundary_margin_median": {
                mode: float(np.median(values))
                for mode, values in round_mode_margin.items()
            },
        })
    return trajectories, per_round


def _active_gp_mode_counts(
    arm: Path, manifest: dict, through_round: int,
) -> dict[int, dict]:
    state = torch.load(
        arm / "resume_state_latest.pt", map_location="cpu", weights_only=False,
    )
    evidence = state["gp_evidence"]
    config = manifest["config"]
    if config["gp_reference_mode"] != "sliding_success_per_gamma_current_phi":
        raise ValueError("diagnostic expects sliding_success_per_gamma_current_phi")
    result = {}
    for round_i in range(1, through_round + 1):
        active = (
            [] if round_i == 1 else _sliding_success_gp_rows(
                evidence,
                config["gammas"],
                int(config["gp_buffer_cap"]),
                through_round=round_i - 1,
                selector=config["gp_sliding_row_selector"],
            )
        )
        counts = Counter(
            (f"{float(row.gamma):g}", MODE_NAMES[int(row.sample_update_mode)])
            for row in active
        )
        trajectory_counts = Counter(
            (f"{float(row.gamma):g}", MODE_NAMES[int(row.sample_update_mode)])
            for row in {
                str(record.trajectory_id): record for record in active
            }.values()
        )
        result[round_i] = {
            "row_counts_by_mode_gamma": {
                f"g{gamma}:{mode}": int(counts[(gamma, mode)])
                for gamma in map(lambda value: f"{float(value):g}", config["gammas"])
                for mode in MODE_NAMES.values()
            },
            "trajectory_counts_by_mode_gamma": {
                f"g{gamma}:{mode}": int(trajectory_counts[(gamma, mode)])
                for gamma in map(lambda value: f"{float(value):g}", config["gammas"])
                for mode in MODE_NAMES.values()
            },
        }
    return result


def _collection_and_gp_summary(
    arm: Path, manifest: dict, through_round: int,
) -> list[dict]:
    active_counts = _active_gp_mode_counts(arm, manifest, through_round)
    rows = []
    for round_row in manifest["rounds"][:through_round]:
        terminal = Counter()
        committed = Counter()
        retry_batches = {}
        attempts = {}
        for gamma, cell in round_row["successful_executed_commit_by_gamma"].items():
            terminal.update(
                MODE_NAMES[int(mode)]
                for mode in cell["success_episode_sample_update_modes"].values()
            )
            committed.update(
                MODE_NAMES[int(mode)] for mode in cell["committed_sample_update_modes"]
            )
            retry_batches[gamma] = int(cell["retry_batches_used"])
            attempts[gamma] = int(cell["attempted_episode_count"])
        rows.append({
            "round": int(round_row["round"]),
            "gp_buffer_by_gamma": round_row["gp_buffer_by_gamma"],
            "gp_active_support": active_counts[int(round_row["round"])],
            "retry_fast_path_contexts": int(
                round_row["retry_verify_all_fast_path_contexts"]
            ),
            "retry_batches_by_gamma": retry_batches,
            "attempted_episodes_by_gamma": attempts,
            "all_terminal_success_modes": dict(terminal),
            "committed_modes": dict(committed),
            "available_trajectories_by_mode_gamma": (
                round_row["available_trajectories_by_mode_gamma"]
            ),
            "sampled_rows_by_mode_gamma": round_row["sampled_rows_by_mode_gamma"],
            "positive_loss": float(round_row["positive_loss"]),
            "optimizer_step": int(round_row["optimizer_step"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--raw-trajectory-file",
        type=Path,
        help=(
            "use exact retained fixed-bank trajectories instead of rerunning "
            "the raw policy on the local device"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.arm / "task_config_resolved.json")
    if args.raw_trajectory_file is not None:
        raw_rows, raw_summary = _saved_raw_diagnostic(
            args.raw_trajectory_file, config, args.round,
        )
    else:
        policy = _checkpoint_policy(
            args.pretrain_dir, args.arm, args.round, device=args.device,
        )
        policy.flow.nfe = int(args.flow_nfe)
        raw_rows, raw_summary = _raw_diagnostic(
            policy,
            config,
            gammas=config.data.gammas,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            round_i=args.round,
        )
    training_rows, training_rounds = _training_trajectories(
        args.arm, config, through_round=args.round,
    )
    manifest = json.loads((args.arm / "manifest.json").read_text())
    gp_rows = _collection_and_gp_summary(args.arm, manifest, args.round)

    bounds = TaskEnvironment(config).bounds
    payload = {
        "status": "TRUNK3_R10_OOB_DIAGNOSIS_COMPLETE",
        "source": {
            "arm": str(args.arm.resolve()),
            "checkpoint": str((args.arm / f"checkpoint_{args.round:03d}.pt").resolve()),
            "round": args.round,
            "fixed_seed": args.seed,
            "episodes_per_gamma": args.episodes,
            "flow_nfe": args.flow_nfe,
            "sampling_temperature": 1.0,
        },
        "taskspace": {
            "bounds": np.round(bounds, 5).tolist(),
            "z_bounds_m": [float(bounds[2, 0]), float(bounds[2, 1])],
            "verifier_gate": (
                "only the dense first executed segment is taskspace-gated; "
                "the unexecuted H-1 tail is not taskspace-gated"
            ),
            "native_execution_cost": (
                "the configured exponential term is zero for every in-bounds "
                "predicted knot and activates only after a knot leaves the box"
            ),
        },
        "raw_summary": raw_summary,
        "raw_trajectories": raw_rows,
        "training_summary": {
            "trajectory_count": len(training_rows),
            "expected_count": args.round * len(config.data.gammas) * 12,
            "mode_counts": dict(Counter(row["mode"] for row in training_rows)),
            "gamma_counts": dict(Counter(f"{row['gamma']:g}" for row in training_rows)),
            "near_wall_lt_0.10m": int(sum(
                row["minimum_margin_m"] < 0.10 for row in training_rows
            )),
            "minimum_boundary_margin_quantiles_m": _quantiles(
                row["minimum_margin_m"] for row in training_rows
            ),
        },
        "training_rounds": training_rounds,
        "training_trajectories": training_rows,
        "collection_and_gp": gp_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "raw_summary": raw_summary,
        "training_summary": payload["training_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
