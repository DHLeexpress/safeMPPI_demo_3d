#!/usr/bin/env python3
"""Build compact data for the six-option PRE2 multi-sphere visual."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


GAMMAS = (0.1, 0.3, 0.5, 1.0)
ROUNDS = (1, 2, 3, 4, 5)
COORDINATE_SCALE_M = 0.001
BOWLING_SUMMARY_KEYS = (
    "attempts",
    "terminal_successes",
    "stable_classified_successes",
    "ambiguous_successes",
    "route_counts",
    "observed_routes",
    "coverage_count",
    "coverage_fraction",
    "status_counts",
    "vertical_dominant_decision_stages",
    "decision_stages",
    "vertical_dominant_decision_fraction",
    "vertical_sign_counts",
    "mean_abs_decision_vertical_m",
    "max_abs_decision_vertical_m",
)
ROUND_STAT_KEYS = (
    "attempted_episode_count",
    "success",
    "NVP",
    "timeout",
    "retry_batches_by_gamma",
    "round_success_commit_complete",
    "optimizer_step",
    "learning_rate",
    "positive_loss",
    "ESS_over_K",
    "marginal_ESS_over_K",
    "uncertainty_uplift",
    "gp_buffer",
    "gather_s",
    "update_s",
    "round_total_s",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _gamma_key(value: float) -> str:
    return f"{float(value):g}"


def _quantize(values: Any) -> list[int]:
    array = np.asarray(values, np.float64)
    if not bool(np.isfinite(array).all()):
        raise ValueError("geometry must be finite")
    return np.rint(array / COORDINATE_SCALE_M).astype(np.int32).reshape(-1).tolist()


def _downsample_path(values: Any, max_points: int) -> tuple[int, list[int]]:
    points = np.asarray(values, np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 2:
        raise ValueError("trajectory path must have shape [T,>=3] with T>=2")
    points = points[:, :3]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points)
        points = points[np.unique(np.rint(indices).astype(int))]
    quantized = np.rint(points / COORDINATE_SCALE_M).astype(np.int32)
    quantized[1:] -= quantized[:-1].copy()
    return len(points), quantized.reshape(-1).tolist()


def _task_geometry(config: dict[str, Any]) -> dict[str, Any]:
    task = config["taskspace"]
    origin = np.asarray(task["origin"], np.float64)
    upper = origin + np.asarray(task["size"], np.float64)
    randomization = config["scene_randomization"]
    return {
        "start_q": _quantize(task["start"][:3]),
        "goal_q": _quantize(task["goal"]),
        "bounds_q": _quantize(np.column_stack([origin, upper])),
        "physical_radius_q": _quantize([randomization["physical_radius_m"]])[0],
        "effective_radius_q": _quantize([randomization["radius_m"]])[0],
        "dt_s": float(config["safemppi"]["dt"]),
    }


def _scene_map(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    scenes: dict[str, np.ndarray] = {}
    for row in manifest.get("lab_scene_ledger", []):
        scene_hash = str(row["scene_hash"])
        spheres = np.asarray(row["spheres"], np.float64)
        if spheres.shape != (6, 4):
            raise ValueError(f"scene {scene_hash} does not contain six spheres")
        previous = scenes.get(scene_hash)
        if previous is not None and not np.array_equal(previous, spheres):
            raise ValueError(f"scene hash {scene_hash} has conflicting geometry")
        scenes[scene_hash] = spheres
    if not scenes:
        raise ValueError("manifest has no scene ledger")
    return scenes


def _committed_keys(
    manifest: dict[str, Any],
) -> tuple[set[tuple[int, float, int]], dict[int, dict[str, Any]]]:
    if manifest.get("status") != "SAFE_FLOW_EXPANSION_COMPLETE":
        raise ValueError("expansion manifest is not complete")
    round_rows = {
        int(row["round"]): row for row in manifest.get("rounds", [])
    }
    if tuple(sorted(round_rows)) != ROUNDS:
        raise ValueError("expansion manifest must contain exactly rounds 1..5")
    keys: set[tuple[int, float, int]] = set()
    for round_index in ROUNDS:
        detail_by_gamma = round_rows[round_index].get(
            "successful_executed_commit_by_gamma", {}
        )
        for gamma in GAMMAS:
            detail = detail_by_gamma.get(_gamma_key(gamma))
            if not isinstance(detail, dict):
                raise ValueError(
                    f"round {round_index} lacks gamma={gamma:g} commit metadata"
                )
            episodes = [int(value) for value in detail["committed_episode_ids"]]
            if len(episodes) != 2 or len(set(episodes)) != 2:
                raise ValueError(
                    f"round {round_index} gamma={gamma:g} must commit two trajectories"
                )
            for episode in episodes:
                key = (round_index, gamma, episode)
                if key in keys:
                    raise ValueError(f"duplicate committed trajectory key {key}")
                keys.add(key)
    if len(keys) != 40:
        raise ValueError(f"expected 40 committed trajectories, found {len(keys)}")
    return keys, round_rows


def _event_sources(expansion: Path) -> list[Path]:
    paths = [expansion / f"events_round_{round_index:03d}.pt" for round_index in ROUNDS]
    if all(path.is_file() for path in paths):
        return paths
    combined = expansion / "events.pt"
    if combined.is_file():
        return [combined]
    missing = [path for path in paths if not path.is_file()]
    raise FileNotFoundError(
        "missing committed event logs: " + ", ".join(map(str, missing))
    )


def _load_committed_events(
    sources: list[Path],
    committed: set[tuple[int, float, int]],
) -> dict[tuple[int, float, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, float, int], list[dict[str, Any]]] = defaultdict(list)
    seen_steps: set[tuple[int, float, int, int]] = set()
    for path in sources:
        events = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(events, list):
            raise ValueError(f"event log is not a list: {path}")
        for event in events:
            key = (
                int(event["round"]),
                float(event["gamma"]),
                int(event["episode"]),
            )
            if key not in committed:
                continue
            step_key = (*key, int(event["step"]))
            if step_key in seen_steps:
                raise ValueError(f"duplicate event step {step_key}")
            seen_steps.add(step_key)
            grouped[key].append(event)
    missing = committed - set(grouped)
    if missing:
        raise ValueError(f"committed event traces are missing: {sorted(missing)}")
    return grouped


def _round_stats(round_rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for round_index in ROUNDS:
        source = round_rows[round_index]
        row = {key: source.get(key) for key in ROUND_STAT_KEYS}
        retries = source.get("retry_batches_by_gamma", {})
        row["extra_retry_batches_by_gamma"] = {
            str(key): max(0, int(value) - 1) for key, value in retries.items()
        }
        output[str(round_index)] = row
    return output


def _dense_option(
    manifest: dict[str, Any],
    event_sources: list[Path],
    max_path_points: int,
) -> dict[str, Any]:
    committed, round_rows = _committed_keys(manifest)
    scenes = _scene_map(manifest)
    grouped = _load_committed_events(event_sources, committed)
    trajectories = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in rows]
        if steps != list(range(len(rows))):
            raise ValueError(f"committed trace {key} has non-contiguous steps")
        statuses = [row.get("status") for row in rows if row.get("status") is not None]
        if statuses != ["SUCCESS"]:
            raise ValueError(f"committed trace {key} is not one terminal SUCCESS")
        scene_hashes = {str(row["scene_hash"]) for row in rows}
        if len(scene_hashes) != 1:
            raise ValueError(f"committed trace {key} changes scene")
        scene_hash = scene_hashes.pop()
        if scene_hash not in scenes:
            raise ValueError(f"scene {scene_hash} is absent from manifest ledger")
        path = np.asarray(
            [rows[0]["robot"][:3]] + [row["robot_after"][:3] for row in rows],
            np.float64,
        )
        point_count, path_q = _downsample_path(path, max_path_points)
        first = rows[0]
        trajectories.append({
            "id": f"r{key[0]}_g{_gamma_key(key[1])}_e{key[2]}",
            "round": key[0],
            "gamma": key[1],
            "episode": key[2],
            "status": "SUCCESS",
            "steps": len(rows),
            "path_points": point_count,
            "path_q": path_q,
            "z_min_q": _quantize([path[:, 2].min()])[0],
            "z_max_q": _quantize([path[:, 2].max()])[0],
            "flow_base_std": float(first["flow_base_std"]),
            "pair_id": first.get("paired_scene_id"),
            "pair_member": first.get("paired_scene_member_name"),
            "source_scene": {
                "hash": scene_hash,
                "sphere_count": 6,
                "spheres_xyzr_q": _quantize(scenes[scene_hash]),
            },
        })
    by_round = Counter(row["round"] for row in trajectories)
    by_gamma = Counter(_gamma_key(row["gamma"]) for row in trajectories)
    if by_round != Counter({round_index: 8 for round_index in ROUNDS}):
        raise ValueError(f"dense committed count by round is wrong: {by_round}")
    if by_gamma != Counter({_gamma_key(gamma): 10 for gamma in GAMMAS}):
        raise ValueError(f"dense committed count by gamma is wrong: {by_gamma}")
    return {
        "id": "dense_committed_r1_r5",
        "label": "Dense-z committed · rounds 1–5 · all 40",
        "kind": "dense_committed",
        "rounds": list(ROUNDS),
        "trajectory_count": len(trajectories),
        "status_counts": {"SUCCESS": len(trajectories)},
        "round_stats": _round_stats(round_rows),
        "trajectories": trajectories,
    }


def _slim_bowling_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: source.get(key) for key in BOWLING_SUMMARY_KEYS if key in source
    }


def _compact_route(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": str(route["code"]),
        "stable_code": str(route["stable_code"]),
        "vertical_sign": str(route["decision_vertical_sign"]),
        "decision_xyz_q": _quantize(route["decision_xyz_m"]),
        "minimum_margin_m": float(route["minimum_decision_margin_m"]),
        "vertical_dominant": [
            bool(value) for value in route["decision_vertical_dominant"]
        ],
    }


def _bowling_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(row["status"]) for row in rows)
    route_counts = Counter(
        str(row["bowling_route"]["stable_code"])
        for row in rows
        if row["status"] == "SUCCESS"
        and isinstance(row.get("bowling_route"), dict)
        and "X" not in str(row["bowling_route"]["stable_code"])
    )
    pooled_coverage = summary.get("pooled", {})
    recorded_counts = pooled_coverage.get("route_counts")
    if recorded_counts is not None:
        normalized = {str(key): int(value) for key, value in recorded_counts.items()}
        if normalized != {key: route_counts.get(key, 0) for key in normalized}:
            raise ValueError("bowling route counts disagree with raw rows")
    return {
        "pooled": _slim_bowling_summary(pooled_coverage),
        "per_gamma": {
            str(key): _slim_bowling_summary(value)
            for key, value in summary.get("per_gamma", {}).items()
        },
        "status_counts": dict(sorted(statuses.items())),
    }


def _bowling_options(
    evaluation: dict[str, Any],
    max_path_points: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evaluation.get("schema") != "bowling_123_fixed_bank_coverage_merge_v1":
        raise ValueError("bowling eval is not the audited merged coverage schema")
    contract = evaluation.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("merged bowling eval lacks contract")
    rows_by_round = evaluation.get("rows")
    summaries = evaluation.get("summary")
    if not isinstance(rows_by_round, dict) or not isinstance(summaries, dict):
        raise ValueError("bowling eval must contain object-valued rows and summary")
    missing = [str(value) for value in ROUNDS if str(value) not in rows_by_round]
    if missing:
        raise ValueError("bowling eval lacks rounds: " + ", ".join(missing))
    fixed_scene = contract.get("scene", {})
    spheres = np.asarray(fixed_scene.get("spheres"), np.float64)
    if spheres.shape != (6, 4):
        raise ValueError("bowling eval does not define one fixed six-sphere scene")
    scene = {
        "hash": str(fixed_scene["scene_hash"]),
        "sphere_count": 6,
        "spheres_xyzr_q": _quantize(spheres),
    }
    options = []
    declared_options = evaluation.get("visual_options")
    if not isinstance(declared_options, list) or [
        int(row.get("round", -1)) for row in declared_options
    ] != list(ROUNDS):
        raise ValueError("merged bowling visual options must be rounds 1..5")
    for round_index in ROUNDS:
        source_rows = rows_by_round[str(round_index)]
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"bowling round {round_index} has no rows")
        trajectories = []
        for index, row in enumerate(source_rows):
            status = str(row["status"])
            if str(row.get("scene_hash")) != scene["hash"]:
                raise ValueError(
                    f"bowling round {round_index} row changed fixed scene"
                )
            route = row.get("bowling_route")
            path = row.get("arc_length_resampled_path_xyz")
            if status == "SUCCESS" and (not isinstance(route, dict) or path is None):
                raise ValueError(
                    f"bowling round {round_index} SUCCESS row lacks route/path"
                )
            if status != "SUCCESS" and path is not None:
                raise ValueError(
                    f"bowling round {round_index} non-SUCCESS row has a path"
                )
            if path is None:
                path_points, path_q = 0, None
            else:
                path_points, path_q = _downsample_path(path, max_path_points)
            trajectories.append({
                "id": f"r{round_index}_g{_gamma_key(row['gamma'])}_e{row['episode']}_{index}",
                "gamma": float(row["gamma"]),
                "episode": int(row["episode"]),
                "rollout_seed": int(row["rollout_seed"]),
                "status": status,
                "path_points": path_points,
                "path_q": path_q,
                "route": _compact_route(route) if isinstance(route, dict) else None,
                "min_clearance_m": row.get("min_clearance_m"),
                "time_to_goal_s": row.get("time_to_goal_s"),
                "window_validity": row.get("window_validity"),
            })
        options.append({
            "id": f"bowling_r{round_index}",
            "label": f"Bowling 1+2+3 · round {round_index}",
            "kind": "bowling_fixed_eval",
            "rounds": [round_index],
            "trajectory_count": len(trajectories),
            "paths_available": sum(row["path_q"] is not None for row in trajectories),
            "scene": scene,
            "summary": _bowling_summary(summaries[str(round_index)], source_rows),
            "trajectories": trajectories,
        })
    return options, {
        "schema": evaluation.get("schema"),
        "status": evaluation.get("status"),
        "sampling_temperature": contract.get("sampling_temperature"),
        "sigma_tilt_used": contract.get("sigma_tilt_used"),
        "rollouts_per_gamma": contract.get("rollouts_per_gamma"),
        "common_random_numbers_across_checkpoints": contract.get(
            "common_random_numbers_across_checkpoints"
        ),
        "coverage_contract": contract.get("bowling_coverage_contract"),
        "scene_contract": {
            "hash": contract.get("scene_hash"),
            "schema": contract.get("scene_schema"),
            "construction": contract.get("construction"),
            "checkpoint_sha256_by_round": contract.get(
                "checkpoint_sha256_by_round"
            ),
        },
    }


def build_payload(
    *,
    expansion: Path,
    bowling_eval: Path,
    task_config: Path,
    max_path_points: int = 30,
) -> dict[str, Any]:
    if max_path_points < 2:
        raise ValueError("max_path_points must be at least two")
    manifest_path = expansion / "manifest.json"
    manifest = _read_json(manifest_path)
    event_sources = _event_sources(expansion)
    evaluation = _read_json(bowling_eval)
    config = _read_json(task_config)
    dense = _dense_option(manifest, event_sources, max_path_points)
    bowling, bowling_contract = _bowling_options(evaluation, max_path_points)
    options = [dense, *bowling]
    if len(options) != 6 or [row["id"] for row in options] != [
        "dense_committed_r1_r5",
        "bowling_r1",
        "bowling_r2",
        "bowling_r3",
        "bowling_r4",
        "bowling_r5",
    ]:
        raise RuntimeError("visual option contract changed")
    return {
        "schema": "pre2_multisphere_rounds_visual_data_v1",
        "geometry_encoding": {
            "coordinate_scale_m": COORDINATE_SCALE_M,
            "layout": (
                "paths: first xyz absolute then flattened xyz deltas; "
                "spheres: flattened absolute xyzr quadruples"
            ),
            "path_sampling": (
                f"endpoint-preserving even index sampling, at most {max_path_points} points"
            ),
        },
        "task": _task_geometry(config),
        "bowling_contract": bowling_contract,
        "options": options,
        "provenance": {
            "expansion_manifest": _source(manifest_path),
            "expansion_events": [_source(path) for path in event_sources],
            "bowling_eval": _source(bowling_eval),
            "task_config": _source(task_config),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--bowling-eval", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-path-points", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=700_000)
    args = parser.parse_args()
    payload = build_payload(
        expansion=args.expansion,
        bowling_eval=args.bowling_eval,
        task_config=args.task_config,
        max_path_points=args.max_path_points,
    )
    encoded = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ) + "\n"
    size = len(encoded.encode("utf-8"))
    if size > args.max_bytes:
        raise RuntimeError(
            f"visual data is {size:,} bytes, above --max-bytes {args.max_bytes:,}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(f"{args.output} ({size:,} bytes; {len(payload['options'])} options)")


if __name__ == "__main__":
    main()
