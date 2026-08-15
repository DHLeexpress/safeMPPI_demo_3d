#!/usr/bin/env python3
"""Convert randomized multi-sphere raw evaluation rows to paper-curve JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


VALID_STATUSES = frozenset({"SUCCESS", "COLLISION", "OOB", "TIMEOUT"})
DEFAULT_GAMMAS = (0.1, 0.3, 0.5, 1.0)
SCHEMA = "multisphere_random_raw_curve_cells_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rounds(raw: str) -> tuple[int, ...]:
    values: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lower_raw, upper_raw = token.split("-", 1)
            lower, upper = int(lower_raw), int(upper_raw)
            if lower < 0 or upper < lower:
                raise ValueError(f"invalid round range {token!r}")
            values.update(range(lower, upper + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError("rounds must be nonnegative")
            values.add(value)
    if not values:
        raise ValueError("--expected-rounds must not be empty")
    return tuple(sorted(values))


def parse_gammas(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    if (
        not values
        or len(values) != len(set(values))
        or any(not math.isfinite(value) for value in values)
    ):
        raise ValueError("--expected-gammas must be finite and unique")
    return values


def _matching_gamma(value: Any, expected: tuple[float, ...]) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid gamma value {value!r}") from error
    matches = [
        gamma for gamma in expected
        if math.isclose(observed, gamma, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"observed gamma {observed:g} is outside expected set {expected}"
        )
    return matches[0]


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _bernoulli(values: Iterable[bool]) -> dict[str, float | int]:
    flags = tuple(bool(value) for value in values)
    if not flags:
        raise ValueError("Bernoulli metric requires at least one episode")
    count = len(flags)
    mean = sum(flags) / count
    return {
        "mean": float(mean),
        "se": float(math.sqrt(mean * (1.0 - mean) / count)),
        "n": count,
    }


def _continuous(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = tuple(float(value) for value in values)
    if not samples:
        return {"mean": None, "se": 0.0, "n_success": 0}
    mean = sum(samples) / len(samples)
    if len(samples) == 1:
        standard_error = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in samples) / (
            len(samples) - 1
        )
        standard_error = math.sqrt(variance / len(samples))
    return {
        "mean": float(mean),
        "se": float(standard_error),
        "n_success": len(samples),
    }


def _cell_signature(rows: list[dict[str, Any]]) -> tuple | None:
    has_scene = ["scene_hash" in row for row in rows]
    has_seed = ["rollout_seed" in row for row in rows]
    if any(has_scene) != all(has_scene) or any(has_seed) != all(has_seed):
        raise ValueError(
            "scene_hash and rollout_seed must be present for every row in a cell "
            "or absent from every row"
        )
    if not all(has_scene) or not all(has_seed):
        return None
    return tuple(sorted(
        (
            int(row["episode"]),
            str(row["scene_hash"]),
            int(row["rollout_seed"]),
        )
        for row in rows
    ))


def summarize_cell(
    round_i: int,
    gamma: float,
    rows: list[dict[str, Any]],
    *,
    expected_m: int,
    expected_temperature: float,
    checkpoint_sha256: str | None,
    source_sha256: str,
) -> dict[str, Any]:
    if len(rows) != expected_m:
        raise ValueError(
            f"round {round_i}, gamma {gamma:g} has {len(rows)} rows; "
            f"expected M={expected_m}"
        )
    episodes = [int(row["episode"]) for row in rows]
    if len(set(episodes)) != expected_m:
        raise ValueError(
            f"round {round_i}, gamma {gamma:g} has duplicate episode IDs"
        )

    statuses: list[str] = []
    validity: list[bool] = []
    clearances: list[float] = []
    times: list[float] = []
    for episode, row in zip(episodes, rows):
        status = str(row.get("status"))
        if status not in VALID_STATUSES:
            raise ValueError(
                f"round {round_i}, gamma {gamma:g}, episode {episode} has "
                f"invalid status {status!r}"
            )
        statuses.append(status)
        window_validity = _finite_float(
            row.get("window_validity"), "window_validity",
        )
        if not 0.0 <= window_validity <= 1.0:
            raise ValueError("window_validity must lie in [0,1]")
        # Paper V_safe is an episode-level event: every executed window must
        # pass.  A fractional window average is deliberately not averaged here.
        validity.append(window_validity == 1.0)

        if "sampling_temperature" in row:
            temperature = _finite_float(
                row["sampling_temperature"], "sampling_temperature",
            )
            if not math.isclose(
                temperature, expected_temperature, rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError(
                    f"row sampling temperature {temperature:g} differs from "
                    f"expected {expected_temperature:g}"
                )
        if status == "SUCCESS":
            clearances.append(_finite_float(
                row.get("min_clearance_m"), "successful min_clearance_m",
            ))
            times.append(_finite_float(
                row.get("time_to_goal_s"), "successful time_to_goal_s",
            ))

    cell: dict[str, Any] = {
        "round": int(round_i),
        "gamma": float(gamma),
        "mode": "randomized_dense_z_raw_cost_ranked_deployment",
        "temp": float(expected_temperature),
        "m": int(expected_m),
        "n": int(expected_m),
        "SR": _bernoulli(status == "SUCCESS" for status in statuses),
        # CR remains physical collision only.  OOB is retained separately.
        "CR": _bernoulli(status == "COLLISION" for status in statuses),
        "OOB": _bernoulli(status == "OOB" for status in statuses),
        "timeout": _bernoulli(status == "TIMEOUT" for status in statuses),
        "v_safe": _bernoulli(validity),
        "clearance": _continuous(clearances),
        "time": _continuous(times),
        "source_raw_eval_sha256": source_sha256,
    }
    if checkpoint_sha256 is not None:
        cell["checkpoint_sha256"] = checkpoint_sha256
    return cell


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return payload


def _metadata(path: Path, payload: dict[str, Any], keys: tuple[str, ...]) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        **{key: payload[key] for key in keys if key in payload},
    }


def _discover(path: Path, filename: str) -> Path | None:
    for directory in (path.parent, path.parent.parent):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def convert(
    raw_eval_paths: list[Path],
    *,
    expected_rounds: tuple[int, ...],
    expected_gammas: tuple[float, ...],
    expected_m: int,
    expected_temperature: float,
    task_config_path: Path | None = None,
    expansion_manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_m < 1:
        raise ValueError("--expected-m must be positive")
    if not math.isfinite(expected_temperature) or expected_temperature < 0.0:
        raise ValueError("--expected-temperature must be finite and nonnegative")
    if not raw_eval_paths:
        raise ValueError("at least one --raw-eval is required")

    cells: dict[tuple[int, float], tuple[list[dict[str, Any]], dict]] = {}
    sources = []
    discovered_task_configs: dict[str, dict] = {}
    discovered_manifests: dict[str, dict] = {}
    for raw_path in raw_eval_paths:
        raw_path = raw_path.resolve()
        payload = _load_json(raw_path, "raw evaluation")
        rows_by_round = payload.get("rows")
        if not isinstance(rows_by_round, dict) or not rows_by_round:
            raise ValueError(f"raw evaluation {raw_path} has no rows mapping")
        source_sha = sha256_file(raw_path)
        temperature = _finite_float(
            payload.get("sampling_temperature"), "sampling_temperature",
        )
        if not math.isclose(
            temperature, expected_temperature, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(
                f"{raw_path} sampling temperature {temperature:g} differs from "
                f"expected {expected_temperature:g}"
            )
        if payload.get("sigma_tilt_used") not in (None, False):
            raise ValueError(f"{raw_path} is not an untilted raw evaluation")
        binding = payload.get("artifact_binding", {})
        if not isinstance(binding, dict):
            raise ValueError("artifact_binding must be a JSON object")
        checkpoint_hashes = binding.get("checkpoint_sha256_by_round", {})
        if not isinstance(checkpoint_hashes, dict):
            raise ValueError("checkpoint_sha256_by_round must be a JSON object")

        source_rounds: list[int] = []
        source_gammas: set[float] = set()
        for round_raw, round_rows in rows_by_round.items():
            round_i = int(round_raw)
            if round_i not in expected_rounds:
                raise ValueError(
                    f"{raw_path} contains unexpected round {round_i}"
                )
            if not isinstance(round_rows, list) or not round_rows:
                raise ValueError(f"round {round_i} rows must be a nonempty list")
            grouped: dict[float, list[dict[str, Any]]] = {}
            for row in round_rows:
                if not isinstance(row, dict):
                    raise ValueError("raw evaluation rows must be JSON objects")
                gamma = _matching_gamma(row.get("gamma"), expected_gammas)
                grouped.setdefault(gamma, []).append(row)
                source_gammas.add(gamma)
            for gamma, gamma_rows in grouped.items():
                key = (round_i, gamma)
                if key in cells:
                    raise ValueError(
                        f"duplicate round-gamma cell across raw evaluations: {key}"
                    )
                cells[key] = (gamma_rows, {
                    "source_sha256": source_sha,
                    "checkpoint_sha256": checkpoint_hashes.get(str(round_i)),
                })
            source_rounds.append(round_i)

        source_record = {
            "path": str(raw_path),
            "sha256": source_sha,
            "status": payload.get("status"),
            "sampling_temperature": temperature,
            "rounds": sorted(set(source_rounds)),
            "gammas": sorted(source_gammas),
            "artifact_binding": binding,
            "scene_bank_contract": {
                key: payload.get("scene_bank", {}).get(key)
                for key in (
                    "schema", "evaluation_seed", "shared_across_rounds",
                    "shared_across_gamma", "sampler",
                )
                if isinstance(payload.get("scene_bank"), dict)
            },
        }
        sources.append(source_record)

        discovered_task = _discover(raw_path, "task_config_resolved.json")
        if discovered_task is not None:
            data = _load_json(discovered_task, "task config")
            metadata = _metadata(
                discovered_task, data,
                ("taskspace", "scene_randomization", "data"),
            )
            discovered_task_configs[metadata["sha256"]] = metadata
        discovered_manifest = _discover(raw_path, "manifest.json")
        if discovered_manifest is not None:
            data = _load_json(discovered_manifest, "expansion manifest")
            metadata = _metadata(
                discovered_manifest, data,
                (
                    "task_profile", "lab_scene_schema",
                    "lab_scene_randomization", "config",
                ),
            )
            expected_hash = binding.get("expansion_manifest_sha256")
            metadata["matches_raw_artifact_binding"] = (
                expected_hash is None or expected_hash == metadata["sha256"]
            )
            discovered_manifests[metadata["sha256"]] = metadata

    expected_cells = {
        (round_i, gamma)
        for round_i in expected_rounds for gamma in expected_gammas
    }
    missing = sorted(expected_cells - set(cells))
    extra = sorted(set(cells) - expected_cells)
    if missing or extra:
        raise ValueError(
            f"incomplete round-gamma grid; missing={missing}, extra={extra}"
        )

    signatures = []
    output_rows = []
    for round_i in expected_rounds:
        for gamma in expected_gammas:
            cell_rows, cell_source = cells[(round_i, gamma)]
            signatures.append(_cell_signature(cell_rows))
            output_rows.append(summarize_cell(
                round_i,
                gamma,
                cell_rows,
                expected_m=expected_m,
                expected_temperature=expected_temperature,
                checkpoint_sha256=cell_source["checkpoint_sha256"],
                source_sha256=cell_source["source_sha256"],
            ))
    available_signatures = [signature for signature in signatures if signature]
    crn_verified = (
        len(available_signatures) == len(signatures)
        and len(set(available_signatures)) == 1
    )

    if task_config_path is not None:
        task_config_path = task_config_path.resolve()
        data = _load_json(task_config_path, "task config")
        metadata = _metadata(
            task_config_path, data,
            ("taskspace", "scene_randomization", "data"),
        )
        discovered_task_configs = {metadata["sha256"]: metadata}
    if expansion_manifest_path is not None:
        expansion_manifest_path = expansion_manifest_path.resolve()
        data = _load_json(expansion_manifest_path, "expansion manifest")
        metadata = _metadata(
            expansion_manifest_path, data,
            (
                "task_profile", "lab_scene_schema",
                "lab_scene_randomization", "config",
            ),
        )
        bound_hashes = {
            source["artifact_binding"].get("expansion_manifest_sha256")
            for source in sources
            if source["artifact_binding"].get("expansion_manifest_sha256")
            is not None
        }
        if bound_hashes and bound_hashes != {metadata["sha256"]}:
            raise ValueError(
                "explicit expansion manifest does not match raw-evaluation "
                f"artifact binding: {sorted(bound_hashes)}"
            )
        metadata["matches_raw_artifact_binding"] = True
        discovered_manifests = {metadata["sha256"]: metadata}

    provenance = {
        "status": "MULTISPHERE_RANDOM_RAW_CURVE_JSONL_COMPLETE",
        "schema": SCHEMA,
        "grid": {
            "rounds": list(expected_rounds),
            "gammas": list(expected_gammas),
            "cells": len(output_rows),
            "M_per_gamma_round": expected_m,
            "sampling_temperature": expected_temperature,
        },
        "metric_contract": {
            "SR": "episode Bernoulli: status == SUCCESS",
            "CR": "episode Bernoulli: physical status == COLLISION; OOB excluded",
            "OOB": "episode Bernoulli: status == OOB",
            "timeout": "episode Bernoulli: status == TIMEOUT",
            "v_safe": (
                "episode Bernoulli: window_validity == 1.0, so every executed "
                "terminal-truncated window is in-bounds, collision-free and "
                "GREEN-certified"
            ),
            "clearance": "mean and sample SE over successful trajectories only",
            "time": "mean and sample SE over successful trajectories only",
            "binary_intervals_in_renderer": "95% Wilson using M trials",
            "continuous_intervals_in_renderer": "mean +/- 1.95996398454 SE",
        },
        "common_random_numbers": {
            "verified_across_all_round_gamma_cells": crn_verified,
            "signature_fields": ["episode", "scene_hash", "rollout_seed"],
        },
        "sources": sources,
        "task_configs": list(discovered_task_configs.values()),
        "expansion_manifests": list(discovered_manifests.values()),
    }
    return output_rows, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-eval", action="append", type=Path, required=True,
        help="raw_eval.json input; repeat for nonoverlapping checkpoint shards",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provenance-output", type=Path,
        help="default: OUTPUT with .provenance.json suffix",
    )
    parser.add_argument("--expected-rounds", default="0-5")
    parser.add_argument(
        "--expected-gammas",
        default=",".join(f"{gamma:g}" for gamma in DEFAULT_GAMMAS),
    )
    parser.add_argument("--expected-m", type=int, default=50)
    parser.add_argument("--expected-temperature", type=float, default=1.0)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--expansion-manifest", type=Path)
    args = parser.parse_args()

    try:
        expected_rounds = parse_rounds(args.expected_rounds)
        expected_gammas = parse_gammas(args.expected_gammas)
        rows, provenance = convert(
            args.raw_eval,
            expected_rounds=expected_rounds,
            expected_gammas=expected_gammas,
            expected_m=args.expected_m,
            expected_temperature=args.expected_temperature,
            task_config_path=args.task_config,
            expansion_manifest_path=args.expansion_manifest,
        )
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ))
    provenance_path = (
        args.provenance_output
        if args.provenance_output is not None
        else args.output.with_suffix(".provenance.json")
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance["output"] = {
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "rows": len(rows),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.output)
    print(provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
