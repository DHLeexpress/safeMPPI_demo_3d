#!/usr/bin/env python3
"""Merge split fixed-bowling checkpoint evaluations into one audit artifact."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.bowling_coverage import (  # noqa: E402
    BOWLING_ROUTE_CODES,
    summarize_bowling_coverage,
)


COMPLETE_STATUS = (
    "LAB_CLUTTER_FIXED_SCENE_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
)
DEFAULT_EXPECTED_ROUNDS = tuple(range(6))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gamma_key(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid gamma value: {value!r}") from error
    if not (numeric == numeric and abs(numeric) != float("inf")):
        raise ValueError(f"gamma must be finite: {value!r}")
    return f"{numeric:g}"


def _round_key(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint round: {value!r}") from error
    if str(value) not in {str(numeric), f"{numeric:03d}"}:
        raise ValueError(f"checkpoint round is not canonical: {value!r}")
    if numeric < 0:
        raise ValueError(f"checkpoint round must be nonnegative: {numeric}")
    return str(numeric)


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _normalized_seed_contract(payload: dict, source: Path) -> dict[str, list[int]]:
    raw = _mapping(
        payload.get("rollout_seeds_by_gamma"),
        f"{source}: rollout_seeds_by_gamma",
    )
    normalized: dict[str, list[int]] = {}
    for raw_gamma, raw_seeds in raw.items():
        gamma = _gamma_key(raw_gamma)
        if gamma in normalized:
            raise ValueError(
                f"{source}: duplicate normalized gamma in seed contract: {gamma}"
            )
        seeds = [
            int(seed) for seed in _list(
                raw_seeds, f"{source}: seeds for gamma={gamma}",
            )
        ]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"{source}: duplicate rollout seed for gamma={gamma}"
            )
        normalized[gamma] = seeds
    if not normalized:
        raise ValueError(f"{source}: empty rollout seed contract")
    return normalized


def _scene_contract(payload: dict, source: Path) -> dict[str, Any]:
    provenance = _mapping(
        payload.get("scene_provenance"), f"{source}: scene_provenance",
    )
    if provenance.get("mode") != "bowling_123":
        raise ValueError(f"{source}: fixed scene is not bowling_123")
    if provenance.get("shared_across_rounds") is not True:
        raise ValueError(f"{source}: fixed scene is not shared across rounds")
    if provenance.get("shared_across_gamma") is not True:
        raise ValueError(f"{source}: fixed scene is not shared across gamma")
    scene = _mapping(provenance.get("scene"), f"{source}: fixed scene")
    scene_hash = _required_text(
        scene.get("scene_hash"), f"{source}: fixed scene scene_hash",
    )
    spheres = _list(scene.get("spheres"), f"{source}: fixed scene spheres")
    if len(spheres) != 6:
        raise ValueError(f"{source}: bowling_123 must have exactly six spheres")
    concrete = _mapping(
        payload.get("concrete_config"), f"{source}: concrete_config",
    )
    if concrete.get("scene_hash") != scene_hash:
        raise ValueError(
            f"{source}: concrete config and fixed scene hashes disagree"
        )
    coverage = _mapping(
        payload.get("bowling_coverage_contract"),
        f"{source}: bowling_coverage_contract",
    )
    if tuple(coverage.get("route_codes", ())) != BOWLING_ROUTE_CODES:
        raise ValueError(f"{source}: bowling route-code contract changed")
    if int(coverage.get("coverage_denominator", -1)) != len(
        BOWLING_ROUTE_CODES
    ):
        raise ValueError(f"{source}: bowling coverage denominator is not eight")
    if coverage.get("full_3d_homotopy_claimed") is not False:
        raise ValueError(
            f"{source}: projected coverage must not claim full 3-D homotopy"
        )
    return {
        "scene_hash": scene_hash,
        "scene": scene,
        "scene_schema": provenance.get("schema"),
        "construction": provenance.get("construction"),
        "concrete_config_sha256": _required_text(
            concrete.get("sha256"), f"{source}: concrete config sha256",
        ),
        "bowling_coverage_contract": coverage,
    }


def _evaluation_contract(
    payload: dict,
    source: Path,
    expected_rollouts_per_gamma: int,
) -> dict[str, Any]:
    if payload.get("status") != COMPLETE_STATUS:
        raise ValueError(f"{source}: evaluation is not complete temperature-1 raw")
    if float(payload.get("sampling_temperature", float("nan"))) != 1.0:
        raise ValueError(f"{source}: bowling coverage must use temperature 1.0")
    if payload.get("sigma_tilt_used") is not False:
        raise ValueError(f"{source}: bowling coverage unexpectedly used tilting")
    if payload.get("common_random_numbers_across_checkpoints") is not True:
        raise ValueError(f"{source}: checkpoint seed sharing is not declared")
    rollouts = int(payload.get("rollouts_per_gamma", -1))
    if rollouts != int(expected_rollouts_per_gamma):
        raise ValueError(
            f"{source}: expected {expected_rollouts_per_gamma} rollouts per "
            f"gamma, found {rollouts}"
        )
    seeds = _normalized_seed_contract(payload, source)
    for gamma, gamma_seeds in seeds.items():
        if len(gamma_seeds) != rollouts:
            raise ValueError(
                f"{source}: gamma={gamma} has {len(gamma_seeds)} seeds, "
                f"expected {rollouts}"
            )
    artifact = _mapping(
        payload.get("artifact_binding"), f"{source}: artifact_binding",
    )
    if artifact.get("round_zero_model_bitwise_equal_to_pretrained") is not True:
        raise ValueError(f"{source}: round zero is not bound to PRE bitwise")
    checkpoint_hashes = _mapping(
        artifact.get("checkpoint_sha256_by_round"),
        f"{source}: checkpoint hashes",
    )
    pretrained_sha = _required_text(
        artifact.get("pretrained_checkpoint_sha256"),
        f"{source}: pretrained checkpoint sha256",
    )
    pretrain_manifest_sha = _required_text(
        artifact.get("pretrain_manifest_sha256"),
        f"{source}: pretrain manifest sha256",
    )
    round_zero_sha = _required_text(
        checkpoint_hashes.get("0"), f"{source}: round-zero checkpoint sha256",
    )
    return {
        "scene": _scene_contract(payload, source),
        "rollout_seeds_by_gamma": seeds,
        "rollouts_per_gamma": rollouts,
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "pretrain_manifest_sha256": pretrain_manifest_sha,
        "round_zero_checkpoint_sha256": round_zero_sha,
    }


def _require_same_contract(
    expected: dict[str, Any],
    observed: dict[str, Any],
    source: Path,
) -> None:
    if observed != expected:
        differing = sorted(
            key for key in set(expected) | set(observed)
            if expected.get(key) != observed.get(key)
        )
        raise ValueError(
            f"{source}: bowling evaluation contract differs in: "
            + ", ".join(differing)
        )


def _validate_success_signature(row: dict, source: Path, round_i: int) -> None:
    signature = row.get("bowling_route")
    if not isinstance(signature, dict):
        raise ValueError(
            f"{source}: round {round_i} SUCCESS row lacks bowling_route"
        )
    code = str(signature.get("code", ""))
    stable = str(signature.get("stable_code", ""))
    if code not in BOWLING_ROUTE_CODES:
        raise ValueError(
            f"{source}: round {round_i} has invalid raw route code {code!r}"
        )
    if len(stable) != 3 or any(bit not in "LRX" for bit in stable):
        raise ValueError(
            f"{source}: round {round_i} has invalid stable code {stable!r}"
        )
    for key in (
        "decision_vertical_m",
        "decision_vertical_dominant",
    ):
        values = _list(
            signature.get(key),
            f"{source}: round {round_i} {key}",
        )
        if len(values) != 3:
            raise ValueError(
                f"{source}: round {round_i} {key} must have three entries"
            )
        if key == "decision_vertical_m":
            try:
                finite = all(
                    float(value) == float(value)
                    and abs(float(value)) != float("inf")
                    for value in values
                )
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(
                    f"{source}: round {round_i} {key} must be finite"
                )
        elif any(type(value) is not bool for value in values):
            raise ValueError(
                f"{source}: round {round_i} {key} must be boolean"
            )
    vertical_sign = str(signature.get("decision_vertical_sign", ""))
    if len(vertical_sign) != 3 or any(value not in "APB" for value in vertical_sign):
        raise ValueError(
            f"{source}: round {round_i} has invalid vertical sign "
            f"{vertical_sign!r}"
        )
    path = row.get("arc_length_resampled_path_xyz")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError(
            f"{source}: round {round_i} SUCCESS row lacks visualizable path"
        )


def _validated_round_rows(
    payload: dict,
    source: Path,
    seed_contract: dict[str, list[int]],
    scene_hash: str,
) -> dict[int, list[dict]]:
    raw_rounds = _mapping(payload.get("rows"), f"{source}: rows")
    raw_summary = _mapping(payload.get("summary"), f"{source}: summary")
    normalized_summary_round_list = [
        _round_key(key) for key in raw_summary
    ]
    if len(normalized_summary_round_list) != len(
        set(normalized_summary_round_list)
    ):
        raise ValueError(f"{source}: duplicate normalized summary round")
    normalized_summary_rounds = set(normalized_summary_round_list)
    rounds: dict[int, list[dict]] = {}
    for raw_round, raw_rows in raw_rounds.items():
        round_key = _round_key(raw_round)
        round_i = int(round_key)
        if round_i in rounds:
            raise ValueError(f"{source}: duplicate normalized round {round_i}")
        if round_key not in normalized_summary_rounds:
            raise ValueError(f"{source}: round {round_i} lacks summary")
        rows = _list(raw_rows, f"{source}: rows for round {round_i}")
        seen_seed_cells: set[tuple[str, int]] = set()
        seen_episode_cells: set[tuple[str, int]] = set()
        observed: dict[str, list[int]] = {
            gamma: [] for gamma in seed_contract
        }
        for row in rows:
            row = _mapping(row, f"{source}: round {round_i} row")
            gamma = _gamma_key(row.get("gamma"))
            if gamma not in seed_contract:
                raise ValueError(
                    f"{source}: round {round_i} has unexpected gamma={gamma}"
                )
            rollout_seed = int(row.get("rollout_seed"))
            episode = int(row.get("episode"))
            seed_cell = (gamma, rollout_seed)
            episode_cell = (gamma, episode)
            if seed_cell in seen_seed_cells or episode_cell in seen_episode_cells:
                raise ValueError(
                    f"{source}: duplicate round/cell at round={round_i}, "
                    f"gamma={gamma}, seed={rollout_seed}, episode={episode}"
                )
            seen_seed_cells.add(seed_cell)
            seen_episode_cells.add(episode_cell)
            observed[gamma].append(rollout_seed)
            if row.get("scene_hash") != scene_hash:
                raise ValueError(
                    f"{source}: round {round_i} row scene hash changed"
                )
            status = row.get("status")
            if status not in {"SUCCESS", "COLLISION", "OOB", "TIMEOUT"}:
                raise ValueError(
                    f"{source}: round {round_i} has invalid status {status!r}"
                )
            if status == "SUCCESS":
                _validate_success_signature(row, source, round_i)
        for gamma, expected_seeds in seed_contract.items():
            if sorted(observed[gamma]) != sorted(expected_seeds):
                raise ValueError(
                    f"{source}: round {round_i}, gamma={gamma} does not "
                    "match the fixed rollout seed bank"
                )
        gamma_order = {gamma: index for index, gamma in enumerate(seed_contract)}
        seed_order = {
            gamma: {seed: index for index, seed in enumerate(seeds)}
            for gamma, seeds in seed_contract.items()
        }
        rows.sort(key=lambda row: (
            gamma_order[_gamma_key(row["gamma"])],
            seed_order[_gamma_key(row["gamma"])][int(row["rollout_seed"])],
        ))
        rounds[round_i] = rows
    if set(map(str, rounds)) != normalized_summary_rounds:
        extra = sorted(normalized_summary_rounds - set(map(str, rounds)))
        raise ValueError(f"{source}: summary has rounds without rows: {extra}")
    return rounds


def _coverage_summary(rows: list[dict]) -> dict[str, Any]:
    summary = summarize_bowling_coverage(rows)
    status_counts = Counter(str(row["status"]) for row in rows)
    signatures = [
        row["bowling_route"] for row in rows
        if row["status"] == "SUCCESS"
    ]
    dominant_stage_count = sum(
        sum(bool(value) for value in signature["decision_vertical_dominant"])
        for signature in signatures
    )
    decision_stage_count = 3 * len(signatures)
    vertical_sign_counts = Counter(
        str(signature["decision_vertical_sign"])
        for signature in signatures
    )
    summary.update({
        "coverage_out_of_8": f"{summary['coverage_count']}/8",
        "status_counts": {
            status: int(status_counts.get(status, 0))
            for status in ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
        },
        "vertical_dominant_decision_stages": int(dominant_stage_count),
        "decision_stages": int(decision_stage_count),
        "vertical_dominant_decision_fraction": (
            float(dominant_stage_count / decision_stage_count)
            if decision_stage_count else None
        ),
        "vertical_sign_counts": dict(sorted(vertical_sign_counts.items())),
    })
    return summary


def _validate_declared_coverage(
    payload: dict,
    round_i: int,
    rows: list[dict],
    gammas: list[str],
    source: Path,
) -> None:
    declared_round = _mapping(
        payload["summary"][str(round_i)],
        f"{source}: declared round {round_i} summary",
    )
    declared = _mapping(
        declared_round.get("bowling_coverage"),
        f"{source}: declared round {round_i} bowling coverage",
    )
    cells = {
        "pooled": rows,
        **{
            gamma: [
                row for row in rows if _gamma_key(row["gamma"]) == gamma
            ]
            for gamma in gammas
        },
    }
    declared_cells = {
        "pooled": declared.get("pooled"),
        **{
            gamma: _mapping(
                declared.get("per_gamma"),
                f"{source}: declared per-gamma coverage",
            ).get(gamma)
            for gamma in gammas
        },
    }
    for cell, cell_rows in cells.items():
        claimed = _mapping(
            declared_cells[cell],
            f"{source}: round {round_i} declared cell {cell}",
        )
        recomputed = summarize_bowling_coverage(cell_rows)
        for key in (
            "attempts",
            "terminal_successes",
            "classified_successes",
            "stable_classified_successes",
            "ambiguous_successes",
            "route_counts",
            "observed_routes",
            "coverage_count",
        ):
            if claimed.get(key) != recomputed[key]:
                raise ValueError(
                    f"{source}: round {round_i} declared {cell} coverage "
                    f"disagrees for {key}"
                )


def merge_bowling_evaluations(
    inputs: list[Path],
    *,
    expected_rounds: tuple[int, ...] = DEFAULT_EXPECTED_ROUNDS,
    expected_rollouts_per_gamma: int = 50,
) -> dict[str, Any]:
    """Validate and merge disjoint split-checkpoint evaluation artifacts."""
    if not inputs:
        raise ValueError("at least one fixed_scene_raw_eval.json is required")
    if int(expected_rollouts_per_gamma) < 1:
        raise ValueError("expected rollouts per gamma must be positive")
    expected_round_set = set(map(int, expected_rounds))
    if len(expected_round_set) != len(expected_rounds):
        raise ValueError("expected checkpoint rounds contain duplicates")

    shared_contract = None
    merged_rows: dict[int, list[dict]] = {}
    source_rows = []
    checkpoint_hashes: dict[str, str] = {}
    for raw_path in inputs:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        contract = _evaluation_contract(
            payload, path, int(expected_rollouts_per_gamma),
        )
        if shared_contract is None:
            shared_contract = contract
        else:
            _require_same_contract(shared_contract, contract, path)
        assert shared_contract is not None
        rounds = _validated_round_rows(
            payload,
            path,
            shared_contract["rollout_seeds_by_gamma"],
            shared_contract["scene"]["scene_hash"],
        )
        overlap = sorted(set(rounds) & set(merged_rows))
        if overlap:
            raise ValueError(
                f"{path}: duplicate evaluated checkpoint rounds: {overlap}"
            )
        artifact = payload["artifact_binding"]
        binding = artifact["checkpoint_sha256_by_round"]
        for round_i in rounds:
            round_key = str(round_i)
            checkpoint_hash = binding.get(round_key)
            if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
                raise ValueError(
                    f"{path}: missing checkpoint hash for round {round_i}"
                )
            previous = checkpoint_hashes.get(round_key)
            if previous is not None and previous != checkpoint_hash:
                raise ValueError(
                    f"{path}: conflicting checkpoint hash for round {round_i}"
                )
            checkpoint_hashes[round_key] = checkpoint_hash
            _validate_declared_coverage(
                payload,
                round_i,
                rounds[round_i],
                list(shared_contract["rollout_seeds_by_gamma"]),
                path,
            )
        merged_rows.update(rounds)
        source_rows.append({
            "path": str(path),
            "sha256": _sha256_file(path),
            "rounds": sorted(rounds),
            "runtime_device": payload.get("runtime_device"),
            "expansion_manifest_sha256": artifact.get(
                "expansion_manifest_sha256"
            ),
        })

    source_rows.sort(key=lambda row: (row["rounds"], row["path"]))
    actual_rounds = set(merged_rows)
    if actual_rounds != expected_round_set:
        raise ValueError(
            "merged checkpoint rounds do not match the required boundary: "
            f"missing={sorted(expected_round_set - actual_rounds)}, "
            f"extra={sorted(actual_rounds - expected_round_set)}"
        )
    assert shared_contract is not None
    gammas = list(shared_contract["rollout_seeds_by_gamma"])
    summaries: dict[str, dict[str, Any]] = {}
    serialized_rows: dict[str, list[dict]] = {}
    for round_i in sorted(merged_rows):
        rows = merged_rows[round_i]
        summaries[str(round_i)] = {
            "pooled": _coverage_summary(rows),
            "per_gamma": {
                gamma: _coverage_summary([
                    row for row in rows
                    if _gamma_key(row["gamma"]) == gamma
                ])
                for gamma in gammas
            },
        }
        serialized_rows[str(round_i)] = rows

    scene_contract = shared_contract["scene"]
    return {
        "schema": "bowling_123_fixed_bank_coverage_merge_v1",
        "status": "BOWLING_COVERAGE_R0_TO_R5_COMPLETE",
        "required_rounds": sorted(expected_round_set),
        "sources": source_rows,
        "contract": {
            "scene_hash": scene_contract["scene_hash"],
            "scene": scene_contract["scene"],
            "scene_schema": scene_contract["scene_schema"],
            "construction": scene_contract["construction"],
            "concrete_config_sha256": scene_contract[
                "concrete_config_sha256"
            ],
            "bowling_coverage_contract": scene_contract[
                "bowling_coverage_contract"
            ],
            "sampling_temperature": 1.0,
            "sigma_tilt_used": False,
            "common_random_numbers_across_checkpoints": True,
            "gammas": [float(gamma) for gamma in gammas],
            "rollouts_per_gamma": int(expected_rollouts_per_gamma),
            "attempts_per_checkpoint": (
                len(gammas) * int(expected_rollouts_per_gamma)
            ),
            "rollout_seeds_by_gamma": shared_contract[
                "rollout_seeds_by_gamma"
            ],
            "projected_route_codes": list(BOWLING_ROUTE_CODES),
            "coverage_denominator": len(BOWLING_ROUTE_CODES),
            "pretrained_checkpoint_sha256": shared_contract[
                "pretrained_checkpoint_sha256"
            ],
            "pretrain_manifest_sha256": shared_contract[
                "pretrain_manifest_sha256"
            ],
            "round_zero_checkpoint_sha256": shared_contract[
                "round_zero_checkpoint_sha256"
            ],
            "checkpoint_sha256_by_round": {
                key: checkpoint_hashes[key]
                for key in map(str, sorted(expected_round_set))
            },
        },
        "summary": summaries,
        "rows": serialized_rows,
        "visual_options": [
            {
                "id": f"bowling_r{round_i}",
                "label": f"Bowling evaluation · R{round_i}",
                "round": round_i,
                "summary_key": f"summary.{round_i}",
                "rows_key": f"rows.{round_i}",
            }
            for round_i in sorted(expected_round_set)
            if round_i > 0
        ],
        "visual_contract": {
            "top_level_bowling_option_count": sum(
                round_i > 0 for round_i in expected_round_set
            ),
            "round_zero_is_numeric_baseline_only": 0 in expected_round_set,
            "final_selector_adds_one_dense_collection_option": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="repeat once per split fixed_scene_raw_eval.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-rollouts-per-gamma",
        type=int,
        default=50,
        help="fail closed unless every gamma/checkpoint uses this M (default 50)",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    merged = merge_bowling_evaluations(
        args.input,
        expected_rollouts_per_gamma=args.expected_rollouts_per_gamma,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, allow_nan=False) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
