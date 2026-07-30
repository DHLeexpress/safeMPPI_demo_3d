#!/usr/bin/env python3
"""Paired fail-fast gate for lab-clutter pretraining checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "SR": lambda row: row["status"] == "SUCCESS",
    "CR": lambda row: row["status"] == "COLLISION",
    "OOB": lambda row: row["status"] == "OOB",
    "window_validity": lambda row: row["window_validity"],
}
SPLITS = {
    "id": "raw_audit",
    "ood": "ood_raw_audit",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return payload


def _row_key(row: dict[str, Any]) -> tuple[float, int, str, str]:
    try:
        gamma = float(row["gamma"])
        episode = int(row["episode"])
        scene_id = str(row["scene_id"])
        scene_hash = str(row["scene_hash"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "audit rows require gamma, episode, scene_id, and scene_hash"
        ) from exc
    if scene_id in {"", "None"} or scene_hash in {"", "None"}:
        raise ValueError("paired clutter audits require nonempty scene identity")
    return gamma, episode, scene_id, scene_hash


def _indexed_rows(
    manifest: dict[str, Any],
    field: str,
    *,
    label: str,
) -> dict[tuple[float, int, str, str], dict[str, Any]]:
    rows = manifest.get(field)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: {field} must be a nonempty list")
    indexed: dict[tuple[float, int, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: {field} contains a non-object row")
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"{label}: duplicate paired row {key}")
        for metric in METRICS:
            if metric == "window_validity":
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{label}: row {key} has invalid {metric}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"{label}: row {key} has non-finite {metric}"
                    )
            elif not isinstance(row.get("status"), str):
                raise ValueError(f"{label}: row {key} has invalid status")
        indexed[key] = row
    return indexed


def _metric_values(
    keys: list[tuple[float, int, str, str]],
    rows: dict[tuple[float, int, str, str], dict[str, Any]],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [float(METRICS[metric](rows[key])) for key in keys],
        dtype=np.float64,
    )


def _summaries(
    keys: list[tuple[float, int, str, str]],
    baseline: dict[tuple[float, int, str, str], dict[str, Any]],
    candidate: dict[tuple[float, int, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    gammas = sorted({key[0] for key in keys})
    summary: dict[str, Any] = {"pooled": {}, "per_gamma": {}}
    deltas: dict[str, np.ndarray] = {}
    for metric in METRICS:
        before = _metric_values(keys, baseline, metric)
        after = _metric_values(keys, candidate, metric)
        deltas[metric] = after - before
        summary["pooled"][metric] = {
            "baseline": float(np.mean(before)),
            "candidate": float(np.mean(after)),
            "delta": float(np.mean(after - before)),
        }
    for gamma in gammas:
        gamma_keys = [key for key in keys if key[0] == gamma]
        gamma_summary: dict[str, Any] = {}
        for metric in METRICS:
            before = _metric_values(gamma_keys, baseline, metric)
            after = _metric_values(gamma_keys, candidate, metric)
            gamma_summary[metric] = {
                "baseline": float(np.mean(before)),
                "candidate": float(np.mean(after)),
                "delta": float(np.mean(after - before)),
            }
        summary["per_gamma"][str(gamma)] = gamma_summary
    return summary, deltas


def _cluster_bootstrap(
    keys: list[tuple[float, int, str, str]],
    deltas: dict[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    clusters = sorted({(key[1], key[2], key[3]) for key in keys})
    cluster_to_indices = {
        cluster: np.asarray(
            [
                index
                for index, key in enumerate(keys)
                if (key[1], key[2], key[3]) == cluster
            ],
            dtype=np.int64,
        )
        for cluster in clusters
    }
    gamma_set = {key[0] for key in keys}
    if any(
        {keys[index][0] for index in indices} != gamma_set
        for indices in cluster_to_indices.values()
    ):
        raise ValueError(
            "each scene cluster must contain the same complete gamma sweep"
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(clusters), size=(replicates, len(clusters))
    )
    result: dict[str, Any] = {
        "seed": int(seed),
        "replicates": int(replicates),
        "cluster_unit": "episode+scene_id+scene_hash_shared_across_gammas",
        "cluster_count": len(clusters),
        "metrics": {},
    }
    for metric, row_delta in deltas.items():
        cluster_delta = np.asarray([
            float(np.mean(row_delta[cluster_to_indices[cluster]]))
            for cluster in clusters
        ])
        samples = np.mean(cluster_delta[draws], axis=1)
        result["metrics"][metric] = {
            "mean": float(np.mean(samples)),
            "p10": float(np.quantile(samples, 0.10)),
            "p50": float(np.quantile(samples, 0.50)),
            "p90": float(np.quantile(samples, 0.90)),
        }
    return result


def _criterion(
    value: float,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    tolerance = 1e-12
    if comparison == ">=":
        passed = value >= threshold - tolerance
    elif comparison == ">":
        passed = value > threshold
    elif comparison == "<=":
        passed = value <= threshold + tolerance
    else:
        raise ValueError(f"unsupported comparison {comparison}")
    return {
        "value": float(value),
        "comparison": comparison,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def compare_manifests(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260730,
) -> dict[str, Any]:
    split_results: dict[str, Any] = {}
    for offset, (split, field) in enumerate(SPLITS.items()):
        baseline = _indexed_rows(
            baseline_manifest, field, label="baseline"
        )
        candidate = _indexed_rows(
            candidate_manifest, field, label="candidate"
        )
        if set(baseline) != set(candidate):
            missing = sorted(set(baseline) - set(candidate))
            extra = sorted(set(candidate) - set(baseline))
            raise ValueError(
                f"{split}: paired row mismatch; "
                f"missing_candidate={missing[:3]}, extra_candidate={extra[:3]}"
            )
        keys = sorted(baseline)
        gammas = sorted({key[0] for key in keys})
        if len(gammas) != 4:
            raise ValueError(
                f"{split}: gate requires exactly four gammas, got {gammas}"
            )
        summary, deltas = _summaries(keys, baseline, candidate)
        split_results[split] = {
            "row_count": len(keys),
            "gammas": gammas,
            **summary,
            "cluster_bootstrap": _cluster_bootstrap(
                keys,
                deltas,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + offset,
            ),
        }

    id_result = split_results["id"]
    ood_result = split_results["ood"]
    id_gamma_sr = [
        row["SR"]["delta"]
        for row in id_result["per_gamma"].values()
    ]
    ood_gamma_sr = [
        row["SR"]["delta"]
        for row in ood_result["per_gamma"].values()
    ]
    criteria = {
        "id_pooled_sr_delta": _criterion(
            id_result["pooled"]["SR"]["delta"], 0.08, ">="
        ),
        "id_cluster_bootstrap_sr_delta_p10": _criterion(
            id_result["cluster_bootstrap"]["metrics"]["SR"]["p10"],
            0.0,
            ">",
        ),
        "ood_pooled_sr_delta": _criterion(
            ood_result["pooled"]["SR"]["delta"], -0.05, ">="
        ),
        "worst_ood_gamma_sr_delta": _criterion(
            min(ood_gamma_sr), -0.10, ">="
        ),
        "id_window_validity_loss": _criterion(
            -id_result["pooled"]["window_validity"]["delta"],
            0.02,
            "<=",
        ),
        "ood_window_validity_loss": _criterion(
            -ood_result["pooled"]["window_validity"]["delta"],
            0.03,
            "<=",
        ),
        "id_nonnegative_gamma_sr_count": _criterion(
            float(sum(delta >= 0.0 for delta in id_gamma_sr)),
            3.0,
            ">=",
        ),
        "worst_id_gamma_sr_delta": _criterion(
            min(id_gamma_sr), -0.05, ">="
        ),
    }
    passed = all(item["passed"] for item in criteria.values())
    return {
        "status": "GRU_PRETRAIN_GATE_PASS" if passed
        else "GRU_PRETRAIN_GATE_FAIL",
        "passed": passed,
        "pairing": {
            "identity": ["gamma", "episode", "scene_id", "scene_hash"],
            "requires_identical_rows": True,
        },
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_replicates": int(bootstrap_replicates),
        "splits": split_results,
        "gate": {
            "criteria": criteria,
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    result = compare_manifests(
        _load_manifest(args.baseline),
        _load_manifest(args.candidate),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed"],
        "output": str(args.output.resolve()),
    }, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
