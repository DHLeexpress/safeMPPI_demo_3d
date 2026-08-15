#!/usr/bin/env python3
"""Merge disjoint gamma-specific mirrored-pair archives for pretraining."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.mirrored_pair_collection import MIRRORED_PAIR_SCHEMA  # noqa: E402


MERGED_SCHEMA = "merged_gamma_exclusive_mirrored_pairs_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_raw(raw: dict) -> dict:
    value = copy.deepcopy(raw)
    value["data"] = dict(value["data"])
    value["data"]["gammas"] = []
    # Quota-generation controls are archive bookkeeping, not rollout science.
    # Ignoring them lets disjoint 400+600 generations be merged while all
    # dynamics, geometry, controller, and acceptance settings stay identical.
    value["data"]["episodes_per_gamma"] = 0
    value["data"]["max_attempts_per_gamma"] = 0
    return value


def merge_archives(
    sources: list[Path],
    output: Path,
    *,
    expected_successes_per_gamma: int,
) -> dict:
    if len(sources) < 2:
        raise ValueError("merge requires at least two gamma-specific archives")
    target = int(expected_successes_per_gamma)
    if target < 2 or target % 2:
        raise ValueError("expected success count must be positive and even")
    sources = [Path(source).resolve() for source in sources]
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output {output}")

    loaded = []
    for source in sources:
        manifest = json.loads((source / "manifest.json").read_text())
        if (
            manifest.get("schema") != MIRRORED_PAIR_SCHEMA
            or manifest.get("status") != "COMPLETE_EXACT_PAIRED_SUCCESS_QUOTA"
        ):
            raise ValueError(f"source is not a complete mirrored archive: {source}")
        if len(manifest.get("gammas", [])) != 1:
            raise ValueError(f"source must contain exactly one gamma: {source}")
        gamma = float(manifest["gammas"][0])
        key = f"{gamma:.9g}"
        accepted_count = int(
            manifest["accepted_counts_by_gamma"].get(key, -1)
        )
        if accepted_count < 2 or accepted_count % 2:
            raise ValueError(
                f"source gamma={gamma:g} has invalid paired run count "
                f"{accepted_count}"
            )
        if len(manifest["runs"]) != accepted_count:
            raise ValueError(f"source gamma={gamma:g} run count changed")
        config = json.loads((source / "resolved_config.json").read_text())
        loaded.append((source, manifest, config, gamma, accepted_count))

    counts_by_gamma: dict[float, int] = {}
    for _, _, _, gamma, accepted_count in loaded:
        counts_by_gamma[gamma] = (
            counts_by_gamma.get(gamma, 0) + accepted_count
        )
    gammas = sorted(counts_by_gamma)
    incomplete = {
        gamma: count for gamma, count in counts_by_gamma.items()
        if count != target
    }
    if incomplete:
        raise ValueError(
            "gamma-specific source generations do not sum to the expected "
            f"{target} runs: {incomplete}"
        )
    reference = _normalized_raw(loaded[0][2])
    for source, _, raw, _, _ in loaded[1:]:
        if _normalized_raw(raw) != reference:
            raise ValueError(f"source scientific configs differ beyond gamma: {source}")

    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "runs"
    run_dir.mkdir()
    merged_runs = []
    pair_rows = []
    source_rows = []
    seen_scene_hashes: set[str] = set()
    seen_pair_ids: set[str] = set()
    for source_index, (
        source, manifest, _, gamma, accepted_count,
    ) in enumerate(sorted(loaded, key=lambda item: (item[3], str(item[0])))):
        grouped: dict[str, list[dict]] = {}
        for row in manifest["runs"]:
            if (
                not row.get("accepted")
                or not row.get("pair_admitted")
                or not row.get("individual_accepted")
            ):
                raise ValueError(f"source contains a non-admitted run: {source}")
            if not float(row["gamma"]) == gamma:
                raise ValueError(f"source run gamma mismatch: {source}")
            grouped.setdefault(str(row["pair_id"]), []).append(row)
        if len(grouped) != accepted_count // 2 or any(
            len(rows) != 2
            or {int(row["pair_member_index"]) for row in rows} != {0, 1}
            for rows in grouped.values()
        ):
            raise ValueError(
                f"source does not contain exactly {target // 2} complete pairs: {source}"
            )
        for pair_id, rows in sorted(grouped.items()):
            merged_pair_id = pair_id
            if merged_pair_id in seen_pair_ids:
                merged_pair_id = f"source{source_index:02d}_{pair_id}"
            if merged_pair_id in seen_pair_ids:
                raise ValueError("pair_id namespace collision across sources")
            seen_pair_ids.add(merged_pair_id)
            pair_rows.append({
                "pair_id": merged_pair_id,
                "gamma": gamma,
                "member_scene_hashes": [
                    str(row["scene_hash"])
                    for row in sorted(rows, key=lambda row: row["pair_member_index"])
                ],
                "pair_admitted": True,
            })
            for row in sorted(rows, key=lambda row: row["pair_member_index"]):
                scene_hash = str(row["scene_hash"])
                if scene_hash in seen_scene_hashes:
                    raise ValueError("scene geometry was reused across merged runs")
                seen_scene_hashes.add(scene_hash)
                source_file = source / str(row["file"])
                if not source_file.is_file():
                    raise FileNotFoundError(source_file)
                destination_stem = (
                    f"g{gamma:g}_{merged_pair_id}_{row['pair_member']}_"
                    f"{scene_hash[:12]}"
                ).replace(".", "p")
                destination_name = f"{destination_stem}.npz"
                destination = run_dir / destination_name
                try:
                    os.link(source_file, destination)
                except OSError:
                    shutil.copy2(source_file, destination)
                merged = copy.deepcopy(row)
                merged["pair_id"] = merged_pair_id
                merged["source_scene_id"] = str(row["scene_id"])
                merged["source_scene_index"] = int(row["scene_index"])
                merged["scene_id"] = (
                    f"source{source_index:02d}_{row['scene_id']}"
                )
                merged["scene_index"] = len(merged_runs)
                merged["file"] = str(destination.relative_to(output))
                merged_runs.append(merged)
        source_rows.append({
            "path": str(source),
            "gamma": gamma,
            "manifest_sha256": _sha256(source / "manifest.json"),
            "accepted_runs": len(manifest["runs"]),
            "attempted_pairs": len(manifest["scene_bank"]["pairs"]),
            "admitted_pairs": sum(
                bool(row["pair_admitted"])
                for row in manifest["scene_bank"]["pairs"]
            ),
            "expert_metrics": manifest["metrics"],
        })

    resolved = copy.deepcopy(loaded[0][2])
    resolved["data"] = dict(resolved["data"])
    resolved["data"].update({
        "gammas": gammas,
        "episodes_per_gamma": target,
        "max_attempts_per_gamma": max(
            target,
            int(resolved["data"].get("max_attempts_per_gamma") or target),
        ),
    })
    resolved["domain_randomization"] = dict(
        resolved.get("domain_randomization", {})
    )
    resolved["domain_randomization"].update({
        "gamma_exclusive_scene_streams": True,
        "scene_bank_shared_across_gamma": False,
        "mirrored_pair_admission": "both_success=>2; otherwise=>0",
        "merged_gamma_archives": True,
    })
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n"
    )
    load_config(output / "resolved_config.json")

    scenes = [{
        "scene_index": int(row["scene_index"]),
        "scene_id": str(row["scene_id"]),
        "scene_seed": int(row["scene_seed"]),
        "scene_hash": str(row["scene_hash"]),
        "spheres": row["spheres"],
        "cylinders": row["cylinders"],
        "gamma": float(row["gamma"]),
        "pair_id": str(row["pair_id"]),
        "pair_member": str(row["pair_member"]),
    } for row in merged_runs]
    manifest = {
        "kind": (
            "Merged gamma-exclusive mirrored-pair randomized-cylinder "
            "SafeMPPI pretraining archive"
        ),
        "schema": MERGED_SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE_MERGED_PAIRED_SUCCESS_QUOTA",
        "config": "resolved_config.json",
        "gammas": gammas,
        "target_successes_per_gamma": target,
        "accepted_counts_by_gamma": {
            f"{gamma:.9g}": sum(
                float(row["gamma"]) == gamma for row in merged_runs
            )
            for gamma in gammas
        },
        "trajectory_count": len(merged_runs),
        "admitted_pair_count": len(pair_rows),
        "sampling_distribution": {
            "unconditioned_geometry": True,
            "gamma_exclusive_scene_streams": True,
            "scene_reuse_across_gamma": False,
            "successful_pair_training_contribution": 2,
            "failed_pair_training_contribution": 0,
            "rollout_retries_on_same_pair": 0,
        },
        "scene_bank": {
            "shared_across_gamma": False,
            "scenes": scenes,
            "pairs": pair_rows,
        },
        "sources": source_rows,
        "runs": merged_runs,
        "attempts": merged_runs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-successes-per-gamma", type=int, default=400,
    )
    args = parser.parse_args()
    manifest = merge_archives(
        args.source,
        args.output,
        expected_successes_per_gamma=args.expected_successes_per_gamma,
    )
    print(json.dumps({
        "status": manifest["status"],
        "gammas": manifest["gammas"],
        "trajectory_count": manifest["trajectory_count"],
        "admitted_pair_count": manifest["admitted_pair_count"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
