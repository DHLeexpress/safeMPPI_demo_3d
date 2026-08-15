"""Pretrain and qualify the Minhyuk-frame reference flow policy."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.expansion import mean_pairwise_lengthscale
from safe_mppi.flow_model import ConditionalFlowMLP
from safe_mppi.helios_remote import (
    add_helios_arguments,
    run_pretraining_on_helios,
)
from safe_mppi.lab_reference_flow_task import (
    LAB_RAW_CONTEXT_SCHEMA,
    LAB_REFERENCE_CONTEXT_DIM,
    LAB_ROUTE_MODES,
    lab_reference_demo_windows,
    raw_reference_rollout,
)
from safe_mppi.lab_visual_flow import (
    LAB_HP100_CHANNELS,
    LAB_HP100_DYNAMIC_FACE_COUNT,
    LAB_HP100_EXACT_MEMORY_DIM,
    LAB_HP100_EXACT_MEMORY_PACKED_DIM,
    LAB_HP100_EXACT_MEMORY_SCHEMA,
    LAB_HP100_FRAME,
    LAB_HP100_GRID_SHAPE,
    LAB_HP100_HISTORY_PACKED_DIM,
    LAB_HP100_HISTORY_SCHEMA,
    LAB_HP100_PACKED_DIM,
    LAB_HP100_PLANE_CHANNELS,
    LAB_HP100_RADIAL_EDGES,
    LAB_HP100_SCHEMA,
    LAB_RADIAL_VISUAL_CHANNELS,
    LAB_RADIAL_VISUAL_ENCODER_CHANNELS,
    LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE,
    LAB_RADIAL_VISUAL_FRAME,
    LAB_RADIAL_VISUAL_GRID_SHAPE,
    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
    LAB_RADIAL_VISUAL_PACKED_DIM,
    LAB_RADIAL_VISUAL_RADIAL_EDGES,
    LAB_RADIAL_VISUAL_SCHEMA,
    LAB_VISUAL_CHANNELS,
    LAB_VISUAL_FRAME,
    LAB_VISUAL_GRID_SHAPE,
    LAB_VISUAL_HISTORY_LENGTH,
    LAB_VISUAL_HISTORY_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_HISTORY_STEP_DIM,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    LabNonuniformRadialFlowPolicy,
    LabNonuniformRadialHistoryFlowPolicy,
    LabUniformHp100ExactMemoryFlowPolicy,
    LabUniformHp100FlowPolicy,
    LabUniformHp100HistoryFlowPolicy,
    LabVisualHistoryFlowPolicy,
    LabVisualFlowPolicy,
)
from safe_mppi.path_focused_clutter import (
    PATH_FOCUSED_DISTRIBUTIONS,
    path_focused_scene_bank,
)


ROOT = Path(__file__).resolve().parents[1]
CFM_TRAINING_RNG_OFFSET = 1_000_003


def cfm_training_rng_seed(seed: int) -> int:
    return int(seed) + CFM_TRAINING_RNG_OFFSET


def source_archive_digest(run_dir: str | Path) -> dict:
    """Hash the manifest, resolved config, and every manifest-declared run."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative_names = {"manifest.json", "resolved_config.json"}
    relative_names.update(
        str(row["file"])
        for row in manifest.get("runs", ())
        if row.get("file") is not None
    )

    digest = hashlib.sha256()
    digest.update(b"safe_mppi_declared_demo_archive_v1\0")
    for name in sorted(relative_names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"archive file must remain under demo dir: {name}")
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as error:
            raise ValueError(
                f"archive file must remain under demo dir: {name}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"archive provenance file missing: {path}")
        encoded_name = relative.as_posix().encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "algorithm": "sha256",
        "schema": "safe_mppi_declared_demo_archive_v1",
        "sha256": digest.hexdigest(),
        "file_count": len(relative_names),
    }


def split_provenance(
    metadata: list[dict],
    training_ids: torch.Tensor,
    validation_ids: torch.Tensor,
    seed: int,
) -> dict:
    """Return reproducible split metadata for randomized and legacy archives."""
    all_scene_hashes = sorted({
        str(row["scene_hash"])
        for row in metadata
        if row.get("scene_hash") is not None
    })
    observed_pair_ids = {
        str(row["pair_id"])
        for row in metadata
        if row.get("pair_id") is not None
    }
    if observed_pair_ids and any(
        row.get("pair_id") is None for row in metadata
    ):
        raise ValueError("archive mixes paired and unpaired metadata rows")

    def selected_scene_hashes(indices: torch.Tensor) -> list[str]:
        return sorted({
            str(metadata[int(index)]["scene_hash"])
            for index in indices.tolist()
            if metadata[int(index)].get("scene_hash") is not None
        })

    def selected_pair_ids(indices: torch.Tensor) -> list[str]:
        return sorted({
            str(metadata[int(index)]["pair_id"])
            for index in indices.tolist()
            if metadata[int(index)].get("pair_id") is not None
        })

    return {
        "split_seed": int(seed),
        "split_unit": (
            "gamma_exclusive_mirrored_pair"
            if observed_pair_ids else (
                "scene_sha256_across_all_gamma"
                if all_scene_hashes else "gamma_and_trajectory_seed"
            )
        ),
        "randomized_scene_count": len(all_scene_hashes),
        "training_scene_hashes": selected_scene_hashes(training_ids),
        "validation_scene_hashes": selected_scene_hashes(validation_ids),
        "mirrored_pair_count": len(observed_pair_ids),
        "training_pair_ids": selected_pair_ids(training_ids),
        "validation_pair_ids": selected_pair_ids(validation_ids),
    }


RADIAL_CONTEXT_CACHE_SCHEMA = "lab_radial_context_cache_v1"
RADIAL_CONTEXT_CACHE_FILES = (
    "contexts.npy",
    "plans.npy",
    "metadata.json",
    "training_ids.npy",
    "validation_ids.npy",
)
LEGACY_RADIAL_CONTEXT_BUILDER_IMPLEMENTATION_DIGEST = {
    "algorithm": "sha256",
    "schema": "lab_radial_context_builder_source_v2",
    "sha256": "e47491ae19c102f619e31915a46773062851d0b1bfd930930d2c856245006f0f",
    "modules": [
        "safe_mppi.lab_reference_flow_task",
        "safe_mppi.lab_visual_flow",
        "safe_mppi.geometry",
        "safe_mppi.environment",
    ],
}


def context_builder_implementation_digest() -> dict:
    """Return the sealed v2 digest without invalidating legacy mmap caches."""
    return dict(LEGACY_RADIAL_CONTEXT_BUILDER_IMPLEMENTATION_DIGEST)


def metadata_row_key_digest(metadata: list[dict]) -> dict:
    """Bind cached array order to its authoritative demonstration windows."""
    keys = []
    for index, row in enumerate(metadata):
        missing = {"scene_hash", "gamma", "seed", "t", "file"}.difference(row)
        if missing:
            raise ValueError(
                f"cache metadata row {index} lacks keys {sorted(missing)}"
            )
        keys.append([
            row["scene_hash"],
            float(row["gamma"]).hex(),
            int(row["seed"]),
            int(row["t"]),
            str(row["file"]),
        ])
    encoded = json.dumps(
        keys,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "schema": "lab_radial_context_ordered_row_keys_v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "count": len(keys),
        "fields": ["scene_hash", "gamma_hex", "seed", "t", "file"],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_radial_context_cache(
    demo_dir: str | Path,
    output: str | Path,
    *,
    split_seed: int,
    validate_archive: bool = True,
) -> dict:
    """Build one immutable history-superset cache for all radial arms."""
    demo_dir = Path(demo_dir).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite context cache {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.building-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"stale cache build directory {temporary}")
    temporary.mkdir()
    try:
        contexts, plans, metadata, _ = lab_reference_demo_windows(
            demo_dir,
            validate_archive=validate_archive,
            context_schema=LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        )
        if contexts.shape[1:] != (LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,):
            raise RuntimeError("radial-history cache context shape changed")
        if plans.shape[1:] != (10, 3) or len(plans) != len(contexts):
            raise RuntimeError("radial-history cache plan shape changed")
        training_ids, validation_ids = trajectory_split(metadata, split_seed)
        arrays = {
            "contexts.npy": np.asarray(contexts, np.float32),
            "plans.npy": np.asarray(plans, np.float32),
            "training_ids.npy": training_ids.numpy().astype(np.int64),
            "validation_ids.npy": validation_ids.numpy().astype(np.int64),
        }
        for name, values in arrays.items():
            np.save(temporary / name, values, allow_pickle=False)
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, separators=(",", ":")) + "\n"
        )
        files = {}
        for name in RADIAL_CONTEXT_CACHE_FILES:
            path = temporary / name
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        manifest = {
            "status": "LAB_RADIAL_CONTEXT_CACHE_COMPLETE",
            "schema": RADIAL_CONTEXT_CACHE_SCHEMA,
            "source_demo_dir": str(demo_dir),
            "archive_replay_validation": bool(validate_archive),
            "source_archive_digest": source_archive_digest(demo_dir),
            "resolved_config_sha256": _sha256_file(
                demo_dir / "resolved_config.json"
            ),
            "context_builder_implementation_digest": (
                context_builder_implementation_digest()
            ),
            "ordered_row_key_digest": metadata_row_key_digest(metadata),
            "stored_context_schema": LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
            "supported_context_models": ["radial_hp3d", "radial_hp3d_gru"],
            "stored_context_shape": list(contexts.shape),
            "stored_context_dtype": str(contexts.dtype),
            "plan_shape": list(plans.shape),
            "plan_dtype": str(plans.dtype),
            "grid_shape": list(LAB_RADIAL_VISUAL_GRID_SHAPE),
            "grid_channels": list(LAB_RADIAL_VISUAL_CHANNELS),
            "encoder_grid_shape": list(
                LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE
            ),
            "encoder_grid_channels": list(
                LAB_RADIAL_VISUAL_ENCODER_CHANNELS
            ),
            "grid_frame": LAB_RADIAL_VISUAL_FRAME,
            "radial_edges": list(LAB_RADIAL_VISUAL_RADIAL_EDGES),
            "history_length": LAB_VISUAL_HISTORY_LENGTH,
            "history_step_dim": LAB_VISUAL_HISTORY_STEP_DIM,
            "split_provenance": split_provenance(
                metadata,
                training_ids,
                validation_ids,
                split_seed,
            ),
            "files": files,
        }
        manifest_path = temporary / "cache_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "CACHE_COMPLETE.json").write_text(json.dumps({
            "status": "LAB_RADIAL_CONTEXT_CACHE_COMPLETE",
            "schema": RADIAL_CONTEXT_CACHE_SCHEMA,
            "cache_manifest_sha256": _sha256_file(manifest_path),
        }, indent=2) + "\n")
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_radial_context_cache(
    cache_dir: str | Path,
    demo_dir: str | Path,
    *,
    context_model: str,
    split_seed: int,
):
    """Load a verified cache as copy-on-write mmap arrays."""
    if context_model not in {"radial_hp3d", "radial_hp3d_gru"}:
        raise ValueError("radial context cache requires a radial context model")
    cache_dir = Path(cache_dir)
    demo_dir = Path(demo_dir).resolve()
    manifest_path = cache_dir / "cache_manifest.json"
    complete_path = cache_dir / "CACHE_COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"radial context cache is incomplete: {complete_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"radial context cache is incomplete: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    if complete != {
        "status": "LAB_RADIAL_CONTEXT_CACHE_COMPLETE",
        "schema": RADIAL_CONTEXT_CACHE_SCHEMA,
        "cache_manifest_sha256": _sha256_file(manifest_path),
    }:
        raise ValueError("radial context cache completion seal mismatch")
    required_contract = {
        "status": "LAB_RADIAL_CONTEXT_CACHE_COMPLETE",
        "schema": RADIAL_CONTEXT_CACHE_SCHEMA,
        "stored_context_schema": LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        "grid_shape": list(LAB_RADIAL_VISUAL_GRID_SHAPE),
        "grid_channels": list(LAB_RADIAL_VISUAL_CHANNELS),
        "encoder_grid_shape": list(
            LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE
        ),
        "encoder_grid_channels": list(
            LAB_RADIAL_VISUAL_ENCODER_CHANNELS
        ),
        "grid_frame": LAB_RADIAL_VISUAL_FRAME,
        "radial_edges": list(LAB_RADIAL_VISUAL_RADIAL_EDGES),
        "history_length": LAB_VISUAL_HISTORY_LENGTH,
        "history_step_dim": LAB_VISUAL_HISTORY_STEP_DIM,
    }
    for key, expected in required_contract.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"radial context cache {key} mismatch: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    if int(manifest["split_provenance"]["split_seed"]) != int(split_seed):
        raise ValueError("radial context cache split seed mismatch")
    if manifest.get(
        "context_builder_implementation_digest"
    ) != context_builder_implementation_digest():
        raise ValueError(
            "radial context cache context-builder implementation mismatch"
        )
    if manifest.get("source_archive_digest") != source_archive_digest(demo_dir):
        raise ValueError("radial context cache source archive mismatch")
    if manifest.get("resolved_config_sha256") != _sha256_file(
        demo_dir / "resolved_config.json"
    ):
        raise ValueError("radial context cache resolved config mismatch")
    declared_files = manifest.get("files", {})
    if set(declared_files) != set(RADIAL_CONTEXT_CACHE_FILES):
        raise ValueError("radial context cache file declaration mismatch")
    for name in RADIAL_CONTEXT_CACHE_FILES:
        path = cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"radial context cache file missing: {path}")
        declaration = declared_files[name]
        if int(declaration.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"radial context cache size mismatch: {name}")
        if declaration.get("sha256") != _sha256_file(path):
            raise ValueError(f"radial context cache hash mismatch: {name}")

    contexts = np.load(
        cache_dir / "contexts.npy",
        mmap_mode="c",
        allow_pickle=False,
    )
    plans = np.load(
        cache_dir / "plans.npy",
        mmap_mode="c",
        allow_pickle=False,
    )
    training_ids = torch.from_numpy(np.load(
        cache_dir / "training_ids.npy",
        allow_pickle=False,
    ))
    validation_ids = torch.from_numpy(np.load(
        cache_dir / "validation_ids.npy",
        allow_pickle=False,
    ))
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if (
        list(contexts.shape) != manifest["stored_context_shape"]
        or str(contexts.dtype) != manifest["stored_context_dtype"]
        or list(plans.shape) != manifest["plan_shape"]
        or str(plans.dtype) != manifest["plan_dtype"]
        or len(contexts) != len(plans)
        or len(contexts) != len(metadata)
    ):
        raise ValueError("radial context cache array contract mismatch")
    if manifest.get("ordered_row_key_digest") != metadata_row_key_digest(
        metadata
    ):
        raise ValueError("radial context cache ordered row-key mismatch")
    expected_training, expected_validation = trajectory_split(
        metadata, split_seed,
    )
    if not torch.equal(training_ids, expected_training) or not torch.equal(
        validation_ids, expected_validation,
    ):
        raise ValueError("radial context cache split contents mismatch")
    if context_model == "radial_hp3d":
        contexts = contexts[:, :LAB_RADIAL_VISUAL_PACKED_DIM]
    expected_dim = (
        LAB_RADIAL_VISUAL_PACKED_DIM
        if context_model == "radial_hp3d"
        else LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM
    )
    if contexts.shape[1] != expected_dim:
        raise ValueError("radial context cache model view mismatch")
    return (
        contexts,
        plans,
        metadata,
        load_config(demo_dir / "resolved_config.json"),
        training_ids,
        validation_ids,
        manifest,
    )


@torch.no_grad()
def pretrained_sample_calibration(
    policy,
    contexts: torch.Tensor,
    metadata: list[dict],
    training_ids: torch.Tensor,
    *,
    seed: int,
    count: int = 50,
) -> tuple[torch.Tensor, list[int]]:
    """Embed 50 pretrained-policy samples on gamma-balanced train contexts."""
    if count < 2:
        raise ValueError("RBF calibration requires at least two samples")
    training = np.asarray(training_ids.tolist(), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 1)
    gammas = sorted({
        float(metadata[int(index)]["gamma"]) for index in training
    })
    selected: list[int] = []
    base_count, remainder = divmod(int(count), len(gammas))
    for gamma_index, gamma in enumerate(gammas):
        target = base_count + int(gamma_index < remainder)
        available = training[[
            np.isclose(
                float(metadata[int(index)]["gamma"]),
                gamma,
                rtol=0.0,
                atol=1.0e-7,
            )
            for index in training
        ]]
        if len(available) < target:
            raise ValueError(
                f"RBF calibration needs {target} training contexts "
                f"for gamma={gamma:g}, found {len(available)}"
            )
        selected.extend(
            rng.choice(available, size=target, replace=False).tolist()
        )
    selected_contexts = contexts[torch.tensor(selected, dtype=torch.long)]
    generator = torch.Generator().manual_seed(int(seed) + 2)
    plans, bases = [], []
    for context in selected_contexts:
        plan, base = policy.sample_with_base(
            context, 1, generator, base_std=1.0,
        )
        plans.append(plan[0])
        bases.append(base[0])
    features = policy.embed(
        selected_contexts,
        torch.stack(plans),
        base=torch.stack(bases),
    )
    if (
        features.shape[0] != count
        or not bool(torch.isfinite(features).all())
    ):
        raise RuntimeError(
            "pretrained-policy RBF calibration produced invalid features"
        )
    return features.cpu(), selected


def trajectory_split(metadata: list[dict], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Disjoint train/validation split over whole demonstration trajectories."""
    paired_rows = [row.get("pair_id") is not None for row in metadata]
    generator = np.random.default_rng(seed)
    if any(paired_rows):
        if not all(paired_rows):
            raise ValueError("archive mixes paired and unpaired metadata rows")
        groups = sorted({
            (float(row["gamma"]), str(row["pair_id"]))
            for row in metadata
        })
        validation_groups = set()
        for gamma in sorted({group[0] for group in groups}):
            candidates = [group for group in groups if group[0] == gamma]
            if len(candidates) < 2:
                raise ValueError(
                    "paired pretraining requires at least two pairs per gamma"
                )
            count = max(1, int(round(0.1 * len(candidates))))
            chosen = generator.choice(len(candidates), size=count, replace=False)
            validation_groups.update(
                candidates[int(index)] for index in chosen
            )
        validation = [
            index for index, row in enumerate(metadata)
            if (float(row["gamma"]), str(row["pair_id"]))
            in validation_groups
        ]
        training = [
            index for index, row in enumerate(metadata)
            if (float(row["gamma"]), str(row["pair_id"]))
            not in validation_groups
        ]
        if not training or not validation:
            raise ValueError(
                "pair-disjoint split produced an empty train or validation set"
            )
        return torch.tensor(training), torch.tensor(validation)

    randomized_scenes = {
        str(row["scene_hash"])
        for row in metadata
        if row.get("scene_hash") is not None
    }
    if randomized_scenes:
        if any(row.get("scene_hash") is None for row in metadata):
            raise ValueError(
                "randomized archive mixes rows with and without scene hashes"
            )
        scenes = sorted(randomized_scenes)
        if len(scenes) < 2:
            raise ValueError(
                "randomized pretraining requires at least two distinct scenes"
            )
        count = max(1, int(round(0.1 * len(scenes))))
        chosen = generator.choice(len(scenes), size=count, replace=False)
        validation_scenes = {scenes[int(index)] for index in chosen}
        validation = [
            index for index, row in enumerate(metadata)
            if str(row.get("scene_hash")) in validation_scenes
        ]
        training = [
            index for index, row in enumerate(metadata)
            if str(row.get("scene_hash")) not in validation_scenes
        ]
        if not training or not validation:
            raise ValueError(
                "scene-disjoint split produced an empty train or validation set"
            )
        return torch.tensor(training), torch.tensor(validation)

    groups = sorted({
        (
            float(row["gamma"]),
            str(
                row.get("scene_id")
                if row.get("scene_id") is not None
                else row["seed"]
            ),
        )
        for row in metadata
    })
    validation_groups = set()
    for gamma in sorted({group[0] for group in groups}):
        candidates = [group for group in groups if group[0] == gamma]
        count = max(1, int(round(0.1 * len(candidates))))
        chosen = generator.choice(len(candidates), size=count, replace=False)
        validation_groups.update(candidates[int(index)] for index in chosen)
    validation = [
        index for index, row in enumerate(metadata)
        if (
            float(row["gamma"]),
            str(
                row.get("scene_id")
                if row.get("scene_id") is not None
                else row["seed"]
            ),
        ) in validation_groups
    ]
    training = [
        index for index, row in enumerate(metadata)
        if (
            float(row["gamma"]),
            str(
                row.get("scene_id")
                if row.get("scene_id") is not None
                else row["seed"]
            ),
        ) not in validation_groups
    ]
    return torch.tensor(training), torch.tensor(validation)


def gamma_mirror_balanced_epoch_order(
    training_ids: torch.Tensor,
    metadata: list[dict],
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Order one epoch into exact gamma-balanced, mirror-complete batches.

    Each unit contains endpoint-aligned windows from both members of one
    admitted mirrored pair.  A batch contains the same number of units from
    every gamma, then shuffles only within that already-balanced batch.
    """
    gammas = sorted({float(metadata[int(index)]["gamma"]) for index in training_ids})
    if not gammas or batch_size % (2 * len(gammas)) != 0:
        raise ValueError(
            "gamma_mirror_balanced batch_size must be divisible by "
            "2 * number_of_gammas"
        )
    ids_by_gamma_pair_member: dict[
        float, dict[str, dict[str, list[int]]]
    ] = {gamma: {} for gamma in gammas}
    for raw_index in training_ids.tolist():
        index = int(raw_index)
        row = metadata[index]
        pair_id = row.get("pair_id")
        member = row.get("pair_member")
        if pair_id is None or member not in {"source", "mirror"}:
            raise ValueError(
                "gamma_mirror_balanced requires pair_id and source/mirror "
                "metadata for every training window"
            )
        gamma = float(row["gamma"])
        ids_by_gamma_pair_member[gamma].setdefault(
            str(pair_id), {"source": [], "mirror": []},
        )[str(member)].append(index)

    units_by_gamma: dict[float, list[tuple[int, int]]] = {}
    for gamma, pairs in ids_by_gamma_pair_member.items():
        units = []
        for pair_id, members in sorted(pairs.items()):
            source = sorted(members["source"], key=lambda index: metadata[index]["t"])
            mirror = sorted(members["mirror"], key=lambda index: metadata[index]["t"])
            if len(source) != len(mirror) or not source:
                raise ValueError(
                    f"mirrored pair {pair_id} has unequal/empty window counts"
                )
            units.extend(zip(source, mirror, strict=True))
        units_by_gamma[gamma] = units
    unit_counts = {len(units) for units in units_by_gamma.values()}
    if len(unit_counts) != 1:
        raise ValueError(
            "gamma_mirror_balanced requires equal window-pair counts per gamma"
        )

    shuffled = {}
    for gamma, units in units_by_gamma.items():
        permutation = torch.randperm(len(units), generator=generator).tolist()
        shuffled[gamma] = [units[index] for index in permutation]
    units_per_gamma_batch = batch_size // (2 * len(gammas))
    order: list[int] = []
    total_units = unit_counts.pop()
    for start in range(0, total_units, units_per_gamma_batch):
        batch: list[int] = []
        for gamma in gammas:
            for unit in shuffled[gamma][start:start + units_per_gamma_batch]:
                batch.extend(unit)
        permutation = torch.randperm(len(batch), generator=generator).tolist()
        order.extend(batch[index] for index in permutation)
    if len(order) != len(training_ids) or len(set(order)) != len(order):
        raise RuntimeError("balanced epoch order lost or duplicated windows")
    return torch.tensor(order, dtype=torch.long)


def training_batch_balance_audit(
    order: torch.Tensor,
    metadata: list[dict],
    batch_size: int,
) -> dict:
    """Quantify gamma imbalance and incomplete mirror units in an epoch."""
    gammas = sorted({float(metadata[int(index)]["gamma"]) for index in order})
    gamma_spreads = []
    incomplete = 0
    pair_units = 0
    for start in range(0, len(order), batch_size):
        indices = [int(index) for index in order[start:start + batch_size]]
        counts = [
            sum(float(metadata[index]["gamma"]) == gamma for index in indices)
            for gamma in gammas
        ]
        gamma_spreads.append(max(counts) - min(counts))
        observed: dict[tuple[float, str], dict[str, int]] = {}
        for index in indices:
            row = metadata[index]
            key = (
                float(row["gamma"]),
                str(row.get("pair_id")),
            )
            counts_by_member = observed.setdefault(
                key, {"source": 0, "mirror": 0},
            )
            member = str(row.get("pair_member"))
            if member in counts_by_member:
                counts_by_member[member] += 1
        pair_units += len(observed)
        incomplete += sum(
            counts["source"] != counts["mirror"]
            for counts in observed.values()
        )
    return {
        "batch_count": int((len(order) + batch_size - 1) // batch_size),
        "maximum_gamma_count_spread": int(max(gamma_spreads, default=0)),
        "mean_gamma_count_spread": float(np.mean(gamma_spreads)),
        "incomplete_mirror_unit_fraction": (
            float(incomplete / pair_units) if pair_units else None
        ),
    }


def parameter_update_audit(
    initial_parameters: dict[str, torch.Tensor],
    policy: torch.nn.Module,
) -> dict:
    """Report selected-checkpoint movement for encoder, trunk, and head."""
    groups = {
        "visual_encoder": "grid_encoder.",
        "flow_trunk": "flow.trunk.",
        "flow_head": "flow.head.",
    }
    current = dict(policy.named_parameters())
    report = {}
    for label, prefix in groups.items():
        names = [name for name in initial_parameters if name.startswith(prefix)]
        delta_squared = 0.0
        initial_squared = 0.0
        changed = 0
        parameter_count = 0
        for name in names:
            before = initial_parameters[name].double()
            after = current[name].detach().cpu().double()
            delta = after - before
            delta_squared += float(delta.square().sum())
            initial_squared += float(before.square().sum())
            changed += int(bool((delta != 0.0).any()))
            parameter_count += before.numel()
        report[label] = {
            "tensor_count": len(names),
            "changed_tensor_count": changed,
            "parameter_count": int(parameter_count),
            "delta_l2": float(np.sqrt(delta_squared)),
            "relative_delta_l2": float(
                np.sqrt(delta_squared) / max(np.sqrt(initial_squared), 1.0e-12)
            ),
        }
    report["all_groups_changed"] = all(
        row["changed_tensor_count"] == row["tensor_count"] > 0
        for row in report.values()
        if isinstance(row, dict)
    )
    return report


def deterministic_validation_audit(
    policy: torch.nn.Module,
    contexts: torch.Tensor,
    plans: torch.Tensor,
    metadata: list[dict],
    validation_ids: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    cuda_amp: bool,
) -> dict:
    """Evaluate one checkpoint with fixed, shared CFM noise and flow times."""
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        values = []
        with torch.no_grad():
            for start in range(0, len(validation_ids), 256):
                indices = validation_ids[start:start + 256]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=bool(cuda_amp),
                ):
                    losses = policy.cfm_loss(
                        contexts[indices].to(device),
                        plans[indices].to(device),
                        reduction="none",
                    )
                values.append(losses.float().cpu())
        losses = torch.cat(values)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    by_gamma = {}
    for gamma in sorted({
        float(metadata[int(index)]["gamma"]) for index in validation_ids
    }):
        mask = torch.tensor([
            float(metadata[int(index)]["gamma"]) == gamma
            for index in validation_ids
        ], dtype=torch.bool)
        selected = losses[mask]
        by_gamma[f"{gamma:g}"] = {
            "windows": int(mask.sum()),
            "mean_cfm_loss": float(selected.mean()),
        }
    return {
        "seed": int(seed),
        "windows": len(validation_ids),
        "mean_cfm_loss": float(losses.mean()),
        "by_gamma": by_gamma,
    }


def validate_early_stopping_contract(
    *,
    epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
) -> None:
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(patience) < 0:
        raise ValueError("patience must be nonnegative")
    if not np.isfinite(min_delta) or float(min_delta) < 0.0:
        raise ValueError("min_delta must be finite and nonnegative")
    if int(min_epochs) < 0 or int(min_epochs) > int(epochs):
        raise ValueError("min_epochs must lie in [0, epochs]")


def validate_training_efficiency_contract(
    *,
    max_windows_per_trajectory: int | None,
    cuda_amp: bool,
    device: torch.device,
) -> None:
    if (
        max_windows_per_trajectory is not None
        and int(max_windows_per_trajectory) < 2
    ):
        raise ValueError(
            "max_windows_per_trajectory must be at least 2 so both trajectory "
            "endpoints remain represented"
        )
    if bool(cuda_amp) and device.type != "cuda":
        raise ValueError("CUDA AMP requires a CUDA training device")


def window_sampling_provenance(
    metadata: list[dict],
    max_windows_per_trajectory: int | None,
) -> dict:
    observed_counts: dict[str, int] = {}
    trajectories = {}
    for row in metadata:
        filename = str(row["file"])
        observed_counts[filename] = observed_counts.get(filename, 0) + 1
        if "trajectory_available_windows" not in row:
            continue
        value = (
            int(row["trajectory_available_windows"]),
            int(row["trajectory_selected_windows"]),
        )
        if filename in trajectories and trajectories[filename] != value:
            raise ValueError(
                f"inconsistent trajectory window counts for {filename}"
            )
        trajectories[filename] = value
    for filename, observed in observed_counts.items():
        trajectories.setdefault(filename, (observed, observed))
    return {
        "schema": "endpoint_stratified_per_trajectory_v1",
        "enabled": max_windows_per_trajectory is not None,
        "max_windows_per_trajectory": (
            None
            if max_windows_per_trajectory is None
            else int(max_windows_per_trajectory)
        ),
        "endpoint_inclusive": True,
        "trajectory_count": len(trajectories),
        "available_windows": sum(value[0] for value in trajectories.values()),
        "selected_windows": sum(value[1] for value in trajectories.values()),
        "all_trajectories_represented": all(
            selected > 0 for _, selected in trajectories.values()
        ),
    }


def train(
    policy,
    contexts,
    plans,
    metadata,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    recovery_path: Path | None = None,
    training_ids: torch.Tensor | None = None,
    validation_ids: torch.Tensor | None = None,
    patience: int = 0,
    min_delta: float = 0.0,
    min_epochs: int = 0,
    cuda_amp: bool = False,
    batch_sampler: str = "random",
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    warmup_epochs: int = 0,
    gradient_clip_norm: float = 0.0,
    ema_decay: float = 0.0,
    deterministic_validation_seed: int | None = None,
):
    validate_early_stopping_contract(
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        min_epochs=min_epochs,
    )
    validate_training_efficiency_contract(
        max_windows_per_trajectory=None,
        cuda_amp=cuda_amp,
        device=device,
    )
    if (training_ids is None) != (validation_ids is None):
        raise ValueError(
            "training_ids and validation_ids must be supplied together"
        )
    if training_ids is None:
        training_ids, validation_ids = trajectory_split(metadata, seed)
    else:
        expected_training, expected_validation = trajectory_split(
            metadata, seed,
        )
        if not torch.equal(training_ids, expected_training) or not torch.equal(
            validation_ids, expected_validation,
        ):
            raise ValueError(
                "cached train/validation split does not match current metadata"
            )
    if batch_sampler not in {"random", "gamma_mirror_balanced"}:
        raise ValueError("batch_sampler must be random or gamma_mirror_balanced")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be adam or adamw")
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= int(warmup_epochs) < int(epochs):
        raise ValueError("warmup_epochs must lie in [0, epochs)")
    if not np.isfinite(gradient_clip_norm) or gradient_clip_norm < 0.0:
        raise ValueError("gradient_clip_norm must be finite and nonnegative")
    if not np.isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must lie in [0,1)")
    generator = torch.Generator().manual_seed(seed)
    optimizer_type = (
        torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_type(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / float(warmup_epochs),
            end_factor=1.0,
            total_iters=int(warmup_epochs),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs) - int(warmup_epochs)),
            eta_min=0.03 * learning_rate,
        )
        schedule = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[int(warmup_epochs)],
        )
    else:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=0.03 * learning_rate,
        )
    ema_policy = copy.deepcopy(policy).to(device) if ema_decay > 0.0 else None
    if ema_policy is not None:
        ema_policy.requires_grad_(False)

    def validation_loss(model) -> float:
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        )
        if deterministic_validation_seed is not None:
            torch.manual_seed(int(deterministic_validation_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(deterministic_validation_seed))
        try:
            with torch.no_grad():
                validation_losses = []
                for start in range(0, len(validation_ids), 256):
                    indices = validation_ids[start:start + 256]
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=bool(cuda_amp),
                    ):
                        values = model.cfm_loss(
                            contexts[indices].to(device),
                            plans[indices].to(device),
                            reduction="none",
                        )
                    validation_losses.append(values.float().cpu())
                return float(torch.cat(validation_losses).mean())
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
    history = []
    best_epoch = -1
    best_validation = float("inf")
    best_state = None
    significant_reference = float("inf")
    consecutive_without_improvement = 0
    early_stop_triggered = False
    for epoch in range(epochs):
        if batch_sampler == "gamma_mirror_balanced":
            order = gamma_mirror_balanced_epoch_order(
                training_ids, metadata, batch_size, generator,
            )
        else:
            order = training_ids[
                torch.randperm(len(training_ids), generator=generator)
            ]
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bool(cuda_amp),
            ):
                loss = policy.cfm_loss(
                    contexts[indices].to(device),
                    plans[indices].to(device),
                    reduction="mean",
                )
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), gradient_clip_norm,
                )
            optimizer.step()
            if ema_policy is not None:
                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        ema_policy.parameters(), policy.parameters(), strict=True,
                    ):
                        ema_parameter.lerp_(parameter, 1.0 - ema_decay)
                    for ema_buffer, buffer in zip(
                        ema_policy.buffers(), policy.buffers(), strict=True,
                    ):
                        ema_buffer.copy_(buffer)
            losses.append(float(loss.detach()))
        schedule.step()
        online_validation = validation_loss(policy)
        ema_validation = (
            validation_loss(ema_policy) if ema_policy is not None else None
        )
        selected_model = (
            "ema"
            if ema_validation is not None and ema_validation < online_validation
            else "online"
        )
        validation = (
            float(ema_validation)
            if selected_model == "ema" else float(online_validation)
        )
        record = {
            "epoch": epoch,
            "train": float(np.mean(losses)),
            "valid": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if ema_policy is not None:
            record.update({
                "valid_online": float(online_validation),
                "valid_ema": float(ema_validation),
                "selected_model": selected_model,
            })
        history.append(record)
        if significant_reference - validation > float(min_delta):
            significant_reference = validation
            consecutive_without_improvement = 0
        else:
            consecutive_without_improvement += 1
        if validation < best_validation:
            best_epoch = epoch
            best_validation = validation
            selected_policy = ema_policy if selected_model == "ema" else policy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in selected_policy.state_dict().items()
            }
            if recovery_path is not None:
                temporary = recovery_path.with_suffix(
                    recovery_path.suffix + ".tmp"
                )
                torch.save({
                    "epoch": best_epoch,
                    "validation_loss": best_validation,
                    "selected_model": selected_model,
                    "model": best_state,
                }, temporary)
                temporary.replace(recovery_path)
        should_stop = (
            int(patience) > 0
            and epoch + 1 >= int(min_epochs)
            and consecutive_without_improvement >= int(patience)
        )
        if epoch % 50 == 0 or epoch == epochs - 1 or should_stop:
            print(
                f"epoch {epoch:4d} train {history[-1]['train']:.4f} "
                f"valid {validation:.4f}",
                flush=True,
            )
        if should_stop:
            early_stop_triggered = True
            break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    policy.load_state_dict(best_state, strict=True)
    early_stopping = {
        "enabled": int(patience) > 0,
        "triggered": early_stop_triggered,
        "patience": int(patience),
        "min_delta": float(min_delta),
        "min_epochs": int(min_epochs),
        "requested_epochs": int(epochs),
        "actual_epochs": len(history),
        "stopped_after_epoch": (
            len(history) - 1 if early_stop_triggered else None
        ),
        "consecutive_without_min_delta_improvement": (
            consecutive_without_improvement
        ),
        "significant_reference_validation": significant_reference,
        "checkpoint_selection": "absolute_minimum_validation_loss",
    }
    return (
        history,
        training_ids,
        validation_ids,
        best_epoch,
        best_validation,
        early_stopping,
    )


def audit(
    policy,
    config,
    episodes: int,
    seed0: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[list[dict], list[dict]]:
    scene_randomization = config.raw.get("scene_randomization", {})
    clutter_scenes = None
    if scene_randomization.get("enabled"):
        if scene_randomization.get("distribution") in PATH_FOCUSED_DISTRIBUTIONS:
            clutter_scenes = path_focused_scene_bank(
                config, episodes, seed=seed0,
            )
        else:
            if scene_randomization.get("obstacle_family") != "vertical_cylinders":
                raise ValueError(
                    "legacy pretraining qualification supports cylinders only"
                )
            from safe_mppi.lab_clutter import cylinder_scene_bank
            clutter_scenes = cylinder_scene_bank(
                episodes,
                seed=seed0,
                bounds=config.taskspace.bounds,
                start=np.asarray(config.taskspace.start),
                goal=np.asarray(config.taskspace.goal),
                cylinder_count=int(scene_randomization["count"]),
                cylinder_radius_m=float(scene_randomization["radius_m"]),
                min_surface_gap_m=float(
                    scene_randomization["minimum_obstacle_surface_gap_m"]
                ),
                start_surface_gap_m=float(
                    scene_randomization["minimum_start_surface_clearance_m"]
                ),
                goal_surface_gap_m=float(
                    scene_randomization["minimum_goal_surface_clearance_m"]
                ),
                boundary_surface_gap_m=float(
                    scene_randomization[
                        "minimum_taskspace_wall_surface_clearance_m"
                    ]
                ),
            )
    randomized_clutter = clutter_scenes is not None
    if clutter_scenes is not None:
        from safe_mppi.lab_clutter import config_for_scene
    rollout_kwargs = (
        {}
        if torch.device(device).type == "cpu"
        else {"device": device}
    )
    rows = []
    for gamma in config.data.gammas:
        for episode in range(episodes):
            rollout_config = (
                config_for_scene(config, clutter_scenes[episode])
                if clutter_scenes is not None else config
            )
            result = raw_reference_rollout(
                policy,
                rollout_config,
                float(gamma),
                seed0 + 37 * episode,
                **rollout_kwargs,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(episode),
                "seed": int(seed0 + 37 * episode),
                "scene_id": (
                    clutter_scenes[episode].scene_id
                    if clutter_scenes is not None else None
                ),
                "scene_hash": (
                    clutter_scenes[episode].scene_hash
                    if clutter_scenes is not None else None
                ),
                "status": result["status"],
                "mode": result["mode"],
                "window_validity": float(result["window_validity"]),
                "min_clearance_m": result["min_clearance_m"],
                "time_to_goal_s": result["time_to_goal_s"],
            })

    summaries = []
    for gamma in config.data.gammas:
        group = [row for row in rows if row["gamma"] == float(gamma)]
        successes = [row for row in group if row["status"] == "SUCCESS"]
        modes = {
            row["mode"] for row in successes if row["mode"] in LAB_ROUTE_MODES
        }
        counts = (
            None
            if randomized_clutter else {
                mode: sum(row["mode"] == mode for row in successes)
                for mode in LAB_ROUTE_MODES
            }
        )
        summaries.append({
            "gamma": float(gamma),
            "episodes": len(group),
            "SR": float(np.mean([row["status"] == "SUCCESS" for row in group])),
            "CR": float(np.mean([row["status"] == "COLLISION" for row in group])),
            "OOB": float(np.mean([row["status"] == "OOB" for row in group])),
            "timeout": float(np.mean([row["status"] == "TIMEOUT" for row in group])),
            "window_validity": float(np.mean([
                row["window_validity"] for row in group
            ])),
            "route_coverage": (
                None
                if randomized_clutter
                else len(modes) / len(LAB_ROUTE_MODES)
            ),
            "route_counts": counts,
            "successful_min_clearance_m": (
                float(np.mean([
                    row["min_clearance_m"] for row in successes
                    if row["min_clearance_m"] is not None
                ]))
                if successes else None
            ),
            "successful_time_to_goal_s": (
                float(np.mean([
                    row["time_to_goal_s"] for row in successes
                    if row["time_to_goal_s"] is not None
                ]))
                if successes else None
            ),
        })
    return rows, summaries


def plot_results(
    history,
    summaries,
    output: Path,
    ood_summaries=None,
) -> None:
    columns = 3 if ood_summaries is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(5.6 * columns, 4.1))
    train_history = [row for row in history if row.get("train") is not None]
    if train_history:
        axes[0].plot(
            [row["epoch"] for row in train_history],
            [row["train"] for row in train_history],
            label="train",
        )
    axes[0].plot(
        [row["epoch"] for row in history],
        [row["valid"] for row in history],
        label="trajectory-disjoint validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="CFM loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    def metric_bars(axis, values, title):
        x = np.arange(len(values))
        width = 0.2
        metrics = [
            ("SR", "SR"),
            ("CR", "CR"),
            ("window_validity", "Window validity"),
            ("OOB", "OOB"),
        ]
        for offset, (key, label) in enumerate(metrics):
            axis.bar(
                x + (offset - 1.5) * width,
                [row[key] for row in values],
                width,
                label=label,
            )
        axis.set_xticks(
            x,
            [rf"$\gamma={row['gamma']:g}$" for row in values],
        )
        axis.set_ylim(0.0, 1.03)
        axis.set_title(title)
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)

    metric_bars(axes[1], summaries, "Cylinder ID")
    if ood_summaries is not None:
        metric_bars(axes[2], ood_summaries, "Sphere OOD")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_pretraining_policy(
    context_model: str,
    config,
    *,
    hidden: int,
    representation_dim: int,
    grid_token_dim: int,
    history_token_dim: int,
    nfe: int,
    trunk_depth: int,
):
    common = {
        "plan_shape": (10, 3),
        "hidden": int(hidden),
        "representation_dim": int(representation_dim),
        "control_limit": config.safemppi.demo_u_max,
        "nfe": int(nfe),
        "trunk_depth": int(trunk_depth),
        "time_features": "raw1",
    }
    if context_model == "uniform_hp100":
        return LabUniformHp100FlowPolicy(
            grid_token_dim=grid_token_dim,
            **common,
        )
    if context_model == "uniform_hp100_exact_memory":
        return LabUniformHp100ExactMemoryFlowPolicy(
            grid_token_dim=grid_token_dim,
            **common,
        )
    if context_model == "uniform_hp100_gru":
        if int(history_token_dim) != 32:
            raise ValueError(
                "uniform_hp100_gru requires history-token-dim=32"
            )
        return LabUniformHp100HistoryFlowPolicy(
            grid_token_dim=grid_token_dim,
            history_token_dim=history_token_dim,
            history_length=LAB_VISUAL_HISTORY_LENGTH,
            **common,
        )
    if context_model == "radial_hp3d":
        return LabNonuniformRadialFlowPolicy(
            grid_token_dim=grid_token_dim,
            **common,
        )
    if context_model == "radial_hp3d_gru":
        if int(history_token_dim) != 32:
            raise ValueError(
                "radial_hp3d_gru requires history-token-dim=32"
            )
        return LabNonuniformRadialHistoryFlowPolicy(
            grid_token_dim=grid_token_dim,
            history_token_dim=history_token_dim,
            history_length=LAB_VISUAL_HISTORY_LENGTH,
            **common,
        )
    if context_model == "visual_hp3d":
        return LabVisualFlowPolicy(
            grid_token_dim=grid_token_dim,
            **common,
        )
    if context_model == "visual_hp3d_gru":
        return LabVisualHistoryFlowPolicy(
            grid_token_dim=grid_token_dim,
            history_token_dim=history_token_dim,
            history_length=LAB_VISUAL_HISTORY_LENGTH,
            **common,
        )
    if context_model == "raw10":
        return ConditionalFlowMLP(
            context_dim=LAB_REFERENCE_CONTEXT_DIM,
            **common,
        )
    raise ValueError(f"unsupported context model {context_model!r}")


def pretraining_arch(
    context_model: str,
    config,
    *,
    hidden: int,
    representation_dim: int,
    grid_token_dim: int,
    history_token_dim: int,
    nfe: int,
    trunk_depth: int,
) -> dict:
    arch = {
        "plan_shape": [10, 3],
        "hidden": int(hidden),
        "representation_dim": int(representation_dim),
        "control_limit": config.safemppi.demo_u_max,
        "nfe": int(nfe),
        "trunk_depth": int(trunk_depth),
        "time_features": "raw1",
    }
    if context_model in {
        "uniform_hp100",
        "uniform_hp100_exact_memory",
        "uniform_hp100_gru",
    }:
        arch.update({
            "kind": (
                LAB_HP100_EXACT_MEMORY_SCHEMA
                if context_model == "uniform_hp100_exact_memory"
                else (
                    LAB_HP100_HISTORY_SCHEMA
                    if context_model == "uniform_hp100_gru"
                    else LAB_HP100_SCHEMA
                )
            ),
            "grid_token_dim": int(grid_token_dim),
            "grid_shape": list(LAB_HP100_GRID_SHAPE),
            "grid_channels": list(LAB_HP100_CHANNELS),
            "grid_frame": LAB_HP100_FRAME,
            "radial_edges": list(LAB_HP100_RADIAL_EDGES),
            "plane_face_count": LAB_HP100_DYNAMIC_FACE_COUNT,
            "plane_row_channels": list(LAB_HP100_PLANE_CHANNELS),
        })
    elif context_model in {"radial_hp3d", "radial_hp3d_gru"}:
        arch.update({
            "kind": (
                LAB_RADIAL_VISUAL_HISTORY_SCHEMA
                if context_model == "radial_hp3d_gru"
                else LAB_RADIAL_VISUAL_SCHEMA
            ),
            "grid_token_dim": int(grid_token_dim),
            "grid_shape": list(LAB_RADIAL_VISUAL_GRID_SHAPE),
            "grid_channels": list(LAB_RADIAL_VISUAL_CHANNELS),
            "encoder_grid_shape": list(
                LAB_RADIAL_VISUAL_ENCODER_GRID_SHAPE
            ),
            "encoder_grid_channels": list(
                LAB_RADIAL_VISUAL_ENCODER_CHANNELS
            ),
            "grid_frame": LAB_RADIAL_VISUAL_FRAME,
            "radial_edges": list(LAB_RADIAL_VISUAL_RADIAL_EDGES),
        })
    elif context_model in {"visual_hp3d", "visual_hp3d_gru"}:
        arch.update({
            "kind": (
                LAB_VISUAL_HISTORY_SCHEMA
                if context_model == "visual_hp3d_gru"
                else LAB_VISUAL_SCHEMA
            ),
            "grid_token_dim": int(grid_token_dim),
            "grid_shape": list(LAB_VISUAL_GRID_SHAPE),
            "grid_channels": list(LAB_VISUAL_CHANNELS),
            "grid_frame": LAB_VISUAL_FRAME,
        })
    else:
        arch.update({
            "kind": "conditional_flow_mlp",
            "context_dim": LAB_REFERENCE_CONTEXT_DIM,
        })
    if context_model in {
        "radial_hp3d_gru",
        "uniform_hp100_gru",
        "visual_hp3d_gru",
    }:
        arch.update({
            "history_token_dim": int(history_token_dim),
            "history_length": LAB_VISUAL_HISTORY_LENGTH,
        })
    return arch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_helios_arguments(parser)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=(
            ROOT
            / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/lab_reference_flow",
    )
    parser.add_argument(
        "--recover-best-checkpoint",
        type=Path,
        help=(
            "package an interrupted training run's atomic best-model recovery "
            "checkpoint without applying any additional optimizer step"
        ),
    )
    parser.add_argument(
        "--recovered-observed-through-epoch",
        type=int,
        help=(
            "last epoch confirmed by the interrupted run log; required with "
            "--recover-best-checkpoint and recorded as provenance only"
        ),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="consecutive sub-min-delta epochs; 0 disables early stopping",
    )
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--batch-sampler",
        choices=("random", "gamma_mirror_balanced"),
        default="random",
    )
    parser.add_argument(
        "--optimizer", choices=("adam", "adamw"), default="adam",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument(
        "--deterministic-validation-seed",
        type=int,
        default=None,
        help=(
            "reuse identical flow noise/time draws at every validation epoch "
            "and for online-vs-EMA checkpoint comparison"
        ),
    )
    parser.add_argument(
        "--max-windows-per-trajectory",
        type=int,
        default=None,
        help=(
            "optional endpoint-inclusive deterministic window cap applied "
            "before context construction; default uses every window"
        ),
    )
    parser.add_argument(
        "--cuda-amp",
        action="store_true",
        help="use CUDA bfloat16 autocast for CFM training and validation",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--representation-dim", type=int, default=32)
    parser.add_argument("--grid-token-dim", type=int, default=32)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument(
        "--context-model",
        choices=(
            "raw10",
            "visual_hp3d",
            "visual_hp3d_gru",
            "radial_hp3d",
            "radial_hp3d_gru",
            "uniform_hp100",
            "uniform_hp100_exact_memory",
            "uniform_hp100_gru",
        ),
        default="raw10",
    )
    parser.add_argument("--history-token-dim", type=int, default=16)
    parser.add_argument("--trunk-depth", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--context-cache",
        type=Path,
        default=None,
        help="verified mmap cache built by build_lab_radial_context_cache.py",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-episodes", type=int, default=50)
    parser.add_argument("--audit-seed", type=int, default=91000)
    parser.add_argument(
        "--ood-config",
        type=Path,
        default=None,
        help="optional path-focused sphere config for an OOD raw audit",
    )
    parser.add_argument("--ood-audit-episodes", type=int, default=None)
    parser.add_argument("--ood-audit-seed", type=int, default=191000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.recover_best_checkpoint is None) != (
        args.recovered_observed_through_epoch is None
    ):
        parser.error(
            "--recover-best-checkpoint and "
            "--recovered-observed-through-epoch are required together"
        )
    if (
        args.recovered_observed_through_epoch is not None
        and args.recovered_observed_through_epoch < 0
    ):
        parser.error("--recovered-observed-through-epoch must be nonnegative")
    if args.helios:
        raise SystemExit(run_pretraining_on_helios(
            args,
            sys.argv[1:],
            ROOT,
            script_name=Path(__file__).name,
        ))
    validate_early_stopping_contract(
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
    )
    validate_training_efficiency_contract(
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        cuda_amp=args.cuda_amp,
        device=torch.device(args.device),
    )
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        parser.error("--weight-decay must be finite and nonnegative")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must lie in [0, epochs)")
    if (
        not np.isfinite(args.gradient_clip_norm)
        or args.gradient_clip_norm < 0.0
    ):
        parser.error("--gradient-clip-norm must be finite and nonnegative")
    if not np.isfinite(args.ema_decay) or not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay must lie in [0,1)")
    if args.context_model in {
        "uniform_hp100",
        "uniform_hp100_exact_memory",
        "uniform_hp100_gru",
    } and (
        args.grid_token_dim not in {64, 128, 256}
        or args.trunk_depth != 3
    ):
        parser.error(
            "uniform_hp100 variants require --grid-token-dim 64, 128, or 256 "
            "--trunk-depth 3"
        )
    if (
        args.context_model == "uniform_hp100_gru"
        and args.history_token_dim != 32
    ):
        parser.error("uniform_hp100_gru requires --history-token-dim 32")
    if (
        args.context_model in {
            "uniform_hp100",
            "uniform_hp100_exact_memory",
            "uniform_hp100_gru",
        }
        and args.context_cache is not None
    ):
        parser.error(
            "uniform_hp100 variants use compact plane contexts and do not "
            "accept the legacy radial mmap cache"
        )
    if (
        args.max_windows_per_trajectory is not None
        and args.context_cache is not None
    ):
        parser.error(
            "--max-windows-per-trajectory must be applied while contexts are "
            "built and cannot be combined with a prebuilt context cache"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    context_schema = {
        "raw10": LAB_RAW_CONTEXT_SCHEMA,
        "visual_hp3d": LAB_VISUAL_SCHEMA,
        "visual_hp3d_gru": LAB_VISUAL_HISTORY_SCHEMA,
        "radial_hp3d": LAB_RADIAL_VISUAL_SCHEMA,
        "radial_hp3d_gru": LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        "uniform_hp100": LAB_HP100_SCHEMA,
        "uniform_hp100_exact_memory": LAB_HP100_EXACT_MEMORY_SCHEMA,
        "uniform_hp100_gru": LAB_HP100_HISTORY_SCHEMA,
    }[args.context_model]
    cache_manifest = None
    cached_training_ids = None
    cached_validation_ids = None
    if args.context_cache is None:
        contexts_np, plans_np, metadata, config = lab_reference_demo_windows(
            args.demo_dir,
            context_schema=context_schema,
            max_windows_per_trajectory=args.max_windows_per_trajectory,
        )
    else:
        (
            contexts_np,
            plans_np,
            metadata,
            config,
            cached_training_ids,
            cached_validation_ids,
            cache_manifest,
        ) = load_radial_context_cache(
            args.context_cache,
            args.demo_dir,
            context_model=args.context_model,
            split_seed=args.seed,
        )
    archive_digest = source_archive_digest(args.demo_dir)
    sampling_provenance = window_sampling_provenance(
        metadata,
        args.max_windows_per_trajectory,
    )
    expected_context_dim = {
        "raw10": LAB_REFERENCE_CONTEXT_DIM,
        "visual_hp3d": LAB_VISUAL_PACKED_DIM,
        "visual_hp3d_gru": LAB_VISUAL_HISTORY_PACKED_DIM,
        "radial_hp3d": LAB_RADIAL_VISUAL_PACKED_DIM,
        "radial_hp3d_gru": LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
        "uniform_hp100": LAB_HP100_PACKED_DIM,
        "uniform_hp100_exact_memory": LAB_HP100_EXACT_MEMORY_PACKED_DIM,
        "uniform_hp100_gru": LAB_HP100_HISTORY_PACKED_DIM,
    }[args.context_model]
    if contexts_np.shape[1] != expected_context_dim:
        raise RuntimeError("lab reference context contract changed unexpectedly")
    contexts = torch.from_numpy(contexts_np)
    plans = torch.from_numpy(plans_np)
    policy_initialization_seed = int(args.seed)
    torch.manual_seed(args.seed)
    policy = build_pretraining_policy(
        args.context_model,
        config,
        hidden=args.hidden,
        representation_dim=args.representation_dim,
        grid_token_dim=args.grid_token_dim,
        history_token_dim=args.history_token_dim,
        nfe=args.nfe,
        trunk_depth=args.trunk_depth,
    )
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
    }
    training_random_seed = cfm_training_rng_seed(args.seed)
    torch.manual_seed(training_random_seed)
    device = torch.device(args.device)
    policy.to(device)
    print(
        f"[dataset] {len(contexts)} windows from "
        f"{len({(row['gamma'], row['seed']) for row in metadata})} trajectories",
        flush=True,
    )
    recovery_provenance = None
    if args.recover_best_checkpoint is None:
        (
            history,
            training_ids,
            validation_ids,
            best_epoch,
            best_validation,
            early_stopping,
        ) = train(
            policy,
            contexts,
            plans,
            metadata,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
            recovery_path=args.output / ".best_training_state.pt",
            training_ids=cached_training_ids,
            validation_ids=cached_validation_ids,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            cuda_amp=args.cuda_amp,
            batch_sampler=args.batch_sampler,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            gradient_clip_norm=args.gradient_clip_norm,
            ema_decay=args.ema_decay,
            deterministic_validation_seed=args.deterministic_validation_seed,
        )
        selected_checkpoint_parameter_source = history[best_epoch].get(
            "selected_model", "online",
        )
    else:
        recovery_path = args.recover_best_checkpoint.resolve()
        if not recovery_path.is_file():
            raise FileNotFoundError(recovery_path)
        recovered = torch.load(
            recovery_path, map_location="cpu", weights_only=False,
        )
        required = {"epoch", "validation_loss", "selected_model", "model"}
        if not isinstance(recovered, dict) or not required.issubset(recovered):
            raise ValueError(
                "recovery checkpoint must contain epoch, validation_loss, "
                "selected_model, and model"
            )
        best_epoch = int(recovered["epoch"])
        best_validation = float(recovered["validation_loss"])
        selected_checkpoint_parameter_source = str(recovered["selected_model"])
        if selected_checkpoint_parameter_source not in {"online", "ema"}:
            raise ValueError("recovery selected_model must be online or ema")
        if best_epoch > int(args.recovered_observed_through_epoch):
            raise ValueError(
                "recovery best epoch exceeds the last observed training epoch"
            )
        policy.load_state_dict(recovered["model"], strict=True)
        if cached_training_ids is None:
            training_ids, validation_ids = trajectory_split(metadata, args.seed)
        else:
            training_ids, validation_ids = (
                cached_training_ids, cached_validation_ids,
            )
        history = [{
            "epoch": best_epoch,
            "train": None,
            "valid": best_validation,
            "learning_rate": None,
            "selected_model": selected_checkpoint_parameter_source,
            "recovered_best_checkpoint": True,
        }]
        early_stopping = {
            "enabled": int(args.patience) > 0,
            "triggered": None,
            "interrupted": True,
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "min_epochs": int(args.min_epochs),
            "requested_epochs": int(args.epochs),
            "actual_epochs_observed_at_least": (
                int(args.recovered_observed_through_epoch) + 1
            ),
            "observed_through_epoch": int(
                args.recovered_observed_through_epoch
            ),
            "checkpoint_selection": "absolute_minimum_validation_loss",
        }
        recovery_provenance = {
            "kind": "interrupted_best_model_packaging",
            "source": str(recovery_path),
            "sha256": _sha256_file(recovery_path),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "selected_model": selected_checkpoint_parameter_source,
            "optimizer_state_available": False,
            "scheduler_state_available": False,
            "rng_state_available": False,
            "additional_optimizer_steps": 0,
        }
    training_history_path = args.output / "training_history.json"
    training_history_path.write_text(json.dumps(history, indent=2) + "\n")

    update_audit = parameter_update_audit(initial_parameters, policy)
    validation_audit = deterministic_validation_audit(
        policy,
        contexts,
        plans,
        metadata,
        validation_ids,
        seed=(
            args.deterministic_validation_seed
            if args.deterministic_validation_seed is not None
            else args.seed + 2_000_003
        ),
        device=device,
        cuda_amp=args.cuda_amp,
    )
    balance_audit = None
    if args.batch_sampler == "gamma_mirror_balanced":
        balanced_order = gamma_mirror_balanced_epoch_order(
            training_ids,
            metadata,
            args.batch_size,
            torch.Generator().manual_seed(args.seed + 71),
        )
        random_order = training_ids[torch.randperm(
            len(training_ids),
            generator=torch.Generator().manual_seed(args.seed + 71),
        )]
        balance_audit = {
            "selected_sampler": training_batch_balance_audit(
                balanced_order, metadata, args.batch_size,
            ),
            "random_permutation_reference": training_batch_balance_audit(
                random_order, metadata, args.batch_size,
            ),
        }

    policy.cpu()
    calibration, calibration_ids = pretrained_sample_calibration(
        policy,
        contexts,
        metadata,
        training_ids,
        seed=args.seed,
    )
    lengthscale = mean_pairwise_lengthscale(calibration)
    policy.to(device)
    rows, summaries = audit(
        policy,
        config,
        args.audit_episodes,
        args.audit_seed,
        device=device,
    )
    demo_manifest = json.loads(
        (args.demo_dir / "manifest.json").read_text()
    )
    training_scene_hashes = {
        str(row["scene_hash"])
        for row in demo_manifest.get("scene_bank", {}).get("scenes", [])
        if row.get("scene_hash") is not None
    }
    id_audit_scene_hashes = {
        str(row["scene_hash"])
        for row in rows if row.get("scene_hash") is not None
    }
    id_overlap = training_scene_hashes & id_audit_scene_hashes
    if id_overlap:
        raise RuntimeError(
            "raw ID audit scene bank overlaps the demonstration geometry bank"
        )
    ood_rows = None
    ood_summaries = None
    if args.ood_config is not None:
        ood_config = load_config(args.ood_config)
        ood_rows, ood_summaries = audit(
            policy,
            ood_config,
            (
                args.audit_episodes
                if args.ood_audit_episodes is None
                else args.ood_audit_episodes
            ),
            args.ood_audit_seed,
            device=device,
        )
    policy.cpu()

    arch = pretraining_arch(
        args.context_model,
        config,
        hidden=args.hidden,
        representation_dim=args.representation_dim,
        grid_token_dim=args.grid_token_dim,
        history_token_dim=args.history_token_dim,
        nfe=args.nfe,
        trunk_depth=args.trunk_depth,
    )
    history_model = args.context_model in {
        "visual_hp3d_gru",
        "radial_hp3d_gru",
        "uniform_hp100_gru",
    }
    exact_memory_model = (
        args.context_model == "uniform_hp100_exact_memory"
    )
    checkpoint = {
        "model": policy.state_dict(),
        "arch": arch,
        "contract": {
            "policy_output": "pre_smoothing_raw_acceleration_command",
            "stateful_governor_in_policy": False,
            "past_raw_action_history_in_policy": history_model,
            "exact_previous_raw_and_applied_in_policy": exact_memory_model,
            "deployment_smoothing_and_tracking": "external",
            "policy_initialization_seed": policy_initialization_seed,
            "cfm_training_rng_seed": training_random_seed,
            "cfm_rng_reset_after_policy_construction": True,
            "training_batch_sampler": args.batch_sampler,
            "checkpoint_parameter_source": (
                selected_checkpoint_parameter_source
            ),
        },
    }
    torch.save(checkpoint, args.output / "pretrained.pt")
    torch.save(calibration, args.output / "calibration_features.pt")
    calibration_contexts = contexts[
        torch.as_tensor(calibration_ids, dtype=torch.long)
    ].detach().cpu().clone()
    torch.save(
        calibration_contexts,
        args.output / "calibration_contexts.pt",
    )
    split_details = split_provenance(
        metadata,
        training_ids,
        validation_ids,
        args.seed,
    )
    manifest = {
        "kind": "lab raw-command reference-flow pretraining",
        "source_demo_dir": str(args.demo_dir.resolve()),
        "source_archive_digest": archive_digest,
        "context_model": args.context_model,
        "context_schema": context_schema,
        "external_context_dim": expected_context_dim,
        "encoded_context_dim": (
            LAB_REFERENCE_CONTEXT_DIM
            if args.context_model == "raw10"
            else (
                7
                + args.grid_token_dim
                + (args.history_token_dim if history_model else 0)
                + (
                    LAB_HP100_EXACT_MEMORY_DIM
                    if exact_memory_model else 0
                )
            )
        ),
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "past_raw_action_history_in_policy": history_model,
        "exact_previous_raw_and_applied_in_policy": exact_memory_model,
        "history_length": (
            LAB_VISUAL_HISTORY_LENGTH
            if history_model else 0
        ),
        "history_token_dim": (
            args.history_token_dim
            if history_model else 0
        ),
        "history_encoder_must_freeze_during_expansion": history_model,
        "trunk_depth": args.trunk_depth,
        "grid_token_dim": (
            args.grid_token_dim
            if args.context_model != "raw10" else 0
        ),
        "trainable_parameter_count": int(sum(
            parameter.numel() for parameter in policy.parameters()
        )),
        "context_cache": (
            None
            if cache_manifest is None
            else {
                "path": str(args.context_cache.resolve()),
                "schema": cache_manifest["schema"],
                "source_archive_digest": cache_manifest[
                    "source_archive_digest"
                ],
                "split_provenance": cache_manifest["split_provenance"],
                "archive_replay_validation": cache_manifest[
                    "archive_replay_validation"
                ],
                "files": cache_manifest["files"],
            }
        ),
        "deployment_smoothing_and_tracking": "external",
        "windows": len(contexts),
        "window_sampling": sampling_provenance,
        "training_windows": len(training_ids),
        "validation_windows": len(validation_ids),
        "epochs": args.epochs,
        "requested_epochs": args.epochs,
        "actual_epochs": (
            len(history)
            if recovery_provenance is None
            else int(args.recovered_observed_through_epoch) + 1
        ),
        "early_stopping": early_stopping,
        "batch_size": args.batch_size,
        "training_method": {
            "batch_sampler": args.batch_sampler,
            "optimizer": args.optimizer,
            "weight_decay": float(args.weight_decay),
            "warmup_epochs": int(args.warmup_epochs),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "ema_decay": float(args.ema_decay),
            "deterministic_validation_seed": (
                args.deterministic_validation_seed
            ),
            "selected_checkpoint_parameter_source": (
                selected_checkpoint_parameter_source
            ),
            "batch_balance_audit": balance_audit,
            "parameter_update_audit": update_audit,
            "selected_checkpoint_validation_audit": validation_audit,
        },
        "cuda_amp": {
            "enabled": bool(args.cuda_amp),
            "dtype": "bfloat16" if args.cuda_amp else None,
            "device_type": device.type,
        },
        "learning_rate": args.learning_rate,
        "policy_initialization_seed": policy_initialization_seed,
        "cfm_training_rng_seed": training_random_seed,
        "cfm_rng_reset_after_policy_construction": True,
        "final_train_loss": history[-1]["train"],
        "final_valid_loss": history[-1]["valid"],
        "recovery_provenance": recovery_provenance,
        "training_history_artifact": {
            "file": training_history_path.name,
            "records": len(history),
            "sha256": _sha256_file(training_history_path),
        },
        "selected_epoch": best_epoch,
        "selected_valid_loss": best_validation,
        "checkpoint_selection": "minimum_trajectory_disjoint_validation_loss",
        **split_details,
        "rbf_lengthscale": lengthscale,
        "rbf_calibration": {
            "source": "pretrained_policy_samples_on_training_contexts",
            "count": len(calibration_ids),
            "gamma_balanced": True,
            "flow_time": 0.9,
            "paired_base_noise": True,
            "context_indices": calibration_ids,
            "context_artifact": "calibration_contexts.pt",
            "context_artifact_sha256": _sha256_file(
                args.output / "calibration_contexts.pt"
            ),
            "scene_hashes": sorted({
                str(metadata[index]["scene_hash"])
                for index in calibration_ids
                if metadata[index].get("scene_hash") is not None
            }),
        },
        "raw_temperature": 1.0,
        "raw_audit_episodes_per_gamma": args.audit_episodes,
        "raw_audit_seed": args.audit_seed,
        "raw_audit_scene_overlap_count": len(id_overlap),
        "raw_audit_scene_bank_disjoint_from_demo": True,
        "raw_audit_summary": summaries,
        "raw_audit": rows,
        "ood_raw_audit_config": (
            str(args.ood_config.resolve())
            if args.ood_config is not None else None
        ),
        "ood_raw_audit_episodes_per_gamma": (
            args.audit_episodes
            if args.ood_config is not None
            and args.ood_audit_episodes is None
            else args.ood_audit_episodes
        ),
        "ood_raw_audit_seed": (
            args.ood_audit_seed if args.ood_config is not None else None
        ),
        "ood_raw_audit_summary": ood_summaries,
        "ood_raw_audit": ood_rows,
    }
    (args.output / "pretrain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    plot_results(
        history,
        summaries,
        args.output / "pretrain_qualification.png",
        ood_summaries,
    )
    (args.output / ".best_training_state.pt").unlink(missing_ok=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
