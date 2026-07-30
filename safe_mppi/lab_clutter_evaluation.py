"""Fixed-seed raw evaluation for fixed- or variable-count sphere expansion."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import torch

from .ball_flow_theta import start_goal_frame
from .committed_success_visualize import (
    COMMITTED_COLOR,
    COMMITTED_WINDOW_COLOR,
    OTHER_SUCCESS_COLOR,
    resolve_committed_success,
)
from .config import ObstacleConfig
from .environment import TaskEnvironment
from .expansion_visualize import (
    round_sigma_statistics,
    within_round_normalized_sigma,
)
from .lab_clutter import ClutterScene, start_goal_path_diagnostics
from .lab_clutter_expansion import (
    LAB_CLUTTER_SCENE_SCHEMA,
    LabClutterExpansionPolicyAdapter,
    LabClutterSphereExpansionTask,
    canonical_sphere_rows,
    scene_sha256,
    sphere_scene_spec_from_config,
)
from .lab_flow_evaluation import _validate_replay_provenance
from .lab_reference_flow_task import raw_reference_rollout
from .lab_visual_flow import (
    LAB_VISUAL_HISTORY_LENGTH,
    LAB_VISUAL_HISTORY_STEP_DIM,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_SCHEMA,
    load_lab_reference_policy,
)


LAB_CLUTTER_TASK_PROFILE = (
    "minhyuk_lab_random_three_sphere_visual_expansion"
)
PATH_FOCUSED_SPHERE_TASK_PROFILE = (
    "minhyuk_lab_path_focused_variable_sphere_visual_expansion"
)
EVALUATION_SCENE_SEED_STRIDE = 1009
START_PROBE_SCENE_SEED_OFFSET = 17
FIXED_SCENE_SEED_OFFSET = 104_729
FIXED_SCENE_ROLLOUT_SEED_OFFSET = 1_000_003
PATH_SPREAD_RESAMPLE_POINTS = 64
ONE_SIGMA_COVERAGE = 0.6826894921370859
ONE_SIGMA_Z = 1.0
BOOTSTRAP_REPLICATES = 1_000
PATH_FOCUSED_SPHERE_SCENE_SCHEMA = (
    "lab_path_focused_variable_spheres_v2"
)
SUPPORTED_SPHERE_SCENE_SCHEMAS = frozenset({
    LAB_CLUTTER_SCENE_SCHEMA,
    PATH_FOCUSED_SPHERE_SCENE_SCHEMA,
})


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_round_zero_equivalence(
    pretrained_path: str | Path,
    checkpoint_zero_path: str | Path,
) -> None:
    """Require r0 to contain the exact pretrained model state."""
    pretrained = torch.load(
        Path(pretrained_path), map_location="cpu", weights_only=False,
    )
    checkpoint_zero = torch.load(
        Path(checkpoint_zero_path), map_location="cpu", weights_only=False,
    )
    if not isinstance(pretrained, dict) or not isinstance(
        pretrained.get("model"), dict,
    ):
        raise ValueError("pretrained checkpoint lacks a model state mapping")
    if (
        not isinstance(checkpoint_zero, dict)
        or int(checkpoint_zero.get("round", -1)) != 0
        or checkpoint_zero.get("pretrained") is not True
        or not isinstance(checkpoint_zero.get("model"), dict)
    ):
        raise ValueError(
            "checkpoint_000.pt is not a declared pretrained round-zero state"
        )
    expected = pretrained["model"]
    observed = checkpoint_zero["model"]
    if list(expected) != list(observed):
        raise ValueError(
            "round-zero model keys differ from the pretrained checkpoint"
        )
    for name, expected_tensor in expected.items():
        observed_tensor = observed[name]
        if (
            not isinstance(expected_tensor, torch.Tensor)
            or not isinstance(observed_tensor, torch.Tensor)
            or expected_tensor.shape != observed_tensor.shape
            or expected_tensor.dtype != observed_tensor.dtype
            or not torch.equal(expected_tensor, observed_tensor)
        ):
            raise ValueError(
                "round-zero model tensor differs from the pretrained "
                f"checkpoint: {name}"
            )


def _evaluation_artifact_binding(args, rounds: list[int]) -> dict:
    """Bind an evaluation to the exact architecture and model-state files."""
    pretrained = Path(args.pretrain_dir) / "pretrained.pt"
    pretrain_manifest = Path(args.pretrain_dir) / "pretrain_manifest.json"
    expansion_manifest = Path(args.expansion) / "manifest.json"
    checkpoints = {
        str(int(round_i)): (
            Path(args.expansion) / f"checkpoint_{int(round_i):03d}.pt"
        )
        for round_i in rounds
    }
    required = {
        "pretrained checkpoint": pretrained,
        "pretrain manifest": pretrain_manifest,
        "expansion manifest": expansion_manifest,
        **{
            f"checkpoint round {round_i}": path
            for round_i, path in checkpoints.items()
        },
    }
    missing = [
        f"{label}: {path}" for label, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "evaluation artifact binding is incomplete: " + "; ".join(missing)
        )
    checkpoint_zero = checkpoints.get("0")
    if checkpoint_zero is None:
        raise ValueError("evaluation artifact binding requires round zero")
    _validate_round_zero_equivalence(pretrained, checkpoint_zero)
    return {
        "pretrained_checkpoint_sha256": _sha256_file(pretrained),
        "pretrain_manifest_sha256": _sha256_file(pretrain_manifest),
        "expansion_manifest_sha256": _sha256_file(expansion_manifest),
        "round_zero_model_bitwise_equal_to_pretrained": True,
        "checkpoint_sha256_by_round": {
            round_i: _sha256_file(path)
            for round_i, path in checkpoints.items()
        },
    }


def is_lab_clutter_evaluation_manifest(manifest: dict) -> bool:
    """Return whether a completed lab manifest has the clutter eval contract."""
    profile = manifest.get("task_profile")
    if profile not in {
        LAB_CLUTTER_TASK_PROFILE,
        PATH_FOCUSED_SPHERE_TASK_PROFILE,
    }:
        return False
    conditioning = manifest.get("lab_conditioning")
    if (
        not isinstance(conditioning, dict)
        or conditioning.get("context_schema") not in {
            LAB_VISUAL_SCHEMA,
            LAB_VISUAL_HISTORY_SCHEMA,
        }
    ):
        raise ValueError(
            "sphere-clutter expansion requires the visual lab context schema"
        )
    if conditioning["context_schema"] == LAB_VISUAL_HISTORY_SCHEMA:
        history = conditioning.get("history_encoder")
        if (
            not isinstance(history, dict)
            or history.get("present") is not True
            or not isinstance(
                history.get("frozen_during_expansion"), bool,
            )
            or not isinstance(
                history.get("explicit_unfreeze_flag"), bool,
            )
            or history["explicit_unfreeze_flag"]
            == history["frozen_during_expansion"]
        ):
            raise ValueError(
                "visual-history expansion requires an explicit, consistent "
                "GRU freeze contract"
            )
    ledger = manifest.get("lab_scene_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError(
            "sphere-clutter expansion manifest requires a nonempty scene ledger"
        )
    schemas = {
        row.get("schema")
        for row in ledger
        if isinstance(row, dict)
    }
    expected_schema = (
        LAB_CLUTTER_SCENE_SCHEMA
        if profile == LAB_CLUTTER_TASK_PROFILE
        else PATH_FOCUSED_SPHERE_SCENE_SCHEMA
    )
    if (
        len(schemas) != 1
        or schemas != {expected_schema}
        or not schemas.issubset(SUPPORTED_SPHERE_SCENE_SCHEMAS)
        or any(not isinstance(row, dict) for row in ledger)
    ):
        raise ValueError(
            "sphere-clutter expansion scene ledger schema mismatch"
        )
    return True


def _scene_record(
    scene_spec,
    env: TaskEnvironment,
    *,
    scene_seed: int,
    episode: int | None = None,
) -> dict:
    spheres = scene_spec.sample(env, int(scene_seed))
    scene_hash = scene_sha256(env, spheres)
    scene = ClutterScene(
        index=int(scene_seed),
        seed=int(scene_seed),
        spheres=tuple(tuple(map(float, row)) for row in spheres),
        cylinders=(),
        scene_hash=scene_hash,
    )
    row = {
        "scene_seed": int(scene_seed),
        "scene_hash": scene_hash,
        "obstacle_count": int(len(spheres)),
        "spheres": spheres.tolist(),
        "start_goal_path_diagnostics": start_goal_path_diagnostics(
            scene,
            start=env.start,
            goal=env.goal,
            soft_clearance_target_m=env.mppi.soft_clearance_target,
        ),
    }
    if episode is not None:
        row["episode"] = int(episode)
    return row


def _fixed_evaluation_scene_bank(
    config,
    episodes: int,
    domain_seed: int,
) -> dict:
    """Materialize every randomized evaluation scene before loading a policy."""
    if int(episodes) < 1:
        raise ValueError("evaluation episodes must be positive")
    scene_spec = sphere_scene_spec_from_config(config)
    env = TaskEnvironment(config)
    scenes = [
        _scene_record(
            scene_spec,
            env,
            episode=episode,
            scene_seed=(
                int(domain_seed)
                + EVALUATION_SCENE_SEED_STRIDE * episode
            ),
        )
        for episode in range(int(episodes))
    ]
    start_probe_scene = _scene_record(
        scene_spec,
        env,
        scene_seed=int(domain_seed) + START_PROBE_SCENE_SEED_OFFSET,
    )
    return {
        "schema": scene_spec.scene_schema,
        "evaluation_seed": int(domain_seed),
        "configured_sampler_domain_seed": int(scene_spec.domain_seed),
        "sampler": {
            "implementation": f"{type(scene_spec).__name__}.sample",
            "rng": (
                "numpy.random.default_rng("
                "SeedSequence([configured_sampler_domain_seed, scene_seed]))"
            ),
            "obstacle_family": "spheres",
            "count": (
                int(scene_spec.max_count)
                if scene_spec.scene_schema == LAB_CLUTTER_SCENE_SCHEMA
                else None
            ),
            "count_min": int(
                getattr(getattr(scene_spec, "spec", None), "count_min",
                        scene_spec.max_count)
            ),
            "count_max": int(scene_spec.max_count),
            "radius_m": float(scene_spec.radius),
            "minimum_obstacle_surface_gap_m": float(
                scene_spec.minimum_surface_margin
            ),
            "minimum_start_goal_surface_clearance_m": float(
                scene_spec.endpoint_margin
            ),
            "minimum_taskspace_wall_surface_clearance_m": float(
                scene_spec.boundary_surface_margin
            ),
            "max_attempts": int(scene_spec.max_attempts),
        },
        "rollout_scene_seed_formula": (
            f"domain_seed + {EVALUATION_SCENE_SEED_STRIDE} * episode"
        ),
        "start_probe_scene_seed_formula": (
            f"domain_seed + {START_PROBE_SCENE_SEED_OFFSET}"
        ),
        "shared_across_rounds": True,
        "shared_across_gamma": True,
        "scenes": scenes,
        "start_probe_scene": start_probe_scene,
    }


def _expansion_scene_hashes(manifest: dict) -> set[str]:
    ledger = manifest.get("lab_scene_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError(
            "cannot establish evaluation disjointness without lab_scene_ledger"
        )
    hashes = set()
    for row in ledger:
        if not isinstance(row, dict):
            raise ValueError("lab_scene_ledger rows must be objects")
        value = row.get("scene_hash", row.get("sha256"))
        if not isinstance(value, str) or not value:
            raise ValueError(
                "every lab_scene_ledger row requires scene_hash or sha256"
            )
        hashes.add(value)
    return hashes


def _evaluation_scene_provenance(
    config,
    episodes: int,
    domain_seed: int,
    manifest: dict,
) -> dict:
    bank = _fixed_evaluation_scene_bank(config, episodes, domain_seed)
    evaluation_hashes = {
        row["scene_hash"] for row in bank["scenes"]
    }
    evaluation_hashes.add(bank["start_probe_scene"]["scene_hash"])
    expansion_hashes = _expansion_scene_hashes(manifest)
    overlap = sorted(evaluation_hashes & expansion_hashes)
    if overlap:
        raise ValueError(
            "evaluation scene bank overlaps expansion lab_scene_ledger: "
            + ", ".join(overlap)
        )
    return {
        **bank,
        "evaluation_unique_scene_count": len(evaluation_hashes),
        "expansion_unique_scene_count": len(expansion_hashes),
        "overlap_count": 0,
    }


def _scene_config(config, spheres: np.ndarray):
    obstacles = ObstacleConfig(
        spheres=tuple(tuple(map(float, row)) for row in spheres),
        cylinders=(),
    )
    return replace(config, obstacles=obstacles)


def _arc_length_resample(
    states: np.ndarray,
    count: int = PATH_SPREAD_RESAMPLE_POINTS,
) -> np.ndarray:
    """Resample one xyz path at common normalized arc-length coordinates."""
    path = np.asarray(states, float)
    if path.ndim != 2 or path.shape[1] < 3 or len(path) < 1:
        raise ValueError("trajectory states must have shape [T,>=3] with T>=1")
    if int(count) < 2 or not bool(np.isfinite(path[:, :3]).all()):
        raise ValueError("path resampling needs finite xyz values and count>=2")
    xyz = path[:, :3]
    if len(xyz) == 1:
        return np.repeat(xyz, int(count), axis=0)
    arc = np.concatenate([
        np.zeros(1, float),
        np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)),
    ])
    keep = np.concatenate([[True], np.diff(arc) > 1.0e-12])
    arc = arc[keep]
    xyz = xyz[keep]
    if len(arc) == 1 or arc[-1] <= 1.0e-12:
        return np.repeat(xyz[:1], int(count), axis=0)
    arc /= arc[-1]
    grid = np.linspace(0.0, 1.0, int(count))
    return np.column_stack([
        np.interp(grid, arc, xyz[:, coordinate])
        for coordinate in range(3)
    ])


def successful_path_spread(
    rows: list[dict],
    count: int = PATH_SPREAD_RESAMPLE_POINTS,
) -> float | None:
    """Mean pairwise RMS distance between successful arc-length-resampled paths."""
    paths = [
        _arc_length_resample(row["states"], count)
        for row in rows
        if row.get("status") == "SUCCESS" and row.get("states") is not None
    ]
    if len(paths) < 2:
        return None
    pairwise = []
    for first in range(len(paths)):
        for second in range(first + 1, len(paths)):
            difference = paths[first] - paths[second]
            pairwise.append(float(np.sqrt(np.mean(np.sum(
                difference * difference, axis=1,
            )))))
    return float(np.mean(pairwise))


def _resampling_row_token(row: dict) -> str:
    return "|".join([
        str(row.get("scene_hash", "")),
        str(row.get("rollout_seed", row.get("episode", ""))),
        f"{float(row.get('gamma', 0.0)):.9g}",
        str(row.get("status", "")),
    ])


def _stable_resampling_seed(rows: list[dict], label: str) -> int:
    """Derive an order-independent deterministic seed from the evaluation cell."""
    tokens = sorted(_resampling_row_token(row) for row in rows)
    digest = hashlib.sha256(
        (str(label) + "\n" + "\n".join(tokens)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _wilson_one_sigma(successes: int, count: int) -> dict:
    if int(count) < 1:
        return {
            "lower": None,
            "upper": None,
            "n": 0,
            "method": "wilson_z1",
        }
    count = int(count)
    successes = int(successes)
    proportion = successes / count
    denominator = 1.0 + ONE_SIGMA_Z * ONE_SIGMA_Z / count
    center = (
        proportion + ONE_SIGMA_Z * ONE_SIGMA_Z / (2.0 * count)
    ) / denominator
    radius = (
        ONE_SIGMA_Z
        * np.sqrt(
            proportion * (1.0 - proportion) / count
            + ONE_SIGMA_Z * ONE_SIGMA_Z / (4.0 * count * count)
        )
        / denominator
    )
    return {
        "lower": float(max(0.0, center - radius)),
        "upper": float(min(1.0, center + radius)),
        "n": count,
        "method": "wilson_z1",
    }


def _bootstrap_mean_one_sigma(
    values,
    *,
    rows: list[dict],
    label: str,
) -> dict:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(rows):
        raise ValueError("bootstrap values and trajectory rows must align")
    ordered = sorted(
        zip(rows, values),
        key=lambda pair: (_resampling_row_token(pair[0]), float(pair[1])),
    )
    rows = [pair[0] for pair in ordered if np.isfinite(pair[1])]
    values = np.asarray([
        pair[1] for pair in ordered if np.isfinite(pair[1])
    ], dtype=float)
    count = len(values)
    if count < 1:
        return {
            "lower": None,
            "upper": None,
            "n": 0,
            "method": "trajectory_bootstrap_central_68.27pct",
            "replicates": BOOTSTRAP_REPLICATES,
        }
    rng = np.random.default_rng(_stable_resampling_seed(rows, label))
    indices = rng.integers(
        0, count, size=(BOOTSTRAP_REPLICATES, count),
    )
    samples = values[indices].mean(axis=1)
    tail = 0.5 * (1.0 - ONE_SIGMA_COVERAGE)
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "n": int(count),
        "method": "trajectory_bootstrap_central_68.27pct",
        "replicates": BOOTSTRAP_REPLICATES,
    }


def _bootstrap_path_spread_one_sigma(rows: list[dict]) -> dict:
    success_rows = [
        row for row in rows
        if row.get("status") == "SUCCESS" and row.get("states") is not None
    ]
    success_rows.sort(key=_resampling_row_token)
    count = len(success_rows)
    empty = {
        "lower": None,
        "upper": None,
        "n": int(count),
        "method": "trajectory_bootstrap_central_68.27pct",
        "replicates": BOOTSTRAP_REPLICATES,
    }
    if count < 2:
        return empty
    paths = np.asarray([
        _arc_length_resample(row["states"], PATH_SPREAD_RESAMPLE_POINTS)
        for row in success_rows
    ])
    differences = paths[:, None] - paths[None]
    distances = np.sqrt(np.mean(np.sum(
        differences * differences, axis=-1,
    ), axis=-1))
    triangle = np.triu_indices(count, 1)
    rng = np.random.default_rng(
        _stable_resampling_seed(success_rows, "successful_path_spread_m")
    )
    samples = np.empty(BOOTSTRAP_REPLICATES, float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, count, size=count)
        samples[replicate] = float(
            distances[np.ix_(selected, selected)][triangle].mean()
        )
    tail = 0.5 * (1.0 - ONE_SIGMA_COVERAGE)
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        **empty,
        "lower": float(lower),
        "upper": float(upper),
    }


def _fixed_scene_provenance(
    config,
    scene_seed: int,
    manifest: dict,
    randomized_scene_bank: dict,
) -> dict:
    """Preregister one scene disjoint from gathering and randomized evaluation."""
    scene_spec = sphere_scene_spec_from_config(config)
    scene = _scene_record(
        scene_spec,
        TaskEnvironment(config),
        scene_seed=int(scene_seed),
    )
    expansion_hashes = _expansion_scene_hashes(manifest)
    randomized_hashes = {
        row["scene_hash"] for row in randomized_scene_bank["scenes"]
    }
    randomized_hashes.add(
        randomized_scene_bank["start_probe_scene"]["scene_hash"]
    )
    if scene["scene_hash"] in expansion_hashes:
        raise ValueError(
            "fixed visualization scene overlaps expansion lab_scene_ledger"
        )
    if scene["scene_hash"] in randomized_hashes:
        raise ValueError(
            "fixed visualization scene overlaps randomized evaluation bank"
        )
    return {
        "schema": scene_spec.scene_schema,
        "preregistered_before_checkpoint_evaluation": True,
        "shared_across_rounds": True,
        "shared_across_gamma": True,
        "scene": scene,
        "expansion_overlap_count": 0,
        "randomized_evaluation_overlap_count": 0,
    }


def _write_fixed_scene_config(config, scene: dict, output: Path) -> dict:
    """Materialize the preregistered evaluation map as a deployment config."""
    payload = copy.deepcopy(config.raw)
    payload["obstacles"] = {
        "spheres": scene["spheres"],
        "cylinders": [],
    }
    randomization = dict(payload.get("scene_randomization", {}))
    randomization["enabled"] = False
    payload["scene_randomization"] = randomization
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return {
        "file": output.name,
        "sha256": _sha256_file(output),
        "scene_hash": str(scene["scene_hash"]),
    }


def _checkpoint_policy(
    pretrain_dir: Path,
    expansion: Path,
    round_i: int,
    *,
    device: str | torch.device = "cpu",
):
    policy = load_lab_reference_policy(pretrain_dir / "pretrained.pt")
    payload = torch.load(
        expansion / f"checkpoint_{round_i:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(device).eval()


def _summarize(
    rows: list[dict],
    *,
    path_spread_domain: str | None = None,
) -> dict:
    success = [row for row in rows if row["status"] == "SUCCESS"]
    clearance_rows = [
        row for row in success if row["min_clearance_m"] is not None
    ]
    clearance = [row["min_clearance_m"] for row in clearance_rows]
    time_rows = [
        row for row in success if row["time_to_goal_s"] is not None
    ]
    times = [row["time_to_goal_s"] for row in time_rows]
    count = len(rows)
    statuses = ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
    status_keys = ("SR", "CR", "OOB", "timeout")
    one_sigma = {
        key: _wilson_one_sigma(
            sum(row["status"] == status for row in rows), count,
        )
        for key, status in zip(status_keys, statuses)
    }
    one_sigma["window_validity"] = _bootstrap_mean_one_sigma(
        [row["window_validity"] for row in rows],
        rows=rows,
        label="window_validity",
    )
    one_sigma["successful_min_clearance_m"] = _bootstrap_mean_one_sigma(
        clearance,
        rows=clearance_rows,
        label="successful_min_clearance_m",
    )
    one_sigma["successful_time_to_goal_s"] = _bootstrap_mean_one_sigma(
        times,
        rows=time_rows,
        label="successful_time_to_goal_s",
    )
    one_sigma["successful_path_spread_m"] = (
        _bootstrap_path_spread_one_sigma(rows)
        if path_spread_domain is not None
        else {
            "lower": None,
            "upper": None,
            "n": len(success),
            "method": "not_applicable_cross_scene_or_gamma",
        }
    )
    return {
        "episodes": count,
        "SR": float(np.mean([
            row["status"] == "SUCCESS" for row in rows
        ])) if rows else None,
        "CR": float(np.mean([
            row["status"] == "COLLISION" for row in rows
        ])) if rows else None,
        "OOB": float(np.mean([
            row["status"] == "OOB" for row in rows
        ])) if rows else None,
        "timeout": float(np.mean([
            row["status"] == "TIMEOUT" for row in rows
        ])) if rows else None,
        "window_validity": float(np.mean([
            row["window_validity"] for row in rows
        ])) if rows else None,
        "successful_min_clearance_m": (
            float(np.mean(clearance)) if clearance else None
        ),
        "successful_time_to_goal_s": (
            float(np.mean(times)) if times else None
        ),
        # A distance between paths from different obstacle scenes or gamma
        # conditions is not a coverage statistic.  Keep the field explicit but
        # unavailable outside one fixed-scene, single-gamma cell.
        "successful_path_spread_m": (
            successful_path_spread(rows)
            if path_spread_domain is not None else None
        ),
        "successful_path_spread_domain": (
            path_spread_domain
            if path_spread_domain is not None
            else "not_applicable_cross_scene_or_gamma"
        ),
        "one_sigma": one_sigma,
    }


def _row_obstacle_count(row: dict) -> int:
    value = row.get("obstacle_count")
    if value is None:
        value = len(row.get("spheres", ()))
    value = int(value)
    if value < 1:
        raise ValueError("every clutter evaluation row needs an obstacle count")
    return value


def _raw_rows(
    policy,
    config,
    gammas,
    scene_bank,
    domain_seed: int,
    *,
    device: str | torch.device = "cpu",
):
    rollout_kwargs = {"sampling_temperature": 1.0}
    if torch.device(device).type != "cpu":
        rollout_kwargs["device"] = device
    rows = []
    for gamma in gammas:
        for scene in scene_bank:
            episode = int(scene["episode"])
            rollout_seed = int(domain_seed) + 37 * episode
            spheres = np.asarray(scene["spheres"], np.float32)
            scene_config = _scene_config(config, spheres)
            result = raw_reference_rollout(
                policy,
                scene_config,
                float(gamma),
                rollout_seed,
                **rollout_kwargs,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(episode),
                "rollout_seed": int(rollout_seed),
                "scene_seed": int(scene["scene_seed"]),
                "scene_hash": str(scene["scene_hash"]),
                "obstacle_count": int(len(spheres)),
                "spheres": spheres.tolist(),
                **result,
            })
    return rows


def _fixed_scene_rows(
    policy,
    config,
    gammas,
    scene: dict,
    rollouts: int,
    seed: int,
    *,
    device: str | torch.device = "cpu",
):
    """Independent temperature-1 raw rollouts in one preregistered scene."""
    if int(rollouts) < 1:
        raise ValueError("--fixed-scene-rollouts must be positive")
    spheres = np.asarray(scene["spheres"], np.float32)
    scene_config = _scene_config(config, spheres)
    rollout_kwargs = {"sampling_temperature": 1.0}
    if torch.device(device).type != "cpu":
        rollout_kwargs["device"] = device
    rows = []
    for gamma_index, gamma in enumerate(gammas):
        for rollout in range(int(rollouts)):
            rollout_seed = (
                int(seed)
                + FIXED_SCENE_ROLLOUT_SEED_OFFSET
                + 10_007 * gamma_index
                + 37 * rollout
            )
            result = raw_reference_rollout(
                policy,
                scene_config,
                float(gamma),
                rollout_seed,
                **rollout_kwargs,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(rollout),
                "rollout_seed": int(rollout_seed),
                "scene_seed": int(scene["scene_seed"]),
                "scene_hash": str(scene["scene_hash"]),
                "obstacle_count": int(len(spheres)),
                "spheres": spheres.tolist(),
                **result,
            })
    return rows


@torch.no_grad()
def _start_probe(
    policy,
    config,
    gammas,
    samples: int,
    scene: dict,
    *,
    device: str | torch.device = "cpu",
):
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=policy.context_schema,
        device=device,
        tight_corridor=True,
    )
    wrapped = LabClutterExpansionPolicyAdapter(
        policy, verifier_suffix_dim=task.verifier_suffix_dim,
    )
    rows = []
    for gamma_index, gamma in enumerate(gammas):
        scene_seed = int(scene["scene_seed"])
        state = task.reset(float(gamma), 0, scene_seed)
        if state["scene_hash"] != scene["scene_hash"]:
            raise RuntimeError("start-probe scene does not match the fixed scene bank")
        context = task.context(state, float(gamma))
        generator = torch.Generator(device=torch.device(device)).manual_seed(
            scene_seed + 7919 * gamma_index
        )
        candidates = wrapped.sample(
            context, int(samples), generator, base_std=1.0,
        )
        verified = task.verify(context, candidates, float(gamma))
        rows.extend({
            "gamma": float(gamma),
            "sample": int(index),
            "valid": bool(result.valid),
            "margin": float(result.margin),
            "scene_hash": state["scene_hash"],
        } for index, result in enumerate(verified))
    return rows


def _plot_curves(summaries: dict, gammas: list[float], output: Path):
    rounds = sorted(map(int, summaries))
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(gammas) - 1, 1)
        )
        for index, gamma in enumerate(gammas)
    }
    specs = (
        ("CR", "Collision rate"),
        ("window_validity", "Validity"),
        ("successful_min_clearance_m", "Min. clearance [m]"),
        ("successful_time_to_goal_s", "Time-to-goal [s]"),
    )
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 18,
        "axes.labelsize": 15,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for axis, (key, title) in zip(axes.flat, specs):
        for gamma in gammas:
            values = [
                summaries[str(round_i)]["per_gamma"][f"{gamma:g}"][key]
                for round_i in rounds
            ]
            intervals = [
                summaries[str(round_i)]["per_gamma"][f"{gamma:g}"][
                    "one_sigma"
                ][key]
                for round_i in rounds
            ]
            axis.plot(
                rounds, values, marker="o", color=colors[gamma],
                label=rf"$\gamma={gamma:g}$",
            )
            lower = [
                interval["lower"] if interval["lower"] is not None else np.nan
                for interval in intervals
            ]
            upper = [
                interval["upper"] if interval["upper"] is not None else np.nan
                for interval in intervals
            ]
            axis.fill_between(
                rounds, lower, upper, color=colors[gamma], alpha=0.14,
                linewidth=0.0,
            )
        axis.set_title(title)
        axis.set_xlabel("Expansion round")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_count_curves(
    summaries: dict,
    obstacle_counts: list[int],
    output: Path,
):
    """Plot randomized-domain metrics stratified by sphere count."""
    rounds = sorted(map(int, summaries))
    cmap = plt.get_cmap("viridis")
    colors = {
        count: cmap(
            0.12 + 0.76 * index / max(len(obstacle_counts) - 1, 1)
        )
        for index, count in enumerate(obstacle_counts)
    }
    specs = (
        ("CR", "Collision rate"),
        ("window_validity", "Validity"),
        ("successful_min_clearance_m", "Min. clearance [m]"),
        ("successful_time_to_goal_s", "Time-to-goal [s]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for axis, (key, title) in zip(axes.flat, specs):
        for count in obstacle_counts:
            cells = [
                summaries[str(round_i)]["per_obstacle_count"].get(str(count))
                for round_i in rounds
            ]
            values = [
                (
                    cell[key]
                    if cell is not None and cell[key] is not None
                    else np.nan
                )
                for cell in cells
            ]
            lower = [
                (
                    cell["one_sigma"][key]["lower"]
                    if cell is not None
                    and cell["one_sigma"][key]["lower"] is not None
                    else np.nan
                )
                for cell in cells
            ]
            upper = [
                (
                    cell["one_sigma"][key]["upper"]
                    if cell is not None
                    and cell["one_sigma"][key]["upper"] is not None
                    else np.nan
                )
                for cell in cells
            ]
            axis.plot(
                rounds, values, marker="o", color=colors[count],
                label=rf"$N={count}$",
            )
            axis.fill_between(
                rounds, lower, upper, color=colors[count], alpha=0.14,
                linewidth=0.0,
            )
        axis.set_title(title)
        axis.set_xlabel("Expansion round")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _draw_sphere(axis, sphere, physical_radius):
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    x = sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v))
    y = sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v))
    z = sphere[2] + sphere[3] * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(
        x, y, z, color="#9da3a6", linewidth=0.35, alpha=0.5,
    )
    x = sphere[0] + physical_radius * np.outer(np.cos(u), np.sin(v))
    y = sphere[1] + physical_radius * np.outer(np.sin(u), np.sin(v))
    z = sphere[2] + physical_radius * np.outer(
        np.ones_like(u), np.cos(v)
    )
    axis.plot_wireframe(
        x, y, z, color="#5c6266", linewidth=0.5, alpha=0.78,
    )


def _plot_gallery(per_round_rows, rounds, gammas, config, output: Path):
    physical_radius = float(
        config.raw["scene_randomization"]["physical_radius_m"]
    )
    fig = plt.figure(figsize=(4.2 * len(gammas), 3.8 * len(rounds)))
    for row_index, round_i in enumerate(rounds):
        for column_index, gamma in enumerate(gammas):
            axis = fig.add_subplot(
                len(rounds),
                len(gammas),
                row_index * len(gammas) + column_index + 1,
                projection="3d",
            )
            record = next(
                row for row in per_round_rows[round_i]
                if row["gamma"] == gamma and row["episode"] == 0
            )
            for sphere in record["spheres"]:
                _draw_sphere(
                    axis,
                    np.asarray(sphere, float),
                    physical_radius,
                )
            states = np.asarray(record["states"], float)
            axis.plot(
                states[:, 0], states[:, 1], states[:, 2],
                color="#1076a8", linewidth=2.0,
            )
            if record["status"] != "SUCCESS":
                axis.scatter(
                    *states[-1, :3], marker="x", s=50, color="#c8321b",
                )
            axis.scatter(
                *config.taskspace.start[:3], marker="s", s=24, color="black",
            )
            axis.scatter(
                *config.taskspace.goal, marker="*", s=80, color="#f1c40f",
                edgecolor="black",
            )
            axis.set(
                xlim=config.taskspace.bounds[0],
                ylim=config.taskspace.bounds[1],
                zlim=config.taskspace.bounds[2],
            )
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$")
            if column_index == 0:
                axis.text2D(
                    -0.18, 0.5, f"round {round_i}",
                    transform=axis.transAxes, rotation=90,
                    va="center", fontsize=13,
                )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_fixed_scene_gallery(
    per_round_rows,
    summaries,
    rounds,
    gammas,
    config,
    scene,
    output: Path,
):
    """Overlay M independent raw rollouts in one scene shared by every cell."""
    physical_radius = float(
        config.raw["scene_randomization"]["physical_radius_m"]
    )
    spheres = np.asarray(scene["spheres"], float)
    fig = plt.figure(figsize=(4.5 * len(gammas), 4.0 * len(rounds)))
    for row_index, round_i in enumerate(rounds):
        for column_index, gamma in enumerate(gammas):
            axis = fig.add_subplot(
                len(rounds),
                len(gammas),
                row_index * len(gammas) + column_index + 1,
                projection="3d",
            )
            for sphere in spheres:
                _draw_sphere(axis, sphere, physical_radius)
            records = [
                row for row in per_round_rows[int(round_i)]
                if float(row["gamma"]) == float(gamma)
            ]
            for record in records:
                states = np.asarray(record["states"], float)
                success = record["status"] == "SUCCESS"
                axis.plot(
                    states[:, 0], states[:, 1], states[:, 2],
                    color=("#1076a8" if success else "#7d838b"),
                    linewidth=(1.25 if success else 0.75),
                    alpha=(0.62 if success else 0.30),
                )
                if not success:
                    axis.scatter(
                        *states[-1, :3], marker="x", s=20,
                        color="#c8321b", alpha=0.65,
                    )
            axis.scatter(
                *config.taskspace.start[:3], marker="s", s=25, color="black",
            )
            axis.scatter(
                *config.taskspace.goal, marker="*", s=85, color="#f1c40f",
                edgecolor="black",
            )
            axis.set(
                xlim=config.taskspace.bounds[0],
                ylim=config.taskspace.bounds[1],
                zlim=config.taskspace.bounds[2],
            )
            cell = summaries[str(int(round_i))]["per_gamma"][f"{gamma:g}"]
            spread = cell["successful_path_spread_m"]
            axis.set_title(
                rf"$\gamma={gamma:g}$ | SR={cell['SR']:.2f} | "
                + (
                    rf"spread={spread:.3f} m"
                    if spread is not None else "spread=n/a"
                ),
                fontsize=10,
            )
            if column_index == 0:
                axis.text2D(
                    -0.18, 0.5, f"round {round_i}",
                    transform=axis.transAxes, rotation=90,
                    va="center", fontsize=13,
                )
    fig.suptitle(
        "Preregistered fixed-scene raw temperature-1 rollouts "
        "(separate from randomized-domain metrics)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _local_path(path: np.ndarray, env: TaskEnvironment, frame: np.ndarray):
    return (np.asarray(path, float) - env.start[:3]) @ frame


def _decode_event_scene(event: dict, task: LabClutterSphereExpansionTask):
    """Recover exact governor memory and geometry from a clutter event."""
    context = np.asarray(event.get("context"), np.float32).reshape(-1)
    history_dim = (
        LAB_VISUAL_HISTORY_LENGTH * LAB_VISUAL_HISTORY_STEP_DIM
        if task.context_schema == LAB_VISUAL_HISTORY_SCHEMA else 0
    )
    governor_start = 7 + history_dim
    scene_start = governor_start + 6
    expected = scene_start + int(task.scene_spec.packed_dim)
    if len(context) != expected:
        raise ValueError(
            "clutter mechanism event lacks low7 + optional history + "
            "governor6 + packed-sphere "
            f"context: expected {expected}, found {len(context)}"
        )
    gamma = float(event["gamma"])
    if not np.isclose(float(context[6]), gamma, atol=1.0e-6, rtol=0.0):
        raise ValueError("clutter event gamma disagrees with compact context")
    expected_position = task.env.goal - context[:3]
    robot = np.asarray(event["robot"], np.float32).reshape(-1)
    if len(robot) < 6 or not np.allclose(
        robot[:3], expected_position, atol=2.0e-5, rtol=0.0,
    ):
        raise ValueError("clutter event robot disagrees with compact context")
    previous_applied = context[
        governor_start:governor_start + 3
    ].copy()
    previous_raw = context[
        governor_start + 3:governor_start + 6
    ].copy()
    spheres = task.scene_spec.unpack(task.env, context[scene_start:])
    scene_env = task._environment(spheres)
    return {
        "previous_applied": previous_applied,
        "previous_raw": previous_raw,
        "spheres": spheres,
        "scene_hash": scene_sha256(scene_env, spheres),
        "env": scene_env,
    }


def _event_scene_index(events, task, manifest):
    """Fail closed unless every episode has one ledger-backed fixed scene."""
    ledger_hashes = _expansion_scene_hashes(manifest)
    index = {}
    for event in events:
        key = (
            int(event["round"]),
            float(event["gamma"]),
            int(event["episode"]),
        )
        decoded = _decode_event_scene(event, task)
        if decoded["scene_hash"] not in ledger_hashes:
            raise ValueError(
                "event scene hash is absent from expansion lab_scene_ledger"
            )
        previous = index.get(key)
        if previous is None:
            index[key] = decoded
        elif (
            previous["scene_hash"] != decoded["scene_hash"]
            or not np.array_equal(previous["spheres"], decoded["spheres"])
        ):
            raise ValueError(
                "sphere geometry changed within one expansion episode"
            )
    return index


def _draw_scene_projections(
    side,
    head,
    env: TaskEnvironment,
    frame: np.ndarray,
    spheres: np.ndarray,
    physical_radius: float,
):
    """Draw every physical sphere and its inflated safety shell."""
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    for sphere in canonical_sphere_rows(spheres):
        center = (sphere[:3] - env.start[:3]) @ frame
        effective_radius = float(sphere[3])
        side.fill(
            center[0] + physical_radius * np.cos(theta),
            center[2] + physical_radius * np.sin(theta),
            color="#666c70", alpha=0.48, zorder=2,
        )
        head.fill(
            center[1] + physical_radius * np.cos(theta),
            center[2] + physical_radius * np.sin(theta),
            color="#666c70", alpha=0.48, zorder=2,
        )
        side.plot(
            center[0] + effective_radius * np.cos(theta),
            center[2] + effective_radius * np.sin(theta),
            color="#9da3a6", lw=1.0, ls="--", alpha=0.85, zorder=1,
        )
        head.plot(
            center[1] + effective_radius * np.cos(theta),
            center[2] + effective_radius * np.sin(theta),
            color="#9da3a6", lw=1.0, ls="--", alpha=0.85, zorder=1,
        )
    goal = (env.goal - env.start[:3]) @ frame
    for axis, first, second in ((side, 0, 2), (head, 1, 2)):
        axis.scatter(0.0, 0.0, marker="s", color="#111111", s=25, zorder=10)
        axis.scatter(
            goal[first], goal[second], marker="*", color="#ffca28",
            edgecolor="#6a4e00", s=95, zorder=10,
        )
        axis.grid(alpha=0.18)
        axis.set_aspect("equal", adjustable="box")
    corners = np.asarray([
        [x, y, z]
        for x in env.bounds[0]
        for y in env.bounds[1]
        for z in env.bounds[2]
    ], float)
    local_corners = _local_path(corners, env, frame)
    side.set_xlim(
        float(local_corners[:, 0].min()) - 0.1,
        float(local_corners[:, 0].max()) + 0.1,
    )
    vertical_limits = (
        float(local_corners[:, 2].min()) - 0.1,
        float(local_corners[:, 2].max()) + 0.1,
    )
    side.set_ylim(vertical_limits)
    head.set_xlim(
        float(local_corners[:, 1].max()) + 0.1,
        float(local_corners[:, 1].min()) - 0.1,
    )
    head.set_ylim(vertical_limits)


def _ranked_committed_episodes(cell):
    by_episode = {episode.episode: episode for episode in cell.committed_episodes}
    output = []
    for episode_id in cell.committed_episode_ids:
        if episode_id not in by_episode:
            raise ValueError(
                f"committed episode {episode_id} has no resolved event trace"
            )
        output.append(by_episode[episode_id])
    return output


def _window_positions_for_episode(cell, episode):
    details = cell.detail.get("committed_trajectories", ())
    match = [
        row for row in details
        if int(row["episode_id"]) == int(episode.episode)
    ]
    if len(match) != 1:
        raise ValueError(
            "each plural committed episode needs one trajectory detail"
        )
    positions = []
    for window_id in match[0]["committed_window_ids"]:
        try:
            start = int(str(window_id).rsplit(":w", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid committed window ID {window_id!r}") from error
        if start >= len(episode.executed_events):
            raise ValueError("committed window starts after executed trajectory")
        positions.append(
            np.asarray(episode.executed_events[start]["robot"][:3], float)
        )
    return np.asarray(positions, float).reshape(-1, 3)


def _candidate_paths(event, task, frame):
    decoded = _decode_event_scene(event, task)
    state6 = np.asarray(event["robot"], np.float32).reshape(-1)[:6]
    paths = []
    for candidate in event["candidates"]:
        states, _, _ = task._rollout_plan(
            state6,
            decoded["previous_applied"],
            np.asarray(candidate, np.float32),
        )
        paths.append(_local_path(states[:, :3], task.env, frame))
    return np.asarray(paths), decoded


def _draw_episode_history(
    side,
    head,
    rows,
    frame_step,
    env,
    frame,
    sigma_statistics,
    cmap,
):
    past = [
        event for event in rows if int(event["step"]) <= int(frame_step)
    ]
    segments_side, segments_head, colors = [], [], []
    for event in past:
        if event["chosen_local"] is None:
            continue
        start = _local_path(
            np.asarray(event["robot"][:3])[None], env, frame,
        )[0]
        stop = _local_path(
            np.asarray(event["robot_after"][:3])[None], env, frame,
        )[0]
        candidate = event["selected"][event["chosen_local"]]
        normalized = within_round_normalized_sigma(
            float(event["sigma_K"][candidate]), sigma_statistics,
        )
        colors.append(cmap(normalized))
        segments_side.append([[start[0], start[2]], [stop[0], stop[2]]])
        segments_head.append([[start[1], start[2]], [stop[1], stop[2]]])
    if segments_side:
        side.add_collection(LineCollection(
            np.asarray(segments_side), colors=colors,
            linewidths=1.8, alpha=0.90,
        ))
        head.add_collection(LineCollection(
            np.asarray(segments_head), colors=colors,
            linewidths=1.8, alpha=0.90,
        ))
    return past


def _draw_live_episode_panel(
    side,
    head,
    rows,
    frame_step,
    task,
    frame,
    physical_radius,
    sigma_statistics,
    cmap,
):
    decoded = _decode_event_scene(rows[0], task)
    _draw_scene_projections(
        side, head, task.env, frame, decoded["spheres"], physical_radius,
    )
    past = _draw_episode_history(
        side, head, rows, frame_step, task.env, frame,
        sigma_statistics, cmap,
    )
    if not past:
        return
    current = past[-1]
    if int(current["step"]) == int(frame_step):
        paths, current_scene = _candidate_paths(current, task, frame)
        if current_scene["scene_hash"] != decoded["scene_hash"]:
            raise ValueError("candidate event changed scene within episode")
        selected = tuple(int(value) for value in current["selected"])
        for candidate, path in enumerate(paths):
            if candidate not in selected:
                side.plot(
                    path[:, 0], path[:, 2], color="#9aa0a6",
                    lw=0.35, alpha=0.16,
                )
                head.plot(
                    path[:, 1], path[:, 2], color="#9aa0a6",
                    lw=0.35, alpha=0.16,
                )
        for local, candidate in enumerate(selected):
            verification = current["verification"][local]
            path = paths[candidate]
            if verification["valid"]:
                side.plot(
                    path[:, 0], path[:, 2], color="#17964b",
                    lw=3.0, alpha=0.42,
                )
                head.plot(
                    path[:, 1], path[:, 2], color="#17964b",
                    lw=3.0, alpha=0.42,
                )
            normalized = within_round_normalized_sigma(
                float(current["sigma_K"][candidate]), sigma_statistics,
            )
            color = cmap(normalized)
            side.plot(
                path[:, 0], path[:, 2], color=color,
                lw=1.15, ls="--", alpha=0.97,
            )
            head.plot(
                path[:, 1], path[:, 2], color=color,
                lw=1.15, ls="--", alpha=0.97,
            )
        if current["chosen_local"] is not None:
            chosen = selected[int(current["chosen_local"])]
            path = paths[chosen]
            side.plot(path[:2, 0], path[:2, 2], color="#1468b3", lw=3.2)
            head.plot(path[:2, 1], path[:2, 2], color="#1468b3", lw=3.2)
    terminal = rows[-1]
    if (
        int(terminal["step"]) <= int(frame_step)
        and terminal.get("status") is not None
    ):
        point = np.asarray(
            terminal["robot_after"]
            if terminal["chosen_local"] is not None else terminal["robot"],
            float,
        )[:3]
        local = _local_path(point[None], task.env, frame)[0]
        success = terminal["status"] == "SUCCESS"
        marker, color = ("*", "#159447") if success else ("x", "#c8321b")
        side.scatter(local[0], local[2], marker=marker, color=color, s=48)
        head.scatter(local[1], local[2], marker=marker, color=color, s=48)


def _draw_committed_episode_panel(
    side,
    head,
    episode,
    cell,
    scene,
    env,
    frame,
    physical_radius,
):
    _draw_scene_projections(
        side, head, env, frame, scene["spheres"], physical_radius,
    )
    path = _local_path(episode.path, env, frame)
    side.plot(path[:, 0], path[:, 2], color=COMMITTED_COLOR, lw=3.0)
    head.plot(path[:, 1], path[:, 2], color=COMMITTED_COLOR, lw=3.0)
    side.scatter(
        path[-1, 0], path[-1, 2], marker="o", color=OTHER_SUCCESS_COLOR,
        edgecolor="#0c5429", s=42,
    )
    head.scatter(
        path[-1, 1], path[-1, 2], marker="o", color=OTHER_SUCCESS_COLOR,
        edgecolor="#0c5429", s=42,
    )
    positions = _window_positions_for_episode(cell, episode)
    if len(positions):
        local = _local_path(positions, env, frame)
        side.scatter(
            local[:, 0], local[:, 2], s=16, color=COMMITTED_WINDOW_COLOR,
            edgecolor="white", linewidth=0.25, zorder=9,
        )
        head.scatter(
            local[:, 1], local[:, 2], s=16, color=COMMITTED_WINDOW_COLOR,
            edgecolor="white", linewidth=0.25, zorder=9,
        )


def _plot_clutter_mechanism_multiview(
    task,
    committed_cells,
    event_scenes,
    rounds,
    gamma,
    physical_radius,
    output,
):
    """One row per committed episode, never mixing randomized scenes."""
    frame = start_goal_frame(task.env)
    rows = []
    for round_i in rounds:
        cell = committed_cells[(int(round_i), float(gamma))]
        for rank, episode in enumerate(_ranked_committed_episodes(cell)):
            rows.append((int(round_i), rank, cell, episode))
    if not rows:
        raise ValueError("no committed episodes for clutter mechanism multiview")
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(12.4, 3.7 * len(rows)), squeeze=False,
    )
    for row_index, (round_i, rank, cell, episode) in enumerate(rows):
        key = (round_i, float(gamma), int(episode.episode))
        scene = event_scenes[key]
        side, head = axes[row_index]
        _draw_committed_episode_panel(
            side, head, episode, cell, scene, task.env, frame, physical_radius,
        )
        side.set_title(
            rf"round {round_i}, $\gamma={gamma:g}$, commit rank {rank + 1}; "
            f"scene {scene['scene_hash'][:8]}"
        )
        head.set_title("head-on")
        side.set_xlabel("start-goal axis [m]")
        side.set_ylabel("vertical [m]")
        head.set_xlabel("transverse [m]")
        head.set_ylabel("vertical [m]")
    handles = (
        Line2D(
            [0], [0], color=COMMITTED_COLOR, lw=3.0,
            label="committed SUCCESS trajectory",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=COMMITTED_WINDOW_COLOR,
            markeredgecolor="white", markersize=7,
            label="exact Adam-admitted window start",
        ),
        Line2D(
            [0], [0], color="#9da3a6", lw=1.0, ls="--",
            label="inflated safety shell",
        ),
    )
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = Path(output)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _clutter_mechanism_video(
    task,
    manifest,
    events,
    committed_cells,
    event_scenes,
    rounds,
    gamma,
    physical_radius,
    output,
):
    """Per-committed-episode K/B movie with one fixed scene per panel."""
    selected_events = [
        event for event in events
        if int(event["round"]) in set(map(int, rounds))
        and np.isclose(
            float(event["gamma"]), float(gamma), atol=1.0e-8, rtol=0.0,
        )
    ]
    if not selected_events:
        raise ValueError("no clutter mechanism events match rounds/gamma")
    sigma_by_round = round_sigma_statistics(selected_events)
    sigma_scope = (
        "successful traces"
        if manifest.get("event_log") == "committed_success"
        else "all gathered traces"
    )
    frame = start_goal_frame(task.env)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0.0, 1.0, clip=True)
    committed_by_round = {
        int(round_i): _ranked_committed_episodes(
            committed_cells[(int(round_i), float(gamma))]
        )
        for round_i in rounds
    }
    max_rows = max(map(len, committed_by_round.values()))
    if max_rows < 1:
        raise ValueError("mechanism video requires committed successful episodes")
    grouped = {}
    for event in selected_events:
        key = (
            int(event["round"]),
            float(event["gamma"]),
            int(event["episode"]),
        )
        grouped.setdefault(key, []).append(event)
    for rows in grouped.values():
        rows.sort(key=lambda event: int(event["step"]))

    # The dimensions remain even at 120 dpi for yuv420p/libx264, including
    # plural committed-episode layouts.
    fig = plt.figure(figsize=(13.2, max(4.0, 3.5 * max_rows)))
    writer = FFMpegWriter(
        fps=8,
        codec="libx264",
        bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=120):
        for round_i in sorted(map(int, rounds)):
            episodes = committed_by_round[round_i]
            episode_rows = {
                episode.episode: grouped[(
                    round_i, float(gamma), int(episode.episode),
                )]
                for episode in episodes
            }
            steps = sorted({
                int(event["step"])
                for rows in episode_rows.values() for event in rows
            })
            indices = np.unique(np.linspace(
                0, len(steps) - 1, min(36, len(steps)), dtype=int,
            ))
            for frame_step in [steps[index] for index in indices]:
                fig.clear()
                grid = fig.add_gridspec(
                    max_rows, 2, wspace=0.16, hspace=0.28,
                )
                axes = []
                for rank in range(max_rows):
                    side = fig.add_subplot(grid[rank, 0])
                    head = fig.add_subplot(grid[rank, 1])
                    axes.extend([side, head])
                    if rank >= len(episodes):
                        side.axis("off")
                        head.axis("off")
                        continue
                    episode = episodes[rank]
                    rows = episode_rows[episode.episode]
                    _draw_live_episode_panel(
                        side, head, rows, frame_step, task, frame,
                        physical_radius, sigma_by_round[round_i], cmap,
                    )
                    scene = event_scenes[(
                        round_i, float(gamma), int(episode.episode),
                    )]
                    side.set_title(
                        f"commit rank {rank + 1}, episode {episode.episode}, "
                        f"scene {scene['scene_hash'][:8]}"
                    )
                    head.set_title("head-on")
                    side.set_ylabel("vertical [m]")
                    head.set_ylabel("vertical [m]")
                    if rank == max_rows - 1:
                        side.set_xlabel("start-goal axis [m]")
                        head.set_xlabel("transverse [m]")
                    if rank == 0:
                        side.legend(
                            handles=(
                                Line2D(
                                    [0], [0], color="#9aa0a6", lw=1.0,
                                    label="unqueried K-plan",
                                ),
                                Line2D(
                                    [0], [0], color=cmap(0.65), lw=1.2,
                                    ls="--", label="selected B (marginal sigma)",
                                ),
                                Line2D(
                                    [0], [0], color="#17964b", lw=3.0,
                                    alpha=0.55, label="verifier-positive",
                                ),
                                Line2D(
                                    [0], [0], color="#1468b3", lw=3.2,
                                    label="executed first step",
                                ),
                            ),
                            loc="lower right",
                            fontsize=7,
                            framealpha=0.86,
                        )
                row = manifest["rounds"][round_i - 1]
                fig.suptitle(
                    rf"round {round_i} | $\gamma={gamma:g}$ | step {frame_step} | "
                    rf"$K={manifest['config']['K']}$, $B={manifest['config']['B']}$ | "
                    f"ESS/K={row['ESS_over_K']:.3f}, "
                    f"uplift={row['uncertainty_uplift']:+.3f}",
                    fontsize=13,
                    weight="bold",
                )
                scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                colorbar = fig.colorbar(
                    scalar, ax=axes, fraction=0.018, pad=0.025,
                )
                colorbar.set_label(
                    r"within-round normalized marginal "
                    rf"$\widetilde{{\sigma}}_n(\phi_s)$ ({sigma_scope})"
                )
                writer.grab_frame()

            fig.clear()
            grid = fig.add_gridspec(max_rows, 2, wspace=0.16, hspace=0.28)
            cell = committed_cells[(round_i, float(gamma))]
            for rank in range(max_rows):
                side = fig.add_subplot(grid[rank, 0])
                head = fig.add_subplot(grid[rank, 1])
                if rank >= len(episodes):
                    side.axis("off")
                    head.axis("off")
                    continue
                episode = episodes[rank]
                scene = event_scenes[(
                    round_i, float(gamma), int(episode.episode),
                )]
                _draw_committed_episode_panel(
                    side, head, episode, cell, scene, task.env, frame,
                    physical_radius,
                )
                side.set_title(
                    f"committed rank {rank + 1}; scene "
                    f"{scene['scene_hash'][:8]}"
                )
                head.set_title("head-on")
                if rank == 0:
                    side.legend(
                        handles=(
                            Line2D(
                                [0], [0], color=COMMITTED_COLOR, lw=3.0,
                                label="committed SUCCESS",
                            ),
                            Line2D(
                                [0], [0], marker="o", color="none",
                                markerfacecolor=COMMITTED_WINDOW_COLOR,
                                markeredgecolor="white", markersize=7,
                                label="Adam-admitted window start",
                            ),
                        ),
                        loc="lower right",
                        fontsize=7,
                        framealpha=0.86,
                    )
            fig.suptitle(
                rf"round {round_i} | $\gamma={gamma:g}$ | exact current-round "
                "committed SUCCESS trajectories and Adam-admitted starts\n"
                f"{len(cell.committed_trajectory_ids)} trajectories, "
                f"{len(cell.committed_window_ids)} window starts",
                fontsize=13,
                weight="bold",
            )
            for _ in range(12):
                writer.grab_frame()
    plt.close(fig)


def evaluate_lab_clutter_expansion(
    args,
    config,
    pretrain_manifest,
    manifest,
):
    """Evaluate raw temperature-1 policy on disjoint randomized and fixed scenes."""
    del pretrain_manifest
    device = torch.device(getattr(args, "device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"evaluation device {device} requires CUDA")
    requested_output = getattr(args, "evaluation_output", None)
    output = (
        Path(requested_output)
        if requested_output is not None
        else args.expansion / "eval"
    )
    if (
        requested_output is not None
        and output.exists()
        and (
            not output.is_dir()
            or any(output.iterdir())
        )
    ):
        raise FileExistsError(
            f"refusing to overwrite explicit evaluation output {output}"
        )
    gammas = [float(value) for value in config.data.gammas]
    scene_bank = _evaluation_scene_provenance(
        config,
        int(args.episodes),
        int(args.seed),
        manifest,
    )
    fixed_scene_seed = getattr(args, "fixed_scene_seed", None)
    if fixed_scene_seed is None:
        fixed_scene_seed = int(args.seed) + FIXED_SCENE_SEED_OFFSET
    fixed_scene_rollouts = int(
        getattr(args, "fixed_scene_rollouts", 10)
    )
    if fixed_scene_rollouts < 1:
        raise ValueError("--fixed-scene-rollouts must be positive")
    fixed_scene = _fixed_scene_provenance(
        config,
        int(fixed_scene_seed),
        manifest,
        scene_bank,
    )
    evaluation_scene_spec = sphere_scene_spec_from_config(config)
    minimum_count = int(
        getattr(
            getattr(evaluation_scene_spec, "spec", None),
            "count_min",
            evaluation_scene_spec.max_count,
        )
    )
    obstacle_counts = list(range(
        minimum_count, int(evaluation_scene_spec.max_count) + 1,
    ))
    total_rounds = int(manifest["config"]["rounds"])
    rounds = sorted({
        0,
        *range(0, total_rounds + 1, int(args.stride)),
        total_rounds,
    })
    requested_display_rounds = getattr(args, "video_rounds", None)
    if requested_display_rounds is None:
        display_rounds = sorted({
            0,
            total_rounds // 2,
            total_rounds,
        })
    else:
        display_rounds = sorted(set(map(int, requested_display_rounds)))
    if (
        not display_rounds
        or any(
            round_i < 0 or round_i > total_rounds
            for round_i in display_rounds
        )
    ):
        raise ValueError(
            f"--video-rounds must lie in [0,{total_rounds}], "
            f"got {display_rounds}"
        )
    rounds = sorted(set(rounds) | set(display_rounds))
    artifact_binding = _evaluation_artifact_binding(args, rounds)

    per_round_rows = {}
    fixed_per_round_rows = {}
    summaries = {}
    fixed_summaries = {}
    probes = {}
    device_kwargs = (
        {} if device.type == "cpu" else {"device": device}
    )
    for round_i in rounds:
        policy = _checkpoint_policy(
            args.pretrain_dir, args.expansion, round_i, **device_kwargs,
        )
        rows = _raw_rows(
            policy,
            config,
            gammas,
            scene_bank["scenes"],
            int(args.seed),
            **device_kwargs,
        )
        fixed_rows = _fixed_scene_rows(
            policy,
            config,
            gammas,
            fixed_scene["scene"],
            fixed_scene_rollouts,
            int(args.seed),
            **device_kwargs,
        )
        probe_rows = _start_probe(
            policy,
            config,
            gammas,
            int(args.probe_samples),
            scene_bank["start_probe_scene"],
            **device_kwargs,
        )
        per_round_rows[round_i] = rows
        fixed_per_round_rows[round_i] = fixed_rows
        probes[str(round_i)] = probe_rows
        per_gamma = {
            f"{gamma:g}": _summarize([
                row for row in rows if row["gamma"] == gamma
            ])
            for gamma in gammas
        }
        fixed_per_gamma = {
            f"{gamma:g}": _summarize(
                [
                    row for row in fixed_rows
                    if row["gamma"] == gamma
                ],
                path_spread_domain="fixed_scene_single_gamma",
            )
            for gamma in gammas
        }
        per_obstacle_count = {
            str(count): _summarize([
                row for row in rows
                if _row_obstacle_count(row) == count
            ])
            for count in obstacle_counts
        }
        per_gamma_obstacle_count = {
            f"{gamma:g}": {
                str(count): _summarize([
                    row for row in rows
                    if row["gamma"] == gamma
                    and _row_obstacle_count(row) == count
                ])
                for count in obstacle_counts
            }
            for gamma in gammas
        }
        summaries[str(round_i)] = {
            "pooled": _summarize(rows),
            "per_gamma": per_gamma,
            "per_obstacle_count": per_obstacle_count,
            "per_gamma_obstacle_count": per_gamma_obstacle_count,
            "start_probe_validity": float(np.mean([
                row["valid"] for row in probe_rows
            ])),
        }
        fixed_summaries[str(round_i)] = {
            # Pooled path spread is deliberately N/A because gamma is a
            # conditioning variable; only per-gamma fixed-scene cells report it.
            "pooled": _summarize(fixed_rows),
            "per_gamma": fixed_per_gamma,
        }
        pooled = summaries[str(round_i)]["pooled"]
        print(
            f"round {round_i:3d}: SR={pooled['SR']:.3f} "
            f"CR={pooled['CR']:.3f} OOB={pooled['OOB']:.3f} "
            f"Vwin={pooled['window_validity']:.3f}",
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    concrete_config = _write_fixed_scene_config(
        config,
        fixed_scene["scene"],
        output / "fixed_scene_config.json",
    )
    slim = {
        str(round_i): [{
            key: value for key, value in row.items()
            if key not in {
                "states", "controls", "applied_controls", "dense_steps",
            }
        } for row in rows]
        for round_i, rows in per_round_rows.items()
    }
    fixed_slim = {}
    for round_i, rows in fixed_per_round_rows.items():
        serialized_rows = []
        for row in rows:
            serialized = {
                key: value for key, value in row.items()
                if key not in {
                    "states", "controls", "applied_controls", "dense_steps",
                }
            }
            serialized["arc_length_resampled_path_xyz"] = (
                _arc_length_resample(
                    row["states"], PATH_SPREAD_RESAMPLE_POINTS,
                ).tolist()
                if row["status"] == "SUCCESS"
                else None
            )
            serialized_rows.append(serialized)
        fixed_slim[str(round_i)] = serialized_rows
    fixed_seed_map = {
        f"{gamma:g}": [
            int(row["rollout_seed"])
            for row in fixed_per_round_rows[rounds[0]]
            if row["gamma"] == gamma
        ]
        for gamma in gammas
    }
    for round_i in rounds[1:]:
        current_seed_map = {
            f"{gamma:g}": [
                int(row["rollout_seed"])
                for row in fixed_per_round_rows[round_i]
                if row["gamma"] == gamma
            ]
            for gamma in gammas
        }
        if current_seed_map != fixed_seed_map:
            raise RuntimeError(
                "fixed-scene rollout seeds changed across checkpoints"
            )
    (output / "raw_eval.json").write_text(json.dumps({
        "status": "LAB_CLUTTER_RAW_TEMPERATURE1_EVALUATION_COMPLETE",
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "runtime_device": str(device),
        "one_sigma_contract": {
            "coverage": ONE_SIGMA_COVERAGE,
            "rates": "Wilson score interval with z=1",
            "continuous_metrics": (
                "deterministic trajectory-level percentile bootstrap, "
                "central 68.27%"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "obstacle_counts": obstacle_counts,
        "artifact_binding": artifact_binding,
        "scene_bank": scene_bank,
        "summary": summaries,
        "rows": slim,
        "start_probe_rows": probes,
    }, indent=2) + "\n")
    (output / "fixed_scene_raw_eval.json").write_text(json.dumps({
        "status": "LAB_CLUTTER_FIXED_SCENE_RAW_TEMPERATURE1_EVALUATION_COMPLETE",
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "runtime_device": str(device),
        "one_sigma_contract": {
            "coverage": ONE_SIGMA_COVERAGE,
            "rates": "Wilson score interval with z=1",
            "continuous_metrics": (
                "deterministic trajectory-level percentile bootstrap, "
                "central 68.27%"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "obstacle_count": int(
            fixed_scene["scene"]["obstacle_count"]
        ),
        "artifact_binding": artifact_binding,
        "rollouts_per_gamma": fixed_scene_rollouts,
        "path_spread_resample_points": PATH_SPREAD_RESAMPLE_POINTS,
        "common_random_numbers_across_checkpoints": True,
        "concrete_config": concrete_config,
        "rollout_seeds_by_gamma": fixed_seed_map,
        "displayed_checkpoints": display_rounds,
        "scene_provenance": fixed_scene,
        "summary": fixed_summaries,
        "rows": fixed_slim,
    }, indent=2) + "\n")
    if getattr(args, "metrics_only", False):
        print("[outputs]", output, flush=True)
        return

    _plot_curves(
        summaries, gammas, output / "raw_curves.png",
    )
    _plot_count_curves(
        summaries, obstacle_counts, output / "raw_curves_by_obstacle_count.png",
    )
    _plot_gallery(
        per_round_rows,
        display_rounds,
        gammas,
        config,
        output / "raw_gallery.png",
    )
    _plot_fixed_scene_gallery(
        fixed_per_round_rows,
        fixed_summaries,
        display_rounds,
        gammas,
        config,
        fixed_scene["scene"],
        output / "fixed_scene_raw_gallery.png",
    )

    if not getattr(args, "screening_only", False):
        events_path = args.expansion / "events.pt"
        if not events_path.is_file():
            raise FileNotFoundError(
                "clutter mechanism video requires --event-log full or "
                "committed_success during expansion"
            )
        mechanism_rounds = [
            round_i for round_i in display_rounds if round_i > 0
        ]
        if not mechanism_rounds:
            raise ValueError(
                "clutter mechanism video needs at least one positive "
                "--video-rounds checkpoint"
            )
        matching_gamma = [
            gamma for gamma in gammas
            if np.isclose(
                gamma, float(args.video_gamma), atol=1.0e-8, rtol=0.0,
            )
        ]
        if len(matching_gamma) != 1:
            raise ValueError(
                f"--video-gamma {args.video_gamma:g} is not one of {gammas}"
            )
        video_gamma = matching_gamma[0]
        events = torch.load(events_path, map_location="cpu", weights_only=False)
        committed_cells = resolve_committed_success(manifest, events)
        _validate_replay_provenance(manifest, committed_cells)
        policy = _checkpoint_policy(
            args.pretrain_dir, args.expansion, total_rounds, **device_kwargs,
        )
        task = LabClutterSphereExpansionTask(
            config,
            context_schema=policy.context_schema,
            device=device,
            tight_corridor=True,
        )
        event_scenes = _event_scene_index(events, task, manifest)
        physical_radius = float(
            config.raw["scene_randomization"]["physical_radius_m"]
        )
        _plot_clutter_mechanism_multiview(
            task,
            committed_cells,
            event_scenes,
            mechanism_rounds,
            video_gamma,
            physical_radius,
            output / "mechanism_multiview.png",
        )
        _clutter_mechanism_video(
            task,
            manifest,
            events,
            committed_cells,
            event_scenes,
            mechanism_rounds,
            video_gamma,
            physical_radius,
            output / "mechanism.mp4",
        )
    print("[outputs]", output, flush=True)
