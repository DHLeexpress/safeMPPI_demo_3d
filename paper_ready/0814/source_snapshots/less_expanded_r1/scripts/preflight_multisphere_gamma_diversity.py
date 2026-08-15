#!/usr/bin/env python3
"""Fixed-scene PRE2 gamma-diversity check with dense z clutter.

This is a pure-sampling diagnostic.  It never updates the policy, writes a
replay buffer, or contributes trajectories to an expansion quota.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.expansion import (  # noqa: E402
    mean_pairwise_lengthscale,
)
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    LAB_CLUTTER_GOVERNOR_DIM,
    scene_sha256,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (  # noqa: E402
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_visual_flow import (  # noqa: E402
    LabUniformHp100Rasterizer,
)
from safe_mppi.path_focused_clutter import (  # noqa: E402
    double_hourglass_halfwidth,
)
import scripts.preflight_multisphere_expansion_pre2 as _base_preflight  # noqa: E402
from scripts.preflight_multisphere_expansion_pre2 import (  # noqa: E402
    CostArm,
    _counter_seed,
    _finalize_episode,
    _fresh_shared_scene_state,
    _json_ready,
    _new_episode,
    _sha256,
    _summary,
    _task,
)
from scripts.research_multisphere_expansion_pre2 import (  # noqa: E402
    _learned_phi_calibration,
    _set_flow_nfe,
)


GAMMAS = (0.1, 0.3, 0.5, 1.0)
SCENE_COUNT = 5
ANCHORED_SCENE_COUNT = 2
SPHERE_COUNT = 6
Z_MIN_M = 0.7
Z_MAX_M = 1.1
ANCHOR_FRACTION = 0.2
ANCHOR_Z_M = 0.9
FLOW_NFE = 16
K = 16
B = 8
BETA = 0.1
FLOW_BASE_STD = 1.0
ARM = CostArm(
    "wall250_axis5_control005",
    wall_weight=250.0,
    wall_target_m=0.15,
    axis_weight=5.0,
    axis_radius_m=1.1,
    control_weight=0.05,
)


class _PairedLatentPolicy:
    """Bypass PRE2 gamma salting so matched gamma rows share exact bases."""

    def __init__(self, adapter):
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    @torch.no_grad()
    def sample_many_with_base(
        self, contexts, count, generators, base_std=1.0,
    ):
        policy_contexts = self.adapter._policy_context(contexts)
        if len(policy_contexts) != len(generators):
            raise ValueError("one paired generator is required per context")
        core = self.adapter.policy
        encoded = core.encode_context(policy_contexts)
        flow = core.flow
        bases = torch.stack([
            torch.randn(
                count,
                flow.plan_dim,
                device=policy_contexts.device,
                generator=generator,
            ) * float(base_std)
            for generator in generators
        ])
        seeds = [int(generator.initial_seed()) for generator in generators]
        for seed in set(seeds):
            indices = [index for index, value in enumerate(seeds) if value == seed]
            reference = bases[indices[0]]
            if any(
                not torch.equal(reference, bases[index])
                for index in indices[1:]
            ):
                raise RuntimeError("paired gamma rows did not receive exact bases")
        repeated_context = encoded[:, None, :].expand(
            len(policy_contexts), count, encoded.shape[-1]
        ).reshape(len(policy_contexts) * count, encoded.shape[-1])
        integrated = flow._integrate_flow(
            bases.reshape(len(policy_contexts) * count, flow.plan_dim),
            repeated_context,
        ).reshape(len(policy_contexts), count, *flow.plan_shape)
        if flow.control_limit is not None:
            integrated = integrated.clamp(
                -flow.control_limit, flow.control_limit,
            )
        return integrated, bases.reshape(
            len(policy_contexts), count, *flow.plan_shape,
        )


def _paired_stream_key(stream: str) -> str:
    prefix, separator, _ = str(stream).rpartition("_g")
    if not separator or not prefix.startswith("dense_s"):
        raise ValueError(f"unexpected dense-scene stream {stream!r}")
    return prefix


def _rollout_batch_paired(*, policy, **kwargs):
    """Use identical latent/acquisition RNG state across gamma per scene."""
    original_counter_seed = _base_preflight._counter_seed

    def paired_counter_seed(master: int, *coordinates: Any) -> int:
        if (
            len(coordinates) == 4
            and coordinates[0] == "preflight_gather"
        ):
            _, stream, _gamma, control_step = coordinates
            return original_counter_seed(
                master,
                "paired_preflight_gather",
                _paired_stream_key(str(stream)),
                int(control_step),
            )
        return original_counter_seed(master, *coordinates)

    _base_preflight._counter_seed = paired_counter_seed
    try:
        return _base_preflight._rollout_batch(
            policy=_PairedLatentPolicy(policy), **kwargs,
        )
    finally:
        _base_preflight._counter_seed = original_counter_seed


def _generate_sphere_rows(
    env,
    scene_spec,
    *,
    scene_seed: int,
    anchored: bool,
) -> np.ndarray:
    """Generate one six-sphere scene under the requested dense-z law."""
    spec = scene_spec.spec
    if spec.count_min != SPHERE_COUNT or spec.count_max != SPHERE_COUNT:
        raise ValueError("dense-z diagnostic requires exactly six spheres")
    if spec.transverse_halfwidth_control_points is None:
        raise ValueError("dense-z diagnostic requires double-hourglass controls")
    radius = float(np.float32(spec.modeled_radius_m))
    if not np.isclose(radius, 0.2405, rtol=0.0, atol=1.0e-6):
        raise ValueError("dense-z diagnostic requires modeled radius 0.2405 m")
    pair_distance = 2.0 * radius + spec.minimum_surface_gap_m
    if not np.isclose(pair_distance, 0.681, rtol=0.0, atol=2.0e-6):
        raise ValueError("dense-z diagnostic requires 0.681 m center spacing")

    start = np.asarray(env.start[:3], np.float64)
    goal = np.asarray(env.goal, np.float64)
    horizontal = goal[:2] - start[:2]
    horizontal_length = float(np.linalg.norm(horizontal))
    if horizontal_length <= 1.0e-12:
        raise ValueError("start-goal path must have horizontal extent")
    horizontal /= horizontal_length
    normal = np.asarray([-horizontal[1], horizontal[0]], np.float64)
    lower = env.bounds[:, 0] + radius + spec.boundary_surface_gap_m
    upper = env.bounds[:, 1] - radius - spec.boundary_surface_gap_m
    endpoint_distance = radius + spec.endpoint_surface_gap_m
    rng = np.random.default_rng(int(scene_seed))

    centers: list[np.ndarray] = []
    if anchored:
        anchor = start + ANCHOR_FRACTION * (goal - start)
        anchor[2] = ANCHOR_Z_M
        centers.append(anchor.astype(np.float32).astype(np.float64))

    for _ in range(spec.max_layout_attempts):
        along = float(rng.uniform(spec.longitudinal_min, spec.longitudinal_max))
        halfwidth = double_hourglass_halfwidth(
            along, spec.transverse_halfwidth_control_points,
        )
        lateral = float(rng.uniform(-halfwidth, halfwidth))
        xy = start[:2] + along * (goal[:2] - start[:2]) + normal * lateral
        candidate = np.asarray([
            xy[0], xy[1], rng.uniform(Z_MIN_M, Z_MAX_M),
        ], np.float32).astype(np.float64)
        if bool((candidate < lower).any() or (candidate > upper).any()):
            continue
        if (
            float(np.linalg.norm(candidate - start)) < endpoint_distance
            or float(np.linalg.norm(candidate - goal)) < endpoint_distance
        ):
            continue
        if any(
            float(np.linalg.norm(candidate - center)) < pair_distance
            for center in centers
        ):
            continue
        centers.append(candidate)
        if len(centers) == SPHERE_COUNT:
            break
    if len(centers) != SPHERE_COUNT:
        raise RuntimeError(
            f"could not place six dense-z spheres for seed {scene_seed}"
        )
    spheres = np.concatenate([
        np.asarray(centers, np.float32),
        np.full((SPHERE_COUNT, 1), radius, np.float32),
    ], axis=1)
    return scene_spec.validate(env, spheres)


def _scene_geometry_summary(
    spheres: np.ndarray,
    *,
    anchor_expected: np.ndarray | None,
) -> dict[str, Any]:
    centers = np.asarray(spheres, np.float64)[:, :3]
    pairwise = [
        float(np.linalg.norm(centers[first] - centers[second]))
        for first in range(len(centers))
        for second in range(first + 1, len(centers))
    ]
    anchor_index = None
    if anchor_expected is not None:
        distances = np.linalg.norm(centers - anchor_expected[None], axis=1)
        anchor_index = int(np.argmin(distances))
        if float(distances[anchor_index]) > 1.0e-6:
            raise RuntimeError("canonical scene rows lost the exact path anchor")
    return {
        "minimum_center_distance_m": float(min(pairwise)),
        "z_min_m": float(centers[:, 2].min()),
        "z_max_m": float(centers[:, 2].max()),
        "anchor_index": anchor_index,
        "anchor_center_m": (
            None if anchor_expected is None else anchor_expected.astype(
                np.float32
            )
        ),
    }


def _aligned_path_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, np.float64).reshape(-1, 3)
    second = np.asarray(second, np.float64).reshape(-1, 3)
    grid = np.linspace(0.0, 1.0, 64)
    first_time = np.linspace(0.0, 1.0, len(first))
    second_time = np.linspace(0.0, 1.0, len(second))
    aligned_first = np.column_stack([
        np.interp(grid, first_time, first[:, axis]) for axis in range(3)
    ])
    aligned_second = np.column_stack([
        np.interp(grid, second_time, second[:, axis]) for axis in range(3)
    ])
    return float(np.mean(np.linalg.norm(
        aligned_first - aligned_second, axis=1,
    )))


def _progress_signature(
    path: np.ndarray,
    *,
    start: np.ndarray,
    goal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(path, np.float64).reshape(-1, 3)
    start = np.asarray(start, np.float64).reshape(3)
    goal = np.asarray(goal, np.float64).reshape(3)
    horizontal = goal[:2] - start[:2]
    length = float(np.linalg.norm(horizontal))
    direction = horizontal / length
    normal = np.asarray([-direction[1], direction[0]], np.float64)
    displacement = path - start
    progress = (displacement[:, :2] @ direction) / length
    lateral = displacement[:, :2] @ normal
    running = np.maximum.accumulate(progress)
    keep = np.concatenate([[True], np.diff(running) > 1.0e-6])
    return running[keep], lateral[keep], path[keep, 2]


def _progress_lateral_z_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    start: np.ndarray,
    goal: np.ndarray,
) -> float | None:
    first_s, first_lateral, first_z = _progress_signature(
        first, start=start, goal=goal,
    )
    second_s, second_lateral, second_z = _progress_signature(
        second, start=start, goal=goal,
    )
    lower = max(0.2, float(first_s[0]), float(second_s[0]))
    upper = min(0.8, float(first_s[-1]), float(second_s[-1]))
    if upper - lower < 0.05:
        return None
    grid = np.linspace(lower, upper, 64)
    first_lz = np.column_stack([
        np.interp(grid, first_s, first_lateral),
        np.interp(grid, first_s, first_z),
    ])
    second_lz = np.column_stack([
        np.interp(grid, second_s, second_lateral),
        np.interp(grid, second_s, second_z),
    ])
    return float(np.mean(np.linalg.norm(first_lz - second_lz, axis=1)))


def _scene_diversity(
    rows: list[dict[str, Any]],
    *,
    start: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "SUCCESS"]
    all_distances = []
    successful_distances = []
    progress_distances = []
    successful_progress_distances = []
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            distance = _aligned_path_distance(
                rows[first]["path"], rows[second]["path"],
            )
            all_distances.append(distance)
            progress_distance = _progress_lateral_z_distance(
                rows[first]["path"],
                rows[second]["path"],
                start=start,
                goal=goal,
            )
            if progress_distance is not None:
                progress_distances.append(progress_distance)
            if (
                rows[first]["status"] == "SUCCESS"
                and rows[second]["status"] == "SUCCESS"
            ):
                successful_distances.append(distance)
                if progress_distance is not None:
                    successful_progress_distances.append(progress_distance)
    signatures = {}
    for row in rows:
        progress, lateral, z = _progress_signature(
            row["path"], start=start, goal=goal,
        )
        middle = (progress >= 0.2) & (progress <= 0.8)
        signatures[f"{float(row['gamma']):g}"] = {
            "status": row["status"],
            "failed_path": row["status"] != "SUCCESS",
            "maximum_progress_fraction": float(progress[-1]),
            "mean_lateral_m": (
                float(np.mean(lateral[middle])) if bool(middle.any()) else None
            ),
            "mean_z_m": (
                float(np.mean(z[middle])) if bool(middle.any()) else None
            ),
        }
    return {
        "successful_gamma_count": len(successful),
        "statuses": {
            f"{float(row['gamma']):g}": row["status"] for row in rows
        },
        "signatures": signatures,
        "mean_pairwise_all_path_distance_m": (
            float(np.mean(all_distances)) if all_distances else None
        ),
        "mean_pairwise_success_path_distance_m": (
            float(np.mean(successful_distances))
            if successful_distances else None
        ),
        "minimum_pairwise_success_path_distance_m": (
            float(np.min(successful_distances))
            if successful_distances else None
        ),
        "mean_pairwise_progress_lateral_z_distance_m": (
            float(np.mean(progress_distances))
            if progress_distances else None
        ),
        "mean_pairwise_success_progress_lateral_z_distance_m": (
            float(np.mean(successful_progress_distances))
            if successful_progress_distances else None
        ),
        "minimum_pairwise_success_progress_lateral_z_distance_m": (
            float(np.min(successful_progress_distances))
            if successful_progress_distances else None
        ),
    }


def _run_fixed_gallery(
    args,
    task,
    policy,
    lengthscale: float,
    rasterizer,
) -> dict[str, Any]:
    episodes = []
    scenes = []
    next_episode = 200_000
    anchor = np.asarray(task.env.start[:3], np.float64) + (
        ANCHOR_FRACTION
        * (np.asarray(task.env.goal, np.float64) - task.env.start[:3])
    )
    anchor[2] = ANCHOR_Z_M
    for scene_index in range(1, SCENE_COUNT + 1):
        anchored = scene_index <= ANCHORED_SCENE_COUNT
        scene_seed = _counter_seed(args.seed, "dense_z_scene", scene_index)
        spheres = _generate_sphere_rows(
            task.env,
            task.scene_spec,
            scene_seed=scene_seed,
            anchored=anchored,
        )
        base = task.reset(0.5, next_episode, scene_seed)
        base["spheres"] = spheres
        base["scene_seed"] = int(scene_seed)
        base["scene_hash"] = scene_sha256(
            task._environment(spheres), spheres,
        )
        geometry = _scene_geometry_summary(
            spheres,
            anchor_expected=anchor if anchored else None,
        )
        scenes.append({
            "scene_index": scene_index,
            "scene_seed": int(scene_seed),
            "scene_hash": str(base["scene_hash"]),
            "law": (
                "dense_z_uniform_with_path_fraction_0.2_anchor"
                if anchored else "dense_z_uniform_no_anchor"
            ),
            "spheres": spheres,
            "geometry": geometry,
        })
        for gamma in GAMMAS:
            state = _fresh_shared_scene_state(task, base)
            episodes.append(_new_episode(
                task,
                gamma=gamma,
                episode=next_episode,
                reset_seed=scene_seed,
                stream=f"dense_s{scene_index}_g{gamma:g}",
                K=K,
                B=B,
                capture=math.isclose(gamma, 0.5),
                state=state,
            ))
            next_episode += 1
    _rollout_batch_paired(
        task=task,
        policy=policy,
        lengthscale=lengthscale,
        episodes=episodes,
        K=K,
        B=B,
        beta=BETA,
        base_std=FLOW_BASE_STD,
        master_seed=args.seed,
        max_steps=args.max_steps,
        capture_stride=args.capture_stride,
        rasterizer=rasterizer,
    )
    rows = [_finalize_episode(task, row) for row in episodes]
    for row in rows:
        row["failed_path"] = row["status"] != "SUCCESS"
    for scene in scenes:
        scene_rows = [
            row for row in rows
            if row["stream"].startswith(
                f"dense_s{scene['scene_index']}_"
            )
        ]
        hashes = {row["scene"]["scene_hash"] for row in scene_rows}
        if hashes != {scene["scene_hash"]}:
            raise RuntimeError("fixed scene geometry changed across gamma")
        scene["summary"] = _summary(scene_rows)
        scene["diversity"] = _scene_diversity(
            scene_rows,
            start=task.env.start[:3],
            goal=task.env.goal,
        )
    return {
        "provenance": (
            "fixed-scene gamma counterfactual diagnostic only; never eligible "
            "for replay, an expansion update, or true-DR quota evidence"
        ),
        "scenes": scenes,
        "rows": rows,
        "summary": _summary(rows),
        "by_gamma": {
            f"{gamma:g}": _summary([
                row for row in rows if row["gamma"] == gamma
            ])
            for gamma in GAMMAS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=81232)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--capture-stride", type=int, default=10)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"--device {args.device!r} requires CUDA")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing to overwrite nonempty output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.task_config)
    scene_spec = sphere_scene_spec_from_config(config)
    policy = load_lab_clutter_pre2_expansion_policy(
        args.pretrain_dir / "pretrained.pt",
        verifier_suffix_dim=(
            LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim
        ),
    ).to(args.device)
    policy.eval()
    _set_flow_nfe(policy, FLOW_NFE)
    manifest = json.loads(
        (args.pretrain_dir / "pretrain_manifest.json").read_text()
    )
    calibration = _learned_phi_calibration(
        policy,
        args.pretrain_dir,
        manifest,
        config,
        args.seed,
        lab_profile=True,
        flow_base_std=FLOW_BASE_STD,
        paired_noised_representation=True,
    )
    measured_lengthscale = mean_pairwise_lengthscale(calibration)
    if not np.isfinite(measured_lengthscale) or measured_lengthscale <= 0.0:
        raise RuntimeError("PRE2 calibration did not define a lengthscale")
    rasterizer = LabUniformHp100Rasterizer().to(args.device).eval()
    task = _task(config, policy, scene_spec, ARM, args.device)
    gallery = _run_fixed_gallery(
        args, task, policy, measured_lengthscale, rasterizer,
    )
    contract = {
        "kind": "pure sampling; no optimizer and no replay/GP buffer writes",
        "provenance": gallery["provenance"],
        "pretrained_sha256": _sha256(args.pretrain_dir / "pretrained.pt"),
        "pretrain_manifest_sha256": _sha256(
            args.pretrain_dir / "pretrain_manifest.json"
        ),
        "task_config_sha256": _sha256(args.task_config),
        "source_sha256": _sha256(Path(__file__)),
        "parent_preflight_source_sha256": _sha256(
            ROOT / "scripts" / "preflight_multisphere_expansion_pre2.py"
        ),
        "policy_nfe": FLOW_NFE,
        "K": K,
        "B": B,
        "retry_B": B,
        "beta": BETA,
        "flow_base_std": FLOW_BASE_STD,
        "gp_buffer_rows": 0,
        "round1_uncertainty_interpretation": (
            "initial sigma_K is tied at one; first draw is uniform and later "
            "B slots use conditional within-K RBF diversity"
        ),
        "rbf_lengthscale_measured": measured_lengthscale,
        "rbf_lengthscale_manifest": manifest.get("rbf_lengthscale"),
        "arm": asdict(ARM),
        "scene_count": SCENE_COUNT,
        "gamma_values": GAMMAS,
        "same_scene_across_gamma": True,
        "paired_latent_counterfactual": True,
        "paired_latent_contract": (
            "within each scene/control step, all active gamma rows use exact "
            "paired K Gaussian bases and identical acquisition RNG state; "
            "only the gamma-conditioned context/vector field differs"
        ),
        "anchored_scenes": [1, 2],
        "unanchored_scenes": [3, 4, 5],
        "sphere_count": SPHERE_COUNT,
        "scene_law_registration": "unregistered diagnostic only",
        "xy_distribution": (
            "registered double-hourglass longitudinal/lateral h(s) law"
        ),
        "sphere_z_distribution": (
            "independent Uniform[0.7, 1.1], replacing and not clipped by "
            "min(0.2, 0.5*h(s))"
        ),
        "anchor_fraction": ANCHOR_FRACTION,
        "anchor_z_m": ANCHOR_Z_M,
        "minimum_center_distance_m": 0.681,
    }
    output = {"contract": contract, "fixed_gallery": gallery}
    torch.save(output, args.output / "diversity.pt")
    compact = _json_ready(output)
    for row in compact["fixed_gallery"]["rows"]:
        for key in ("dense_path", "commands", "frames"):
            row.pop(key, None)
    (args.output / "summary.json").write_text(
        json.dumps(compact, indent=2, allow_nan=False) + "\n"
    )
    print(f"[done] {args.output / 'diversity.pt'}", flush=True)


if __name__ == "__main__":
    main()
