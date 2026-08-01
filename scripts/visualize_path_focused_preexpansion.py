#!/usr/bin/env python3
"""Render provenance-bound pre-expansion examples for path-focused clutter."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.lab_clutter import config_for_scene  # noqa: E402
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402
from safe_mppi.path_focused_clutter import path_focused_scene_bank  # noqa: E402


GAMMAS = (0.1, 0.3, 0.5, 1.0)
ROW_LABELS = (
    "SafeMPPI | cylinder ID",
    "Pretrained | cylinder ID",
    "Pretrained | sphere OOD",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense_path(result: dict) -> np.ndarray:
    states = np.asarray(result["states"], float)
    dense = np.asarray(result["dense_steps"], float)
    if not len(dense):
        return states[:, :3]
    return np.concatenate([states[:1, :3], dense.reshape(-1, 3)])


def _arc_resample(path: np.ndarray, count: int = 64) -> np.ndarray:
    path = np.asarray(path, float).reshape(-1, 3)
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    if len(path) == 1:
        return np.repeat(path, count, axis=0)
    distance = np.r_[0.0, np.cumsum(
        np.linalg.norm(np.diff(path, axis=0), axis=1)
    )]
    sample = np.linspace(0.0, distance[-1], count)
    return np.column_stack([
        np.interp(sample, distance, path[:, axis])
        for axis in range(3)
    ])


def _features(
    path: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    spheres: np.ndarray,
    cylinders: np.ndarray,
) -> dict:
    sampled = _arc_resample(path)
    start = np.asarray(start, float).reshape(-1)[:3]
    goal = np.asarray(goal, float).reshape(3)
    direct = float(np.linalg.norm(goal - start))
    direction = (goal - start) / direct
    along = (sampled - start) @ direction
    projection = start[None] + along[:, None] * direction[None]
    transverse = sampled - projection
    path_length = float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum())
    realized = float(np.linalg.norm(sampled[-1] - sampled[0]))

    interacted = 0
    obstacle_count = 0
    spheres = np.asarray(spheres, float).reshape(-1, 4)
    cylinders = np.asarray(cylinders, float).reshape(-1, 3)
    if len(spheres):
        surface = (
            np.linalg.norm(
                sampled[:, None] - spheres[None, :, :3], axis=2,
            )
            - spheres[None, :, 3]
        )
        interacted += int(np.sum(surface.min(axis=0) <= 0.30))
        obstacle_count += len(spheres)
    if len(cylinders):
        surface = (
            np.linalg.norm(
                sampled[:, None, :2] - cylinders[None, :, :2], axis=2,
            )
            - cylinders[None, :, 2]
        )
        interacted += int(np.sum(surface.min(axis=0) <= 0.30))
        obstacle_count += len(cylinders)
    return {
        "path_length_excess_ratio": (
            path_length / max(realized, 1e-12) - 1.0
        ),
        "transverse_rms_m": float(np.sqrt(np.mean(np.sum(
            transverse * transverse, axis=1,
        )))),
        "interaction_fraction": (
            float(interacted / obstacle_count) if obstacle_count else 0.0
        ),
        "resampled_path": sampled,
    }


def _rank_groups(groups: list[dict], start: np.ndarray, goal: np.ndarray):
    direct = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)[:3]))
    time_limit = float(np.quantile([
        np.mean([item["time_to_goal_s"] for item in group["items"]])
        for group in groups
    ], 0.95))
    ranked = []
    for group in groups:
        if np.mean([
            item["time_to_goal_s"] for item in group["items"]
        ]) > time_limit:
            continue
        features = [item["features"] for item in group["items"]]
        spread = float(np.mean([
            np.sqrt(np.mean(np.sum(
                (
                    features[left]["resampled_path"]
                    - features[right]["resampled_path"]
                ) ** 2,
                axis=1,
            ))) / direct
            for left, right in combinations(range(len(features)), 2)
        ]))
        nonstraight = float(np.mean([
            feature["transverse_rms_m"] / direct
            + max(feature["path_length_excess_ratio"], 0.0)
            + 0.25 * feature["interaction_fraction"]
            for feature in features
        ]))
        ranked.append({
            **group,
            "score": nonstraight + spread,
            "nonstraight_score": nonstraight,
            "cross_gamma_spread": spread,
        })
    return sorted(
        ranked,
        key=lambda row: (-row["score"], str(row["key"])),
    )


def _expert_groups(demo_dir: Path):
    manifest = json.loads((demo_dir / "manifest.json").read_text())
    config = load_config(demo_dir / "resolved_config.json")
    by_scene = {}
    for row in manifest["runs"]:
        if row.get("accepted") and row.get("success"):
            by_scene.setdefault(str(row["scene_id"]), {})[
                float(row["gamma"])
            ] = row
    shared = [
        scene_id for scene_id, rows in by_scene.items()
        if all(gamma in rows for gamma in GAMMAS)
    ]
    groups = []
    for scene_id in shared:
        items = []
        for gamma in GAMMAS:
            row = by_scene[scene_id][gamma]
            data = np.load(demo_dir / row["file"])
            path = np.asarray(data["dense_positions"], float)
            spheres = np.asarray(
                data["spheres"] if "spheres" in data.files else (),
                float,
            ).reshape(-1, 4)
            cylinders = np.asarray(
                data["cylinders"] if "cylinders" in data.files else (),
                float,
            ).reshape(-1, 3)
            items.append({
                "gamma": gamma,
                "path": path,
                "spheres": spheres,
                "cylinders": cylinders,
                "status": "SUCCESS",
                "time_to_goal_s": float(row["time_to_goal_s"]),
                "seed": int(row["seed"]),
                "scene_hash": str(row["scene_hash"]),
                "features": _features(
                    path,
                    config.taskspace.start,
                    config.taskspace.goal,
                    spheres,
                    cylinders,
                ),
            })
        groups.append({"key": scene_id, "items": items})
    return _rank_groups(
        groups, config.taskspace.start, config.taskspace.goal,
    ), config, manifest


@torch.no_grad()
def _policy_groups(
    policy,
    manifest: dict,
    summary_key: str,
    rows_key: str,
    config_path: Path,
    episodes_key: str,
    seed_key: str,
    device: str,
):
    del summary_key  # The complete summary remains manifest-bound for plotting.
    config = load_config(config_path)
    audit_rows = manifest[rows_key]
    by_episode = {}
    for row in audit_rows:
        if row["status"] == "SUCCESS":
            by_episode.setdefault(int(row["episode"]), {})[
                float(row["gamma"])
            ] = row
    shared = [
        episode for episode, rows in by_episode.items()
        if all(gamma in rows for gamma in GAMMAS)
    ]
    scenes = path_focused_scene_bank(
        config,
        int(manifest[episodes_key]),
        seed=int(manifest[seed_key]),
    )
    groups = []
    for episode in shared:
        scene = scenes[episode]
        scene_config = config_for_scene(config, scene)
        items = []
        for gamma in GAMMAS:
            declared = by_episode[episode][gamma]
            if str(declared["scene_hash"]) != str(scene.scene_hash):
                raise RuntimeError("audit scene lineage does not reproduce")
            result = raw_reference_rollout(
                policy,
                scene_config,
                gamma,
                int(declared["seed"]),
                device=device,
                sampling_temperature=1.0,
            )
            if result["status"] != declared["status"]:
                raise RuntimeError(
                    "checkpoint replay does not reproduce audit status"
                )
            path = _dense_path(result)
            spheres = np.asarray(scene.spheres, float).reshape(-1, 4)
            cylinders = np.asarray(scene.cylinders, float).reshape(-1, 3)
            items.append({
                "gamma": gamma,
                "path": path,
                "spheres": spheres,
                "cylinders": cylinders,
                "status": result["status"],
                "time_to_goal_s": float(result["time_to_goal_s"]),
                "seed": int(declared["seed"]),
                "scene_hash": str(scene.scene_hash),
                "features": _features(
                    path,
                    config.taskspace.start,
                    config.taskspace.goal,
                    spheres,
                    cylinders,
                ),
            })
        groups.append({"key": int(episode), "items": items})
    return _rank_groups(
        groups, config.taskspace.start, config.taskspace.goal,
    ), config


def _draw_cylinders(axis, cylinders, z_bounds, physical_radius):
    theta = np.linspace(0.0, 2.0 * np.pi, 28)
    z = np.asarray(z_bounds, float)
    theta_grid, z_grid = np.meshgrid(theta, z)
    for x, y, modeled_radius in np.asarray(cylinders, float):
        axis.plot_surface(
            x + modeled_radius * np.cos(theta_grid),
            y + modeled_radius * np.sin(theta_grid),
            z_grid,
            color="#6aaed6",
            alpha=0.12,
            linewidth=0.0,
        )
        axis.plot_surface(
            x + physical_radius * np.cos(theta_grid),
            y + physical_radius * np.sin(theta_grid),
            z_grid,
            color="#8d9295",
            alpha=0.72,
            linewidth=0.0,
        )


def _draw_spheres(axis, spheres, physical_radius):
    azimuth = np.linspace(0.0, 2.0 * np.pi, 25)
    polar = np.linspace(0.0, np.pi, 13)
    unit_x = np.outer(np.cos(azimuth), np.sin(polar))
    unit_y = np.outer(np.sin(azimuth), np.sin(polar))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))
    for x, y, z, modeled_radius in np.asarray(spheres, float):
        axis.plot_wireframe(
            x + modeled_radius * unit_x,
            y + modeled_radius * unit_y,
            z + modeled_radius * unit_z,
            color="#6aaed6",
            alpha=0.22,
            linewidth=0.35,
        )
        axis.plot_surface(
            x + physical_radius * unit_x,
            y + physical_radius * unit_y,
            z + physical_radius * unit_z,
            color="#8d9295",
            alpha=0.68,
            linewidth=0.0,
        )


def _draw_obstacles(axis, item, config):
    physical = float(
        config.raw["scene_randomization"]["physical_radius_m"]
    )
    if len(item["cylinders"]):
        _draw_cylinders(
            axis,
            item["cylinders"],
            config.taskspace.bounds[2],
            physical,
        )
    if len(item["spheres"]):
        _draw_spheres(axis, item["spheres"], physical)


def _style_3d(axis, config):
    axis.set(
        xlim=config.taskspace.bounds[0],
        ylim=config.taskspace.bounds[1],
        zlim=config.taskspace.bounds[2],
        xlabel=r"$x$ [m]",
        ylabel=r"$y$ [m]",
        zlabel=r"$z$ [m]",
    )
    axis.view_init(elev=24, azim=-58)
    axis.grid(alpha=0.18)


def _plot_gallery(selections, configs, output: Path):
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / (len(GAMMAS) - 1)
        )
        for index, gamma in enumerate(GAMMAS)
    }
    fig = plt.figure(figsize=(17.2, 12.5))
    for row_index, (selection, config) in enumerate(zip(selections, configs)):
        for column_index, gamma in enumerate(GAMMAS):
            axis = fig.add_subplot(
                len(selections),
                len(GAMMAS),
                row_index * len(GAMMAS) + column_index + 1,
                projection="3d",
            )
            item = selection["items"][column_index]
            _draw_obstacles(axis, item, config)
            path = item["path"]
            axis.plot(
                path[:, 0], path[:, 1], path[:, 2],
                color=colors[gamma], linewidth=2.4,
            )
            axis.scatter(
                *config.taskspace.start[:3],
                marker="s", s=25, color="black",
            )
            axis.scatter(
                *config.taskspace.goal,
                marker="*", s=90, color="#f1c40f", edgecolor="black",
            )
            _style_3d(axis, config)
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$", fontsize=18)
            if column_index == 0:
                axis.text2D(
                    -0.18, 0.5, ROW_LABELS[row_index],
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=14,
                )
    fig.suptitle(
        "Curated successful trajectories—not a rate estimate",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.975))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_overlays(selections, configs, output: Path):
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / (len(GAMMAS) - 1)
        )
        for index, gamma in enumerate(GAMMAS)
    }
    fig = plt.figure(figsize=(15.5, 12.0))
    for row_index, (selection, config) in enumerate(zip(selections, configs)):
        axis3d = fig.add_subplot(3, 3, row_index * 3 + 1, projection="3d")
        representative = selection["items"][0]
        _draw_obstacles(axis3d, representative, config)
        axis_xy = fig.add_subplot(3, 3, row_index * 3 + 2)
        axis_sz = fig.add_subplot(3, 3, row_index * 3 + 3)
        start = np.asarray(config.taskspace.start[:3], float)
        goal = np.asarray(config.taskspace.goal, float)
        direction = (goal - start) / np.linalg.norm(goal - start)
        for item in selection["items"]:
            gamma = item["gamma"]
            path = item["path"]
            axis3d.plot(
                path[:, 0], path[:, 1], path[:, 2],
                color=colors[gamma], linewidth=2.2,
                label=rf"$\gamma={gamma:g}$",
            )
            axis_xy.plot(
                path[:, 0], path[:, 1],
                color=colors[gamma], linewidth=2.0,
            )
            longitudinal = (path - start) @ direction
            axis_sz.plot(
                longitudinal, path[:, 2],
                color=colors[gamma], linewidth=2.0,
            )
        axis3d.scatter(*start, marker="s", s=24, color="black")
        axis3d.scatter(
            *goal, marker="*", s=85, color="#f1c40f", edgecolor="black",
        )
        _style_3d(axis3d, config)
        axis3d.set_title(ROW_LABELS[row_index], fontsize=14)
        axis_xy.set(
            xlim=config.taskspace.bounds[0],
            ylim=config.taskspace.bounds[1],
            xlabel=r"$x$ [m]",
            ylabel=r"$y$ [m]",
            title="Top view",
        )
        axis_xy.scatter(*start[:2], marker="s", s=20, color="black")
        axis_xy.scatter(
            *goal[:2], marker="*", s=70,
            color="#f1c40f", edgecolor="black",
        )
        axis_xy.grid(alpha=0.2)
        axis_sz.set(
            xlabel="Longitudinal distance [m]",
            ylabel=r"$z$ [m]",
            ylim=config.taskspace.bounds[2],
            title="Goal-aligned side view",
        )
        axis_sz.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], color=colors[gamma], linewidth=2.2,
                   label=rf"$\gamma={gamma:g}$")
        for gamma in GAMMAS
    ]
    handles.extend([
        Patch(facecolor="#8d9295", alpha=0.72, label="physical obstacle"),
        Patch(facecolor="#6aaed6", alpha=0.14, label="inflated model"),
    ])
    fig.legend(
        handles=handles, loc="upper center", ncol=6,
        bbox_to_anchor=(0.5, 0.985), frameon=False,
    )
    fig.suptitle(
        "Curated successful trajectory overlays—not a rate estimate",
        fontsize=19,
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_audit(manifest: dict, output: Path):
    summaries = (
        ("Cylinder ID", manifest["raw_audit_summary"]),
        ("Sphere OOD", manifest["ood_raw_audit_summary"]),
    )
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / (len(GAMMAS) - 1)
        )
        for index, gamma in enumerate(GAMMAS)
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    width = 0.18
    metrics = (("SR", "SR"), ("CR", "CR"), ("OOB", "OOB"),
               ("window_validity", "Validity"))
    for axis, (title, rows) in zip(axes, summaries):
        x = np.arange(len(GAMMAS))
        for index, (key, label) in enumerate(metrics):
            axis.bar(
                x + (index - 1.5) * width,
                [row[key] for row in rows],
                width,
                label=label,
            )
        axis.set_xticks(x, [rf"$\gamma={gamma:g}$" for gamma in GAMMAS])
        axis.set_ylim(0.0, 1.03)
        axis.set_title(title)
        axis.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Rate")
    axes[1].legend(ncol=2, fontsize=9)
    fig.suptitle(
        "Unbiased raw temperature-1 audit | 100 episodes per gamma",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _serializable_selection(selection: dict) -> dict:
    return {
        "key": selection["key"],
        "score": selection["score"],
        "nonstraight_score": selection["nonstraight_score"],
        "cross_gamma_spread": selection["cross_gamma_spread"],
        "items": [{
            "gamma": item["gamma"],
            "seed": item["seed"],
            "scene_hash": item["scene_hash"],
            "status": item["status"],
            "time_to_goal_s": item["time_to_goal_s"],
            "path_length_excess_ratio": (
                item["features"]["path_length_excess_ratio"]
            ),
            "transverse_rms_m": item["features"]["transverse_rms_m"],
            "interaction_fraction": item["features"]["interaction_fraction"],
        } for item in selection["items"]],
    }


def _choose(ranked: list[dict], requested):
    if requested is None:
        return ranked[0]
    for row in ranked:
        if str(row["key"]) == str(requested):
            return row
    raise ValueError(f"requested qualitative key {requested!r} is unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--ood-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expert-scene-id")
    parser.add_argument("--id-episode", type=int)
    parser.add_argument("--ood-episode", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.pretrain_dir / "pretrained.pt"
    manifest_path = args.pretrain_dir / "pretrain_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    policy = load_lab_reference_policy(checkpoint).to(args.device).eval()

    expert_ranked, expert_config, expert_manifest = _expert_groups(
        args.demo_dir,
    )
    id_ranked, id_config = _policy_groups(
        policy,
        manifest,
        "raw_audit_summary",
        "raw_audit",
        args.demo_dir / "resolved_config.json",
        "raw_audit_episodes_per_gamma",
        "raw_audit_seed",
        args.device,
    )
    ood_ranked, ood_config = _policy_groups(
        policy,
        manifest,
        "ood_raw_audit_summary",
        "ood_raw_audit",
        args.ood_config,
        "ood_raw_audit_episodes_per_gamma",
        "ood_raw_audit_seed",
        args.device,
    )
    selections = (
        _choose(expert_ranked, args.expert_scene_id),
        _choose(id_ranked, args.id_episode),
        _choose(ood_ranked, args.ood_episode),
    )
    configs = (expert_config, id_config, ood_config)

    _plot_gallery(
        selections,
        configs,
        args.output_dir / "curated_preexpansion_gallery.png",
    )
    _plot_overlays(
        selections,
        configs,
        args.output_dir / "curated_preexpansion_overlays.png",
    )
    _plot_audit(
        manifest,
        args.output_dir / "unbiased_pretrained_audit.png",
    )
    provenance = {
        "status": "PATH_FOCUSED_PREEXPANSION_VISUALIZATION_COMPLETE",
        "caption": (
            "Curated successful qualitative examples selected by a fixed "
            "geometric score; not an estimate of SR, CR, OOB, or Validity."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "pretrain_manifest": str(manifest_path.resolve()),
        "pretrain_manifest_sha256": _sha256(manifest_path),
        "expert_manifest_sha256": _sha256(args.demo_dir / "manifest.json"),
        "ood_config": str(args.ood_config.resolve()),
        "ood_config_sha256": _sha256(args.ood_config),
        "selection_pool_sizes": {
            "expert_all_gamma_success_scenes": len(expert_ranked),
            "pretrained_id_all_gamma_success_scenes": len(id_ranked),
            "pretrained_ood_all_gamma_success_scenes": len(ood_ranked),
        },
        "selections": {
            label: _serializable_selection(selection)
            for label, selection in zip(ROW_LABELS, selections)
        },
        "unbiased_audit": {
            "episodes_per_gamma": manifest["raw_audit_episodes_per_gamma"],
            "cylinder_id": manifest["raw_audit_summary"],
            "sphere_ood": manifest["ood_raw_audit_summary"],
        },
        "source_archive": {
            "accepted_runs": len(expert_manifest["runs"]),
            "attempted_runs": len(expert_manifest["attempts"]),
        },
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance["selections"], indent=2), flush=True)
    print(f"[output] {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
