"""Fixed-seed raw evaluation for randomized three-sphere lab expansion."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ObstacleConfig
from .environment import TaskEnvironment
from .lab_clutter import ClutterScene, start_goal_path_diagnostics
from .lab_clutter_expansion import (
    LAB_CLUTTER_SCENE_SCHEMA,
    LabClutterExpansionPolicyAdapter,
    LabClutterSphereExpansionTask,
    RandomThreeSphereScene,
    scene_sha256,
)
from .lab_reference_flow_task import raw_reference_rollout
from .lab_visual_flow import LAB_VISUAL_SCHEMA, load_lab_reference_policy


LAB_CLUTTER_TASK_PROFILE = (
    "minhyuk_lab_random_three_sphere_visual_expansion"
)
EVALUATION_SCENE_SEED_STRIDE = 1009
START_PROBE_SCENE_SEED_OFFSET = 17


def is_lab_clutter_evaluation_manifest(manifest: dict) -> bool:
    """Return whether a completed lab manifest has the clutter eval contract."""
    if manifest.get("task_profile") != LAB_CLUTTER_TASK_PROFILE:
        return False
    conditioning = manifest.get("lab_conditioning")
    if (
        not isinstance(conditioning, dict)
        or conditioning.get("context_schema") != LAB_VISUAL_SCHEMA
    ):
        raise ValueError(
            "random-three-sphere expansion requires the visual lab context schema"
        )
    ledger = manifest.get("lab_scene_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError(
            "random-three-sphere expansion manifest requires a nonempty scene ledger"
        )
    if any(
        not isinstance(row, dict)
        or row.get("schema") != LAB_CLUTTER_SCENE_SCHEMA
        for row in ledger
    ):
        raise ValueError(
            "random-three-sphere expansion scene ledger schema mismatch"
        )
    return True


def _scene_record(
    scene_spec: RandomThreeSphereScene,
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
    scene_spec = RandomThreeSphereScene.from_config(config)
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
        "schema": LAB_CLUTTER_SCENE_SCHEMA,
        "evaluation_seed": int(domain_seed),
        "configured_sampler_domain_seed": int(scene_spec.domain_seed),
        "sampler": {
            "implementation": "RandomThreeSphereScene.sample",
            "rng": (
                "numpy.random.default_rng("
                "SeedSequence([configured_sampler_domain_seed, scene_seed]))"
            ),
            "obstacle_family": "spheres",
            "count": 3,
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


def _checkpoint_policy(pretrain_dir: Path, expansion: Path, round_i: int):
    policy = load_lab_reference_policy(pretrain_dir / "pretrained.pt")
    payload = torch.load(
        expansion / f"checkpoint_{round_i:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy.load_state_dict(payload["model"], strict=True)
    return policy.eval()


def _summarize(rows: list[dict]) -> dict:
    success = [row for row in rows if row["status"] == "SUCCESS"]
    clearance = [
        row["min_clearance_m"] for row in success
        if row["min_clearance_m"] is not None
    ]
    times = [
        row["time_to_goal_s"] for row in success
        if row["time_to_goal_s"] is not None
    ]
    count = len(rows)
    return {
        "episodes": count,
        "SR": float(np.mean([
            row["status"] == "SUCCESS" for row in rows
        ])),
        "CR": float(np.mean([
            row["status"] == "COLLISION" for row in rows
        ])),
        "OOB": float(np.mean([
            row["status"] == "OOB" for row in rows
        ])),
        "timeout": float(np.mean([
            row["status"] == "TIMEOUT" for row in rows
        ])),
        "window_validity": float(np.mean([
            row["window_validity"] for row in rows
        ])),
        "successful_min_clearance_m": (
            float(np.mean(clearance)) if clearance else None
        ),
        "successful_time_to_goal_s": (
            float(np.mean(times)) if times else None
        ),
    }


def _raw_rows(policy, config, gammas, scene_bank, domain_seed: int):
    rows = []
    for gamma in gammas:
        for scene in scene_bank:
            episode = int(scene["episode"])
            spheres = np.asarray(scene["spheres"], np.float32)
            scene_config = _scene_config(config, spheres)
            result = raw_reference_rollout(
                policy,
                scene_config,
                float(gamma),
                int(domain_seed) + 37 * episode,
                sampling_temperature=1.0,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(episode),
                "scene_seed": int(scene["scene_seed"]),
                "scene_hash": str(scene["scene_hash"]),
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
):
    task = LabClutterSphereExpansionTask(
        config,
        context_schema=policy.context_schema,
        tight_corridor=True,
    )
    wrapped = LabClutterExpansionPolicyAdapter(policy)
    rows = []
    for gamma_index, gamma in enumerate(gammas):
        scene_seed = int(scene["scene_seed"])
        state = task.reset(float(gamma), 0, scene_seed)
        if state["scene_hash"] != scene["scene_hash"]:
            raise RuntimeError("start-probe scene does not match the fixed scene bank")
        context = task.context(state, float(gamma))
        generator = torch.Generator().manual_seed(
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
            axis.plot(
                rounds, values, marker="o", color=colors[gamma],
                label=rf"$\gamma={gamma:g}$",
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


def evaluate_lab_clutter_expansion(
    args,
    config,
    pretrain_manifest,
    manifest,
):
    """Evaluate raw temperature-1 policy on one disjoint fixed scene bank."""
    del pretrain_manifest
    gammas = [float(value) for value in config.data.gammas]
    scene_bank = _evaluation_scene_provenance(
        config,
        int(args.episodes),
        int(args.seed),
        manifest,
    )
    total_rounds = int(manifest["config"]["rounds"])
    rounds = sorted({
        0,
        *range(0, total_rounds + 1, int(args.stride)),
        total_rounds,
    })
    per_round_rows = {}
    summaries = {}
    probes = {}
    for round_i in rounds:
        policy = _checkpoint_policy(
            args.pretrain_dir, args.expansion, round_i,
        )
        rows = _raw_rows(
            policy,
            config,
            gammas,
            scene_bank["scenes"],
            int(args.seed),
        )
        probe_rows = _start_probe(
            policy,
            config,
            gammas,
            int(args.probe_samples),
            scene_bank["start_probe_scene"],
        )
        per_round_rows[round_i] = rows
        probes[str(round_i)] = probe_rows
        per_gamma = {
            f"{gamma:g}": _summarize([
                row for row in rows if row["gamma"] == gamma
            ])
            for gamma in gammas
        }
        summaries[str(round_i)] = {
            "pooled": _summarize(rows),
            "per_gamma": per_gamma,
            "start_probe_validity": float(np.mean([
                row["valid"] for row in probe_rows
            ])),
        }
        pooled = summaries[str(round_i)]["pooled"]
        print(
            f"round {round_i:3d}: SR={pooled['SR']:.3f} "
            f"CR={pooled['CR']:.3f} OOB={pooled['OOB']:.3f} "
            f"Vwin={pooled['window_validity']:.3f}",
            flush=True,
        )

    output = args.expansion / "eval"
    output.mkdir(exist_ok=True)
    slim = {
        str(round_i): [{
            key: value for key, value in row.items()
            if key not in {
                "states", "controls", "applied_controls", "dense_steps",
            }
        } for row in rows]
        for round_i, rows in per_round_rows.items()
    }
    (output / "raw_eval.json").write_text(json.dumps({
        "status": "LAB_CLUTTER_RAW_TEMPERATURE1_EVALUATION_COMPLETE",
        "sampling_temperature": 1.0,
        "sigma_tilt_used": False,
        "scene_bank": scene_bank,
        "summary": summaries,
        "rows": slim,
        "start_probe_rows": probes,
    }, indent=2) + "\n")
    if not args.metrics_only:
        _plot_curves(
            summaries, gammas, output / "raw_curves.png",
        )
        gallery_rounds = sorted({
            rounds[0], rounds[len(rounds) // 2], rounds[-1],
        })
        _plot_gallery(
            per_round_rows,
            gallery_rounds,
            gammas,
            config,
            output / "raw_gallery.png",
        )
    print("[outputs]", output, flush=True)
