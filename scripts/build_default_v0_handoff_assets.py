#!/usr/bin/env python3
"""Reproduce the paired Stage-1 ``default_v0`` handoff comparison.

This is evaluation-only.  It replays one declared raw temperature-one seed
with the pretrained policy and the packaged r3 checkpoint, writes the exact
trajectory archives, and renders a side-by-side gamma overlay.  It does not
run expansion, uncertainty tilting, a verifier controller, or fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402
from safe_mppi.lab_visual_flow import load_lab_reference_policy  # noqa: E402


GAMMAS = (0.1, 0.3, 0.5, 1.0)
COLORS = {
    gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / 3.0)
    for index, gamma in enumerate(GAMMAS)
}
EXPECTED_PRETRAINED_SHA256 = (
    "cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff"
)
EXPECTED_R3_SHA256 = (
    "dfa4b72a04e3b892bbbe7ea152b1f24a1c4a086a4ccd653f0bf67183027fd69c"
)
EXPECTED_SEED = 91_074


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense_path(result: dict) -> np.ndarray:
    states = np.asarray(result["states"], np.float32)
    dense = np.asarray(result["dense_steps"], np.float32)
    if not len(dense):
        return states[:, :3]
    return np.concatenate([states[:1, :3], dense.reshape(-1, 3)])


def _failure_point(result: dict, config) -> np.ndarray | None:
    if result["status"] == "SUCCESS":
        return None
    path = _dense_path(result)
    env = TaskEnvironment(config)
    if result["status"] == "COLLISION":
        values = env.obstacle_clearance(path)
        indices = np.flatnonzero(np.isfinite(values) & (values < 0.0))
    elif result["status"] == "OOB":
        indices = np.flatnonzero(~env.inside_taskspace(path))
    else:
        indices = np.asarray([len(path) - 1], dtype=np.int64)
    return path[int(indices[0]) if len(indices) else -1]


def _draw_sphere(axis, center: np.ndarray, modeled_radius: float) -> None:
    physical_radius = 0.254
    azimuth = np.linspace(0.0, 2.0 * np.pi, 32)
    elevation = np.linspace(0.0, np.pi, 18)
    unit_x = np.outer(np.cos(azimuth), np.sin(elevation))
    unit_y = np.outer(np.sin(azimuth), np.sin(elevation))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(elevation))
    axis.plot_surface(
        center[0] + physical_radius * unit_x,
        center[1] + physical_radius * unit_y,
        center[2] + physical_radius * unit_z,
        color="#8e9398", alpha=0.78, linewidth=0.0, shade=True,
    )
    axis.plot_wireframe(
        center[0] + modeled_radius * unit_x,
        center[1] + modeled_radius * unit_y,
        center[2] + modeled_radius * unit_z,
        color="#2a86b8", alpha=0.32, linewidth=0.45,
        rstride=3, cstride=3,
    )


def _style_axis(axis, config, title: str) -> None:
    origin = np.asarray(config.taskspace.origin, float)
    upper = origin + np.asarray(config.taskspace.size, float)
    axis.set_xlim(origin[0], upper[0])
    axis.set_ylim(origin[1], upper[1])
    axis.set_zlim(origin[2], upper[2])
    axis.set_xlabel(r"$x$ [m]", labelpad=8)
    axis.set_ylabel(r"$y$ [m]", labelpad=8)
    axis.set_zlabel(r"$z$ [m]", labelpad=5)
    axis.view_init(elev=21.0, azim=-56.0)
    axis.set_box_aspect((3.8, 3.5, 2.35))
    axis.set_title(title, fontsize=17, pad=16)
    axis.grid(True, alpha=0.24)


def _save_archive(
    path: Path,
    result: dict,
    *,
    checkpoint: Path,
    config_path: Path,
    gamma: float,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=np.asarray(result["states"], np.float32),
        controls=np.asarray(result["controls"], np.float32),
        executed_controls=np.asarray(result["applied_controls"], np.float32),
        dense_positions=_dense_path(result),
        gamma=np.asarray(gamma, np.float32),
        seed=np.asarray(seed, np.int64),
        status=np.asarray(result["status"]),
        route_mode=np.asarray(result["mode"]),
        sampling_temperature=np.asarray(1.0, np.float32),
        checkpoint_sha256=np.asarray(_sha256(checkpoint)),
        config_sha256=np.asarray(_sha256(config_path)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    handoff = ROOT / "flow_deployment/minhyuk_stage1_handoff"
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=handoff / "checkpoints/hp100_t128_d3.pt",
    )
    parser.add_argument(
        "--expanded-r3",
        type=Path,
        default=handoff / "checkpoints/stage1_default_v0_best_r3.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_ball_stage1_t128.json",
    )
    parser.add_argument("--output-dir", type=Path, default=handoff)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    args = parser.parse_args()

    if _sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("pretrained checkpoint SHA-256 mismatch")
    if _sha256(args.expanded_r3) != EXPECTED_R3_SHA256:
        raise ValueError("packaged r3 checkpoint SHA-256 mismatch")
    if args.seed != EXPECTED_SEED:
        raise ValueError(f"canonical paired seed must be {EXPECTED_SEED}")

    config = load_config(args.config)
    policies = {
        "pretrained r0": load_lab_reference_policy(args.pretrained).eval(),
        "default_v0 r3": load_lab_reference_policy(args.expanded_r3).eval(),
    }
    checkpoints = {
        "pretrained r0": args.pretrained,
        "default_v0 r3": args.expanded_r3,
    }
    expected_status = {
        "pretrained r0": "COLLISION",
        "default_v0 r3": "SUCCESS",
    }
    output = args.output_dir.resolve()
    archives = output / "trajectory_archives"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    replay: dict[tuple[str, float], dict] = {}

    for stage, policy in policies.items():
        tag = "r0" if stage.endswith("r0") else "r3"
        for gamma in GAMMAS:
            result = raw_reference_rollout(
                policy,
                config,
                gamma,
                args.seed,
                device="cpu",
                sampling_temperature=1.0,
            )
            if result["status"] != expected_status[stage]:
                raise RuntimeError(
                    f"{stage} gamma={gamma:g}: expected {expected_status[stage]}, "
                    f"got {result['status']}"
                )
            replay[(stage, gamma)] = result
            gamma_label = "1" if gamma == 1.0 else f"{gamma:g}"
            archive = (
                archives
                / f"stage1_default_v0_{tag}_g{gamma_label}_s{args.seed}_"
                  f"{result['status'].lower()}.npz"
            )
            _save_archive(
                archive,
                result,
                checkpoint=checkpoints[stage],
                config_path=args.config,
                gamma=gamma,
                seed=args.seed,
            )
            rows.append({
                "checkpoint": tag,
                "gamma": gamma,
                "seed": args.seed,
                "status": result["status"],
                "route_mode": result["mode"],
                "minimum_clearance_m": float(result["min_clearance_m"]),
                "time_to_goal_s": (
                    None if result["time_to_goal_s"] is None
                    else float(result["time_to_goal_s"])
                ),
                "window_validity": float(result["window_validity"]),
                "archive": str(archive.relative_to(output)),
                "archive_sha256": _sha256(archive),
            })

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 13,
    })
    figure = plt.figure(figsize=(15.2, 7.0))
    sphere = np.asarray(config.obstacles.spheres[0], float)
    start = np.asarray(config.taskspace.start[:3], float)
    goal = np.asarray(config.taskspace.goal, float)
    for index, stage in enumerate(policies):
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        _draw_sphere(axis, sphere[:3], float(sphere[3]))
        for gamma in GAMMAS:
            result = replay[(stage, gamma)]
            path = _dense_path(result)
            axis.plot(
                path[:, 0], path[:, 1], path[:, 2],
                color=COLORS[gamma], linewidth=3.0, alpha=0.96,
            )
            axis.scatter(
                path[::10, 0], path[::10, 1], path[::10, 2],
                color=[COLORS[gamma]], s=9, alpha=0.75, depthshade=False,
            )
            failure = _failure_point(result, config)
            if failure is not None:
                axis.scatter(
                    *failure, marker="x", s=85, color="#b2182b",
                    linewidth=2.2, depthshade=False,
                )
        axis.scatter(*start, marker="s", s=55, color="black", depthshade=False)
        axis.scatter(
            *goal, marker="*", s=180, color="#f3c623",
            edgecolor="#403a12", linewidth=0.8, depthshade=False,
        )
        title = (
            "Pretrained r0: all four collide"
            if stage.endswith("r0")
            else "default_v0 r3: all four succeed"
        )
        _style_axis(axis, config, title)

    legend = [
        Line2D([0], [0], color=COLORS[gamma], linewidth=3.0,
               label=rf"$\gamma={gamma:g}$")
        for gamma in GAMMAS
    ]
    legend.extend([
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black",
               markersize=7, label="start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#f3c623",
               markeredgecolor="#403a12", markersize=12, label="goal"),
        Line2D([0], [0], marker="x", color="#b2182b", linestyle="none",
               markersize=8, markeredgewidth=2, label="collision"),
    ])
    figure.legend(
        handles=legend, loc="upper center", ncol=7, frameon=False,
        bbox_to_anchor=(0.5, 0.985), fontsize=13,
    )
    figure.suptitle(
        "Stage 1 raw-policy transfer on the fixed sphere",
        fontsize=19, y=1.025,
    )
    figure.text(
        0.5, 0.018,
        "Paired raw temperature-one seed 91074; no uncertainty tilt, "
        "verifier controller, or fallback",
        ha="center", fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.92))
    png = assets / "single_sphere_default_v0_r0_r3_seed91074_gamma_overlay.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    record = {
        "status": "DEFAULT_V0_PAIRED_OVERLAY_COMPLETE",
        "scientific_scope": (
            "Selected qualitative paired seed; aggregate claims come from "
            "default_v0/raw_eval_m20.json."
        ),
        "contract": {
            "seed": args.seed,
            "episode": 2,
            "sampling_temperature": 1.0,
            "raw_sampling": True,
            "uncertainty_tilting": False,
            "verifier_controller": False,
            "fallback": False,
            "device": "cpu",
        },
        "inputs": {
            "pretrained_checkpoint": str(args.pretrained.relative_to(ROOT)),
            "pretrained_checkpoint_sha256": _sha256(args.pretrained),
            "expanded_r3_checkpoint": str(args.expanded_r3.relative_to(ROOT)),
            "expanded_r3_checkpoint_sha256": _sha256(args.expanded_r3),
            "config": str(args.config.relative_to(ROOT)),
            "config_sha256": _sha256(args.config),
        },
        "figure": {
            "png": str(png.relative_to(output)),
            "png_sha256": _sha256(png),
            "pdf": str(pdf.relative_to(output)),
            "pdf_sha256": _sha256(pdf),
        },
        "rows": rows,
    }
    destination = output / "default_v0/paired_overlay_seed91074.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({
        "status": record["status"],
        "png": str(png),
        "record": str(destination),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
