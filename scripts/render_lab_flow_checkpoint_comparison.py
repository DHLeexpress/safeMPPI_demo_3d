#!/usr/bin/env python3
"""Render a qualitative pretrained-versus-expanded successful-reference GIF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_deployment.lab_pretrained import (  # noqa: E402
    load_lab_reference_policy,
    sha256_file,
)
from safe_mppi.ball_flow_theta import start_goal_frame  # noqa: E402
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402
from safe_mppi.lab_reference_flow_task import raw_reference_rollout  # noqa: E402


def _rollouts(checkpoint, config, gammas, seeds):
    policy = load_lab_reference_policy(checkpoint).eval()
    rows = []
    for gamma, seed in zip(gammas, seeds):
        row = raw_reference_rollout(
            policy,
            config,
            float(gamma),
            int(seed),
            sampling_temperature=1.0,
        )
        if row["status"] != "SUCCESS":
            raise RuntimeError(
                f"{checkpoint.name}: gamma={gamma:g}, seed={seed} "
                f"ended {row['status']}, not SUCCESS"
            )
        rows.append({
            **row,
            "gamma": float(gamma),
            "seed": int(seed),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lab_ball_pretrain.json",
    )
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=(0.1, 0.3, 0.5, 1.0),
    )
    parser.add_argument(
        "--pretrained-seeds",
        type=int,
        nargs="+",
        default=(191111, 191111, 191111, 191111),
    )
    parser.add_argument(
        "--expanded-seeds",
        type=int,
        nargs="+",
        default=(191000, 191000, 191000, 191222),
    )
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (
        len(args.gammas)
        == len(args.pretrained_seeds)
        == len(args.expanded_seeds)
    ):
        parser.error("gamma and seed lists must have equal lengths")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = load_config(args.config)
    env = TaskEnvironment(config)
    pretrain_rows = _rollouts(
        args.pretrained, config, args.gammas, args.pretrained_seeds,
    )
    expanded_rows = _rollouts(
        args.expanded, config, args.gammas, args.expanded_seeds,
    )
    frame = start_goal_frame(env)
    center = (env.spheres[0, :3] - env.start[:3]) @ frame
    goal = (env.goal - env.start[:3]) @ frame
    radius = float(env.spheres[0, 3])
    angle = np.linspace(0.0, 2.0 * np.pi, 120)
    paths = []
    for rows in (pretrain_rows, expanded_rows):
        paths.append([
            (
                np.concatenate([
                    row["states"][:1, :3],
                    row["dense_steps"].reshape(-1, 3),
                ])
                - env.start[:3]
            ) @ frame
            for row in rows
        ])
    frames = max(2, int(args.max_frames))

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
    })
    fig, axes = plt.subplots(
        2, len(args.gammas),
        figsize=(3.5 * len(args.gammas), 6.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    lines = []
    row_names = ("Pretrained", "Expanded r20")
    colors = ("#2457a6", "#d1492e")
    for row_index, rows in enumerate((pretrain_rows, expanded_rows)):
        line_row = []
        for column_index, row in enumerate(rows):
            axis = axes[row_index, column_index]
            axis.fill(
                center[0] + radius * np.cos(angle),
                center[2] + radius * np.sin(angle),
                color="#a7adb3",
                alpha=0.55,
            )
            axis.scatter(0.0, 0.0, marker="s", s=22, color="black")
            axis.scatter(
                goal[0], goal[2], marker="*", s=85,
                color="#ffca28", edgecolor="#5f4b00", zorder=8,
            )
            (line,) = axis.plot([], [], color=colors[row_index], lw=2.2)
            line_row.append(line)
            axis.set_title(
                rf"$\gamma={row['gamma']:g}$, {row['mode']}"
                f"\nseed {row['seed']}",
            )
            axis.grid(alpha=0.22)
            if column_index == 0:
                axis.set_ylabel(f"{row_names[row_index]}\nvertical [m]")
            if row_index == 1:
                axis.set_xlabel("start-goal axis [m]")
        lines.append(line_row)
    corners = np.asarray([
        [x, y, z]
        for x in env.bounds[0]
        for y in env.bounds[1]
        for z in env.bounds[2]
    ])
    local_corners = (corners - env.start[:3]) @ frame
    for axis in axes.flat:
        axis.set_xlim(local_corners[:, 0].min(), local_corners[:, 0].max())
        axis.set_ylim(local_corners[:, 2].min(), local_corners[:, 2].max())
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle(
        "Selected successful temperature-1 references (qualitative only)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=12)
    with writer.saving(fig, str(args.output), dpi=105):
        for frame_index in range(frames):
            for row_index in range(2):
                for column_index in range(len(args.gammas)):
                    path = paths[row_index][column_index]
                    stop = max(
                        1,
                        int(
                            np.ceil(
                                (frame_index + 1)
                                * len(path)
                                / frames
                            )
                        ),
                    )
                    lines[row_index][column_index].set_data(
                        path[:stop, 0], path[:stop, 2],
                    )
            writer.grab_frame()
        for _ in range(18):
            writer.grab_frame()
    plt.close(fig)

    manifest = {
        "status": "LAB_FLOW_SELECTED_SUCCESS_COMPARISON_COMPLETE",
        "scope": (
            "Curated successful raw temperature-1 references for qualitative "
            "deployment comparison; not an unbiased evaluation."
        ),
        "pretrained": {
            "checkpoint": str(args.pretrained),
            "sha256": sha256_file(args.pretrained),
            "runs": [
                {
                    "gamma": row["gamma"],
                    "seed": row["seed"],
                    "mode": row["mode"],
                }
                for row in pretrain_rows
            ],
        },
        "expanded": {
            "checkpoint": str(args.expanded),
            "sha256": sha256_file(args.expanded),
            "runs": [
                {
                    "gamma": row["gamma"],
                    "seed": row["seed"],
                    "mode": row["mode"],
                }
                for row in expanded_rows
            ],
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
