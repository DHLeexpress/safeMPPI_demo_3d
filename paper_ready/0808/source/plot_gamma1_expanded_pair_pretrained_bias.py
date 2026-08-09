#!/usr/bin/env python3
"""Plot two expanded and eight pretrained gamma-1 frozen trajectories in 3D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def load(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            dense = np.concatenate([
                np.asarray(data["states"][:1, :3], dtype=float),
                np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3),
            ])
            rows.append({
                "dense": dense,
                "seed": int(scalar(data, "seed")),
                "theta": float(scalar(data, "theta_deg")),
            })
    return rows


def setup(ax, start: np.ndarray, goal: np.ndarray, sphere: np.ndarray) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 22)
    center, radius = sphere[:3], float(sphere[3])
    ax.plot_surface(
        center[0] + radius * np.outer(np.cos(u), np.sin(v)),
        center[1] + radius * np.outer(np.sin(u), np.sin(v)),
        center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
        color="#AFAFAF", alpha=0.38, linewidth=0.0, shade=True,
    )
    ax.scatter(*start, marker="s", s=54, color="#111111", depthshade=False)
    ax.scatter(*goal, marker="*", s=92, color="#D8A500", edgecolor="#5B4700",
               linewidth=0.6, depthshade=False)
    ax.text(*start, "  start", fontsize=10)
    ax.text(*goal, "  goal", fontsize=10)
    ax.set(xlim=(-2.25, 0.9), ylim=(-1.65, 1.65), zlim=(0.35, 1.85))
    ax.set_box_aspect((3.15, 3.3, 1.5))
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.view_init(elev=25, azim=30)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.025)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expanded-string-safe", type=Path)
    parser.add_argument("--expanded-mirrored", type=Path)
    parser.add_argument("--pretrained-biased", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config = json.loads((bundle / "config" / "task_config_resolved.json").read_text())
    start = np.asarray(config["taskspace"]["start"][:3], dtype=float)
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    sphere = np.asarray(config["obstacles"]["spheres"][0], dtype=float)
    expanded = load(
        args.expanded_string_safe.resolve() if args.expanded_string_safe else
        bundle / "trajectories" / "expanded_string_safe_v1"
    ) + load(
        args.expanded_mirrored.resolve() if args.expanded_mirrored else
        bundle / "trajectories" / "expanded_mirrored_above_v1"
    )
    pretrained = load(
        args.pretrained_biased.resolve() if args.pretrained_biased else
        bundle / "trajectories" / "pretrained_gamma1_biased_left_v1"
    )
    if len(expanded) != 2 or len(pretrained) != 8:
        raise ValueError("expected two expanded and eight pretrained trajectories")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 12, "axes.titlesize": 15,
        "axes.labelsize": 13,
    })
    fig = plt.figure(figsize=(14.0, 6.2), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    for ax in axes:
        setup(ax, start, goal, sphere)
    colors = ("#0072B2", "#D55E00")
    for row, color in zip(sorted(expanded, key=lambda item: item["seed"]), colors):
        dense = np.asarray(row["dense"])
        axes[0].plot(*dense.T, color=color, linewidth=2.7,
                     label=rf"seed {row['seed']}, $\theta={row['theta']:.1f}^\circ$")
    for index, row in enumerate(pretrained):
        dense = np.asarray(row["dense"])
        axes[1].plot(*dense.T, color="#8E3B9D", alpha=0.54 + 0.05 * (index % 4),
                     linewidth=1.75)
    axes[0].set_title(r"Expanded, $\gamma=1.0$: two frozen above paths")
    axes[1].set_title(r"Pretrained, $\gamma=1.0$: eight frozen left-biased paths")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    axes[1].legend(handles=[Line2D(
        [0], [0], color="#8E3B9D", lw=2.2,
        label="8 selected left successes from the fixed bank",
    )], loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    output = args.output.resolve() if args.output else bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    png = output / "gamma1_expanded_pair_vs_pretrained_bias8_3d.png"
    pdf = output / "gamma1_expanded_pair_vs_pretrained_bias8_3d.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
