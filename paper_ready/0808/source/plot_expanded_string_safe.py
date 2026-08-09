#!/usr/bin/env python3
"""Compare the frozen straight-over-ball and side-above gamma-1 references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def task_frame(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    forward = goal - start
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    up -= float(up @ forward) * forward
    up /= np.linalg.norm(up)
    lateral = np.cross(up, forward)
    lateral /= np.linalg.norm(lateral)
    return np.column_stack([forward, lateral, up])


def load_local(path: Path, center: np.ndarray, frame: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        dense = np.concatenate([
            np.asarray(data["states"][:1, :3], dtype=float),
            np.asarray(data["dense_steps"], dtype=float).reshape(-1, 3),
        ])
        return {
            "local": (dense - center[None]) @ frame,
            "seed": int(scalar(data, "seed")),
            "theta": float(scalar(data, "theta_deg")),
        }


def minimum_string_distance(local: np.ndarray, physical_radius: float) -> float:
    overhead = local[:, 2] >= physical_radius
    return float(np.linalg.norm(local[overhead, :2], axis=1).min())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replacement", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config = json.loads((bundle / "config" / "task_config_resolved.json").read_text())
    start = np.asarray(config["taskspace"]["start"][:3], dtype=float)
    goal = np.asarray(config["taskspace"]["goal"], dtype=float)
    sphere = np.asarray(config["obstacles"]["spheres"][0], dtype=float)
    physical_radius = float(config["stage1"]["physical_sphere_radius_m"])
    modeled_radius = float(config["stage1"]["modeled_radius_m"])
    frame = task_frame(start, goal)
    old = load_local(
        bundle / "trajectories" / "expanded_supplement_v1"
        / "gamma_1_mode_above_seed_137364.npz",
        sphere[:3], frame,
    )
    replacement_path = (
        args.replacement.resolve()
        if args.replacement is not None
        else bundle / "trajectories" / "expanded_string_safe_v1"
        / "gamma_1_mode_above_seed_131629.npz"
    )
    new = load_local(replacement_path, sphere[:3], frame)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
    })
    fig, ax = plt.subplots(figsize=(7.0, 6.4), constrained_layout=True)
    ax.add_patch(Circle(
        (0.0, 0.0), modeled_radius, facecolor="#BDBDBD", alpha=0.16,
        edgecolor="#555555", linestyle="--", linewidth=1.2,
    ))
    ax.add_patch(Circle(
        (0.0, 0.0), physical_radius, facecolor="#888888", alpha=0.55,
        edgecolor="#333333", linewidth=1.0,
    ))
    ax.plot([0.0, 0.0], [physical_radius, 0.82], color="#222222", linewidth=3.0)
    ax.text(0.025, 0.73, "suspension line", color="#222222", fontsize=11)
    for row, color, name in (
        (old, "#D55E00", "old straight-above"),
        (new, "#009E73", "new side-above"),
    ):
        local = np.asarray(row["local"])
        visible = np.abs(local[:, 0]) <= 0.9
        distance = minimum_string_distance(local, physical_radius)
        ax.plot(
            local[visible, 1], local[visible, 2], color=color, linewidth=2.4,
            label=(rf"{name}: $\theta={row['theta']:.1f}^\circ$, seed {row['seed']}"
                   + rf"; string sep. {distance:.3f} m"),
        )
    ax.axhline(0.0, color="#A0A0A0", linewidth=0.55)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.72, 0.82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("local left [m]")
    ax.set_ylabel("vertical [m]")
    ax.set_title(r"Expanded $\gamma=1.0$: suspension-line alternative")
    ax.grid(color="#EEEEEE", linewidth=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    output = args.output.resolve() if args.output is not None else bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    png = output / "expanded_gamma1_string_safe_headon.png"
    pdf = output / "expanded_gamma1_string_safe_headon.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
