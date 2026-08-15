"""Visualize a reproducible pre/post expansion slice of the learned CFM field.

The learned plan flow is 30-dimensional.  This diagnostic fixes one policy
context and all plan coordinates except the first action's head-on transverse
and vertical components, then compares that two-dimensional slice at one flow
time.  It also compares temperature-1 first-action samples on the same context
and keeps committed training quotas separate from closed-loop route outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from safe_mppi.ball_flow_theta import start_goal_frame
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_flow_evaluation import _checkpoint_policy
from safe_mppi.lab_flow_expansion import LabFlowExpansionTask


ROOT = Path(__file__).resolve().parents[1]
MODES = ("below", "above", "left", "right")
MODE_COLORS = {
    "below": "#1468b3",
    "above": "#c8321b",
    "left": "#17964b",
    "right": "#8a3ffc",
}


def _probe_context(policy, config, gamma: float, position_fraction: float, device):
    env = TaskEnvironment(config)
    task = LabFlowExpansionTask(
        config,
        context_schema=policy.context_schema,
        tight_corridor=True,
        device=device,
    )
    state = task.reset(float(gamma), 0, 74123)
    state["x"] = np.asarray(env.start, np.float32).copy()
    state["x"][:3] = (
        env.start[:3]
        + float(position_fraction) * (env.goal - env.start[:3])
    )
    context = task.context(state, float(gamma)).detach().to(device)
    return context[..., :policy.context_dim], env


@torch.no_grad()
def _field_slice(policy, context, frame, flow_time: float, grid_size: int):
    limit = float(policy.control_limit or 0.3)
    values = np.linspace(-limit, limit, int(grid_size), dtype=np.float32)
    transverse, vertical = np.meshgrid(values, values)
    local = np.zeros((transverse.size, 3), np.float32)
    local[:, 1] = transverse.ravel()
    local[:, 2] = vertical.ravel()
    world = local @ np.asarray(frame, np.float32).T

    encoded = policy.encode_context(context)
    points = torch.zeros(
        len(world), policy.flow.plan_dim,
        device=context.device, dtype=context.dtype,
    )
    points[:, :3] = torch.from_numpy(world).to(context.device)
    times = torch.full(
        (len(world),), float(flow_time),
        device=context.device, dtype=context.dtype,
    )
    velocity = policy.flow(
        points,
        times,
        encoded.reshape(1, -1).expand(len(world), -1),
    )
    local_velocity = (
        velocity[:, :3].detach().cpu().numpy()
        @ np.asarray(frame, np.float32)
    )
    return {
        "x": transverse,
        "y": vertical,
        "u": local_velocity[:, 1].reshape(transverse.shape),
        "v": local_velocity[:, 2].reshape(vertical.shape),
    }


def _quadrant_counts(local_actions: np.ndarray) -> dict[str, int]:
    angle = np.degrees(np.arctan2(local_actions[:, 2], local_actions[:, 1]))
    return {
        "below": int(np.sum((angle >= -135.0) & (angle < -45.0))),
        "above": int(np.sum((angle >= 45.0) & (angle < 135.0))),
        "left": int(np.sum((angle >= -45.0) & (angle < 45.0))),
        "right": int(np.sum((angle >= 135.0) | (angle < -135.0))),
    }


def _normalized_entropy(counts: dict[str, int]) -> float:
    values = np.asarray([counts[mode] for mode in MODES], float)
    probability = values / values.sum()
    positive = probability[probability > 0.0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(MODES)))


def _field_change_metrics(before, after) -> dict[str, object]:
    before_mag = np.hypot(before["u"], before["v"])
    after_mag = np.hypot(after["u"], after["v"])
    delta_mag = np.hypot(
        after["u"] - before["u"], after["v"] - before["v"],
    )
    before_min = np.unravel_index(np.argmin(before_mag), before_mag.shape)
    after_min = np.unravel_index(np.argmin(after_mag), after_mag.shape)
    dot = before["u"] * after["u"] + before["v"] * after["v"]
    denominator = before_mag * after_mag
    valid = denominator > 1.0e-8
    return {
        "rms_before": float(np.sqrt(np.mean(before_mag ** 2))),
        "rms_after": float(np.sqrt(np.mean(after_mag ** 2))),
        "rms_change": float(np.sqrt(np.mean(delta_mag ** 2))),
        "relative_rms_change": float(
            np.sqrt(np.mean(delta_mag ** 2))
            / max(np.sqrt(np.mean(before_mag ** 2)), 1.0e-12)
        ),
        "mean_direction_cosine": float(np.mean(dot[valid] / denominator[valid])),
        "minimum_speed_point_before": [
            float(before["x"][before_min]), float(before["y"][before_min]),
        ],
        "minimum_speed_point_after": [
            float(after["x"][after_min]), float(after["y"][after_min]),
        ],
    }


def _field_preview(before, after, size: int = 5) -> list[dict[str, float]]:
    indices = np.linspace(0, before["x"].shape[0] - 1, size, dtype=int)
    return [
        {
            "x": round(float(before["x"][row, column]), 5),
            "y": round(float(before["y"][row, column]), 5),
            "before_u": round(float(before["u"][row, column]), 5),
            "before_v": round(float(before["v"][row, column]), 5),
            "after_u": round(float(after["u"][row, column]), 5),
            "after_v": round(float(after["v"][row, column]), 5),
        }
        for row in indices
        for column in indices
    ]


@torch.no_grad()
def _sample_first_actions(policy, context, frame, samples: int, seed: int):
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    plans, bases = policy.sample_with_base(
        context, int(samples), generator, base_std=1.0,
    )
    first = plans[:, 0, :].detach().cpu().numpy()
    return first @ np.asarray(frame, np.float32), bases.detach().cpu()


def _committed_counts(manifest: dict) -> dict[str, int]:
    if "trajectory_counts_by_mode" in manifest:
        return {
            mode: int(manifest["trajectory_counts_by_mode"][mode])
            for mode in MODES
        }
    counts = Counter()
    for round_row in manifest["rounds"]:
        for detail in round_row[
            "successful_executed_commit_by_gamma"
        ].values():
            for mode in detail.get("committed_sample_update_modes", []):
                counts[MODES[int(mode)]] += 1
    return {mode: int(counts[mode]) for mode in MODES}


def _raw_counts(raw_eval: dict, round_i: int) -> dict[str, int]:
    counts = raw_eval["summary"][str(int(round_i))]["pooled"]["route_counts"]
    return {mode: int(counts[mode]) for mode in MODES}


def _draw_sectors(axis, limit):
    for degrees in (45.0, 135.0, 225.0, 315.0):
        angle = np.radians(degrees)
        axis.plot(
            [0.0, limit * np.cos(angle)],
            [0.0, limit * np.sin(angle)],
            color="#6b7280", lw=0.7, ls="--", alpha=0.65,
        )


def _plot(
    output: Path,
    before_field,
    after_field,
    before_actions,
    after_actions,
    committed,
    raw_before,
    raw_after,
    metadata,
):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.1))

    fields = (before_field, after_field)
    magnitudes = [np.hypot(field["u"], field["v"]) for field in fields]
    norm = Normalize(0.0, max(float(value.max()) for value in magnitudes))
    for axis, field, magnitude, title in zip(
        axes[0, :2], fields, magnitudes,
        (
            f"Before (checkpoint {metadata['before_round']})",
            f"After (checkpoint {metadata['after_round']})",
        ),
    ):
        axis.quiver(
            field["x"], field["y"], field["u"], field["v"], magnitude,
            cmap="viridis", norm=norm, angles="xy", scale_units="xy",
            scale=8.0, width=0.004,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("first-action transverse slice")
        axis.set_ylabel("first-action vertical slice")
        axis.grid(alpha=0.18)

    delta_u = after_field["u"] - before_field["u"]
    delta_v = after_field["v"] - before_field["v"]
    delta = np.hypot(delta_u, delta_v)
    axes[0, 2].quiver(
        before_field["x"], before_field["y"], delta_u, delta_v, delta,
        cmap="magma", angles="xy", scale_units="xy", scale=6.0, width=0.004,
    )
    axes[0, 2].set_title("Field change (after − before)")
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_xlabel("first-action transverse slice")
    axes[0, 2].set_ylabel("first-action vertical slice")
    axes[0, 2].grid(alpha=0.18)

    sample_limit = max(
        float(np.abs(before_actions[:, 1:3]).max()),
        float(np.abs(after_actions[:, 1:3]).max()),
    ) * 1.05
    rng = np.random.default_rng(1209)
    for axis, actions, counts, title in (
        (axes[1, 0], before_actions, metadata["open_loop_before"], "Temperature-1 actions: before"),
        (axes[1, 1], after_actions, metadata["open_loop_after"], "Temperature-1 actions: after"),
    ):
        chosen = rng.choice(len(actions), min(1800, len(actions)), replace=False)
        axis.scatter(
            actions[chosen, 1], actions[chosen, 2],
            s=5, alpha=0.20, color="#2563eb", linewidths=0,
        )
        _draw_sectors(axis, sample_limit)
        axis.set_xlim(-sample_limit, sample_limit)
        axis.set_ylim(-sample_limit, sample_limit)
        axis.set_aspect("equal")
        axis.set_title(
            title + "\n"
            + "/".join(f"{mode[0]}{counts[mode]}" for mode in MODES)
            + f"  H₄={_normalized_entropy(counts):.3f}"
        )
        axis.set_xlabel("transverse acceleration [m/s²]")
        axis.set_ylabel("vertical acceleration [m/s²]")
        axis.grid(alpha=0.18)

    series = (
        ("Committed\ntraining inputs", committed),
        ("Raw success routes\nbefore", raw_before),
        ("Raw success routes\nafter", raw_after),
    )
    x = np.arange(len(MODES))
    width = 0.24
    for index, (label, counts) in enumerate(series):
        values = np.asarray([counts[mode] for mode in MODES], float)
        values = values / max(values.sum(), 1.0)
        axes[1, 2].bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"{label} (H₄={_normalized_entropy(counts):.2f})",
            alpha=0.82,
        )
    axes[1, 2].axhline(0.25, color="black", lw=1.0, ls="--", alpha=0.65)
    axes[1, 2].set_xticks(x, MODES)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_ylabel("share among the four bins")
    axes[1, 2].set_title("Balanced commits ≠ uniform learned routes")
    axes[1, 2].legend(frameon=False, loc="upper right")
    axes[1, 2].grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Conditional flow-matching change at one fixed approach context\n"
        f"γ={metadata['gamma']:g}, position={metadata['position_fraction']:.2f} "
        f"of start→goal, flow time t={metadata['flow_time']:.2f}",
        fontsize=14,
    )
    fig.text(
        0.5, 0.008,
        "Vector panels are one documented 2-D slice of the 30-D plan field; "
        "action quadrants are not closed-loop obstacle-crossing modes.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.93))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--position-fraction", type=float, default=0.25)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=880000)
    parser.add_argument(
        "--flow-nfe", type=int,
        help="override Euler integration steps for the pre/post action samples",
    )
    parser.add_argument("--before-round", type=int, default=0)
    parser.add_argument("--after-round", type=int, default=1)
    args = parser.parse_args()

    if not 0.0 <= args.position_fraction <= 1.0:
        parser.error("--position-fraction must lie in [0,1]")
    if not 0.0 <= args.flow_time <= 1.0:
        parser.error("--flow-time must lie in [0,1]")
    if args.flow_nfe is not None and args.flow_nfe <= 0:
        parser.error("--flow-nfe must be positive")
    if args.before_round < 0 or args.after_round <= args.before_round:
        parser.error("require 0 <= --before-round < --after-round")

    device = torch.device(args.device)
    config = load_config(args.expansion / "task_config_resolved.json")
    manifest = json.loads((args.expansion / "manifest.json").read_text())
    raw_eval = json.loads(args.raw_eval.read_text())
    before = _checkpoint_policy(
        args.pretrain_dir, args.expansion, args.before_round, device,
    )
    after = _checkpoint_policy(
        args.pretrain_dir, args.expansion, args.after_round, device,
    )
    if args.flow_nfe is not None:
        before.flow.nfe = int(args.flow_nfe)
        after.flow.nfe = int(args.flow_nfe)
    before_context, env = _probe_context(
        before, config, args.gamma, args.position_fraction, device,
    )
    after_context, _ = _probe_context(
        after, config, args.gamma, args.position_fraction, device,
    )
    frame = start_goal_frame(env)

    before_field = _field_slice(
        before, before_context, frame, args.flow_time, args.grid_size,
    )
    after_field = _field_slice(
        after, after_context, frame, args.flow_time, args.grid_size,
    )
    before_actions, before_bases = _sample_first_actions(
        before, before_context, frame, args.samples, args.seed,
    )
    after_actions, after_bases = _sample_first_actions(
        after, after_context, frame, args.samples, args.seed,
    )
    if not torch.equal(before_bases, after_bases):
        raise RuntimeError("pre/post sampling did not use identical base latents")

    metadata = {
        "schema": "safe_mppi_flow_change_slice_v1",
        "gamma": float(args.gamma),
        "position_fraction": float(args.position_fraction),
        "flow_time": float(args.flow_time),
        "sampling_temperature": 1.0,
        "flow_nfe": int(before.flow.nfe),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "before_round": int(args.before_round),
        "after_round": int(args.after_round),
        "slice": (
            "all 30-D flow coordinates fixed to zero except first-action "
            "head-on transverse and vertical coordinates"
        ),
        "open_loop_before": _quadrant_counts(before_actions),
        "open_loop_after": _quadrant_counts(after_actions),
        "committed": _committed_counts(manifest),
        "raw_routes_before": _raw_counts(raw_eval, args.before_round),
        "raw_routes_after": _raw_counts(raw_eval, args.after_round),
        "field_change": _field_change_metrics(before_field, after_field),
        "field_preview": _field_preview(before_field, after_field),
    }
    metadata["entropy"] = {
        key: _normalized_entropy(metadata[key])
        for key in (
            "open_loop_before", "open_loop_after", "committed",
            "raw_routes_before", "raw_routes_after",
        )
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(
        args.output,
        before_field,
        after_field,
        before_actions,
        after_actions,
        metadata["committed"],
        metadata["raw_routes_before"],
        metadata["raw_routes_after"],
        metadata,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
