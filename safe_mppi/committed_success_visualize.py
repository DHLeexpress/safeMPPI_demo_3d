"""Fail-closed gathering visualizations for committed successful trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COMMITTED_COLOR = "#d99a00"
COMMITTED_WINDOW_COLOR = "#1468b3"
OTHER_SUCCESS_COLOR = "#17964b"
FAILED_COLOR = "#7d838b"
NVP_COLOR = "#c8321b"
TARGET_NVP_COLOR = "#e67e22"

_WINDOW_ID = re.compile(
    r"^(?P<trajectory>r\d{4}:g[^:]+:e\d{6}):w(?P<start>\d{6})$"
)


@dataclass(frozen=True)
class EpisodeTrace:
    round: int
    gamma: float
    episode: int
    events: tuple[Mapping[str, Any], ...]
    executed_events: tuple[Mapping[str, Any], ...]
    path: np.ndarray
    status: str
    nvp_reason: str | None


@dataclass(frozen=True)
class GatheringCell:
    round: int
    gamma: float
    episodes: tuple[EpisodeTrace, ...]
    committed_trajectory_ids: tuple[str, ...]
    committed_episode_ids: tuple[int, ...]
    committed_trajectory_id: str | None
    committed_episode_id: int | None
    committed_window_ids: tuple[str, ...]
    committed_window_context_ids: tuple[int, ...]
    committed_window_positions: np.ndarray
    detail: Mapping[str, Any]

    @property
    def committed_episodes(self) -> tuple[EpisodeTrace, ...]:
        matches = tuple(
            episode for episode in self.episodes
            if episode.episode in self.committed_episode_ids
        )
        if len(matches) != len(self.committed_episode_ids):
            raise ValueError(
                "committed episodes must each resolve exactly once for "
                f"round={self.round}, gamma={self.gamma:g}; "
                f"expected {len(self.committed_episode_ids)}, found {len(matches)}"
            )
        return matches

    @property
    def committed_episode(self) -> EpisodeTrace | None:
        if self.committed_episode_id is None:
            return None
        return next(
            episode for episode in self.committed_episodes
            if episode.episode == self.committed_episode_id
        )


def has_committed_success_schema(manifest: Mapping[str, Any]) -> bool:
    rows = manifest.get("rounds", ())
    present = [
        "successful_executed_commit_by_gamma" in row for row in rows
    ]
    if any(present) and not all(present):
        raise ValueError(
            "committed-success schema is present for only some expansion rounds"
        )
    return bool(present and all(present))


def _episode_path(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], np.ndarray]:
    executed = tuple(row for row in rows if row.get("chosen_local") is not None)
    if not rows:
        raise ValueError("cannot build an episode trace from no events")
    points = [np.asarray(rows[0]["robot"][:3], float)]
    points.extend(
        np.asarray(row["robot_after"][:3], float) for row in executed
    )
    return executed, np.asarray(points, float)


def group_episode_traces(
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, float, int], EpisodeTrace]:
    grouped: dict[tuple[int, float, int], list[Mapping[str, Any]]] = {}
    for event in events:
        key = (
            int(event["round"]),
            float(event["gamma"]),
            int(event["episode"]),
        )
        grouped.setdefault(key, []).append(event)

    traces: dict[tuple[int, float, int], EpisodeTrace] = {}
    for key, unsorted in grouped.items():
        rows = tuple(sorted(unsorted, key=lambda event: int(event["step"])))
        steps = [int(event["step"]) for event in rows]
        if len(steps) != len(set(steps)):
            raise ValueError(f"duplicate event step in episode trace {key}")
        if any(event.get("status") is not None for event in rows[:-1]):
            raise ValueError(f"episode trace continues after a terminal event: {key}")
        executed, path = _episode_path(rows)
        terminal = rows[-1].get("status") or "TIMEOUT"
        traces[key] = EpisodeTrace(
            round=key[0],
            gamma=key[1],
            episode=key[2],
            events=rows,
            executed_events=executed,
            path=path,
            status=str(terminal),
            nvp_reason=(
                str(rows[-1].get("nvp_reason"))
                if rows[-1].get("nvp_reason") is not None else None
            ),
        )
    return traces


def _trajectory_id(round_i: int, gamma: float, episode: int) -> str:
    return f"r{round_i:04d}:g{gamma:.9g}:e{episode:06d}"


def resolve_committed_success(
    manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, float], GatheringCell]:
    """Resolve authoritative manifest keys to logged executed trajectories.

    No trajectory is selected from visual appearance, route mode, or terminal
    status alone. The manifest must name it, and every named window must map to
    an executed window-start event on that exact successful episode.
    """
    if not has_committed_success_schema(manifest):
        raise ValueError("manifest has no committed-success schema")
    gammas = tuple(float(gamma) for gamma in manifest["config"]["gammas"])
    traces = group_episode_traces(events)
    committed_only = manifest.get("event_log") == "committed_success"
    expected_success_trace_keys: set[tuple[int, float, int]] = set()
    output: dict[tuple[int, float], GatheringCell] = {}

    for row in manifest["rounds"]:
        round_i = int(row["round"])
        details = row["successful_executed_commit_by_gamma"]
        expected_gamma_keys = {f"{gamma:.9g}" for gamma in gammas}
        if set(details) != expected_gamma_keys:
            raise ValueError(
                f"round {round_i} committed gamma keys {sorted(details)} do not "
                f"match declared gammas {sorted(expected_gamma_keys)}"
            )

        round_trajectory_ids: list[str] = []
        round_window_ids: list[str] = []
        for gamma in gammas:
            gamma_key = f"{gamma:.9g}"
            detail = details[gamma_key]
            cell_episodes = tuple(sorted(
                (
                    trace for (trace_round, trace_gamma, _), trace in traces.items()
                    if trace_round == round_i and trace_gamma == gamma
                ),
                key=lambda trace: trace.episode,
            ))
            actual_successes = [
                trace for trace in cell_episodes if trace.status == "SUCCESS"
            ]
            if int(detail["success_episode_count"]) != len(actual_successes):
                raise ValueError(
                    f"round {round_i}, gamma {gamma:g} success count mismatch: "
                    f"manifest={detail['success_episode_count']}, "
                    f"events={len(actual_successes)}"
                )
            if committed_only:
                declared_success_ids = {
                    int(value)
                    for value in detail["success_episode_above_fractions"]
                }
                actual_episode_ids = {
                    trace.episode for trace in cell_episodes
                }
                if any(trace.status != "SUCCESS" for trace in cell_episodes):
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g} committed-success "
                        "event log contains a non-SUCCESS trace"
                    )
                if actual_episode_ids != declared_success_ids:
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g} successful event "
                        "trace IDs do not exactly match the resolved SUCCESS "
                        f"IDs: events={sorted(actual_episode_ids)}, "
                        f"manifest={sorted(declared_success_ids)}"
                    )
                expected_success_trace_keys.update(
                    (round_i, gamma, episode_id)
                    for episode_id in declared_success_ids
                )

            trajectory_id = detail["committed_trajectory_id"]
            episode_id = detail["committed_episode_id"]
            primary_window_ids = tuple(
                str(value) for value in detail["committed_window_ids"]
            )
            if int(detail["committed_window_count"]) != len(primary_window_ids):
                raise ValueError(
                    f"round {round_i}, gamma {gamma:g} committed window count mismatch"
                )

            if "committed_trajectories" in detail:
                trajectories = tuple(detail["committed_trajectories"])
                trajectory_ids = tuple(
                    str(value) for value in detail["committed_trajectory_ids"]
                )
                episode_ids = tuple(
                    int(value) for value in detail["committed_episode_ids"]
                )
                all_window_ids = tuple(
                    str(value)
                    for value in detail["all_committed_window_ids"]
                )
                if int(detail["committed_trajectory_count"]) != len(trajectories):
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g} committed trajectory "
                        "count mismatch"
                    )
                if int(detail["all_committed_window_count"]) != len(all_window_ids):
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g} all-window count mismatch"
                    )
                if trajectory_ids != tuple(
                    str(value["trajectory_id"]) for value in trajectories
                ) or episode_ids != tuple(
                    int(value["episode_id"]) for value in trajectories
                ):
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g} plural committed IDs "
                        "do not match trajectory details"
                    )
                if trajectories:
                    primary = trajectories[0]
                    if (
                        str(trajectory_id) != str(primary["trajectory_id"])
                        or int(episode_id) != int(primary["episode_id"])
                        or primary_window_ids != tuple(
                            str(value)
                            for value in primary["committed_window_ids"]
                        )
                    ):
                        raise ValueError(
                            f"round {round_i}, gamma {gamma:g} primary aliases "
                            "do not match the first committed trajectory"
                        )
                elif not (
                    trajectory_id is None
                    and episode_id is None
                    and not primary_window_ids
                    and not trajectory_ids
                    and not episode_ids
                    and not all_window_ids
                ):
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g} has partial empty "
                        "committed-trajectory keys"
                    )
            else:
                trajectories = (
                    ({
                        "trajectory_id": trajectory_id,
                        "episode_id": episode_id,
                        "executed_steps": detail["executed_steps"],
                        "committed_window_count": detail[
                            "committed_window_count"
                        ],
                        "committed_window_ids": primary_window_ids,
                    },)
                    if trajectory_id is not None and episode_id is not None
                    else ()
                )
                trajectory_ids = (
                    (str(trajectory_id),) if trajectories else ()
                )
                episode_ids = ((int(episode_id),) if trajectories else ())
                all_window_ids = primary_window_ids

            context_ids: list[int] = []
            positions: list[np.ndarray] = []
            gamma_window_ids: list[str] = []
            for trajectory in trajectories:
                current_id = str(trajectory["trajectory_id"])
                current_episode = int(trajectory["episode_id"])
                current_window_ids = tuple(
                    str(value) for value in trajectory["committed_window_ids"]
                )
                if int(trajectory["committed_window_count"]) != len(
                    current_window_ids
                ):
                    raise ValueError(
                        f"round {round_i}, gamma {gamma:g}, episode "
                        f"{current_episode} committed window count mismatch"
                    )
                expected_id = _trajectory_id(
                    round_i, gamma, current_episode
                )
                if current_id != expected_id:
                    raise ValueError(
                        f"committed trajectory ID {current_id!r} does not match "
                        f"{expected_id!r}"
                    )
                trace_key = (round_i, gamma, current_episode)
                if trace_key not in traces:
                    raise ValueError(
                        f"committed trajectory {current_id} has no event trace"
                    )
                trace = traces[trace_key]
                if trace.status != "SUCCESS":
                    raise ValueError(
                        f"committed trajectory {current_id} terminates as "
                        f"{trace.status}, not SUCCESS"
                    )
                if int(trajectory["executed_steps"]) != len(trace.executed_events):
                    raise ValueError(
                        f"committed trajectory {current_id} executed-step mismatch"
                    )
                for window_id in current_window_ids:
                    match = _WINDOW_ID.fullmatch(window_id)
                    if (
                        match is None
                        or match.group("trajectory") != current_id
                    ):
                        raise ValueError(
                            f"committed window ID {window_id!r} does not belong to "
                            f"{current_id!r}"
                        )
                    start = int(match.group("start"))
                    if start >= len(trace.executed_events):
                        raise ValueError(
                            f"committed window {window_id} starts after the executed trace"
                        )
                    event = trace.executed_events[start]
                    context_ids.append(int(event["context_id"]))
                    positions.append(np.asarray(event["robot"][:3], float))
                round_trajectory_ids.append(current_id)
                gamma_window_ids.extend(current_window_ids)

            if tuple(gamma_window_ids) != all_window_ids:
                raise ValueError(
                    f"round {round_i}, gamma {gamma:g} aggregate committed "
                    "window IDs disagree with trajectory details"
                )
            round_window_ids.extend(gamma_window_ids)

            output[(round_i, gamma)] = GatheringCell(
                round=round_i,
                gamma=gamma,
                episodes=cell_episodes,
                committed_trajectory_ids=trajectory_ids,
                committed_episode_ids=episode_ids,
                committed_trajectory_id=(
                    str(trajectory_id) if trajectory_id is not None else None
                ),
                committed_episode_id=(
                    int(episode_id) if episode_id is not None else None
                ),
                committed_window_ids=all_window_ids,
                committed_window_context_ids=tuple(context_ids),
                committed_window_positions=(
                    np.asarray(positions, float).reshape(-1, 3)
                    if positions else np.empty((0, 3), float)
                ),
                detail=detail,
            )

        if list(row["committed_trajectory_ids"]) != round_trajectory_ids:
            raise ValueError(
                f"round {round_i} top-level committed trajectory IDs disagree "
                "with per-gamma details"
            )
        if list(row["committed_window_ids"]) != round_window_ids:
            raise ValueError(
                f"round {round_i} top-level committed window IDs disagree "
                "with per-gamma details"
            )
    if committed_only and set(traces) != expected_success_trace_keys:
        extras = set(traces).difference(expected_success_trace_keys)
        missing = expected_success_trace_keys.difference(traces)
        raise ValueError(
            "committed-success event log must contain exactly the resolved "
            f"SUCCESS traces; extras={sorted(extras)}, missing={sorted(missing)}"
        )
    return output


def _draw_scene_projections(ax_side, ax_head, env) -> None:
    sphere = np.asarray(env.spheres[0], float)
    theta = np.linspace(0.0, 2.0 * np.pi, 140)
    ax_side.fill(
        sphere[0] + sphere[3] * np.cos(theta),
        sphere[2] + sphere[3] * np.sin(theta),
        color="#8f969f",
        alpha=0.46,
        zorder=1,
    )
    ax_head.fill(
        sphere[1] + sphere[3] * np.cos(theta),
        sphere[2] + sphere[3] * np.sin(theta),
        color="#8f969f",
        alpha=0.46,
        zorder=1,
    )
    for axis, start_columns, goal_columns in (
        (ax_side, (0, 2), (0, 2)),
        (ax_head, (1, 2), (1, 2)),
    ):
        axis.scatter(
            env.start[start_columns[0]],
            env.start[start_columns[1]],
            marker="s",
            color="#111111",
            s=18,
            zorder=8,
        )
        axis.scatter(
            env.goal[goal_columns[0]],
            env.goal[goal_columns[1]],
            marker="*",
            color="#ffca28",
            edgecolor="#6a4e00",
            s=70,
            zorder=8,
        )


def _draw_target_region(ax_side, ax_head, target_region: str) -> None:
    if target_region not in {"above_wedge", "above_halfspace"}:
        raise ValueError(f"unknown target region {target_region!r}")
    ax_side.axhspan(2.0, 2.95, color="#f0a33a", alpha=0.045, zorder=0)
    ax_side.axhline(2.0, color="#d47b18", lw=0.8, ls=":", alpha=0.8)
    y = np.linspace(-0.95, 0.95, 140)
    boundary = (
        2.0 + np.abs(y)
        if target_region == "above_wedge" else np.full_like(y, 2.0)
    )
    ax_head.fill_between(y, boundary, 2.95, color="#f0a33a", alpha=0.07, zorder=0)
    ax_head.plot(y, boundary, color="#d47b18", lw=0.8, ls=":", alpha=0.8)


def draw_gathering_cell(
    ax_side,
    ax_head,
    env,
    cell: GatheringCell,
    *,
    target_region: str,
    gate_active: bool,
    annotate: bool = True,
) -> None:
    """Draw one round/gamma cell without assigning training meaning to failures."""
    _draw_scene_projections(ax_side, ax_head, env)
    if gate_active:
        _draw_target_region(ax_side, ax_head, target_region)

    committed = cell.committed_episode
    committed_episode_ids = set(cell.committed_episode_ids)
    for episode in cell.episodes:
        path = episode.path
        if episode.episode in committed_episode_ids:
            for axis, columns in ((ax_side, (0, 2)), (ax_head, (1, 2))):
                axis.plot(
                    path[:, columns[0]],
                    path[:, columns[1]],
                    color="#5a4300",
                    lw=4.4,
                    alpha=0.75,
                    zorder=5,
                )
                axis.plot(
                    path[:, columns[0]],
                    path[:, columns[1]],
                    color=COMMITTED_COLOR,
                    lw=2.9,
                    alpha=0.98,
                    zorder=6,
                )
                axis.scatter(
                    path[-1, columns[0]],
                    path[-1, columns[1]],
                    marker="*",
                    color=COMMITTED_COLOR,
                    edgecolor="#5a4300",
                    linewidth=0.6,
                    s=70,
                    zorder=9,
                )
        elif episode.status == "SUCCESS":
            ax_side.plot(
                path[:, 0], path[:, 2],
                color=OTHER_SUCCESS_COLOR, lw=0.9, alpha=0.38, zorder=3,
            )
            ax_head.plot(
                path[:, 1], path[:, 2],
                color=OTHER_SUCCESS_COLOR, lw=0.9, alpha=0.38, zorder=3,
            )
        else:
            ax_side.plot(
                path[:, 0], path[:, 2],
                color=FAILED_COLOR, lw=0.65, alpha=0.28, zorder=2,
            )
            ax_head.plot(
                path[:, 1], path[:, 2],
                color=FAILED_COLOR, lw=0.65, alpha=0.28, zorder=2,
            )
            marker = "o" if episode.status == "TIMEOUT" else "x"
            color = (
                TARGET_NVP_COLOR
                if episode.nvp_reason == "TARGET" else NVP_COLOR
            )
            for axis, columns in ((ax_side, (0, 2)), (ax_head, (1, 2))):
                axis.scatter(
                    path[-1, columns[0]],
                    path[-1, columns[1]],
                    marker=marker,
                    color=(FAILED_COLOR if marker == "o" else color),
                    s=20,
                    linewidth=1.0,
                    zorder=7,
                )

    if len(cell.committed_window_positions):
        positions = cell.committed_window_positions
        ax_side.scatter(
            positions[:, 0], positions[:, 2],
            color=COMMITTED_WINDOW_COLOR, edgecolor="white", linewidth=0.25,
            s=15, zorder=10,
        )
        ax_head.scatter(
            positions[:, 1], positions[:, 2],
            color=COMMITTED_WINDOW_COLOR, edgecolor="white", linewidth=0.25,
            s=15, zorder=10,
        )

    success = sum(episode.status == "SUCCESS" for episode in cell.episodes)
    nvp = sum(episode.status == "NVP" for episode in cell.episodes)
    timeout = sum(episode.status == "TIMEOUT" for episode in cell.episodes)
    if annotate:
        label = (
            f"S {success}  NVP {nvp}  TO {timeout}\n"
            f"committed trajectories {len(cell.committed_trajectory_ids)}  "
            f"windows {len(cell.committed_window_ids)}"
        )
        if committed is None:
            label += "\nNO COMMITTED SUCCESS"
        ax_side.text(
            0.02, 0.02, label, transform=ax_side.transAxes,
            fontsize=6.2, va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            zorder=20,
        )

    ax_side.set_xlim(-0.15, 3.2)
    ax_side.set_ylim(1.1, 2.95)
    ax_head.set_xlim(0.95, -0.95)
    ax_head.set_ylim(1.1, 2.95)
    for axis in (ax_side, ax_head):
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.16)


def plot_gathering_commit_gallery(
    env,
    manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    target_region: str,
    target_gate_start_round: int | None,
) -> Path:
    """Render an event-derived round-by-gamma gallery, separate from raw evaluation."""
    cells = resolve_committed_success(manifest, events)
    rounds = tuple(int(row["round"]) for row in manifest["rounds"])
    gammas = tuple(float(gamma) for gamma in manifest["config"]["gammas"])
    fig = plt.figure(figsize=(5.25 * len(gammas), 2.65 * len(rounds)))
    outer = fig.add_gridspec(
        len(rounds), len(gammas), hspace=0.28, wspace=0.18,
    )
    for row_index, round_i in enumerate(rounds):
        for column_index, gamma in enumerate(gammas):
            inner = outer[row_index, column_index].subgridspec(
                1, 2, width_ratios=(1.55, 1.0), wspace=0.06,
            )
            ax_side = fig.add_subplot(inner[0, 0])
            ax_head = fig.add_subplot(inner[0, 1])
            draw_gathering_cell(
                ax_side,
                ax_head,
                env,
                cells[(round_i, gamma)],
                target_region=target_region,
                gate_active=(
                    target_gate_start_round is not None
                    and round_i >= target_gate_start_round
                ),
            )
            if row_index == 0:
                ax_side.set_title(rf"$\gamma={gamma:g}$: side", fontsize=10)
                ax_head.set_title("head-on", fontsize=8)
            if column_index == 0:
                ax_side.set_ylabel(f"round {round_i}\n" + r"$z$ [m]", fontsize=8)
            else:
                ax_side.set_yticklabels([])
            ax_head.set_yticklabels([])
            if row_index == len(rounds) - 1:
                ax_side.set_xlabel(r"$x$ [m]", fontsize=8)
                ax_head.set_xlabel(r"$y$ [m]", fontsize=8)
            else:
                ax_side.set_xticklabels([])
                ax_head.set_xticklabels([])
            ax_side.tick_params(labelsize=6)
            ax_head.tick_params(labelsize=6)

    legend = (
        Line2D([0], [0], color=COMMITTED_COLOR, lw=3.0,
               label="committed SUCCESS"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COMMITTED_WINDOW_COLOR, markeredgecolor="white",
               markersize=6, label=r"committed $\leq H$ suffix start"),
        Line2D([0], [0], color=OTHER_SUCCESS_COLOR, lw=1.0,
               label="other gathering SUCCESS (replay not asserted)"),
        Line2D([0], [0], color=FAILED_COLOR, lw=0.8,
               label="failed gathering attempt (diagnostic only)"),
        Line2D([0], [0], marker="x", color=NVP_COLOR, lw=0,
               label="verifier/progress NVP"),
        Line2D([0], [0], marker="x", color=TARGET_NVP_COLOR, lw=0,
               label="target-region NVP"),
    )
    fig.legend(
        handles=legend, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 0.975), frameon=False, fontsize=8,
    )
    fig.suptitle(
        "Expansion gathering evidence | not raw evaluation\n"
        "Failed attempts are diagnostic; only blue-dotted windows are asserted committed",
        fontsize=15,
        weight="bold",
        y=0.997,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output
