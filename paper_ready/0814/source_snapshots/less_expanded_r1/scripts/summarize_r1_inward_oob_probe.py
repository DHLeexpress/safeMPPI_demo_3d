#!/usr/bin/env python3
"""Summarize the live paired r1 inward-route/OOB probe without actuating it."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = (
    ROOT / "results/stage1_single_ball_t128/0811_pre2_r1_inward_oob_probe"
)
MODES = ("below", "above", "left", "right")
GAMMAS = ("0.1", "0.3", "0.5", "1")
EVALUATION_STATUS = "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE"
FIXED_THRESHOLDS = {
    "SR_min": 0.30,
    "OOB_max": 0.075,
    "CR_max": 0.65,
    "window_validity_min": 0.80,
    "route_coverage_min": 0.75,
    "right_success_min": 1,
    "x_max_exits_max": 3,
}
SAMPLE_UPDATE_RE = re.compile(
    r"\[sample-update\]\s+r(?P<round>\d+)\s+gamma\s+"
    r"(?P<gamma>[0-9.]+)\s+retry batch\s+"
    r"(?P<retry>\d+)/(?P<cap_last>\d+)\s+\|\s+"
    r"guided b(?P<gb>\d+)/a(?P<ga>\d+)/l(?P<gl>\d+)/r(?P<gr>\d+)"
    r"\s+\|\s+unguided b(?P<ub>\d+)/a(?P<ua>\d+)/l(?P<ul>\d+)/r(?P<ur>\d+)"
)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _gamma_key(mapping: dict, gamma: str) -> Any:
    for key in (gamma, f"{float(gamma):g}", str(float(gamma))):
        if key in mapping:
            return mapping[key]
    return None


def _counts(values) -> list[int]:
    result = [0, 0, 0, 0]
    for value in values or []:
        try:
            mode = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= mode < 4:
            result[mode] += 1
    return result


def _pid_alive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        os.kill(int(path.read_text().strip()), 0)
    except (OSError, ValueError):
        return False
    return True


def _latest_log_support(stage: Path, name: str, output: Path) -> dict:
    candidates = [
        stage / "logs" / f"{name}.log",
        stage / "logs" / f"{name}.screen.log",
        output / "helios.log",
        output / "helios_early_stop.log",
    ]
    paths = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    latest = {}
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            match = SAMPLE_UPDATE_RE.search(line)
            if match is None or int(match.group("round")) != 1:
                continue
            gamma = f"{float(match.group('gamma')):g}"
            guided = [int(match.group(key)) for key in ("gb", "ga", "gl", "gr")]
            unguided = [int(match.group(key)) for key in ("ub", "ua", "ul", "ur")]
            latest[gamma] = {
                "retry_batch": int(match.group("retry")),
                "retry_resample_batch_cap": int(match.group("cap_last")) + 1,
                "guided_b_a_l_r": guided,
                "unguided_b_a_l_r": unguided,
                "terminal_support_b_a_l_r": [
                    left + right for left, right in zip(guided, unguided)
                ],
                "source": str(path.resolve()),
            }
    return latest


def _terminal_support(
    stage: Path,
    name: str,
    output: Path,
    metric: dict | None,
    early_stop: dict | None,
) -> dict:
    support = _latest_log_support(stage, name, output)
    if early_stop is not None:
        stopped = early_stop.get("terminal_success_support_b_a_l_r", {})
        for gamma in GAMMAS:
            value = _gamma_key(stopped, gamma)
            if value is None:
                continue
            row = support.setdefault(gamma, {})
            row.update({
                "retry_batch": row.get(
                    "retry_batch", early_stop.get("stopped_at_retry_batch")
                ),
                "terminal_support_b_a_l_r": [int(item) for item in value],
                "source": str((output / "EARLY_STOPPED.json").resolve()),
            })
    if metric is not None:
        details = metric.get("successful_executed_commit_by_gamma", {})
        retries = metric.get("retry_batches_by_gamma", {})
        for gamma in GAMMAS:
            detail = _gamma_key(details, gamma)
            if not isinstance(detail, dict):
                continue
            found = detail.get("success_episode_sample_update_modes", {})
            row = support.setdefault(gamma, {})
            row.update({
                "retry_batch": _gamma_key(retries, gamma),
                "terminal_support_b_a_l_r": _counts(found.values()),
                "committed_b_a_l_r": _counts(
                    detail.get("committed_sample_update_modes")
                ),
                "source": str((output / "metrics.jsonl").resolve()),
            })
    for gamma, row in support.items():
        counts = row.get("terminal_support_b_a_l_r")
        if counts is not None:
            row["missing_from_exact_quota_b_a_l_r"] = [
                max(3 - int(value), 0) for value in counts
            ]
    return {gamma: support.get(gamma) for gamma in GAMMAS}


def _task_geometry(plan: dict, output: Path) -> dict | None:
    candidates = [output / "task_config_resolved.json"]
    configured = plan.get("common", {}).get("task_config")
    if configured:
        candidates.append(Path(configured))
    task = next((value for path in candidates if (value := _read_json(path))), None)
    if task is None:
        return None
    try:
        taskspace = task["taskspace"]
        origin = np.asarray(taskspace["origin"], float)
        size = np.asarray(taskspace["size"], float)
        sphere = np.asarray(task["obstacles"]["spheres"][0], float)
        return {
            "start": np.asarray(taskspace["start"][:3], float),
            "goal": np.asarray(taskspace["goal"][:3], float),
            "bounds": np.column_stack([origin, origin + size]),
            "sphere": sphere,
            "dt": float(task.get("safemppi", {}).get("dt", 0.1)),
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(sum(values) / len(values)) if values else None


def _path_geometry(points: np.ndarray, task: dict) -> dict:
    centre, radius = task["sphere"][:3], float(task["sphere"][3])
    bounds = task["bounds"]
    crossing_radius = None
    x = points[:, 0]
    crossings = np.flatnonzero((x[:-1] < centre[0]) & (x[1:] >= centre[0]))
    if len(crossings):
        index = int(crossings[0])
        fraction = (centre[0] - x[index]) / max(
            x[index + 1] - x[index], 1.0e-12
        )
        crossing = points[index] + fraction * (points[index + 1] - points[index])
        crossing_radius = float(np.linalg.norm(crossing[1:] - centre[1:]))
    return {
        "time_to_goal_s": float((len(points) - 1) * task["dt"]),
        "obstacle_crossing_radius_m": crossing_radius,
        "min_obstacle_clearance_m_state_knots": float(
            np.linalg.norm(points - centre, axis=1).min() - radius
        ),
        "x_max_margin_m": float((bounds[0, 1] - points[:, 0]).min()),
        "y_min_margin_m": float((points[:, 1] - bounds[1, 0]).min()),
    }


def _committed_geometry(
    plan: dict, output: Path, metric: dict | None
) -> dict | None:
    events_path = output / "events_round_001.pt"
    task = _task_geometry(plan, output)
    if metric is None or task is None or not events_path.is_file():
        return None
    try:
        events = torch.load(events_path, map_location="cpu", weights_only=False)
    except (OSError, EOFError, RuntimeError):
        return None
    grouped = {}
    for event in events:
        try:
            key = (float(event["gamma"]), int(event["episode"]))
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(event)
    rows = []
    details = metric.get("successful_executed_commit_by_gamma", {})
    for gamma in GAMMAS:
        detail = _gamma_key(details, gamma)
        if not isinstance(detail, dict):
            continue
        for episode, mode in zip(
            detail.get("committed_episode_ids", []),
            detail.get("committed_sample_update_modes", []),
        ):
            episode_events = grouped.get((float(gamma), int(episode)), [])
            if not episode_events:
                continue
            episode_events.sort(key=lambda event: int(event["step"]))
            points = np.asarray(
                [event["robot"][:3] for event in episode_events]
                + [episode_events[-1]["robot_after"][:3]],
                float,
            )
            row = _path_geometry(points, task)
            row.update({
                "gamma": float(gamma),
                "episode": int(episode),
                "mode": MODES[int(mode)],
            })
            rows.append(row)
    keys = (
        "time_to_goal_s",
        "obstacle_crossing_radius_m",
        "min_obstacle_clearance_m_state_knots",
        "x_max_margin_m",
        "y_min_margin_m",
    )

    def aggregate(selected: list[dict]) -> dict:
        return {"count": len(selected), **{
            key: _mean(selected, key) for key in keys
        }}

    return {
        "measurement_note": (
            "Acquisition event logs preserve 0.1-s state knots, not integration "
            "substeps; minimum clearance and wall margins are knot-based."
        ),
        "pooled_by_mode": {
            mode: aggregate([row for row in rows if row["mode"] == mode])
            for mode in MODES
        },
        "per_gamma_mode": {
            gamma: {
                mode: aggregate([
                    row for row in rows
                    if row["mode"] == mode
                    and float(row["gamma"]) == float(gamma)
                ])
                for mode in MODES
            }
            for gamma in GAMMAS
        },
    }


def _evaluation_candidates(stage: Path, name: str, output: Path) -> list[Path]:
    paths = [
        *output.glob("fixed_eval*/raw_eval.json"),
        *output.glob("evaluations/*/raw_eval.json"),
    ]
    evaluation_root = output.parent / "evaluations"
    if evaluation_root.is_dir():
        paths.extend(evaluation_root.glob(f"{name}*/raw_eval.json"))
    stage_evaluations = stage / "evaluations"
    if stage_evaluations.is_dir():
        paths.extend(stage_evaluations.glob(f"{name}*/raw_eval.json"))
    return sorted(set(paths), key=lambda path: (path.stat().st_mtime, str(path)))


def _load_evaluation(stage: Path, name: str, output: Path) -> tuple[dict, Path] | None:
    published = []
    for path in _evaluation_candidates(stage, name, output):
        payload = _read_json(path)
        if payload is None or payload.get("status") != EVALUATION_STATUS:
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        rounds = sorted((int(key) for key in summary if str(key).isdigit()))
        published.append((max(rounds, default=-1), payload, path))
    if not published:
        return None
    _, payload, path = max(published, key=lambda row: (row[0], row[2].stat().st_mtime))
    return payload, path


def _first_exit(points: np.ndarray, bounds: np.ndarray) -> str | None:
    violations = np.column_stack([
        points[:, 0] < bounds[0, 0], points[:, 0] > bounds[0, 1],
        points[:, 1] < bounds[1, 0], points[:, 1] > bounds[1, 1],
        points[:, 2] < bounds[2, 0], points[:, 2] > bounds[2, 1],
    ])
    indices = np.flatnonzero(violations.any(axis=1))
    if not len(indices):
        return None
    face = int(np.flatnonzero(violations[int(indices[0])])[0])
    return ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")[face]


def _evaluation_summary(
    stage: Path, plan: dict, name: str, output: Path
) -> dict | None:
    loaded = _load_evaluation(stage, name, output)
    if loaded is None:
        return None
    evaluation, raw_eval_path = loaded
    task = _task_geometry(plan, output)
    trajectories_path = raw_eval_path.parent / "raw_trajectories.pt"
    trajectories = None
    if trajectories_path.is_file():
        try:
            trajectories = torch.load(
                trajectories_path, map_location="cpu", weights_only=False
            )
        except (OSError, EOFError, RuntimeError):
            trajectories = None
    rounds = {}
    for round_text in ("0", "1"):
        source = evaluation.get("summary", {}).get(round_text)
        if not isinstance(source, dict):
            continue
        pooled = source.get("pooled", {})
        routes = pooled.get("route_counts", {})
        exits = {face: 0 for face in (
            "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"
        )}
        exits_by_route: dict[str, dict[str, int]] = {}
        trajectory_rows = None
        if isinstance(trajectories, dict):
            trajectory_rows = trajectories.get(
                int(round_text), trajectories.get(round_text)
            )
        if trajectory_rows is not None and task is not None:
            for row in trajectory_rows:
                if row.get("status") != "OOB":
                    continue
                dense = np.asarray(row.get("dense_steps", []))
                if dense.size:
                    points = np.vstack([
                        np.asarray(row["states"])[0, :3], dense.reshape(-1, 3)
                    ])
                else:
                    points = np.asarray(row["states"])[:, :3]
                face = _first_exit(points, task["bounds"])
                if face is None:
                    continue
                exits[face] += 1
                route = str(row.get("mode", "none"))
                exits_by_route.setdefault(route, {})[face] = (
                    exits_by_route.setdefault(route, {}).get(face, 0) + 1
                )
        rounds[round_text] = {
            "SR": pooled.get("SR"),
            "CR": pooled.get("CR"),
            "OOB": pooled.get("OOB"),
            "timeout": pooled.get("timeout"),
            "window_validity": pooled.get("window_validity"),
            "successful_min_clearance_m": pooled.get(
                "successful_min_clearance_m"
            ),
            "successful_time_to_goal_s": pooled.get(
                "successful_time_to_goal_s"
            ),
            "route_coverage": pooled.get("route_coverage"),
            "route_counts_b_a_l_r": [int(routes.get(mode, 0)) for mode in MODES],
            "first_exit_faces": exits,
            "first_exit_faces_by_route": exits_by_route,
        }
    return {
        "raw_eval": str(raw_eval_path.resolve()),
        "raw_trajectories": (
            str(trajectories_path.resolve()) if trajectories_path.is_file() else None
        ),
        "rounds": rounds,
    }


def _r1_metric(output: Path) -> dict | None:
    return next(
        (row for row in _read_metrics(output / "metrics.jsonl")
         if int(row.get("round", -1)) == 1),
        None,
    )


def _state(output: Path, evaluation: dict | None, early_stop: dict | None) -> str:
    if evaluation is not None and "1" in evaluation.get("rounds", {}):
        return "evaluated"
    if (output / "checkpoint_001.pt").is_file():
        return "evaluation_pending"
    if early_stop is not None:
        return "early_stopped"
    if (output / "FAILED.json").is_file():
        return "failed"
    lock = Path(f"{output}.control_braking_process.lock/pid")
    if _pid_alive(lock):
        return "active"
    if (output / "checkpoint_000.pt").is_file():
        return "collecting_or_stale"
    return "not_started"


def _bool_all(checks: dict) -> bool | None:
    values = list(checks.values())
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def _go_checks(arm: dict, control: dict | None) -> dict:
    support = arm["terminal_support"]
    exact = []
    right_support = []
    for gamma in GAMMAS:
        row = support.get(gamma)
        committed = row.get("committed_b_a_l_r") if isinstance(row, dict) else None
        terminal = (
            row.get("terminal_support_b_a_l_r") if isinstance(row, dict) else None
        )
        exact.append(None if committed is None else committed == [3, 3, 3, 3])
        right_support.append(None if terminal is None else terminal[3] >= 1)
    acquisition = {
        "checkpoint1_present": arm["committed_round"] is not None
        and arm["committed_round"] >= 1,
        "exact_3_per_mode_per_gamma": (
            False if False in exact else None if None in exact else True
        ),
        "right_terminal_support_every_gamma": (
            False if False in right_support
            else None if None in right_support else True
        ),
    }
    geometry = arm.get("committed_geometry")
    control_geometry = control.get("committed_geometry") if control else None
    right = (
        geometry.get("pooled_by_mode", {}).get("right") if geometry else None
    )
    control_right = (
        control_geometry.get("pooled_by_mode", {}).get("right")
        if control_geometry else None
    )
    probe_radius = right.get("obstacle_crossing_radius_m") if right else None
    control_radius = (
        control_right.get("obstacle_crossing_radius_m") if control_right else None
    )
    geometry_checks = {
        "right_crossing_radius_moves_inward": (
            None if probe_radius is None or control_radius is None
            else float(probe_radius) < float(control_radius)
        ),
        "right_committed_in_every_gamma": acquisition[
            "exact_3_per_mode_per_gamma"
        ],
    }
    evaluation = arm.get("fixed_evaluation")
    r1 = evaluation.get("rounds", {}).get("1") if evaluation else None
    fixed = {
        "SR_gte_0.30": None if r1 is None else float(r1["SR"]) >= 0.30,
        "OOB_lte_0.075": None if r1 is None else float(r1["OOB"]) <= 0.075,
        "CR_lte_0.65": None if r1 is None else float(r1["CR"]) <= 0.65,
        "validity_gte_0.80": (
            None if r1 is None else float(r1["window_validity"]) >= 0.80
        ),
        "coverage_gte_3_of_4": (
            None if r1 is None else float(r1["route_coverage"]) >= 0.75
        ),
        "right_success_gte_1": (
            None if r1 is None else r1["route_counts_b_a_l_r"][3] >= 1
        ),
        "x_max_exits_lte_3": (
            None if r1 is None else r1["first_exit_faces"]["x_max"] <= 3
        ),
    }
    groups = {
        "acquisition": {**acquisition, "passed": _bool_all(acquisition)},
        "geometry": {
            **geometry_checks,
            "probe_right_crossing_radius_m": probe_radius,
            "historical_control_right_crossing_radius_m": control_radius,
            "passed": _bool_all(geometry_checks),
        },
        "fixed_eval": {**fixed, "passed": _bool_all(fixed)},
    }
    groups["passed"] = _bool_all({
        name: row["passed"] for name, row in groups.items()
    })
    return groups


def _arm(
    stage: Path, plan: dict, record: dict, *, include_go: bool,
    control: dict | None = None,
) -> dict:
    name = record["name"]
    output = Path(record.get("output", stage / "arms" / name))
    metric = _r1_metric(output)
    early_stop = _read_json(output / "EARLY_STOPPED.json")
    evaluation = _evaluation_summary(stage, plan, name, output)
    checkpoints = sorted(output.glob("checkpoint_*.pt"))
    result = {
        **record,
        "output": str(output.resolve()),
        "state": _state(output, evaluation, early_stop),
        "committed_round": (
            int(checkpoints[-1].stem.rsplit("_", 1)[-1]) if checkpoints else None
        ),
        "early_stop": early_stop,
        "terminal_support": _terminal_support(
            stage, name, output, metric, early_stop
        ),
        "committed_geometry": _committed_geometry(plan, output, metric),
        "fixed_evaluation": evaluation,
    }
    if include_go:
        result["go_checks"] = _go_checks(result, control)
    return result


def _delta(probe: dict, control: dict | None) -> dict | None:
    if control is None:
        return None
    output = {}
    probe_r1 = (probe.get("fixed_evaluation") or {}).get("rounds", {}).get("1")
    control_r1 = (control.get("fixed_evaluation") or {}).get("rounds", {}).get("1")
    if probe_r1 is not None and control_r1 is not None:
        output["fixed_r1"] = {
            key: float(probe_r1[key]) - float(control_r1[key])
            for key in ("SR", "CR", "OOB", "timeout", "window_validity")
        }
        output["fixed_r1"]["x_max_exits"] = (
            probe_r1["first_exit_faces"]["x_max"]
            - control_r1["first_exit_faces"]["x_max"]
        )
    probe_geometry = probe.get("committed_geometry")
    control_geometry = control.get("committed_geometry")
    if probe_geometry is not None and control_geometry is not None:
        output["committed_geometry_by_mode"] = {}
        for mode in MODES:
            left = probe_geometry["pooled_by_mode"][mode]
            right = control_geometry["pooled_by_mode"][mode]
            output["committed_geometry_by_mode"][mode] = {
                key: (
                    None if left[key] is None or right[key] is None
                    else float(left[key]) - float(right[key])
                )
                for key in (
                    "time_to_goal_s", "obstacle_crossing_radius_m",
                    "min_obstacle_clearance_m_state_knots",
                    "x_max_margin_m", "y_min_margin_m",
                )
            }
    return output or None


def build(stage: Path) -> dict:
    plan = _read_json(stage / "PROBE_PLAN.json")
    if plan is None:
        raise FileNotFoundError(stage / "PROBE_PLAN.json")
    control_path = plan.get("causal_control")
    control = None
    if control_path:
        control_record = {"name": Path(control_path).name, "output": control_path}
        control = _arm(
            stage, plan, control_record, include_go=False
        )
    arms = []
    for record in plan.get("arms", []):
        arm = _arm(stage, plan, record, include_go=True, control=control)
        arm["delta_from_historical_control"] = _delta(arm, control)
        arms.append(arm)
    go_arms = [arm["name"] for arm in arms if arm["go_checks"]["passed"] is True]
    unresolved = [
        arm for arm in arms
        if arm["state"] not in {"evaluated", "early_stopped", "failed"}
    ]
    status = "GO" if go_arms else "RUNNING" if unresolved else "NO_GO"
    return {
        "status": status,
        "stage": str(stage.resolve()),
        "go_arms": go_arms,
        "fixed_thresholds": FIXED_THRESHOLDS,
        "historical_control": control,
        "arms": arms,
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _markdown(report: dict) -> str:
    lines = [
        "# R1 inward-route/OOB probe",
        "",
        f"Status: **{report['status']}**; GO arms: "
        + (", ".join(report["go_arms"]) or "none"),
        "",
        "## Acquisition support",
        "",
        "| arm | state | gamma | retry | terminal b/a/l/r | committed b/a/l/r |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        for gamma in GAMMAS:
            row = arm["terminal_support"].get(gamma) or {}
            terminal = row.get("terminal_support_b_a_l_r")
            committed = row.get("committed_b_a_l_r")
            lines.append(
                f"| {arm['name']} | {arm['state']} | {gamma} | "
                f"{row.get('retry_batch', '--')} | "
                f"{'/'.join(map(str, terminal)) if terminal else '--'} | "
                f"{'/'.join(map(str, committed)) if committed else '--'} |"
            )
    lines.extend([
        "", "## Committed r1 geometry", "",
        "| arm | mode | n | TtG s | crossing radius m | knot clearance m | x-max margin m | y-min margin m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    displayed = []
    if report.get("historical_control") is not None:
        displayed.append(("historical-control", report["historical_control"]))
    displayed.extend((arm["name"], arm) for arm in report["arms"])
    for label, arm in displayed:
        geometry = arm.get("committed_geometry")
        if geometry is None:
            continue
        for mode in MODES:
            row = geometry["pooled_by_mode"][mode]
            lines.append(
                f"| {label} | {mode} | {row['count']} | "
                f"{_fmt(row['time_to_goal_s'])} | "
                f"{_fmt(row['obstacle_crossing_radius_m'])} | "
                f"{_fmt(row['min_obstacle_clearance_m_state_knots'])} | "
                f"{_fmt(row['x_max_margin_m'])} | "
                f"{_fmt(row['y_min_margin_m'])} |"
            )
    lines.extend([
        "", "## Fixed-bank evaluation", "",
        "| arm | round | SR | CR | OOB | timeout | validity | clearance m | TtG s | routes b/a/l/r | exits x-max/y-min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, arm in displayed:
        evaluation = arm.get("fixed_evaluation")
        if evaluation is None:
            continue
        for round_text, row in evaluation["rounds"].items():
            lines.append(
                f"| {label} | r{round_text} | {_fmt(row['SR'])} | "
                f"{_fmt(row['CR'])} | {_fmt(row['OOB'])} | "
                f"{_fmt(row['timeout'])} | {_fmt(row['window_validity'])} | "
                f"{_fmt(row['successful_min_clearance_m'])} | "
                f"{_fmt(row['successful_time_to_goal_s'], 2)} | "
                f"{'/'.join(map(str, row['route_counts_b_a_l_r']))} | "
                f"{row['first_exit_faces']['x_max']}/"
                f"{row['first_exit_faces']['y_min']} |"
            )
    lines.extend(["", "## GO checks", ""])
    for arm in report["arms"]:
        checks = arm["go_checks"]
        lines.append(
            f"- **{arm['name']}**: acquisition={checks['acquisition']['passed']}, "
            f"geometry={checks['geometry']['passed']}, "
            f"fixed_eval={checks['fixed_eval']['passed']}, "
            f"overall={checks['passed']}; right crossing "
            f"{_fmt(checks['geometry']['probe_right_crossing_radius_m'])} m "
            f"vs control "
            f"{_fmt(checks['geometry']['historical_control_right_crossing_radius_m'])} m."
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build(args.stage)
    if args.write:
        _atomic_write(
            args.stage / "CURRENT_SUMMARY.json",
            json.dumps(report, indent=2) + "\n",
        )
        _atomic_write(args.stage / "CURRENT_SUMMARY.md", _markdown(report))
    print(json.dumps({
        "status": report["status"],
        "go_arms": report["go_arms"],
        "arms": [{
            "name": arm["name"],
            "state": arm["state"],
            "committed_round": arm["committed_round"],
            "go": arm["go_checks"]["passed"],
        } for arm in report["arms"]],
    }, indent=2))


if __name__ == "__main__":
    main()
