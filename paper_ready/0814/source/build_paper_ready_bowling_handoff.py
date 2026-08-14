#!/usr/bin/env python3
"""Select and package paper-ready PRE2/S4 bowling trajectories."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROUTES = ("LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR")
GAMMAS = (0.1, 0.3, 0.5, 1.0)
Z_CENTER_M = 0.9
EFFECTIVE_RADIUS_M = 0.2405
Z_MARGIN_M = 0.3
Z_LOW_M = Z_CENTER_M - EFFECTIVE_RADIUS_M - Z_MARGIN_M
Z_HIGH_M = Z_CENTER_M + EFFECTIVE_RADIUS_M + Z_MARGIN_M
MIN_Z_OCCUPANCY = 0.9
GOAL_JIGGLE_RADIUS_M = 0.8
MAX_GOAL_REVERSE_STEP_M = 0.002
MAX_GOAL_CUMULATIVE_REVERSE_M = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(paths: list[Path], expected_round: int) -> list[dict[str, Any]]:
    unique: dict[tuple[float, int], dict[str, Any]] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = payload.get(str(expected_round))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: missing round {expected_round}")
        for row in rows:
            key = (float(row["gamma"]), int(row["episode"]))
            if key in unique:
                raise ValueError(f"duplicate rollout {key} across raw banks")
            unique[key] = row
    return sorted(unique.values(), key=lambda row: (float(row["gamma"]), int(row["episode"])))


def _load_cfm_rows(specs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows = []
    paths = {}
    for index, spec in enumerate(specs):
        if "=" in spec:
            regime, raw_path = spec.split("=", 1)
            regime = regime.strip()
        else:
            regime, raw_path = None, spec
        path = Path(raw_path).expanduser().resolve()
        path_key = regime or f"combined_{index}"
        if regime == "" or path_key in paths:
            raise ValueError(f"invalid or duplicate CFM--MPPI source: {path_key!r}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise ValueError(f"{path}: CFM--MPPI raw artifact must be a list")
        paths[path_key] = path
        for source in payload:
            row = dict(source)
            row["regime"] = regime or str(source["regime"])
            row["episode"] = int(row.get("trial", row.get("episode", 0)))
            rows.append(row)
    return sorted(rows, key=lambda row: (
        str(row["regime"]), float(row["gamma"]), int(row["episode"]),
    )), paths


def _stable_route(row: dict[str, Any]) -> str | None:
    route = row.get("bowling_route")
    if not isinstance(route, dict):
        return None
    code = str(route.get("stable_code", ""))
    return code if code in ROUTES else None


def _resampled_curvature(states: Any, spacing_m: float = 0.05) -> float:
    positions = np.asarray(states, np.float64)[:, :3]
    segment_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segment_length)))
    keep = np.concatenate(([True], np.diff(arclength) > 1e-7))
    positions = positions[keep]
    arclength = arclength[keep]
    if len(positions) < 3 or arclength[-1] < 2.0 * spacing_m:
        return float("inf")
    query = np.arange(0.0, arclength[-1] + 1e-9, spacing_m)
    if arclength[-1] - query[-1] > 0.5 * spacing_m:
        query = np.append(query, arclength[-1])
    sampled = np.column_stack([
        np.interp(query, arclength, positions[:, axis]) for axis in range(3)
    ])
    segments = np.diff(sampled, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    directions = segments / np.maximum(lengths[:, None], 1e-12)
    angles = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0))
    return float(np.mean(angles) / spacing_m)


def _quality(row: dict[str, Any], goal: Any) -> dict[str, float | bool | int]:
    states = np.asarray(row["states"], np.float64)
    occupancy = float(np.mean((states[:, 2] >= Z_LOW_M) & (states[:, 2] <= Z_HIGH_M)))
    route = row.get("bowling_route") or {}
    decisions = np.asarray(route.get("decision_xyz_m", []), np.float64)
    decision_z = (
        float(np.mean(np.abs(decisions[:, 2] - Z_CENTER_M)))
        if decisions.ndim == 2 and decisions.shape[1] >= 3 and len(decisions)
        else float("inf")
    )
    curvature = _resampled_curvature(states)
    goal_distance = np.linalg.norm(states[:, :3] - np.asarray(goal, np.float64), axis=1)
    goal_progress = goal_distance[:-1] - goal_distance[1:]
    terminal = goal_progress[goal_distance[:-1] <= GOAL_JIGGLE_RADIUS_M]
    reverse = np.maximum(-terminal, 0.0)
    max_reverse = float(np.max(reverse, initial=0.0))
    cumulative_reverse = float(np.sum(reverse))
    goal_progress_pass = (
        max_reverse <= MAX_GOAL_REVERSE_STEP_M
        and cumulative_reverse <= MAX_GOAL_CUMULATIVE_REVERSE_M
    )
    # Dimensionless, fixed before gamma selection: z alignment is primary and
    # curvature is the secondary tie-break requested for paper trajectories.
    score = 0.65 * min(decision_z / 0.4, 3.0) + 0.35 * min(curvature / 3.0, 3.0)
    return {
        "z_band_occupancy": occupancy,
        "hard_z_pass": occupancy >= MIN_Z_OCCUPANCY,
        "decision_mean_abs_z_m": decision_z,
        "mean_curvature_rad_per_m": curvature,
        "quality_score": float(score),
        "goal_jiggle_radius_m": GOAL_JIGGLE_RADIUS_M,
        "goal_reverse_step_count": int(np.count_nonzero(reverse > 0.0)),
        "max_goal_reverse_step_m": max_reverse,
        "cumulative_goal_reverse_m": cumulative_reverse,
        "hard_goal_progress_pass": goal_progress_pass,
    }


def _annotate(rows: list[dict[str, Any]], goal: Any) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item["stable_route"] = _stable_route(row)
        item["paper_quality"] = _quality(row, goal)
        annotated.append(item)
    return annotated


def _successful(rows: list[dict[str, Any]], gamma: float, trial_limit: int | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if np.isclose(float(row["gamma"]), gamma)
        and str(row["status"]) == "SUCCESS"
        and row["stable_route"] in ROUTES
        and (trial_limit is None or int(row["episode"]) < trial_limit)
    ]


def _discovery_trials(rows: list[dict[str, Any]], gamma: float) -> int | None:
    observed: set[str] = set()
    for row in sorted(_successful(rows, gamma), key=lambda item: int(item["episode"])):
        observed.add(row["stable_route"])
        if len(observed) == len(ROUTES):
            return int(row["episode"]) + 1
    return None


def _best_by_route(rows: list[dict[str, Any]], gamma: float) -> dict[str, dict[str, Any]]:
    selected = {}
    for route in ROUTES:
        candidates = [
            row for row in _successful(rows, gamma)
            if row["stable_route"] == route
            and row["paper_quality"]["hard_z_pass"]
            and row["paper_quality"]["hard_goal_progress_pass"]
        ]
        candidates.sort(key=lambda row: (
            row["paper_quality"]["quality_score"],
            row["paper_quality"]["decision_mean_abs_z_m"],
            row["paper_quality"]["mean_curvature_rad_per_m"],
            int(row["episode"]),
        ))
        if candidates:
            selected[route] = candidates[0]
    return selected


def _select_expanded_eight(rows: list[dict[str, Any]], gamma: float) -> list[dict[str, Any]]:
    """Keep the best eligible route representatives, then fill to eight."""
    representatives = _best_by_route(rows, gamma)
    selected = list(representatives.values())
    used = {(float(row["gamma"]), int(row["episode"])) for row in selected}
    remainder = [
        row for row in _successful(rows, gamma)
        if row["paper_quality"]["hard_z_pass"]
        and row["paper_quality"]["hard_goal_progress_pass"]
        and (float(row["gamma"]), int(row["episode"])) not in used
    ]
    remainder.sort(key=lambda row: (
        row["paper_quality"]["quality_score"],
        row["paper_quality"]["cumulative_goal_reverse_m"],
        int(row["episode"]),
    ))
    selected.extend(remainder[:max(0, 8 - len(selected))])
    if len(selected) != 8:
        raise ValueError(f"expanded gamma={gamma:g} has only {len(selected)} eligible trajectories")
    return sorted(selected, key=lambda row: (ROUTES.index(row["stable_route"]), int(row["episode"])))


def _choose_gamma(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    audits = {}
    ranking = []
    for gamma in GAMMAS:
        best = _best_by_route(rows, gamma)
        mean_score = float(np.mean([
            row["paper_quality"]["quality_score"] for row in best.values()
        ])) if best else float("inf")
        discovery = _discovery_trials(rows, gamma)
        audits[str(gamma)] = {
            "hard_mode_count": len(best),
            "hard_modes": sorted(best),
            "mean_representative_quality_score": mean_score if np.isfinite(mean_score) else None,
            "eight_mode_discovery_trials": discovery,
            "available_trials": 1 + max(
                [int(row["episode"]) for row in rows if np.isclose(float(row["gamma"]), gamma)],
                default=-1,
            ),
        }
        ranking.append((-len(best), mean_score, discovery or 10**9, gamma))
    ranking.sort()
    chosen = float(ranking[0][3])
    if audits[str(chosen)]["hard_mode_count"] != 8:
        raise ValueError(f"no gamma has eight hard-eligible modes: {audits}")
    return chosen, audits


def _choose_cfm_gamma(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    balanced = [row for row in rows if row["regime"] == "balanced"]
    if not balanced:
        raise ValueError("CFM--MPPI rows require a balanced regime")
    audits = {}
    ranking = []
    for gamma in GAMMAS:
        candidates = [
            row for row in balanced if np.isclose(float(row["gamma"]), gamma)
        ]
        successes = [row for row in candidates if row["status"] == "SUCCESS"]
        stable = [row for row in successes if row["stable_route"] in ROUTES]
        modes = sorted({row["stable_route"] for row in stable})
        success_rate = len(successes) / max(len(candidates), 1)
        validity = float(np.mean([
            float(row["window_validity"]) for row in candidates
        ])) if candidates else 0.0
        quality = float(np.mean([
            float(row["paper_quality"]["quality_score"]) for row in stable
        ])) if stable else float("inf")
        audits[str(gamma)] = {
            "attempts": len(candidates),
            "successes": len(successes),
            "collisions": sum(row["status"] == "COLLISION" for row in candidates),
            "success_rate": success_rate,
            "mean_window_validity": validity,
            "stable_mode_count": len(modes),
            "stable_modes": modes,
            "mean_stable_quality_score": quality if np.isfinite(quality) else None,
        }
        ranking.append((-success_rate, -validity, -len(modes), quality, gamma))
    ranking.sort()
    return float(ranking[0][-1]), audits


def _select_pre2(rows: list[dict[str, Any]], gamma: float, trials: int) -> list[dict[str, Any]]:
    available_episodes = {
        int(row["episode"]) for row in rows if np.isclose(float(row["gamma"]), gamma)
    }
    missing = sorted(set(range(trials)) - available_episodes)
    if missing:
        raise ValueError(
            f"PRE2 gamma={gamma:g} is not a matched {trials}-trial bank; "
            f"missing episodes begin {missing[:5]}"
        )
    candidates = [
        row for row in _successful(rows, gamma, trials)
        if row["paper_quality"]["hard_z_pass"]
    ]
    by_route = {
        route: sorted(
            [row for row in candidates if row["stable_route"] == route],
            key=lambda row: (row["paper_quality"]["quality_score"], int(row["episode"])),
        )
        for route in ROUTES
    }
    available = [route for route in ROUTES if by_route[route]]
    if len(available) < 2:
        raise ValueError(f"PRE2 did not expose two hard-eligible modes in {trials} trials")
    available.sort(key=lambda route: (-len(by_route[route]), by_route[route][0]["paper_quality"]["quality_score"]))
    chosen_modes = available[:3]
    selected = [by_route[route][0] for route in chosen_modes[1:]]
    used = {(float(row["gamma"]), int(row["episode"])) for row in selected}
    dominant = chosen_modes[0]
    pool = by_route[dominant] + [
        row for route in chosen_modes[1:] for row in by_route[route][1:]
    ]
    pool.sort(key=lambda row: (
        row["stable_route"] != dominant,
        row["paper_quality"]["quality_score"],
        int(row["episode"]),
    ))
    for row in pool:
        key = (float(row["gamma"]), int(row["episode"]))
        if key not in used:
            selected.append(row)
            used.add(key)
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError(f"PRE2 has only {len(selected)} selectable trajectories")
    return selected


def _select_balanced_eight(rows: list[dict[str, Any]], gamma: float) -> list[dict[str, Any]]:
    """Choose at most eight hard-eligible successes with maximum route balance."""
    by_route = {
        route: sorted(
            [
                row for row in _successful(rows, gamma)
                if row["stable_route"] == route and row["paper_quality"]["hard_z_pass"]
            ],
            key=lambda row: (row["paper_quality"]["quality_score"], int(row["episode"])),
        )
        for route in ROUTES
    }
    selected = []
    depth = 0
    while len(selected) < 8:
        added = False
        for route in ROUTES:
            if depth < len(by_route[route]):
                selected.append(by_route[route][depth])
                added = True
                if len(selected) == 8:
                    break
        if not added:
            break
        depth += 1
    if not selected:
        raise ValueError(f"SafeMPPI gamma={gamma:g} has no hard-eligible stable success")
    return selected


def _encode_states(states: Any) -> dict[str, Any]:
    values = np.asarray(states, np.float64)
    positions = np.rint(values[:, :3] * 1000.0).astype(np.int64)
    deltas = np.diff(positions, axis=0)
    speed = np.rint(np.linalg.norm(values[:, 3:6], axis=1) * 1000.0).astype(np.int64)
    blob = b"".join((
        positions[0].astype("<i2").tobytes(),
        deltas.astype("<i2").tobytes(),
        speed.astype("<u2").tobytes(),
    ))
    return {"n": int(len(values)), "b": base64.b64encode(blob).decode("ascii")}


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _viz_row(row: dict[str, Any], model: str) -> dict[str, Any]:
    regime = row.get("regime")
    identifier = f"{model}:g{float(row['gamma']):g}:e{int(row['episode'])}"
    if regime:
        identifier = f"{model}:{regime}:g{float(row['gamma']):g}:e{int(row['episode'])}"
    return {
        "id": identifier,
        "model": model,
        "gamma": float(row["gamma"]),
        "episode": int(row["episode"]),
        "seed": int(row["rollout_seed"]),
        "route": row["stable_route"],
        "status": str(row.get("status", "SUCCESS")),
        "regime": regime,
        "normalized_goal": row.get("normalized_goal"),
        "normalized_safety": row.get("normalized_safety"),
        "states": _encode_states(row["states"]),
        "clearance": row.get("min_clearance_m"),
        "ttg": row.get("time_to_goal_s"),
        "validity": row.get("window_validity"),
        "quality": _json_finite(row["paper_quality"]),
    }


def _portable_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = dict(row)
    keep.pop("stable_route", None)
    keep.pop("paper_quality", None)
    return keep


def _selection_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = {
        "gamma": float(row["gamma"]),
        "episode": int(row["episode"]),
        "seed": int(row["rollout_seed"]),
        "route": row["stable_route"],
        **_json_finite(row["paper_quality"]),
    }
    if row.get("regime") is not None:
        selected.update({
            "regime": str(row["regime"]),
            "status": str(row["status"]),
            "normalized_goal": float(row["normalized_goal"]),
            "normalized_safety": float(row["normalized_safety"]),
        })
    return selected


def _outer_document(inner: str) -> str:
    csp = (
        "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src data: blob:; "
        "font-src data:; connect-src data: blob:; frame-src 'self'; object-src 'none'; "
        "base-uri 'none'; form-action 'none'"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="{csp}">
<title>Paper-ready Bowling Route Handoff</title><style>:root{{color-scheme:light dark}}html,body{{margin:0}}iframe{{display:block;width:100%;height:100vh;border:0}}</style></head>
<body><iframe sandbox="allow-scripts" referrerpolicy="no-referrer" title="Paper-ready bowling route handoff" srcdoc="{html.escape(inner, quote=True)}"></iframe></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre2-eval", type=Path, required=True)
    parser.add_argument("--pre2-raw", type=Path, action="append", required=True)
    parser.add_argument("--expanded-raw", type=Path, action="append", required=True)
    parser.add_argument("--safemppi-raw", type=Path)
    parser.add_argument("--cfm-raw", action="append", default=[])
    parser.add_argument(
        "--cfm-paper-gamma",
        type=float,
        help="fixed comparison gamma to promote instead of data-driven selection",
    )
    parser.add_argument("--legacy-data", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-output", type=Path)
    args = parser.parse_args()

    evaluation = json.loads(args.pre2_eval.read_text())
    scene = evaluation["scene"]
    goal = scene["goal"]
    pre2 = _annotate(_load_rows(args.pre2_raw, 0), goal)
    expanded = _annotate(_load_rows(args.expanded_raw, 4), goal)
    safemppi = None
    if args.safemppi_raw is not None:
        raw = torch.load(args.safemppi_raw, map_location="cpu", weights_only=False)
        if not isinstance(raw.get("safemppi"), list):
            raise ValueError("SafeMPPI raw bank is missing its safemppi rows")
        safemppi = _annotate(raw["safemppi"], goal)
    cfm, cfm_paths = _load_cfm_rows(args.cfm_raw)
    cfm = _annotate(cfm, goal) if cfm else []
    cfm_gamma = None
    cfm_gamma_audit = None
    if cfm:
        selected_by_data, cfm_gamma_audit = _choose_cfm_gamma(cfm)
        cfm_gamma = (
            float(args.cfm_paper_gamma)
            if args.cfm_paper_gamma is not None else selected_by_data
        )
        if not any(np.isclose(cfm_gamma, gamma) for gamma in GAMMAS):
            raise ValueError(f"CFM--MPPI paper gamma is not configured: {cfm_gamma:g}")
    chosen_gamma, gamma_audit = _choose_gamma(expanded)
    discovery_trials = _discovery_trials(expanded, chosen_gamma)
    assert discovery_trials is not None

    paper_expanded = list(_best_by_route(expanded, chosen_gamma).values())
    paper_expanded.sort(key=lambda row: ROUTES.index(row["stable_route"]))
    paper_pre2 = _select_pre2(pre2, chosen_gamma, discovery_trials)
    groups: dict[str, list[dict[str, Any]] | None] = {
        "paper-ready-pre2": paper_pre2,
        "paper-ready-expanded": paper_expanded,
        "paper-ready-cfmmppi": None if not cfm else [
            row for row in cfm if np.isclose(float(row["gamma"]), cfm_gamma)
        ],
        "paper-ready-safemppi": None if safemppi is None else [
            row for gamma in GAMMAS for row in _select_balanced_eight(safemppi, gamma)
        ],
        "not-paper-ready-pre2": [],
        "not-paper-ready-expanded": [],
        "not-paper-ready-cfmmppi": None if not cfm else [
            row for row in cfm if not np.isclose(float(row["gamma"]), cfm_gamma)
        ],
    }
    for gamma in GAMMAS:
        if np.isclose(gamma, chosen_gamma):
            continue
        groups["not-paper-ready-expanded"].extend(_select_expanded_eight(expanded, gamma))
        available_trials = gamma_audit[str(gamma)]["available_trials"]
        try:
            groups["not-paper-ready-pre2"].extend(_select_pre2(pre2, gamma, available_trials))
        except ValueError:
            fallback = [
                row for row in _successful(pre2, gamma)
                if row["paper_quality"]["hard_z_pass"]
            ]
            fallback.sort(key=lambda row: row["paper_quality"]["quality_score"])
            groups["not-paper-ready-pre2"].extend(fallback[:8])

    pre2_observed = sorted({row["stable_route"] for row in _successful(pre2, chosen_gamma, discovery_trials)})
    audit = {
        "selected_gamma": chosen_gamma,
        "expanded_trials_to_all_8_modes": discovery_trials,
        "pre2_trials_matched": discovery_trials,
        "pre2_modes_in_matched_trials": pre2_observed,
        "pre2_mode_count_in_matched_trials": len(pre2_observed),
        "expanded_gamma_search": gamma_audit,
        "hard_z_contract": {
            "center_m": Z_CENTER_M,
            "effective_radius_m": EFFECTIVE_RADIUS_M,
            "margin_m": Z_MARGIN_M,
            "band_m": [Z_LOW_M, Z_HIGH_M],
            "minimum_state_occupancy": MIN_Z_OCCUPANCY,
        },
        "goal_progress_contract": {
            "terminal_radius_m": GOAL_JIGGLE_RADIUS_M,
            "maximum_reverse_step_m": MAX_GOAL_REVERSE_STEP_M,
            "maximum_cumulative_reverse_m": MAX_GOAL_CUMULATIVE_REVERSE_M,
            "scope": "displayed closed-loop control states",
        },
        "quality_score": "0.65*min(mean|decision_z-0.9|/0.4,3) + 0.35*min(mean resampled curvature/3,3); lower is better",
    }
    if safemppi is not None:
        audit["safemppi"] = {
            "source_git_sha": "dabb5011dfc674864e1de275a1e1c2adab58f4af",
            "controller_sha256": "dfc91a26ccac2818c902215bf4d9a06e405d5878e5c6af0be2f75c4f68106dad",
            "attempts_per_gamma": {
                str(gamma): int(sum(np.isclose(float(row["gamma"]), gamma) for row in safemppi))
                for gamma in GAMMAS
            },
            "selected_route_counts": {
                str(gamma): {
                    route: int(sum(
                        np.isclose(float(row["gamma"]), gamma)
                        and row["stable_route"] == route
                        for row in groups["paper-ready-safemppi"]
                    ))
                    for route in ROUTES
                }
                for gamma in GAMMAS
            },
        }
    if cfm:
        audit["cfmmppi"] = {
            "selected_gamma": cfm_gamma,
            "gamma_selection_rule": (
                "fixed to the expanded paper-ready gamma for matched comparison"
                if args.cfm_paper_gamma is not None else
                "balanced success rate, then validity, stable mode count, "
                "trajectory quality, gamma"
            ),
            "data_driven_gamma_without_fixed_match": selected_by_data,
            "gamma_search": cfm_gamma_audit,
            "proposal_count": 32,
            "elite_count": 8,
            "copies_per_elite": 32,
            "nfe": 16,
            "cbf_alpha": 0.5,
            "cbf_margin_m": 0.1,
            "regimes": {
                regime: {
                    "normalized_goal": float(rows[0]["normalized_goal"]),
                    "normalized_safety": float(rows[0]["normalized_safety"]),
                }
                for regime in sorted({row["regime"] for row in cfm})
                for rows in [[item for item in cfm if item["regime"] == regime]]
            },
        }
    legacy_groups = {}
    if args.legacy_data is not None:
        legacy = json.loads(args.legacy_data.read_text())
        legacy_groups = {
            "legacy-pre2": legacy["pre2"],
            "legacy-s4": legacy["s4"],
            "legacy-s4-distinct": [row for row in legacy["s4"] if row.get("curated")],
        }
    payload = {
        "contract": "faithful raw NFE16 · M1 · speed400 · fixed bowling 1–2–3",
        "dt": 0.1,
        "routes": list(ROUTES),
        "gammas": list(GAMMAS),
        "scene": scene,
        "audit": audit,
        "groups": {
            name: None if rows is None else [_viz_row(row, "PRE2" if "pre2" in name else "Expanded") for row in rows]
            for name, rows in groups.items()
        },
    }
    if groups["paper-ready-safemppi"] is not None:
        payload["groups"]["paper-ready-safemppi"] = [
            _viz_row(row, "SafeMPPI") for row in groups["paper-ready-safemppi"]
        ]
    for name in ("paper-ready-cfmmppi", "not-paper-ready-cfmmppi"):
        if groups[name] is not None:
            payload["groups"][name] = [
                _viz_row(row, "CFM-MPPI") for row in groups[name]
            ]
    payload["groups"].update(legacy_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "paper_ready_bowling_selection.json"
    handoff_path = args.output_dir / "paper_ready_bowling_handoff.pt"
    inner_path = args.output_dir / "paper_ready_bowling_handoff.inner.html"
    site_path = args.site_output or args.output_dir / "paper_ready_bowling_handoff.html"
    template = args.template.read_text()
    marker = "__PAPER_READY_BOWLING_DATA__"
    if template.count(marker) != 1:
        raise ValueError("template must contain exactly one data marker")
    inner = template.replace(marker, json.dumps(payload, separators=(",", ":"), allow_nan=False))
    inner_path.write_text(inner)
    site_path.parent.mkdir(parents=True, exist_ok=True)
    site_path.write_text(_outer_document(inner))
    selection_path.write_text(json.dumps({
        **audit,
        "groups": {
            name: None if rows is None else [_selection_row(row) for row in rows]
            for name, rows in groups.items()
        },
        "input_sha256": {
            "pre2_eval": _sha256(args.pre2_eval),
            **{f"pre2_raw_{i}": _sha256(path) for i, path in enumerate(args.pre2_raw)},
            **{f"expanded_raw_{i}": _sha256(path) for i, path in enumerate(args.expanded_raw)},
            **({"safemppi_raw": _sha256(args.safemppi_raw)} if args.safemppi_raw else {}),
            **{
                f"cfmmppi_{regime}": _sha256(path)
                for regime, path in sorted(cfm_paths.items())
            },
        },
    }, indent=2) + "\n")
    torch.save({
        "audit": audit,
        "scene": scene,
        "groups": {
            name: None if rows is None else [_portable_row(row) for row in rows]
            for name, rows in groups.items()
        },
    }, handoff_path)
    print(json.dumps({
        "selected_gamma": chosen_gamma,
        "discovery_trials": discovery_trials,
        "pre2_modes": pre2_observed,
        "paper_expanded_routes": [row["stable_route"] for row in paper_expanded],
        "cfmmppi_selected_gamma": cfm_gamma,
        "selection": str(selection_path),
        "handoff": str(handoff_path),
        "site": str(site_path),
    }, indent=2))


if __name__ == "__main__":
    main()
