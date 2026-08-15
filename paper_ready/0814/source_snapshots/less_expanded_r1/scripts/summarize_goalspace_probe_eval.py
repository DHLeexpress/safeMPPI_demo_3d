#!/usr/bin/env python3
"""Summarize fixed-bank goal-space probes and their dual-boundary audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
MODES = ("below", "above", "left", "right")
METRICS = (
    "SR",
    "CR",
    "OOB",
    "timeout",
    "window_validity",
    "successful_min_clearance_m",
    "successful_time_to_goal_s",
    "route_coverage",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arm(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("arm must be LABEL=EVAL_DIR")
    label, raw_path = text.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("arm must be LABEL=EVAL_DIR")
    return label, Path(raw_path)


def _bounds(text: str) -> str:
    values = text.split(",")
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "bounds must be xmin,xmax,ymin,ymax,zmin,zmax"
        )
    for value in values:
        float(value)
    return text


def _monotone_nonincreasing(values: list[float | None]) -> bool | None:
    if any(value is None for value in values):
        return None
    finite = [float(value) for value in values]
    return all(
        right <= left + 1e-12
        for left, right in zip(finite, finite[1:])
    )


def _mean_se(values: list[float]) -> tuple[float | None, float]:
    if not values:
        return None, 0.0
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, float(math.sqrt(variance / len(values)))


def _paper_curve_row(
    round_i: int,
    gamma: float,
    rows: list[dict],
    summary: dict,
) -> dict:
    gamma_rows = [
        row for row in rows if float(row["gamma"]) == float(gamma)
    ]
    successes = [
        row for row in gamma_rows if row["status"] == "SUCCESS"
    ]
    validity_mean, validity_se = _mean_se([
        float(row["window_validity"]) for row in gamma_rows
    ])
    clearance_mean, clearance_se = _mean_se([
        float(row["min_clearance_m"])
        for row in successes if row.get("min_clearance_m") is not None
    ])
    time_mean, time_se = _mean_se([
        float(row["time_to_goal_s"])
        for row in successes if row.get("time_to_goal_s") is not None
    ])
    return {
        "round": round_i,
        "gamma": float(gamma),
        "m": len(gamma_rows),
        "temp": 1.0,
        "CR": {"mean": float(summary["CR"]), "se": 0.0},
        "v_safe": {"mean": validity_mean, "se": validity_se},
        "clearance": {"mean": clearance_mean, "se": clearance_se},
        "time": {"mean": time_mean, "se": time_se},
    }


def _metric_row(values: dict) -> dict:
    routes = values.get("route_counts", {})
    route_counts = {
        mode: int(routes.get(mode, 0)) for mode in MODES
    }
    success_count = sum(route_counts.values())
    left_right = route_counts["left"] + route_counts["right"]
    below_above = route_counts["below"] + route_counts["above"]
    shares = [
        route_counts[mode] / success_count if success_count else 0.0
        for mode in MODES
    ]
    return {
        **{key: values.get(key) for key in METRICS},
        "route_counts": route_counts,
        "route_symmetry": {
            "left_right_signed_bias": (
                (route_counts["right"] - route_counts["left"])
                / left_right if left_right else None
            ),
            "left_right_absolute_imbalance": (
                abs(route_counts["right"] - route_counts["left"])
                / left_right if left_right else None
            ),
            "below_above_signed_bias": (
                (route_counts["above"] - route_counts["below"])
                / below_above if below_above else None
            ),
            "below_above_absolute_imbalance": (
                abs(route_counts["above"] - route_counts["below"])
                / below_above if below_above else None
            ),
            "four_mode_l1_from_uniform": (
                sum(abs(share - 0.25) for share in shares)
                if success_count else None
            ),
        },
    }


def _ensure_audit(
    eval_dir: Path,
    old_bounds: str,
    new_bounds: str,
) -> Path:
    trajectories = eval_dir / "raw_trajectories.pt"
    audit = eval_dir / "dual_bounds_audit.json"
    regenerate = not audit.is_file() or audit.stat().st_mtime < trajectories.stat().st_mtime
    if not regenerate:
        payload = json.loads(audit.read_text())
        expected_old = [
            [float(value) for value in old_bounds.split(",")[offset:offset + 2]]
            for offset in range(0, 6, 2)
        ]
        expected_new = [
            [float(value) for value in new_bounds.split(",")[offset:offset + 2]]
            for offset in range(0, 6, 2)
        ]
        regenerate = (
            Path(payload.get("source", "")) != trajectories.resolve()
            or payload.get("old_bounds") != expected_old
            or payload.get("new_bounds") != expected_new
        )
    if regenerate:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/audit_taskspace_boundary_crossings.py"),
                "--trajectories",
                str(trajectories),
                "--output",
                str(audit),
                f"--old-bounds={old_bounds}",
                f"--new-bounds={new_bounds}",
            ],
            check=True,
        )
    return audit


def _audit_round(audit: dict, round_i: int) -> dict:
    episodes = [
        row for row in audit["episodes"] if int(row["round"]) == round_i
    ]
    new_faces = {}
    for row in episodes:
        face = row.get("first_new_exit_face")
        if face is not None:
            new_faces[face] = new_faces.get(face, 0) + 1
    old_y = [row for row in episodes if row["crossed_old_y_min"]]
    turnaround = [
        float(row["turnaround_after_old_y_crossing_s"])
        for row in old_y
        if row["turnaround_after_old_y_crossing_s"] is not None
    ]
    x_max = int(new_faces.get("x_max", 0))
    y_min = int(new_faces.get("y_min", 0))
    goal_side_total = x_max + y_min
    return {
        "episodes": len(episodes),
        "old_y_min_crossings": len(old_y),
        "old_y_min_crossing_then_return": sum(
            row["returned_inside_old_y_min"] for row in old_y
        ),
        "old_y_min_crossing_then_success": sum(
            row["successful_after_old_y_crossing"] for row in old_y
        ),
        "old_y_min_crossing_then_return_and_success": sum(
            row["returned_inside_old_y_min"]
            and row["successful_after_old_y_crossing"]
            for row in old_y
        ),
        "new_boundary_crossings": sum(
            row["new_boundary_crossing"] for row in episodes
        ),
        "new_first_exit_faces": {
            key: int(new_faces[key]) for key in sorted(new_faces)
        },
        "goal_side_exit_symmetry": {
            "x_max": x_max,
            "y_min": y_min,
            "signed_bias_xmax_minus_ymin": (
                (x_max - y_min) / goal_side_total
                if goal_side_total else None
            ),
            "absolute_imbalance": (
                abs(x_max - y_min) / goal_side_total
                if goal_side_total else None
            ),
        },
        "turnaround_after_old_y_crossing_s_mean": (
            sum(turnaround) / len(turnaround) if turnaround else None
        ),
    }


def _gamma_trend(per_gamma: dict, gamma_order: list[str]) -> dict:
    rows = [_metric_row(per_gamma[gamma]) for gamma in gamma_order]
    ttg = [row["successful_time_to_goal_s"] for row in rows]
    clearance = [row["successful_min_clearance_m"] for row in rows]
    return {
        "gamma_order": [float(gamma) for gamma in gamma_order],
        "SR": [row["SR"] for row in rows],
        "TtG_s": ttg,
        "clearance_m": clearance,
        "TtG_nonincreasing_with_gamma": _monotone_nonincreasing(ttg),
        "clearance_nonincreasing_with_gamma": _monotone_nonincreasing(
            clearance
        ),
    }


def _delta(left: dict, right: dict) -> dict:
    pooled = {}
    for key in METRICS:
        before = left["pooled"].get(key)
        after = right["pooled"].get(key)
        pooled[key] = (
            None if before is None or after is None else float(after - before)
        )
    routes = {
        mode: (
            right["pooled"]["route_counts"][mode]
            - left["pooled"]["route_counts"][mode]
        )
        for mode in MODES
    }
    route_symmetry = {}
    for key, before in left["pooled"]["route_symmetry"].items():
        after = right["pooled"]["route_symmetry"][key]
        route_symmetry[key] = (
            None if before is None or after is None else float(after - before)
        )
    boundary = {
        key: right["boundary"][key] - left["boundary"][key]
        for key in (
            "old_y_min_crossings",
            "old_y_min_crossing_then_return",
            "old_y_min_crossing_then_success",
            "old_y_min_crossing_then_return_and_success",
            "new_boundary_crossings",
        )
    }
    for face in ("x_max", "y_min"):
        boundary[f"goal_side_{face}_exits"] = (
            right["boundary"]["goal_side_exit_symmetry"][face]
            - left["boundary"]["goal_side_exit_symmetry"][face]
        )
    before_imbalance = left["boundary"]["goal_side_exit_symmetry"][
        "absolute_imbalance"
    ]
    after_imbalance = right["boundary"]["goal_side_exit_symmetry"][
        "absolute_imbalance"
    ]
    boundary["goal_side_exit_absolute_imbalance"] = (
        None
        if before_imbalance is None or after_imbalance is None
        else float(after_imbalance - before_imbalance)
    )
    return {
        "pooled": pooled,
        "route_counts": routes,
        "route_symmetry": route_symmetry,
        "boundary": boundary,
    }


def _fmt(value, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _markdown(report: dict) -> str:
    lines = [
        "# Goal-space fixed-bank report",
        "",
        (
            f"Protocol: seed {report['protocol']['seed']}, NFE "
            f"{report['protocol']['flow_nfe']}, "
            f"{report['protocol']['episodes_per_gamma']} episodes/gamma; "
            f"fixed-bank identity validated: "
            f"{report['protocol']['fixed_bank_identity_validated']}."
        ),
        "",
        "| arm | round | SR | CR | OOB | timeout | validity | clearance m | TtG s | coverage | b/a/l/r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arm in report["arms"].items():
        for round_text, row in arm["rounds"].items():
            pooled = row["pooled"]
            routes = pooled["route_counts"]
            lines.append(
                f"| {label} | r{round_text} | {_fmt(pooled['SR'])} | "
                f"{_fmt(pooled['CR'])} | {_fmt(pooled['OOB'])} | "
                f"{_fmt(pooled['timeout'])} | "
                f"{_fmt(pooled['window_validity'])} | "
                f"{_fmt(pooled['successful_min_clearance_m'])} | "
                f"{_fmt(pooled['successful_time_to_goal_s'], 2)} | "
                f"{_fmt(pooled['route_coverage'])} | "
                f"{routes['below']}/{routes['above']}/"
                f"{routes['left']}/{routes['right']} |"
            )
    lines.extend([
        "",
        "| arm | round | old-y crossed | returned | success after crossing | return+success | new exits | x-max/y-min | face imbalance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, arm in report["arms"].items():
        for round_text, row in arm["rounds"].items():
            boundary = row["boundary"]
            symmetry = boundary["goal_side_exit_symmetry"]
            lines.append(
                f"| {label} | r{round_text} | "
                f"{boundary['old_y_min_crossings']} | "
                f"{boundary['old_y_min_crossing_then_return']} | "
                f"{boundary['old_y_min_crossing_then_success']} | "
                f"{boundary['old_y_min_crossing_then_return_and_success']} | "
                f"{boundary['new_boundary_crossings']} | "
                f"{symmetry['x_max']}/{symmetry['y_min']} | "
                f"{_fmt(symmetry['absolute_imbalance'])} |"
            )
    lines.extend(["", "## Gamma trends", ""])
    for label, arm in report["arms"].items():
        for round_text, row in arm["rounds"].items():
            trend = row["gamma_trend"]
            lines.append(
                f"- {label} r{round_text}: SR={trend['SR']}; "
                f"TtG={trend['TtG_s']} (nonincreasing="
                f"{trend['TtG_nonincreasing_with_gamma']}); clearance="
                f"{trend['clearance_m']} (nonincreasing="
                f"{trend['clearance_nonincreasing_with_gamma']})."
            )
    comparisons = report.get("comparisons_to_legacy", {})
    if comparisons:
        lines.extend(["", "## Delta from legacy old-space w50", ""])
        for label, rounds in comparisons.items():
            for round_text, delta in rounds.items():
                lines.append(
                    f"- {label} r{round_text}: delta SR "
                    f"{_fmt(delta['pooled']['SR'])}, delta CR "
                    f"{_fmt(delta['pooled']['CR'])}, delta OOB "
                    f"{_fmt(delta['pooled']['OOB'])}."
                )
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict:
    task = json.loads(args.task_config.read_text())
    gammas = [f"{float(value):g}" for value in task["data"]["gammas"]]
    new_bounds = args.new_bounds or ",".join(
        f"{float(value):g}"
        for pair in zip(
            task["taskspace"]["origin"], task["taskspace"]["size"]
        )
        for value in (pair[0], pair[0] + pair[1])
    )
    arms = {}
    bank_identity = None
    bank_valid = True
    inputs = {str(args.task_config.resolve()): _sha256(args.task_config)}

    for label, eval_dir in args.arm:
        raw_eval = eval_dir / "raw_eval.json"
        trajectories_path = eval_dir / "raw_trajectories.pt"
        if not raw_eval.is_file() or not trajectories_path.is_file():
            raise FileNotFoundError(f"incomplete evaluation: {eval_dir}")
        evaluation = json.loads(raw_eval.read_text())
        if evaluation.get("status") != "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE":
            raise ValueError(f"evaluation is not published: {raw_eval}")
        if evaluation.get("sampling_temperature") != 1.0:
            raise ValueError(f"not a temperature-1 raw evaluation: {raw_eval}")
        if evaluation.get("sigma_tilt_used") is not False:
            raise ValueError(f"raw evaluation unexpectedly used tilt: {raw_eval}")
        if evaluation.get("flow_nfe_override") != args.flow_nfe:
            raise ValueError(
                f"expected NFE {args.flow_nfe}, got "
                f"{evaluation.get('flow_nfe_override')} in {raw_eval}"
            )
        trajectories = torch.load(
            trajectories_path, map_location="cpu", weights_only=False
        )
        audit_path = _ensure_audit(
            eval_dir, args.old_bounds, new_bounds,
        )
        audit = json.loads(audit_path.read_text())
        rounds = {}
        paper_curve_rows = []
        for round_text in sorted(evaluation["summary"], key=int):
            round_i = int(round_text)
            rows = trajectories.get(round_i, trajectories.get(round_text))
            if rows is None:
                raise KeyError(f"round {round_i} missing from {trajectories_path}")
            identity = sorted(
                (f"{float(row['gamma']):g}", int(row["episode"]))
                for row in rows
            )
            expected = sorted(
                (gamma, episode)
                for gamma in gammas
                for episode in range(args.episodes_per_gamma)
            )
            bank_valid &= identity == expected
            if bank_identity is None:
                bank_identity = identity
            else:
                bank_valid &= identity == bank_identity
            summary = evaluation["summary"][round_text]
            for gamma in gammas:
                paper_curve_rows.append(_paper_curve_row(
                    round_i,
                    float(gamma),
                    rows,
                    summary["per_gamma"][gamma],
                ))
            rounds[round_text] = {
                "pooled": _metric_row(summary["pooled"]),
                "per_gamma": {
                    gamma: _metric_row(summary["per_gamma"][gamma])
                    for gamma in gammas
                },
                "gamma_trend": _gamma_trend(summary["per_gamma"], gammas),
                "boundary": _audit_round(audit, round_i),
            }
        deltas = {}
        round_numbers = sorted(map(int, rounds))
        for left, right in zip(round_numbers, round_numbers[1:]):
            deltas[f"r{left}_to_r{right}"] = _delta(
                rounds[str(left)], rounds[str(right)]
            )
        arms[label] = {
            "eval_dir": str(eval_dir.resolve()),
            "rounds": rounds,
            "adjacent_round_deltas": deltas,
            "paper_curve_rows": paper_curve_rows,
        }
        for path in (raw_eval, trajectories_path, audit_path):
            inputs[str(path.resolve())] = _sha256(path)

    report = {
        "status": "GOALSPACE_FIXED_BANK_REPORT_COMPLETE",
        "protocol": {
            "seed": args.seed,
            "flow_nfe": args.flow_nfe,
            "episodes_per_gamma": args.episodes_per_gamma,
            "sampling_temperature": 1.0,
            "sigma_tilt_used": False,
            "episode_seed_formula": "seed + 37 * episode",
            "fixed_bank_identity_validated": bank_valid,
            "gammas": [float(value) for value in gammas],
        },
        "old_bounds": [float(value) for value in args.old_bounds.split(",")],
        "new_bounds": [float(value) for value in new_bounds.split(",")],
        "arms": arms,
        "inputs_sha256": inputs,
    }
    legacy_label = getattr(args, "legacy_arm", None)
    if legacy_label is not None:
        if legacy_label not in arms:
            raise ValueError(f"legacy arm not found: {legacy_label}")
        legacy = arms[legacy_label]
        report["legacy_arm"] = legacy_label
        report["legacy_sr_curve"] = {
            round_text: row["pooled"]["SR"]
            for round_text, row in legacy["rounds"].items()
        }
        report["comparisons_to_legacy"] = {}
        for label, arm in arms.items():
            if label == legacy_label:
                continue
            common = sorted(
                set(legacy["rounds"]) & set(arm["rounds"]), key=int,
            )
            report["comparisons_to_legacy"][label] = {
                round_text: _delta(
                    legacy["rounds"][round_text], arm["rounds"][round_text]
                )
                for round_text in common
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_arm, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--old-bounds",
        type=_bounds,
        default="-2.5,1.3,-1.7,1.8,0.1,1.7",
    )
    parser.add_argument("--new-bounds", type=_bounds)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--flow-nfe", type=int, default=12)
    parser.add_argument("--episodes-per-gamma", type=int, default=40)
    parser.add_argument(
        "--legacy-arm",
        help="arm label used as the old-space roundwise comparison",
    )
    args = parser.parse_args()
    report = build_report(args)
    if not report["protocol"]["fixed_bank_identity_validated"]:
        raise ValueError("evaluations do not share the approved fixed bank")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, arm in report["arms"].items():
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-") or "arm"
        jsonl = args.output.parent / f"{args.output.stem}_{slug}.jsonl"
        jsonl.write_text("".join(
            json.dumps(row) + "\n" for row in arm["paper_curve_rows"]
        ))
        arm["paper_curve_jsonl"] = str(jsonl.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(_markdown(report))
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve()),
        "markdown": str(markdown.resolve()),
        "arms": list(report["arms"]),
    }, indent=2))


if __name__ == "__main__":
    main()
