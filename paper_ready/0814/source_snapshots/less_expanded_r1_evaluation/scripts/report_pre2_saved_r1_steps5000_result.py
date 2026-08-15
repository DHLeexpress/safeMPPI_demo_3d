#!/usr/bin/env python3
"""Write the final paired native-raw comparison for the saved-r1 study."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/stage1_single_ball_t128/0812_pre2_saved_r1_steps5000_r7"
MODES = ("below", "above", "left", "right")
ARMS = {
    "w150_steps5000": {
        "label": "speed150 / 5,000 Adam steps",
        "speed_weight": 150,
        "optimizer_steps_per_round": 5000,
        "raw_eval": STAGE / "final_inputs/w150_steps5000_eval/raw_eval.json",
    },
    "w300_steps2500": {
        "label": "speed300 / 2,500 Adam steps",
        "speed_weight": 300,
        "optimizer_steps_per_round": 2500,
        "raw_eval": STAGE / "final_inputs/w300_steps2500_eval/raw_eval.json",
    },
    "w300_steps5000": {
        "label": "speed300 / 5,000 Adam steps",
        "speed_weight": 300,
        "optimizer_steps_per_round": 5000,
        "raw_eval": STAGE / "final_inputs/w300_steps5000_eval/raw_eval.json",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _route(counts: dict[str, Any]) -> dict[str, Any]:
    normalized = {mode: int(counts.get(mode, 0)) for mode in MODES}
    total = sum(normalized.values())
    shares = {
        mode: normalized[mode] / total if total else 0.0 for mode in MODES
    }
    entropy = -sum(
        share * math.log(share) for share in shares.values() if share > 0
    ) / math.log(4)
    return {
        "counts": normalized,
        "shares": shares,
        "successful_episodes": total,
        "coverage_modes": sum(value > 0 for value in normalized.values()),
        "normalized_entropy": entropy,
        "minimum_share": min(shares.values()),
        "L1_from_uniform": sum(abs(share - 0.25) for share in shares.values()),
    }


def _metric(cell: dict[str, Any]) -> dict[str, float]:
    return {
        "SR": float(cell["SR"]),
        "CR": float(cell["CR"]),
        "OOB": float(cell["OOB"]),
        "timeout": float(cell["timeout"]),
        "OOB_plus_timeout": float(cell["OOB"]) + float(cell["timeout"]),
        "validity": float(cell["window_validity"]),
        "clearance_m": float(cell["successful_min_clearance_m"]),
        "TtG_s": float(cell["successful_time_to_goal_s"]),
    }


def _trend(values: list[float], direction: int, tolerance: float) -> dict[str, Any]:
    deltas = np.diff(np.asarray(values, dtype=np.float64))
    favorable = direction * deltas
    strict_favorable = int(np.sum(favorable > 1e-12))
    flat = int(np.sum(np.abs(deltas) <= 1e-12))
    adverse = int(len(deltas) - strict_favorable - flat)
    tail = np.asarray(values[-3:], dtype=np.float64)
    tail_slope = float(np.polyfit(np.arange(3), tail, 1)[0])
    return {
        "values": values,
        "desired_direction": "increase" if direction > 0 else "decrease",
        "strict_favorable_steps": strict_favorable,
        "flat_steps": flat,
        "adverse_steps": adverse,
        "favorable_or_flat_fraction": float(np.mean(favorable >= -1e-12)),
        "net_change": values[-1] - values[0],
        "tail_r5_r7_slope_per_round": tail_slope,
        "tail_r5_r7_range": float(np.ptp(tail)),
        "tail_plateau_tolerance_per_round": tolerance,
        "tail_plateau": abs(tail_slope) <= tolerance,
    }


def _binomial_two_sided(left: int, right: int) -> float:
    total = left + right
    if not total:
        return 1.0
    small = min(left, right)
    return min(
        1.0,
        2.0 * sum(math.comb(total, index) for index in range(small + 1))
        / (2**total),
    )


def _paired(winner: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    a = winner["rows"]["7"]
    b = baseline["rows"]["7"]
    identifiers_a = [(row["gamma"], row["episode"]) for row in a]
    identifiers_b = [(row["gamma"], row["episode"]) for row in b]
    if identifiers_a != identifiers_b:
        raise ValueError("r7 native evaluation rows are not paired")
    transitions = Counter(
        f"{old['status']}->{new['status']}" for new, old in zip(a, b)
    )
    success_gain = sum(
        old["status"] != "SUCCESS" and new["status"] == "SUCCESS"
        for new, old in zip(a, b)
    )
    success_loss = sum(
        old["status"] == "SUCCESS" and new["status"] != "SUCCESS"
        for new, old in zip(a, b)
    )
    collision_removed = sum(
        old["status"] == "COLLISION" and new["status"] != "COLLISION"
        for new, old in zip(a, b)
    )
    collision_added = sum(
        old["status"] != "COLLISION" and new["status"] == "COLLISION"
        for new, old in zip(a, b)
    )
    oob_removed = sum(
        old["status"] == "OOB" and new["status"] != "OOB"
        for new, old in zip(a, b)
    )
    oob_added = sum(
        old["status"] != "OOB" and new["status"] == "OOB"
        for new, old in zip(a, b)
    )
    return {
        "paired_rows": len(a),
        "status_transitions_baseline_to_winner": dict(sorted(transitions.items())),
        "success": {
            "gained": success_gain,
            "lost": success_loss,
            "exact_two_sided_discordant_p": _binomial_two_sided(
                success_gain, success_loss
            ),
        },
        "collision": {
            "removed": collision_removed,
            "added": collision_added,
            "exact_two_sided_discordant_p": _binomial_two_sided(
                collision_removed, collision_added
            ),
        },
        "OOB": {
            "removed": oob_removed,
            "added": oob_added,
            "exact_two_sided_discordant_p": _binomial_two_sided(
                oob_removed, oob_added
            ),
        },
    }


def _gamma_trend(per_gamma: dict[str, Any]) -> dict[str, Any]:
    gammas = sorted(per_gamma, key=float)
    values = {gamma: _metric(per_gamma[gamma]) for gamma in gammas}
    clearance = [values[gamma]["clearance_m"] for gamma in gammas]
    ttg = [values[gamma]["TtG_s"] for gamma in gammas]
    return {
        "gamma_order": gammas,
        "per_gamma": values,
        "minimum_gamma_SR": min(values[gamma]["SR"] for gamma in gammas),
        "SR_range": max(values[gamma]["SR"] for gamma in gammas)
        - min(values[gamma]["SR"] for gamma in gammas),
        "clearance_nonincreasing_pair_fraction": sum(
            right <= left + 1e-12 for left, right in zip(clearance, clearance[1:])
        ) / 3,
        "TtG_nonincreasing_pair_fraction": sum(
            right <= left + 1e-12 for left, right in zip(ttg, ttg[1:])
        ) / 3,
        "high_gamma_clearance_le_low_gamma": clearance[-1] <= clearance[0],
        "high_gamma_faster_or_equal": ttg[-1] <= ttg[0],
    }


def _load_arm(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = spec["raw_eval"]
    payload = json.loads(path.read_text())
    expected_rounds = {str(index) for index in range(8)}
    if payload.get("status") != "LAB_RAW_TEMPERATURE1_EVALUATION_COMPLETE":
        raise ValueError(f"incomplete raw evaluation: {path}")
    if set(payload.get("summary", {})) != expected_rounds:
        raise ValueError(f"missing r0-r7 summary: {path}")
    if set(payload.get("rows", {})) != expected_rounds:
        raise ValueError(f"missing r0-r7 rows: {path}")
    rounds: dict[str, Any] = {}
    for index in range(8):
        key = str(index)
        pooled = payload["summary"][key]["pooled"]
        rounds[key] = {
            **_metric(pooled),
            "routes": _route(pooled["route_counts"]),
        }
    trends = {
        "SR": _trend([rounds[str(i)]["SR"] for i in range(8)], 1, 0.01),
        "CR": _trend([rounds[str(i)]["CR"] for i in range(8)], -1, 0.01),
        "OOB_plus_timeout": _trend(
            [rounds[str(i)]["OOB_plus_timeout"] for i in range(8)], -1, 0.01
        ),
        "validity": _trend(
            [rounds[str(i)]["validity"] for i in range(8)], 1, 0.005
        ),
        "clearance_m": _trend(
            [rounds[str(i)]["clearance_m"] for i in range(1, 8)], 1, 0.002
        ),
        "TtG_s": _trend(
            [rounds[str(i)]["TtG_s"] for i in range(1, 8)], 1, 0.10
        ),
    }
    r7 = rounds["7"]
    exact_goal = {
        "SR_eq_1": r7["SR"] == 1.0,
        "CR_eq_0": r7["CR"] == 0.0,
        "OOB_plus_timeout_eq_0": r7["OOB_plus_timeout"] == 0.0,
        "validity_eq_1": r7["validity"] == 1.0,
        "coverage_4_of_4": r7["routes"]["coverage_modes"] == 4,
        "clearance_tail_plateau": trends["clearance_m"]["tail_plateau"],
        "TtG_tail_plateau": trends["TtG_s"]["tail_plateau"],
    }
    prior_final = {
        "SR_ge_0.95": r7["SR"] >= 0.95,
        "every_gamma_SR_ge_0.93": min(
            cell["SR"]
            for cell in _gamma_trend(payload["summary"]["7"]["per_gamma"])[
                "per_gamma"
            ].values()
        ) >= 0.93,
        "CR_le_0.03": r7["CR"] <= 0.03,
        "OOB_plus_timeout_le_0.03": r7["OOB_plus_timeout"] <= 0.03,
        "validity_ge_0.95": r7["validity"] >= 0.95,
        "coverage_4_of_4": r7["routes"]["coverage_modes"] == 4,
    }
    result = {
        "label": spec["label"],
        "speed_weight": spec["speed_weight"],
        "optimizer_steps_per_round": spec["optimizer_steps_per_round"],
        "raw_eval": str(path.resolve()),
        "raw_eval_sha256": _sha256(path),
        "evaluation": {
            "seed": 91000,
            "episodes_per_gamma": 40,
            "gammas": [0.1, 0.3, 0.5, 1.0],
            "native_raw_deployment": True,
            "evaluation_execution_shaping": False,
        },
        "rounds": rounds,
        "trends": trends,
        "gamma_r7": _gamma_trend(payload["summary"]["7"]["per_gamma"]),
        "exact_aspirational_goal_checks": exact_goal,
        "prior_final_threshold_checks": prior_final,
    }
    return payload, result


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Saved-r1 optimizer-exposure study: final native-raw result",
        "",
        "## Verdict",
        "",
        "**Winner: speed300 / 5,000 Adam steps.** It is the only arm that ends "
        "r7 with the best joint SR, CR, OOB+timeout, validity, full coverage, and "
        "route balance. It is a strong directional result, not paper-ready: CR=.150 "
        "and OOB=.0625 remain above the target, validity=.938 is below .95, and the "
        "formal r5-r7 clearance/TtG plateau checks do not pass.",
        "",
        "All numbers use the identical native fixed bank: seed91000, 40 episodes/γ "
        "(160/round), γ={.1,.3,.5,1.0}, with no verifier, speed penalty, or margin "
        "selector applied during deployment.",
        "",
        "## r7 comparison",
        "",
        "| Arm | SR | CR | OOB | TO | Validity | Clearance | TtG | Coverage | b/a/l/r | H₄ | min share | L1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    order = ("w300_steps5000", "w150_steps5000", "w300_steps2500")
    for name in order:
        arm = report["arms"][name]
        r = arm["rounds"]["7"]
        routes = r["routes"]
        counts = routes["counts"]
        lines.append(
            f"| {arm['label']} | {_fmt(r['SR'])} | {_fmt(r['CR'])} | "
            f"{_fmt(r['OOB'],4)} | {_fmt(r['timeout'],4)} | "
            f"{_fmt(r['validity'])} | {_fmt(r['clearance_m'])}m | "
            f"{_fmt(r['TtG_s'])}s | {routes['coverage_modes']}/4 | "
            f"{counts['below']}/{counts['above']}/{counts['left']}/{counts['right']} | "
            f"{_fmt(routes['normalized_entropy'])} | {_fmt(routes['minimum_share'])} | "
            f"{_fmt(routes['L1_from_uniform'])} |"
        )
    lines += [
        "",
        "W300/5000 versus the paired W300/2500 control isolates optimizer exposure: "
        "SR +.3125, CR −.2750, OOB −.0375, validity +.0741, and coverage 2/4→4/4. "
        "It is faster rather than slower (TtG 8.455→8.200s), so the gain is not a "
        "timeout/slowdown trick. Its clearance is lower (.160→.106m), showing that "
        "stronger fitting recovers more difficult successes rather than simply "
        "selecting only the widest-clearance survivors.",
        "",
        "## Full pooled curves",
    ]
    for name in order:
        arm = report["arms"][name]
        lines += [
            "",
            f"### {arm['label']}",
            "",
            "| r | SR | CR | OOB | TO | Validity | Clearance | TtG | Coverage | b/a/l/r | H₄ | min share | L1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for index in range(8):
            r = arm["rounds"][str(index)]
            routes = r["routes"]
            counts = routes["counts"]
            lines.append(
                f"| {index} | {_fmt(r['SR'])} | {_fmt(r['CR'])} | "
                f"{_fmt(r['OOB'],4)} | {_fmt(r['timeout'],4)} | "
                f"{_fmt(r['validity'])} | {_fmt(r['clearance_m'])}m | "
                f"{_fmt(r['TtG_s'])}s | {routes['coverage_modes']}/4 | "
                f"{counts['below']}/{counts['above']}/{counts['left']}/{counts['right']} | "
                f"{_fmt(routes['normalized_entropy'])} | "
                f"{_fmt(routes['minimum_share'])} | "
                f"{_fmt(routes['L1_from_uniform'])} |"
            )
    lines += ["", "## r7 gamma diagnostics", ""]
    for name in order:
        arm = report["arms"][name]
        gamma = arm["gamma_r7"]
        lines += [
            f"### {arm['label']}",
            "",
            "| γ | SR | CR | OOB | Validity | Clearance | TtG |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for value in gamma["gamma_order"]:
            cell = gamma["per_gamma"][value]
            lines.append(
                f"| {value} | {_fmt(cell['SR'])} | {_fmt(cell['CR'])} | "
                f"{_fmt(cell['OOB'])} | {_fmt(cell['validity'])} | "
                f"{_fmt(cell['clearance_m'])}m | {_fmt(cell['TtG_s'])}s |"
            )
        lines += [
            "",
            f"High-γ endpoint trend: clearance "
            f"{'PASS' if gamma['high_gamma_clearance_le_low_gamma'] else 'FAIL'}, "
            f"TtG {'PASS' if gamma['high_gamma_faster_or_equal'] else 'FAIL'}; "
            f"pairwise fractions={gamma['clearance_nonincreasing_pair_fraction']:.2f}/"
            f"{gamma['TtG_nonincreasing_pair_fraction']:.2f}; "
            f"minimum γ SR={gamma['minimum_gamma_SR']:.3f}.",
            "",
        ]
    lines += [
        "## Stability and plateau diagnosis",
        "",
        "Strict step counts below use all adjacent checkpoints. Plateau is a linear "
        "slope over r5-r7 with tolerances 0.002m/round for clearance and 0.10s/round "
        "for TtG.",
        "",
        "| Arm | SR ↑ | CR ↓ | Validity ↑ | Clearance ↑ (r1-r7) | TtG ↑ (r1-r7) | Clearance plateau | TtG plateau |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        arm = report["arms"][name]
        t = arm["trends"]
        lines.append(
            f"| {arm['label']} | {t['SR']['strict_favorable_steps']}/7 | "
            f"{t['CR']['strict_favorable_steps']}/7 | "
            f"{t['validity']['strict_favorable_steps']}/7 | "
            f"{t['clearance_m']['strict_favorable_steps']}/6 | "
            f"{t['TtG_s']['strict_favorable_steps']}/6 | "
            f"{'PASS' if t['clearance_m']['tail_plateau'] else 'FAIL'} "
            f"({t['clearance_m']['tail_r5_r7_slope_per_round']:+.4f}m/r) | "
            f"{'PASS' if t['TtG_s']['tail_plateau'] else 'FAIL'} "
            f"({t['TtG_s']['tail_r5_r7_slope_per_round']:+.3f}s/r) |"
        )
    winner = report["arms"]["w300_steps5000"]
    lines += [
        "",
        "W300/5000 is closest to the desired learning shape: SR improves on 6/7 "
        "transitions, timeout remains exactly zero, clearance reaches a practical "
        "near-flat .111→.106→.106m over r5-r7, and all four modes remain present. "
        "The predeclared strict plateau test narrowly rejects clearance "
        f"({winner['trends']['clearance_m']['tail_r5_r7_slope_per_round']:+.4f}m/r "
        "versus ±.002), while TtG still jiggles enough to reject plateau. OOB is "
        "not monotone and remains the main safety failure after collision.",
        "",
        "## Paired episode evidence at r7",
        "",
    ]
    for baseline in ("w300_steps2500", "w150_steps5000"):
        pair = report["paired_r7"][f"w300_steps5000_vs_{baseline}"]
        base_label = report["arms"][baseline]["label"]
        lines += [
            f"- Versus {base_label}: success gained/lost "
            f"{pair['success']['gained']}/{pair['success']['lost']} "
            f"(exact discordant p={pair['success']['exact_two_sided_discordant_p']:.3g}); "
            f"collision removed/added {pair['collision']['removed']}/"
            f"{pair['collision']['added']} "
            f"(p={pair['collision']['exact_two_sided_discordant_p']:.3g}); "
            f"OOB removed/added {pair['OOB']['removed']}/{pair['OOB']['added']} "
            f"(p={pair['OOB']['exact_two_sided_discordant_p']:.3g})."
        ]
    lines += [
        "",
        "These paired counts are descriptive fixed-bank evidence; the episode rows "
        "share scenes/seeds and should not be treated as independent paper-level "
        "replicates.",
        "",
        "## Scientific target check",
        "",
        "No arm is paper-ready. W300/5000 passes coverage4/4 and both high-γ endpoint "
        "trends, but misses SR≥.95, every-γ SR≥.93, CR≤.03, OOB+timeout≤.03, and "
        "validity≥.95. It is the unambiguous promotion candidate, not a final winner.",
        "",
        "## Visualization bundle",
        "",
        "- Interactive native-raw 3D + all-arm curves: "
        "`final_report/interactive/speed-exposure-r7-final.html`",
        "- Interactive provenance and input hashes: "
        "`final_report/interactive/speed-exposure-r7-final.provenance.json`",
        "- Rendered visual-QA snapshot: "
        "`final_report/static/speed-exposure-r7-final.png`",
        "- Round-1 acquisition/exposure diagnosis: `R1_ACQUISITION_DIAGNOSIS.md` "
        "and `R1_OPTIMIZER_EXPOSURE_RESULT.md`",
        "",
        "Exact raw-evaluation hashes and every computed value are stored in "
        "`FINAL_RESULT.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payloads: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    for name, spec in ARMS.items():
        payloads[name], arms[name] = _load_arm(spec)
    # r0 must be exactly common across all paired evaluations.
    r0 = [arms[name]["rounds"]["0"] for name in ARMS]
    if any(cell != r0[0] for cell in r0[1:]):
        raise ValueError("paired evaluations do not share the same r0 summary")
    report = {
        "status": "SAVED_R1_OPTIMIZER_EXPOSURE_FINAL_NATIVE_RAW_COMPARISON",
        "stage": str(STAGE.resolve()),
        "winner": "w300_steps5000",
        "winner_label": ARMS["w300_steps5000"]["label"],
        "paper_ready": False,
        "verdict": (
            "W300/5000 is the clear paired promotion winner but is not paper-ready; "
            "collision and OOB remain material and strict clearance/TtG plateau "
            "checks fail."
        ),
        "arms": arms,
        "paired_r7": {
            "w300_steps5000_vs_w300_steps2500": _paired(
                payloads["w300_steps5000"], payloads["w300_steps2500"]
            ),
            "w300_steps5000_vs_w150_steps5000": _paired(
                payloads["w300_steps5000"], payloads["w150_steps5000"]
            ),
        },
        "methods": {
            "route_entropy": "-sum(p*ln(p))/ln(4), successful routes only",
            "route_L1": "sum(abs(p_mode-0.25)), successful routes only",
            "plateau": "absolute linear slope over r5-r7 <= 0.002m/round clearance or 0.10s/round TtG",
            "paired_test": "two-sided exact binomial test on discordant fixed-bank episode outcomes",
        },
        "artifacts": {
            "interactive": str(
                STAGE / "final_report/interactive/speed-exposure-r7-final.html"
            ),
            "interactive_provenance": str(
                STAGE
                / "final_report/interactive/speed-exposure-r7-final.provenance.json"
            ),
            "visual_qa_snapshot": str(
                STAGE / "final_report/static/speed-exposure-r7-final.png"
            ),
            "r1_acquisition_diagnosis": str(STAGE / "R1_ACQUISITION_DIAGNOSIS.md"),
            "r1_optimizer_exposure": str(STAGE / "R1_OPTIMIZER_EXPOSURE_RESULT.md"),
        },
    }
    json_path = STAGE / "FINAL_RESULT.json"
    md_path = STAGE / "FINAL_RESULT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(report))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
