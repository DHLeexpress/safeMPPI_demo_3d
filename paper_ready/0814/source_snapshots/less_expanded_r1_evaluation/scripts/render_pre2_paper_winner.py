#!/usr/bin/env python3
"""Render final-winner PRE2 raw curves and eight-sector coverage crowns."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
DEFAULT_RESULT = (
    ROOT / "results/stage1_single_ball_t128/0810_pre2_paper_closed_loop"
)
RENDERER = Path(
    "/Users/dhl/Documents/safe_flow_expansion/scripts/"
    "paper_b1_margin50_trends.py"
)
MODES = ("below", "above", "left", "right")
MODE_SHORT = {"below": "B", "above": "A", "left": "L", "right": "R"}
MODE_COLORS = {
    "below": "#2b6cb0",
    "above": "#dd6b20",
    "left": "#2f855a",
    "right": "#805ad5",
}


def _se(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _load_raw(path: Path) -> dict[int, tuple[dict, list[dict], float]]:
    payload = json.loads(path.read_text())
    cells = {}
    for round_text, summary in payload["summary"].items():
        round_i = int(round_text)
        cells[round_i] = (
            summary,
            payload["rows"][round_text],
            float(payload.get("sampling_temperature", 1.0)),
        )
    return cells


def _arm_cells(result: Path, stage: str, arm: str) -> dict[int, tuple[dict, list[dict], float]]:
    cells = {}
    baseline = result / "evaluations/baseline/pre2/r000/raw_eval.json"
    if baseline.is_file():
        cells.update(_load_raw(baseline))
    else:
        for path in sorted((result / "evaluations").glob("*/*/r*/raw_eval.json")):
            candidate = _load_raw(path)
            if 0 in candidate:
                cells[0] = candidate[0]
                break
    for path in sorted((result / "evaluations" / stage / arm).glob("r*/raw_eval.json")):
        cells.update(_load_raw(path))
    return cells


def _metric_rows(cells: dict[int, tuple[dict, list[dict], float]]) -> list[dict]:
    output = []
    for round_i, (summary, rows, temperature) in sorted(cells.items()):
        for gamma_text, cell in sorted(
            summary["per_gamma"].items(), key=lambda item: float(item[0])
        ):
            gamma = float(gamma_text)
            gamma_rows = [row for row in rows if float(row["gamma"]) == gamma]
            successes = [row for row in gamma_rows if row["status"] == "SUCCESS"]
            validity = [float(row["window_validity"]) for row in gamma_rows]
            clearance = [float(row["min_clearance_m"]) for row in successes]
            time = [float(row["time_to_goal_s"]) for row in successes]
            output.append({
                "round": round_i,
                "gamma": gamma,
                "m": len(gamma_rows),
                "temp": temperature,
                "CR": {"mean": float(cell["CR"]), "se": 0.0},
                "v_safe": {
                    "mean": float(cell["window_validity"]),
                    "se": _se(validity),
                },
                "clearance": {
                    "mean": float(np.mean(clearance)) if clearance else None,
                    "se": _se(clearance),
                },
                "time": {
                    "mean": float(np.mean(time)) if time else None,
                    "se": _se(time),
                },
            })
    return output


def _coverage_crown(
    cells: dict[int, tuple[dict, list[dict], float]], output: Path,
) -> None:
    rounds = sorted(cells)
    columns = min(4, max(1, len(rounds)))
    rows_n = math.ceil(len(rounds) / columns)
    figure, axes = plt.subplots(
        rows_n, columns, subplot_kw={"projection": "polar"},
        figsize=(4.2 * columns, 4.25 * rows_n), squeeze=False,
    )
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": shutil.which("latex") is not None,
    })
    sectors = [
        (mode, band)
        for mode in MODES
        for band in ("low", "high")
    ]
    all_values = []
    crown_rows = []
    for round_i in rounds:
        _, raw_rows, _ = cells[round_i]
        successes = [row for row in raw_rows if row["status"] == "SUCCESS"]
        counts = []
        for mode, band in sectors:
            count = sum(
                row.get("mode") == mode
                and ((float(row["gamma"]) <= 0.3) == (band == "low"))
                for row in successes
            )
            counts.append(count)
        values = [count / len(successes) if successes else 0.0 for count in counts]
        all_values.extend(values)
        crown_rows.append((round_i, raw_rows, successes, counts, values))
    radial_max = max(0.25, math.ceil(max(all_values, default=0.0) * 20.0) / 20.0)
    angles = np.arange(8) * (2.0 * math.pi / 8.0)
    width = 2.0 * math.pi / 8.0 * 0.84
    for axis, (round_i, raw_rows, successes, counts, values) in zip(
        axes.reshape(-1), crown_rows
    ):
        for index, ((mode, band), value) in enumerate(zip(sectors, values)):
            axis.bar(
                angles[index], value, width=width,
                color=MODE_COLORS[mode],
                alpha=0.48 if band == "low" else 0.95,
                edgecolor="white", linewidth=0.8,
            )
        axis.plot(
            np.linspace(0, 2.0 * math.pi, 361),
            np.full(361, 0.125), color="black", linestyle="--",
            linewidth=0.9, alpha=0.7,
        )
        axis.set_ylim(0.0, radial_max)
        axis.set_xticks(angles)
        axis.set_xticklabels([
            f"{MODE_SHORT[mode]}$_{{{'L' if band == 'low' else 'H'}}}$"
            for mode, band in sectors
        ], fontsize=11)
        axis.set_yticklabels([])
        sr = len(successes) / len(raw_rows) if raw_rows else 0.0
        axis.set_title(
            f"round {round_i}  |  SR {sr:.1%}\n"
            f"success n={len(successes)}",
            fontsize=13, fontweight="bold", pad=19,
        )
        axis.grid(alpha=0.25)
    for axis in axes.reshape(-1)[len(rounds):]:
        axis.set_visible(False)
    figure.suptitle(
        "C. 3D single sphere - coverage evolution",
        fontsize=22, fontweight="bold", x=0.02, ha="left",
    )
    figure.text(
        0.5, 0.012,
        "Each mode is split into low $\\gamma$ (0.1, 0.3; light) and "
        "high $\\gamma$ (0.5, 1.0; dark). Dashed crown = uniform 1/8 share.",
        ha="center", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps({
        "status": "PRE2_EIGHT_SECTOR_COVERAGE_CROWN_COMPLETE",
        "sector_definition": {
            "modes": list(MODES),
            "low_gamma": [0.1, 0.3],
            "high_gamma": [0.5, 1.0],
            "normalization": "share among all successful raw rollouts in the round",
            "uniform_reference_per_sector": 0.125,
        },
        "rounds": [
            {
                "round": round_i,
                "successes": len(successes),
                "episodes": len(raw_rows),
                "sector_counts": {
                    f"{mode}_{band}": count
                    for (mode, band), count in zip(sectors, counts)
                },
                "sector_shares": {
                    f"{mode}_{band}": value
                    for (mode, band), value in zip(sectors, values)
                },
            }
            for round_i, raw_rows, successes, counts, values in crown_rows
        ],
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True,
        help="STAGE/ARM, for example initial/quad_high_sr_trunk3",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir or args.result_root / "paper_curves"
    data_dir = outdir / "data"
    asset_dir = outdir / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for spec in args.arm:
        if "/" not in spec:
            raise ValueError("--arm must be STAGE/ARM")
        stage, arm = spec.split("/", 1)
        cells = _arm_cells(args.result_root, stage, arm)
        if not cells:
            raise FileNotFoundError(f"no full evaluation cells for {spec}")
        stem = f"{stage}_{arm}_raw_trends"
        jsonl = data_dir / f"{stem}.jsonl"
        metric_rows = _metric_rows(cells)
        jsonl.write_text("".join(json.dumps(row) + "\n" for row in metric_rows))
        crown = asset_dir / f"{stage}_{arm}_coverage_crown"
        _coverage_crown(cells, crown)
        if not args.no_render:
            subprocess.run([
                sys.executable, str(RENDERER),
                "--arm", f"{arm}={jsonl}",
                "--panel-heading", "C. 3D single sphere",
                "--outdir", str(asset_dir),
                "--stem", stem,
            ], check=True)
        rendered.append({
            "arm": spec,
            "rounds": sorted(cells),
            "jsonl": str(jsonl),
            "curve_stem": str(asset_dir / stem),
            "coverage_crown_stem": str(crown),
        })
    manifest = outdir / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "status": "PRE2_PAPER_WINNER_RENDER_COMPLETE",
        "panel_heading": "C. 3D single sphere",
        "renderer": str(RENDERER),
        "coverage_sector_definition": "mode x {low-gamma, high-gamma}",
        "arms": rendered,
    }, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
