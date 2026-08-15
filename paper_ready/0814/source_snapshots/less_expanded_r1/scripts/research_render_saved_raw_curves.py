"""Re-render paper curves from a saved raw_eval.json without new rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_mppi.lab_flow_evaluation import _plot_curves, _plot_sr_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--x-label", default="Checkpoint stage")
    args = parser.parse_args()
    payload = json.loads(args.raw_eval.read_text())
    per_stage = {
        int(stage): rows for stage, rows in payload["rows"].items()
    }
    summaries = payload["summary"]
    gammas = sorted({
        float(row["gamma"])
        for rows in per_stage.values() for row in rows
    })
    output = args.raw_eval.parent
    _plot_curves(
        per_stage, summaries, gammas, output / "raw_curves.png",
        x_label=args.x_label,
    )
    _plot_sr_coverage(
        per_stage, gammas, output / "raw_sr_coverage.png",
        x_label=args.x_label,
    )


if __name__ == "__main__":
    main()
