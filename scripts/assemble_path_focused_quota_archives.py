#!/usr/bin/env python3
"""Assemble complete disjoint-gamma quota archives into one demo archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.path_focused_archive import (  # noqa: E402
    assemble_path_focused_quota_archives,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = assemble_path_focused_quota_archives(
        args.inputs, args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "gammas": manifest["gammas"],
        "accepted_counts_by_gamma": manifest["accepted_counts_by_gamma"],
        "attempts": len(manifest["attempts"]),
        "runs": len(manifest["runs"]),
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
