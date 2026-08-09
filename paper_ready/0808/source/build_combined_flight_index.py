#!/usr/bin/env python3
"""Build the authoritative combined 0808 index without altering frozen inputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    sources = (
        bundle / "flight_references" / "FLIGHT_INDEX.csv",
        bundle / "safemppi" / "flight_references" / "FLIGHT_INDEX.csv",
    )
    rows: list[dict[str, str]] = []
    columns: list[str] | None = None
    for source in sources:
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if columns is None:
                columns = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != columns:
                raise ValueError(f"index columns differ in {source}")
            rows.extend(reader)
    if len(rows) != 42 or len({row["flight_id"] for row in rows}) != 42:
        raise ValueError("expected 42 unique 0808 flight references")
    target = bundle / "FLIGHT_INDEX_ALL.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
