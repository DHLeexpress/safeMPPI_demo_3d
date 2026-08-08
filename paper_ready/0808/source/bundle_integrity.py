#!/usr/bin/env python3
"""Write or verify the strict SHA-256 inventory for the 0808 bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_in(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checksum_path = root / "SHA256SUMS"
    current = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in files_in(root)
    }
    if args.write:
        checksum_path.write_text("".join(
            f"{digest}  {relative}\n"
            for relative, digest in current.items()
        ))
        print(f"wrote {len(current)} checksums to {checksum_path}")
        return

    if not checksum_path.exists():
        raise SystemExit(f"missing {checksum_path}")
    expected = {}
    for line in checksum_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    if set(expected) != set(current):
        missing = sorted(set(expected) - set(current))
        extra = sorted(set(current) - set(expected))
        raise SystemExit(f"file-set mismatch: missing={missing}, extra={extra}")
    mismatched = [
        relative for relative in expected
        if expected[relative] != current[relative]
    ]
    if mismatched:
        raise SystemExit(f"checksum mismatch: {mismatched}")
    print(f"OK: {len(current)} files match SHA256SUMS")


if __name__ == "__main__":
    main()
