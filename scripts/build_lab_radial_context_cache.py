"""Build the immutable mmap dataset shared by all radial architecture arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pretrain_lab_reference_flow import build_radial_context_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--skip-archive-replay-validation",
        action="store_true",
        help="skip only the dynamics replay; provenance hashes remain required",
    )
    args = parser.parse_args()
    manifest = build_radial_context_cache(
        args.demo_dir,
        args.output,
        split_seed=args.split_seed,
        validate_archive=not args.skip_archive_replay_validation,
    )
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output.resolve()),
        "windows": manifest["stored_context_shape"][0],
        "source_archive_sha256": manifest[
            "source_archive_digest"
        ]["sha256"],
        "split_seed": manifest["split_provenance"]["split_seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
