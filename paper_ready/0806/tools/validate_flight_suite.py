#!/usr/bin/env python3
"""Read-only structural and hash validation for the frozen 0806 suite."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hashes(suite: Path) -> int:
    count = 0
    for line in (suite / "FROZEN.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = suite / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"hash mismatch: {relative}: {actual} != {expected}")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    manifest = json.loads((suite / "suite_manifest.json").read_text())
    if manifest["trajectory_cells"] != 16:
        raise ValueError("suite manifest does not declare 16 trajectory cells")

    references = 0
    videos = 0
    source_runs = 0
    for scene in sorted((suite / "scenes").iterdir()):
        config_sha = sha256_file(scene / "concrete_config.json")
        declared = next(
            row for row in manifest["scenes"] if row["scene_id"] == scene.name
        )
        if config_sha != declared["config_sha256"]:
            raise ValueError(f"config hash mismatch for {scene.name}")
        for policy in ("safemppi", "pretrained"):
            runs = sorted((scene / policy).glob("run_*.npz"))
            events = sorted((scene / policy).glob("events_run_*.pt"))
            if len(runs) != 4 or len(events) != 4:
                raise ValueError(f"expected four runs/events in {scene}/{policy}")
            source_runs += len(runs)
        scene_references = sorted((scene / "flight_references").glob("*_100hz.npz"))
        if len(scene_references) != 8:
            raise ValueError(f"expected eight flight references in {scene}")
        for path in scene_references:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "time_s", "position_ref", "velocity_ref", "acceleration_ref",
                    "raw_controls_10hz", "executed_controls_10hz", "gamma", "seed",
                }
                if not required.issubset(data.files):
                    raise ValueError(f"flight reference fields missing in {path}")
                n = len(data["time_s"])
                if any(data[key].shape != (n, 3) for key in (
                    "position_ref", "velocity_ref", "acceleration_ref"
                )):
                    raise ValueError(f"100 Hz arrays are misaligned in {path}")
            references += 1
        scene_videos = sorted((scene / "videos").glob("*.mp4"))
        if len(scene_videos) != 8:
            raise ValueError(f"expected eight videos in {scene}")
        for video in scene_videos:
            sidecar = json.loads(Path(str(video) + ".json").read_text())
            if sha256_file(video) != sidecar["mp4_sha256"]:
                raise ValueError(f"video sidecar mismatch for {video}")
            videos += 1

    hashed_files = validate_hashes(suite)
    result = {
        "status": "PASS",
        "scenes": 2,
        "source_runs": source_runs,
        "flight_references": references,
        "videos": videos,
        "hashed_files": hashed_files,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
