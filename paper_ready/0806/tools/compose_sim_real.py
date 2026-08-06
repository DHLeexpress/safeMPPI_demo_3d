#!/usr/bin/env python3
"""Compose an immutable simulation MP4 with a derived real-flight MP4."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "0806_flight_demonstration_suite" in args.output.resolve().parts:
        raise ValueError("composite must be written to an operator run")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sim_duration = duration(args.simulation)
    real_duration = duration(args.real)
    target = max(sim_duration, real_duration)
    sim_pad = max(0.0, target - sim_duration)
    real_pad = max(0.0, target - real_duration)
    filter_graph = (
        f"[0:v]fps=30,setpts=PTS-STARTPTS,"
        f"scale=1440:720:force_original_aspect_ratio=decrease,"
        f"pad=1440:720:(ow-iw)/2:(oh-ih)/2:white,"
        f"tpad=stop_mode=clone:stop_duration={sim_pad:.6f}[sim];"
        f"[1:v]fps=30,setpts=PTS-STARTPTS,"
        f"scale=1440:720:force_original_aspect_ratio=decrease,"
        f"pad=1440:720:(ow-iw)/2:(oh-ih)/2:white,"
        f"tpad=stop_mode=clone:stop_duration={real_pad:.6f}[real];"
        "[sim][real]hstack=inputs=2[out]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(args.simulation),
            "-i", str(args.real), "-filter_complex", filter_graph,
            "-map", "[out]", "-t", f"{target:.6f}", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(args.output),
        ],
        check=True,
    )
    manifest = {
        "status": "P0806_SIM_REAL_COMPOSITE_COMPLETE",
        "semantics": (
            "derived side-by-side asset; the immutable simulation MP4 is "
            "referenced byte-for-byte and is not modified"
        ),
        "simulation": str(args.simulation.resolve()),
        "simulation_sha256": sha256_file(args.simulation),
        "real": str(args.real.resolve()),
        "real_sha256": sha256_file(args.real),
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "resolution": [2880, 720],
        "panels": {"left": "frozen simulation", "right": "measured flight"},
    }
    Path(str(args.output) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
