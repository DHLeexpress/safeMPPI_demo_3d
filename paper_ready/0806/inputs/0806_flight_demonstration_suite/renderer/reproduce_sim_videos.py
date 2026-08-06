#!/usr/bin/env python3
"""Re-render all 16 frozen simulation videos and require byte equality.

Run this with the pinned renderer environment described in
``rendering_environment.json``.  Outputs must be written outside this frozen
suite.  The checked-in source NPZ/PT archives are the authoritative inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if "0806_flight_demonstration_suite" in args.output.resolve().parts:
        raise ValueError("reproduction output must be outside the frozen suite")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    renderer = args.suite / "renderer" / "p0806_render2.py"
    tasks = []
    for scene in sorted((args.suite / "scenes").iterdir()):
        scene_output = args.output / scene.name
        scene_output.mkdir()
        for policy in ("safemppi", "pretrained"):
            display_cap = "160" if policy == "safemppi" else "1"
            for run_npz in sorted((scene / policy).glob("run_*.npz")):
                events = scene / policy / f"events_{run_npz.stem}.pt"
                gamma_text = run_npz.stem.split("_g", 1)[1].split("_s", 1)[0]
                expected = scene / "videos" / f"{policy}_symmetric_g{gamma_text}.mp4"
                reproduced = scene_output / expected.name
                tasks.append(
                    (
                        scene.name,
                        policy,
                        gamma_text,
                        expected,
                        reproduced,
                        [
                            sys.executable,
                            str(renderer),
                            "--mode", policy,
                            "--repo", str(args.repo),
                            "--config", str(scene / "concrete_config.json"),
                            "--run-npz", str(run_npz),
                            "--events", str(events),
                            "--display-cap", display_cap,
                            "--output", str(reproduced),
                        ],
                    )
                )

    def reproduce(task: tuple) -> dict:
        scene_name, policy, gamma_text, expected, reproduced, command = task
        subprocess.run(command, check=True)
        expected_sha = sha256_file(expected)
        reproduced_sha = sha256_file(reproduced)
        if expected_sha != reproduced_sha:
            raise ValueError(
                f"byte mismatch for {scene_name}/{expected.name}: "
                f"{reproduced_sha} != {expected_sha}"
            )
        return {
            "scene": scene_name,
            "policy": policy,
            "gamma": float(gamma_text),
            "expected_sha256": expected_sha,
            "reproduced_sha256": reproduced_sha,
            "byte_identical": True,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        results = list(executor.map(reproduce, tasks))
    results.sort(key=lambda row: (row["scene"], row["policy"], row["gamma"]))
    if len(results) != 16:
        raise ValueError(f"expected 16 videos, reproduced {len(results)}")
    report = {"status": "PASS", "count": len(results), "videos": results}
    (args.output / "BYTE_IDENTICAL_REPRODUCTION.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"status": "PASS", "count": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
