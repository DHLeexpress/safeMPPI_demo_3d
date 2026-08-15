#!/usr/bin/env python3
"""Prepare or directly launch the faithful-M1 speed400 experiments on GPU1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))

from safe_mppi.helios_remote import (  # noqa: E402
    HELIOS_HOST,
    REMOTE_ARTIFACT_BASE,
    REMOTE_PYTHON,
    REMOTE_SOURCE_BASE,
    _source_id,
    _source_stage_lock,
    _ssh_master,
    _stage_source,
)


SOURCE_STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_hybrid_spooler_sweep"
STAGE = ROOT / "results/stage2_multi_sphere_n6/0812_pre2_speed400_faithful_gpu1"
SOURCE_SPEC = SOURCE_STAGE / "specs/02_1e29491aa246.json"
REFERENCE_FAITHFUL_R5 = (
    SOURCE_STAGE / "winner_bundle_dense_speed400"
    / "eval_m50_m1_faithful_r0_r5/raw_eval.json"
)
AUTHORITATIVE_R5_BUNDLE = SOURCE_STAGE / "winner_bundle_dense_speed400"
CONFIRM = "I_UNDERSTAND_THIS_RUNS_TWO_FAITHFUL_SPEED400_EXPERIMENTS_ON_GPU1"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.launch and args.confirm != CONFIRM:
        parser.error(f"--launch requires --confirm {CONFIRM}")
    if not SOURCE_SPEC.is_file():
        raise FileNotFoundError(SOURCE_SPEC)
    if not REFERENCE_FAITHFUL_R5.is_file():
        raise FileNotFoundError(REFERENCE_FAITHFUL_R5)
    if not (AUTHORITATIVE_R5_BUNDLE / "resume_state_latest.pt").is_file():
        raise FileNotFoundError(AUTHORITATIVE_R5_BUNDLE)
    if (STAGE / "LAUNCH.json").exists():
        raise RuntimeError("refusing duplicate launch")

    original = json.loads(SOURCE_SPEC.read_text())
    source_id = _source_id(ROOT)
    worker_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    remote_stage = f"{REMOTE_ARTIFACT_BASE}/direct_runs/{STAGE.name}"
    control = f"{remote_stage}/control"
    source_r5 = f"{remote_stage}/inputs/authoritative_r5_bundle"
    extension_output = f"{remote_stage}/experiments/speed400_exact_r5_to_r10"
    stability_output = f"{remote_stage}/experiments/speed400_r5_stability"
    r5_pretrain = f"{remote_stage}/inputs/r5_pretrain_package"
    reference_faithful_r5 = f"{remote_stage}/inputs/reference_faithful_r5_raw_eval.json"
    spec_payload = {
        "source_id": source_id,
        "scientific_source": original["remote_source"],
        "resume_source": (
            "/home/dohyun/.cache/safeMPPI_demo_3d/"
            "5c8a57779f16-505fd5d96acf"
        ),
        "source_r5": source_r5,
        "source_pretrain": original["remote_pretrain"],
        "task_config": original["remote_task_config"],
        "scene_bank": original["remote_scene_bank"],
        "reference_faithful_r5": reference_faithful_r5,
        "original_expansion_args": original["expansion_args"],
        "extension_output": extension_output,
        "stability_output": stability_output,
        "r5_pretrain": r5_pretrain,
        "control": control,
        "physical_gpu": 1,
    }
    spec_payload["config_hash"] = _canonical_sha256(spec_payload)
    local_spec = STAGE / "spec.json"
    _write(local_spec, spec_payload)

    with _ssh_master(HELIOS_HOST) as ssh:
        with _source_stage_lock(source_id):
            _stage_source(ssh, ROOT, worker_source)
    remote_spec = f"{control}/spec.json"
    subprocess.run([
        "ssh", HELIOS_HOST,
        f"mkdir -p {shlex.quote(control)} {shlex.quote(remote_stage + '/experiments')} "
        f"{shlex.quote(remote_stage + '/inputs')}",
    ], check=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh", str(local_spec),
        f"{HELIOS_HOST}:{remote_spec}",
    ], check=True)
    subprocess.run([
        "rsync", "-az", "-e", "ssh", str(REFERENCE_FAITHFUL_R5),
        f"{HELIOS_HOST}:{reference_faithful_r5}",
    ], check=True)
    subprocess.run([
        "rsync", "-a", "-e", "ssh", f"{AUTHORITATIVE_R5_BUNDLE}/",
        f"{HELIOS_HOST}:{source_r5}/",
    ], check=True)

    status = "PREPARED_NOT_LAUNCHED"
    pid = None
    log = f"{control}/run.log"
    if args.launch:
        # The original R5 arm is immutable.  A copy-on-write clone is the exact
        # resume input and keeps every checkpoint, event, archive, Adam state,
        # and RNG transaction intact.
        prepare = (
            f"test ! -e {shlex.quote(extension_output)} && "
            f"cp -a --reflink=auto {shlex.quote(source_r5)} {shlex.quote(extension_output)}"
        )
        subprocess.run(["ssh", HELIOS_HOST, prepare], check=True)
        worker = shlex.join([
            REMOTE_PYTHON,
            f"{worker_source}/scripts/run_multisphere_speed400_faithful_gpu1.py",
            "--spec", remote_spec,
        ])
        shell = (
            "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
            "export CUDA_VISIBLE_DEVICES=1; "
            "export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16; "
            f"nohup {worker} >> {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
        )
        pid = int(subprocess.check_output(
            ["ssh", HELIOS_HOST, f"bash -lc {shlex.quote(shell)}"], text=True,
        ).strip())
        status = "RUNNING"

    payload = {
        "schema_version": 1,
        "status": status,
        "created_unix": time.time(),
        "physical_gpu": 1,
        "remote_pid": pid,
        "remote_log": log,
        "remote_control": control,
        "spec": spec_payload,
        "contract": {
            "experiment_a": (
                "bit-identical authoritative dense-speed400 R5 clone, exact "
                "faithful-M1 revalidation, then exact-state R6-R10 continuation"
            ),
            "experiment_b": (
                "R5 model fresh initialization; new paired scenes, cumulative "
                "safe-success replay, LR 5e-6, first-layer scale .25, 20 effective "
                "repeats per cumulative row across two passes"
            ),
            "evaluation": (
                "faithful raw NFE16 M1 only; fixed unseen 50 scenes/gamma; "
                "200 attempts/checkpoint; no verifier/progress selection"
            ),
            "M8_used": False,
        },
    }
    _write(STAGE / ("LAUNCH.json" if args.launch else "PREFLIGHT.json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
