#!/usr/bin/env python3
"""Stage and run the PRE2 sampling preflight on Helios GPU 1 or 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_mppi.helios_remote import (  # noqa: E402
    HELIOS_HOST,
    REMOTE_ARTIFACT_BASE,
    REMOTE_SOURCE_BASE,
    _check_remote_gpu,
    _collect_detached_log,
    _remote_detached_command,
    _rsync,
    _run_remote_job_detached,
    _sha256,
    _source_id,
    _source_stage_lock,
    _ssh,
    _ssh_master,
    _stage_pretrain,
    _stage_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "preflight"), required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helios-gpu", type=int, default=1)
    args, extra = parser.parse_known_args()
    if args.helios_gpu not in {1, 3}:
        parser.error("PRE2 preflight permits only Helios GPU 1 or 3")
    local_output = args.output.resolve()
    if local_output.exists() and (
        not local_output.is_dir() or any(local_output.iterdir())
    ):
        parser.error(f"refusing to overwrite nonempty output {local_output}")
    pretrain = args.pretrain_dir.resolve()
    task_config = args.task_config.resolve()
    try:
        relative_config = task_config.relative_to(ROOT)
    except ValueError:
        parser.error("--task-config must be inside the repository")
    source_id = _source_id(ROOT)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    run_id = (
        f"{local_output.name}-{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    remote_output = f"{REMOTE_ARTIFACT_BASE}/diagnostics/{run_id}"
    with _ssh_master(HELIOS_HOST) as ssh:
        _check_remote_gpu(ssh, args.helios_gpu)
        with _source_stage_lock(source_id):
            _stage_source(ssh, ROOT, remote_source)
            remote_pretrain, checkpoint_sha = _stage_pretrain(ssh, pretrain)
        remote_args = [
            "--phase", args.phase,
            "--pretrain-dir", remote_pretrain,
            "--task-config", f"{remote_source}/{relative_config}",
            "--output", remote_output,
            "--device", "cuda:0",
            *extra,
        ]
        _ssh(
            ssh,
            HELIOS_HOST,
            f"mkdir -p {shlex.quote(str(Path(remote_output).parent))}",
        )
        job_token = uuid.uuid4().hex
        command, status_file, log_file = _remote_detached_command(
            remote_source,
            remote_output,
            "preflight_multisphere_expansion_pre2.py",
            remote_args,
            args.helios_gpu,
            share_gpu=True,
            job_token=job_token,
        )
        plain_ssh = ["ssh"]
        try:
            _run_remote_job_detached(
                ssh,
                command,
                job_token,
                status_file,
                log_file,
            )
        except subprocess.CalledProcessError:
            _collect_detached_log(plain_ssh, log_file, remote_output)
            local_output.mkdir(parents=True, exist_ok=True)
            _rsync(
                plain_ssh,
                f"{HELIOS_HOST}:{remote_output}/",
                f"{local_output}/",
            )
            raise
        _collect_detached_log(plain_ssh, log_file, remote_output)
        local_output.mkdir(parents=True, exist_ok=True)
        _rsync(
            plain_ssh,
            f"{HELIOS_HOST}:{remote_output}/",
            f"{local_output}/",
        )
    expected = (
        "calibration.pt" if args.phase == "calibration" else "preflight.pt"
    )
    if not (local_output / expected).is_file() or not (
        local_output / "summary.json"
    ).is_file():
        raise RuntimeError("Helios preflight completed without final artifacts")
    (local_output / ".helios_remote.json").write_text(json.dumps({
        "status": "HELIOS_REMOTE_PRE2_PREFLIGHT_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "remote_output": remote_output,
        "remote_source": remote_source,
        "source_id": source_id,
        "pretrained_sha256": checkpoint_sha,
        "task_config_sha256": _sha256(task_config),
    }, indent=2) + "\n")
    print(f"[helios] remote output: {remote_output}")
    print(f"[helios] local output:  {local_output}")


if __name__ == "__main__":
    main()
