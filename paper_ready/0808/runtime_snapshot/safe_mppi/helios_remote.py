"""Run existing expansion/evaluation entry points on a Helios GPU.

This module is transport only: it stages the current source and input artifacts,
rewrites local paths for the remote process, and copies the resulting artifact
directory back.  It does not change any scientific CLI option.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import uuid


HELIOS_HOST = "dohyun@helios.robotics.caltech.edu"
REMOTE_PYTHON = "/home/dohyun/miniforge3/envs/cfm_mppi/bin/python"
REMOTE_SOURCE_BASE = "/home/dohyun/.cache/safeMPPI_demo_3d"
REMOTE_ARTIFACT_BASE = "/data3/research1/safeMPPI_remote_cli"
REMOTE_METADATA = ".helios_remote.json"


def add_helios_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--helios",
        action="store_true",
        help=(
            "run this unchanged entry point on Helios, then copy artifacts "
            "back to the requested local output directory"
        ),
    )
    parser.add_argument(
        "--helios-gpu",
        type=int,
        default=1,
        help=(
            "physical Helios GPU index selected through CUDA_VISIBLE_DEVICES; "
            "the remote Python process sees it as cuda:0 (default: 1)"
        ),
    )
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--helios-share-gpu",
        dest="helios_share_gpu",
        action="store_true",
        default=True,
        help=(
            "start immediately even when the selected Helios GPU is busy "
            "(default)"
        ),
    )
    policy.add_argument(
        "--helios-queue-gpu",
        dest="helios_share_gpu",
        action="store_false",
        help=(
            "wait for exclusive access through the per-GPU queue instead of "
            "the default immediate sharing policy"
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_id(root: Path) -> str:
    digest = hashlib.sha256()
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "nogit"
    for directory in ("safe_mppi", "scripts", "configs"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return f"{head[:12]}-{digest.hexdigest()[:12]}"


def _without_option(argv: list[str], name: str, takes_value: bool) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            index += 2 if takes_value else 1
            continue
        if takes_value and token.startswith(f"{name}="):
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _replace_option(argv: list[str], name: str, value: str) -> list[str]:
    return _without_option(argv, name, takes_value=True) + [name, value]


def _remote_argv(argv: list[str]) -> list[str]:
    result = _without_option(argv, "--helios", takes_value=False)
    result = _without_option(result, "--helios-gpu", takes_value=True)
    result = _without_option(result, "--helios-share-gpu", takes_value=False)
    return _without_option(result, "--helios-queue-gpu", takes_value=False)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


@contextmanager
def _ssh_master(host: str):
    socket = Path(tempfile.gettempdir()) / (
        f"safe-mppi-{os_getpid()}-{uuid.uuid4().hex[:8]}"
    )
    base = ["ssh", "-S", str(socket)]
    _run([
        "ssh", "-o", "ControlMaster=yes", "-o", "ControlPersist=600",
        "-o", f"ControlPath={socket}", "-Nf", host,
    ])
    try:
        yield base
    finally:
        subprocess.run(base + ["-O", "exit", host], check=False)


def os_getpid() -> int:
    # Kept as a tiny seam for deterministic unit tests without importing os
    # throughout the transport helper.
    import os
    return os.getpid()


def _ssh(ssh: list[str], host: str, command: str) -> None:
    _run(ssh + [host, command])


def _ssh_output(ssh: list[str], host: str, command: str) -> str:
    return subprocess.check_output(
        ssh + [host, command], text=True,
    ).strip()


def _rsync(
    ssh: list[str], source: str, destination: str, *, delete: bool = False,
) -> None:
    command = ["rsync", "-az"]
    if delete:
        command.append("--delete")
    command += ["-e", shlex.join(ssh), source, destination]
    _run(command)


def _stage_source(ssh: list[str], root: Path, remote_source: str) -> None:
    _ssh(ssh, HELIOS_HOST, f"mkdir -p {shlex.quote(remote_source)}")
    excludes = [
        ".git", "results", "docs", "flow_deployment", "tmp", "outputs",
        "deploy_sim", "examples", "tests", "__pycache__", ".pytest_cache",
        ".DS_Store",
    ]
    command = ["rsync", "-az", "--delete"]
    for value in excludes:
        command += ["--exclude", value]
    command += [
        "-e", shlex.join(ssh), f"{root}/",
        f"{HELIOS_HOST}:{remote_source}/",
    ]
    _run(command)


def _stage_pretrain(
    ssh: list[str], pretrain_dir: Path,
) -> tuple[str, str]:
    checkpoint = pretrain_dir / "pretrained.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing pretrained checkpoint: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    remote = f"{REMOTE_ARTIFACT_BASE}/pretrains/{checkpoint_sha[:16]}"
    _ssh(ssh, HELIOS_HOST, f"mkdir -p {shlex.quote(remote)}")
    _rsync(ssh, f"{pretrain_dir}/", f"{HELIOS_HOST}:{remote}/")
    return remote, checkpoint_sha


def _assert_local_output_available(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite local output {path}")


def _check_remote_gpu(ssh: list[str], gpu: int) -> None:
    if gpu < 0:
        raise ValueError("--helios-gpu must be nonnegative")
    script = f"""
set -euo pipefail
gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
  awk -F, -v selected={gpu} '$1 + 0 == selected {{gsub(/ /, \"\", $2); print $2}}')
test -n \"$gpu_uuid\" || {{ echo 'unknown physical GPU index {gpu}' >&2; exit 2; }}
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | \
    sed 's/ //g' | grep -Fxq \"$gpu_uuid\"; then
  echo 'WARNING: sharing busy Helios physical GPU {gpu}' >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid \
    --format=csv,noheader | grep -F "$gpu_uuid" >&2 || true
fi
"""
    _ssh(ssh, HELIOS_HOST, f"bash -lc {shlex.quote(script)}")


def _assert_remote_gpu_exists(ssh: list[str], gpu: int) -> None:
    if gpu < 0:
        raise ValueError("--helios-gpu must be nonnegative")
    _ssh(
        ssh,
        HELIOS_HOST,
        f"nvidia-smi -i {gpu} --query-gpu=uuid --format=csv,noheader",
    )


def _remote_command(
    remote_source: str,
    script_name: str,
    argv: list[str],
    gpu: int,
    *,
    share_gpu: bool = False,
    job_token: str | None = None,
) -> str:
    job_token = job_token or uuid.uuid4().hex
    job_file = f"/tmp/safe-mppi-demo3d-job-{job_token}.pid"
    command = shlex.join([
        REMOTE_PYTHON, f"{remote_source}/scripts/{script_name}", *argv,
    ])
    gpu_gate = ""
    if not share_gpu:
        gpu_gate = (
            f"exec 9>/tmp/safe-mppi-demo3d-gpu-{gpu}.lock; "
            f"echo '[helios] queued for physical GPU {gpu}' >&2; "
            "flock 9; "
            "gpu_uuid=$(nvidia-smi --query-gpu=index,uuid "
            "--format=csv,noheader | "
            f"awk -F, -v selected={gpu} "
            "'$1 + 0 == selected {gsub(/ /, \"\", $2); print $2}'); "
            "while nvidia-smi --query-compute-apps=gpu_uuid "
            "--format=csv,noheader | sed 's/ //g' | "
            "grep -Fxq \"$gpu_uuid\"; do sleep 5; done; "
            f"echo '[helios] acquired physical GPU {gpu}' >&2; "
        )
    body = (
        "set -euo pipefail; "
        f"job_file={shlex.quote(job_file)}; "
        "child_pid=''; "
        "cleanup() { "
        "status=$?; trap - EXIT INT TERM HUP; "
        "if [ -n \"${child_pid:-}\" ] && kill -0 \"$child_pid\" 2>/dev/null; then "
        "kill -TERM -- \"-$child_pid\" 2>/dev/null || "
        "kill -TERM \"$child_pid\" 2>/dev/null || true; "
        "for _cleanup_wait in $(seq 1 20); do "
        "kill -0 \"$child_pid\" 2>/dev/null || break; sleep 0.5; done; "
        "kill -KILL -- \"-$child_pid\" 2>/dev/null || "
        "kill -KILL \"$child_pid\" 2>/dev/null || true; "
        "wait \"$child_pid\" 2>/dev/null || true; "
        "fi; "
        "rm -f \"$job_file\"; exit \"$status\"; "
        "}; "
        "trap 'exit 130' INT; trap 'exit 143' TERM HUP; trap cleanup EXIT; "
        "printf '%s\\n' \"$$\" > \"$job_file\"; "
        f"cd {shlex.quote(remote_source)}; "
        f"{gpu_gate}"
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
        f"setsid {command} & child_pid=$!; "
        "printf '%s %s\\n' \"$$\" \"$child_pid\" > \"$job_file\"; "
        "wait \"$child_pid\""
    )
    return f"bash -lc {shlex.quote(body)}"


def _cancel_remote_job(ssh: list[str], job_token: str) -> None:
    """Terminate only the queued/running process group owned by one launcher."""
    job_file = f"/tmp/safe-mppi-demo3d-job-{job_token}.pid"
    script = f"""
set +e
job_file={shlex.quote(job_file)}
if [ -r "$job_file" ]; then
  read -r shell_pid child_pid < "$job_file"
  if [[ "${{child_pid:-}}" =~ ^[0-9]+$ ]]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
  fi
  if [[ "${{shell_pid:-}}" =~ ^[0-9]+$ ]]; then
    kill -TERM "$shell_pid" 2>/dev/null || true
  fi
  for _cleanup_wait in $(seq 1 20); do
    child_alive=0
    shell_alive=0
    [[ "${{child_pid:-}}" =~ ^[0-9]+$ ]] && kill -0 "$child_pid" 2>/dev/null && child_alive=1
    [[ "${{shell_pid:-}}" =~ ^[0-9]+$ ]] && kill -0 "$shell_pid" 2>/dev/null && shell_alive=1
    (( child_alive == 0 && shell_alive == 0 )) && break
    sleep 0.5
  done
  if [[ "${{child_pid:-}}" =~ ^[0-9]+$ ]]; then
    kill -KILL -- "-$child_pid" 2>/dev/null || kill -KILL "$child_pid" 2>/dev/null || true
  fi
  if [[ "${{shell_pid:-}}" =~ ^[0-9]+$ ]]; then
    kill -KILL "$shell_pid" 2>/dev/null || true
  fi
  rm -f "$job_file"
fi
"""
    _ssh(ssh, HELIOS_HOST, f"bash -lc {shlex.quote(script)}")


def _run_remote_job(
    ssh: list[str], command: str, job_token: str,
) -> None:
    try:
        _ssh(ssh, HELIOS_HOST, command)
    except KeyboardInterrupt:
        print("\n[helios] cancelling queued/running remote job...", file=sys.stderr)
        try:
            _cancel_remote_job(ssh, job_token)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"[helios] remote cleanup warning: {error}", file=sys.stderr)
        raise


def _remote_files_are_complete(
    ssh: list[str], paths: list[str],
) -> bool:
    command = " && ".join(
        f"test -s {shlex.quote(path)}" for path in paths
    )
    try:
        _ssh(ssh, HELIOS_HOST, command)
    except subprocess.CalledProcessError:
        return False
    return True


def _write_metadata(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / REMOTE_METADATA).write_text(json.dumps(payload, indent=2) + "\n")


def run_expansion_on_helios(
    args: argparse.Namespace, argv: list[str], root: Path,
    script_name: str = "run_ball_expansion.py",
) -> int:
    local_output = args.output.resolve()
    _assert_local_output_available(local_output)
    pretrain_dir = args.pretrain_dir.resolve()
    source_id = _source_id(root)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    run_id = f"{local_output.name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    remote_output = f"{REMOTE_ARTIFACT_BASE}/runs/{run_id}"

    with _ssh_master(HELIOS_HOST) as ssh:
        if args.helios_share_gpu:
            _check_remote_gpu(ssh, args.helios_gpu)
        else:
            _assert_remote_gpu_exists(ssh, args.helios_gpu)
        _stage_source(ssh, root, remote_source)
        remote_pretrain, checkpoint_sha = _stage_pretrain(ssh, pretrain_dir)
        remote_args = _remote_argv(argv)
        remote_args = _replace_option(remote_args, "--pretrain-dir", remote_pretrain)
        remote_args = _replace_option(remote_args, "--output", remote_output)
        remote_args = _replace_option(remote_args, "--device", "cuda:0")
        if args.lab_task_config is not None:
            config = args.lab_task_config.resolve()
            try:
                relative = config.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "--helios currently requires --lab-task-config inside the repo"
                ) from error
            remote_args = _replace_option(
                remote_args, "--lab-task-config", f"{remote_source}/{relative}",
            )
        _ssh(
            ssh, HELIOS_HOST,
            f"mkdir -p {shlex.quote(str(Path(remote_output).parent))}",
        )
        job_token = uuid.uuid4().hex
        recovered_exit_code = None
        try:
            _run_remote_job(
                ssh,
                _remote_command(
                    remote_source, script_name, remote_args,
                    args.helios_gpu, share_gpu=args.helios_share_gpu,
                    job_token=job_token,
                ),
                job_token,
            )
        except subprocess.CalledProcessError as error:
            required = [
                f"{remote_output}/manifest.json",
                f"{remote_output}/checkpoint_{args.rounds:03d}.pt",
                f"{remote_output}/query_archive.pt",
                f"{remote_output}/metrics.jsonl",
            ]
            if args.event_log in {"full", "committed_success"}:
                required.append(f"{remote_output}/events.pt")
            if not _remote_files_are_complete(ssh, required):
                raise RuntimeError(
                    f"Helios expansion exited with status {error.returncode} "
                    f"and its completion artifacts are incomplete: {remote_output}"
                ) from None
            recovered_exit_code = int(error.returncode)
            print(
                "[helios] remote process reported a nonzero exit after writing "
                "all completion artifacts; recovering the completed run",
                file=sys.stderr,
            )
        local_output.mkdir(parents=True, exist_ok=True)
        _rsync(
            ssh, f"{HELIOS_HOST}:{remote_output}/", f"{local_output}/",
        )

    _write_metadata(local_output, {
        "status": "HELIOS_REMOTE_EXPANSION_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "gpu_policy": (
            "shared" if args.helios_share_gpu else "exclusive_queue"
        ),
        "remote_output": remote_output,
        "remote_source": remote_source,
        "source_id": source_id,
        "pretrained_sha256": checkpoint_sha,
        "recovered_remote_exit_code": recovered_exit_code,
    })
    print(f"[helios] remote output: {remote_output}")
    print(f"[helios] local output:  {local_output}")
    return 0


def run_evaluation_on_helios(
    args: argparse.Namespace, argv: list[str], root: Path,
    script_name: str = "evaluate_ball_expansion.py",
) -> int:
    local_expansion = args.expansion.resolve()
    metadata_path = local_expansion / REMOTE_METADATA
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} is missing; run expansion with --helios first"
        )
    expansion_metadata = json.loads(metadata_path.read_text())
    remote_expansion = expansion_metadata["remote_output"]
    local_eval = (
        args.evaluation_output.resolve()
        if args.evaluation_output is not None
        else local_expansion / "eval"
    )
    _assert_local_output_available(local_eval)
    source_id = _source_id(root)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    remote_eval = (
        f"{REMOTE_ARTIFACT_BASE}/evaluations/"
        f"{local_expansion.name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    with _ssh_master(HELIOS_HOST) as ssh:
        if args.helios_share_gpu:
            _check_remote_gpu(ssh, args.helios_gpu)
        else:
            _assert_remote_gpu_exists(ssh, args.helios_gpu)
        _stage_source(ssh, root, remote_source)
        remote_pretrain, checkpoint_sha = _stage_pretrain(
            ssh, args.pretrain_dir.resolve(),
        )
        remote_args = _remote_argv(argv)
        remote_args = _replace_option(remote_args, "--pretrain-dir", remote_pretrain)
        remote_args = _replace_option(remote_args, "--expansion", remote_expansion)
        remote_args = _replace_option(remote_args, "--evaluation-output", remote_eval)
        remote_args = _replace_option(remote_args, "--device", "cuda:0")
        _ssh(
            ssh, HELIOS_HOST,
            f"mkdir -p {shlex.quote(str(Path(remote_eval).parent))}",
        )
        job_token = uuid.uuid4().hex
        recovered_exit_code = None
        try:
            _run_remote_job(
                ssh,
                _remote_command(
                    remote_source, script_name, remote_args,
                    args.helios_gpu, share_gpu=args.helios_share_gpu,
                    job_token=job_token,
                ),
                job_token,
            )
        except subprocess.CalledProcessError as error:
            required = [f"{remote_eval}/raw_eval.json"]
            if not args.metrics_only:
                # --gallery-view decides which gallery is rendered, so require
                # the one this invocation asked for rather than assuming the
                # side view is always written.
                gallery = (
                    "raw_gallery_headon.png"
                    if getattr(args, "gallery_view", "side") == "head_on"
                    else "raw_gallery.png"
                )
                required.append(f"{remote_eval}/{gallery}")
            if not args.metrics_only and not args.screening_only:
                required.append(f"{remote_eval}/mechanism.mp4")
            if not _remote_files_are_complete(ssh, required):
                raise RuntimeError(
                    f"Helios evaluation exited with status {error.returncode} "
                    f"and its completion artifacts are incomplete: {remote_eval}"
                ) from None
            recovered_exit_code = int(error.returncode)
            print(
                "[helios] remote evaluator reported a nonzero exit after "
                "writing all completion artifacts; recovering the completed "
                "evaluation",
                file=sys.stderr,
            )
        legacy_eval = f"{remote_expansion}/eval"
        actual_remote_eval = _ssh_output(
            ssh,
            HELIOS_HOST,
            "bash -lc " + shlex.quote(
                f"if [ -d {shlex.quote(remote_eval)} ] && "
                f"[ -n \"$(find {shlex.quote(remote_eval)} -mindepth 1 "
                "-maxdepth 1 -print -quit)\" ]; then "
                f"printf %s {shlex.quote(remote_eval)}; "
                f"elif [ -d {shlex.quote(legacy_eval)} ] && "
                f"[ -n \"$(find {shlex.quote(legacy_eval)} -mindepth 1 "
                "-maxdepth 1 -print -quit)\" ]; then "
                f"printf %s {shlex.quote(legacy_eval)}; "
                "else echo 'remote evaluator produced no artifacts' >&2; exit 4; fi"
            ),
        )
        local_eval.mkdir(parents=True, exist_ok=True)
        _rsync(
            ssh, f"{HELIOS_HOST}:{actual_remote_eval}/", f"{local_eval}/",
        )

    _write_metadata(local_eval, {
        "status": "HELIOS_REMOTE_EVALUATION_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "gpu_policy": (
            "shared" if args.helios_share_gpu else "exclusive_queue"
        ),
        "remote_output": actual_remote_eval,
        "remote_expansion": remote_expansion,
        "remote_source": remote_source,
        "source_id": source_id,
        "pretrained_sha256": checkpoint_sha,
        "recovered_remote_exit_code": recovered_exit_code,
    })
    print(f"[helios] remote evaluation: {remote_eval}")
    print(f"[helios] local evaluation:  {local_eval}")
    return 0
