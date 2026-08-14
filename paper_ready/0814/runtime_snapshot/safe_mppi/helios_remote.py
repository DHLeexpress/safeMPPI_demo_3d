"""Run existing expansion/evaluation entry points on a Helios GPU.

This module is transport only: it stages the current source and input artifacts,
rewrites local paths for the remote process, and copies the resulting artifact
directory back.  It does not change any scientific CLI option.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Callable
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
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--helios-detached",
        dest="helios_detached",
        action="store_true",
        default=True,
        help=(
            "keep a long expansion alive if the controlling SSH channel "
            "drops; remote stdout is written to helios.log and completion "
            "is polled over reconnectable short SSH sessions (default)"
        ),
    )
    transport.add_argument(
        "--helios-attached",
        dest="helios_detached",
        action="store_false",
        help="use the legacy SSH-lifetime-bound streaming transport",
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
    parser.add_argument(
        "--helios-stage-local-expansion",
        action="store_true",
        help=(
            "for evaluation of a locally produced offline checkpoint package, "
            "stage that expansion directory directly instead of requiring "
            ".helios_remote.json"
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
    result = _without_option(result, "--helios-detached", takes_value=False)
    result = _without_option(result, "--helios-attached", takes_value=False)
    result = _without_option(result, "--helios-share-gpu", takes_value=False)
    result = _without_option(result, "--helios-queue-gpu", takes_value=False)
    return _without_option(
        result, "--helios-stage-local-expansion", takes_value=False,
    )


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
        if socket.exists():
            subprocess.run(
                base + ["-O", "exit", host], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def os_getpid() -> int:
    # Kept as a tiny seam for deterministic unit tests without importing os
    # throughout the transport helper.
    import os
    return os.getpid()


def _ssh(ssh: list[str], host: str, command: str) -> None:
    _run(ssh + [host, command])


@contextmanager
def _source_stage_lock(source_id: str):
    """Serialize content-addressed rsync staging across local launchers."""
    lock = Path(tempfile.gettempdir()) / f"safe-mppi-stage-{source_id}.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _stage_expansion_resume(
    ssh: list[str], resume_dir: Path, remote_output: str,
) -> None:
    state = resume_dir / "resume_state_latest.pt"
    metadata = resume_dir / "resume_state.json"
    if not state.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            "--resume-from requires resume_state_latest.pt and "
            f"resume_state.json in {resume_dir}"
        )
    _ssh(ssh, HELIOS_HOST, f"mkdir -p {shlex.quote(remote_output)}")
    _rsync(ssh, f"{resume_dir}/", f"{HELIOS_HOST}:{remote_output}/")


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


def _remote_detached_command(
    remote_source: str,
    remote_output: str,
    script_name: str,
    argv: list[str],
    gpu: int,
    *,
    share_gpu: bool = False,
    job_token: str | None = None,
) -> tuple[str, str, str]:
    """Build a reconnect-safe launcher plus its status and log paths."""
    job_token = job_token or uuid.uuid4().hex
    job_file = f"/tmp/safe-mppi-demo3d-job-{job_token}.pid"
    status_file = f"/tmp/safe-mppi-demo3d-job-{job_token}.status"
    # Keep the transport log outside the scientific output until the child
    # exits: both entry points intentionally reject nonempty output dirs.
    log_file = f"{remote_output}.helios.log"
    command = shlex.join([
        REMOTE_PYTHON, f"{remote_source}/scripts/{script_name}", *argv,
    ])
    gpu_gate = ""
    if not share_gpu:
        gpu_gate = (
            f"exec 9>/tmp/safe-mppi-demo3d-gpu-{gpu}.lock; "
            f"echo '[helios] queued for physical GPU {gpu}'; "
            "flock 9; "
            "gpu_uuid=$(nvidia-smi --query-gpu=index,uuid "
            "--format=csv,noheader | "
            f"awk -F, -v selected={gpu} "
            "'$1 + 0 == selected {gsub(/ /, \"\", $2); print $2}'); "
            "while nvidia-smi --query-compute-apps=gpu_uuid "
            "--format=csv,noheader | sed 's/ //g' | "
            "grep -Fxq \"$gpu_uuid\"; do sleep 5; done; "
            f"echo '[helios] acquired physical GPU {gpu}'; "
        )
    worker = (
        "set -uo pipefail; status=125; "
        f"job_file={shlex.quote(job_file)}; "
        f"status_file={shlex.quote(status_file)}; "
        "finish() { "
        "tmp_status=\"${status_file}.$$\"; "
        "printf '%s\\n' \"$status\" > \"$tmp_status\"; "
        "mv \"$tmp_status\" \"$status_file\"; "
        "rm -f \"$job_file\"; "
        "}; trap finish EXIT; "
        "printf '%s\\n' \"$$\" > \"$job_file\"; "
        f"cd {shlex.quote(remote_source)}; "
        f"{gpu_gate}"
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
        f"setsid {command} & child_pid=$!; "
        "printf '%s %s\\n' \"$$\" \"$child_pid\" > \"$job_file\"; "
        "wait \"$child_pid\"; status=$?; exit 0"
    )
    launcher = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(str(Path(remote_output).parent))}; "
        f"rm -f {shlex.quote(status_file)}; "
        f"nohup bash -lc {shlex.quote(worker)} "
        f"> {shlex.quote(log_file)} 2>&1 < /dev/null & "
        "printf '%s\\n' \"$!\""
    )
    return f"bash -lc {shlex.quote(launcher)}", status_file, log_file


def _run_remote_job_detached(
    launch_ssh: list[str],
    command: str,
    job_token: str,
    status_file: str,
    log_file: str,
    *,
    poll_seconds: float = 15.0,
    max_connection_failures: int = 120,
    progress_sync: Callable[[], None] | None = None,
    progress_sync_seconds: float = 240.0,
) -> None:
    """Launch once, then poll over fresh SSH connections until completion."""
    _ssh(launch_ssh, HELIOS_HOST, command)
    plain_ssh = ["ssh", "-o", "ConnectTimeout=15"]
    connection_failures = 0
    previous_line = None
    last_progress_sync = 0.0
    poll = (
        f"if test -s {shlex.quote(status_file)}; then "
        f"printf 'DONE '; cat {shlex.quote(status_file)}; "
        "else printf 'RUNNING '; "
        f"tail -n 1 {shlex.quote(log_file)} 2>/dev/null || true; fi"
    )
    try:
        while True:
            try:
                output = _ssh_output(plain_ssh, HELIOS_HOST, poll)
                connection_failures = 0
            except (OSError, subprocess.CalledProcessError) as error:
                connection_failures += 1
                if connection_failures >= max_connection_failures:
                    raise RuntimeError(
                        "lost contact with detached Helios job after "
                        f"{connection_failures} consecutive polls"
                    ) from error
                time.sleep(poll_seconds)
                continue
            if output.startswith("DONE "):
                status = int(output.split(maxsplit=1)[1])
                if status:
                    raise subprocess.CalledProcessError(status, command)
                return
            if (
                progress_sync is not None
                and time.monotonic() - last_progress_sync
                >= progress_sync_seconds
            ):
                try:
                    progress_sync()
                except (OSError, subprocess.CalledProcessError) as error:
                    print(
                        f"[helios] committed-state mirror warning: {error}",
                        file=sys.stderr, flush=True,
                    )
                last_progress_sync = time.monotonic()
            if output != previous_line:
                print(f"[helios-detached] {output}", flush=True)
                previous_line = output
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n[helios] cancelling detached remote job...", file=sys.stderr)
        try:
            _cancel_remote_job(plain_ssh, job_token)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"[helios] remote cleanup warning: {error}", file=sys.stderr)
        raise


def _collect_detached_log(
    ssh: list[str], log_file: str, remote_output: str,
) -> None:
    script = (
        f"mkdir -p {shlex.quote(remote_output)}; "
        f"if test -f {shlex.quote(log_file)}; then "
        f"mv {shlex.quote(log_file)} "
        f"{shlex.quote(f'{remote_output}/helios.log')}; fi"
    )
    _ssh(ssh, HELIOS_HOST, script)


def _sync_committed_expansion_progress(
    ssh: list[str], remote_output: str, local_output: Path,
) -> None:
    """Mirror only atomically published round-boundary recovery artifacts."""
    local_output.mkdir(parents=True, exist_ok=True)
    include = [
        "checkpoint_*.pt", "query_archive_round_*.pt", "metrics.jsonl",
        "resume_state_latest.pt", "resume_state.json", "events_round_*.pt",
        "fa_alloc_log.json", "first_action_stats.json", "manifest.json",
        "manifest_before_resume_round_*.json", "RESUME_IN_PROGRESS.json",
        "query_archive.pt", "gp_evidence.pt", "events.pt", "FAILED.json",
        "task_config_resolved.json", "RECIPE.sh", "command.sh",
    ]
    command = ["rsync", "-az"]
    for pattern in include:
        command.extend(["--include", pattern])
    command.extend([
        "--exclude", ".*.tmp", "--exclude", "*", "-e", shlex.join(ssh),
        f"{HELIOS_HOST}:{remote_output}/", f"{local_output}/",
    ])
    _run(command)


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


def _required_metrics_evaluation_artifacts(
    remote_eval: str, script_name: str,
) -> list[str]:
    required = [f"{remote_eval}/raw_eval.json"]
    if script_name == "evaluate_sphere_clutter_expansion.py":
        required.extend([
            f"{remote_eval}/fixed_scene_raw_eval.json",
            f"{remote_eval}/fixed_scene_config.json",
        ])
    return required


def _write_metadata(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / REMOTE_METADATA).write_text(json.dumps(payload, indent=2) + "\n")


def _sync_collection_progress(
    ssh: list[str], remote_output: str, local_output: Path,
) -> None:
    """Mirror atomically committed mirrored-pair collection artifacts."""
    local_output.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync", "-az",
        "--include", "attempt_shards/",
        "--include", "attempt_shards/*.npz",
        "--include", "pair_quota_contract.json",
        "--include", "resolved_config.json",
        "--include", "progress.json",
        "--include", "manifest.json",
        "--include", "metrics.json",
        "--include", "FAILED_collection.json",
        "--exclude", ".*.tmp",
        "--exclude", "*",
        "-e", shlex.join(ssh),
        f"{HELIOS_HOST}:{remote_output}/", f"{local_output}/",
    ]
    _run(command)


def run_collection_on_helios(
    args: argparse.Namespace,
    argv: list[str],
    root: Path,
    *,
    script_name: str = "collect_mirrored_pair_success_quota.py",
) -> int:
    """Stage and run a standalone output-producing collection entry point."""
    local_output = args.output.resolve()
    _assert_local_output_available(local_output)
    config = args.config.resolve()
    try:
        config_relative = config.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "--helios collection requires --config inside the repository"
        ) from error

    source_id = _source_id(root)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    run_id = (
        f"{local_output.name}-{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    remote_output = f"{REMOTE_ARTIFACT_BASE}/collections/{run_id}"
    with _ssh_master(HELIOS_HOST) as ssh:
        if args.helios_share_gpu:
            _check_remote_gpu(ssh, args.helios_gpu)
        else:
            _assert_remote_gpu_exists(ssh, args.helios_gpu)
        with _source_stage_lock(source_id):
            _stage_source(ssh, root, remote_source)
        remote_args = _remote_argv(argv)
        remote_args = _replace_option(
            remote_args, "--config", f"{remote_source}/{config_relative}",
        )
        remote_args = _replace_option(remote_args, "--output", remote_output)
        remote_args = _replace_option(remote_args, "--device", "cuda:0")
        _ssh(
            ssh,
            HELIOS_HOST,
            f"mkdir -p {shlex.quote(str(Path(remote_output).parent))}",
        )
        job_token = uuid.uuid4().hex
        completion_ssh = ssh
        try:
            if args.helios_detached:
                command, status_file, log_file = _remote_detached_command(
                    remote_source,
                    remote_output,
                    script_name,
                    remote_args,
                    args.helios_gpu,
                    share_gpu=args.helios_share_gpu,
                    job_token=job_token,
                )
                completion_ssh = ["ssh"]
                try:
                    _run_remote_job_detached(
                        ssh,
                        command,
                        job_token,
                        status_file,
                        log_file,
                        progress_sync=lambda: _sync_collection_progress(
                            ["ssh"], remote_output, local_output,
                        ),
                    )
                finally:
                    _collect_detached_log(
                        completion_ssh, log_file, remote_output,
                    )
            else:
                _run_remote_job(
                    ssh,
                    _remote_command(
                        remote_source,
                        script_name,
                        remote_args,
                        args.helios_gpu,
                        share_gpu=args.helios_share_gpu,
                        job_token=job_token,
                    ),
                    job_token,
                )
        except subprocess.CalledProcessError as error:
            local_output.mkdir(parents=True, exist_ok=True)
            _rsync(
                completion_ssh,
                f"{HELIOS_HOST}:{remote_output}/",
                f"{local_output}/",
            )
            raise RuntimeError(
                f"Helios collection exited with status {error.returncode}; "
                f"failure artifacts were mirrored from {remote_output}"
            ) from None
        local_output.mkdir(parents=True, exist_ok=True)
        _rsync(
            completion_ssh,
            f"{HELIOS_HOST}:{remote_output}/",
            f"{local_output}/",
        )

    if not (local_output / "manifest.json").is_file():
        raise RuntimeError("Helios collection completed without manifest.json")
    _write_metadata(local_output, {
        "status": "HELIOS_REMOTE_COLLECTION_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "gpu_policy": "shared" if args.helios_share_gpu else "exclusive_queue",
        "detached": bool(args.helios_detached),
        "remote_output": remote_output,
        "remote_source": remote_source,
        "source_id": source_id,
    })
    print(f"[helios] remote collection: {remote_output}")
    print(f"[helios] local collection:  {local_output}")
    return 0


def run_pretraining_on_helios(
    args: argparse.Namespace,
    argv: list[str],
    root: Path,
    *,
    script_name: str = "pretrain_lab_reference_flow.py",
) -> int:
    """Stage a declared demo archive and run pretraining on Helios."""
    local_output = args.output.resolve()
    _assert_local_output_available(local_output)
    demo_dir = args.demo_dir.resolve()
    for required in (demo_dir / "manifest.json", demo_dir / "resolved_config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    demo_id = _sha256(demo_dir / "manifest.json")[:16]
    remote_demo = f"{REMOTE_ARTIFACT_BASE}/demos/{demo_dir.name}-{demo_id}"

    source_id = _source_id(root)
    remote_source = f"{REMOTE_SOURCE_BASE}/{source_id}"
    run_id = (
        f"{local_output.name}-{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    remote_output = f"{REMOTE_ARTIFACT_BASE}/pretraining_runs/{run_id}"
    with _ssh_master(HELIOS_HOST) as ssh:
        if args.helios_share_gpu:
            _check_remote_gpu(ssh, args.helios_gpu)
        else:
            _assert_remote_gpu_exists(ssh, args.helios_gpu)
        with _source_stage_lock(source_id):
            _stage_source(ssh, root, remote_source)
            _ssh(ssh, HELIOS_HOST, f"mkdir -p {shlex.quote(remote_demo)}")
            _rsync(ssh, f"{demo_dir}/", f"{HELIOS_HOST}:{remote_demo}/")

        remote_args = _remote_argv(argv)
        remote_args = _replace_option(remote_args, "--demo-dir", remote_demo)
        remote_args = _replace_option(remote_args, "--output", remote_output)
        remote_args = _replace_option(remote_args, "--device", "cuda:0")
        recovery_checkpoint = getattr(args, "recover_best_checkpoint", None)
        if recovery_checkpoint is not None:
            resolved_recovery = recovery_checkpoint.resolve()
            if not resolved_recovery.is_file():
                raise FileNotFoundError(resolved_recovery)
            recovery_id = _sha256(resolved_recovery)[:16]
            remote_recovery_dir = (
                f"{REMOTE_ARTIFACT_BASE}/pretraining_recovery/{recovery_id}"
            )
            remote_recovery = (
                f"{remote_recovery_dir}/{resolved_recovery.name}"
            )
            _ssh(
                ssh, HELIOS_HOST,
                f"mkdir -p {shlex.quote(remote_recovery_dir)}",
            )
            _rsync(
                ssh, str(resolved_recovery),
                f"{HELIOS_HOST}:{remote_recovery}",
            )
            remote_args = _replace_option(
                remote_args,
                "--recover-best-checkpoint",
                remote_recovery,
            )
        ood_config = getattr(args, "ood_config", None)
        if ood_config is not None:
            resolved_ood = ood_config.resolve()
            try:
                relative = resolved_ood.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "--helios pretraining requires --ood-config inside the repository"
                ) from error
            remote_args = _replace_option(
                remote_args, "--ood-config", f"{remote_source}/{relative}",
            )
        _ssh(
            ssh,
            HELIOS_HOST,
            f"mkdir -p {shlex.quote(str(Path(remote_output).parent))}",
        )
        job_token = uuid.uuid4().hex
        completion_ssh = ssh
        try:
            if args.helios_detached:
                command, status_file, log_file = _remote_detached_command(
                    remote_source,
                    remote_output,
                    script_name,
                    remote_args,
                    args.helios_gpu,
                    share_gpu=args.helios_share_gpu,
                    job_token=job_token,
                )
                completion_ssh = ["ssh"]
                try:
                    _run_remote_job_detached(
                        ssh,
                        command,
                        job_token,
                        status_file,
                        log_file,
                    )
                finally:
                    _collect_detached_log(
                        completion_ssh, log_file, remote_output,
                    )
            else:
                _run_remote_job(
                    ssh,
                    _remote_command(
                        remote_source,
                        script_name,
                        remote_args,
                        args.helios_gpu,
                        share_gpu=args.helios_share_gpu,
                        job_token=job_token,
                    ),
                    job_token,
                )
        except subprocess.CalledProcessError as error:
            local_output.mkdir(parents=True, exist_ok=True)
            _rsync(
                completion_ssh,
                f"{HELIOS_HOST}:{remote_output}/",
                f"{local_output}/",
            )
            raise RuntimeError(
                f"Helios pretraining exited with status {error.returncode}; "
                f"artifacts were mirrored from {remote_output}"
            ) from None
        local_output.mkdir(parents=True, exist_ok=True)
        _rsync(
            completion_ssh,
            f"{HELIOS_HOST}:{remote_output}/",
            f"{local_output}/",
        )

    required_outputs = (
        local_output / "pretrained.pt",
        local_output / "pretrain_manifest.json",
        local_output / "training_history.json",
    )
    if any(not path.is_file() for path in required_outputs):
        raise RuntimeError("Helios pretraining completed without final artifacts")
    _write_metadata(local_output, {
        "status": "HELIOS_REMOTE_PRETRAINING_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "gpu_policy": "shared" if args.helios_share_gpu else "exclusive_queue",
        "detached": bool(args.helios_detached),
        "remote_output": remote_output,
        "remote_demo": remote_demo,
        "remote_source": remote_source,
        "source_id": source_id,
    })
    print(f"[helios] remote pretraining: {remote_output}")
    print(f"[helios] local pretraining:  {local_output}")
    return 0


def run_expansion_on_helios(
    args: argparse.Namespace, argv: list[str], root: Path,
    script_name: str = "run_ball_expansion.py",
) -> int:
    local_output = args.output.resolve()
    resume_dir = (
        args.resume_from.resolve()
        if getattr(args, "resume_from", None) is not None else None
    )
    if resume_dir is None:
        _assert_local_output_available(local_output)
    elif resume_dir != local_output:
        raise ValueError("--resume-from must equal --output for Helios resume")
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
        with _source_stage_lock(source_id):
            _stage_source(ssh, root, remote_source)
            remote_pretrain, checkpoint_sha = _stage_pretrain(
                ssh, pretrain_dir,
            )
            bootstrap_query_archive = getattr(
                args, "bootstrap_query_archive", None,
            )
            remote_bootstrap_query_archive = None
            if bootstrap_query_archive is not None:
                bootstrap_query_archive = bootstrap_query_archive.resolve()
                if (
                    not bootstrap_query_archive.is_file()
                    or bootstrap_query_archive.suffix != ".pt"
                ):
                    raise FileNotFoundError(bootstrap_query_archive)
                bootstrap_sha = _sha256(bootstrap_query_archive)
                remote_bootstrap_dir = (
                    f"{REMOTE_ARTIFACT_BASE}/expansion_bootstraps/"
                    f"{bootstrap_sha[:16]}"
                )
                remote_bootstrap_query_archive = (
                    f"{remote_bootstrap_dir}/{bootstrap_query_archive.name}"
                )
                _ssh(
                    ssh, HELIOS_HOST,
                    f"mkdir -p {shlex.quote(remote_bootstrap_dir)}",
                )
                _rsync(
                    ssh, str(bootstrap_query_archive),
                    f"{HELIOS_HOST}:{remote_bootstrap_query_archive}",
                )
        remote_args = _remote_argv(argv)
        remote_args = _replace_option(remote_args, "--pretrain-dir", remote_pretrain)
        remote_args = _replace_option(remote_args, "--output", remote_output)
        remote_args = _replace_option(remote_args, "--device", "cuda:0")
        if remote_bootstrap_query_archive is not None:
            remote_args = _replace_option(
                remote_args,
                "--bootstrap-query-archive",
                remote_bootstrap_query_archive,
            )
        if resume_dir is not None:
            _stage_expansion_resume(ssh, resume_dir, remote_output)
            remote_args = _replace_option(
                remote_args, "--resume-from", remote_output,
            )
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
        completion_ssh = ssh
        try:
            if args.helios_detached:
                command, status_file, log_file = _remote_detached_command(
                    remote_source, remote_output, script_name, remote_args,
                    args.helios_gpu, share_gpu=args.helios_share_gpu,
                    job_token=job_token,
                )
                # The staging ControlMaster may expire during a long detached
                # run.  Completion checks and rsync therefore reconnect.
                completion_ssh = ["ssh"]
                try:
                    _run_remote_job_detached(
                        ssh, command, job_token, status_file, log_file,
                        progress_sync=lambda: _sync_committed_expansion_progress(
                            ["ssh"], remote_output, local_output,
                        ),
                    )
                finally:
                    _collect_detached_log(
                        completion_ssh, log_file, remote_output,
                    )
            else:
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
            if not _remote_files_are_complete(completion_ssh, required):
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
            completion_ssh,
            f"{HELIOS_HOST}:{remote_output}/", f"{local_output}/",
        )

    _write_metadata(local_output, {
        "status": "HELIOS_REMOTE_EXPANSION_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "gpu_policy": (
            "shared" if args.helios_share_gpu else "exclusive_queue"
        ),
        "detached": bool(args.helios_detached),
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
    stage_local_expansion = bool(
        getattr(args, "helios_stage_local_expansion", False)
    )
    if not metadata_path.is_file() and not stage_local_expansion:
        raise FileNotFoundError(
            f"{metadata_path} is missing; run expansion with --helios first"
        )
    remote_expansion = None
    if metadata_path.is_file():
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
        with _source_stage_lock(source_id):
            _stage_source(ssh, root, remote_source)
            remote_pretrain, checkpoint_sha = _stage_pretrain(
                ssh, args.pretrain_dir.resolve(),
            )
        if stage_local_expansion:
            remote_expansion = f"{remote_eval}.input_expansion"
            _ssh(
                ssh, HELIOS_HOST,
                f"mkdir -p {shlex.quote(remote_expansion)}",
            )
            _rsync(
                ssh, f"{local_expansion}/",
                f"{HELIOS_HOST}:{remote_expansion}/",
            )
        if remote_expansion is None:
            raise RuntimeError("remote expansion path was not resolved")
        remote_args = _remote_argv(argv)
        remote_args = _replace_option(remote_args, "--pretrain-dir", remote_pretrain)
        remote_args = _replace_option(remote_args, "--expansion", remote_expansion)
        remote_args = _replace_option(remote_args, "--evaluation-output", remote_eval)
        remote_args = _replace_option(remote_args, "--device", "cuda:0")
        lab_task_config = getattr(args, "lab_task_config", None)
        if lab_task_config is not None:
            config = lab_task_config.resolve()
            try:
                relative = config.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "--helios currently requires --lab-task-config inside the repo"
                ) from error
            remote_args = _replace_option(
                remote_args,
                "--lab-task-config",
                f"{remote_source}/{relative}",
            )
        _ssh(
            ssh, HELIOS_HOST,
            f"mkdir -p {shlex.quote(str(Path(remote_eval).parent))}",
        )
        scene_bank_json = getattr(args, "scene_bank_json", None)
        if scene_bank_json is not None:
            local_scene_bank = scene_bank_json.resolve()
            if not local_scene_bank.is_file():
                raise FileNotFoundError(local_scene_bank)
            remote_scene_bank = f"{remote_eval}.input_scene_bank.json"
            _rsync(
                ssh, str(local_scene_bank),
                f"{HELIOS_HOST}:{remote_scene_bank}",
            )
            remote_args = _replace_option(
                remote_args, "--scene-bank-json", remote_scene_bank,
            )
        expansion_manifest = getattr(args, "expansion_manifest", None)
        if expansion_manifest is not None:
            local_manifest = expansion_manifest.resolve()
            if not local_manifest.is_file():
                raise FileNotFoundError(local_manifest)
            manifest_payload = json.loads(local_manifest.read_text())
            remote_manifest = f"{remote_eval}.input_manifest.json"
            _rsync(
                ssh, str(local_manifest),
                f"{HELIOS_HOST}:{remote_manifest}",
            )
            remote_args = _replace_option(
                remote_args, "--expansion-manifest", remote_manifest,
            )
            source_checkpoint = manifest_payload.get("source_checkpoint")
            if source_checkpoint is not None:
                local_checkpoint = Path(source_checkpoint).resolve()
                if not local_checkpoint.is_file():
                    raise FileNotFoundError(local_checkpoint)
                checkpoint_round = int(manifest_payload["checkpoint_round"])
                remote_snapshot = f"{remote_eval}.input_expansion"
                _ssh(
                    ssh, HELIOS_HOST,
                    f"mkdir -p {shlex.quote(remote_snapshot)}",
                )
                _rsync(
                    ssh, str(local_checkpoint),
                    f"{HELIOS_HOST}:{remote_snapshot}/"
                    f"checkpoint_{checkpoint_round:03d}.pt",
                )
                task_config = local_expansion / "task_config_resolved.json"
                if task_config.is_file():
                    _rsync(
                        ssh, str(task_config),
                        f"{HELIOS_HOST}:{remote_snapshot}/"
                        "task_config_resolved.json",
                    )
                remote_args = _replace_option(
                    remote_args, "--expansion", remote_snapshot,
                )
        job_token = uuid.uuid4().hex
        recovered_exit_code = None
        completion_ssh = ssh
        try:
            if args.helios_detached:
                command, status_file, log_file = _remote_detached_command(
                    remote_source, remote_eval, script_name, remote_args,
                    args.helios_gpu, share_gpu=args.helios_share_gpu,
                    job_token=job_token,
                )
                completion_ssh = ["ssh"]
                try:
                    _run_remote_job_detached(
                        ssh, command, job_token, status_file, log_file,
                    )
                finally:
                    _collect_detached_log(
                        completion_ssh, log_file, remote_eval,
                    )
            else:
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
            required = _required_metrics_evaluation_artifacts(
                remote_eval, script_name,
            )
            if not getattr(args, "metrics_only", True):
                # --gallery-view decides which gallery is rendered, so require
                # the one this invocation asked for rather than assuming the
                # side view is always written.
                gallery = (
                    "raw_gallery_headon.png"
                    if getattr(args, "gallery_view", "side") == "head_on"
                    else "raw_gallery.png"
                )
                required.append(f"{remote_eval}/{gallery}")
            if (
                not getattr(args, "metrics_only", True)
                and not getattr(args, "screening_only", True)
            ):
                required.append(f"{remote_eval}/mechanism.mp4")
            if not _remote_files_are_complete(completion_ssh, required):
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
            completion_ssh,
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
            completion_ssh,
            f"{HELIOS_HOST}:{actual_remote_eval}/", f"{local_eval}/",
        )

    _write_metadata(local_eval, {
        "status": "HELIOS_REMOTE_EVALUATION_COMPLETE",
        "host": HELIOS_HOST,
        "physical_gpu": args.helios_gpu,
        "gpu_policy": (
            "shared" if args.helios_share_gpu else "exclusive_queue"
        ),
        "detached": bool(args.helios_detached),
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
