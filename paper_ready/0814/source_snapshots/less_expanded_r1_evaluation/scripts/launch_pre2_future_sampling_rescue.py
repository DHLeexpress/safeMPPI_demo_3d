#!/usr/bin/env python3
"""Dry-run or enqueue one exact W300/Adam5000 future-sampling rescue round."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/Users/dhl/Documents/safeMPPI_demo_3d")
sys.path.insert(0, str(ROOT))

from safe_mppi.helios_remote import (  # noqa: E402
    HELIOS_HOST, REMOTE_PYTHON, REMOTE_SOURCE_BASE,
    _source_id, _source_stage_lock, _ssh_master, _stage_source,
)
from scripts.run_pre2_future_sampling_rescue import (  # noqa: E402
    APPROVED_RECIPE_SHA256, HASHED_FIELDS, HISTORICAL_SEED,
    NATIVE_EVALUATION, _canonical_sha256, _validate_spec,
)


LOCAL_STAGE = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_saved_r1_steps5000_r7/"
    "r10_finish/future_sampling_rescue"
)
SOURCE_SPEC = ROOT / (
    "results/stage1_single_ball_t128/0812_pre2_saved_r1_steps5000_r7/"
    "r10_finish/spec--6de15c31b4b7.json"
)
REMOTE_BASE = (
    "/data3/research1/safeMPPI_remote_cli/spooled_sweeps/"
    "0812_pre2_saved_r1_steps5000_r7/r7_r10_finish"
)
DEFAULT_PARENT_OUTPUT = (
    f"{REMOTE_BASE}/snapshots/w300_steps5000_exact_r8"
)
REMOTE_RESCUE_STAGE = f"{REMOTE_BASE}/future_sampling_rescue"
TSP_SOCKETS = {
    1: "/tmp/smppi-speedband-r2-g1.sock",
    3: "/tmp/smppi-speedband-r2-g3.sock",
}
CONFIRM = "I_UNDERSTAND_THIS_LAUNCHES_ONE_FUTURE_SAMPLING_ROUND_ONLY"


REMOTE_PARENT_INSPECTION = r"""
import hashlib
import json
from pathlib import Path
import sys
import torch

output = Path(sys.argv[1])
expected_round = int(sys.argv[2])

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected object: ' + str(path))
    return value

if (output / 'RESUME_IN_PROGRESS.json').exists():
    raise RuntimeError('parent has RESUME_IN_PROGRESS.json')
paths = {
    'checkpoint': output / f'checkpoint_{expected_round:03d}.pt',
    'query': output / f'query_archive_round_{expected_round:03d}.pt',
    'events': output / f'events_round_{expected_round:03d}.pt',
    'resume_json': output / 'resume_state.json',
    'resume_state': output / 'resume_state_latest.pt',
    'manifest': output / 'manifest.json',
    'metrics': output / 'metrics.jsonl',
}
missing = [str(path) for path in paths.values() if not path.is_file() or not path.stat().st_size]
if missing:
    raise FileNotFoundError('incomplete parent: ' + ', '.join(missing))
metadata = read_json(paths['resume_json'])
optimizer_step = expected_round * 5000
if not (
    metadata.get('status') == 'COMMITTED_ROUND_RESUME'
    and int(metadata.get('version', -1)) == 1
    and int(metadata.get('completed_round', -1)) == expected_round
    and int(metadata.get('next_round', -1)) == expected_round + 1
    and int(metadata.get('optimizer_step', -1)) == optimizer_step
):
    raise RuntimeError('parent resume metadata mismatch')
state = torch.load(paths['resume_state'], map_location='cpu', weights_only=False)
if not (
    isinstance(state, dict)
    and state.get('status') == 'COMMITTED_ROUND_RESUME'
    and int(state.get('version', -1)) == 1
    and int(state.get('completed_round', -1)) == expected_round
    and int(state['optimizer_metadata']['_safe_mppi_schedule_step']) == optimizer_step
):
    raise RuntimeError('parent resume payload mismatch')
config = state['config']
guards = {
    'seed': 82410, 'inner_steps': 5000,
    'successful_trajectories_per_gamma': 12,
    'K': 16, 'B': 8, 'retry_B': 8,
    'execution_cost_band_fraction': 0.05,
    'replay_scope': 'cumulative', 'replay_rounds': 100,
    'retry_verify_all_fast_path': False,
    'paired_noised_representation': True,
}
for key, expected in guards.items():
    if config.get(key) != expected:
        raise RuntimeError(f'parent config drift: {key}={config.get(key)!r}')
manifest = read_json(paths['manifest'])
metrics = [json.loads(line) for line in paths['metrics'].read_text().splitlines() if line]
if [int(row.get('round', -1)) for row in metrics] != list(range(1, expected_round + 1)):
    raise RuntimeError('parent metrics round sequence mismatch')
if not (state.get('round_rows') == manifest.get('rounds') == metrics):
    raise RuntimeError('parent histories differ')
print(json.dumps({
    'completed_round': expected_round,
    'optimizer_step': optimizer_step,
    'historical_seed': int(config['seed']),
    'future_sampling_seed': config.get('future_sampling_seed'),
    'checkpoint_sha256': sha(paths['checkpoint']),
    'resume_state_sha256': sha(paths['resume_state']),
    'resume_json_sha256': sha(paths['resume_json']),
    'manifest_sha256': sha(paths['manifest']),
    'metrics_sha256': sha(paths['metrics']),
}, sort_keys=True))
"""


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _remote_parent_contract(
    parent_output: str, parent_round: int, *, pickle_source: str,
) -> dict[str, Any]:
    command = (
        f"cd {shlex.quote(pickle_source)} && "
        f"PYTHONPATH=. {shlex.quote(REMOTE_PYTHON)} -c "
        f"{shlex.quote(REMOTE_PARENT_INSPECTION)} "
        f"{shlex.quote(parent_output)} {int(parent_round)}"
    )
    payload = subprocess.check_output(
        ["ssh", HELIOS_HOST, command], text=True,
    )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("remote parent contract is not an object")
    return value


def _default_branch_output(
    parent_round: int, future_sampling_seed: int,
    parent_contract: dict[str, Any],
) -> str:
    checkpoint_tag = parent_contract["checkpoint_sha256"][:12]
    return (
        f"{REMOTE_RESCUE_STAGE}/branches/"
        f"from_r{parent_round:03d}_{checkpoint_tag}/seed{future_sampling_seed}"
    )


def build_spec(
    *, future_sampling_seed: int,
    parent_output: str = DEFAULT_PARENT_OUTPUT,
    parent_round: int = 8,
    branch_output: str | None = None,
    physical_gpu: int = 3,
    parent_contract: dict[str, Any] | None = None,
    source_id: str | None = None,
    remote_source: str | None = None,
    source_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = _read(SOURCE_SPEC) if source_spec is None else deepcopy(source_spec)
    args = deepcopy(source["expansion_args"])
    if _canonical_sha256(args) != APPROVED_RECIPE_SHA256:
        raise ValueError("local source spec is not the approved W300/Adam5000 recipe")
    source_id = _source_id(ROOT) if source_id is None else source_id
    remote_source = (
        f"{REMOTE_SOURCE_BASE}/{source_id}"
        if remote_source is None else remote_source
    )
    if parent_contract is None:
        # The historical staged source is guaranteed to deserialize this snapshot;
        # launch-time validation repeats the check with the newly staged source.
        parent_contract = _remote_parent_contract(
            parent_output, parent_round, pickle_source=source["remote_source"],
        )
    else:
        parent_contract = deepcopy(parent_contract)
    if branch_output is None:
        branch_output = _default_branch_output(
            parent_round, future_sampling_seed, parent_contract,
        )
    target_round = parent_round + 1
    spec: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pre2_future_sampling_one_round_rescue",
        "name": (
            f"w300_steps5000_r{parent_round:03d}_r{target_round:03d}_"
            f"future{future_sampling_seed}"
        ),
        "physical_gpu": physical_gpu,
        "source_id": source_id,
        "source_config_hash": source["config_hash"],
        "approved_recipe_sha256": APPROVED_RECIPE_SHA256,
        "remote_source": remote_source,
        "remote_pretrain": source["remote_pretrain"],
        "remote_task_config": source["remote_task_config"],
        "parent_output": parent_output,
        "branch_output": branch_output,
        "remote_control": f"{REMOTE_RESCUE_STAGE}/control",
        "parent_round": parent_round,
        "target_round": target_round,
        "historical_seed": HISTORICAL_SEED,
        "future_sampling_seed": future_sampling_seed,
        "expansion_args": args,
        "parent_contract": parent_contract,
        "evaluation": deepcopy(NATIVE_EVALUATION),
    }
    spec["hash_payload"] = {field: spec[field] for field in HASHED_FIELDS}
    spec["config_hash"] = _canonical_sha256(spec["hash_payload"])
    _validate_spec(spec, check_cuda=False)
    return spec


def _remote_duplicate_scan(spec: dict[str, Any], socket: str) -> None:
    key = f"{spec['name']}--{spec['config_hash'][:12]}"
    label = f"future-r{spec['parent_round']}-r{spec['target_round']}:{spec['config_hash'][:8]}"
    code = r"""
import os
from pathlib import Path
import sys

needle = sys.argv[1].encode()
self_pid = os.getpid()
excluded = set()
pid = self_pid
while pid > 1 and pid not in excluded:
    excluded.add(pid)
    try:
        pid = int(Path(f'/proc/{pid}/stat').read_text().split()[3])
    except Exception:
        break
matches = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in excluded:
        continue
    try:
        command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if b'run_pre2_future_sampling_rescue.py' in command and needle in command:
        matches.append({'pid': int(entry.name), 'command': command.decode(errors='replace')})
if matches:
    raise RuntimeError('duplicate live target: ' + repr(matches))
"""
    status = Path(spec["remote_control"]) / "status"
    command = (
        "set -euo pipefail; "
        f"test ! -e {shlex.quote(spec['branch_output'])}; "
        f"test ! -e {shlex.quote(str(status / (key + '.RUNNING.json')))}; "
        f"test ! -e {shlex.quote(str(status / (key + '.COMPLETE.json')))}; "
        f"cd {shlex.quote(spec['remote_source'])}; "
        f"PYTHONPATH=. {shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)} "
        f"{shlex.quote(spec['config_hash'][:12])}; "
        f"if export TS_SOCKET={shlex.quote(socket)}; tsp -l | grep -F -- "
        f"{shlex.quote(label)} >/dev/null; then exit 23; fi"
    )
    subprocess.run(["ssh", HELIOS_HOST, command], check=True)


def remote_validate(spec: dict[str, Any]) -> None:
    actual = _remote_parent_contract(
        spec["parent_output"], int(spec["parent_round"]),
        pickle_source=spec["remote_source"],
    )
    if actual != spec["parent_contract"]:
        raise RuntimeError("remote parent changed between spec creation and launch")
    _remote_duplicate_scan(spec, TSP_SOCKETS[int(spec["physical_gpu"])])


def _enqueue(spec: dict[str, Any], remote_spec: str) -> tuple[int, str, str]:
    gpu = int(spec["physical_gpu"])
    socket = TSP_SOCKETS[gpu]
    label = (
        f"future-r{spec['parent_round']}-r{spec['target_round']}:"
        f"{spec['config_hash'][:8]}"
    )
    log = (
        f"{spec['remote_control']}/logs/{spec['name']}--"
        f"{spec['config_hash'][:12]}.log"
    )
    worker = shlex.join([
        REMOTE_PYTHON,
        f"{spec['remote_source']}/scripts/run_pre2_future_sampling_rescue.py",
        "--spec", remote_spec,
    ])
    shell = (
        "set -euo pipefail; export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; "
        f"exec {worker} >> {shlex.quote(log)} 2>&1"
    )
    submit = (
        f"export TS_SOCKET={shlex.quote(socket)} TS_MAXFINISHED=200; "
        f"tsp -B -L {shlex.quote(label)} bash -lc {shlex.quote(shell)}"
    )
    output = subprocess.check_output(
        ["ssh", HELIOS_HOST, submit], text=True,
    ).strip()
    if not output.isdigit():
        raise RuntimeError(f"unexpected task-spooler job ID: {output!r}")
    return int(output), socket, log


def launch(spec: dict[str, Any]) -> dict[str, Any]:
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    marker = LOCAL_STAGE / (
        f"LAUNCHED--{spec['name']}--{spec['config_hash'][:12]}.json"
    )
    lock_path = LOCAL_STAGE / ".launch.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another future-sampling launcher is active") from error
        if marker.exists():
            raise RuntimeError(f"refusing duplicate launch: {marker}")
        with _ssh_master(HELIOS_HOST) as ssh:
            with _source_stage_lock(spec["source_id"]):
                _stage_source(ssh, ROOT, spec["remote_source"])
        remote_validate(spec)

        local_spec = LOCAL_STAGE / "specs" / (
            f"{spec['name']}--{spec['config_hash'][:12]}.json"
        )
        _atomic_json(local_spec, spec)
        remote_specs = f"{spec['remote_control']}/specs"
        remote_spec = f"{remote_specs}/{local_spec.name}"
        directories = (
            remote_specs, f"{spec['remote_control']}/logs",
            f"{spec['remote_control']}/locks",
            f"{spec['remote_control']}/status",
        )
        subprocess.run([
            "ssh", HELIOS_HOST,
            "mkdir -p " + " ".join(shlex.quote(path) for path in directories),
        ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", "ssh", str(local_spec),
            f"{HELIOS_HOST}:{remote_spec}",
        ], check=True)
        job_id, socket, log = _enqueue(spec, remote_spec)
        payload = {
            "schema_version": 1, "status": "LAUNCHED",
            "created_unix": time.time(), "name": spec["name"],
            "config_hash": spec["config_hash"],
            "physical_gpu": spec["physical_gpu"], "tsp_socket": socket,
            "tsp_job_id": job_id, "local_spec": str(local_spec),
            "remote_spec": remote_spec, "remote_log": log,
            "parent_output": spec["parent_output"],
            "branch_output": spec["branch_output"],
            "parent_round": spec["parent_round"],
            "target_round": spec["target_round"],
            "future_sampling_seed": spec["future_sampling_seed"],
            "evaluation": spec["evaluation"],
            "selection_scope": (
                "seed93211 calibration bank only; final seed91000 is not used"
            ),
        }
        _atomic_json(marker, payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-sampling-seed", type=int, required=True)
    parser.add_argument("--parent-output", default=DEFAULT_PARENT_OUTPUT)
    parser.add_argument("--parent-round", type=int, default=8)
    parser.add_argument("--branch-output")
    parser.add_argument("--gpu", type=int, choices=(1, 3), default=3)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    spec = build_spec(
        future_sampling_seed=args.future_sampling_seed,
        parent_output=args.parent_output,
        parent_round=args.parent_round,
        branch_output=args.branch_output,
        physical_gpu=args.gpu,
    )
    if args.launch:
        if args.confirm != CONFIRM:
            parser.error(f"--launch requires --confirm {CONFIRM}")
        payload = launch(spec)
    else:
        payload = {
            "status": "DRY_RUN_VALIDATION_PASSED",
            "launch_performed": False,
            "source_spec": str(SOURCE_SPEC),
            "spec": spec,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
