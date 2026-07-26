#!/usr/bin/env python3
"""Run a frozen ball-flow policy through Minhyuk's unchanged deploy_sim plant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy_sim import run_offline as native  # noqa: E402
from flow_deployment.bridge import (  # noqa: E402
    EndpointSimilarity,
    FlowDeploymentController,
    load_flow_policy,
    sha256_file,
    verify_deploy_sim_lock,
)
from flow_deployment.visualize import save_frame_bridge_figure  # noqa: E402
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _trace_arrays(rows):
    if not rows:
        return {}
    return {
        "episode": np.asarray([row["episode"] for row in rows], np.int32),
        "step": np.asarray([row["step"] for row in rows], np.int32),
        "seed": np.asarray([row["seed"] for row in rows], np.int64),
        "gamma": np.asarray([row["gamma"] for row in rows], np.float32),
        "target_state": np.stack([row["target_state"] for row in rows]),
        "source_state": np.stack([row["source_state"] for row in rows]),
        "context": np.stack([row["context"] for row in rows]),
        "source_plan": np.stack([row["source_plan"] for row in rows]),
        "target_plan": np.stack([row["target_plan"] for row in rows]),
        "saturation_scale": np.stack(
            [row["saturation_scale"] for row in rows]
        ),
        "action": np.stack([row["action"] for row in rows]),
    }


def _validate_source_contract(source_config, policy):
    expected_start = np.array([0.0, 0.0, 2.0])
    expected_goal = np.array([3.0, 0.0, 2.0])
    expected_sphere = np.array([1.5, 0.0, 2.0, 0.254])
    actual = {
        "start": np.asarray(source_config.taskspace.start[:3], float),
        "goal": np.asarray(source_config.taskspace.goal, float),
        "sphere": np.asarray(source_config.obstacles.spheres[0], float),
    }
    expected = {
        "start": expected_start,
        "goal": expected_goal,
        "sphere": expected_sphere,
    }
    mismatches = [
        key for key in expected
        if not np.allclose(actual[key], expected[key], atol=1.0e-10)
    ]
    if mismatches:
        raise ValueError(f"canonical source task mismatch: {mismatches}")
    if not np.isclose(float(source_config.safemppi.dt), 0.1):
        raise ValueError("canonical policy bridge requires source dt=0.1")
    if tuple(policy.plan_shape) != (10, 3):
        raise ValueError("canonical policy bridge requires H=10, action_dim=3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/crazyflie_mppi_corner.json",
    )
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        default=(
            REPO_ROOT / "results/global50_reference/"
            "pretrain_global10_h48p32_s0"
        ),
    )
    parser.add_argument("--expansion", type=Path)
    parser.add_argument("--round", dest="round_index", type=int)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/flow_deployment/pretrained_corner",
    )
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if (args.expansion is None) != (args.round_index is None):
        parser.error("--expansion and --round must be supplied together")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    lock_before = verify_deploy_sim_lock(REPO_ROOT)
    target_config = load_config(args.config)
    target_env = TaskEnvironment(target_config)
    source_config_path = args.pretrain_dir / "demo_config.json"
    source_config = load_config(source_config_path)
    source_env = TaskEnvironment(source_config)
    if len(source_env.spheres) != 1 or len(target_env.spheres) != 1:
        raise ValueError("temporary bridge requires one sphere in each task")

    frame = EndpointSimilarity.from_endpoints(
        source_env.start[:3],
        source_env.goal,
        target_env.start[:3],
        target_env.goal,
    )
    policy, policy_provenance = load_flow_policy(
        args.pretrain_dir, args.expansion, args.round_index,
    )
    _validate_source_contract(source_config, policy)
    policy_provenance["demo_config"] = str(source_config_path.resolve())
    policy_provenance["demo_config_sha256"] = sha256_file(source_config_path)
    controller = FlowDeploymentController(
        policy,
        frame,
        target_env.spheres[0],
        target_action_limit=float(target_config.safemppi.demo_u_max),
        device=args.device,
    )
    gamma = (
        float(args.gamma)
        if args.gamma is not None
        else float(target_config.data.gammas[0])
    )

    episode_rows = []
    trace_rows = []
    results = []
    for episode in range(args.episodes):
        seed = args.seed_start + episode
        swarm = native.Swarm(
            np.array([target_env.start[0], target_env.start[1], 0.0]),
            seed=seed,
        )
        result = native.harness.run(
            controller,
            target_env,
            target_config,
            swarm,
            gamma=gamma,
            seed=seed,
            verbose=False,
        )
        metrics = native.harness.summarize(result, target_env, verbose=False)
        episode_rows.append({
            "episode": episode,
            "seed": seed,
            "reached": bool(result["reached"]),
            "outcome": result["outcome"],
            **metrics,
        })
        for step, row in enumerate(controller.trace):
            trace_rows.append({"episode": episode, "step": step, **row})
        results.append(result)

    successful = [
        index for index, result in enumerate(results) if result["reached"]
    ]
    selected = successful[0] if successful else 0
    selected_result = results[selected]
    native_base = native.save(
        selected_result,
        target_env,
        target_config,
        args.output / "native_deploy_sim",
        gamma,
        args.gif,
        True,
    )
    target_path = np.asarray([
        [row["x"], row["y"], row["z"]]
        for row in selected_result["log"]
    ])
    bridge_png, bridge_pdf = save_frame_bridge_figure(
        args.output / "frame_bridge.png",
        frame,
        source_env.spheres[0],
        target_env.spheres[0],
        target_path,
    )
    arrays = _trace_arrays(trace_rows)
    if arrays:
        np.savez_compressed(args.output / "controller_trace.npz", **arrays)

    success_rate = len(successful) / args.episodes
    saturation = arrays.get("saturation_scale")
    saturated_fraction = (
        float(np.mean(saturation < 1.0 - 1.0e-7))
        if saturation is not None and saturation.size
        else 0.0
    )
    contract = {
        "status": "FLOW_DEPLOYMENT_DIAGNOSTIC_COMPLETE",
        "scope": (
            "Frozen-policy offline interconnection diagnostic; no online "
            "expansion, motion-capture collection, or flight safety guarantee."
        ),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "policy": policy_provenance,
        "frame": frame.contract(),
        "controller": controller.contract(),
        "gamma": gamma,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "success_rate": success_rate,
        "plan_knot_saturated_fraction": saturated_fraction,
        "successful_episode_indices": successful,
        "selected_episode": selected,
        "selected_native_base": None if native_base is None else str(native_base),
        "frame_bridge_png": str(bridge_png),
        "frame_bridge_pdf": str(bridge_pdf),
        "deploy_sim_lock_before": lock_before,
        "deploy_sim_lock_after": verify_deploy_sim_lock(REPO_ROOT),
    }
    (args.output / "episode_metrics.json").write_text(
        json.dumps(episode_rows, indent=2, default=_jsonable) + "\n"
    )
    (args.output / "run_contract.json").write_text(
        json.dumps(contract, indent=2, default=_jsonable) + "\n"
    )
    print(json.dumps({
        "status": contract["status"],
        "episodes": args.episodes,
        "success_rate": success_rate,
        "selected_episode": selected,
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
