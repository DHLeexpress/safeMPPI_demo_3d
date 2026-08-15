"""Compare PRE and expansion checkpoints on one fixed bowling scene.

The deployment contract is deliberately identical to the synchronized raw-M8
hybrid evaluator: draw M raw flow plans, shortlist by execution cost, then use
the nominal first-step margin only inside the configured bounded-regret band.
Verifier/progress labels never participate in action selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_multisphere_min_cost_deployment import _rollout, _summary
from safe_mppi.bowling_coverage import (
    bowling_route_signature,
    summarize_bowling_coverage,
)
from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_GOVERNOR_DIM,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (
    bowling_123_spheres,
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimate(path: np.ndarray, limit: int = 64) -> list[list[float]]:
    values = np.asarray(path, np.float64)[:, :3]
    if len(values) > limit:
        indices = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(int))
        values = values[indices]
    return np.round(values, 4).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint-rounds", default="0,3")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--samples-per-step", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument(
        "--save-raw-trajectories",
        action="store_true",
        help=(
            "also preserve full state/control sequences in raw_trajectories.pt; "
            "this does not change deployment or summary computation"
        ),
    )
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument("--execution-clearance-exp-weight", type=float, default=15.0)
    parser.add_argument("--execution-clearance-target-m", type=float, default=0.6)
    parser.add_argument("--execution-clearance-exp-temperature", type=float, default=0.15)
    parser.add_argument("--execution-taskspace-quadratic-weight", type=float, default=250.0)
    parser.add_argument("--execution-taskspace-quadratic-target-m", type=float, default=0.15)
    parser.add_argument("--execution-axis-cylinder-quadratic-weight", type=float, default=5.0)
    parser.add_argument("--execution-axis-cylinder-radius-m", type=float, default=1.1)
    parser.add_argument("--execution-control-weight", type=float, default=0.05)
    parser.add_argument("--execution-obstacle-speed-weight", type=float, default=400.0)
    parser.add_argument("--execution-cost-band-fraction", type=float, default=0.05)
    args = parser.parse_args()

    rounds = tuple(sorted({int(value) for value in args.checkpoint_rounds.split(",")}))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.episodes < 1 or args.samples_per_step < 1:
        parser.error("episodes and samples-per-step must be positive")

    config = load_config(args.expansion / "task_config_resolved.json")
    scene_spec = sphere_scene_spec_from_config(config)
    wrapped = load_lab_clutter_pre2_expansion_policy(
        args.pretrain_dir / "pretrained.pt",
        verifier_suffix_dim=LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim,
    ).to(args.device).eval()
    wrapped.policy.nfe = 16
    wrapped.policy.flow.nfe = 16
    task = LabClutterPre2MultiPairExpansionTask(
        config,
        context_schema=wrapped.context_schema,
        device=args.device,
        tight_corridor=False,
        verifier_mode="full_polytope",
        verifier_solver="analytic",
        execution_clearance_exp_weight=args.execution_clearance_exp_weight,
        execution_clearance_target_m=args.execution_clearance_target_m,
        execution_clearance_exp_temperature=args.execution_clearance_exp_temperature,
        execution_taskspace_quadratic_weight=args.execution_taskspace_quadratic_weight,
        execution_taskspace_quadratic_target_m=args.execution_taskspace_quadratic_target_m,
        execution_axis_cylinder_quadratic_weight=args.execution_axis_cylinder_quadratic_weight,
        execution_axis_cylinder_radius_m=args.execution_axis_cylinder_radius_m,
        execution_control_weight=args.execution_control_weight,
        execution_obstacle_speed_weight=args.execution_obstacle_speed_weight,
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=args.seed,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        # Evaluation supplies the fixed bowling spheres directly to _rollout;
        # the multi-pair task itself intentionally keeps fixed-scene collection
        # disabled.
        fixed_scene_layout="none",
        scene_spec=scene_spec,
    )
    spheres = bowling_123_spheres(task.env.start, task.env.goal, scene_spec.radius)
    base_env = task._environment(spheres)
    gammas = [float(value) for value in config.data.gammas]

    rows_by_round: dict[str, list[dict]] = {}
    raw_rows_by_round: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    checkpoint_hashes: dict[str, str] = {}
    total = len(rounds) * len(gammas) * args.episodes
    completed = 0
    for round_i in rounds:
        checkpoint_path = args.expansion / f"checkpoint_{round_i:03d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        wrapped.policy.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_hashes[str(round_i)] = _sha256(checkpoint_path)
        rows: list[dict] = []
        raw_rows: list[dict] = []
        for gamma in gammas:
            for episode in range(args.episodes):
                rollout_seed = args.seed + 37 * episode
                result = _rollout(
                    wrapped,
                    task,
                    config,
                    spheres,
                    gamma,
                    rollout_seed,
                    args.samples_per_step,
                    args.sampling_temperature,
                    args.execution_cost_band_fraction,
                )
                route = None
                if result["status"] == "SUCCESS":
                    route = bowling_route_signature(
                        result["states"],
                        base_env.start,
                        base_env.goal,
                        sphere_radius_m=scene_spec.radius,
                    )
                rows.append({
                    "round": round_i,
                    "gamma": gamma,
                    "episode": episode,
                    "rollout_seed": rollout_seed,
                    "status": result["status"],
                    "min_clearance_m": result["min_clearance_m"],
                    "time_to_goal_s": result["time_to_goal_s"],
                    "window_validity": result["window_validity"],
                    "mean_chosen_execution_cost": result["mean_chosen_execution_cost"],
                    "mean_chosen_step_margin": result["mean_chosen_step_margin"],
                    "bowling_route": route,
                    "path_xyz_m": _decimate(result["states"]),
                })
                if args.save_raw_trajectories:
                    raw_rows.append({
                        "round": round_i,
                        "gamma": gamma,
                        "episode": episode,
                        "rollout_seed": rollout_seed,
                        "bowling_route": route,
                        **result,
                    })
                completed += 1
                if completed % 20 == 0 or completed == total:
                    print(f"[bowling M8] {completed}/{total}", flush=True)
        rows_by_round[str(round_i)] = rows
        if args.save_raw_trajectories:
            raw_rows_by_round[str(round_i)] = raw_rows
        summaries[str(round_i)] = {
            "pooled": _summary(rows),
            "per_gamma": {
                f"{gamma:g}": _summary([row for row in rows if row["gamma"] == gamma])
                for gamma in gammas
            },
            "bowling": summarize_bowling_coverage(rows),
        }

    args.output.mkdir(parents=True)
    payload = {
        "status": "FIXED_BOWLING_PRE_VS_EXPANSION_COMPLETE",
        "scene": {
            "layout": "bowling_1_2_3",
            "spheres": np.round(spheres, 6).tolist(),
            "start": np.round(base_env.start[:3], 6).tolist(),
            "goal": np.round(base_env.goal[:3], 6).tolist(),
            "bounds": np.round(base_env.bounds, 6).tolist(),
        },
        "deployment_contract": {
            "samples_per_step": args.samples_per_step,
            "sampling_temperature": args.sampling_temperature,
            "NFE": 16,
            "verifier_or_progress_used_for_selection": False,
            "obstacle_exponential": [
                args.execution_clearance_exp_weight,
                args.execution_clearance_target_m,
                args.execution_clearance_exp_temperature,
            ],
            "wall": [
                args.execution_taskspace_quadratic_weight,
                args.execution_taskspace_quadratic_target_m,
            ],
            "axis": [
                args.execution_axis_cylinder_quadratic_weight,
                args.execution_axis_cylinder_radius_m,
            ],
            "control_weight": args.execution_control_weight,
            "obstacle_conditioned_speed_weight": args.execution_obstacle_speed_weight,
            "cost_band_fraction": args.execution_cost_band_fraction,
        },
        "checkpoint_rounds": list(rounds),
        "artifact_binding": {
            "checkpoint_sha256_by_round": checkpoint_hashes,
            "pretrained_sha256": _sha256(args.pretrain_dir / "pretrained.pt"),
        },
        "summary": summaries,
        "rows": rows_by_round,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    (args.output / "bowling_eval.json").write_text(serialized)
    if args.save_raw_trajectories:
        torch.save(raw_rows_by_round, args.output / "raw_trajectories.pt")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
