"""Replay deterministic faithful-M1 evaluation rows with full trajectories.

The canonical raw deployment artifact intentionally omits state/control arrays.
This helper replays the first stored row for each requested round/status using
the stored scene, gamma, and rollout seed, then records both the authoritative
outcome and the replayed full state sequence for visualization audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_multisphere_min_cost_deployment import _rollout
from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_GOVERNOR_DIM,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)


def _json_array(value: np.ndarray, digits: int = 5) -> list:
    return np.round(np.asarray(value, np.float64), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rounds", default="1,3,5,7,9")
    parser.add_argument("--statuses", default="SUCCESS,COLLISION,OOB")
    parser.add_argument("--per-status", type=int, default=1)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rounds = tuple(int(value) for value in args.rounds.split(","))
    statuses = tuple(value.strip().upper() for value in args.statuses.split(","))
    if args.per_status < 1:
        parser.error("--per-status must be positive")

    canonical = json.loads(args.raw_eval.read_text())
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
        execution_clearance_exp_weight=15.0,
        execution_clearance_target_m=0.6,
        execution_clearance_exp_temperature=0.15,
        execution_taskspace_quadratic_weight=250.0,
        execution_taskspace_quadratic_target_m=0.15,
        execution_axis_cylinder_quadratic_weight=5.0,
        execution_axis_cylinder_radius_m=1.1,
        execution_control_weight=0.05,
        execution_obstacle_speed_weight=400.0,
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=91000,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        fixed_scene_layout="none",
        scene_spec=scene_spec,
    )

    full_rows: list[dict] = []
    compact_rows: list[dict] = []
    for round_i in rounds:
        checkpoint = torch.load(
            args.expansion / f"checkpoint_{round_i:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        wrapped.policy.load_state_dict(checkpoint["model"], strict=True)
        canonical_rows = canonical["rows"][str(round_i)]
        for status in statuses:
            selected = [
                row for row in canonical_rows if row["status"] == status
            ][: args.per_status]
            if len(selected) != args.per_status:
                raise RuntimeError(
                    f"round {round_i} has only {len(selected)} rows for {status}"
                )
            for selection_index, source in enumerate(selected):
                result = _rollout(
                    wrapped,
                    task,
                    config,
                    source["spheres"],
                    float(source["gamma"]),
                    int(source["rollout_seed"]),
                    samples_per_step=1,
                    sampling_temperature=1.0,
                    execution_cost_band_fraction=0.05,
                )
                identity = {
                    "round": round_i,
                    "requested_status": status,
                    "selection_index": selection_index,
                    "gamma": float(source["gamma"]),
                    "episode": int(source["episode"]),
                    "rollout_seed": int(source["rollout_seed"]),
                    "scene_seed": int(source["scene_seed"]),
                    "scene_hash": str(source["scene_hash"]),
                    "spheres": source["spheres"],
                    "authoritative_status": str(source["status"]),
                    "replay_status": str(result["status"]),
                    "status_match": result["status"] == source["status"],
                    "authoritative_min_clearance_m": source["min_clearance_m"],
                    "authoritative_time_to_goal_s": source["time_to_goal_s"],
                    "replay_min_clearance_m": result["min_clearance_m"],
                    "replay_time_to_goal_s": result["time_to_goal_s"],
                    "replay_window_validity": result["window_validity"],
                }
                full_rows.append({**identity, **result})
                compact_rows.append({
                    **identity,
                    "states": _json_array(result["states"]),
                })
                print(
                    f"[trajectory] r{round_i} {status} episode={source['episode']} "
                    f"replay={result['status']} states={len(result['states'])}",
                    flush=True,
                )

    args.output.mkdir(parents=True)
    torch.save(full_rows, args.output / "raw_trajectories.pt")
    payload = {
        "status": "FAITHFUL_M1_TRAJECTORY_REPLAY_COMPLETE",
        "contract": {
            "NFE": 16,
            "samples_per_step": 1,
            "sampling_temperature": 1.0,
            "execution_obstacle_exponential": [15.0, 0.6, 0.15],
            "wall": [250.0, 0.15],
            "axis": [5.0, 1.1],
            "control_weight": 0.05,
            "obstacle_conditioned_speed_weight": 400.0,
            "cost_band_fraction": 0.05,
            "selection": "first canonical row in stored order per round/status",
            "device": args.device,
        },
        "rounds": list(rounds),
        "statuses": list(statuses),
        "per_status": args.per_status,
        "all_statuses_match": all(row["status_match"] for row in compact_rows),
        "rows": compact_rows,
    }
    (args.output / "trajectory_visual_data.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
