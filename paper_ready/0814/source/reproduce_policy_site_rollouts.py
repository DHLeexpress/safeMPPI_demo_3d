#!/usr/bin/env python3
"""Re-run the PRE2/R1/expanded site selections from their exact seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


BUNDLE = Path(__file__).resolve().parents[1]
_early_model = (
    sys.argv[sys.argv.index("--model") + 1]
    if "--model" in sys.argv and sys.argv.index("--model") + 1 < len(sys.argv)
    else None
)
if _early_model == "less-expanded":
    _runtime_root = BUNDLE / "source_snapshots/less_expanded_r1_evaluation"
    _scripts_root = _runtime_root / "scripts"
else:
    _runtime_root = BUNDLE / "runtime_snapshot"
    _scripts_root = BUNDLE / "source"
sys.path[:0] = [str(_scripts_root), str(_runtime_root)]

from evaluate_multisphere_min_cost_deployment import _rollout  # noqa: E402
from safe_mppi.bowling_coverage import bowling_route_signature  # noqa: E402
from safe_mppi.config import load_config  # noqa: E402
from safe_mppi.lab_clutter_expansion import (  # noqa: E402
    LAB_CLUTTER_GOVERNOR_DIM,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (  # noqa: E402
    bowling_123_spheres,
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_clutter_pre2_multipair_expansion import (  # noqa: E402
    LabClutterPre2MultiPairExpansionTask,
)


MODEL_GROUPS = {
    "pre2": ("paper-ready-pre2", "not-paper-ready-pre2"),
    "less-expanded": ("paper-ready-less-expanded",),
    "expanded": ("paper-ready-expanded", "not-paper-ready-expanded"),
}


def _same_array(left, right) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=tuple(MODEL_GROUPS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="roll out arbitrary seeds instead of the frozen site selection",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="conditioning gamma used with --seeds (default: 0.1)",
    )
    parser.add_argument(
        "--verify-frozen",
        action="store_true",
        help="require array identity with the bundled selected trajectories",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.seeds and args.verify_frozen:
        raise ValueError("--verify-frozen cannot be combined with --seeds")

    selection = json.loads(
        (BUNDLE / "selections/paper_ready_bowling_selection.json").read_text()
    )
    config = load_config(BUNDLE / "config/task_config_resolved.json")
    scene_spec = sphere_scene_spec_from_config(config)
    wrapped = load_lab_clutter_pre2_expansion_policy(
        BUNDLE / "checkpoints/pre2/pretrained.pt",
        verifier_suffix_dim=LAB_CLUTTER_GOVERNOR_DIM + scene_spec.packed_dim,
    ).to(args.device).eval()
    wrapped.policy.nfe = 16
    wrapped.policy.flow.nfe = 16
    if args.model in ("less-expanded", "expanded"):
        checkpoint_path = (
            BUNDLE / "checkpoints/less_expanded/checkpoint_001.pt"
            if args.model == "less-expanded" else
            BUNDLE / "checkpoints/expanded/checkpoint_004.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        wrapped.policy.load_state_dict(checkpoint["model"], strict=True)

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
        paired_scene_seed=0,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        fixed_scene_layout="none",
        scene_spec=scene_spec,
    )
    spheres = bowling_123_spheres(task.env.start, task.env.goal, scene_spec.radius)
    frozen = torch.load(
        BUNDLE / "trajectories/paper_ready_bowling_handoff.pt",
        map_location="cpu",
        weights_only=False,
    )["groups"]
    output = {}
    groups = (
        [f"custom-{args.model}"]
        if args.seeds else list(MODEL_GROUPS[args.model])
    )
    for group in groups:
        output[group] = []
        expected_by_key = {
            (float(row["gamma"]), int(row["episode"]), int(row["rollout_seed"])): row
            for row in frozen.get(group, [])
        }
        selected_rows = (
            [
                {"gamma": args.gamma, "episode": index, "seed": seed}
                for index, seed in enumerate(args.seeds)
            ]
            if args.seeds else selection["groups"][group]
        )
        for selected in selected_rows:
            gamma = float(selected["gamma"])
            episode = int(selected["episode"])
            seed = int(selected["seed"])
            result = _rollout(
                wrapped, task, config, spheres, gamma, seed,
                samples_per_step=1,
                sampling_temperature=1.0,
                execution_cost_band_fraction=0.05,
            )
            route = None
            if result["status"] == "SUCCESS":
                route = bowling_route_signature(
                    result["states"], task.env.start, task.env.goal,
                    sphere_radius_m=scene_spec.radius,
                )
            row = {
                "gamma": gamma,
                "episode": episode,
                "rollout_seed": seed,
                "bowling_route": route,
                **result,
            }
            if args.verify_frozen:
                expected = expected_by_key[(gamma, episode, seed)]
                for key in ("states", "controls", "applied_controls", "dense_steps"):
                    if not _same_array(row[key], expected[key]):
                        raise RuntimeError(f"{group} seed={seed}: {key} is not bit-identical")
                if row["status"] != expected["status"]:
                    raise RuntimeError(f"{group} seed={seed}: status mismatch")
            output[group].append(row)
            print(f"{group} gamma={gamma:g} seed={seed} {row['status']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
