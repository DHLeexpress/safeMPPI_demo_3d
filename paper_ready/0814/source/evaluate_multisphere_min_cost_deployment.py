"""Evaluate raw best-of-M cost-ranked deployment on a fixed clutter seed bank.

This is an explicit deployment ablation, not the canonical raw curve.  At
every closed-loop step it samples exactly M flow plans and executes the global
minimum under the same native + wall + axis-cylinder + control cost used by
round-1 expansion.  Verifier validity and progress never enter selection;
collision, OOB, success, timeout, and window validity are measured only from
the resulting executed closed-loop trajectory.
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
ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SOURCE_FILES = (
    "scripts/evaluate_multisphere_min_cost_deployment.py",
    "safe_mppi/config.py",
    "safe_mppi/environment.py",
    "safe_mppi/execution_costs.py",
    "safe_mppi/flow_model.py",
    "safe_mppi/lab_visual_flow.py",
    "safe_mppi/lab_clutter_evaluation.py",
    "safe_mppi/lab_clutter_expansion.py",
    "safe_mppi/lab_clutter_pre2_expansion.py",
    "safe_mppi/lab_clutter_pre2_multipair_expansion.py",
    "safe_mppi/lab_reference_flow_task.py",
    "safe_mppi/expansion_multisphere_multipair.py",
)

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.helios_remote import (
    add_helios_arguments,
    run_evaluation_on_helios,
)
from safe_mppi.lab_clutter_evaluation import _scene_config, _summarize
from safe_mppi.lab_clutter_expansion import (
    LAB_CLUTTER_GOVERNOR_DIM,
    scene_sha256,
    sphere_scene_spec_from_config,
)
from safe_mppi.lab_clutter_pre2_expansion import (
    load_lab_clutter_pre2_expansion_policy,
)
from safe_mppi.lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)
from safe_mppi.lab_reference_flow_task import (
    reference_window_validity_fraction,
)
from safe_mppi.ball_flow_task import nominal_hp_chain_margins
from safe_mppi.expansion_multisphere_multipair import (
    bounded_regret_step_margin_choice,
)


@torch.no_grad()
def _rollout(
    wrapped,
    task,
    config,
    spheres,
    gamma: float,
    seed: int,
    samples_per_step: int,
    sampling_temperature: float,
    execution_cost_band_fraction: float,
):
    spheres = np.asarray(spheres, np.float32)
    env = task._environment(spheres)
    state = {
        "x": env.start.copy(),
        "previous_applied": np.zeros(3, np.float32),
        "previous_raw": np.zeros(3, np.float32),
        "spheres": spheres,
        "scene_seed": int(seed),
        "scene_hash": scene_sha256(env, spheres),
        "steps": 0,
        "collided": False,
        "oob": False,
    }
    states = [state["x"].copy()]
    controls = []
    applied_controls = []
    dense_steps = []
    chosen_costs = []
    chosen_step_margins = []
    sampled_counts = []
    status = "TIMEOUT"

    for step in range(config.taskspace.max_steps):
        context = task.context(state, float(gamma))
        # Re-seed each closed-loop step so M=1 is the exact first-candidate
        # subset of M>1 for the same model/temperature/scene/gamma rollout.
        generator = torch.Generator(device=task.device).manual_seed(
            int(seed) * 100_000 + int(step)
        )
        candidates = wrapped.sample(
            context,
            samples_per_step,
            generator,
            base_std=sampling_temperature,
        )
        sampled_counts.append(len(candidates))
        costs = []
        step_margins = []
        for candidate in candidates:
            plan = candidate.detach().cpu().numpy().reshape(-1, 3)
            predicted, _, _ = task._rollout_plan(
                np.asarray(state["x"], np.float32),
                np.asarray(state["previous_applied"], np.float32),
                plan,
            )
            costs.append(task.execution_cost_breakdown(
                env,
                predicted,
                plan,
                np.asarray(state["previous_raw"], np.float32),
            ))
            step_margins.append(float(nominal_hp_chain_margins(
                env,
                predicted[:2, :3],
                float(gamma),
                config.safemppi.sensing_range,
            )[0]))
        total_costs = [float(value["total"]) for value in costs]
        chosen = bounded_regret_step_margin_choice(
            total_costs,
            step_margins,
            execution_cost_band_fraction,
        )
        candidate = candidates[chosen]
        chosen_costs.append(float(costs[chosen]["total"]))
        chosen_step_margins.append(float(step_margins[chosen]))
        predicted, applied, dense = task._rollout_plan(
            np.asarray(state["x"], np.float32),
            np.asarray(state["previous_applied"], np.float32),
            candidate.detach().cpu().numpy().reshape(-1, 3),
        )
        updated = task.advance(state, candidate)
        if not np.allclose(updated["x"], predicted[1], atol=2.0e-6, rtol=0.0):
            raise RuntimeError("deployment advance disagrees with verifier dynamics")
        states.append(updated["x"].copy())
        controls.append(candidate[0].detach().cpu().numpy().copy())
        applied_controls.append(applied[0].copy())
        dense_steps.append(dense[0].copy())
        state = updated
        terminal = task.terminal(state)
        if terminal is not None:
            status = terminal
            break

    states_array = np.asarray(states, np.float32)
    controls_array = np.asarray(controls, np.float32).reshape(-1, 3)
    applied_array = np.asarray(applied_controls, np.float32).reshape(-1, 3)
    dense_array = np.asarray(dense_steps, np.float32).reshape(
        len(controls_array), config.safemppi.integration_substeps, 3,
    )
    dense_path = (
        np.concatenate([states_array[:1, :3], dense_array.reshape(-1, 3)])
        if len(dense_array) else states_array[:, :3]
    )
    clearance = env.obstacle_clearance(dense_path)
    finite = clearance[np.isfinite(clearance)]
    return {
        "status": status,
        "states": states_array,
        "controls": controls_array,
        "applied_controls": applied_array,
        "dense_steps": dense_array,
        "mode": None,
        "min_clearance_m": float(finite.min()) if len(finite) else None,
        "time_to_goal_s": (
            float(len(controls_array) * config.safemppi.dt)
            if status == "SUCCESS" else None
        ),
        "window_validity": reference_window_validity_fraction(
            _scene_config(config, spheres),
            states_array,
            dense_array,
            float(gamma),
        ),
        "sampling_temperature": float(sampling_temperature),
        "samples_per_step": int(samples_per_step),
        "mean_sampled_candidates": (
            float(np.mean(sampled_counts)) if sampled_counts else 0.0
        ),
        "mean_chosen_execution_cost": (
            float(np.mean(chosen_costs)) if chosen_costs else None
        ),
        "mean_chosen_step_margin": (
            float(np.mean(chosen_step_margins))
            if chosen_step_margins else None
        ),
        "execution_cost_band_fraction": float(
            execution_cost_band_fraction
        ),
    }


def _summary(rows):
    value = _summarize(rows)
    value["status_counts"] = {
        status: int(sum(row["status"] == status for row in rows))
        for status in ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
    }
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_provenance() -> dict:
    source_sha256 = {
        relative: _sha256(ROOT / relative)
        for relative in EVALUATION_SOURCE_FILES
    }
    aggregate = hashlib.sha256(json.dumps(
        source_sha256, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "source_file_sha256": source_sha256,
        "source_aggregate_sha256": aggregate,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _parse_rounds(raw: str | None, single: int | None) -> tuple[int, ...]:
    if raw is not None and single is not None:
        raise ValueError(
            "--checkpoint-round and --checkpoint-rounds are mutually exclusive"
        )
    if raw is None:
        values = (1 if single is None else int(single),)
    else:
        parsed = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lower_raw, upper_raw = token.split("-", 1)
                lower, upper = int(lower_raw), int(upper_raw)
                if lower < 0 or upper < lower:
                    raise ValueError(f"invalid checkpoint range {token!r}")
                parsed.extend(range(lower, upper + 1))
            else:
                parsed.append(int(token))
        values = tuple(sorted(set(parsed)))
    if not values or any(value < 0 for value in values):
        raise ValueError("checkpoint rounds must be nonnegative")
    return tuple(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_helios_arguments(parser)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--checkpoint-round", type=int)
    parser.add_argument(
        "--checkpoint-rounds",
        help="comma-separated rounds/ranges, e.g. 0-5",
    )
    parser.add_argument("--scene-bank-json", type=Path, required=True)
    parser.add_argument(
        "--evaluation-output", "--output", dest="evaluation_output",
        type=Path, required=True,
    )
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument(
        "--execution-clearance-exp-weight", type=float, default=15.0,
    )
    parser.add_argument(
        "--execution-clearance-target-m", type=float, default=0.6,
    )
    parser.add_argument(
        "--execution-clearance-exp-temperature", type=float, default=0.15,
    )
    parser.add_argument("--execution-taskspace-quadratic-weight", type=float, default=250.0)
    parser.add_argument("--execution-taskspace-quadratic-target-m", type=float, default=0.15)
    parser.add_argument("--execution-axis-cylinder-quadratic-weight", type=float, default=5.0)
    parser.add_argument("--execution-axis-cylinder-radius-m", type=float, default=1.1)
    parser.add_argument("--execution-control-weight", type=float, default=0.05)
    parser.add_argument(
        "--execution-obstacle-speed-weight", type=float, default=0.0,
    )
    parser.add_argument(
        "--execution-cost-band-fraction", type=float, default=0.0,
    )
    parser.add_argument("--seed", type=int, default=91000)
    args = parser.parse_args()

    if args.helios:
        return run_evaluation_on_helios(
            args,
            sys.argv[1:],
            Path(__file__).resolve().parents[1],
            script_name="evaluate_multisphere_min_cost_deployment.py",
        )

    try:
        checkpoint_rounds = _parse_rounds(
            args.checkpoint_rounds, args.checkpoint_round,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.evaluation_output.exists():
        raise FileExistsError(
            f"refusing to overwrite {args.evaluation_output}"
        )
    if args.episodes < 1 or args.samples_per_step < 1:
        parser.error("episodes and samples-per-step must be positive")
    if not np.isfinite(args.sampling_temperature) or args.sampling_temperature < 0.0:
        parser.error("sampling temperature must be finite and nonnegative")
    if (
        not np.isfinite(args.execution_clearance_exp_weight)
        or args.execution_clearance_exp_weight < 0.0
        or not np.isfinite(args.execution_clearance_target_m)
        or args.execution_clearance_target_m < 0.0
        or not np.isfinite(args.execution_clearance_exp_temperature)
        or args.execution_clearance_exp_temperature <= 0.0
    ):
        parser.error("invalid exponential obstacle-cost parameters")
    if (
        not np.isfinite(args.execution_obstacle_speed_weight)
        or args.execution_obstacle_speed_weight < 0.0
    ):
        parser.error("obstacle speed weight must be finite and nonnegative")
    if (
        not np.isfinite(args.execution_cost_band_fraction)
        or not 0.0 <= args.execution_cost_band_fraction <= 1.0
    ):
        parser.error("execution cost-band fraction must lie in [0,1]")
    config = load_config(args.expansion / "task_config_resolved.json")
    task_config_path = args.expansion / "task_config_resolved.json"
    task_config_payload = json.loads(task_config_path.read_text())
    scene_payload = json.loads(args.scene_bank_json.read_text())
    scenes = scene_payload["scene_bank"]["scenes"]
    expansion_manifest_path = args.expansion / "manifest.json"
    expansion_manifest_payload = json.loads(
        expansion_manifest_path.read_text()
    )
    expansion_scene_hashes = {
        str(row["scene_hash"])
        for row in expansion_manifest_payload.get("lab_scene_ledger", [])
        if isinstance(row, dict) and row.get("scene_hash")
    }
    evaluation_scene_hashes = {
        str(row["scene_hash"]) for row in scenes
    }
    overlap = sorted(expansion_scene_hashes & evaluation_scene_hashes)
    if overlap:
        parser.error(
            "fixed evaluation scene bank overlaps expansion collection: "
            + ", ".join(overlap)
        )
    if args.episodes > len(scenes):
        parser.error(
            f"requested {args.episodes} scenes but bank has only {len(scenes)}"
        )
    scenes = scenes[:args.episodes]
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
        execution_clearance_exp_weight=(
            args.execution_clearance_exp_weight
        ),
        execution_clearance_target_m=(
            args.execution_clearance_target_m
        ),
        execution_clearance_exp_temperature=(
            args.execution_clearance_exp_temperature
        ),
        execution_taskspace_quadratic_weight=(
            args.execution_taskspace_quadratic_weight
        ),
        execution_taskspace_quadratic_target_m=(
            args.execution_taskspace_quadratic_target_m
        ),
        execution_axis_cylinder_quadratic_weight=(
            args.execution_axis_cylinder_quadratic_weight
        ),
        execution_axis_cylinder_radius_m=(
            args.execution_axis_cylinder_radius_m
        ),
        execution_control_weight=args.execution_control_weight,
        execution_obstacle_speed_weight=(
            args.execution_obstacle_speed_weight
        ),
        paired_scene_rotation="start_goal_axis_180",
        paired_scene_seed=args.seed,
        paired_scene_pair_count=5,
        paired_scene_max_replacements_per_slot=1,
        fixed_scene_layout="none",
        scene_spec=scene_spec,
    )
    gammas = [float(value) for value in config.data.gammas]
    rows_by_round = {}
    summary_by_round = {}
    checkpoint_hashes = {}
    total = len(checkpoint_rounds) * len(gammas) * len(scenes)
    completed = 0
    for checkpoint_round in checkpoint_rounds:
        checkpoint_path = (
            args.expansion / f"checkpoint_{checkpoint_round:03d}.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        wrapped.policy.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_hashes[str(checkpoint_round)] = _sha256(checkpoint_path)
        rows = []
        for gamma in gammas:
            for scene in scenes:
                episode = int(scene["episode"])
                rollout_seed = args.seed + 37 * episode
                result = _rollout(
                    wrapped,
                    task,
                    config,
                    scene["spheres"],
                    gamma,
                    rollout_seed,
                    args.samples_per_step,
                    args.sampling_temperature,
                    args.execution_cost_band_fraction,
                )
                rows.append({
                    "round": int(checkpoint_round),
                    "gamma": gamma,
                    "episode": episode,
                    "rollout_seed": rollout_seed,
                    "scene_seed": int(scene["scene_seed"]),
                    "scene_hash": str(scene["scene_hash"]),
                    "obstacle_count": int(scene["obstacle_count"]),
                    "spheres": scene["spheres"],
                    **result,
                })
                completed += 1
                if completed % max(total // 20, 1) == 0 or completed == total:
                    print(f"[best-of-M] {completed}/{total}", flush=True)
        rows_by_round[str(checkpoint_round)] = rows
        summary_by_round[str(checkpoint_round)] = {
            "pooled": _summary(rows),
            "per_gamma": {
                f"{gamma:g}": _summary([
                    row for row in rows if row["gamma"] == gamma
                ])
                for gamma in gammas
            },
        }

    args.evaluation_output.mkdir(parents=True)
    slim = {
        round_i: [{
            key: value for key, value in row.items()
            if key not in {
                "states", "controls", "applied_controls", "dense_steps",
            }
        } for row in rows]
        for round_i, rows in rows_by_round.items()
    }
    expansion_manifest = expansion_manifest_path
    payload = {
        "status": "RAW_COST_RANKED_BEST_OF_M_DEPLOYMENT_COMPLETE",
        "canonical_raw_curve": False,
        "selection": (
            "bounded-regret hybrid among exactly M raw flow plans: shortlist "
            "by execution cost, then maximize first-step nominal H_P margin; "
            "verifier validity and progress are not computed or used"
        ),
        "cost": (
            "round-1 configured native obstacle/goal/smooth/progress/taskspace "
            "+ interior wall + axis-cylinder + control override"
        ),
        "execution_obstacle_exponential": {
            "weight": float(args.execution_clearance_exp_weight),
            "clearance_target_m": float(args.execution_clearance_target_m),
            "temperature_m": float(args.execution_clearance_exp_temperature),
            "formula": (
                "weight * mean_h exp((clearance_target_m - d_h) / "
                "temperature_m)"
            ),
            "scope": "raw cost ranking only; verifier unused",
        },
        "execution_obstacle_conditioned_speed": {
            "weight": float(args.execution_obstacle_speed_weight),
            "clearance_target_m": float(args.execution_clearance_target_m),
            "temperature_m": float(args.execution_clearance_exp_temperature),
            "formula": (
                "weight * mean_h sigmoid((clearance_target_m-clearance_h)/"
                "temperature_m) * ||predicted_velocity_h||^2"
            ),
        },
        "execution_margin_hybrid": {
            "cost_band_fraction": float(
                args.execution_cost_band_fraction
            ),
            "formula": (
                "J <= J_min + fraction*(J_max-J_min), then maximum "
                "first-step nominal H_P margin with lower-cost tie break"
            ),
            "nominal_margin_is_not_verifier_label": True,
        },
        "execution_taskspace_quadratic": {
            "weight": float(args.execution_taskspace_quadratic_weight),
            "target_m": float(args.execution_taskspace_quadratic_target_m),
        },
        "execution_axis_cylinder_quadratic": {
            "weight": float(args.execution_axis_cylinder_quadratic_weight),
            "radius_m": float(args.execution_axis_cylinder_radius_m),
        },
        "execution_control": {
            "weight": float(args.execution_control_weight),
        },
        "selection_uses_verifier_or_progress": False,
        "outcomes": "post-execution closed-loop evaluation only",
        "checkpoint_rounds": list(checkpoint_rounds),
        "checkpoint_round": (
            int(checkpoint_rounds[0]) if len(checkpoint_rounds) == 1 else None
        ),
        "samples_per_step": int(args.samples_per_step),
        "sampling_temperature": float(args.sampling_temperature),
        "sigma_tilt_used": False,
        "NFE": 16,
        "scene_bank_source": str(args.scene_bank_json.resolve()),
        "scene_bank": scene_payload["scene_bank"],
        "scene_bank_disjointness": {
            "expansion_unique_scene_count": len(expansion_scene_hashes),
            "evaluation_unique_scene_count": len(evaluation_scene_hashes),
            "overlap_count": 0,
            "expansion_scene_hashes_sha256": hashlib.sha256(
                json.dumps(
                    sorted(expansion_scene_hashes),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "evaluation_scene_hashes_sha256": hashlib.sha256(
                json.dumps(
                    sorted(evaluation_scene_hashes),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "artifact_binding": {
            "checkpoint_sha256_by_round": checkpoint_hashes,
            "expansion_manifest_sha256": _sha256(expansion_manifest),
            "task_config_sha256": _sha256(task_config_path),
            "task_config_semantic_sha256": hashlib.sha256(
                json.dumps(
                    task_config_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        },
        "software_provenance": _software_provenance(),
        "summary": summary_by_round,
        "rows": slim,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    (args.evaluation_output / "deployment_eval.json").write_text(serialized)
    (args.evaluation_output / "raw_eval.json").write_text(serialized)
    if args.save_raw_trajectories:
        torch.save(rows_by_round, args.evaluation_output / "raw_trajectories.pt")
    print(json.dumps(summary_by_round, indent=2), flush=True)


if __name__ == "__main__":
    main()
