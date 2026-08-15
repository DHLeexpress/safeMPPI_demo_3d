"""Raw-command lab task adapter for the Minhyuk-frame flow policy.

The 10-D policy predicts the pre-smoothing acceleration command. It does not
model governor state, plant state, or trajectory tracking. The external
simulator/deployment harness applies command smoothing and reference integration
exactly once.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .ball_flow_task import PLAN_H, build_context
from .ball_flow_theta import theta_name, trajectory_crossing_theta
from .config import ExperimentConfig, ObstacleConfig, load_config
from .environment import ReferenceGovernor, TaskEnvironment
from .verifier_polytope import certify_window


LAB_REFERENCE_CONTEXT_DIM = 10
LAB_ROUTE_MODES = ("below", "above", "left", "right")
LAB_RAW_CONTEXT_SCHEMA = "lab_raw10_v1"


def stratified_window_starts(
    window_count: int,
    max_windows_per_trajectory: int | None,
) -> np.ndarray:
    """Select deterministic, endpoint-inclusive starts across one trajectory."""
    window_count = int(window_count)
    if window_count < 0:
        raise ValueError("window_count must be nonnegative")
    if max_windows_per_trajectory is None:
        return np.arange(window_count, dtype=np.int64)
    maximum = int(max_windows_per_trajectory)
    if maximum < 2:
        raise ValueError(
            "max_windows_per_trajectory must be at least 2 so both trajectory "
            "endpoints remain represented"
        )
    if window_count <= maximum:
        return np.arange(window_count, dtype=np.int64)
    starts = np.rint(
        np.linspace(0, window_count - 1, maximum)
    ).astype(np.int64)
    if (
        len(starts) != maximum
        or starts[0] != 0
        or starts[-1] != window_count - 1
        or np.any(np.diff(starts) <= 0)
    ):
        raise RuntimeError("stratified window selection contract failed")
    return starts


def _run_environment(
    config: ExperimentConfig,
    data,
    row: dict,
) -> TaskEnvironment:
    """Reconstruct the exact per-run scene when an archive randomizes it."""
    if "spheres" not in data.files and "cylinders" not in data.files:
        if row.get("scene_hash") is not None or row.get("scene_id") is not None:
            raise ValueError(
                f"randomized run {row['file']} is missing obstacle arrays"
            )
        return TaskEnvironment(config)
    spheres = np.asarray(
        data["spheres"] if "spheres" in data.files else (),
        np.float32,
    ).reshape(-1, 4)
    cylinders = np.asarray(
        data["cylinders"] if "cylinders" in data.files else (),
        np.float32,
    ).reshape(-1, 3)
    from .lab_clutter import obstacle_scene_hash

    observed_hash = obstacle_scene_hash(spheres, cylinders)
    expected_hash = str(row.get("scene_hash", observed_hash))
    if observed_hash != expected_hash:
        raise ValueError(
            f"per-run obstacle hash mismatch for {row['file']}"
        )
    if "scene_hash" in data.files and str(data["scene_hash"].item()) != expected_hash:
        raise ValueError(
            f"NPZ scene hash mismatch for {row['file']}"
        )
    scalar_fields = (
        ("scene_id", str),
        ("scene_index", int),
        ("scene_seed", int),
        ("seed", int),
    )
    for key, cast in scalar_fields:
        if key in data.files and row.get(key) is not None:
            observed = cast(data[key].item())
            expected = cast(row.get(f"source_{key}", row[key]))
            if observed != expected:
                raise ValueError(
                    f"NPZ {key} mismatch for {row['file']}: "
                    f"{observed!r} != {expected!r}"
                )
    if "gamma" in data.files and not np.isclose(
        float(data["gamma"].item()),
        float(row["gamma"]),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError(f"NPZ gamma mismatch for {row['file']}")
    for key, width in (("spheres", 4), ("cylinders", 3)):
        if row.get(key) is not None:
            expected = np.asarray(row[key], np.float32).reshape(-1, width)
            observed = spheres if key == "spheres" else cylinders
            if not np.array_equal(observed, expected):
                raise ValueError(
                    f"manifest {key} mismatch for {row['file']}"
                )
    obstacles = ObstacleConfig(
        spheres=tuple(tuple(map(float, values)) for values in spheres),
        cylinders=tuple(tuple(map(float, values)) for values in cylinders),
    )
    return TaskEnvironment(replace(config, obstacles=obstacles))


def _raw_history_before(
    controls: np.ndarray,
    start: int,
) -> np.ndarray:
    from .lab_visual_flow import (
        LAB_VISUAL_HISTORY_LENGTH,
        LAB_VISUAL_HISTORY_STEP_DIM,
    )

    history = np.zeros(
        (LAB_VISUAL_HISTORY_LENGTH, LAB_VISUAL_HISTORY_STEP_DIM),
        np.float32,
    )
    values = np.asarray(controls, np.float32).reshape(-1, 3)
    tail = values[max(0, int(start) - LAB_VISUAL_HISTORY_LENGTH):int(start)]
    if len(tail):
        history[-len(tail):, :3] = tail
        history[-len(tail):, 3] = 1.0
    return history


def _append_raw_history(
    history: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    updated = np.roll(np.asarray(history, np.float32), -1, axis=0)
    updated[-1, :3] = np.asarray(command, np.float32).reshape(3)
    updated[-1, 3] = 1.0
    return updated


def _raw_and_applied_memories(
    controls: np.ndarray,
    mppi_config,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact pre-step raw/applied memory for every trajectory state."""
    values = np.asarray(controls, np.float32).reshape(-1, 3)
    previous_raw = np.zeros((len(values) + 1, 3), np.float32)
    previous_applied = np.zeros((len(values) + 1, 3), np.float32)
    applied = np.zeros(3, np.float32)
    smooth = float(mppi_config.deployment_accel_smooth)
    limit = float(mppi_config.demo_u_max)
    for step, command in enumerate(values):
        raw = np.clip(command, -limit, limit).astype(np.float32)
        applied = (
            smooth * raw + (1.0 - smooth) * applied
        ).astype(np.float32)
        previous_raw[step + 1] = raw
        previous_applied[step + 1] = applied
    return previous_raw, previous_applied


def policy_context(
    policy,
    env: TaskEnvironment,
    state6: np.ndarray,
    gamma: float,
    raw_history: np.ndarray | None = None,
    previous_raw: np.ndarray | None = None,
    previous_applied: np.ndarray | None = None,
) -> np.ndarray:
    """Build the external context declared by a lab policy."""
    schema = getattr(policy, "context_schema", LAB_RAW_CONTEXT_SCHEMA)
    if schema == LAB_RAW_CONTEXT_SCHEMA:
        return build_context(env, state6, gamma)
    from .lab_visual_flow import (
        LAB_HP100_EXACT_MEMORY_SCHEMA,
        LAB_HP100_HISTORY_SCHEMA,
        LAB_HP100_SCHEMA,
        LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
        LAB_RADIAL_VISUAL_SCHEMA,
        LAB_VISUAL_HISTORY_SCHEMA,
        LAB_VISUAL_SCHEMA,
        build_nonuniform_radial_visual_history_context,
        build_nonuniform_radial_visual_context,
        build_uniform_hp100_exact_memory_context,
        build_uniform_hp100_history_context,
        build_uniform_hp100_context,
        build_visual_context,
        build_visual_history_context,
    )
    if schema == LAB_VISUAL_SCHEMA:
        return build_visual_context(env, state6, gamma)
    if schema == LAB_RADIAL_VISUAL_SCHEMA:
        return build_nonuniform_radial_visual_context(env, state6, gamma)
    if schema == LAB_HP100_SCHEMA:
        return build_uniform_hp100_context(env, state6, gamma)
    if schema == LAB_HP100_EXACT_MEMORY_SCHEMA:
        if previous_raw is None or previous_applied is None:
            raise ValueError(
                "uniform H_P exact-memory policy requires previous raw and "
                "applied commands"
            )
        return build_uniform_hp100_exact_memory_context(
            env,
            state6,
            gamma,
            previous_raw,
            previous_applied,
        )
    if schema == LAB_HP100_HISTORY_SCHEMA:
        if raw_history is None:
            raise ValueError(
                "uniform H_P history policy context requires past raw commands"
            )
        return build_uniform_hp100_history_context(
            env, state6, gamma, raw_history,
        )
    if schema == LAB_RADIAL_VISUAL_HISTORY_SCHEMA:
        if raw_history is None:
            raise ValueError(
                "radial visual-history policy context requires past raw "
                "commands"
            )
        return build_nonuniform_radial_visual_history_context(
            env,
            state6,
            gamma,
            raw_history,
        )
    if schema == LAB_VISUAL_HISTORY_SCHEMA:
        if raw_history is None:
            raise ValueError(
                "visual-history policy context requires past raw commands"
            )
        return build_visual_history_context(
            env, state6, gamma, raw_history,
        )
    raise ValueError(f"unsupported lab policy context schema {schema!r}")


def lab_reference_demo_windows(
    run_dir: str | Path,
    *,
    validate_archive: bool = True,
    context_schema: str = LAB_RAW_CONTEXT_SCHEMA,
    max_windows_per_trajectory: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], ExperimentConfig]:
    """Load declared contexts and pre-smoothing raw-command windows.

    An optional positive cap selects deterministic endpoint-inclusive strata
    before contexts are constructed.  Thus every declared trajectory remains
    represented without paying the context-construction cost for discarded
    windows.  ``None`` preserves the historical all-window behavior.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = load_config(run_dir / "resolved_config.json")
    contexts, plans, metadata = [], [], []
    scene_bank = manifest.get("scene_bank")
    scene_rows = {}
    if scene_bank is not None:
        scene_rows = {
            str(row["scene_id"]): row
            for row in scene_bank["scenes"]
        }
        if len(scene_rows) != len(scene_bank["scenes"]):
            raise ValueError("scene bank contains duplicate scene_id values")
        unconditioned = bool(
            manifest.get("sampling_distribution", {}).get(
                "unconditioned_geometry", False,
            )
        )
        if not unconditioned:
            expected_gammas = set(map(float, config.data.gammas))
            for scene_id in scene_rows:
                observed = {
                    float(row["gamma"]) for row in manifest["runs"]
                    if str(row.get("scene_id")) == scene_id
                }
                if observed != expected_gammas:
                    raise ValueError(
                        f"scene {scene_id} does not contain every configured gamma"
                    )

    for row in manifest["runs"]:
        if not row.get("accepted", False):
            raise ValueError("lab archive manifest contains an unaccepted run")
        data = np.load(run_dir / row["file"])
        if scene_rows:
            scene_id = str(row.get("scene_id"))
            if scene_id not in scene_rows:
                raise ValueError(
                    f"run references undeclared scene {scene_id!r}"
                )
            declared = scene_rows[scene_id]
            if str(row.get("scene_hash")) != str(declared["scene_hash"]):
                raise ValueError(
                    f"run scene hash disagrees with scene bank for {scene_id}"
                )
        env = _run_environment(config, data, row)
        states = np.asarray(data["states"], np.float32)
        controls = np.asarray(data["controls"], np.float32).reshape(-1, 3)
        if len(states) != len(controls) + 1:
            raise ValueError("lab reference states and controls are misaligned")
        if not np.allclose(
            states[0], env.start, atol=1.0e-6, rtol=0.0,
        ):
            raise ValueError(
                f"stored start state does not match config for {row['file']}"
            )
        if validate_archive:
            governor = ReferenceGovernor(config.safemppi)
            state = states[0].copy()
            replay = [state.copy()]
            for control in controls:
                state, _, _ = governor.step(state, control)
                replay.append(state.copy())
            if not np.allclose(
                np.asarray(replay), states, atol=1.0e-6, rtol=0.0,
            ):
                raise ValueError(
                    "stored states do not match externally governed raw commands"
                )

        gamma = float(row["gamma"])
        from .lab_visual_flow import LAB_HP100_EXACT_MEMORY_SCHEMA
        previous_raw = previous_applied = None
        if context_schema == LAB_HP100_EXACT_MEMORY_SCHEMA:
            previous_raw, previous_applied = _raw_and_applied_memories(
                controls, config.safemppi,
            )
        available_window_count = max(0, len(controls) - PLAN_H + 1)
        selected_starts = stratified_window_starts(
            available_window_count,
            max_windows_per_trajectory,
        )
        for start in selected_starts.tolist():
            if context_schema == LAB_RAW_CONTEXT_SCHEMA:
                context = build_context(env, states[start], gamma)
            else:
                from .lab_visual_flow import (
                    LAB_HP100_EXACT_MEMORY_SCHEMA,
                    LAB_HP100_HISTORY_SCHEMA,
                    LAB_HP100_SCHEMA,
                    LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
                    LAB_RADIAL_VISUAL_SCHEMA,
                    LAB_VISUAL_HISTORY_SCHEMA,
                    LAB_VISUAL_SCHEMA,
                    build_nonuniform_radial_visual_history_context,
                    build_nonuniform_radial_visual_context,
                    build_uniform_hp100_exact_memory_context,
                    build_uniform_hp100_history_context,
                    build_uniform_hp100_context,
                    build_visual_context,
                    build_visual_history_context,
                )
                if context_schema == LAB_VISUAL_SCHEMA:
                    context = build_visual_context(
                        env, states[start], gamma,
                    )
                elif context_schema == LAB_RADIAL_VISUAL_SCHEMA:
                    context = build_nonuniform_radial_visual_context(
                        env, states[start], gamma,
                    )
                elif context_schema == LAB_HP100_SCHEMA:
                    context = build_uniform_hp100_context(
                        env, states[start], gamma,
                    )
                elif context_schema == LAB_HP100_EXACT_MEMORY_SCHEMA:
                    context = build_uniform_hp100_exact_memory_context(
                        env,
                        states[start],
                        gamma,
                        previous_raw[start],
                        previous_applied[start],
                    )
                elif context_schema == LAB_HP100_HISTORY_SCHEMA:
                    context = build_uniform_hp100_history_context(
                        env,
                        states[start],
                        gamma,
                        _raw_history_before(controls, start),
                    )
                elif context_schema == LAB_RADIAL_VISUAL_HISTORY_SCHEMA:
                    context = build_nonuniform_radial_visual_history_context(
                        env,
                        states[start],
                        gamma,
                        _raw_history_before(controls, start),
                    )
                elif context_schema == LAB_VISUAL_HISTORY_SCHEMA:
                    context = build_visual_history_context(
                        env,
                        states[start],
                        gamma,
                        _raw_history_before(controls, start),
                    )
                else:
                    raise ValueError(
                        f"unsupported lab context schema {context_schema!r}"
                    )
            contexts.append(context)
            plans.append(controls[start:start + PLAN_H])
            metadata.append({
                "gamma": gamma,
                "seed": int(row["seed"]),
                "t": int(start),
                "file": str(row["file"]),
                "scene_id": row.get("scene_id"),
                "scene_index": row.get("scene_index"),
                "scene_hash": row.get("scene_hash"),
                "pair_id": row.get("pair_id"),
                "pair_member": row.get("pair_member"),
                "trajectory_available_windows": available_window_count,
                "trajectory_selected_windows": len(selected_starts),
            })

    return (
        np.asarray(contexts, np.float32),
        np.asarray(plans, np.float32),
        metadata,
        config,
    )


def reference_window_validity_fraction(
    config: ExperimentConfig,
    states: np.ndarray,
    dense_steps: np.ndarray,
    gamma: float,
) -> float:
    """Window-level in-bounds, collision-free, GREEN validity."""
    states = np.asarray(states, np.float32)
    dense_steps = np.asarray(dense_steps, np.float32)
    if (
        len(states) < 2
        or dense_steps.ndim != 3
        or dense_steps.shape[0] != len(states) - 1
        or dense_steps.shape[2] != 3
    ):
        return 0.0
    env = TaskEnvironment(config)
    valid = 0
    count = len(states) - 1
    for start in range(count):
        stop = min(start + PLAN_H, count)
        dense = np.concatenate([
            states[start:start + 1, :3],
            dense_steps[start:stop].reshape(-1, 3),
        ])
        clearance = env.obstacle_clearance(dense)
        collision_free = bool(
            np.isinf(clearance).all()
            or (
                np.isfinite(clearance).any()
                and float(clearance.min()) > 0.0
            )
        )
        certified, _ = certify_window(
            states[start:stop + 1, :3],
            env.spheres,
            env.cylinders,
            float(gamma),
            config.safemppi.sensing_range,
        )
        valid += int(
            env.inside_taskspace(dense).all()
            and collision_free
            and certified
        )
    return valid / count


@torch.no_grad()
def raw_reference_rollout(
    policy,
    config: ExperimentConfig,
    gamma: float,
    seed: int,
    *,
    device: str | torch.device = "cpu",
    max_steps: int | None = None,
    sampling_temperature: float = 1.0,
) -> dict:
    """Raw rollout with external smoothing/integration applied exactly once."""
    if not np.isfinite(sampling_temperature) or sampling_temperature < 0.0:
        raise ValueError("sampling_temperature must be finite and nonnegative")
    env = TaskEnvironment(config)
    governor = ReferenceGovernor(config.safemppi)
    state = env.start.copy()
    raw_history = _raw_history_before(np.empty((0, 3)), 0)
    states = [state.copy()]
    controls = []
    applied_controls = []
    dense_steps = []
    status = "TIMEOUT"

    for step in range(max_steps or config.taskspace.max_steps):
        context = torch.from_numpy(
            policy_context(
                policy,
                env,
                state,
                float(gamma),
                raw_history=raw_history,
                previous_raw=(
                    np.clip(
                        controls[-1],
                        -float(config.safemppi.demo_u_max),
                        float(config.safemppi.demo_u_max),
                    ).astype(np.float32)
                    if controls else np.zeros(3, np.float32)
                ),
                previous_applied=governor.previous_applied,
            )
        ).to(device)
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(int(seed) * 100_000 + step)
        plan = policy.sample(
            context,
            1,
            generator,
            base_std=float(sampling_temperature),
        )[0].detach().cpu().numpy()
        control = plan.reshape(PLAN_H, 3)[0]
        raw_history = _append_raw_history(raw_history, control)
        state, applied, dense = governor.step(state, control)
        states.append(state.copy())
        controls.append(control.copy())
        applied_controls.append(applied.copy())
        dense_steps.append(dense.copy())
        clearance = env.obstacle_clearance(dense)
        if np.isfinite(clearance).any() and float(clearance.min()) < 0.0:
            status = "COLLISION"
            break
        if not env.inside_taskspace(dense).all():
            status = "OOB"
            break
        if env.reached(state[:3]):
            status = "SUCCESS"
            break

    states_array = np.asarray(states, np.float32)
    controls_array = np.asarray(controls, np.float32).reshape(-1, 3)
    applied_array = np.asarray(applied_controls, np.float32).reshape(-1, 3)
    dense_array = np.asarray(dense_steps, np.float32).reshape(
        len(controls_array), config.safemppi.integration_substeps, 3,
    )
    dense_path = (
        np.concatenate([states_array[:1, :3], dense_array.reshape(-1, 3)])
        if len(controls_array)
        else states_array[:, :3]
    )
    clearance = env.obstacle_clearance(dense_path)
    finite = clearance[np.isfinite(clearance)]
    crossing = (
        trajectory_crossing_theta(env, dense_path)
        if len(env.spheres) == 1 else None
    )
    return {
        "status": status,
        "states": states_array,
        "controls": controls_array,
        "applied_controls": applied_array,
        "dense_steps": dense_array,
        "mode": theta_name(crossing),
        "min_clearance_m": float(finite.min()) if len(finite) else None,
        "time_to_goal_s": (
            float(len(controls_array) * config.safemppi.dt)
            if status == "SUCCESS"
            else None
        ),
        "window_validity": reference_window_validity_fraction(
            config,
            states_array,
            dense_array,
            float(gamma),
        ),
        "sampling_temperature": float(sampling_temperature),
    }


class LabReferenceFlowController:
    """deploy_sim-compatible raw flow controller with configurable base scale."""

    def __init__(
        self,
        policy,
        env: TaskEnvironment,
        sampling_temperature: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        schema = getattr(policy, "context_schema", LAB_RAW_CONTEXT_SCHEMA)
        if schema == LAB_RAW_CONTEXT_SCHEMA:
            if int(policy.context_dim) != LAB_REFERENCE_CONTEXT_DIM:
                raise ValueError("raw lab deployment requires the 10-D context")
        else:
            from .lab_visual_flow import (
                LAB_HP100_EXACT_MEMORY_PACKED_DIM,
                LAB_HP100_EXACT_MEMORY_SCHEMA,
                LAB_HP100_HISTORY_PACKED_DIM,
                LAB_HP100_HISTORY_SCHEMA,
                LAB_HP100_PACKED_DIM,
                LAB_HP100_SCHEMA,
                LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM,
                LAB_RADIAL_VISUAL_HISTORY_SCHEMA,
                LAB_RADIAL_VISUAL_PACKED_DIM,
                LAB_RADIAL_VISUAL_SCHEMA,
                LAB_VISUAL_HISTORY_PACKED_DIM,
                LAB_VISUAL_HISTORY_SCHEMA,
                LAB_VISUAL_PACKED_DIM,
                LAB_VISUAL_SCHEMA,
            )
            expected = {
                LAB_VISUAL_SCHEMA: LAB_VISUAL_PACKED_DIM,
                LAB_HP100_SCHEMA: LAB_HP100_PACKED_DIM,
                LAB_HP100_EXACT_MEMORY_SCHEMA: (
                    LAB_HP100_EXACT_MEMORY_PACKED_DIM
                ),
                LAB_HP100_HISTORY_SCHEMA: LAB_HP100_HISTORY_PACKED_DIM,
                LAB_RADIAL_VISUAL_SCHEMA: LAB_RADIAL_VISUAL_PACKED_DIM,
                LAB_RADIAL_VISUAL_HISTORY_SCHEMA: (
                    LAB_RADIAL_VISUAL_HISTORY_PACKED_DIM
                ),
                LAB_VISUAL_HISTORY_SCHEMA: LAB_VISUAL_HISTORY_PACKED_DIM,
            }
            if schema not in expected or int(policy.context_dim) != expected[schema]:
                raise ValueError("unknown lab visual context contract")
        if tuple(policy.plan_shape) != (PLAN_H, 3):
            raise ValueError("lab deployment requires H=10, 3-D plans")
        if not np.isfinite(sampling_temperature) or sampling_temperature < 0.0:
            raise ValueError("sampling_temperature must be finite and nonnegative")
        self.policy = policy.to(device).eval()
        self.env = env
        self.device = torch.device(device)
        self.sampling_temperature = float(sampling_temperature)
        self.trace = []
        self.raw_history = _raw_history_before(np.empty((0, 3)), 0)
        self.previous_raw = np.zeros(3, np.float32)
        self.previous_applied = np.zeros(3, np.float32)

    def reset(self):
        self.trace = []
        self.raw_history = _raw_history_before(np.empty((0, 3)), 0)
        self.previous_raw = np.zeros(3, np.float32)
        self.previous_applied = np.zeros(3, np.float32)

    @torch.no_grad()
    def plan(self, state, goal, gamma, seed=0):
        state = np.asarray(state, np.float32).reshape(6)
        goal = np.asarray(goal, np.float32).reshape(3)
        if not np.allclose(goal, self.env.goal, atol=1.0e-6, rtol=0.0):
            raise ValueError("controller goal does not match its lab environment")
        context = torch.from_numpy(
            policy_context(
                self.policy,
                self.env,
                state,
                float(gamma),
                raw_history=self.raw_history,
                previous_raw=self.previous_raw,
                previous_applied=self.previous_applied,
            )
        ).to(self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        plan = self.policy.sample(
            context,
            1,
            generator,
            base_std=self.sampling_temperature,
        )[0]
        action = plan[0].detach().cpu().numpy().astype(np.float32)
        self.raw_history = _append_raw_history(self.raw_history, action)
        limited_action = np.clip(
            action,
            -float(self.env.mppi.demo_u_max),
            float(self.env.mppi.demo_u_max),
        ).astype(np.float32)
        smooth = float(self.env.mppi.deployment_accel_smooth)
        self.previous_applied = (
            smooth * limited_action
            + (1.0 - smooth) * self.previous_applied
        ).astype(np.float32)
        self.previous_raw = limited_action
        self.trace.append({
            "seed": int(seed),
            "gamma": float(gamma),
            "state": state.copy(),
            "action": action.copy(),
        })
        return action, {
            "sampling_temperature": self.sampling_temperature,
            "raw_plan": plan.detach().cpu().numpy(),
        }
