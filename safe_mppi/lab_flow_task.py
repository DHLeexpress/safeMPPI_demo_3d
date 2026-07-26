"""Minhyuk-lab data adapter for a future raw-command flow policy.

The deployment reference governor is stateful.  A policy that predicts the raw
acceleration command therefore needs the previous applied acceleration in
addition to the original 10-D ball context.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ball_flow_task import PLAN_H, build_context
from .config import ExperimentConfig, load_config
from .environment import ReferenceGovernor, TaskEnvironment


LAB_CONTEXT_DIM = 13


def build_lab_context(
    env: TaskEnvironment,
    state6: np.ndarray,
    gamma: float,
    previous_applied: np.ndarray,
) -> np.ndarray:
    """Return ``[ball_context_10, previous_applied_acceleration_3]``."""
    previous_applied = np.asarray(previous_applied, np.float32).reshape(3)
    return np.concatenate([
        build_context(env, state6, gamma),
        previous_applied,
    ]).astype(np.float32)


def governed_plan_states(
    env: TaskEnvironment,
    state6: np.ndarray,
    raw_plan: np.ndarray,
    previous_applied: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate raw commands through the exact lab reference recurrence."""
    governor = ReferenceGovernor(env.mppi)
    governor.previous_applied = np.asarray(
        previous_applied, np.float32,
    ).reshape(3).copy()
    state = np.asarray(state6, np.float32).reshape(6).copy()
    states = [state.copy()]
    applied_controls = []
    dense_steps = []
    for raw_control in np.asarray(raw_plan, np.float32).reshape(-1, 3):
        state, applied, dense = governor.step(state, raw_control)
        states.append(state.copy())
        applied_controls.append(applied.copy())
        dense_steps.append(dense)
    return (
        np.asarray(states, np.float32),
        np.asarray(applied_controls, np.float32),
        np.concatenate(dense_steps, axis=0),
    )


def lab_demo_windows(
    run_dir: str | Path,
    *,
    validate_archive: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], ExperimentConfig]:
    """Load governed demonstrations as raw-command H-step training windows.

    The returned plans are the raw SafeMPPI commands.  The previous applied
    acceleration is appended to every context, making the smoothing recurrence
    Markov.  This loader intentionally rejects legacy or mixed archives.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = load_config(run_dir / "resolved_config.json")
    if config.data.rollout_dynamics != "minhyuk_reference_governor":
        raise ValueError("lab_demo_windows requires a Minhyuk-governed archive")

    env = TaskEnvironment(config)
    contexts, plans, meta = [], [], []
    for row in manifest["runs"]:
        if not row.get("accepted", False):
            raise ValueError("lab archive manifest contains an unaccepted run")
        data = np.load(run_dir / row["file"])
        states = np.asarray(data["states"], np.float32)
        raw_controls = np.asarray(data["controls"], np.float32).reshape(-1, 3)
        applied_controls = np.asarray(
            data["executed_controls"], np.float32,
        ).reshape(-1, 3)
        if len(states) != len(raw_controls) + 1:
            raise ValueError("lab archive states and raw controls are misaligned")
        if applied_controls.shape != raw_controls.shape:
            raise ValueError("lab archive raw and applied controls are misaligned")

        if validate_archive:
            replay_states, replay_applied, _ = governed_plan_states(
                env,
                states[0],
                raw_controls,
                np.zeros(3, np.float32),
            )
            if not np.allclose(replay_states, states, atol=1.0e-6, rtol=0.0):
                raise ValueError("stored states do not match the Minhyuk governor")
            if not np.allclose(
                replay_applied, applied_controls, atol=1.0e-6, rtol=0.0,
            ):
                raise ValueError(
                    "stored applied controls do not match the Minhyuk governor"
                )

        gamma = float(row["gamma"])
        for start in range(len(raw_controls) - PLAN_H + 1):
            previous_applied = (
                np.zeros(3, np.float32)
                if start == 0 else applied_controls[start - 1]
            )
            contexts.append(build_lab_context(
                env, states[start], gamma, previous_applied,
            ))
            plans.append(raw_controls[start:start + PLAN_H])
            meta.append({
                "gamma": gamma,
                "seed": int(row["seed"]),
                "t": int(start),
                "previous_applied": previous_applied.tolist(),
            })

    return (
        np.asarray(contexts, np.float32),
        np.asarray(plans, np.float32),
        meta,
        config,
    )
