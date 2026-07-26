"""Export contract between a future lab flow policy and plant tracking.

The future policy must generate raw H-step controls, apply the stateful lab
reference governor exactly once, and export the resulting governed reference.
The calibrated plant consumes that frozen export; it does not call the policy
or governor online.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class GovernedReference:
    dense_positions: np.ndarray
    executed_controls: np.ndarray
    raw_controls: np.ndarray | None
    gamma: float
    seed: int
    source: str

    def validate(
        self,
        *,
        integration_substeps: int,
        action_limit: float,
    ) -> "GovernedReference":
        dense = np.asarray(self.dense_positions, np.float32)
        applied = np.asarray(self.executed_controls, np.float32)
        raw = (
            None
            if self.raw_controls is None
            else np.asarray(self.raw_controls, np.float32)
        )
        if dense.ndim != 2 or dense.shape[1:] != (3,):
            raise ValueError("dense_positions must have shape [1+N*S,3]")
        if applied.ndim != 2 or applied.shape[1:] != (3,):
            raise ValueError("executed_controls must have shape [N,3]")
        if len(dense) != 1 + len(applied) * integration_substeps:
            raise ValueError("dense positions and executed controls are misaligned")
        if raw is not None and raw.shape != applied.shape:
            raise ValueError("raw and executed controls must have the same shape")
        arrays = [dense, applied] + ([] if raw is None else [raw])
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("governed reference contains non-finite values")
        if float(np.max(np.abs(applied), initial=0.0)) > action_limit + 1.0e-6:
            raise ValueError("executed controls exceed the declared action limit")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must lie in (0,1]")
        return self


class LabReferenceGenerator(Protocol):
    """Placeholder implemented later by a pretrained/expanded lab flow model."""

    def generate(self, *, gamma: float, seed: int) -> GovernedReference:
        ...


def load_governed_reference(
    path: str | Path,
    *,
    gamma: float,
    seed: int,
    integration_substeps: int,
    action_limit: float,
) -> GovernedReference:
    """Load either an accepted SafeMPPI export or a future policy export."""
    path = Path(path)
    data = np.load(path)
    required = {"dense_positions", "executed_controls"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"governed reference is missing {missing}")
    reference = GovernedReference(
        dense_positions=np.asarray(data["dense_positions"], np.float32),
        executed_controls=np.asarray(data["executed_controls"], np.float32),
        raw_controls=(
            np.asarray(data["controls"], np.float32)
            if "controls" in data.files else None
        ),
        gamma=float(gamma),
        seed=int(seed),
        source=str(path.resolve()),
    )
    return reference.validate(
        integration_substeps=integration_substeps,
        action_limit=action_limit,
    )
