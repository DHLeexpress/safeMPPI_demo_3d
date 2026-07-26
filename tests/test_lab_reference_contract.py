import json
from pathlib import Path

import numpy as np
import pytest

from flow_deployment.lab_reference_contract import (
    GovernedReference,
    load_governed_reference,
)
from safe_mppi.config import load_config


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"


def test_accepted_archive_matches_future_policy_export_contract():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    config = load_config(ARCHIVE / "resolved_config.json")
    row = manifest["runs"][0]
    reference = load_governed_reference(
        ARCHIVE / row["file"],
        gamma=row["gamma"],
        seed=row["seed"],
        integration_substeps=config.safemppi.integration_substeps,
        action_limit=config.safemppi.demo_u_max,
    )
    assert reference.raw_controls.shape == reference.executed_controls.shape
    assert len(reference.dense_positions) == (
        1
        + len(reference.executed_controls)
        * config.safemppi.integration_substeps
    )


def test_governed_reference_rejects_misaligned_or_nonfinite_exports():
    with pytest.raises(ValueError, match="misaligned"):
        GovernedReference(
            dense_positions=np.zeros((10, 3), np.float32),
            executed_controls=np.zeros((1, 3), np.float32),
            raw_controls=None,
            gamma=0.3,
            seed=0,
            source="test",
        ).validate(integration_substeps=10, action_limit=0.3)

    bad = np.zeros((11, 3), np.float32)
    bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GovernedReference(
            dense_positions=bad,
            executed_controls=np.zeros((1, 3), np.float32),
            raw_controls=None,
            gamma=0.3,
            seed=0,
            source="test",
        ).validate(integration_substeps=10, action_limit=0.3)
