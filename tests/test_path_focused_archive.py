import errno
import json
from pathlib import Path

import numpy as np
import pytest

from safe_mppi.config import load_config
from safe_mppi.lab_reference_flow_task import lab_reference_demo_windows
from safe_mppi.lab_visual_flow import LAB_RADIAL_VISUAL_SCHEMA
from safe_mppi.path_focused_archive import (
    assemble_path_focused_quota_archives,
)
from safe_mppi.path_focused_collection import (
    collect_path_focused_success_quota,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_cylinders_path_v2.json"


def _fake_controller(*args, **kwargs):
    del args, kwargs
    return object()


def _episode(env, controller, gamma, seed, rollout_dynamics):
    del controller, rollout_dynamics
    scene_index = int(seed) // 1009
    success = not (np.isclose(gamma, 0.3) and scene_index == 0)
    controls = np.zeros((10, 3), np.float32)
    states = np.zeros((11, 6), np.float32)
    states[0] = env.start
    endpoint = env.goal if success else (env.start[:3] + env.goal) / 2.0
    states[:, :3] = np.linspace(env.start[:3], endpoint, 11)
    row = {
        "gamma": float(gamma),
        "seed": int(seed),
        "success": bool(success),
        "collision": False,
        "taskspace_violation": False,
        "steps": 10,
        "time_to_goal_s": 1.0 if success else None,
        "min_clearance_m": 0.2,
        "mean_feasible_fraction": 1.0,
        "minimum_feasible_fraction": 1.0,
        "all_infeasible": False,
        "all_infeasible_steps": 0,
        "minimum_controller_one_step_slack": 0.1,
        "minimum_online_one_step_slack": 0.1,
        "mean_plan_time_ms": 1.0,
        "mean_control_variation_mps2": 0.0,
        "mean_applied_control_variation_mps2": 0.0,
        "max_abs_control_mps2": 0.0,
        "max_abs_applied_control_mps2": 0.0,
        "peak_speed_mps": 0.0,
        "peak_vertical_speed_mps": 0.0,
        "deployment_speed_compatible": True,
        "fraction_internal_states_below_z_bias_plane": 0.0,
        "mean_internal_z_m": float(env.start[2]),
    }
    arrays = {
        "states": states,
        "controls": controls,
        "executed_controls": controls.copy(),
        "dense_positions": states[:, :3].copy(),
        "gamma": np.asarray(gamma, np.float32),
        "seed": np.asarray(seed, np.int64),
    }
    return row, arrays


def _source(tmp_path, gamma, *, target=2, centroid_gain=None):
    output = tmp_path / f"source_g{gamma:g}_{centroid_gain}"
    collect_path_focused_success_quota(
        load_config(CONFIG),
        output,
        target_successes_per_gamma=target,
        max_scene_count=4,
        gammas=(gamma,),
        centroid_gain=centroid_gain,
        episode_runner=_episode,
        controller_factory=_fake_controller,
    )
    return output


def test_assembler_unions_disjoint_gammas_and_preserves_all_attempts(
    tmp_path,
):
    source_a = _source(tmp_path, 0.1)
    source_b = _source(tmp_path, 0.3)
    output = tmp_path / "assembled"
    manifest = assemble_path_focused_quota_archives(
        [source_a, source_b], output,
    )

    assert manifest["status"] == "COMPLETE_EXACT_SUCCESS_QUOTA"
    assert manifest["gammas"] == [0.1, 0.3]
    assert manifest["accepted_counts_by_gamma"] == {"0.1": 2, "0.3": 2}
    assert len(manifest["runs"]) == 4
    assert len(manifest["attempts"]) == 5
    assert sum(not row["accepted"] for row in manifest["attempts"]) == 1
    assert len(manifest["scene_bank"]["scenes"]) == 3

    assembly = json.loads((output / "assembly_contract.json").read_text())
    assert len(assembly["trajectory_files"]) == 4
    assert (
        assembly["publication"]["hardlinks"]
        + assembly["publication"]["copies"]
        == 4
    )
    contexts, plans, metadata, config = lab_reference_demo_windows(
        output,
        validate_archive=False,
        context_schema=LAB_RADIAL_VISUAL_SCHEMA,
    )
    assert contexts.shape[0] == plans.shape[0] == len(metadata) == 4
    assert config.data.gammas == (0.1, 0.3)


def test_assembler_copy_fallback_and_refuses_overwrite(tmp_path, monkeypatch):
    source = _source(tmp_path, 1.0, target=1)

    def no_hardlinks(*args, **kwargs):
        del args, kwargs
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr("safe_mppi.path_focused_archive.os.link", no_hardlinks)
    output = tmp_path / "assembled_copy"
    assemble_path_focused_quota_archives([source], output)
    assembly = json.loads((output / "assembly_contract.json").read_text())
    assert assembly["publication"] == {"hardlinks": 0, "copies": 1}
    declaration = assembly["trajectory_files"][0]
    assert (output / declaration["file"]).is_file()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assemble_path_focused_quota_archives([source], output)


def test_assembler_rejects_overlapping_gamma_and_recipe_mismatch(tmp_path):
    source = _source(tmp_path, 0.1)
    duplicate = tmp_path / "duplicate"
    # A second source needs a distinct directory but may share hardlinked data.
    import shutil

    shutil.copytree(source, duplicate)
    with pytest.raises(ValueError, match="appears in multiple sources"):
        assemble_path_focused_quota_archives(
            [source, duplicate], tmp_path / "duplicate_output",
        )

    different_recipe = _source(tmp_path, 0.3, centroid_gain=0.2)
    with pytest.raises(ValueError, match="resolved recipes differ"):
        assemble_path_focused_quota_archives(
            [source, different_recipe], tmp_path / "recipe_output",
        )


def test_assembler_rejects_tampered_quota_provenance(tmp_path):
    source = _source(tmp_path, 0.1, target=1)
    contract_path = source / "quota_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["target_successes_per_gamma"] = 9
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    with pytest.raises(ValueError, match="target mismatch"):
        assemble_path_focused_quota_archives(
            [source], tmp_path / "tampered_output",
        )
