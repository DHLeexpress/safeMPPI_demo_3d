import json
from pathlib import Path
import subprocess
import sys

import pytest

from safe_mppi.config import load_config
from safe_mppi.path_focused_clutter import PathFocusedClutterSpec
from scripts.generate_path_focused_clutter_scenes import (
    _override_legacy_transverse_std,
    _sampling_payload,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_path_focused_clutter_scenes.py"
LEGACY_CYLINDER_CONFIG = ROOT / "configs/lab_clutter_cylinders_path_v2.json"
MIDPOINT_CYLINDER_CONFIG = (
    ROOT / "configs/lab_clutter_cylinders_path_midpoint_uniform_v2.json"
)
MIDPOINT_SPHERE_CONFIG = (
    ROOT / "configs/lab_clutter_spheres_path_midpoint_uniform_v2.json"
)


def test_preview_cli_renders_midpoint_uniform_configs_with_honest_payload(tmp_path):
    output = tmp_path / "preview"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--scenes",
            "2",
            "--seed",
            "41",
            "--cylinder-config",
            str(MIDPOINT_CYLINDER_CONFIG),
            "--sphere-config",
            str(MIDPOINT_SPHERE_CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "scene_preview.png").stat().st_size > 0
    assert (output / "scene_preview.pdf").stat().st_size > 0
    payload = json.loads((output / "scene_bank.json").read_text())
    cylinder = payload["sampling"]
    sphere = payload["sampling"]["sphere_sampling"]
    assert cylinder["transverse_distribution"] == (
        "uniform_symmetric_with_longitudinal_halfwidth"
    )
    assert cylinder["transverse_halfwidth_scale_m"] == 3.0
    assert "transverse_std_m" not in cylinder
    assert sphere["transverse_distribution"] == (
        "uniform_symmetric_with_longitudinal_halfwidth"
    )
    assert sphere["sphere_z_distribution"] == "truncated_normal"
    assert sphere["sphere_z_mean_m"] == 0.9
    assert sphere["sphere_z_std_m"] == 0.4
    assert sphere["sphere_z_center_range_m"] == [0.754, 1.646]
    assert len(payload["cylinder_scenes"]) == 2
    assert len(payload["sphere_scenes"]) == 2


def test_transverse_std_override_remains_legacy_only():
    legacy = load_config(LEGACY_CYLINDER_CONFIG)
    overridden = _override_legacy_transverse_std(legacy, 0.43)
    payload = _sampling_payload(PathFocusedClutterSpec.from_config(overridden))
    assert payload["transverse_distribution"] == "normal"
    assert payload["transverse_std_m"] == 0.43

    midpoint = load_config(MIDPOINT_CYLINDER_CONFIG)
    with pytest.raises(ValueError, match="only applies to the legacy"):
        _override_legacy_transverse_std(midpoint, 0.43)
