from pathlib import Path
import subprocess
import sys

from safe_mppi.config import load_config
from safe_mppi.lab_clutter_expansion import sphere_scene_spec_from_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lab_clutter_spheres_stage2_three_v2.json"


def test_stage2_config_samples_exactly_three_spheres():
    scene_spec = sphere_scene_spec_from_config(load_config(CONFIG))

    assert scene_spec.spec.count_min == 3
    assert scene_spec.spec.count_max == 3
    assert scene_spec.max_count == 3


def test_truncated_stage2_cli_fails_before_writing_output(tmp_path):
    output = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_sphere_clutter_expansion.py"),
            "--pretrain-dir", str(tmp_path / "missing_pretrain"),
            "--lab-task-config", str(CONFIG),
            "--output", str(output),
            "--beta", "1.0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "line-continuation backslash was omitted" in result.stderr
    assert not output.exists()
