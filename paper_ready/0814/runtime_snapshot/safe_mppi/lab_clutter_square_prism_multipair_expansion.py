"""Isolated multi-pair adapter for one deterministic square-prism scene."""
from __future__ import annotations

from typing import Any

import numpy as np

from .lab_clutter_expansion import scene_sha256
from .lab_clutter_pre2_expansion import (
    _gamma_key,
    rotate_points_180_about_start_goal_axis,
)
from .lab_clutter_pre2_multipair_expansion import (
    LabClutterPre2MultiPairExpansionTask,
)


class LabClutterSquarePrismMultiPairExpansionTask(
    LabClutterPre2MultiPairExpansionTask,
):
    """Reuse one declared fixed geometry across gamma without hash rejection.

    The standard domain-randomized adapter requires every gamma to have a
    distinct scene hash. This stress probe intentionally holds geometry fixed
    so gamma is the only changed conditioning variable.
    """

    def _generate_pair_slot(
        self,
        gamma: float,
        pair_slot: int,
        version: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.paired_scene_pair_count != 1 or int(pair_slot) != 0:
            raise ValueError("square-prism probe supports exactly one pair slot")
        gamma_key = _gamma_key(gamma)
        key = (self.expansion_round, gamma_key, 0, int(version))
        existing = self._multipair_scene_cache.get(key)
        if existing is not None:
            return existing

        randomization = self.config.raw.get("scene_randomization", {})
        if randomization.get("admission_mode") != (
            "deterministic_geometry_only_no_expert_conditioning"
        ):
            raise ValueError(
                "square-prism adapter requires its isolated deterministic config"
            )
        scene_seed = int(np.random.SeedSequence([
            self.paired_scene_seed & 0xFFFFFFFF,
            self.expansion_round & 0xFFFFFFFF,
            gamma_key & 0xFFFFFFFF,
            int(version) & 0xFFFFFFFF,
            0x53515250,
        ]).generate_state(1, dtype=np.uint32)[0])
        source = self.scene_spec.sample(self.env, scene_seed)
        rotated = source.copy()
        rotated[:, :3] = rotate_points_180_about_start_goal_axis(
            source[:, :3], self.env.start, self.env.goal,
        ).astype(np.float32)
        source = self.scene_spec.validate(self.env, source)
        rotated = self.scene_spec.validate(self.env, rotated)
        source_hash = scene_sha256(self._environment(source), source)
        rotated_hash = scene_sha256(self._environment(rotated), rotated)
        if source_hash == rotated_hash:
            raise RuntimeError("square-prism source and mirror are identical")

        recovered = rotated.copy()
        recovered[:, :3] = rotate_points_180_about_start_goal_axis(
            rotated[:, :3], self.env.start, self.env.goal,
        ).astype(np.float32)
        recovered = self.scene_spec.validate(self.env, recovered)
        if not np.allclose(recovered, source, rtol=0.0, atol=2.0e-6):
            raise RuntimeError("square-prism rotation failed involution")

        metadata = {
            "paired_scene_id": (
                f"r{self.expansion_round:03d}_g{float(gamma):.9g}_"
                f"square_prism_v{int(version):02d}"
            ),
            "paired_scene_pair_slot": 0,
            "paired_scene_label_base": 0,
            "paired_scene_version": int(version),
            "paired_scene_seed": scene_seed,
            "paired_scene_proposal_index": 0,
            "paired_source_scene_hash": source_hash,
            "paired_rotated_scene_hash": rotated_hash,
            "paired_rotation": "start_goal_axis_180",
            "cross_gamma_scene_reuse": True,
        }
        cached = (source, rotated, metadata)
        self._multipair_scene_cache[key] = cached
        return cached
