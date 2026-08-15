"""Isolated exact multi-pair PRE2 clutter expansion task."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .environment import TaskEnvironment
from .lab_clutter_expansion import scene_sha256
from .lab_clutter_pre2_expansion import (
    LabClutterPre2ExpansionTask,
    _gamma_key,
    load_lab_clutter_pre2_expansion_policy,
    rotate_points_180_about_start_goal_axis,
)


_REQUESTED_BOUNDS = np.asarray([
    [-2.5, 1.3],
    [-2.1, 1.8],
    [0.1, 1.7],
], np.float64)
_LEGACY_CONSTRUCTION_BOUNDS = np.asarray([
    [-2.5, 1.3],
    [-1.7, 1.8],
    [0.1, 1.7],
], np.float64)


class LabClutterPre2MultiPairExpansionTask(
    LabClutterPre2ExpansionTask
):
    """Assign one deterministic exact scene pair to every quota pair slot."""

    def __init__(
        self,
        config,
        *args,
        paired_scene_pair_count: int = 5,
        paired_scene_max_replacements_per_slot: int = 1,
        **kwargs,
    ):
        if int(paired_scene_pair_count) < 1:
            raise ValueError("paired_scene_pair_count must be positive")
        if int(paired_scene_max_replacements_per_slot) < 0:
            raise ValueError(
                "paired_scene_max_replacements_per_slot must be nonnegative"
            )
        if kwargs.get("paired_scene_rotation", "none") != (
            "start_goal_axis_180"
        ):
            raise ValueError(
                "isolated multi-pair expansion requires "
                "paired_scene_rotation='start_goal_axis_180'"
            )
        if kwargs.get("fixed_scene_layout", "none") != "none":
            raise ValueError(
                "isolated multi-pair expansion does not support a fixed scene"
            )
        requested_bounds = np.asarray(config.taskspace.bounds, np.float64)
        if not np.allclose(
            requested_bounds, _REQUESTED_BOUNDS, rtol=0.0, atol=1.0e-9,
        ):
            raise ValueError(
                "isolated multi-pair expansion requires taskspace bounds "
                "x=[-2.5,1.3], y=[-2.1,1.8], z=[0.1,1.7]"
            )
        raw_safety = config.raw.get("safety")
        if not isinstance(raw_safety, dict):
            raise ValueError(
                "isolated multi-pair expansion requires an explicit safety box"
            )
        try:
            raw_safety_bounds = np.stack([
                np.asarray(raw_safety["safe_min"], np.float64),
                np.asarray(raw_safety["safe_max"], np.float64),
            ], axis=1)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "isolated multi-pair safety requires 3-D safe_min/safe_max"
            ) from error
        if (
            raw_safety_bounds.shape != (3, 2)
            or not np.allclose(
                raw_safety_bounds,
                _REQUESTED_BOUNDS,
                rtol=0.0,
                atol=1.0e-9,
            )
        ):
            raise ValueError(
                "isolated multi-pair raw safety bounds must exactly match "
                "the requested taskspace"
            )

        # The shared clutter adapter intentionally enforces its historical
        # y=[-1.7,1.8] box.  Construct only that inherited machinery through a
        # frozen dataclass proxy, then restore the requested box before any
        # scene is sampled, packed into H_P context, or sent to the verifier.
        legacy_taskspace = replace(
            config.taskspace,
            origin=tuple(_LEGACY_CONSTRUCTION_BOUNDS[:, 0]),
            size=tuple(np.diff(_LEGACY_CONSTRUCTION_BOUNDS, axis=1)[:, 0]),
        )
        construction_config = replace(config, taskspace=legacy_taskspace)
        super().__init__(construction_config, *args, **kwargs)
        if self._fixed_spheres is not None:
            raise RuntimeError(
                "multi-pair construction unexpectedly materialized a fixed scene"
            )
        self.config = config
        self.env = TaskEnvironment(config)
        delta = self.env.goal - self.env.start[:3]
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-12:
            raise ValueError("multi-pair start and goal must be distinct")
        self.forward = delta / length
        self.reference_z = 0.5 * float(
            self.env.start[2] + self.env.goal[2]
        )
        self._axis_unit = self.forward.copy()
        self.paired_scene_pair_count = int(paired_scene_pair_count)
        self.paired_scene_max_replacements_per_slot = int(
            paired_scene_max_replacements_per_slot
        )
        self._multipair_scene_cache: dict[
            tuple[int, int, int, int],
            tuple[np.ndarray, np.ndarray, dict[str, Any]],
        ] = {}
        self._multipair_current_versions: dict[tuple[int, int, int], int] = {}
        self.paired_scene_replacement_ledger: list[dict[str, Any]] = []

    def _pair_version_key(
        self, gamma: float, pair_slot: int,
    ) -> tuple[int, int, int]:
        return self.expansion_round, _gamma_key(gamma), int(pair_slot)

    def paired_scene_current_version(
        self, gamma: float, pair_slot: int,
    ) -> int:
        if not 0 <= int(pair_slot) < self.paired_scene_pair_count:
            raise ValueError("paired scene pair slot is out of range")
        return self._multipair_current_versions.get(
            self._pair_version_key(gamma, pair_slot), 0,
        )

    def _generate_pair_slot(
        self,
        gamma: float,
        pair_slot: int,
        version: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        gamma_key = _gamma_key(gamma)
        if int(version) < 0:
            raise ValueError("paired scene version must be nonnegative")
        key = (
            self.expansion_round, gamma_key, int(pair_slot), int(version),
        )
        existing = self._multipair_scene_cache.get(key)
        if existing is not None:
            return existing
        used_hashes = {
            str(metadata[hash_name])
            for (round_i, _gamma_i, _slot, _version), (_, _, metadata)
            in self._multipair_scene_cache.items()
            if round_i == self.expansion_round
            for hash_name in (
                "paired_source_scene_hash",
                "paired_rotated_scene_hash",
            )
        }
        pair_seed = int(np.random.SeedSequence([
            self.paired_scene_seed & 0xFFFFFFFF,
            self.expansion_round & 0xFFFFFFFF,
            gamma_key & 0xFFFFFFFF,
            int(pair_slot) & 0xFFFFFFFF,
            int(version) & 0xFFFFFFFF,
            0x4D504149,
        ]).generate_state(1, dtype=np.uint32)[0])
        for proposal_index in range(self.paired_scene_max_proposals):
            scene_seed = int(np.random.SeedSequence([
                pair_seed,
                proposal_index,
            ]).generate_state(1, dtype=np.uint32)[0])
            source = self.scene_spec.sample(self.env, scene_seed)
            rotated = source.copy()
            rotated[:, :3] = rotate_points_180_about_start_goal_axis(
                source[:, :3], self.env.start, self.env.goal,
            ).astype(np.float32)
            try:
                source = self.scene_spec.validate(self.env, source)
                rotated = self.scene_spec.validate(self.env, rotated)
            except ValueError:
                continue
            source_hash = scene_sha256(self._environment(source), source)
            rotated_hash = scene_sha256(self._environment(rotated), rotated)
            if (
                source_hash == rotated_hash
                or source_hash in used_hashes
                or rotated_hash in used_hashes
            ):
                continue
            recovered = rotated.copy()
            recovered[:, :3] = rotate_points_180_about_start_goal_axis(
                rotated[:, :3], self.env.start, self.env.goal,
            ).astype(np.float32)
            recovered = self.scene_spec.validate(self.env, recovered)
            if not np.allclose(
                recovered, source, rtol=0.0, atol=2.0e-6,
            ):
                raise RuntimeError(
                    "multi-pair sphere rotation failed involution"
                )
            metadata = {
                "paired_scene_id": (
                    f"r{self.expansion_round:03d}_"
                    f"g{float(gamma):.9g}_p{int(pair_slot):02d}_"
                    f"v{int(version):02d}"
                ),
                "paired_scene_pair_slot": int(pair_slot),
                "paired_scene_label_base": 2 * int(pair_slot),
                "paired_scene_version": int(version),
                "paired_scene_seed": scene_seed,
                "paired_scene_proposal_index": proposal_index,
                "paired_source_scene_hash": source_hash,
                "paired_rotated_scene_hash": rotated_hash,
                "paired_rotation": "start_goal_axis_180",
            }
            cached = (source, rotated, metadata)
            self._multipair_scene_cache[key] = cached
            return cached
        raise RuntimeError(
            "could not sample a distinct start-goal-axis scene pair for "
            f"round={self.expansion_round}, gamma={float(gamma):g}, "
            f"pair_slot={int(pair_slot)}, version={int(version)} within "
            f"{self.paired_scene_max_proposals} proposals"
        )

    def _multipair_spheres(
        self,
        gamma: float,
        pair_slot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not 0 <= int(pair_slot) < self.paired_scene_pair_count:
            raise ValueError("paired scene pair slot is out of range")
        version = self.paired_scene_current_version(gamma, pair_slot)
        # Generate lower slots first so collision rejection and resulting hashes
        # are independent of which episode happens to request a slot first.
        for slot in range(int(pair_slot) + 1):
            self._generate_pair_slot(
                gamma, slot, self.paired_scene_current_version(gamma, slot),
            )
        return self._multipair_scene_cache[
            (
                self.expansion_round, _gamma_key(gamma), int(pair_slot),
                version,
            )
        ]

    def replace_paired_scene_slot(
        self,
        gamma: float,
        pair_slot: int,
    ) -> dict[str, Any]:
        """Atomically advance one round/gamma slot to a fresh exact pair."""
        slot = int(pair_slot)
        old_version = self.paired_scene_current_version(gamma, slot)
        if old_version >= self.paired_scene_max_replacements_per_slot:
            raise RuntimeError(
                "paired scene replacement limit exhausted for "
                f"gamma={float(gamma):g}, slot={slot}"
            )
        old_source, old_rotated, old_metadata = self._multipair_spheres(
            gamma, slot,
        )
        del old_source, old_rotated
        new_version = old_version + 1
        new_source, new_rotated, new_metadata = self._generate_pair_slot(
            gamma, slot, new_version,
        )
        del new_source, new_rotated
        self._multipair_current_versions[
            self._pair_version_key(gamma, slot)
        ] = new_version
        record = {
            "round": int(self.expansion_round),
            "gamma": float(gamma),
            "pair_slot": slot,
            "old_version": int(old_version),
            "new_version": int(new_version),
            "old_scene_id": str(old_metadata["paired_scene_id"]),
            "new_scene_id": str(new_metadata["paired_scene_id"]),
            "old_source_scene_hash": str(
                old_metadata["paired_source_scene_hash"]
            ),
            "old_rotated_scene_hash": str(
                old_metadata["paired_rotated_scene_hash"]
            ),
            "new_source_scene_hash": str(
                new_metadata["paired_source_scene_hash"]
            ),
            "new_rotated_scene_hash": str(
                new_metadata["paired_rotated_scene_hash"]
            ),
        }
        self.paired_scene_replacement_ledger.append(dict(record))
        return record

    def reset(self, gamma: float, episode: int, seed: int) -> dict[str, Any]:
        if self.paired_scene_rotation == "none":
            return super().reset(gamma, episode, seed)
        label_count = 2 * self.paired_scene_pair_count
        label = int(episode) % label_count
        pair_slot, member = divmod(label, 2)
        source, rotated, pair_metadata = self._multipair_spheres(
            gamma, pair_slot,
        )
        metadata = {
            **pair_metadata,
            "paired_scene_label": int(label),
            "paired_scene_member": int(member),
            "paired_scene_member_name": (
                "original" if member == 0 else "axis_180"
            ),
            "base_scene_seed": int(seed),
        }
        return self._new_state(
            gamma=gamma,
            episode=episode,
            spheres=source if member == 0 else rotated,
            scene_seed=int(pair_metadata["paired_scene_seed"]),
            metadata=metadata,
        )

    def scene_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        metadata = super().scene_metadata(state)
        for name in (
            "paired_scene_pair_slot",
            "paired_scene_label_base",
            "paired_scene_label",
            "paired_scene_version",
        ):
            if name in state:
                metadata[name] = int(state[name])
        return metadata

    def advance(self, state, candidate: torch.Tensor):
        updated = super().advance(state, candidate)
        for name in (
            "paired_scene_pair_slot",
            "paired_scene_label_base",
            "paired_scene_label",
            "paired_scene_version",
        ):
            if name in state:
                updated[name] = state[name]
        return updated

    def successful_trajectory_mode(self, executed_states) -> int | None:
        """Expose the stable pair-slot/member label used by exact quota."""
        if self.paired_scene_rotation == "none":
            return None
        if not executed_states:
            return None
        labels = {
            int(state["paired_scene_label"])
            for state in executed_states
            if "paired_scene_label" in state
        }
        if len(labels) != 1:
            raise RuntimeError(
                "successful multi-pair trajectory lost a stable quota label"
            )
        label = labels.pop()
        if not 0 <= label < 2 * self.paired_scene_pair_count:
            raise RuntimeError("successful multi-pair label is out of range")
        return label

    def successful_trajectory_pair_identity(
        self, executed_states,
    ) -> dict[str, int] | None:
        """Return the immutable slot/label/version identity for one success."""
        if not executed_states:
            return None
        identities = {
            (
                int(state["paired_scene_pair_slot"]),
                int(state["paired_scene_label"]),
                int(state["paired_scene_version"]),
            )
            for state in executed_states
        }
        if len(identities) != 1:
            raise RuntimeError(
                "successful multi-pair trajectory changed slot/label/version"
            )
        slot, label, version = identities.pop()
        if label // 2 != slot:
            raise RuntimeError("successful multi-pair identity is inconsistent")
        return {
            "pair_slot": slot,
            "label": label,
            "version": version,
        }


__all__ = [
    "LabClutterPre2MultiPairExpansionTask",
    "load_lab_clutter_pre2_expansion_policy",
]
