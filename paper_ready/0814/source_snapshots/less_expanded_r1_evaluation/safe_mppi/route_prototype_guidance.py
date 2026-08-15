"""Route-persistent candidate selection from successful action-window traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROUTE_PROTOTYPE_FORMAT = "safemppi_route_action_prototypes_v1"


def prototype_group_key(gamma: float, route: int) -> str:
    return f"gamma={float(gamma):.9g},route={int(route)}"


@dataclass(frozen=True)
class RoutePrototype:
    trajectory_id: str
    states: torch.Tensor
    action_windows: torch.Tensor
    valid_horizons: torch.Tensor


class RoutePrototypeGuide:
    """Match verified candidate windows to a progress-aligned success trace."""

    def __init__(
        self,
        groups: dict[str, list[RoutePrototype]],
        *,
        device: torch.device,
        velocity_weight: float = 0.5,
        cosine_weight: float = 0.25,
        time_decay: float = 0.9,
    ):
        if not np.isfinite(velocity_weight) or velocity_weight < 0.0:
            raise ValueError("velocity_weight must be finite and nonnegative")
        if not np.isfinite(cosine_weight) or cosine_weight < 0.0:
            raise ValueError("cosine_weight must be finite and nonnegative")
        if not np.isfinite(time_decay) or not 0.0 < time_decay <= 1.0:
            raise ValueError("time_decay must lie in (0, 1]")
        self.device = torch.device(device)
        self.velocity_weight = float(velocity_weight)
        self.cosine_weight = float(cosine_weight)
        self.time_decay = float(time_decay)
        self.groups = {
            key: [
                RoutePrototype(
                    trajectory_id=prototype.trajectory_id,
                    states=prototype.states.to(self.device),
                    action_windows=prototype.action_windows.to(self.device),
                    valid_horizons=prototype.valid_horizons.to(self.device),
                )
                for prototype in prototypes
            ]
            for key, prototypes in groups.items()
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device,
        velocity_weight: float = 0.5,
        cosine_weight: float = 0.25,
        time_decay: float = 0.9,
    ) -> "RoutePrototypeGuide":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != ROUTE_PROTOTYPE_FORMAT:
            raise ValueError(f"unsupported route prototype format in {path}")
        groups: dict[str, list[RoutePrototype]] = {}
        for key, rows in payload.get("groups", {}).items():
            prototypes = []
            for row in rows:
                states = torch.as_tensor(row["states"], dtype=torch.float32)
                windows = torch.as_tensor(
                    row["action_windows"], dtype=torch.float32,
                )
                valid_horizons = torch.as_tensor(
                    row["valid_horizons"], dtype=torch.long,
                )
                if states.ndim != 2 or states.shape[1] != 6:
                    raise ValueError(
                        f"route prototype {row['trajectory_id']} states must be [N,6]"
                    )
                if windows.ndim != 3 or windows.shape[0] != states.shape[0]:
                    raise ValueError(
                        f"route prototype {row['trajectory_id']} windows must be [N,H,A]"
                    )
                if (
                    valid_horizons.shape != (states.shape[0],)
                    or torch.any(valid_horizons < 1)
                    or torch.any(valid_horizons > windows.shape[1])
                ):
                    raise ValueError(
                        f"route prototype {row['trajectory_id']} has invalid horizons"
                    )
                prototypes.append(RoutePrototype(
                    trajectory_id=str(row["trajectory_id"]),
                    states=states,
                    action_windows=windows,
                    valid_horizons=valid_horizons,
                ))
            if prototypes:
                groups[str(key)] = prototypes
        if not groups:
            raise ValueError("route prototype bank has no nonempty groups")
        return cls(
            groups,
            device=device,
            velocity_weight=velocity_weight,
            cosine_weight=cosine_weight,
            time_decay=time_decay,
        )

    def group_size(self, gamma: float, route: int) -> int:
        return len(self.groups.get(prototype_group_key(gamma, route), ()))

    def target_window(
        self,
        *,
        episode: dict[str, Any],
        gamma: float,
        route: int,
    ) -> tuple[torch.Tensor | None, int, dict[str, float | int | str]]:
        """Return the progress-matched executed-action suffix for one route."""
        key = prototype_group_key(gamma, route)
        prototypes = self.groups.get(key)
        if not prototypes:
            return None, 0, {"missing_group": key}
        assignment_key = (key, int(episode["retry_batch"]), int(episode["replica"]))
        if episode.get("_route_prototype_assignment") != assignment_key:
            prototype_index = (
                int(episode["retry_batch"]) + int(episode["replica"])
            ) % len(prototypes)
            episode["_route_prototype_assignment"] = assignment_key
            episode["_route_prototype_index"] = prototype_index
            episode["_route_prototype_cursor"] = 0
        prototype_index = int(episode["_route_prototype_index"])
        prototype = prototypes[prototype_index]

        state = torch.as_tensor(
            np.asarray(episode["state"]["x"], np.float32)[:6],
            dtype=torch.float32,
            device=self.device,
        )
        cursor = int(episode.get("_route_prototype_cursor", 0))
        cursor = min(cursor, len(prototype.states) - 1)
        state_delta = prototype.states[cursor:] - state
        state_delta = state_delta.clone()
        state_delta[:, 3:] *= self.velocity_weight
        matched = cursor + int(torch.argmin(torch.sum(state_delta.square(), dim=1)))
        episode["_route_prototype_cursor"] = matched
        valid_horizon = int(prototype.valid_horizons[matched])
        return prototype.action_windows[matched], valid_horizon, {
            "prototype": prototype.trajectory_id,
            "prototype_index": prototype_index,
            "cursor": matched,
        }

    def choose(
        self,
        *,
        episode: dict[str, Any],
        gamma: float,
        route: int,
        candidates: torch.Tensor,
        eligible: list[int],
    ) -> tuple[int | None, dict[str, float | int | str]]:
        if not eligible:
            return None, {"empty_eligible": 1}
        target, valid_horizon, diagnostics = self.target_window(
            episode=episode, gamma=gamma, route=route,
        )
        if target is None:
            return None, diagnostics
        candidate_indices = torch.as_tensor(
            eligible, dtype=torch.long, device=candidates.device,
        )
        plans = candidates.index_select(0, candidate_indices)
        horizon = min(
            plans.shape[1], target.shape[0],
            valid_horizon,
        )
        if horizon < 1:
            return None, {"empty_horizon": 1}
        plans = plans[:, :horizon]
        target = target[:horizon].to(plans.device)
        weights = torch.pow(
            torch.tensor(self.time_decay, device=plans.device),
            torch.arange(horizon, device=plans.device),
        )
        weights = weights / weights.sum()
        mse = torch.sum(
            (plans - target.unsqueeze(0)).square()
            * weights[None, :, None],
            dim=(1, 2),
        ) / plans.shape[2]
        weighted_plans = plans * weights.sqrt()[None, :, None]
        weighted_target = target * weights.sqrt()[:, None]
        plan_flat = weighted_plans.flatten(1)
        target_flat = weighted_target.flatten()
        cosine = torch.nn.functional.cosine_similarity(
            plan_flat,
            target_flat.unsqueeze(0).expand_as(plan_flat),
            dim=1,
            eps=1.0e-8,
        )
        scores = mse + self.cosine_weight * (1.0 - cosine)
        local = int(torch.argmin(scores))
        return int(eligible[local]), {
            **diagnostics,
            "score": float(scores[local].detach().cpu()),
            "cosine": float(cosine[local].detach().cpu()),
        }


def blend_route_proposals(
    candidates: torch.Tensor,
    target: torch.Tensor,
    valid_horizon: int,
    strengths: tuple[float, ...],
) -> torch.Tensor:
    """Replace the tail candidate slots with convex route-window proposals."""
    if candidates.ndim != 3 or target.ndim != 2:
        raise ValueError("candidates and target must be [K,H,A] and [H,A]")
    if candidates.shape[2] != target.shape[1]:
        raise ValueError("candidate and target action dimensions differ")
    if len(strengths) > len(candidates):
        raise ValueError("more proposal strengths than candidate slots")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0
           for value in strengths):
        raise ValueError("proposal strengths must lie in [0,1]")
    horizon = min(int(valid_horizon), candidates.shape[1], target.shape[0])
    if horizon < 1 or not strengths:
        return candidates
    proposed = candidates.clone()
    target = target[:horizon].to(
        device=candidates.device, dtype=candidates.dtype,
    )
    first = len(candidates) - len(strengths)
    for offset, strength in enumerate(strengths):
        index = first + offset
        proposed[index, :horizon] = (
            (1.0 - float(strength)) * candidates[index, :horizon]
            + float(strength) * target
        )
    return proposed
