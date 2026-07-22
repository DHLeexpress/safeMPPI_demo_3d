"""Small conditional flow-matching policy; dimensions are supplied by the task adapter."""
from __future__ import annotations

import math

import torch
from torch import nn


class ConditionalFlowMLP(nn.Module):
    def __init__(self, context_dim: int, plan_shape: tuple[int, ...], hidden: int = 96,
                 representation_dim: int = 32, control_limit: float | None = None):
        super().__init__()
        self.context_dim = int(context_dim)
        self.plan_shape = tuple(int(value) for value in plan_shape)
        self.plan_dim = math.prod(self.plan_shape)
        self.control_limit = control_limit
        self.trunk = nn.Sequential(
            nn.Linear(self.plan_dim + self.context_dim + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, representation_dim), nn.SiLU(),
        )
        self.head = nn.Linear(representation_dim, self.plan_dim)

    @staticmethod
    def _time(t: torch.Tensor) -> torch.Tensor:
        frequency = torch.arange(1, 5, device=t.device, dtype=t.dtype)[None]
        angle = 2.0 * math.pi * t[:, None] * frequency
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)

    def _features(self, x: torch.Tensor, t: torch.Tensor,
                  context: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([x, context, self._time(t)], dim=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x, t, context))

    def cfm_loss(self, contexts: torch.Tensor, candidates: torch.Tensor,
                 reduction: str = "none") -> torch.Tensor:
        target = candidates.reshape(len(candidates), self.plan_dim)
        base = torch.randn_like(target)
        t = torch.rand(len(target), device=target.device)
        point = (1.0 - t[:, None]) * base + t[:, None] * target
        per_sample = (self(point, t, contexts) - (target - base)).square().mean(dim=1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'none' or 'mean'")

    @torch.no_grad()
    def sample(self, context: torch.Tensor, count: int,
               generator: torch.Generator) -> torch.Tensor:
        context = context.reshape(1, -1).expand(count, -1)
        x = torch.randn(count, self.plan_dim, device=context.device, generator=generator)
        nfe = 8
        for index in range(nfe):
            t = torch.full((count,), index / nfe, device=x.device)
            x = x + self(x, t, context) / nfe
        output = x.reshape(count, *self.plan_shape)
        return output.clamp(-self.control_limit, self.control_limit) \
            if self.control_limit is not None else output

    @torch.no_grad()
    def embed(self, context: torch.Tensor, candidates: torch.Tensor,
              flow_time: float = 0.9) -> torch.Tensor:
        if context.ndim == 1:
            context = context[None].expand(len(candidates), -1)
        point = flow_time * candidates.reshape(len(candidates), self.plan_dim)
        t = torch.full((len(candidates),), flow_time, device=point.device)
        return self._features(point, t, context)
