"""Small conditional flow-matching policy; dimensions are supplied by the task adapter."""
from __future__ import annotations

import math

import torch
from torch import nn


class ConditionalFlowMLP(nn.Module):
    def __init__(self, context_dim: int, plan_shape: tuple[int, ...], hidden: int = 96,
                 representation_dim: int = 32, control_limit: float | None = None,
                 nfe: int = 8, trunk_depth: int = 3,
                 time_features: str = "fourier8"):
        super().__init__()
        self.context_dim = int(context_dim)
        self.plan_shape = tuple(int(value) for value in plan_shape)
        self.plan_dim = math.prod(self.plan_shape)
        self.control_limit = control_limit
        self.nfe = int(nfe)
        if time_features not in {"raw1", "fourier8"}:
            raise ValueError("time_features must be 'raw1' or 'fourier8'")
        self.time_features = time_features
        input_dim = self.plan_dim + self.context_dim + (1 if time_features == "raw1" else 8)
        if trunk_depth == 3:
            layers = [nn.Linear(input_dim, hidden), nn.SiLU(),
                      nn.Linear(hidden, hidden), nn.SiLU(),
                      nn.Linear(hidden, representation_dim), nn.SiLU()]
        elif trunk_depth == 2:
            # Shallow spec: input -> hidden -> representation (phi_s) -> plan.
            layers = [nn.Linear(input_dim, hidden), nn.SiLU(),
                      nn.Linear(hidden, representation_dim), nn.SiLU()]
        else:
            raise ValueError("trunk_depth must be 2 or 3")
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(representation_dim, self.plan_dim)

    def expansion_parameter_groups(
            self, base_lr: float, first_layer_lr_scale: float = 1.0,
            ) -> list[dict[str, object]]:
        """Adam groups for a slower input layer during expansion.

        Pretraining still optimizes every parameter at one learning rate.  This
        grouping is only used by the expansion loop, where the newly learned
        conditioning interface should move more slowly than the representation
        layer and output head.
        """
        if base_lr <= 0.0 or not 0.0 < first_layer_lr_scale <= 1.0:
            raise ValueError("require base_lr>0 and first_layer_lr_scale in (0,1]")
        first = list(self.trunk[0].parameters())
        first_ids = {id(parameter) for parameter in first}
        remaining = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in first_ids
        ]
        return [
            {"params": first, "lr": base_lr * first_layer_lr_scale},
            {"params": remaining, "lr": base_lr},
        ]

    def _time(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_features == "raw1":
            return t[:, None]
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
                 reduction: str = "none",
                 loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        target = candidates.reshape(len(candidates), self.plan_dim)
        base = torch.randn_like(target)
        flat_mask = None
        if loss_mask is not None:
            if tuple(loss_mask.shape) != tuple(candidates.shape):
                raise ValueError("loss_mask must match candidates")
            flat_mask = loss_mask.reshape(len(candidates), self.plan_dim).to(
                device=target.device, dtype=target.dtype,
            )
            if bool((flat_mask.sum(dim=1) <= 0.0).any()):
                raise ValueError("each CFM loss mask must contain an active control")
            base = base * flat_mask
        t = torch.rand(len(target), device=target.device)
        point = (1.0 - t[:, None]) * base + t[:, None] * target
        squared = (self(point, t, contexts) - (target - base)).square()
        if flat_mask is None:
            per_sample = squared.mean(dim=1)
        else:
            per_sample = (squared * flat_mask).sum(dim=1) / flat_mask.sum(dim=1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'none' or 'mean'")

    @torch.no_grad()
    def sample(self, context: torch.Tensor, count: int,
               generator: torch.Generator,
               base_std: float = 1.0) -> torch.Tensor:
        return self.sample_with_base(
            context, count, generator, base_std=base_std,
        )[0]

    @torch.no_grad()
    def sample_with_base(self, context: torch.Tensor, count: int,
                         generator: torch.Generator,
                         base_std: float = 1.0,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample plans and return their paired initial Gaussian flow latents."""
        if not math.isfinite(base_std) or base_std < 0.0:
            raise ValueError("base_std must be finite and nonnegative")
        context = context.reshape(1, -1).expand(count, -1)
        x = (
            torch.randn(
                count, self.plan_dim, device=context.device, generator=generator,
            )
            * base_std
        )
        base = x.clone()
        nfe = self.nfe
        for index in range(nfe):
            t = torch.full((count,), index / nfe, device=x.device)
            x = x + self(x, t, context) / nfe
        output = x.reshape(count, *self.plan_shape)
        if self.control_limit is not None:
            output = output.clamp(-self.control_limit, self.control_limit)
        return output, base.reshape(count, *self.plan_shape)

    @torch.no_grad()
    def embed(self, context: torch.Tensor, candidates: torch.Tensor,
              flow_time: float = 0.9, base: torch.Tensor | None = None) -> torch.Tensor:
        """Penultimate features at the flow-path point; ``base`` is the optional CFM base noise."""
        if context.ndim == 1:
            context = context[None].expand(len(candidates), -1)
        point = flow_time * candidates.reshape(len(candidates), self.plan_dim)
        if base is not None:
            point = point + (1.0 - flow_time) * base.reshape(len(candidates), self.plan_dim)
        t = torch.full((len(candidates),), flow_time, device=point.device)
        return self._features(point, t, context)
