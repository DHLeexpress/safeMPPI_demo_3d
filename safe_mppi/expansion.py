"""Task-agnostic, single-arm Safe Flow Expansion reference.

The task adapter owns dynamics, context construction, nominal ``H_P``, the full
verifier, and the execution cost.  This file owns only the B1 expansion loop:
RBF uncertainty, budgeted acquisition, fail-closed execution, recent positive
replay, and optional signed negative gradients.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ExpansionConfig:
    rounds: int = 10
    gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    parallel_episodes: int = 2
    max_steps: int = 100
    K: int = 16
    B: int = 4
    batch_size: int = 32
    inner_steps: int | None = None  # None = one exact pass over every eligible D+ row.
    learning_rate: float = 3.0e-5
    replay_rounds: int = 2
    gp_buffer_cap: int = 256
    gp_noise: float = 1.0e-2
    rbf_lengthscale: float | None = None
    beta: float = 0.05
    adaptive_beta: bool = False
    ess_target: float = 0.5
    negative_alpha: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if self.rounds < 1 or self.parallel_episodes < 1 or self.max_steps < 1:
            raise ValueError("rounds, parallel_episodes, and max_steps must be positive")
        if not (1 <= self.B <= self.K):
            raise ValueError("require 1 <= B <= K")
        if self.batch_size < 1 or (self.inner_steps is not None and self.inner_steps < 1):
            raise ValueError("batch_size and an explicit inner_steps must be positive")
        if self.replay_rounds < 1 or self.gp_buffer_cap < 2:
            raise ValueError("replay_rounds must be positive and gp_buffer_cap at least two")
        if self.gp_noise <= 0.0 or self.beta <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("gp_noise, beta, and learning_rate must be positive")
        if not 0.0 < self.ess_target <= 1.0 or self.negative_alpha < 0.0:
            raise ValueError("ess_target must lie in (0,1] and negative_alpha be nonnegative")


@dataclass(frozen=True)
class Verification:
    """One successfully evaluated full-verifier result.

    ``valid`` is the full-H label stored in D+. ``hp_eligible`` is the separate
    one-step nominal gate used only before execution.  ``execution_cost`` ranks
    candidates that satisfy both.
    """

    valid: bool
    hp_eligible: bool
    margin: float
    execution_cost: float
    error: bool = False


@dataclass
class QueryRecord:
    round: int
    gamma: float
    episode: int
    context_id: int
    context: torch.Tensor
    candidate: torch.Tensor
    verification: Verification
    nvp_context: bool = False


class ExpansionPolicy(Protocol):
    def parameters(self): ...
    def sample(self, context: torch.Tensor, count: int,
               generator: torch.Generator) -> torch.Tensor: ...
    def embed(self, context: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor: ...
    def cfm_loss(self, contexts: torch.Tensor, candidates: torch.Tensor,
                 reduction: str = "none") -> torch.Tensor: ...
    def state_dict(self): ...


class ExpansionTask(Protocol):
    def reset(self, gamma: float, episode: int, seed: int) -> Any: ...
    def context(self, state: Any, gamma: float) -> torch.Tensor: ...
    def verify(self, context: torch.Tensor, candidates: torch.Tensor,
               gamma: float) -> Sequence[Verification]: ...
    def advance(self, state: Any, candidate: torch.Tensor) -> Any: ...
    def terminal(self, state: Any) -> str | None: ...


def _normalize(values: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def mean_pairwise_lengthscale(features: torch.Tensor) -> float:
    values = _normalize(features.detach().to(torch.float64))
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("RBF calibration needs at least two embedding rows")
    lengthscale = float(torch.pdist(values).mean())
    if not math.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("pretrained embeddings do not define a positive length scale")
    return lengthscale


class RBFPosterior:
    """Exact RBF posterior variance on a capped, previous-round D+ buffer."""

    def __init__(self, lengthscale: float, noise: float):
        if lengthscale <= 0.0 or noise <= 0.0:
            raise ValueError("RBF lengthscale and observation noise must be positive")
        self.lengthscale, self.noise = float(lengthscale), float(noise)
        self.X: torch.Tensor | None = None
        self.L: torch.Tensor | None = None

    @staticmethod
    def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a * a).sum(1, keepdim=True) + (b * b).sum(1)[None]
                - 2.0 * a @ b.T).clamp_min(0.0)

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sqdist(a, b) / (2.0 * self.lengthscale ** 2))

    @torch.no_grad()
    def set_buffer(self, features: torch.Tensor | None) -> None:
        if features is None or len(features) == 0:
            self.X = self.L = None
            return
        self.X = _normalize(features.detach())
        kernel = self.kernel(self.X, self.X).to(torch.float64)
        eye = torch.eye(len(kernel), dtype=torch.float64, device=kernel.device)
        jitter = self.noise
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(kernel + jitter * eye)
            if int(info.max()) == 0:
                self.L = factor
                return
            jitter *= 10.0
        raise RuntimeError("RBF posterior Cholesky failed")

    @torch.no_grad()
    def covariance(self, features: torch.Tensor) -> torch.Tensor:
        query = _normalize(features.detach())
        covariance = self.kernel(query, query)
        if self.X is not None:
            cross = self.kernel(query, self.X)
            solved = torch.cholesky_solve(cross.T.to(torch.float64), self.L)
            covariance -= cross @ solved.to(cross.dtype)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += self.noise * torch.eye(
            len(query), dtype=query.dtype, device=query.device)
        return covariance

    @torch.no_grad()
    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        diagonal = torch.diagonal(self.covariance(features)) - self.noise
        return diagonal.clamp_min(0.0).sqrt()

    @torch.no_grad()
    def acquire(self, features: torch.Tensor, B: int, beta: float,
                generator: torch.Generator) -> tuple[list[int], list[float], list[float]]:
        covariance = self.covariance(features)
        remaining = torch.arange(len(features), device=features.device)
        selected, selected_sigma, ess = [], [], []
        for _ in range(B):
            scores = (torch.diagonal(covariance) / (1.0 + self.noise)).clamp(0.0, 1.0)
            weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
            probability = weights / weights.sum()
            local = int(torch.multinomial(probability, 1, generator=generator))
            selected.append(int(remaining[local]))
            selected_sigma.append(float(scores[local].sqrt()))
            ess.append(float(1.0 / (probability.to(torch.float64).square().sum()
                                    * probability.numel())))
            keep = torch.ones(len(remaining), dtype=torch.bool, device=remaining.device)
            keep[local] = False
            if not bool(keep.any()):
                break
            cross = covariance[keep, local]
            denominator = covariance[local, local].clamp_min(1.0e-12)
            covariance = covariance[keep][:, keep] - torch.outer(cross, cross) / denominator
            covariance = 0.5 * (covariance + covariance.T)
            remaining = remaining[keep]
        return selected, selected_sigma, ess


def normalized_ess(scores: torch.Tensor, beta: float) -> float:
    weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
    probability = weights / weights.sum()
    return float(1.0 / (probability.to(torch.float64).square().sum()
                        * probability.numel()))


def calibrate_fixed_beta(score_pools: Sequence[torch.Tensor], target: float,
                         lower: float = 1.0e-5, upper: float = 10.0) -> float:
    """Choose beta once from representative pools; expansion keeps it fixed."""
    if not score_pools or not 0.0 < target <= 1.0:
        raise ValueError("provide score pools and an ESS target in (0,1]")
    for _ in range(60):
        middle = math.sqrt(lower * upper)
        value = float(np.median([normalized_ess(pool, middle) for pool in score_pools]))
        if value < target:
            lower = middle
        else:
            upper = middle
    return math.sqrt(lower * upper)


def _recent(records: Sequence[QueryRecord], round_i: int, width: int,
            *, positive: bool | None = None) -> list[QueryRecord]:
    first = max(0, round_i - width + 1)
    output = [row for row in records if first <= row.round <= round_i]
    if positive is not None:
        output = [row for row in output if row.verification.valid is positive]
    return output


def _balanced_cap(records: Sequence[QueryRecord], cap: int,
                  rng: np.random.Generator) -> list[QueryRecord]:
    """Round/gamma-balanced sampling without replacement for the GP only."""
    cells: dict[tuple[int, float], list[QueryRecord]] = {}
    for row in records:
        cells.setdefault((row.round, row.gamma), []).append(row)
    chosen: list[QueryRecord] = []
    while len(chosen) < cap and any(cells.values()):
        for key in sorted(cells):
            if len(chosen) == cap:
                break
            cell = cells[key]
            if cell:
                chosen.append(cell.pop(int(rng.integers(len(cell)))))
    return chosen


def _equal_mass_weights(records: Sequence[QueryRecord]) -> torch.Tensor:
    """Equal mass over gamma -> (round, episode) -> context -> positive query."""
    gammas = sorted({row.gamma for row in records})
    lineages: dict[float, set[tuple[int, int]]] = {}
    contexts: dict[tuple[float, int, int], set[int]] = {}
    queries: dict[tuple[float, int, int, int], int] = {}
    for row in records:
        lineage = (row.round, row.episode)
        lineages.setdefault(row.gamma, set()).add(lineage)
        contexts.setdefault((row.gamma, *lineage), set()).add(row.context_id)
        query = (row.gamma, *lineage, row.context_id)
        queries[query] = queries.get(query, 0) + 1
    weights = []
    for row in records:
        lineage = (row.gamma, row.round, row.episode)
        query = (*lineage, row.context_id)
        weights.append(1.0 / (
            len(gammas) * len(lineages[row.gamma]) * len(contexts[lineage]) * queries[query]
        ))
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean()


def _stack(records: Sequence[QueryRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.stack([row.context for row in records]),
            torch.stack([row.candidate for row in records]))


def _gradient_norm(gradients: Sequence[torch.Tensor | None], device) -> torch.Tensor:
    return torch.sqrt(sum((gradient.square().sum() for gradient in gradients
                           if gradient is not None),
                          torch.zeros((), device=device)).clamp_min(0.0))


def _update(policy: ExpansionPolicy, optimizer: torch.optim.Optimizer,
            positives: Sequence[QueryRecord], negatives: Sequence[QueryRecord],
            cfg: ExpansionConfig, generator: torch.Generator) -> dict[str, float | int | None]:
    if not positives:
        return {"steps": 0, "positive_loss": None, "negative_loss": None}
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    device = parameters[0].device
    order = torch.randperm(
        len(positives), generator=generator, device=device
    ).cpu().tolist()
    if cfg.inner_steps is not None:
        order = order[:cfg.inner_steps * cfg.batch_size]
    weights = _equal_mass_weights(positives)
    positive_values, negative_values = [], []
    for start in range(0, len(order), cfg.batch_size):
        ids = order[start:start + cfg.batch_size]
        batch = [positives[index] for index in ids]
        contexts, candidates = (value.to(device) for value in _stack(batch))
        per_sample = policy.cfm_loss(contexts, candidates, reduction="none")
        positive_loss = (per_sample * weights[ids].to(device)).mean()
        optimizer.zero_grad()
        if cfg.negative_alpha and negatives:
            negative_ids = torch.randint(
                len(negatives), (len(batch),), generator=generator, device=device
            ).cpu().tolist()
            neg_context, neg_candidate = _stack([negatives[index] for index in negative_ids])
            negative_loss = policy.cfm_loss(
                neg_context.to(device), neg_candidate.to(device), reduction="mean"
            )
            positive_grad = torch.autograd.grad(positive_loss, parameters, allow_unused=True)
            negative_grad = torch.autograd.grad(negative_loss, parameters, allow_unused=True)
            pos_norm = _gradient_norm(positive_grad, device)
            neg_norm = _gradient_norm(negative_grad, device)
            rho = cfg.negative_alpha * pos_norm / (neg_norm + 1.0e-12)
            for parameter, pos, neg in zip(parameters, positive_grad, negative_grad):
                if pos is None and neg is None:
                    parameter.grad = None
                elif pos is None:
                    parameter.grad = -rho * neg.detach()
                elif neg is None:
                    parameter.grad = pos.detach()
                else:
                    parameter.grad = pos.detach() - rho * neg.detach()
            negative_values.append(float(negative_loss.detach()))
        else:
            positive_loss.backward()
        optimizer.step()
        positive_values.append(float(positive_loss.detach()))
    return {
        "steps": len(positive_values),
        "positive_loss": float(np.mean(positive_values)),
        "negative_loss": (float(np.mean(negative_values)) if negative_values else None),
    }


def run_safe_expansion(
    policy: ExpansionPolicy,
    task: ExpansionTask,
    output_dir: str | Path,
    *,
    config: ExpansionConfig = ExpansionConfig(),
    calibration_features: torch.Tensor | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run expert-free B1 expansion; task callbacks supply every task-specific fact."""
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("expansion output_dir must be empty")
    if config.rbf_lengthscale is None:
        if calibration_features is None:
            raise ValueError("provide 50 pretrained embeddings or set rbf_lengthscale explicitly")
        lengthscale = mean_pairwise_lengthscale(calibration_features)
    else:
        lengthscale = float(config.rbf_lengthscale)
    archive: list[QueryRecord] = []
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    rng = np.random.default_rng(config.seed)
    device = next(policy.parameters()).device
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(config.seed)
    beta = float(config.beta)
    round_rows = []
    torch.save({"round": 0, "model": policy.state_dict(),
                "config": asdict(config), "pretrained": True},
               output_dir / "checkpoint_000.pt")
    for round_i in range(1, config.rounds + 1):
        beta_used = beta
        gp = RBFPosterior(lengthscale, config.gp_noise)
        previous_positive = _recent(archive, round_i - 1, config.replay_rounds, positive=True)
        gp_rows = _balanced_cap(previous_positive, config.gp_buffer_cap, rng)
        if gp_rows:
            context, candidate = _stack(gp_rows)
            with torch.no_grad():
                gp.set_buffer(policy.embed(context.to(next(policy.parameters()).device),
                                           candidate.to(next(policy.parameters()).device)))
        episodes = []
        for gamma in config.gammas:
            for replica in range(config.parallel_episodes):
                episode = len(episodes)
                episodes.append({
                    "gamma": gamma, "episode": episode, "status": None,
                    "state": task.reset(gamma, episode, config.seed + 10000 * round_i + episode),
                })
        ess_values, score_pools, nvp = [], [], 0
        context_id = 0
        for step in range(config.max_steps):
            active = [episode for episode in episodes if episode["status"] is None]
            if not active:
                break
            for episode in active:
                gamma = float(episode["gamma"])
                state_before = episode["state"]
                context = task.context(episode["state"], gamma).detach().to(device)
                candidates = policy.sample(context, config.K, torch_rng).detach()
                features = policy.embed(context, candidates)
                sigma_k = gp.sigma(features).detach().cpu()
                score_pools.append(sigma_k)
                selected, selected_sigma, ess = gp.acquire(
                    features, config.B, beta_used, torch_rng
                )
                ess_values.extend(ess)
                queried = candidates[selected]
                results = list(task.verify(context, queried, gamma))
                if len(results) != len(selected):
                    raise RuntimeError("task.verify must return exactly one result per selected query")
                current_records = []
                for candidate, result in zip(queried, results):
                    if result.error:
                        continue
                    record = QueryRecord(
                        round_i, gamma, int(episode["episode"]), context_id,
                        context.detach().cpu(), candidate.detach().cpu(), result,
                    )
                    archive.append(record)
                    current_records.append(record)
                eligible = [index for index, result in enumerate(results)
                            if not result.error and result.valid and result.hp_eligible]
                if not eligible:
                    episode["status"] = "NVP"
                    nvp += 1
                    for record in current_records:
                        record.nvp_context = True
                    chosen = None
                else:
                    chosen = min(eligible, key=lambda index: results[index].execution_cost)
                    episode["state"] = task.advance(episode["state"], queried[chosen])
                    episode["status"] = task.terminal(episode["state"])
                if event_callback is not None:
                    event_callback({
                        "round": round_i, "step": step, "gamma": gamma,
                        "episode": int(episode["episode"]), "context_id": context_id,
                        "state_before": state_before, "state_after": episode["state"],
                        "context": context.detach().cpu(),
                        "candidates": candidates.detach().cpu(),
                        "sigma_K": sigma_k,
                        "selected": selected, "selected_sigma": selected_sigma,
                        "verification": [asdict(result) for result in results],
                        "chosen_local": chosen, "status": episode["status"],
                    })
                context_id += 1
        eligible_positive = _recent(archive, round_i, config.replay_rounds, positive=True)
        negative = [row for row in _recent(archive, round_i, config.replay_rounds)
                    if row.nvp_context]
        update = _update(policy, optimizer, eligible_positive, negative,
                         config, torch_rng)
        if config.adaptive_beta and score_pools:
            beta = calibrate_fixed_beta(score_pools, config.ess_target)
        statuses = [episode["status"] or "TIMEOUT" for episode in episodes]
        row = {
            "round": round_i,
            "queries": sum(record.round == round_i for record in archive),
            "positives": sum(record.round == round_i and record.verification.valid
                             for record in archive),
            "gp_buffer": len(gp_rows),
            "replay_positives": len(eligible_positive),
            "beta": beta_used,
            "beta_next": beta,
            "ESS_over_K": float(np.mean(ess_values)) if ess_values else 1.0,
            "NVP": nvp,
            "success": statuses.count("SUCCESS"),
            "timeout": statuses.count("TIMEOUT"),
            **update,
        }
        round_rows.append(row)
        torch.save({"round": round_i, "model": policy.state_dict(),
                    "config": asdict(config), "pretrained": False},
                   output_dir / f"checkpoint_{round_i:03d}.pt")
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "status": "SAFE_FLOW_EXPANSION_COMPLETE",
        "config": asdict(config),
        "rbf_lengthscale": lengthscale,
        "final_beta": beta,
        "D": len(archive),
        "D_plus": sum(row.verification.valid for row in archive),
        "rounds": round_rows,
        "semantics": {
            "GP": "recent full-H verifier positives only",
            "D": "all successful selected-B verifier queries; verifier errors excluded",
            "D_plus": "full-H verifier positives",
            "execution": "nominal-Hp eligible full-H positives ranked by task execution_cost",
            "failure": "NVP terminates that episode; no expert fallback",
            "replay": "recent D+; gamma-lineage-context-query equal mass",
        },
    }
    torch.save(archive, output_dir / "query_archive.pt")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                          allow_nan=False) + "\n")
    return manifest
