"""Concrete reasoning backends.

DeterministicBackend: near-zero latency, fixed noise on cost estimates
(cheap heuristic, e.g. straight-line distance / greedy rule).

ExpensiveReasoningBackend: models an LLM-like backend. Latency is drawn from
a distribution centered a few seconds. Decision quality is a *distribution*,
not a constant improvement: each call draws its own noise scale from a
range that sometimes beats the deterministic backend and sometimes is
statistically indistinguishable from it (or worse). We must not hardcode
reasoning as always better, so this backend is allowed to be mediocre on
any given call.
"""
from __future__ import annotations

import numpy as np

from src.reasoning.contracts import Candidate, ReasoningBackend


class DeterministicBackend(ReasoningBackend):
    name = "deterministic"

    def __init__(self, noise_std: float = 0.35, latency_mean: float = 0.02, latency_std: float = 0.01):
        self.noise_std = noise_std
        self.latency_mean = latency_mean
        self.latency_std = latency_std

    def estimate(self, candidates: list[Candidate], rng: np.random.Generator) -> tuple[list[float], float]:
        noise = rng.normal(0, self.noise_std, size=len(candidates))
        costs = [c.true_cost + n for c, n in zip(candidates, noise)]
        latency = max(0.001, rng.normal(self.latency_mean, self.latency_std))
        return costs, latency


class ExpensiveReasoningBackend(ReasoningBackend):
    name = "reasoning"

    def __init__(
        self,
        base_noise_std: float = 0.35,
        quality_factor_range: tuple[float, float] = (0.35, 1.25),
        latency_mean: float = 3.0,
        latency_std: float = 1.0,
        failure_prob: float = 0.03,
        failure_noise_std: float = 3.0,
    ):
        self.base_noise_std = base_noise_std
        self.quality_factor_range = quality_factor_range
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.failure_prob = failure_prob
        self.failure_noise_std = failure_noise_std

    def estimate(self, candidates: list[Candidate], rng: np.random.Generator) -> tuple[list[float], float]:
        if rng.random() < self.failure_prob:
            noise_std = self.failure_noise_std
        else:
            qf = rng.uniform(*self.quality_factor_range)
            noise_std = self.base_noise_std * qf
        noise = rng.normal(0, noise_std, size=len(candidates))
        costs = [c.true_cost + n for c, n in zip(candidates, noise)]
        latency = max(0.05, rng.normal(self.latency_mean, self.latency_std))
        return costs, latency
