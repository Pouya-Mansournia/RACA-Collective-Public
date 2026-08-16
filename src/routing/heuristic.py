"""B2: heuristic ambiguity/urgency router (reimplementation of the previous
RACA AdaptiveRouter idea, not a copy of its code).

U(deterministic) = 1.0
U(reasoning) = 1.0 + quality_bonus_scale * ambiguity(x)
utility_gap = U(reasoning) - U(deterministic) = quality_bonus_scale * ambiguity(x)

cost_penalty = lambda_latency * expected_reasoning_latency * (1 + urgency(x))
             + lambda_reliability * reasoning_failure_rate

Escalate to reasoning iff utility_gap - cost_penalty > 0.
"""
from __future__ import annotations

import numpy as np

from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult, Observation


class HeuristicRouter:
    name = "B2_heuristic"

    def __init__(
        self,
        det_backend: DeterministicBackend | None = None,
        reasoning_backend: ExpensiveReasoningBackend | None = None,
        quality_bonus_scale: float = 1.0,
        lambda_latency: float = 0.15,
        lambda_reliability: float = 1.0,
        expected_reasoning_latency: float = 3.0,
        reasoning_failure_rate: float = 0.03,
    ):
        self.det_backend = det_backend or DeterministicBackend()
        self.reasoning_backend = reasoning_backend or ExpensiveReasoningBackend()
        self.quality_bonus_scale = quality_bonus_scale
        self.lambda_latency = lambda_latency
        self.lambda_reliability = lambda_reliability
        self.expected_reasoning_latency = expected_reasoning_latency
        self.reasoning_failure_rate = reasoning_failure_rate

    def det_estimate(self, candidates, rng):
        return self.det_backend.estimate(candidates, rng)

    def utility_gap(self, obs: Observation) -> float:
        benefit = self.quality_bonus_scale * obs.ambiguity
        cost_penalty = (
            self.lambda_latency * self.expected_reasoning_latency * (1 + obs.urgency)
            + self.lambda_reliability * self.reasoning_failure_rate
        )
        return benefit - cost_penalty

    def decide(self, obs: Observation, rng: np.random.Generator) -> DecisionResult:
        if self.utility_gap(obs) > 0:
            idx, latency = self.reasoning_backend.decide(obs.candidates, rng)
            return DecisionResult(chosen_index=idx, latency=obs.det_estimate_latency + latency, used_reasoning=True)
        idx = int(np.argmin(obs.det_estimate_costs))
        return DecisionResult(chosen_index=idx, latency=obs.det_estimate_latency, used_reasoning=False)
