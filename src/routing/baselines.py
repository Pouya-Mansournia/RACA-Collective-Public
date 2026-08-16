"""B0/B1 routing baselines: always deterministic, always expensive reasoning."""
from __future__ import annotations

import numpy as np

from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult, Observation


class AlwaysDeterministic:
    name = "B0_always_deterministic"

    def __init__(self, det_backend: DeterministicBackend | None = None):
        self.det_backend = det_backend or DeterministicBackend()

    def det_estimate(self, candidates, rng):
        return self.det_backend.estimate(candidates, rng)

    def decide(self, obs: Observation, rng: np.random.Generator) -> DecisionResult:
        idx = int(np.argmin(obs.det_estimate_costs))
        return DecisionResult(chosen_index=idx, latency=obs.det_estimate_latency, used_reasoning=False)


class AlwaysExpensiveReasoning:
    name = "B1_always_reasoning"

    def __init__(self, det_backend: DeterministicBackend | None = None,
                 reasoning_backend: ExpensiveReasoningBackend | None = None):
        self.det_backend = det_backend or DeterministicBackend()
        self.reasoning_backend = reasoning_backend or ExpensiveReasoningBackend()

    def det_estimate(self, candidates, rng):
        # still computed (needed for Observation features) but not used for the decision
        return self.det_backend.estimate(candidates, rng)

    def decide(self, obs: Observation, rng: np.random.Generator) -> DecisionResult:
        idx, latency = self.reasoning_backend.decide(obs.candidates, rng)
        total_latency = obs.det_estimate_latency + latency
        return DecisionResult(chosen_index=idx, latency=total_latency, used_reasoning=True)
