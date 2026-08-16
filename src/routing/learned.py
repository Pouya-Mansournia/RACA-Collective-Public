"""B3: learned outcome-aware router. Escalates to the reasoning backend iff
the trained value estimator's P(reasoning worth it | x) exceeds a threshold
tau, tau chosen on a calibration set (never on validation/held-out data).
"""
from __future__ import annotations

import numpy as np

from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult, Observation
from src.reasoning.value_estimator import LogisticValueEstimator


class LearnedRouter:
    name = "B3_learned"

    def __init__(self, estimator: LogisticValueEstimator, tau: float,
                 det_backend: DeterministicBackend | None = None,
                 reasoning_backend: ExpensiveReasoningBackend | None = None):
        self.estimator = estimator
        self.tau = tau
        self.det_backend = det_backend or DeterministicBackend()
        self.reasoning_backend = reasoning_backend or ExpensiveReasoningBackend()

    def det_estimate(self, candidates, rng):
        return self.det_backend.estimate(candidates, rng)

    def _features(self, obs: Observation) -> np.ndarray:
        return np.array([
            obs.ambiguity, obs.urgency, float(obs.candidate_count),
            obs.cost_margin, obs.battery_soc, obs.recent_latency_mean,
        ]).reshape(1, -1)

    def decide(self, obs: Observation, rng: np.random.Generator) -> DecisionResult:
        p = self.estimator.predict_proba(self._features(obs))[0]
        if p > self.tau:
            idx, latency = self.reasoning_backend.decide(obs.candidates, rng)
            return DecisionResult(chosen_index=idx, latency=obs.det_estimate_latency + latency, used_reasoning=True)
        idx = int(np.argmin(obs.det_estimate_costs))
        return DecisionResult(chosen_index=idx, latency=obs.det_estimate_latency, used_reasoning=False)
