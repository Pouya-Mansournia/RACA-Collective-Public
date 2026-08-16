"""V_hat(x): a small interpretable logistic regression predicting whether
reasoning is worth its cost for a decision state x, trained manually with
numpy (no sklearn). Features match the heuristic router's inputs plus a
couple of context features already computed by the simulator.
"""
from __future__ import annotations

import numpy as np

from src.reasoning.counterfactual import CounterfactualRecord

FEATURE_NAMES = ["ambiguity", "urgency", "candidate_count", "cost_margin", "battery_soc", "recent_latency_mean"]


def featurize(record: CounterfactualRecord) -> np.ndarray:
    return np.array([
        record.ambiguity,
        record.urgency,
        float(record.candidate_count),
        record.cost_margin,
        record.battery_soc,
        record.recent_latency_mean,
    ])


def label(record: CounterfactualRecord, lambda_latency: float) -> int:
    from src.reasoning.counterfactual import utility
    u_det = utility(record.regret_det, record.latency_det, lambda_latency)
    u_reasoning = utility(record.regret_reasoning, record.latency_reasoning, lambda_latency)
    return 1 if u_reasoning > u_det else 0


class LogisticValueEstimator:
    def __init__(self, n_features: int = len(FEATURE_NAMES), l2: float = 1e-3):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.l2 = l2
        self.mean_ = np.zeros(n_features)
        self.std_ = np.ones(n_features)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 500, seed: int = 0):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        Xn = self._normalize(X)
        n, d = Xn.shape
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.01, size=d)
        self.bias = 0.0
        for _ in range(epochs):
            z = Xn @ self.weights + self.bias
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = Xn.T @ (p - y) / n + self.l2 * self.weights
            grad_b = np.mean(p - y)
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X)
        z = Xn @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))
