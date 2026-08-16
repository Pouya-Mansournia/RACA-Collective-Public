"""Shared data contracts for the reasoning/routing pipeline."""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np


@dataclasses.dataclass
class Candidate:
    id: int
    true_cost: float  # unknown to the router; only observable through estimates


@dataclasses.dataclass
class Observation:
    robot_id: int
    candidates: list[Candidate]
    det_estimate_costs: list[float]
    det_estimate_latency: float
    ambiguity: float
    urgency: float
    candidate_count: int
    cost_margin: float
    battery_soc: float
    recent_latency_mean: float
    is_hard: bool = False  # ground-truth difficulty-axis label, offline analysis only


@dataclasses.dataclass
class DecisionResult:
    chosen_index: int
    latency: float
    used_reasoning: bool


class ReasoningBackend(ABC):
    """A backend that picks one candidate action given noisy cost estimates."""

    name: str

    @abstractmethod
    def estimate(self, candidates: list[Candidate], rng: np.random.Generator) -> tuple[list[float], float]:
        """Returns (estimated_costs, latency_seconds)."""

    def decide(self, candidates: list[Candidate], rng: np.random.Generator) -> tuple[int, float]:
        costs, latency = self.estimate(candidates, rng)
        return int(np.argmin(costs)), latency
