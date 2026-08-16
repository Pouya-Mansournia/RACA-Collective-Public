"""A2: Centralized Cognitive Manager (RACA-Collective.txt Section 19 / Section 22).

This is the centralized foil to the distributed A3-A6/A7 conditions. Unlike every
distributed mechanism in this codebase (DelegationPolicy, PeerMemory), which only ever
lets a robot see its own observation and its own locally-learned peer estimates, the
CentralizedManager here is a single entity that receives every robot's Observation for
a decision round IN ONE BATCH before any assignment is made, and decides fleet-wide:

  1. whether each robot's decision should use the deterministic backend at all
     (using the SAME calibrated B3 value estimator as A1/A4/A7, so the comparison
     isolates the centralized-vs-distributed control structure, not routing quality), and
  2. if reasoning is worth it, whether the robot should reason about its own decision or
     have that decision REASSIGNED to whichever single robot in the whole fleet the
     manager currently believes has the best capability for it.

This differs from A4 in genuine structure, not just labeling:
  - A4: each robot independently runs DelegationPolicy against its OWN PeerMemory
    (asymmetric, decentralized beliefs; no robot sees another robot's observation).
  - A2: one process sees ALL robots' observations for the round simultaneously and
    holds ONE shared capability table for the whole fleet, and assignment decisions for
    different robots in the same round can affect each other (fleet-wide capacity
    accounting, competition for the same top-capability robot).

Reassignment is not peer-to-peer delegation: the manager picks the single best-capability
candidate from the ENTIRE fleet (subject to a per-robot capacity cap), not a robot's own
locally-preferred peer, and the "capability" table is centrally maintained from every
robot's observed outcomes, not from any one robot's private history.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from src.reasoning.contracts import Observation
from src.reasoning.value_estimator import LogisticValueEstimator


@dataclasses.dataclass
class CentralCapability:
    """One fleet-wide table entry per robot, visible to and updated only by the
    central manager. Distinct from PeerMemory: there is exactly one of these tables
    for the whole fleet, not one asymmetric table per robot."""

    n_assignments: int = 0
    n_success: int = 0

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_assignments if self.n_assignments else 0.5  # uninformative prior


class CentralizedManager:
    """Sees every robot's Observation for a decision round before assigning anything.

    assign_round(observations) returns, for every robot id in the round, one of:
      ("deterministic", None)   -- B3 estimates reasoning is not worth it
      ("reason_self", None)     -- reasoning is worth it and this robot itself is
                                    currently the best (or only) available candidate
      ("reassign", other_id)    -- reasoning is worth it and the manager reassigns the
                                    decision to whichever OTHER robot in the fleet
                                    currently has the best estimated capability,
                                    subject to a per-robot reassignment capacity cap
    """

    def __init__(self, estimator: LogisticValueEstimator, tau: float,
                 capacity_per_robot: int = 2, self_bias: float = 0.05):
        self.estimator = estimator
        self.tau = tau
        self.capacity_per_robot = capacity_per_robot
        # a small bias in favor of the robot reasoning about its own decision, so a
        # brand-new fleet (empty capability table, everyone at the uninformative prior)
        # does not reassign every single decision purely on floating-point ties
        self.self_bias = self_bias
        self.capability: dict[int, CentralCapability] = {}

    def _cap(self, rid: int) -> CentralCapability:
        if rid not in self.capability:
            self.capability[rid] = CentralCapability()
        return self.capability[rid]

    def _features(self, obs: Observation) -> np.ndarray:
        return np.array([[obs.ambiguity, obs.urgency, float(obs.candidate_count),
                           obs.cost_margin, obs.battery_soc, obs.recent_latency_mean]])

    def _p_reason(self, obs: Observation) -> float:
        return float(self.estimator.predict_proba(self._features(obs))[0])

    def assign_round(self, observations: dict[int, Observation]) -> dict[int, tuple[str, int | None]]:
        assignments: dict[int, tuple[str, int | None]] = {}
        loads: dict[int, int] = {}
        # process robots in an order determined by how strongly reasoning is indicated
        # for their decision, so the highest-value reassignments claim capacity first
        # under the fleet-wide capacity cap (a genuinely fleet-wide, round-level
        # allocation decision, not something any single-robot view could make)
        p_by_rid = {rid: self._p_reason(obs) for rid, obs in observations.items()}
        order = sorted(observations.keys(), key=lambda r: -p_by_rid[r])

        for rid in order:
            obs = observations[rid]
            p = p_by_rid[rid]
            if p <= self.tau:
                assignments[rid] = ("deterministic", None)
                continue

            best_id, best_score = rid, self._cap(rid).success_rate + self.self_bias
            for other_id in observations.keys():
                if other_id == rid:
                    continue
                if loads.get(other_id, 0) >= self.capacity_per_robot:
                    continue
                score = self._cap(other_id).success_rate
                if score > best_score:
                    best_id, best_score = other_id, score

            if best_id == rid:
                assignments[rid] = ("reason_self", None)
            else:
                loads[best_id] = loads.get(best_id, 0) + 1
                assignments[rid] = ("reassign", best_id)
        return assignments

    def record_outcome(self, reasoning_robot_id: int, success: bool) -> None:
        """Updates the ONE shared capability table entry for whichever robot actually
        performed the reasoning (itself, or the robot a decision was reassigned to)."""
        cap = self._cap(reasoning_robot_id)
        cap.n_assignments += 1
        cap.n_success += int(success)
