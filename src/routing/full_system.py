"""A7: Full RACA-Collective (RACA-Collective.txt Section 19, "everything combined").

This module introduces no new decision logic; it composes three mechanisms that already
exist and were separately validated:

  - the calibrated learned outcome-aware router (B3, src/routing/learned.py /
    src/reasoning/value_estimator.py) decides WHETHER reasoning is worth it at all,
    replacing DelegationPolicy's own ambiguity-based self/skip comparison as the
    escalation gate,
  - the calibrated DelegationPolicy (src/delegation/policy.py, parameters from
    experiments/phase_d/calibrate_delegation.py) decides WHO reasons once B3 has said
    reasoning is worth it: the robot itself or a peer,
  - local PeerMemory (the C3 condition, src/memory/conditions.py) is each robot's own
    private history of delegation outcomes; Phase 5/11 found no memory condition beats
    any other once the policy is calibrated (all four collapse to numerically identical
    utility), so C3 is used because it is the richest condition capable of producing any
    history-dependent behavior at all, matching the choice already made for A5/Phase 6.

Everything here is DISTRIBUTED: each robot only ever sees its own Observation and its
own PeerMemory. There is no central entity, unlike src/routing/centralized.py (A2).
"""
from __future__ import annotations

import numpy as np

from src.delegation.peer_memory import PeerMemory
from src.delegation.policy import DelegationPolicy
from src.reasoning.contracts import Observation
from src.reasoning.value_estimator import LogisticValueEstimator


class FullSystemRouter:
    """B3 gates whether to reason at all; the calibrated DelegationPolicy then chooses
    self vs. peer, using the robot's own local PeerMemory. Purely a composition of
    pre-existing mechanisms."""

    def __init__(self, estimator: LogisticValueEstimator, tau: float, policy: DelegationPolicy):
        self.estimator = estimator
        self.tau = tau
        self.policy = policy

    def _features(self, obs: Observation) -> np.ndarray:
        return np.array([[obs.ambiguity, obs.urgency, float(obs.candidate_count),
                           obs.cost_margin, obs.battery_soc, obs.recent_latency_mean]])

    def should_reason(self, obs: Observation) -> bool:
        p = float(self.estimator.predict_proba(self._features(obs))[0])
        return p > self.tau

    def decide(self, self_id: int, obs: Observation, peer_memory: PeerMemory,
                peer_ids: list[int], peer_loads: dict[int, int]) -> tuple[str, int | None]:
        """Returns (action, peer_id_or_None); action in {"skip", "self", "delegate"}.

        "skip" is decided by B3 alone (the outcome-aware gate). Once B3 says reasoning
        is worth it, the choice of self vs. delegate uses the SAME calibrated expected
        utility comparison DelegationPolicy uses in A3-A6, just without re-considering
        "skip" (B3 already answered that question, so this does not double-count the
        heuristic ambiguity-based skip option the plain DelegationPolicy also offers)."""
        if not self.should_reason(obs):
            return "skip", None

        eu_self = self.policy.eu_self(obs)
        best_peer, best_peer_eu = None, -float("inf")
        for pid in peer_ids:
            if pid == self_id:
                continue
            if peer_loads.get(pid, 0) >= self.policy.peer_capacity:
                continue
            eu = self.policy.eu_peer(obs, peer_memory.get(pid))
            if eu > best_peer_eu:
                best_peer_eu, best_peer = eu, pid
        if best_peer is not None and best_peer_eu > eu_self:
            return "delegate", best_peer
        return "self", None
