"""Per-robot memory of observed peer behavior (RACA-Collective.txt Section 13).

Nothing here is hardcoded per robot ID; every robot starts with an identical, empty
PeerMemory and only learns differentiated estimates of its peers from what it actually
observes during the run. No roles are assigned.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PeerStats:
    n_requests: int = 0
    n_success: int = 0  # response beat (or matched) what deterministic would have chosen
    sum_latency: float = 0.0
    n_responses: int = 0
    n_rejected: int = 0
    current_load: int = 0  # outstanding requests this round, reset each round

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_responses if self.n_responses else 0.5  # uninformative prior

    @property
    def mean_latency(self) -> float:
        return self.sum_latency / self.n_responses if self.n_responses else 3.0  # pessimistic prior

    @property
    def reliability(self) -> float:
        return self.n_responses / self.n_requests if self.n_requests else 0.5


class PeerMemory:
    """One of these per robot; keyed by peer robot id."""

    def __init__(self):
        self.stats: dict[int, PeerStats] = {}

    def get(self, peer_id: int) -> PeerStats:
        if peer_id not in self.stats:
            self.stats[peer_id] = PeerStats()
        return self.stats[peer_id]

    def record_request(self, peer_id: int):
        self.get(peer_id).n_requests += 1

    def record_rejected(self, peer_id: int):
        self.get(peer_id).n_rejected += 1

    def record_response(self, peer_id: int, success: bool, latency: float):
        s = self.get(peer_id)
        s.n_responses += 1
        s.n_success += int(success)
        s.sum_latency += latency

    def reset_round_load(self):
        for s in self.stats.values():
            s.current_load = 0
