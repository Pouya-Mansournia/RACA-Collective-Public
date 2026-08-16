"""Delegation decision: reason myself, ask a peer, or skip expensive reasoning.

Compares three expected-utility estimates (RACA-Collective.txt Section 13):

  EU(self)  = -expected_regret_if_self_reasons(x) - lambda_latency * own_reasoning_latency_mean
  EU(peer_j) = -expected_regret_if_peer_j_reasons(x, peer_j)
               - lambda_latency * (2 * msg_latency_mean + peer_j.mean_latency)
               - lambda_msg * n_messages_per_roundtrip
  EU(skip)  = -expected_regret_if_skip(x) - 0

expected_regret_if_self_reasons / if_skip are estimated from the same ambiguity feature
the Phase 1-3 heuristic/learned routers use (ambiguity(x) scales expected regret).
expected_regret_if_peer_j_reasons additionally discounts by the peer's OBSERVED
success_rate and reliability from PeerMemory: a peer nobody has successfully used before
is assumed no better than self (uninformative prior), so no peer starts out favored;
whatever a robot learns about peers comes only from its own delegation history.

No predefined roles, no hardcoded "robot X is good at Y". A robot with an empty
PeerMemory always prefers self/skip over an unproven peer.
"""
from __future__ import annotations

from src.delegation.peer_memory import PeerMemory
from src.reasoning.contracts import Observation


class DelegationPolicy:
    def __init__(
        self,
        lambda_latency: float = 0.15,
        lambda_msg: float = 0.01,
        base_regret_scale: float = 1.0,
        self_quality_factor: float = 0.7,  # self-reasoning is assumed as good as own backend's average
        msg_latency_mean: float = 0.05,
        reasoning_latency_mean: float = 3.0,
        peer_capacity: int = 2,  # max concurrent delegation requests a peer will accept per round
        self_load_penalty: float = 1.5,  # a robot with a backed-up recent-latency history looks
                                          # like a worse candidate to reason for itself right now
    ):
        self.lambda_latency = lambda_latency
        self.lambda_msg = lambda_msg
        self.base_regret_scale = base_regret_scale
        self.self_quality_factor = self_quality_factor
        self.msg_latency_mean = msg_latency_mean
        self.reasoning_latency_mean = reasoning_latency_mean
        self.peer_capacity = peer_capacity
        self.self_load_penalty = self_load_penalty

    def _expected_regret(self, obs: Observation, quality_factor: float) -> float:
        # ambiguity(x) proxies P(picking the wrong candidate); cost_margin(x) (available
        # to the router from the deterministic backend's own estimate, not a leaked
        # ground-truth label) proxies the STAKES of a wrong pick -- this is what lets the
        # policy tell a genuinely high-stakes ("hard") decision from a low-stakes one
        # without ever seeing the simulator's internal is_hard flag. quality_factor
        # scales how likely a given backend is to still get it wrong (lower = better).
        stakes = max(obs.cost_margin, 0.05)
        return self.base_regret_scale * obs.ambiguity * quality_factor * stakes

    def eu_self(self, obs: Observation) -> float:
        r = self._expected_regret(obs, self.self_quality_factor)
        # a robot whose own recent decisions have carried a lot of latency (it has been
        # doing a lot of its own reasoning lately) is treated as a worse candidate to
        # reason for itself right now -- this is what creates any incentive to delegate;
        # it is a real load signal (recent_latency_mean), not a hardcoded per-robot rule.
        load_penalty = self.self_load_penalty * obs.recent_latency_mean
        return -r - self.lambda_latency * (self.reasoning_latency_mean + load_penalty)

    def eu_skip(self, obs: Observation) -> float:
        r = self._expected_regret(obs, 1.0)  # deterministic backend quality factor = 1.0 (reference)
        return -r

    def eu_peer(self, obs: Observation, peer_stats) -> float:
        # peers with no track record get an uninformative quality factor (as good as self);
        # peers with an observed success_rate above 0.5 look better than self, below look worse.
        quality_factor = self.self_quality_factor * (1.0 - (peer_stats.success_rate - 0.5))
        quality_factor = max(0.05, quality_factor)
        r = self._expected_regret(obs, quality_factor)
        roundtrip_latency = 2 * self.msg_latency_mean + peer_stats.mean_latency
        reliability_penalty = (1.0 - peer_stats.reliability) * 0.3  # unreliable peers cost extra expected regret
        return -r - reliability_penalty - self.lambda_latency * roundtrip_latency - self.lambda_msg * 2

    def decide(self, self_id: int, obs: Observation, peer_memory: PeerMemory, peer_ids: list[int],
                peer_loads: dict[int, int]) -> tuple[str, int | None]:
        """Returns (action, peer_id_or_None). action in {"self", "skip", "delegate"}."""
        options = {"self": self.eu_self(obs), "skip": self.eu_skip(obs)}
        best_peer, best_peer_eu = None, -float("inf")
        for pid in peer_ids:
            if pid == self_id:
                continue
            if peer_loads.get(pid, 0) >= self.peer_capacity:
                continue  # peer would reject; don't even count this as the chosen option
            eu = self.eu_peer(obs, peer_memory.get(pid))
            if eu > best_peer_eu:
                best_peer_eu, best_peer = eu, pid
        if best_peer is not None:
            options["delegate"] = best_peer_eu
        best_action = max(options, key=options.get)
        if best_action == "delegate":
            return "delegate", best_peer
        return best_action, None
