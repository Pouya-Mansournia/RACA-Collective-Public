"""Bounded-window variant of PeerMemory, used only as a diagnostic ablation
(docs/PHASE_7_REPORT.md "Mechanism investigation" section, experiments/phase_n/).

`PeerMemory`/`PeerStats` (src/delegation/peer_memory.py) accumulate success_rate,
mean_latency, and reliability as CUMULATIVE, UNBOUNDED averages over the entire episode:
once a peer has responded to enough requests, a single early success or failure
contributes a smaller and smaller share of the running average forever, but it never
fully leaves the estimate. `WindowedPeerMemory` answers a narrow, disprovable question:
if a peer's `success_rate` estimate only reflects its most recent `window` responses
(old evidence ages out completely) instead of the full episode history, does the
delegate-in concentration/persistence structure shrink? If unbounded cumulative memory
is what lets a peer's early, possibly-lucky success_rate edge compound into a permanent
selection advantage, bounding the window should weaken that compounding. If concentration
is unchanged under windowing, unbounded-memory accumulation is not the mechanism.

Exposes the same read interface `DelegationPolicy.eu_peer` needs (`.success_rate`,
`.mean_latency`, `.reliability` on the object returned by `get(peer_id)`) and the same
write interface `PeerMemory` exposes (`record_request`, `record_response`,
`reset_round_load`), so it is a drop-in replacement for `PeerMemory` in the simulation
loop. No robot ID or role information is used; every robot still starts with an
identical, empty `WindowedPeerMemory`.
"""
from __future__ import annotations

import collections


class WindowedPeerStats:
    def __init__(self, window: int):
        self.window = window
        # one entry per request (whether or not it later got a response)
        self.requests: collections.deque = collections.deque(maxlen=window)
        # one (success, latency) entry per completed response
        self.responses: collections.deque = collections.deque(maxlen=window)

    @property
    def success_rate(self) -> float:
        if not self.responses:
            return 0.5  # uninformative prior, same as PeerStats
        return sum(s for s, _ in self.responses) / len(self.responses)

    @property
    def mean_latency(self) -> float:
        if not self.responses:
            return 3.0  # pessimistic prior, same as PeerStats
        return sum(l for _, l in self.responses) / len(self.responses)

    @property
    def reliability(self) -> float:
        if not self.requests:
            return 0.5  # uninformative prior, same as PeerStats
        return len(self.responses) / len(self.requests)


class WindowedPeerMemory:
    """Same role/keying as PeerMemory: one instance per robot, keyed by peer id."""

    def __init__(self, window: int = 50):
        self.window = window
        self.stats: dict[int, WindowedPeerStats] = {}

    def get(self, peer_id: int) -> WindowedPeerStats:
        if peer_id not in self.stats:
            self.stats[peer_id] = WindowedPeerStats(self.window)
        return self.stats[peer_id]

    def record_request(self, peer_id: int):
        self.get(peer_id).requests.append(1)

    def record_rejected(self, peer_id: int):
        pass  # not read by eu_peer(); kept only for interface parity with PeerMemory

    def record_response(self, peer_id: int, success: bool, latency: float):
        self.get(peer_id).responses.append((bool(success), float(latency)))

    def reset_round_load(self):
        pass  # PeerMemory.current_load is never read by DelegationPolicy; parity only
