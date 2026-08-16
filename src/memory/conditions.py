"""Phase 5: shared experience / cognitive memory conditions.

C0 - no shared memory: PeerMemory is never updated from delegation outcomes; every
     robot always uses the uninformative prior (success_rate=0.5, mean_latency=3.0).
C1 - global shared memory: a single PeerMemory instance is used by every robot, so an
     observation any robot makes about peer j is instantly visible to all robots (an
     upper bound on how much sharing could help, at zero communication cost -- an
     intentionally unrealistic control, see report).
C2 - peer-to-peer selective sharing: each robot keeps its own local PeerMemory (as in
     Phase 4/C3), but each round, with probability `share_prob`, a random pair of robots
     exchanges their stats for one randomly chosen peer they have both interacted with,
     merging counts. This costs communication (counted through MessageBus) unlike C1.
C3 - local memory + delegation history only (no sharing at all beyond what Phase 4
     already does): identical mechanism to Phase 4's delegation-enabled condition.
"""
from __future__ import annotations

import dataclasses

from src.delegation.peer_memory import PeerMemory, PeerStats


@dataclasses.dataclass
class MemoryConditionConfig:
    name: str
    shared: bool = False       # C1: all robots point at ONE PeerMemory instance
    frozen: bool = False       # C0: never record any outcomes
    share_prob: float = 0.0    # C2: probability of a pairwise merge event per round


def make_peer_memories(n_robots: int, cond: MemoryConditionConfig) -> list[PeerMemory]:
    if cond.shared:
        shared = PeerMemory()
        return [shared for _ in range(n_robots)]
    return [PeerMemory() for _ in range(n_robots)]


def merge_stats(a: PeerStats, b: PeerStats) -> PeerStats:
    return PeerStats(
        n_requests=a.n_requests + b.n_requests,
        n_success=a.n_success + b.n_success,
        sum_latency=a.sum_latency + b.sum_latency,
        n_responses=a.n_responses + b.n_responses,
        n_rejected=a.n_rejected + b.n_rejected,
        current_load=0,
    )


def maybe_share(peer_memories: list[PeerMemory], cond: MemoryConditionConfig, rng, bus, round_idx: int) -> int:
    """C2 only: returns number of sharing messages sent this round."""
    if cond.shared or cond.frozen or cond.share_prob <= 0:
        return 0
    n = len(peer_memories)
    if rng.random() >= cond.share_prob:
        return 0
    a, b = rng.choice(n, size=2, replace=False)
    common = set(peer_memories[a].stats.keys()) & set(peer_memories[b].stats.keys())
    if not common:
        return 0
    target = int(rng.choice(list(common)))
    merged = merge_stats(peer_memories[a].get(target), peer_memories[b].get(target))
    peer_memories[a].stats[target] = merged
    peer_memories[b].stats[target] = dataclasses.replace(merged)
    bus.send(a, b, "memory_share", 1, rng)
    return 1
