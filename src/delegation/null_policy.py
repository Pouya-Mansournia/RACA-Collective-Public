"""Null-model delegation policy (RACA-Collective.txt Section 20 / large-N residual-structure
follow-up, see docs/PHASE_7_REPORT.md "Definitive resolution" section).

`DelegationPolicy` (src/delegation/policy.py) picks the peer with the highest estimated
expected utility, which is itself a function of that peer's PeerMemory track record --
i.e. delegation targets are chosen partly on observed capability/history.

`NullDelegationPolicy` answers a different, narrower question: keep the same
self/skip/delegate expected-utility comparison as the real policy, so the DECISION OF
WHETHER TO DELEGATE is unchanged, but randomize which PEER receives a delegation once
delegation has already won. The recipient is chosen uniformly at random among peers that
are currently willing/available (i.e. under the same peer-capacity constraint the real
policy respects), with NO capability-based and NO memory/history-based selection among
them -- a fair coin among available peers.

Everything else (task arrival process, battery dynamics, robot count, episode length, the
artifact-control machinery in experiments/phase_f/run_specialization_revised.py's
run_episode, expected-utility formulas for self/skip) is unchanged: this class only
overrides which peer is picked once "delegate" already looks favorable.

This is used as an explicit null model: if the real (capability/history-based) policy's
delegate-in concentration statistics are not distinguishable from this pure-random-target
null model's statistics, that is evidence the residual concentration reported in Phase 6/7
is sampling noise rather than structure. If the real policy's statistics are reliably
higher, that rules out "pure chance among willing peers" as the explanation (though it does
not by itself prove interaction-driven emergence -- see the rich-get-richer mechanism check
in experiments/phase_m/run_large_n_resolution.py).
"""
from __future__ import annotations

import numpy as np

from src.delegation.policy import DelegationPolicy
from src.delegation.peer_memory import PeerMemory
from src.reasoning.contracts import Observation


class NullDelegationPolicy(DelegationPolicy):
    def __init__(self, *args, null_rng_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(null_rng_seed)

    def decide(self, self_id: int, obs: Observation, peer_memory: PeerMemory, peer_ids: list[int],
                peer_loads: dict[int, int]) -> tuple[str, int | None]:
        """Preserve the base policy's delegate/non-delegate decision, then randomize
        only the target peer if delegation won."""
        action, real_peer = super().decide(self_id, obs, peer_memory, peer_ids, peer_loads)
        if action != "delegate":
            return action, None

        available = [pid for pid in peer_ids
                     if pid != self_id and peer_loads.get(pid, 0) < self.peer_capacity]
        if not available:
            return "delegate", real_peer
        chosen_peer = available[int(self._rng.integers(0, len(available)))]
        return "delegate", chosen_peer
