import numpy as np

from src.delegation.null_policy import NullDelegationPolicy
from src.delegation.peer_memory import PeerMemory, PeerStats
from src.reasoning.contracts import Candidate, Observation


def make_obs(ambiguity=0.9, cost_margin=0.5):
    candidates = [Candidate(id=0, true_cost=1.0), Candidate(id=1, true_cost=1.0)]
    return Observation(
        robot_id=0,
        candidates=candidates,
        det_estimate_costs=[1.0, 1.05],
        det_estimate_latency=0.1,
        ambiguity=ambiguity,
        urgency=0.0,
        candidate_count=2,
        cost_margin=cost_margin,
        battery_soc=1.0,
        recent_latency_mean=0.0,
    )


def test_null_policy_picks_only_available_peers():
    policy = NullDelegationPolicy(null_rng_seed=1)
    memory = PeerMemory()
    obs = make_obs()
    peer_loads = {1: 0, 2: 0, 3: 2}  # peer 3 at capacity (default peer_capacity=2)
    for _ in range(50):
        action, peer_id = policy.decide(0, obs, memory, [0, 1, 2, 3], peer_loads)
        if action == "delegate":
            assert peer_id in (1, 2)
            assert peer_id != 3


def test_null_policy_ignores_peer_capability_for_target_choice():
    """Two peers with different (but both good enough that "delegate" clearly beats
    "self"/"skip" regardless of which one is picked) observed track records should
    still be chosen with roughly equal frequency under the null model -- target
    selection must be uniform among available peers, independent of PeerStats."""
    memory = PeerMemory()
    good = memory.get(1)
    good.n_requests, good.n_responses, good.n_success = 100, 100, 95  # 0.95 success rate
    okay = memory.get(2)
    okay.n_requests, okay.n_responses, okay.n_success = 100, 100, 85  # 0.85 success rate, still good

    # high-stakes obs (large ambiguity * cost_margin) so both peers' delegate EU
    # clearly beats self/skip EU regardless of which peer the null model picks
    obs = make_obs(ambiguity=1.0, cost_margin=5.0)
    peer_loads = {1: 0, 2: 0}
    counts = {1: 0, 2: 0}
    policy = NullDelegationPolicy(null_rng_seed=42)
    n_trials = 4000
    n_delegate = 0
    for _ in range(n_trials):
        action, peer_id = policy.decide(0, obs, memory, [0, 1, 2], peer_loads)
        if action == "delegate":
            n_delegate += 1
            counts[peer_id] += 1

    assert n_delegate > 0.95 * n_trials  # delegation should be the favored action here
    # roughly 50/50 regardless of peer 1 having a materially better track record
    frac_peer1 = counts[1] / n_delegate
    assert 0.42 < frac_peer1 < 0.58


def test_null_policy_falls_back_to_self_or_skip_when_no_peers_available():
    policy = NullDelegationPolicy(null_rng_seed=7)
    memory = PeerMemory()
    obs = make_obs()
    peer_loads = {1: 5, 2: 5}  # all peers over capacity (default peer_capacity=2)
    action, peer_id = policy.decide(0, obs, memory, [0, 1, 2], peer_loads)
    assert action in ("self", "skip")
    assert peer_id is None


def test_null_policy_deterministic_given_same_rng_seed():
    memory = PeerMemory()
    obs = make_obs()
    peer_loads = {1: 0, 2: 0, 3: 0}
    results_a = []
    results_b = []
    policy_a = NullDelegationPolicy(null_rng_seed=99)
    policy_b = NullDelegationPolicy(null_rng_seed=99)
    for _ in range(20):
        results_a.append(policy_a.decide(0, obs, memory, [0, 1, 2, 3], dict(peer_loads)))
        results_b.append(policy_b.decide(0, obs, memory, [0, 1, 2, 3], dict(peer_loads)))
    assert results_a == results_b
