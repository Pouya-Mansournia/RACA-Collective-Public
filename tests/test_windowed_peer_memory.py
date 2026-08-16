import numpy as np

from src.delegation.windowed_peer_memory import WindowedPeerMemory, WindowedPeerStats


def test_new_peer_uses_uninformative_prior():
    mem = WindowedPeerMemory(window=5)
    stats = mem.get(3)
    assert stats.success_rate == 0.5
    assert stats.mean_latency == 3.0
    assert stats.reliability == 0.5


def test_success_rate_reflects_only_the_most_recent_window():
    mem = WindowedPeerMemory(window=3)
    mem.record_request(1)
    mem.record_response(1, success=False, latency=1.0)
    mem.record_request(1)
    mem.record_response(1, success=False, latency=1.0)
    mem.record_request(1)
    mem.record_response(1, success=False, latency=1.0)
    assert mem.get(1).success_rate == 0.0
    # a run of successes should push the failures out of the window entirely
    for _ in range(3):
        mem.record_request(1)
        mem.record_response(1, success=True, latency=1.0)
    assert mem.get(1).success_rate == 1.0


def test_window_bounds_deque_length():
    mem = WindowedPeerMemory(window=4)
    for i in range(10):
        mem.record_request(2)
        mem.record_response(2, success=(i % 2 == 0), latency=float(i))
    stats = mem.get(2)
    assert len(stats.responses) == 4
    assert len(stats.requests) == 4


def test_reliability_uses_windowed_request_and_response_counts():
    mem = WindowedPeerMemory(window=5)
    for _ in range(5):
        mem.record_request(7)
    # only 2 of the 5 requests got a recorded response
    mem.record_response(7, success=True, latency=1.0)
    mem.record_response(7, success=False, latency=1.0)
    stats = mem.get(7)
    assert stats.reliability == 2 / 5


def test_interface_parity_with_peer_memory_for_delegation_policy():
    """WindowedPeerMemory must be a drop-in replacement for PeerMemory: DelegationPolicy.eu_peer
    only reads .success_rate, .mean_latency, .reliability off whatever peer_memory.get() returns."""
    from src.delegation.policy import DelegationPolicy
    from src.reasoning.contracts import Observation, Candidate

    mem = WindowedPeerMemory(window=10)
    mem.record_request(0)
    mem.record_response(0, success=True, latency=0.5)

    policy = DelegationPolicy()
    obs = Observation(
        robot_id=1,
        candidates=[Candidate(id=0, true_cost=1.0), Candidate(id=1, true_cost=2.0)],
        det_estimate_costs=[1.0, 2.0],
        det_estimate_latency=0.1,
        ambiguity=0.5,
        urgency=0.0,
        candidate_count=2,
        cost_margin=1.0,
        battery_soc=1.0,
        recent_latency_mean=0.0,
        is_hard=False,
    )
    eu = policy.eu_peer(obs, mem.get(0))
    assert isinstance(eu, float)


def test_reset_round_load_is_a_noop_and_does_not_raise():
    mem = WindowedPeerMemory(window=5)
    mem.record_request(0)
    mem.reset_round_load()  # must not error; kept only for interface parity with PeerMemory
