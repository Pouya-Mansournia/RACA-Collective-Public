import numpy as np

from src.delegation.peer_memory import PeerMemory
from src.delegation.policy import DelegationPolicy
from src.environment.simulator import FleetConfig
from src.reasoning.contracts import Observation, Candidate
from src.reasoning.counterfactual import generate_dataset
from src.reasoning.value_estimator import LogisticValueEstimator, featurize, label
from src.routing.centralized import CentralizedManager, CentralCapability
from src.routing.full_system import FullSystemRouter


def small_config():
    return FleetConfig(num_robots=4, num_stations=5, decisions_per_robot=10)


def trained_estimator(seed=1):
    cfg = small_config()
    records = generate_dataset(cfg, seed=seed, n_samples=200)
    X = np.stack([featurize(r) for r in records])
    y = np.array([label(r, 0.15) for r in records], dtype=float)
    return LogisticValueEstimator().fit(X, y, epochs=200)


def make_observation(robot_id, ambiguity, cost_margin=1.0, battery_soc=1.0):
    candidates = [Candidate(id=0, true_cost=5.0), Candidate(id=1, true_cost=8.0)]
    return Observation(
        robot_id=robot_id, candidates=candidates, det_estimate_costs=[5.0, 8.0],
        det_estimate_latency=0.1, ambiguity=ambiguity, urgency=0.0, candidate_count=2,
        cost_margin=cost_margin, battery_soc=battery_soc, recent_latency_mean=0.0,
    )


def test_centralized_manager_sees_all_robots_in_one_batch():
    est = trained_estimator()
    manager = CentralizedManager(est, tau=-1.0)  # tau below any p forces "reason" branch for everyone
    obs = {rid: make_observation(rid, ambiguity=0.9) for rid in range(4)}
    assignments = manager.assign_round(obs)
    assert set(assignments.keys()) == {0, 1, 2, 3}
    for action, target in assignments.values():
        assert action in ("reason_self", "reassign")


def test_centralized_manager_below_tau_uses_deterministic():
    est = trained_estimator()
    manager = CentralizedManager(est, tau=2.0)  # no probability can exceed 2.0
    obs = {rid: make_observation(rid, ambiguity=0.9) for rid in range(4)}
    assignments = manager.assign_round(obs)
    for action, target in assignments.values():
        assert action == "deterministic"
        assert target is None


def test_centralized_manager_reassigns_to_higher_capability_robot():
    est = trained_estimator()
    manager = CentralizedManager(est, tau=-1.0, capacity_per_robot=4, self_bias=0.0)
    # robot 1 has a strong, centrally-recorded track record; robot 0 has none
    for _ in range(20):
        manager.record_outcome(1, success=True)
    obs = {0: make_observation(0, ambiguity=0.9), 1: make_observation(1, ambiguity=0.9)}
    assignments = manager.assign_round(obs)
    # robot 0's decision should be reassigned to robot 1 (the fleet-wide best candidate),
    # not decided locally -- this is the behavior a purely distributed mechanism cannot
    # produce, since no single robot's local PeerMemory could see robot 1's central record
    # without an explicit query/response round trip
    assert assignments[0] == ("reassign", 1)


def test_centralized_manager_respects_capacity_cap():
    est = trained_estimator()
    manager = CentralizedManager(est, tau=-1.0, capacity_per_robot=1, self_bias=0.0)
    for _ in range(20):
        manager.record_outcome(3, success=True)
    obs = {rid: make_observation(rid, ambiguity=0.9) for rid in range(4)}
    assignments = manager.assign_round(obs)
    reassigned_to_3 = sum(1 for a, t in assignments.values() if a == "reassign" and t == 3)
    assert reassigned_to_3 <= 1  # capacity_per_robot=1 caps how many decisions robot 3 absorbs


def test_full_system_router_skips_below_tau():
    est = trained_estimator()
    policy = DelegationPolicy(lambda_latency=0.15)
    router = FullSystemRouter(est, tau=2.0, policy=policy)
    obs = make_observation(0, ambiguity=0.9)
    action, peer = router.decide(0, obs, PeerMemory(), [0, 1, 2], {})
    assert action == "skip"
    assert peer is None


def test_full_system_router_reasons_above_tau_self_or_delegate():
    est = trained_estimator()
    policy = DelegationPolicy(lambda_latency=0.15)
    router = FullSystemRouter(est, tau=-1.0, policy=policy)
    obs = make_observation(0, ambiguity=0.9)
    action, peer = router.decide(0, obs, PeerMemory(), [0, 1, 2], {})
    assert action in ("self", "delegate")
    if action == "delegate":
        assert peer in (1, 2)


def test_full_system_router_is_distributed_not_centralized():
    """Each robot's decision only depends on its own Observation and its own
    PeerMemory, unlike CentralizedManager which requires every robot's Observation
    up front. This is the architectural distinction A7 vs A2 is meant to test."""
    est = trained_estimator()
    policy = DelegationPolicy(lambda_latency=0.15)
    router = FullSystemRouter(est, tau=-1.0, policy=policy)
    obs0 = make_observation(0, ambiguity=0.9)
    pm0 = PeerMemory()
    action_a, _ = router.decide(0, obs0, pm0, [0, 1, 2], {})
    action_b, _ = router.decide(0, obs0, pm0, [0, 1, 2], {})
    assert action_a == action_b  # deterministic given the same local state, no fleet-wide info needed


def test_central_capability_uninformative_prior():
    cap = CentralCapability()
    assert cap.success_rate == 0.5
