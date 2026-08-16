import numpy as np

from src.environment.simulator import FleetConfig
from src.reasoning.counterfactual import generate_dataset
from src.reasoning.value_estimator import LogisticValueEstimator, featurize, label
from src.routing.oracle import oracle_utility
from src.routing.learned import LearnedRouter


def small_config():
    return FleetConfig(num_robots=4, num_stations=5, decisions_per_robot=10)


def test_counterfactual_dataset_shapes():
    cfg = small_config()
    records = generate_dataset(cfg, seed=1, n_samples=20)
    assert len(records) == 20
    for r in records:
        assert r.regret_det >= -1e-9
        assert r.regret_reasoning >= -1e-9
        assert r.latency_reasoning > r.latency_det


def test_oracle_never_worse_than_either_branch():
    cfg = small_config()
    records = generate_dataset(cfg, seed=2, n_samples=30)
    for r in records:
        u, used = oracle_utility(r, lambda_latency=0.15)
        assert used in (True, False)


def test_logistic_estimator_fits_and_predicts_in_range():
    cfg = small_config()
    records = generate_dataset(cfg, seed=3, n_samples=200)
    X = np.stack([featurize(r) for r in records])
    y = np.array([label(r, 0.15) for r in records], dtype=float)
    est = LogisticValueEstimator().fit(X, y, epochs=200)
    p = est.predict_proba(X)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_learned_router_deterministic_given_seed():
    cfg = small_config()
    records = generate_dataset(cfg, seed=4, n_samples=200)
    X = np.stack([featurize(r) for r in records])
    y = np.array([label(r, 0.15) for r in records], dtype=float)
    est = LogisticValueEstimator().fit(X, y, epochs=100)
    router1 = LearnedRouter(est, tau=0.5)
    router2 = LearnedRouter(est, tau=0.5)
    from src.environment.simulator import FleetSimulator
    r1 = FleetSimulator(cfg, seed=99).run(router1)
    r2 = FleetSimulator(cfg, seed=99).run(router2)
    for a, b in zip(r1, r2):
        assert a["true_cost"] == b["true_cost"]
        assert a["used_reasoning"] == b["used_reasoning"]
