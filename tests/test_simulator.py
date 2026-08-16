import numpy as np

from src.environment.simulator import FleetSimulator, FleetConfig
from src.routing.baselines import AlwaysDeterministic, AlwaysExpensiveReasoning
from src.routing.heuristic import HeuristicRouter


def small_config():
    return FleetConfig(num_robots=3, num_stations=5, decisions_per_robot=10)


def test_simulator_determinism_same_seed():
    cfg = small_config()
    sim1 = FleetSimulator(cfg, seed=42)
    sim2 = FleetSimulator(cfg, seed=42)
    r1 = sim1.run(AlwaysDeterministic())
    r2 = sim2.run(AlwaysDeterministic())
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a["true_cost"] == b["true_cost"]
        assert a["latency"] == b["latency"]
        assert a["failed"] == b["failed"]


def test_simulator_different_seeds_differ():
    cfg = small_config()
    r1 = FleetSimulator(cfg, seed=1).run(AlwaysDeterministic())
    r2 = FleetSimulator(cfg, seed=2).run(AlwaysDeterministic())
    costs1 = [r["true_cost"] for r in r1]
    costs2 = [r["true_cost"] for r in r2]
    assert costs1 != costs2


def test_record_count():
    cfg = small_config()
    records = FleetSimulator(cfg, seed=0).run(AlwaysDeterministic())
    assert len(records) == cfg.num_robots * cfg.decisions_per_robot


def test_always_reasoning_has_higher_latency_than_always_deterministic():
    cfg = small_config()
    rec_det = FleetSimulator(cfg, seed=7).run(AlwaysDeterministic())
    rec_llm = FleetSimulator(cfg, seed=7).run(AlwaysExpensiveReasoning())
    lat_det = np.mean([r["latency"] for r in rec_det])
    lat_llm = np.mean([r["latency"] for r in rec_llm])
    assert lat_llm > lat_det


def test_heuristic_router_calls_reasoning_sometimes_not_always():
    cfg = FleetConfig(num_robots=5, num_stations=8, decisions_per_robot=40)
    records = FleetSimulator(cfg, seed=3).run(HeuristicRouter())
    rate = sum(1 for r in records if r["used_reasoning"]) / len(records)
    assert 0.0 < rate < 1.0


def test_regret_nonnegative():
    cfg = small_config()
    records = FleetSimulator(cfg, seed=9).run(AlwaysDeterministic())
    for r in records:
        assert r["regret"] >= -1e-9


def test_battery_failure_resets_soc():
    cfg = FleetConfig(num_robots=2, num_stations=4, decisions_per_robot=200, battery_drain_scale=0.5)
    sim = FleetSimulator(cfg, seed=5)
    records = sim.run(AlwaysDeterministic())
    assert any(r["failed"] for r in records)
    for r in records:
        assert 0.0 <= r["battery_soc"] <= 1.0 + 1e-9
