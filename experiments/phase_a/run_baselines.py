"""Phase 1 / Phase A: run B0, B1, B2 across seeds and configs, save raw +
aggregated results.

Usage: python3 -m experiments.phase_a.run_baselines
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetSimulator, FleetConfig
from src.routing.baselines import AlwaysDeterministic, AlwaysExpensiveReasoning
from src.routing.heuristic import HeuristicRouter
from src.evaluation.metrics import summarize, bootstrap_ci

SEEDS = list(range(1000, 1020))  # 20 independent seeds

CONFIGS = {
    "small_dense": FleetConfig(num_robots=8, num_stations=6, decisions_per_robot=80,
                                min_candidates=2, max_candidates=4, grid_size=20.0),
    "large_sparse": FleetConfig(num_robots=25, num_stations=20, decisions_per_robot=80,
                                 min_candidates=3, max_candidates=8, grid_size=80.0),
}

ROUTERS = {
    "B0_always_deterministic": lambda: AlwaysDeterministic(),
    "B1_always_reasoning": lambda: AlwaysExpensiveReasoning(),
    "B2_heuristic": lambda: HeuristicRouter(),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_records = {}  # (config, router) -> list of per-seed summaries
    raw = {}
    for cfg_name, cfg in CONFIGS.items():
        for router_name, router_factory in ROUTERS.items():
            key = f"{cfg_name}__{router_name}"
            per_seed_summaries = []
            for seed in SEEDS:
                sim = FleetSimulator(cfg, seed=seed)
                router = router_factory()
                records = sim.run(router)
                s = summarize(records)
                s["seed"] = seed
                per_seed_summaries.append(s)
            raw[key] = per_seed_summaries
            all_records[key] = per_seed_summaries
            with open(os.path.join(RESULTS_DIR, f"{key}.json"), "w") as f:
                json.dump(per_seed_summaries, f, indent=2)
    return all_records


def aggregate(all_records: dict) -> dict:
    agg = {}
    for key, summaries in all_records.items():
        metrics = ["completion_rate", "mean_cost", "mean_regret", "mean_latency", "reasoning_call_rate"]
        entry = {}
        for m in metrics:
            vals = [s[m] for s in summaries]
            point, lo, hi = bootstrap_ci(vals, seed=0)
            entry[m] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
        agg[key] = entry
    return agg


def check_reproducibility():
    """Explicitly verify same-seed reruns give identical results."""
    cfg = CONFIGS["small_dense"]
    router1 = AlwaysDeterministic()
    router2 = AlwaysDeterministic()
    r1 = FleetSimulator(cfg, seed=4242).run(router1)
    r2 = FleetSimulator(cfg, seed=4242).run(router2)
    identical = all(
        a["true_cost"] == b["true_cost"] and a["latency"] == b["latency"] and a["failed"] == b["failed"]
        for a, b in zip(r1, r2)
    )
    return identical


if __name__ == "__main__":
    all_records = run_all()
    agg = aggregate(all_records)
    with open(os.path.join(os.path.dirname(__file__), "summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    repro = check_reproducibility()
    with open(os.path.join(os.path.dirname(__file__), "reproducibility_check.json"), "w") as f:
        json.dump({"identical_reruns": repro}, f, indent=2)
    print("Reproducibility check (same seed identical):", repro)
    print(json.dumps(agg, indent=2))
