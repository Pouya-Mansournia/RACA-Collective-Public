"""Phase 1 revised: rerun B0/B1/B2 under the new difficulty-axis cost regime
(src/environment/simulator.py FleetConfig difficulty_hard_fraction/hard_spread_multiplier/
easy_spread_multiplier, see docs/PHASE_2_REPORT.md addendum), with a fresh seed pool disjoint
from the original 1000-1019.

Usage: python -m experiments.phase_a.run_baselines_revised
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

SEEDS = list(range(11000, 11020))  # fresh pool, +10000 shift from original 1000-1019

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
    all_records = {}
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
            all_records[key] = per_seed_summaries
            with open(os.path.join(RESULTS_DIR, f"revised_{key}.json"), "w") as f:
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


if __name__ == "__main__":
    all_records = run_all()
    agg = aggregate(all_records)
    with open(os.path.join(os.path.dirname(__file__), "summary_revised.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))
