"""Phase 2 / Phase B: counterfactual dataset + offline oracle comparison
against B0, B1, B2.

Usage: python3 -m experiments.phase_b.run_oracle_comparison
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetConfig
from src.reasoning.counterfactual import generate_dataset, utility
from src.routing.oracle import oracle_utility
from src.evaluation.metrics import bootstrap_ci

SEEDS = list(range(2000, 2020))  # 20 seeds, disjoint from phase_a seed pool
LAMBDA_LATENCY = 0.15
N_SAMPLES_PER_SEED = 400

CONFIGS = {
    "small_dense": FleetConfig(num_robots=8, num_stations=6, decisions_per_robot=80,
                                min_candidates=2, max_candidates=4, grid_size=20.0),
    "large_sparse": FleetConfig(num_robots=25, num_stations=20, decisions_per_robot=80,
                                 min_candidates=3, max_candidates=8, grid_size=80.0),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def policy_utilities(records, lambda_latency):
    u_b0 = [utility(r.regret_det, r.latency_det, lambda_latency) for r in records]
    u_b1 = [utility(r.regret_reasoning, r.latency_reasoning, lambda_latency) for r in records]
    u_b2 = [
        utility(r.regret_reasoning, r.latency_reasoning, lambda_latency) if r.heuristic_used_reasoning
        else utility(r.regret_det, r.latency_det, lambda_latency)
        for r in records
    ]
    oracle_pairs = [oracle_utility(r, lambda_latency) for r in records]
    u_oracle = [p[0] for p in oracle_pairs]
    oracle_escalation_rate = sum(1 for p in oracle_pairs if p[1]) / len(oracle_pairs)
    return {"B0": u_b0, "B1": u_b1, "B2": u_b2, "Oracle": u_oracle}, oracle_escalation_rate


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}
    for cfg_name, cfg in CONFIGS.items():
        all_records = []
        for seed in SEEDS:
            all_records.extend(generate_dataset(cfg, seed=seed, n_samples=N_SAMPLES_PER_SEED,
                                                  lambda_latency=LAMBDA_LATENCY))
        with open(os.path.join(RESULTS_DIR, f"{cfg_name}_counterfactual_raw.json"), "w") as f:
            json.dump([r.__dict__ for r in all_records], f)

        utils, oracle_escalation_rate = policy_utilities(all_records, LAMBDA_LATENCY)
        entry = {"oracle_escalation_rate": oracle_escalation_rate, "n_samples": len(all_records)}
        for name, vals in utils.items():
            point, lo, hi = bootstrap_ci(vals, seed=0)
            entry[f"utility_{name}"] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
        # value of reasoning = Oracle - B0 (headroom that ANY routing could capture)
        vor_vals = [o - d for o, d in zip(utils["Oracle"], utils["B0"])]
        point, lo, hi = bootstrap_ci(vor_vals, seed=1)
        entry["value_of_reasoning_oracle_minus_b0"] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
        b2_vs_b0_vals = [b2 - b0 for b2, b0 in zip(utils["B2"], utils["B0"])]
        point, lo, hi = bootstrap_ci(b2_vs_b0_vals, seed=2)
        entry["b2_minus_b0"] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
        summary[cfg_name] = entry

    with open(os.path.join(os.path.dirname(__file__), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
