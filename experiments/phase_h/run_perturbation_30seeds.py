"""Phase 8 revised (RACA-Collective.txt Section 21): perturbation/reorganization test on a
larger, fresh, disjoint seed pool (30 seeds instead of the original 6).

Reuses every mechanism from `run_perturbation.py` unchanged (same RNG-fixed simulator,
calibrated... actually uncalibrated `DelegationPolicy` defaults matching the original Phase 8
choice, C3 local-memory-only condition, midpoint hub removal at round 400 of 800) -- only the
seed pool changes, and bootstrap 95% confidence intervals are added, which the original
6-seed report did not have (only raw mean/std).

Seed pool: 50000-50029 (30 seeds), fresh and disjoint from every `range(...)`-based seed pool
used anywhere in `experiments/**/*.py` (highest prior endpoint found by grep: 42020, Phase O's
peer_capacity ablation; the original Phase 8 pool was 21000-21005).

Usage: python -m experiments.phase_h.run_perturbation_30seeds
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.phase_h.run_perturbation import run_one_seed, REMOVE_AT_ROUND

SEEDS = list(range(50000, 50030))  # 30 fresh seeds, disjoint from all prior pools


def bootstrap_ci(values, n_resamples=10000, seed=12345):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    per_seed = [run_one_seed(seed) for seed in SEEDS]

    drops = [r["utility_drop"] for r in per_seed]
    gini_before = [r["delegate_in_gini_among_remaining_before"] for r in per_seed]
    gini_after = [r["delegate_in_gini_among_remaining_after"] for r in per_seed]
    n_reorganized = sum(1 for r in per_seed if r["reorganized"])

    drop_ci = bootstrap_ci(drops)
    gini_before_ci = bootstrap_ci(gini_before)
    gini_after_ci = bootstrap_ci(gini_after)

    summary = {
        "n_seeds": len(SEEDS),
        "seed_pool": [SEEDS[0], SEEDS[-1]],
        "remove_at_round": REMOVE_AT_ROUND,
        "per_seed": per_seed,
        "mean_utility_drop": float(np.mean(drops)),
        "std_utility_drop": float(np.std(drops)),
        "utility_drop_95ci_bootstrap": drop_ci,
        "mean_gini_among_remaining_before": float(np.mean(gini_before)),
        "mean_gini_among_remaining_before_95ci_bootstrap": gini_before_ci,
        "mean_gini_among_remaining_after": float(np.mean(gini_after)),
        "mean_gini_among_remaining_after_95ci_bootstrap": gini_after_ci,
        "n_seeds_reorganized_to_new_hub": n_reorganized,
        "fraction_reorganized": n_reorganized / len(SEEDS),
    }
    with open(os.path.join(results_dir, "perturbation_summary_30seeds.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
