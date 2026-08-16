"""Phase 9 revised (RACA-Collective.txt Section 22): add an N=100 scaling condition.

RACA-Collective.txt Section 22 lists N=5/10/20/50/100 as example fleet sizes; the original
Phase 9 run (`run_scaling.py`) only tested N in {10, 25, 50}. This script adds N=100 on a
fresh, disjoint seed pool, reusing `run_episode` from `run_scaling.py` unchanged (same
mechanism: post-RNG-fix simulator, DelegationPolicy defaults, C3 local-memory-only,
400 decisions/robot -- same budget as the original N=10/25/50 run, kept identical so the
N=100 point is directly comparable rather than confounded by a changed episode length).

Runtime check before committing to this seed budget: a single N=100 episode (400
decisions/robot, same as the original run) took ~8 seconds on this machine, so 4 seeds is
~32 seconds -- well within budget, matching the original run's per-size seed count (4).

Seed pool: 60000-60003 (4 seeds), fresh and disjoint from every `range(...)`-based seed pool
used anywhere in `experiments/**/*.py` (highest prior endpoint found by grep: 42020, Phase O's
peer_capacity ablation).

Usage: python -m experiments.phase_i.run_scaling_n100
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.phase_i.run_scaling import run_episode, gini

FLEET_SIZE = 100
SEEDS_PER_SIZE = 4
BASE_SEED = 60000  # fresh pool, disjoint from all others


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    episodes = [run_episode(FLEET_SIZE, BASE_SEED + i) for i in range(SEEDS_PER_SIZE)]

    with open(os.path.join(results_dir, "scaling_n100_raw.json"), "w") as f:
        json.dump(episodes, f, indent=2)

    summary = {
        "n_robots": FLEET_SIZE,
        "n_seeds": len(episodes),
        "seeds": [BASE_SEED + i for i in range(SEEDS_PER_SIZE)],
        "mean_utility_mean": float(np.mean([e["mean_utility"] for e in episodes])),
        "delegate_in_gini_mean": float(np.mean([e["delegate_in_gini"] for e in episodes])),
        "delegate_in_gini_std": float(np.std([e["delegate_in_gini"] for e in episodes])),
        "messages_per_decision_mean": float(np.mean([e["messages_per_decision"] for e in episodes])),
        "message_latency_per_decision_mean": float(np.mean([e["message_latency_per_decision"] for e in episodes])),
    }
    with open(os.path.join(results_dir, "scaling_n100_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
