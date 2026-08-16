"""Phase O part 3 -- Gini response to peer_capacity, an ablation of the mechanism's
"reinforcement strength" parameter.

`DelegationPolicy.peer_capacity` (default 2) caps how many delegation requests a robot
will accept within a single round; once a robot has already received `peer_capacity`
delegations in the current round, it looks "unavailable" to every later decider that
round. This is the one explicit resource-constraint parameter in the mechanism that
plausibly controls how strongly an early advantage can compound: with peer_capacity=1,
a robot that wins one tie-break in a round immediately blocks every other robot from
also delegating to it that round, forcing traffic to spread to other (possibly
untried, tied) peers; with a very large peer_capacity, a single early-favored peer can
absorb essentially unlimited delegation traffic every round without ever being blocked,
which is the closest thing this mechanism has to an unconstrained "richer gets ever
richer with no cap" regime.

This is not a claim that peer_capacity IS a Polya-urn reinforcement-strength parameter
in the classical sense (see docs/PHASE_7_REPORT.md's new "Mathematical analysis of the
self-reinforcing loop" section for why this system is an argmax/tie-break process, not
a proportional-reinforcement urn) -- it is the most direct available analogue: the
per-round cap that determines how much of a round's delegation traffic one peer can
absorb once it has established a lead. Prediction: Gini should INCREASE monotonically
(or near-monotonically) as peer_capacity increases, since the mechanism relaxing this
cap allows any transient lead to be exploited more completely each round.

Method: fresh, disjoint 20-seed pool (42000-42019; verified disjoint by grep against
every `range(...)`-based seed pool used anywhere in experiments/**/*.py -- the highest
prior endpoint found was 40030, from run_injection_depth_test.py's range(40000, 40030)).
For each seed and each peer_capacity in {1, 2, 3, 5, 10}, run the identical
fully-controlled episode (randomized robot IDs, shuffled processing order, randomized
tie-break -- same machinery as run_large_n_resolution.run_episode_full_control) with a
DelegationPolicy configured with that peer_capacity, and record delegate-in Gini and
role persistence. Report mean +/- 95% bootstrap CI per capacity level (numpy only).

peer_capacity=10 == N_ROBOTS, i.e. effectively unbounded (no robot can ever be
capacity-blocked in a 10-robot fleet within a single round).

Usage: python -m experiments.phase_o.run_peer_capacity_ablation
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.delegation.policy import DelegationPolicy

from experiments.phase_f.run_specialization_revised import N_ROBOTS, LAMBDA_LATENCY, analyze
from experiments.phase_m.run_large_n_resolution import run_episode_full_control, bootstrap_ci

SEEDS = list(range(42000, 42020))  # 20 fresh seeds, disjoint from every prior pool (max prior endpoint 40030)
CAPACITIES = [1, 2, 3, 5, 10]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_capacity_stats = {c: {"gini": [], "persistence": []} for c in CAPACITIES}
    per_seed = {}

    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for cap in CAPACITIES:
            policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY, peer_capacity=cap)
            activity, _ = run_episode_full_control(seed, policy, track_early_luck=False)
            a = analyze(activity)
            per_capacity_stats[cap]["gini"].append(a["delegate_in_gini"])
            per_capacity_stats[cap]["persistence"].append(a["mean_role_persistence_rank_corr"])
            per_seed[str(seed)][str(cap)] = {
                "delegate_in_gini": a["delegate_in_gini"],
                "persistence": a["mean_role_persistence_rank_corr"],
                "max_delegate_in_robot": a["max_delegate_in_robot"],
            }

    summary = {
        "seed_pool": {"start": SEEDS[0], "end": SEEDS[-1], "count": len(SEEDS)},
        "capacities_tested": CAPACITIES,
        "by_capacity": {
            str(cap): {
                "gini_ci": bootstrap_ci(per_capacity_stats[cap]["gini"], seed=cap + 100),
                "persistence_ci": bootstrap_ci(per_capacity_stats[cap]["persistence"], seed=cap + 200),
            }
            for cap in CAPACITIES
        },
    }

    ginis_by_cap = [summary["by_capacity"][str(c)]["gini_ci"]["mean"] for c in CAPACITIES]
    monotonic_nondecreasing = all(ginis_by_cap[i] <= ginis_by_cap[i + 1] + 1e-9 for i in range(len(ginis_by_cap) - 1))
    summary["gini_means_by_capacity"] = dict(zip([str(c) for c in CAPACITIES], ginis_by_cap))
    summary["gini_monotonic_nondecreasing_in_capacity"] = bool(monotonic_nondecreasing)

    with open(os.path.join(RESULTS_DIR, "peer_capacity_ablation_per_seed.json"), "w") as f:
        json.dump(per_seed, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "peer_capacity_ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
