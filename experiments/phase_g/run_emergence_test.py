"""Phase 7: emergence test and artifact controls (RACA-Collective.txt Sections 18/20).

Re-runs the Phase 6 specialization setup under three controls:
  1. randomized robot IDs (labels shuffled after each round, structure re-analyzed
     under the shuffled labels -- same underlying dynamics, different ID assignment)
  2. shuffled per-round decision order (instead of fixed round-robin 0..9)
  3. alternate tie-breaking rule in peer selection is NOT separately implementable
     without changing DelegationPolicy (see report -- the actual mechanism is
     near-continuous EU values, so exact ties essentially never occur; this control
     is addressed qualitatively in the report rather than by a second code path).

If the concentration/persistence structure found in Phase 6 (delegate_in Gini ~0.74,
role persistence ~0.90, robot 0 as hub in 5/6 seeds) survives ID randomization and
order shuffling in SHAPE (similar Gini/persistence) but the specific robot playing the
hub role changes, that is evidence for genuine (if modest) structure, not an ID
artifact. If instead the concentration or the specific hub identity collapses under
these controls, the Phase 6 structure is an artifact.

Usage: python -m experiments.phase_g.run_emergence_test
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.phase_f.run_specialization import (
    SEEDS, N_ROBOTS, run_episode, analyze,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_control_randomized_ids():
    out = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 800000)
        perm = rng.permutation(N_ROBOTS)
        robot_id_map = {raw: int(perm[raw]) for raw in range(N_ROBOTS)}
        activity = run_episode(seed, robot_id_map=robot_id_map)
        out[str(seed)] = analyze(activity)
    return out


def run_control_shuffled_order():
    out = {}
    for seed in SEEDS:
        activity = run_episode(seed, shuffle_order_seed=seed + 810000)
        out[str(seed)] = analyze(activity)
    return out


def cross_seed(all_analysis):
    ginis = [all_analysis[str(s)]["delegate_in_gini"] for s in SEEDS]
    persistences = [all_analysis[str(s)]["mean_role_persistence_rank_corr"] for s in SEEDS]
    max_robots = [all_analysis[str(s)]["max_delegate_in_robot"] for s in SEEDS]
    return {
        "delegate_in_gini_mean": float(np.mean(ginis)), "delegate_in_gini_std": float(np.std(ginis)),
        "role_persistence_mean": float(np.mean(persistences)), "role_persistence_std": float(np.std(persistences)),
        "max_delegate_in_robot_per_seed": max_robots,
        "max_delegate_in_robot_identical_across_seeds": len(set(max_robots)) == 1,
    }


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    randomized = run_control_randomized_ids()
    shuffled = run_control_shuffled_order()
    summary = {
        "randomized_ids": cross_seed(randomized),
        "shuffled_order": cross_seed(shuffled),
    }
    with open(os.path.join(RESULTS_DIR, "randomized_ids_summary.json"), "w") as f:
        json.dump(randomized, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "shuffled_order_summary.json"), "w") as f:
        json.dump(shuffled, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "artifact_control_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
