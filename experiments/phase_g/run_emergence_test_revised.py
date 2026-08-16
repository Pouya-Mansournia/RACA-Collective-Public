"""Phase 7 REVISED (RNG fix): emergence test and artifact controls, rerun after fixing
src/environment/simulator.py's per-robot RNG construction (see Phase 6/7 "Revised (RNG
fix)" report sections). Fresh seed pool (21000-21005), not the original 16000-16005 pool.

Three controls, matching RACA-Collective.txt Section 20:
  1. randomized robot IDs (post-hoc relabeling)
  2. shuffled per-round decision order
  3. alternate tie-breaking in peer selection -- unlike the original Phase 7 report,
     this control IS implemented here. Investigating the RNG-fix result surfaced a
     second, distinct artifact: DelegationPolicy.decide() iterates candidate peers in a
     fixed ascending ID order and keeps the first candidate on an exact expected-utility
     tie. Peers with no track record all carry identical uninformative-prior stats, so
     this is a real, frequent exact tie (not a rare floating-point coincidence), and the
     fixed order means ties are always won by the lowest untried robot ID. This control
     randomizes peer evaluation order per decision so ties are broken uniformly at
     random instead of by ascending ID.

Usage: python -m experiments.phase_g.run_emergence_test_revised
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.phase_f.run_specialization_revised import (
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


def run_control_tie_break_randomized():
    out = {}
    for seed in SEEDS:
        activity = run_episode(seed, tie_break_rng_seed=seed + 950000)
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
    baseline = {str(seed): analyze(run_episode(seed)) for seed in SEEDS}
    randomized = run_control_randomized_ids()
    shuffled = run_control_shuffled_order()
    tie_break = run_control_tie_break_randomized()
    summary = {
        "baseline_revised": cross_seed(baseline),
        "randomized_ids": cross_seed(randomized),
        "shuffled_order": cross_seed(shuffled),
        "tie_break_randomized": cross_seed(tie_break),
    }
    with open(os.path.join(RESULTS_DIR, "baseline_revised_summary.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "randomized_ids_summary_revised.json"), "w") as f:
        json.dump(randomized, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "shuffled_order_summary_revised.json"), "w") as f:
        json.dump(shuffled, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "tie_break_summary_revised.json"), "w") as f:
        json.dump(tie_break, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "artifact_control_summary_revised.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
