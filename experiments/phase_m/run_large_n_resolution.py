"""Phase M -- large-N, null-model-controlled resolution of the residual specialization
structure left open by docs/PHASE_6_REPORT.md and docs/PHASE_7_REPORT.md ("Revised (RNG
fix)" sections: delegate-in Gini ~0.47, role persistence ~0.86, survived randomized robot
IDs, shuffled processing order, and randomized tie-breaking simultaneously, but only tested
on 6 seeds -- too small a budget to separate "real signal" from "noise").

This script does three things, all on a single fresh, disjoint seed pool:

1. STATISTICAL POWER: reruns the fully-controlled condition (randomized robot IDs +
   shuffled processing order + randomized tie-breaking, ALL THREE simultaneously, exactly
   as Phase 7 revised's individual controls did one at a time) on 50 seeds instead of 6,
   and reports 95% bootstrap confidence intervals (10000 resamples, numpy only) for Gini,
   HHI, mean entropy, and role persistence.

2. NULL-MODEL COMPARISON: reruns the identical setup but with delegation targets chosen
   uniformly at random among available/willing peers instead of by expected utility
   (src/delegation/null_policy.py's NullDelegationPolicy), on the SAME 50 seeds, with
   bootstrap CIs for the same statistics. If the real-system CI and the null-model CI
   overlap, the residual structure is not distinguishable from pure chance. If the
   real-system CI is reliably above the null-model CI, that rules out pure chance.

3. RICH-GET-RICHER MECHANISM CHECK: for the real (fully-controlled) run, correlates each
   robot's early success rate (successes among its delegate-in events falling in the first
   10% of the episode's decision rounds) against its final delegate-in share, pooled across
   robots and seeds, using both Pearson and Spearman correlation (numpy only).

Usage: python -m experiments.phase_m.run_large_n_resolution
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetSimulator
from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult
from src.communication.messaging import MessageBus
from src.delegation.policy import DelegationPolicy
from src.delegation.null_policy import NullDelegationPolicy
from src.memory.conditions import MemoryConditionConfig, make_peer_memories

from experiments.phase_f.run_specialization_revised import (
    N_ROBOTS, DECISIONS_PER_ROBOT, CONFIG, N_WINDOWS, LAMBDA_LATENCY,
    gini, hhi, entropy, analyze,
)

# Fresh, disjoint seed pool. Every seed range used anywhere else in this repository was
# grepped (see docs/RESEARCH_LOG.md entry for this phase for the full list); the highest
# endpoint found was 25915 (experiments/phase_l/run_ablation_full.py's B3_CAL_SEEDS,
# range(25910, 25915)). 30000-30049 does not overlap any of: 1000-1020, 2000-2020,
# 3000-3010/3100-3110/3200-3210/3300-3320, 11000-11020, 12000-12020,
# 13000-13010/13100-13110/13200-13210/13300-13320, 14000-14015, 15000-15015,
# 16000-16006, 18000-18010, 19000-19015, 20000-20015, 21000-21006, 23000-23006,
# 23100-23108, 23900-23915, 25100-25110, 25900-25915.
SEEDS = list(range(30000, 30050))  # 50 fresh seeds, disjoint from every prior pool
N_BOOTSTRAP = 10000
EARLY_FRACTION = 0.10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_episode_full_control(seed: int, policy, track_early_luck: bool = False):
    """Runs the same episode structure as experiments/phase_f/run_specialization_revised's
    run_episode, but with ALL THREE artifact controls (randomized robot IDs, shuffled
    processing order, randomized tie-breaking) applied SIMULTANEOUSLY, and an injectable
    policy object (real DelegationPolicy or NullDelegationPolicy) so both the real system
    and the null model run through byte-identical simulator/backend/memory/battery/task
    machinery. Optionally tracks, per delegate-in event, the decision round index and
    success, for the rich-get-richer correlation check."""
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 900000)

    # control 1: randomized robot IDs (post-hoc relabeling of raw robot identity)
    id_rng = np.random.default_rng(seed + 800000)
    perm = id_rng.permutation(N_ROBOTS)
    robot_id_map = {raw: int(perm[raw]) for raw in range(N_ROBOTS)}

    # control 2: shuffled per-round processing order
    order_rng = np.random.default_rng(seed + 810000)

    # control 3: randomized tie-breaking (peer evaluation order shuffled per decision)
    tie_break_rng = np.random.default_rng(seed + 950000)

    cond = MemoryConditionConfig(name="C3_local_memory_only")
    peer_memories = make_peer_memories(N_ROBOTS, cond)

    window_len = DECISIONS_PER_ROBOT // N_WINDOWS
    activity = np.zeros((N_WINDOWS, N_ROBOTS, 4), dtype=int)

    # delegate_in_events[mapped_robot_id] = list of (round_idx, success)
    delegate_in_events = {r: [] for r in range(N_ROBOTS)} if track_early_luck else None

    for round_idx in range(DECISIONS_PER_ROBOT):
        window = min(round_idx // window_len, N_WINDOWS - 1)
        peer_loads = {rid: 0 for rid in range(N_ROBOTS)}
        order = list(range(N_ROBOTS))
        order_rng.shuffle(order)
        for raw_rid in order:
            rid = robot_id_map[raw_rid]
            obs = sim.make_observation(raw_rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            peer_id_order = list(range(N_ROBOTS))
            tie_break_rng.shuffle(peer_id_order)
            action, peer_id = policy.decide(raw_rid, obs, peer_memories[raw_rid], peer_id_order, peer_loads)

            if action == "skip":
                idx, latency = det_idx, obs.det_estimate_latency
                regret = candidates[idx].true_cost - min_cost
                activity[window, rid, 3] += 1
            elif action == "self":
                idx_r, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[raw_rid])
                idx, latency = idx_r, obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
                activity[window, rid, 0] += 1
            else:
                peer_loads[peer_id] = peer_loads.get(peer_id, 0) + 1
                req = bus.send(raw_rid, peer_id, "delegation_request", len(candidates), rng_ctrl)
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[peer_id])
                resp = bus.send(peer_id, raw_rid, "delegation_response", 1, rng_ctrl)
                latency = obs.det_estimate_latency + req.latency + r_latency + resp.latency
                regret = candidates[idx].true_cost - min_cost
                success = regret <= (candidates[det_idx].true_cost - min_cost)
                peer_memories[raw_rid].record_request(peer_id)
                peer_memories[raw_rid].record_response(peer_id, success=success,
                                                          latency=req.latency + r_latency + resp.latency)
                activity[window, rid, 1] += 1
                peer_rid_mapped = robot_id_map[peer_id]
                activity[window, peer_rid_mapped, 2] += 1
                if track_early_luck:
                    delegate_in_events[peer_rid_mapped].append((round_idx, bool(success)))

            sim.apply_decision(raw_rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()

    return activity, delegate_in_events


def bootstrap_ci(values: list[float], n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Percentile bootstrap 95% CI over seed-level statistics. numpy only."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_seeds": int(n),
    }


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    real_stats = {"gini": [], "hhi": [], "entropy": [], "persistence": []}
    null_stats = {"gini": [], "hhi": [], "entropy": [], "persistence": []}
    early_success_rates = []
    final_shares = []

    real_per_seed = {}
    null_per_seed = {}

    for seed in SEEDS:
        real_policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
        activity_real, delegate_events = run_episode_full_control(seed, real_policy, track_early_luck=True)
        a = analyze(activity_real)
        real_stats["gini"].append(a["delegate_in_gini"])
        real_stats["hhi"].append(a["delegate_in_hhi"])
        real_stats["entropy"].append(a["mean_entropy"])
        real_stats["persistence"].append(a["mean_role_persistence_rank_corr"])
        real_per_seed[str(seed)] = a

        total_in = sum(a["delegate_in_counts"])
        early_cutoff = EARLY_FRACTION * DECISIONS_PER_ROBOT
        for r in range(N_ROBOTS):
            events = delegate_events[r]
            early = [succ for (rd, succ) in events if rd < early_cutoff]
            if len(early) == 0:
                continue  # this robot had no delegate-in events in the early window; excluded
            early_success_rate = float(np.mean(early))
            final_share = a["delegate_in_counts"][r] / total_in if total_in > 0 else 0.0
            early_success_rates.append(early_success_rate)
            final_shares.append(final_share)

        null_policy = NullDelegationPolicy(lambda_latency=LAMBDA_LATENCY, null_rng_seed=seed + 970000)
        activity_null, _ = run_episode_full_control(seed, null_policy, track_early_luck=False)
        a_null = analyze(activity_null)
        null_stats["gini"].append(a_null["delegate_in_gini"])
        null_stats["hhi"].append(a_null["delegate_in_hhi"])
        null_stats["entropy"].append(a_null["mean_entropy"])
        null_stats["persistence"].append(a_null["mean_role_persistence_rank_corr"])
        null_per_seed[str(seed)] = a_null

    real_ci = {k: bootstrap_ci(v, seed=1) for k, v in real_stats.items()}
    null_ci = {k: bootstrap_ci(v, seed=2) for k, v in null_stats.items()}

    real_gini_ci = real_ci["gini"]
    null_gini_ci = null_ci["gini"]
    gini_ci_overlap = not (real_gini_ci["ci_lo"] > null_gini_ci["ci_hi"] or null_gini_ci["ci_lo"] > real_gini_ci["ci_hi"])

    pearson_r = pearson(early_success_rates, final_shares) if early_success_rates else None
    spearman_r = spearman(early_success_rates, final_shares) if early_success_rates else None

    summary = {
        "seed_pool": {"start": SEEDS[0], "end": SEEDS[-1], "count": len(SEEDS)},
        "n_bootstrap": N_BOOTSTRAP,
        "real_system_fully_controlled": real_ci,
        "null_model_fully_controlled": null_ci,
        "gini_cis_overlap": gini_ci_overlap,
        "rich_get_richer_check": {
            "n_robot_seed_samples": len(early_success_rates),
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "early_fraction": EARLY_FRACTION,
        },
    }

    with open(os.path.join(RESULTS_DIR, "real_per_seed.json"), "w") as f:
        json.dump(real_per_seed, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "null_per_seed.json"), "w") as f:
        json.dump(null_per_seed, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "large_n_resolution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
