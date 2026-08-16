"""Phase 8 (RACA-Collective.txt Section 21): perturbation / reorganization test.

Uses the same mechanism as the revised Phase 6/7 specialization runs (post-RNG-fix
simulator, DelegationPolicy defaults, C3 local-memory-only condition -- the
"mechanism-richest" configuration used throughout Phases 6/7, chosen there because it is
the condition where interaction-driven differentiation can occur at all, not because it
has the best raw utility).

For each of the 6 fresh specialization seeds (21000-21005), identify that seed's
delegate-in hub robot from the FIRST HALF of the episode (rounds 0-399), then for the
SECOND HALF (rounds 400-799) remove that robot from the fleet's set of eligible
delegation targets and from ever making its own decisions (a stand-in for "the hub goes
offline"): every other robot's remaining decisions are unaffected in mechanism, but they
can no longer delegate to the removed robot and it stops making self decisions too.
Measures per-window fleet utility before/after removal and whether the delegate-in
distribution among the REMAINING 9 robots concentrates on a new robot after removal
(reorganization) or stays diffuse / degrades (no reorganization).

Usage: python -m experiments.phase_h.run_perturbation
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetSimulator, FleetConfig
from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult
from src.communication.messaging import MessageBus
from src.delegation.policy import DelegationPolicy
from src.memory.conditions import MemoryConditionConfig, make_peer_memories

SEEDS = list(range(21000, 21006))  # SAME seeds as the revised Phase 6/7 specialization run
LAMBDA_LATENCY = 0.15
N_ROBOTS = 10
DECISIONS_PER_ROBOT = 800
REMOVE_AT_ROUND = 400  # midpoint
CONFIG = FleetConfig(num_robots=N_ROBOTS, num_stations=10, decisions_per_robot=DECISIONS_PER_ROBOT,
                      min_candidates=2, max_candidates=5, grid_size=30.0)
N_WINDOWS = 8  # finer windows to see before/after transition clearly (100 decisions/window)


def utility(regret, latency):
    return -regret - LAMBDA_LATENCY * latency


def find_hub_first_half(seed: int) -> int:
    """First pass: run rounds 0-399 unperturbed, return the delegate-in hub robot."""
    activity = run_segment(seed, remove_robot=None, n_rounds=REMOVE_AT_ROUND)[0]
    delegate_in_counts = activity.sum(axis=0)[:, 2]
    return int(np.argmax(delegate_in_counts))


def run_segment(seed: int, remove_robot: int | None, n_rounds: int, start_round: int = 0,
                 sim=None, policy=None, bus=None, peer_memories=None, rng_ctrl=None):
    """Runs n_rounds of decisions. If sim/policy/etc are None, fresh state is created
    (used for the first-half hub-identification pass). Returns (activity, state) where
    state is a dict of the mutable objects so a second call can continue seamlessly."""
    fresh = sim is None
    if fresh:
        sim = FleetSimulator(CONFIG, seed=seed)
        policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
        bus = MessageBus()
        cond = MemoryConditionConfig(name="C3_local_memory_only")
        peer_memories = make_peer_memories(N_ROBOTS, cond)
        rng_ctrl = np.random.default_rng(seed + 900000)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()

    window_len = max(1, DECISIONS_PER_ROBOT // N_WINDOWS)
    activity = np.zeros((n_rounds, N_ROBOTS, 4), dtype=int)  # per-round for fine-grained utility trace
    round_utility = np.zeros(n_rounds)

    eligible_peers = [r for r in range(N_ROBOTS) if r != remove_robot] if remove_robot is not None else list(range(N_ROBOTS))

    for local_round in range(n_rounds):
        peer_loads = {rid: 0 for rid in range(N_ROBOTS)}
        round_utils = []
        for rid in range(N_ROBOTS):
            if remove_robot is not None and rid == remove_robot:
                continue  # removed robot makes no decisions
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            action, peer_id = policy.decide(rid, obs, peer_memories[rid], eligible_peers, peer_loads)

            if action == "skip":
                idx, latency = det_idx, obs.det_estimate_latency
                regret = candidates[idx].true_cost - min_cost
                activity[local_round, rid, 3] += 1
            elif action == "self":
                idx_r, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[rid])
                idx, latency = idx_r, obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
                activity[local_round, rid, 0] += 1
            else:
                peer_loads[peer_id] = peer_loads.get(peer_id, 0) + 1
                req = bus.send(rid, peer_id, "delegation_request", len(candidates), rng_ctrl)
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[peer_id])
                resp = bus.send(peer_id, rid, "delegation_response", 1, rng_ctrl)
                latency = obs.det_estimate_latency + req.latency + r_latency + resp.latency
                regret = candidates[idx].true_cost - min_cost
                success = regret <= (candidates[det_idx].true_cost - min_cost)
                peer_memories[rid].record_request(peer_id)
                peer_memories[rid].record_response(peer_id, success=success, latency=req.latency + r_latency + resp.latency)
                activity[local_round, rid, 1] += 1
                activity[local_round, peer_id, 2] += 1

            u = utility(regret, latency)
            round_utils.append(u)
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        round_utility[local_round] = float(np.mean(round_utils)) if round_utils else 0.0
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()

    state = {"sim": sim, "policy": policy, "bus": bus, "peer_memories": peer_memories, "rng_ctrl": rng_ctrl}
    return activity, round_utility, state


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def run_one_seed(seed: int) -> dict:
    # Phase 1: run full episode WITHOUT perturbation to get the "would-be" hub identity
    # and baseline per-window utility, using a fresh independent copy of the sim state.
    act_full, util_full, _ = run_segment(seed, remove_robot=None, n_rounds=DECISIONS_PER_ROBOT)
    first_half_counts = act_full[:REMOVE_AT_ROUND].sum(axis=0)[:, 2]
    hub = int(np.argmax(first_half_counts))

    # Phase 2: rerun from scratch, perturbing (removing the hub) at REMOVE_AT_ROUND.
    act_before, util_before, state = run_segment(seed, remove_robot=None, n_rounds=REMOVE_AT_ROUND)
    act_after, util_after, _ = run_segment(
        seed, remove_robot=hub, n_rounds=DECISIONS_PER_ROBOT - REMOVE_AT_ROUND,
        sim=state["sim"], policy=state["policy"], bus=state["bus"],
        peer_memories=state["peer_memories"], rng_ctrl=state["rng_ctrl"],
    )

    window_len = max(1, DECISIONS_PER_ROBOT // N_WINDOWS)
    all_util = np.concatenate([util_before, util_after])
    window_utils = [float(np.mean(all_util[w * window_len:(w + 1) * window_len]))
                     for w in range(N_WINDOWS)]

    remaining = [r for r in range(N_ROBOTS) if r != hub]
    delegate_in_before = act_before.sum(axis=0)[:, 2]
    delegate_in_after = act_after.sum(axis=0)[:, 2]
    gini_before_remaining = gini(delegate_in_before[remaining])
    gini_after_remaining = gini(delegate_in_after[remaining])
    new_hub = int(remaining[int(np.argmax(delegate_in_after[remaining]))]) if delegate_in_after[remaining].sum() > 0 else None

    mean_util_before = float(np.mean(util_before))
    mean_util_after = float(np.mean(util_after))

    return {
        "seed": seed,
        "removed_hub_robot": hub,
        "mean_utility_before_removal": mean_util_before,
        "mean_utility_after_removal": mean_util_after,
        "utility_drop": mean_util_before - mean_util_after,  # positive = utility got worse (more negative)
        "window_utilities": window_utils,
        "delegate_in_gini_among_remaining_before": gini_before_remaining,
        "delegate_in_gini_among_remaining_after": gini_after_remaining,
        "new_top_delegate_in_robot_after_removal": new_hub,
        "reorganized": new_hub is not None and new_hub != hub,
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    per_seed = [run_one_seed(seed) for seed in SEEDS]

    drops = [r["utility_drop"] for r in per_seed]
    gini_before = [r["delegate_in_gini_among_remaining_before"] for r in per_seed]
    gini_after = [r["delegate_in_gini_among_remaining_after"] for r in per_seed]

    summary = {
        "n_seeds": len(SEEDS),
        "remove_at_round": REMOVE_AT_ROUND,
        "per_seed": per_seed,
        "mean_utility_drop": float(np.mean(drops)),
        "std_utility_drop": float(np.std(drops)),
        "mean_gini_among_remaining_before": float(np.mean(gini_before)),
        "mean_gini_among_remaining_after": float(np.mean(gini_after)),
        "n_seeds_reorganized_to_new_hub": sum(1 for r in per_seed if r["reorganized"]),
    }
    with open(os.path.join(results_dir, "perturbation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
