"""Phase 9 (RACA-Collective.txt Section 22): scaling.

Runs the same specialization mechanism (post-RNG-fix simulator, DelegationPolicy
defaults, C3 local-memory-only) at three fleet sizes (10, 25, 50 robots). Seed budget is
reduced relative to Phase 6 (4 seeds per fleet size instead of 6) and decisions_per_robot
is reduced (400 instead of 800) to keep the N=50 runs computationally tractable within
this task; this is stated explicitly rather than silently changing methodology.

Reports, per fleet size: delegate-in Gini/persistence (does concentration strengthen or
weaken with scale), fleet mean utility, and communication volume (messages/decision) to
check whether communication cost grows faster than fleet size (a sign that distributed
delegation could stop scaling before a hypothetical centralized alternative would).

Usage: python -m experiments.phase_i.run_scaling
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

FLEET_SIZES = [10, 25, 50]
SEEDS_PER_SIZE = 4
BASE_SEED = 22000  # fresh pool, disjoint from all others; 22000+size*10+i per (size, seed_idx)
LAMBDA_LATENCY = 0.15
DECISIONS_PER_ROBOT = 400


def utility(regret, latency):
    return -regret - LAMBDA_LATENCY * latency


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def run_episode(n_robots: int, seed: int) -> dict:
    config = FleetConfig(num_robots=n_robots, num_stations=max(10, n_robots), decisions_per_robot=DECISIONS_PER_ROBOT,
                          min_candidates=2, max_candidates=5, grid_size=30.0 * (n_robots / 10.0) ** 0.5)
    sim = FleetSimulator(config, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 900000)
    cond = MemoryConditionConfig(name="C3_local_memory_only")
    peer_memories = make_peer_memories(n_robots, cond)

    delegate_in_counts = np.zeros(n_robots, dtype=int)
    utilities = []

    for _round in range(DECISIONS_PER_ROBOT):
        peer_loads = {rid: 0 for rid in range(n_robots)}
        for rid in range(n_robots):
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))
            action, peer_id = policy.decide(rid, obs, peer_memories[rid], list(range(n_robots)), peer_loads)

            if action == "skip":
                idx, latency = det_idx, obs.det_estimate_latency
                regret = candidates[idx].true_cost - min_cost
            elif action == "self":
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[rid])
                latency = obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
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
                delegate_in_counts[peer_id] += 1

            utilities.append(utility(regret, latency))
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()

    n_decisions = n_robots * DECISIONS_PER_ROBOT
    comm = bus.summary()
    return {
        "n_robots": n_robots, "seed": seed,
        "mean_utility": float(np.mean(utilities)),
        "delegate_in_gini": gini(delegate_in_counts),
        "messages_per_decision": comm["total_messages"] / n_decisions,
        "message_latency_per_decision": comm["total_message_latency"] / n_decisions,
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    for size in FLEET_SIZES:
        episodes = []
        for i in range(SEEDS_PER_SIZE):
            seed = BASE_SEED + size * 100 + i
            episodes.append(run_episode(size, seed))
        all_results[str(size)] = episodes

    with open(os.path.join(results_dir, "scaling_raw.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    summary = {}
    for size, episodes in all_results.items():
        summary[size] = {
            "n_seeds": len(episodes),
            "mean_utility_mean": float(np.mean([e["mean_utility"] for e in episodes])),
            "delegate_in_gini_mean": float(np.mean([e["delegate_in_gini"] for e in episodes])),
            "delegate_in_gini_std": float(np.std([e["delegate_in_gini"] for e in episodes])),
            "messages_per_decision_mean": float(np.mean([e["messages_per_decision"] for e in episodes])),
            "message_latency_per_decision_mean": float(np.mean([e["message_latency_per_decision"] for e in episodes])),
        }
    with open(os.path.join(results_dir, "scaling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
