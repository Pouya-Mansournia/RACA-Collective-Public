"""Phase 5: shared experience / cognitive memory. Compares C0-C3 memory conditions on
top of the Phase 4 delegation policy and difficulty-axis simulator. Fresh seed pool.

Usage: python -m experiments.phase_e.run_memory
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
from src.memory.conditions import MemoryConditionConfig, make_peer_memories, maybe_share
from src.evaluation.metrics import bootstrap_ci

SEEDS = list(range(15000, 15015))  # fresh pool, disjoint from phase_d's 14000s
LAMBDA_LATENCY = 0.15
CONFIG = FleetConfig(num_robots=10, num_stations=10, decisions_per_robot=150,
                      min_candidates=2, max_candidates=5, grid_size=30.0)

CONDITIONS = {
    "C0_no_shared_memory": MemoryConditionConfig(name="C0_no_shared_memory", frozen=True),
    "C1_global_shared_memory": MemoryConditionConfig(name="C1_global_shared_memory", shared=True),
    "C2_peer_to_peer_selective": MemoryConditionConfig(name="C2_peer_to_peer_selective", share_prob=0.1),
    "C3_local_memory_only": MemoryConditionConfig(name="C3_local_memory_only"),
}


def utility(regret: float, latency: float) -> float:
    return -regret - LAMBDA_LATENCY * latency


def run_episode(seed: int, cond: MemoryConditionConfig) -> dict:
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 700000)

    n_robots = CONFIG.num_robots
    peer_memories = make_peer_memories(n_robots, cond)
    utilities, regrets = [], []
    per_robot_reasoning_calls = [0] * n_robots
    share_events = 0
    memory_entries_total = []

    for _round in range(CONFIG.decisions_per_robot):
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
                per_robot_reasoning_calls[rid] += 1
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[rid])
                latency = obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
            else:
                peer_loads[peer_id] = peer_loads.get(peer_id, 0) + 1
                per_robot_reasoning_calls[peer_id] += 1
                req = bus.send(rid, peer_id, "delegation_request", len(candidates), rng_ctrl)
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[peer_id])
                resp = bus.send(peer_id, rid, "delegation_response", 1, rng_ctrl)
                latency = obs.det_estimate_latency + req.latency + r_latency + resp.latency
                regret = candidates[idx].true_cost - min_cost
                success = regret <= (candidates[det_idx].true_cost - min_cost)
                if not cond.frozen:
                    peer_memories[rid].record_request(peer_id)
                    peer_memories[rid].record_response(peer_id, success=success,
                                                          latency=req.latency + r_latency + resp.latency)

            u = utility(regret, latency)
            utilities.append(u)
            regrets.append(regret)
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in set(id(pm_) for pm_ in peer_memories):
            pass
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()
        share_events += maybe_share(peer_memories, cond, rng_ctrl, bus, _round)

    seen_memories = {id(pm): pm for pm in peer_memories}
    memory_entries_total = sum(len(pm.stats) for pm in seen_memories.values())

    return {
        "seed": seed,
        "condition": cond.name,
        "mean_utility": float(np.mean(utilities)),
        "mean_regret": float(np.mean(regrets)),
        "fleet_reasoning_call_rate": sum(per_robot_reasoning_calls) / (n_robots * CONFIG.decisions_per_robot),
        "communication": bus.summary(),
        "share_events": share_events,
        "distinct_memory_entries": memory_entries_total,
        "n_memory_instances": len(seen_memories),
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    all_results = {name: [] for name in CONDITIONS}
    for seed in SEEDS:
        for name, cond in CONDITIONS.items():
            all_results[name].append(run_episode(seed, cond))

    with open(os.path.join(results_dir, "raw_episodes.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    summary = {}
    for name, episodes in all_results.items():
        util_vals = [e["mean_utility"] for e in episodes]
        u_pt, u_lo, u_hi = bootstrap_ci(util_vals, seed=0)
        summary[name] = {
            "n_seeds": len(episodes),
            "mean_utility": {"mean": u_pt, "ci_lo": u_lo, "ci_hi": u_hi},
            "mean_regret_avg": float(np.mean([e["mean_regret"] for e in episodes])),
            "fleet_reasoning_call_rate_mean": float(np.mean([e["fleet_reasoning_call_rate"] for e in episodes])),
            "total_messages_mean": float(np.mean([e["communication"]["total_messages"] for e in episodes])),
            "share_events_mean": float(np.mean([e["share_events"] for e in episodes])),
            "distinct_memory_entries_mean": float(np.mean([e["distinct_memory_entries"] for e in episodes])),
            "n_memory_instances": episodes[0]["n_memory_instances"],
        }

    with open(os.path.join(os.path.dirname(__file__), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
