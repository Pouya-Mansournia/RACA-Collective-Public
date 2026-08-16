"""Phase 10 (RACA-Collective.txt Section 23): homogeneous (H0) vs controlled
heterogeneous (H1) robots, as a SEPARATE experiment from the homogeneous specialization
result in Phases 6/7.

H0: all 10 robots share identical DelegationPolicy parameters (as in every prior phase).
H1: a controlled capability split -- 3 of 10 robots ("fast/reliable") get a lower
    reasoning_latency_mean (2.0 vs 3.0) and the ExpensiveReasoningBackend they use draws
    with lower cost-estimation noise (see src/reasoning/backends.py failure_prob), i.e. a
    genuine capability difference in latency and reliability, not a hardcoded "role" or
    reward. The other 7 ("standard") are unchanged from H0. Which robots are fast is
    fixed by robot index (0-2) for reproducibility of the manipulation itself, but no
    role or preference is given to them in the DelegationPolicy or PeerMemory code --
    any advantage they get must come from actually being faster/more reliable and other
    robots discovering that through observed peer stats, exactly as Section 23 requires.

Compares: does delegate-in concentration (Gini) track the heterogeneity (fast robots
becoming more heavily delegated to) more strongly / more predictably than the H0
homogeneous result, and does fleet utility improve under H1 relative to H0?

Usage: python -m experiments.phase_j.run_heterogeneity
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

SEEDS = list(range(23000, 23006))  # fresh pool, 6 seeds
LAMBDA_LATENCY = 0.15
N_ROBOTS = 10
DECISIONS_PER_ROBOT = 400
FAST_ROBOTS = {0, 1, 2}  # H1 only: which robots get the capability boost
CONFIG = FleetConfig(num_robots=N_ROBOTS, num_stations=10, decisions_per_robot=DECISIONS_PER_ROBOT,
                      min_candidates=2, max_candidates=5, grid_size=30.0)


def utility(regret, latency):
    return -regret - LAMBDA_LATENCY * latency


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def run_episode(seed: int, heterogeneous: bool) -> dict:
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    fast_backend = ExpensiveReasoningBackend(failure_prob=0.01, latency_mean=1.6, latency_std=0.3)
    std_backend = ExpensiveReasoningBackend()
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 900000)
    cond = MemoryConditionConfig(name="C3_local_memory_only")
    peer_memories = make_peer_memories(N_ROBOTS, cond)
    delegate_in_counts = np.zeros(N_ROBOTS, dtype=int)
    utilities = []

    def backend_for(rid):
        return fast_backend if (heterogeneous and rid in FAST_ROBOTS) else std_backend

    for _round in range(DECISIONS_PER_ROBOT):
        peer_loads = {rid: 0 for rid in range(N_ROBOTS)}
        for rid in range(N_ROBOTS):
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))
            action, peer_id = policy.decide(rid, obs, peer_memories[rid], list(range(N_ROBOTS)), peer_loads)

            if action == "skip":
                idx, latency = det_idx, obs.det_estimate_latency
                regret = candidates[idx].true_cost - min_cost
            elif action == "self":
                idx, r_latency = backend_for(rid).decide(candidates, sim.robot_rngs[rid])
                latency = obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
            else:
                peer_loads[peer_id] = peer_loads.get(peer_id, 0) + 1
                req = bus.send(rid, peer_id, "delegation_request", len(candidates), rng_ctrl)
                idx, r_latency = backend_for(peer_id).decide(candidates, sim.robot_rngs[peer_id])
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

    fast_share = float(delegate_in_counts[list(FAST_ROBOTS)].sum() / max(1, delegate_in_counts.sum())) if heterogeneous else None
    return {
        "seed": seed, "heterogeneous": heterogeneous,
        "mean_utility": float(np.mean(utilities)),
        "delegate_in_gini": gini(delegate_in_counts),
        "delegate_in_counts": delegate_in_counts.tolist(),
        "fast_robot_delegate_in_share": fast_share,
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    h0 = [run_episode(s, heterogeneous=False) for s in SEEDS]
    h1 = [run_episode(s, heterogeneous=True) for s in SEEDS]

    with open(os.path.join(results_dir, "heterogeneity_raw.json"), "w") as f:
        json.dump({"H0_homogeneous": h0, "H1_heterogeneous": h1}, f, indent=2)

    summary = {
        "H0_homogeneous": {
            "mean_utility": float(np.mean([e["mean_utility"] for e in h0])),
            "delegate_in_gini_mean": float(np.mean([e["delegate_in_gini"] for e in h0])),
            "delegate_in_gini_std": float(np.std([e["delegate_in_gini"] for e in h0])),
        },
        "H1_heterogeneous": {
            "mean_utility": float(np.mean([e["mean_utility"] for e in h1])),
            "delegate_in_gini_mean": float(np.mean([e["delegate_in_gini"] for e in h1])),
            "delegate_in_gini_std": float(np.std([e["delegate_in_gini"] for e in h1])),
            "fast_robot_delegate_in_share_mean": float(np.mean([e["fast_robot_delegate_in_share"] for e in h1])),
            "fast_robot_expected_share_if_uniform": len(FAST_ROBOTS) / N_ROBOTS,
        },
    }
    with open(os.path.join(results_dir, "heterogeneity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
