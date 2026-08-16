"""Delegation-policy calibration (RACA-Collective.txt Section 24 seed-pool discipline).

The original Phase 4 report used a single, hand-picked DelegationPolicy parameterization
(self_load_penalty=1.5, lambda_msg=0.01, self_quality_factor=0.7, peer_capacity=2) that was
never calibrated the way Phase 3's tau threshold was (grid search on a dedicated
calibration seed pool, disjoint from held-out seeds, chosen BEFORE touching held-out
seeds). This script closes that gap: grid search over the policy's free parameters on a
calibration seed pool, selecting the configuration that maximizes mean fleet utility with
delegation enabled, evaluated ONLY on the calibration pool. The chosen parameters are then
used, unchanged, on fresh held-out seeds in run_delegation_revised.py.

Calibration seed pool: 18000-18009 (10 seeds), disjoint from every prior pool (14000s
Phase 4 original, 15000s Phase 5 original, and the fresh held-out pools used below).

Usage: python -m experiments.phase_d.calibrate_delegation
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetSimulator, FleetConfig
from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult
from src.communication.messaging import MessageBus
from src.delegation.peer_memory import PeerMemory
from src.delegation.policy import DelegationPolicy

CAL_SEEDS = list(range(18000, 18010))  # 10 seeds, fresh, disjoint from all other pools
LAMBDA_LATENCY = 0.15
CONFIG = FleetConfig(num_robots=10, num_stations=10, decisions_per_robot=150,
                      min_candidates=2, max_candidates=5, grid_size=30.0)

GRID = {
    "self_load_penalty": [0.0, 0.5, 1.5, 3.0],
    "lambda_msg": [0.0, 0.01, 0.05],
    "self_quality_factor": [0.5, 0.7, 0.9],
    "peer_capacity": [2],  # left fixed; not stress-tested at this fleet size (see Phase 4 note)
}


def utility(regret: float, latency: float) -> float:
    return -regret - LAMBDA_LATENCY * latency


def run_episode(seed: int, policy: DelegationPolicy) -> float:
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 600000)
    n_robots = CONFIG.num_robots
    peer_memories = [PeerMemory() for _ in range(n_robots)]
    utilities = []

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

            utilities.append(utility(regret, latency))
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in peer_memories:
            pm.reset_round_load()
    return float(np.mean(utilities))


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())
    grid_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY, **params)
        utils = [run_episode(seed, policy) for seed in CAL_SEEDS]
        grid_results.append({"params": params, "mean_utility": float(np.mean(utils)), "std_utility": float(np.std(utils))})

    grid_results.sort(key=lambda r: r["mean_utility"], reverse=True)
    best = grid_results[0]

    with open(os.path.join(results_dir, "calibration_grid.json"), "w") as f:
        json.dump({"calibration_seeds": CAL_SEEDS, "grid": grid_results, "best": best}, f, indent=2)
    print(json.dumps(best, indent=2))
    return best, grid_results


if __name__ == "__main__":
    run()
