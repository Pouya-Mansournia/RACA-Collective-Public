"""Phase 4 revised: distributed cognitive delegation with a CALIBRATED policy.

Uses the DelegationPolicy parameters chosen by calibrate_delegation.py's grid search on
the calibration seed pool (18000-18009), which is disjoint from both the original Phase 4
seed pool (14000-14014) and the fresh held-out pool used here (19000-19014). Calibration
was performed once, before these held-out numbers were computed, and is not iterated
after seeing this script's output.

Usage: python -m experiments.phase_d.run_delegation_revised
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
from src.delegation.peer_memory import PeerMemory
from src.delegation.policy import DelegationPolicy
from src.evaluation.metrics import bootstrap_ci
from experiments.phase_d.calibrate_delegation import run as run_calibration

SEEDS = list(range(19000, 19015))  # fresh pool, 15 seeds, disjoint from 14000s and 18000s
LAMBDA_LATENCY = 0.15
CONFIG = FleetConfig(num_robots=10, num_stations=10, decisions_per_robot=150,
                      min_candidates=2, max_candidates=5, grid_size=30.0)


def utility(regret: float, latency: float) -> float:
    return -regret - LAMBDA_LATENCY * latency


def run_episode(seed: int, delegation_enabled: bool, policy: DelegationPolicy) -> dict:
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 500000)

    n_robots = CONFIG.num_robots
    peer_memories = [PeerMemory() for _ in range(n_robots)]
    per_robot_reasoning_calls = [0] * n_robots
    utilities, regrets = [], []
    duplicate_or_failed = 0

    for _round in range(CONFIG.decisions_per_robot):
        peer_loads = {rid: 0 for rid in range(n_robots)}
        for rid in range(n_robots):
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            if not delegation_enabled:
                eu_self = policy.eu_self(obs)
                eu_skip = policy.eu_skip(obs)
                action = "self" if eu_self > eu_skip else "skip"
                peer_id = None
            else:
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
                if peer_loads[peer_id] > policy.peer_capacity:
                    duplicate_or_failed += 1
                    bus.send(rid, peer_id, "reject", len(candidates), rng_ctrl)
                    bus.record_duplicate()
                    peer_memories[rid].record_rejected(peer_id)
                    idx, latency = det_idx, obs.det_estimate_latency
                    regret = candidates[idx].true_cost - min_cost
                else:
                    per_robot_reasoning_calls[peer_id] += 1
                    req = bus.send(rid, peer_id, "delegation_request", len(candidates), rng_ctrl)
                    idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[peer_id])
                    resp = bus.send(peer_id, rid, "delegation_response", 1, rng_ctrl)
                    latency = obs.det_estimate_latency + req.latency + r_latency + resp.latency
                    regret = candidates[idx].true_cost - min_cost
                    success = regret <= (candidates[det_idx].true_cost - min_cost)
                    peer_memories[rid].record_request(peer_id)
                    peer_memories[rid].record_response(peer_id, success=success, latency=req.latency + r_latency + resp.latency)

            utilities.append(utility(regret, latency))
            regrets.append(regret)
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in peer_memories:
            pm.reset_round_load()

    n_decisions = n_robots * CONFIG.decisions_per_robot
    return {
        "seed": seed, "delegation_enabled": delegation_enabled,
        "mean_utility": float(np.mean(utilities)),
        "mean_regret": float(np.mean(regrets)),
        "fleet_reasoning_call_rate": sum(per_robot_reasoning_calls) / n_decisions,
        "communication": bus.summary(),
        "duplicate_or_failed_delegations": duplicate_or_failed,
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    best, grid = run_calibration()
    params = best["params"]
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY, **params)

    all_results = {"delegation_enabled": [], "delegation_disabled": []}
    for seed in SEEDS:
        all_results["delegation_disabled"].append(run_episode(seed, False, policy))
        all_results["delegation_enabled"].append(run_episode(seed, True, policy))

    with open(os.path.join(results_dir, "raw_episodes_revised.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    summary = {"calibrated_params": params, "calibration_grid_best_utility": best["mean_utility"]}
    for cond, episodes in all_results.items():
        util_vals = [e["mean_utility"] for e in episodes]
        u_pt, u_lo, u_hi = bootstrap_ci(util_vals, seed=0)
        summary[cond] = {
            "n_seeds": len(episodes),
            "mean_utility": {"mean": u_pt, "ci_lo": u_lo, "ci_hi": u_hi},
            "mean_regret_avg": float(np.mean([e["mean_regret"] for e in episodes])),
            "fleet_reasoning_call_rate_mean": float(np.mean([e["fleet_reasoning_call_rate"] for e in episodes])),
            "total_messages_mean": float(np.mean([e["communication"]["total_messages"] for e in episodes])),
        }

    paired_diff = [
        e_on["mean_utility"] - e_off["mean_utility"]
        for e_on, e_off in zip(all_results["delegation_enabled"], all_results["delegation_disabled"])
    ]
    d_pt, d_lo, d_hi = bootstrap_ci(paired_diff, seed=2)
    summary["paired_utility_diff_enabled_minus_disabled"] = {"mean": d_pt, "ci_lo": d_lo, "ci_hi": d_hi}

    with open(os.path.join(os.path.dirname(__file__), "summary_revised.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
