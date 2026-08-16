"""Phase 4: distributed cognitive delegation. Each robot decides, per decision, whether
to reason itself, delegate to a peer, or skip expensive reasoning, using the
DelegationPolicy (src/delegation/policy.py) built on top of the difficulty-axis
FleetSimulator from Phases 1-3. Communication has an explicit cost (src/communication).

Delegation-enabled vs delegation-disabled (self/skip only, no messaging, equivalent to a
per-robot independent B2-style heuristic escalation) are compared on the SAME seeds so
the comparison is paired. Fresh seed pool disjoint from all Phase 1-3 pools.

Usage: python -m experiments.phase_d.run_delegation
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

SEEDS = list(range(14000, 14015))  # fresh pool, 15 seeds
LAMBDA_LATENCY = 0.15
CONFIG = FleetConfig(num_robots=10, num_stations=10, decisions_per_robot=150,
                      min_candidates=2, max_candidates=5, grid_size=30.0)


def utility(regret: float, latency: float) -> float:
    return -regret - LAMBDA_LATENCY * latency


def run_episode(seed: int, delegation_enabled: bool) -> dict:
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 500000)  # policy/peer-selection randomness, separate stream

    n_robots = CONFIG.num_robots
    peer_memories = [PeerMemory() for _ in range(n_robots)]
    per_robot_reasoning_calls = [0 for _ in range(n_robots)]
    per_robot_self_calls = [0 for _ in range(n_robots)]
    per_robot_delegate_out = [0 for _ in range(n_robots)]
    per_robot_delegate_in = [0 for _ in range(n_robots)]
    per_robot_skip = [0 for _ in range(n_robots)]
    utilities = []
    regrets = []
    duplicate_or_failed = 0

    for _round in range(CONFIG.decisions_per_robot):
        peer_loads = {rid: 0 for rid in range(n_robots)}
        # each robot makes its decision this round in a fixed (but arbitrary) robot-id
        # order; artifact-control variants shuffle this order (see phase_g).
        for rid in range(n_robots):
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            if not delegation_enabled:
                # self/skip only, matching Phase 1's heuristic gate (no peers, no messages)
                eu_self = policy.eu_self(obs)
                eu_skip = policy.eu_skip(obs)
                action = "self" if eu_self > eu_skip else "skip"
                peer_id = None
            else:
                action, peer_id = policy.decide(rid, obs, peer_memories[rid], list(range(n_robots)), peer_loads)

            if action == "skip":
                per_robot_skip[rid] += 1
                idx = det_idx
                latency = obs.det_estimate_latency
                regret = candidates[idx].true_cost - min_cost
            elif action == "self":
                per_robot_self_calls[rid] += 1
                per_robot_reasoning_calls[rid] += 1
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[rid])
                latency = obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
            else:  # delegate
                peer_loads[peer_id] = peer_loads.get(peer_id, 0) + 1
                if peer_loads[peer_id] > policy.peer_capacity:
                    # capacity was checked at decision time but a same-round race can still
                    # occur if two robots pick the same peer in the same round; count it.
                    duplicate_or_failed += 1
                    bus.send(rid, peer_id, "reject", len(candidates), rng_ctrl)
                    bus.record_duplicate()
                    peer_memories[rid].record_rejected(peer_id)
                    idx = det_idx
                    latency = obs.det_estimate_latency
                    regret = candidates[idx].true_cost - min_cost
                else:
                    per_robot_delegate_out[rid] += 1
                    per_robot_delegate_in[peer_id] += 1
                    per_robot_reasoning_calls[peer_id] += 1
                    req = bus.send(rid, peer_id, "delegation_request", len(candidates), rng_ctrl)
                    idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[peer_id])
                    resp = bus.send(peer_id, rid, "delegation_response", 1, rng_ctrl)
                    latency = obs.det_estimate_latency + req.latency + r_latency + resp.latency
                    regret = candidates[idx].true_cost - min_cost
                    success = regret <= (candidates[det_idx].true_cost - min_cost)
                    peer_memories[rid].record_request(peer_id)
                    peer_memories[rid].record_response(peer_id, success=success, latency=req.latency + r_latency + resp.latency)

            u = utility(regret, latency)
            utilities.append(u)
            regrets.append(regret)
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in peer_memories:
            pm.reset_round_load()

    n_decisions = n_robots * CONFIG.decisions_per_robot
    return {
        "seed": seed,
        "delegation_enabled": delegation_enabled,
        "mean_utility": float(np.mean(utilities)),
        "mean_regret": float(np.mean(regrets)),
        "fleet_reasoning_call_rate": sum(per_robot_reasoning_calls) / n_decisions,
        "per_robot_reasoning_calls": per_robot_reasoning_calls,
        "per_robot_self_calls": per_robot_self_calls,
        "per_robot_delegate_out": per_robot_delegate_out,
        "per_robot_delegate_in": per_robot_delegate_in,
        "per_robot_skip": per_robot_skip,
        "communication": bus.summary(),
        "duplicate_or_failed_delegations": duplicate_or_failed,
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    all_results = {"delegation_enabled": [], "delegation_disabled": []}
    for seed in SEEDS:
        all_results["delegation_disabled"].append(run_episode(seed, delegation_enabled=False))
        all_results["delegation_enabled"].append(run_episode(seed, delegation_enabled=True))

    with open(os.path.join(results_dir, "raw_episodes.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    summary = {}
    for cond, episodes in all_results.items():
        util_vals = [e["mean_utility"] for e in episodes]
        regret_vals = [e["mean_regret"] for e in episodes]
        reasoning_rate_vals = [e["fleet_reasoning_call_rate"] for e in episodes]
        msg_vals = [e["communication"]["total_messages"] for e in episodes]
        msg_latency_vals = [e["communication"]["total_message_latency"] for e in episodes]
        dup_vals = [e["duplicate_or_failed_delegations"] for e in episodes]
        u_pt, u_lo, u_hi = bootstrap_ci(util_vals, seed=0)
        r_pt, r_lo, r_hi = bootstrap_ci(regret_vals, seed=1)
        summary[cond] = {
            "n_seeds": len(episodes),
            "mean_utility": {"mean": u_pt, "ci_lo": u_lo, "ci_hi": u_hi},
            "mean_regret": {"mean": r_pt, "ci_lo": r_lo, "ci_hi": r_hi},
            "fleet_reasoning_call_rate_mean": float(np.mean(reasoning_rate_vals)),
            "total_messages_mean": float(np.mean(msg_vals)),
            "total_message_latency_mean": float(np.mean(msg_latency_vals)),
            "duplicate_or_failed_delegations_mean": float(np.mean(dup_vals)),
        }

    # paired comparison (same seeds both conditions)
    paired_diff = [
        e_on["mean_utility"] - e_off["mean_utility"]
        for e_on, e_off in zip(all_results["delegation_enabled"], all_results["delegation_disabled"])
    ]
    d_pt, d_lo, d_hi = bootstrap_ci(paired_diff, seed=2)
    summary["paired_utility_diff_enabled_minus_disabled"] = {"mean": d_pt, "ci_lo": d_lo, "ci_hi": d_hi}

    with open(os.path.join(os.path.dirname(__file__), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
