"""Phase 11 (RACA-Collective.txt Section 19): ablation matrix.

Runs a reduced but real subset of the required ablations on a common fleet config (10
robots, 150 decisions/robot, difficulty-axis simulator, lambda_latency=0.15), 8 fresh
seeds per condition (23100-23107), disjoint from every other pool used in this task. 8
seeds is a smaller budget than Phase 4/5's 15 (stated explicitly, a deliberate
time/seed-budget tradeoff for a 6-condition matrix rather than an unstated shortcut).

Conditions run (subset required by the task, mapped onto existing mechanisms):
  A0 -- Independent Previous-RACA Agents: B2 heuristic router (src/routing/heuristic.py),
        every robot independent, no delegation, no learned routing.
  A1 -- Independent Outcome-Aware Agents: B3 learned router (src/routing/learned.py),
        trained fresh on this fleet config's own dev/cal seeds (13900s/13910s, disjoint
        from Phase 3's pools), every robot independent, no delegation.
  A3 -- Distributed Agents Without Delegation Learning: DelegationPolicy self/skip only
        (no peers), calibrated parameters from experiments/phase_d/calibrate_delegation.py.
  A4 -- Distributed + Delegation (calibrated), memory frozen (peer stats never update,
        so delegation is available but never differentiates peers beyond the
        uninformative prior -- isolates "delegation as an option" from "delegation that
        learns").
  A5 -- Distributed + Delegation (calibrated) + Local Memory: each robot's own PeerMemory
        updates from its own delegation outcomes (C3 mechanism from Phase 5).
  A6 -- Distributed + Delegation (calibrated) + Shared/Peer Memory: one shared PeerMemory
        instance across all robots (C1 mechanism from Phase 5).

Usage: python -m experiments.phase_k.run_ablation
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
from src.routing.heuristic import HeuristicRouter
from src.routing.learned import LearnedRouter
from src.reasoning.counterfactual import generate_dataset
from src.reasoning.value_estimator import LogisticValueEstimator, featurize, label
from src.evaluation.metrics import bootstrap_ci
from experiments.phase_d.calibrate_delegation import run as run_calibration

SEEDS = list(range(23100, 23108))  # 8 fresh seeds, disjoint from all other pools
LAMBDA_LATENCY = 0.15
CONFIG = FleetConfig(num_robots=10, num_stations=10, decisions_per_robot=150,
                      min_candidates=2, max_candidates=5, grid_size=30.0)
B3_DEV_SEEDS = list(range(23900, 23905))
B3_CAL_SEEDS = list(range(23910, 23915))


def utility(regret, latency):
    return -regret - LAMBDA_LATENCY * latency


def train_b3():
    dev_records = []
    for s in B3_DEV_SEEDS:
        dev_records.extend(generate_dataset(CONFIG, seed=s, n_samples=150, lambda_latency=LAMBDA_LATENCY))
    X = np.stack([featurize(r) for r in dev_records])
    y = np.array([label(r, LAMBDA_LATENCY) for r in dev_records], dtype=float)
    est = LogisticValueEstimator().fit(X, y, lr=0.3, epochs=800, seed=0)

    cal_records = []
    for s in B3_CAL_SEEDS:
        cal_records.extend(generate_dataset(CONFIG, seed=s, n_samples=150, lambda_latency=LAMBDA_LATENCY))
    Xc = np.stack([featurize(r) for r in cal_records])
    p = est.predict_proba(Xc)
    best_tau, best_u = 0.5, -np.inf
    for tau in np.linspace(0.05, 0.95, 19):
        utils = []
        for r, pi in zip(cal_records, p):
            if pi > tau:
                utils.append(utility(r.regret_reasoning, r.latency_reasoning))
            else:
                utils.append(utility(r.regret_det, r.latency_det))
        mean_u = float(np.mean(utils))
        if mean_u > best_u:
            best_u, best_tau = mean_u, float(tau)
    return est, best_tau


def run_independent_router_episode(seed: int, router) -> float:
    sim = FleetSimulator(CONFIG, seed=seed)
    rng_router = np.random.default_rng(seed + 400000)
    utilities = []
    for _round in range(CONFIG.decisions_per_robot):
        for rid in range(CONFIG.num_robots):
            obs = sim.make_observation(rid, router.det_estimate)
            result = router.decide(obs, sim.robot_rngs[rid])
            true_costs = [c.true_cost for c in obs.candidates]
            min_cost = min(true_costs)
            regret = obs.candidates[result.chosen_index].true_cost - min_cost
            utilities.append(utility(regret, result.latency))
            sim.apply_decision(rid, obs, result)
    return float(np.mean(utilities))


def run_delegation_episode(seed: int, policy: DelegationPolicy, delegation_enabled: bool,
                            mem_cond: MemoryConditionConfig | None) -> dict:
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 500000)
    n_robots = CONFIG.num_robots
    if mem_cond is not None:
        peer_memories = make_peer_memories(n_robots, mem_cond)
    else:
        from src.delegation.peer_memory import PeerMemory
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

            if not delegation_enabled:
                eu_self, eu_skip = policy.eu_self(obs), policy.eu_skip(obs)
                action, peer_id = ("self" if eu_self > eu_skip else "skip"), None
            else:
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
                if mem_cond is None or not mem_cond.frozen:
                    peer_memories[rid].record_request(peer_id)
                    peer_memories[rid].record_response(peer_id, success=success, latency=req.latency + r_latency + resp.latency)

            utilities.append(utility(regret, latency))
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()
    return {"mean_utility": float(np.mean(utilities))}


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    b2 = HeuristicRouter(lambda_latency=LAMBDA_LATENCY)
    b3_est, b3_tau = train_b3()
    b3 = LearnedRouter(b3_est, b3_tau)

    best, _ = run_calibration()
    params = best["params"]
    cal_policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY, **params)

    results = {}
    results["A0_independent_B2_heuristic"] = [run_independent_router_episode(s, b2) for s in SEEDS]
    results["A1_independent_B3_learned"] = [run_independent_router_episode(s, b3) for s in SEEDS]
    results["A3_distributed_no_delegation_calibrated"] = [
        run_delegation_episode(s, cal_policy, delegation_enabled=False, mem_cond=None)["mean_utility"] for s in SEEDS
    ]
    frozen_cond = MemoryConditionConfig(name="A4_frozen_memory", frozen=True)
    results["A4_delegation_frozen_memory"] = [
        run_delegation_episode(s, cal_policy, delegation_enabled=True, mem_cond=frozen_cond)["mean_utility"] for s in SEEDS
    ]
    local_cond = MemoryConditionConfig(name="A5_local_memory")
    results["A5_delegation_local_memory"] = [
        run_delegation_episode(s, cal_policy, delegation_enabled=True, mem_cond=local_cond)["mean_utility"] for s in SEEDS
    ]
    shared_cond = MemoryConditionConfig(name="A6_shared_memory", shared=True)
    results["A6_delegation_shared_memory"] = [
        run_delegation_episode(s, cal_policy, delegation_enabled=True, mem_cond=shared_cond)["mean_utility"] for s in SEEDS
    ]

    summary = {"calibrated_params": params, "b3_tau": b3_tau, "n_seeds": len(SEEDS)}
    for name, vals in results.items():
        pt, lo, hi = bootstrap_ci(vals, seed=0)
        summary[name] = {"mean_utility": pt, "ci_lo": lo, "ci_hi": hi}

    with open(os.path.join(results_dir, "ablation_raw.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(results_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
