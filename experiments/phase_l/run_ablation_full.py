"""Phase 12 (RACA-Collective.txt Section 19 / 22): A2 and A7, completing the ablation
matrix Phase 11 left with two named gaps.

Runs all eight named ablations (A0-A7) on one common fleet config (10 robots, 150
decisions/robot, difficulty-axis simulator, lambda_latency=0.15), on a FRESH seed pool
(25100-25109, 10 seeds) disjoint from every previously used pool (checked against every
SEEDS/BASE_SEED assignment under experiments/, see docs/RESEARCH_LOG.md; highest prior
pool touched was 23900-23914). B3 is retrained fresh on its own disjoint dev/cal pool
(25900-25904 / 25910-25914).

  A0 -- Independent Previous-RACA Agents: B2 heuristic router.
  A1 -- Independent Outcome-Aware Agents: B3 learned router.
  A2 -- Centralized Cognitive Manager (src/routing/centralized.py, new): one process
        sees every robot's Observation for the round before assigning anything.
  A3 -- Distributed, no delegation (calibrated self/skip only).
  A4 -- Distributed + delegation (calibrated), memory frozen.
  A5 -- Distributed + delegation (calibrated) + local memory.
  A6 -- Distributed + delegation (calibrated) + shared memory.
  A7 -- Full RACA-Collective (src/routing/full_system.py, new): B3 gate + calibrated
        delegation + local memory (C3), distributed.

A separate scaling section (Section 22's "does centralized cognition stop scaling
before distributed cognition") runs A2 and A3 at N=10 and N=50, 4 seeds each (fresh pool
26010-26013 / 26050-26053), measuring wall-clock seconds/round and messages/round
alongside utility.

Usage: python -m experiments.phase_l.run_ablation_full
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetSimulator, FleetConfig
from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult
from src.communication.messaging import MessageBus
from src.delegation.policy import DelegationPolicy
from src.memory.conditions import MemoryConditionConfig, make_peer_memories
from src.delegation.peer_memory import PeerMemory
from src.routing.heuristic import HeuristicRouter
from src.routing.learned import LearnedRouter
from src.routing.centralized import CentralizedManager
from src.routing.full_system import FullSystemRouter
from src.reasoning.counterfactual import generate_dataset
from src.reasoning.value_estimator import LogisticValueEstimator, featurize, label
from src.evaluation.metrics import bootstrap_ci

SEEDS = list(range(25100, 25110))  # 10 fresh seeds, disjoint from all other pools
LAMBDA_LATENCY = 0.15
CONFIG = FleetConfig(num_robots=10, num_stations=10, decisions_per_robot=150,
                      min_candidates=2, max_candidates=5, grid_size=30.0)
B3_DEV_SEEDS = list(range(25900, 25905))
B3_CAL_SEEDS = list(range(25910, 25915))

SCALE_SIZES = [10, 50]
SCALE_SEEDS_PER_SIZE = 4
SCALE_BASE = 26000
SCALE_DECISIONS_PER_ROBOT = 100


def utility(regret, latency):
    return -regret - LAMBDA_LATENCY * latency


def train_b3(config=CONFIG, dev_seeds=B3_DEV_SEEDS, cal_seeds=B3_CAL_SEEDS):
    dev_records = []
    for s in dev_seeds:
        dev_records.extend(generate_dataset(config, seed=s, n_samples=150, lambda_latency=LAMBDA_LATENCY))
    X = np.stack([featurize(r) for r in dev_records])
    y = np.array([label(r, LAMBDA_LATENCY) for r in dev_records], dtype=float)
    est = LogisticValueEstimator().fit(X, y, lr=0.3, epochs=800, seed=0)

    cal_records = []
    for s in cal_seeds:
        cal_records.extend(generate_dataset(config, seed=s, n_samples=150, lambda_latency=LAMBDA_LATENCY))
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


def calibrated_delegation_policy():
    from experiments.phase_d.calibrate_delegation import run as run_calibration
    best, _ = run_calibration()
    params = best["params"]
    return DelegationPolicy(lambda_latency=LAMBDA_LATENCY, **params), params


def run_independent_router_episode(seed: int, router, config=CONFIG) -> float:
    sim = FleetSimulator(config, seed=seed)
    utilities = []
    for _round in range(config.decisions_per_robot):
        for rid in range(config.num_robots):
            obs = sim.make_observation(rid, router.det_estimate)
            result = router.decide(obs, sim.robot_rngs[rid])
            true_costs = [c.true_cost for c in obs.candidates]
            min_cost = min(true_costs)
            regret = obs.candidates[result.chosen_index].true_cost - min_cost
            utilities.append(utility(regret, result.latency))
            sim.apply_decision(rid, obs, result)
    return float(np.mean(utilities))


def run_distributed_episode(seed: int, decide_fn, mem_cond: MemoryConditionConfig | None,
                             config=CONFIG) -> dict:
    """decide_fn(rid, obs, peer_memory, peer_ids, peer_loads) -> (action, peer_id_or_None),
    action in {"skip", "self", "delegate"}. Shared runner for A3-A7 (all distributed)."""
    sim = FleetSimulator(config, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 500000)
    n_robots = config.num_robots
    if mem_cond is not None:
        peer_memories = make_peer_memories(n_robots, mem_cond)
    else:
        peer_memories = [PeerMemory() for _ in range(n_robots)]
    utilities = []

    for _round in range(config.decisions_per_robot):
        peer_loads = {rid: 0 for rid in range(n_robots)}
        for rid in range(n_robots):
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            action, peer_id = decide_fn(rid, obs, peer_memories[rid], list(range(n_robots)), peer_loads)

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
                    peer_memories[rid].record_response(peer_id, success=success,
                                                         latency=req.latency + r_latency + resp.latency)

            utilities.append(utility(regret, latency))
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()
    return {"mean_utility": float(np.mean(utilities))}


def run_centralized_episode(seed: int, manager: CentralizedManager, config=CONFIG,
                             measure_time: bool = False) -> dict:
    """A2. The manager sees every robot's Observation for the round in one batch
    (unlike every distributed condition, where a robot only ever sees its own
    Observation) before assigning anything. Communication cost: every robot must report
    its observation to the center and receive an assignment back (2 messages/robot/round,
    unlike the distributed conditions' zero messages when no delegation happens), plus a
    2-message roundtrip for any reassigned decision, matching Section 14's requirement
    that delegation/reassignment is never free."""
    sim = FleetSimulator(config, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 600000)
    n_robots = config.num_robots
    utilities = []
    wall_seconds = []

    for _round in range(config.decisions_per_robot):
        t0 = time.perf_counter() if measure_time else None
        observations = {}
        candidates_by_rid = {}
        for rid in range(n_robots):
            obs = sim.make_observation(rid, det_backend.estimate)
            observations[rid] = obs
            candidates_by_rid[rid] = obs.candidates
            bus.send(rid, -1, "report_to_center", 1, rng_ctrl)  # robot -> center

        assignments = manager.assign_round(observations)

        for rid in range(n_robots):
            obs = observations[rid]
            candidates = candidates_by_rid[rid]
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))
            action, target = assignments[rid]
            bus.send(-1, rid, "assignment", 1, rng_ctrl)  # center -> robot

            if action == "deterministic":
                idx, latency = det_idx, obs.det_estimate_latency
                regret = candidates[idx].true_cost - min_cost
                used_reasoning = False
            elif action == "reason_self":
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[rid])
                latency = obs.det_estimate_latency + r_latency
                regret = candidates[idx].true_cost - min_cost
                used_reasoning = True
                success = regret <= (candidates[det_idx].true_cost - min_cost)
                manager.record_outcome(rid, success)
            else:  # reassign
                req = bus.send(rid, target, "reassign_request", len(candidates), rng_ctrl)
                idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[target])
                resp = bus.send(target, rid, "reassign_response", 1, rng_ctrl)
                latency = obs.det_estimate_latency + req.latency + r_latency + resp.latency
                regret = candidates[idx].true_cost - min_cost
                used_reasoning = True
                success = regret <= (candidates[det_idx].true_cost - min_cost)
                manager.record_outcome(target, success)

            utilities.append(utility(regret, latency))
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=used_reasoning))
        if measure_time:
            wall_seconds.append(time.perf_counter() - t0)

    out = {"mean_utility": float(np.mean(utilities)), "comm": bus.summary()}
    if measure_time:
        out["mean_wall_seconds_per_round"] = float(np.mean(wall_seconds))
        out["messages_per_decision"] = bus.summary()["total_messages"] / (n_robots * config.decisions_per_robot)
    return out


def run_distributed_episode_timed(seed: int, decide_fn, mem_cond, config) -> dict:
    """Like run_distributed_episode but also measures wall-clock seconds/round, for the
    A2-vs-A3 scaling comparison."""
    sim = FleetSimulator(config, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 700000)
    n_robots = config.num_robots
    peer_memories = [PeerMemory() for _ in range(n_robots)] if mem_cond is None else make_peer_memories(n_robots, mem_cond)
    utilities = []
    wall_seconds = []

    for _round in range(config.decisions_per_robot):
        t0 = time.perf_counter()
        peer_loads = {rid: 0 for rid in range(n_robots)}
        for rid in range(n_robots):
            obs = sim.make_observation(rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))
            action, peer_id = decide_fn(rid, obs, peer_memories[rid], list(range(n_robots)), peer_loads)
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
            utilities.append(utility(regret, latency))
            sim.apply_decision(rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()
        wall_seconds.append(time.perf_counter() - t0)

    return {
        "mean_utility": float(np.mean(utilities)),
        "mean_wall_seconds_per_round": float(np.mean(wall_seconds)),
        "messages_per_decision": bus.summary()["total_messages"] / (n_robots * config.decisions_per_robot),
    }


def run_main_matrix():
    b2 = HeuristicRouter(lambda_latency=LAMBDA_LATENCY)
    b3_est, b3_tau = train_b3()
    b3 = LearnedRouter(b3_est, b3_tau)
    cal_policy, cal_params = calibrated_delegation_policy()
    full_router = FullSystemRouter(b3_est, b3_tau, cal_policy)

    results = {}
    results["A0_independent_B2_heuristic"] = [run_independent_router_episode(s, b2) for s in SEEDS]
    results["A1_independent_B3_learned"] = [run_independent_router_episode(s, b3) for s in SEEDS]

    a2_utils = []
    for s in SEEDS:
        manager = CentralizedManager(b3_est, b3_tau, capacity_per_robot=cal_policy.peer_capacity)
        a2_utils.append(run_centralized_episode(s, manager)["mean_utility"])
    results["A2_centralized_manager"] = a2_utils

    def a3_decide(rid, obs, pm, peer_ids, peer_loads):
        eu_self, eu_skip = cal_policy.eu_self(obs), cal_policy.eu_skip(obs)
        return ("self", None) if eu_self > eu_skip else ("skip", None)

    results["A3_distributed_no_delegation_calibrated"] = [
        run_distributed_episode(s, a3_decide, None)["mean_utility"] for s in SEEDS
    ]

    frozen_cond = MemoryConditionConfig(name="A4_frozen_memory", frozen=True)
    results["A4_delegation_frozen_memory"] = [
        run_distributed_episode(s, cal_policy.decide, frozen_cond)["mean_utility"] for s in SEEDS
    ]
    local_cond = MemoryConditionConfig(name="A5_local_memory")
    results["A5_delegation_local_memory"] = [
        run_distributed_episode(s, cal_policy.decide, local_cond)["mean_utility"] for s in SEEDS
    ]
    shared_cond = MemoryConditionConfig(name="A6_shared_memory", shared=True)
    results["A6_delegation_shared_memory"] = [
        run_distributed_episode(s, cal_policy.decide, shared_cond)["mean_utility"] for s in SEEDS
    ]
    results["A7_full_raca_collective"] = [
        run_distributed_episode(s, full_router.decide, local_cond)["mean_utility"] for s in SEEDS
    ]

    summary = {"calibrated_params": cal_params, "b3_tau": b3_tau, "n_seeds": len(SEEDS)}
    for name, vals in results.items():
        pt, lo, hi = bootstrap_ci(vals, seed=0)
        summary[name] = {"mean_utility": pt, "ci_lo": lo, "ci_hi": hi}
    return results, summary, b3_est, b3_tau, cal_policy


def run_scaling_comparison(b3_est, b3_tau, cal_policy):
    """Section 22: does centralized cognition (A2) stop scaling before distributed
    cognition (A3) does? Measures wall-clock seconds/round, messages/decision, and
    utility at N=10 and N=50 for both A2 and A3.

    A3 here uses the SAME B3 escalation criterion as A2 (rather than the calibrated
    DelegationPolicy self/skip heuristic used in the main A0-A7 matrix): the point of
    this comparison is to isolate the centralized-vs-distributed control structure, so
    both sides must use identical escalation logic and differ only in whether a robot
    that decides to reason must reason itself (A3, distributed, no reassignment option)
    or may have the decision centrally reassigned to a fleet-wide best candidate (A2)."""
    scaling = {}
    for size in SCALE_SIZES:
        cfg = FleetConfig(num_robots=size, num_stations=max(10, size),
                           decisions_per_robot=SCALE_DECISIONS_PER_ROBOT,
                           min_candidates=2, max_candidates=5,
                           grid_size=30.0 * (size / 10.0) ** 0.5)
        a2_rows, a3_rows = [], []

        def a3_decide(rid, obs, pm, peer_ids, peer_loads, _est=b3_est, _tau=b3_tau):
            feats = np.array([[obs.ambiguity, obs.urgency, float(obs.candidate_count),
                                obs.cost_margin, obs.battery_soc, obs.recent_latency_mean]])
            p = float(_est.predict_proba(feats)[0])
            return ("self", None) if p > _tau else ("skip", None)

        for i in range(SCALE_SEEDS_PER_SIZE):
            seed = SCALE_BASE + size * 100 + i
            manager = CentralizedManager(b3_est, b3_tau, capacity_per_robot=cal_policy.peer_capacity)
            a2_rows.append(run_centralized_episode(seed, manager, config=cfg, measure_time=True))
            a3_rows.append(run_distributed_episode_timed(seed + 1000, a3_decide, None, cfg))

        scaling[str(size)] = {
            "A2_centralized": {
                "mean_utility": float(np.mean([r["mean_utility"] for r in a2_rows])),
                "mean_wall_seconds_per_round": float(np.mean([r["mean_wall_seconds_per_round"] for r in a2_rows])),
                "messages_per_decision": float(np.mean([r["messages_per_decision"] for r in a2_rows])),
            },
            "A3_distributed": {
                "mean_utility": float(np.mean([r["mean_utility"] for r in a3_rows])),
                "mean_wall_seconds_per_round": float(np.mean([r["mean_wall_seconds_per_round"] for r in a3_rows])),
                "messages_per_decision": float(np.mean([r["messages_per_decision"] for r in a3_rows])),
            },
        }
    return scaling


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    raw, summary, b3_est, b3_tau, cal_policy = run_main_matrix()
    with open(os.path.join(results_dir, "ablation_full_raw.json"), "w") as f:
        json.dump(raw, f, indent=2)
    with open(os.path.join(results_dir, "ablation_full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=== Main matrix (A0-A7) ===")
    print(json.dumps(summary, indent=2))

    scaling = run_scaling_comparison(b3_est, b3_tau, cal_policy)
    with open(os.path.join(results_dir, "scaling_centralized_vs_distributed.json"), "w") as f:
        json.dump(scaling, f, indent=2)
    print("=== Scaling: A2 (centralized) vs A3 (distributed) at N=10, N=50 ===")
    print(json.dumps(scaling, indent=2))

    return summary, scaling


if __name__ == "__main__":
    run()
