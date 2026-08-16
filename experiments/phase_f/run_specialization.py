"""Phase 6: specialization experiment. Long-horizon (decisions_per_robot large),
homogeneous robots, no predefined roles, using the delegation+memory mechanism from
Phases 4-5 (C3_local_memory_only: each robot's own local, unshared PeerMemory -- picked
because it is the mechanism-richest condition where delegation/memory feedback can
produce history-dependent behavior, not because it had the best raw utility in Phase 5;
Phase 5 in fact found C0/frozen memory had *better* utility, but running frozen memory
here would trivially prevent any interaction-driven differentiation and defeats the
purpose of this phase, which asks whether specialization arises from interaction alone).

Roles are NEVER labeled during the simulation. Only counts are recorded per robot per
decision; all "role" language in this script's output is a post-hoc description applied
in analysis, computed after the run.

Usage: python -m experiments.phase_f.run_specialization
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

SEEDS = list(range(16000, 16006))  # fresh pool, 6 seeds
LAMBDA_LATENCY = 0.15
N_ROBOTS = 10
DECISIONS_PER_ROBOT = 800
CONFIG = FleetConfig(num_robots=N_ROBOTS, num_stations=10, decisions_per_robot=DECISIONS_PER_ROBOT,
                      min_candidates=2, max_candidates=5, grid_size=30.0)
N_WINDOWS = 4  # for role-persistence analysis


def utility(regret, latency):
    return -regret - LAMBDA_LATENCY * latency


def run_episode(seed: int, robot_id_map=None, shuffle_order_seed=None, tie_break="first"):
    """robot_id_map: optional permutation applied to robot identity labels only (artifact
    control: randomize robot IDs). shuffle_order_seed: if set, shuffles per-round decision
    order instead of fixed round-robin (artifact control). tie_break: "first" or "last" --
    which candidate/peer wins on an exact EU tie (artifact control)."""
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 900000)
    order_rng = np.random.default_rng(shuffle_order_seed) if shuffle_order_seed is not None else None

    cond = MemoryConditionConfig(name="C3_local_memory_only")
    peer_memories = make_peer_memories(N_ROBOTS, cond)

    # per-window, per-robot activity counts: self, delegate_out, delegate_in, skip
    window_len = DECISIONS_PER_ROBOT // N_WINDOWS
    activity = np.zeros((N_WINDOWS, N_ROBOTS, 4), dtype=int)  # [window, robot, {self,out,in,skip}]

    for round_idx in range(DECISIONS_PER_ROBOT):
        window = min(round_idx // window_len, N_WINDOWS - 1)
        peer_loads = {rid: 0 for rid in range(N_ROBOTS)}
        order = list(range(N_ROBOTS))
        if order_rng is not None:
            order_rng.shuffle(order)
        for raw_rid in order:
            rid = robot_id_map[raw_rid] if robot_id_map is not None else raw_rid
            obs = sim.make_observation(raw_rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            action, peer_id = policy.decide(raw_rid, obs, peer_memories[raw_rid], list(range(N_ROBOTS)), peer_loads)

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
                peer_rid_mapped = robot_id_map[peer_id] if robot_id_map is not None else peer_id
                activity[window, peer_rid_mapped, 2] += 1

            sim.apply_decision(raw_rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))
        for pm in {id(x): x for x in peer_memories}.values():
            pm.reset_round_load()

    return activity


def entropy(p):
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def hhi(x):
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total == 0:
        return 0.0
    shares = x / total
    return float(np.sum(shares ** 2))


def analyze(activity: np.ndarray) -> dict:
    """activity: [window, robot, 4]. Post-hoc analysis only -- no labels assigned during sim."""
    total_activity = activity.sum(axis=0)  # [robot, 4]
    fleet_activity = total_activity.sum(axis=0)  # [4]

    per_robot_entropy = [entropy(total_activity[r] / total_activity[r].sum()) if total_activity[r].sum() > 0 else 0.0
                          for r in range(N_ROBOTS)]
    delegate_in_counts = total_activity[:, 2]
    delegate_in_gini = gini(delegate_in_counts)
    delegate_in_hhi = hhi(delegate_in_counts)
    self_counts = total_activity[:, 0]
    self_gini = gini(self_counts)

    # role persistence: rank-correlation of per-robot delegate_in fraction between
    # consecutive temporal windows (Spearman via manual rank correlation, no scipy)
    window_totals = activity.sum(axis=2)  # [window, robot]
    window_delegate_in_frac = activity[:, :, 2] / np.maximum(window_totals, 1)
    persistence_corrs = []
    for w in range(N_WINDOWS - 1):
        a = window_delegate_in_frac[w]
        b = window_delegate_in_frac[w + 1]
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        if np.std(ra) == 0 or np.std(rb) == 0:
            persistence_corrs.append(0.0)
        else:
            persistence_corrs.append(float(np.corrcoef(ra, rb)[0, 1]))

    return {
        "fleet_activity_fractions": (fleet_activity / fleet_activity.sum()).tolist(),
        "per_robot_entropy_H_i": per_robot_entropy,
        "mean_entropy": float(np.mean(per_robot_entropy)),
        "delegate_in_counts": delegate_in_counts.tolist(),
        "delegate_in_gini": delegate_in_gini,
        "delegate_in_hhi": delegate_in_hhi,
        "self_reasoning_counts": self_counts.tolist(),
        "self_reasoning_gini": self_gini,
        "window_delegate_in_frac": window_delegate_in_frac.tolist(),
        "role_persistence_rank_corr": persistence_corrs,
        "mean_role_persistence_rank_corr": float(np.mean(persistence_corrs)) if persistence_corrs else 0.0,
        "max_delegate_in_robot": int(np.argmax(delegate_in_counts)),
    }


def run():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    all_analysis = {}
    for seed in SEEDS:
        activity = run_episode(seed)
        np.save(os.path.join(results_dir, f"activity_seed{seed}.npy"), activity)
        all_analysis[str(seed)] = analyze(activity)

    with open(os.path.join(results_dir, "specialization_summary.json"), "w") as f:
        json.dump(all_analysis, f, indent=2)

    # cross-seed reproducibility of SHAPE (not identity): compare gini/hhi/entropy across seeds
    ginis = [all_analysis[str(s)]["delegate_in_gini"] for s in SEEDS]
    hhis = [all_analysis[str(s)]["delegate_in_hhi"] for s in SEEDS]
    mean_entropies = [all_analysis[str(s)]["mean_entropy"] for s in SEEDS]
    persistences = [all_analysis[str(s)]["mean_role_persistence_rank_corr"] for s in SEEDS]
    max_robots = [all_analysis[str(s)]["max_delegate_in_robot"] for s in SEEDS]

    cross_seed = {
        "delegate_in_gini_mean": float(np.mean(ginis)), "delegate_in_gini_std": float(np.std(ginis)),
        "delegate_in_hhi_mean": float(np.mean(hhis)), "delegate_in_hhi_std": float(np.std(hhis)),
        "mean_entropy_mean": float(np.mean(mean_entropies)), "mean_entropy_std": float(np.std(mean_entropies)),
        "role_persistence_mean": float(np.mean(persistences)), "role_persistence_std": float(np.std(persistences)),
        "max_delegate_in_robot_per_seed": max_robots,
        "max_delegate_in_robot_identical_across_seeds": len(set(max_robots)) == 1,
    }
    with open(os.path.join(results_dir, "cross_seed_summary.json"), "w") as f:
        json.dump(cross_seed, f, indent=2)
    print(json.dumps(cross_seed, indent=2))
    return all_analysis, cross_seed


if __name__ == "__main__":
    run()
