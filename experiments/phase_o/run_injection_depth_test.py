"""Phase O part 2 -- does the hub-identity flip rate from a tie-reversal perturbation
decay with how deep into the episode the perturbation is injected?

Background: `run_path_dependence_test.py` showed that reversing the peer-evaluation
order at the FIRST reversible exact-EU tie in an episode (typically very early, since
untried peer pairs are the main source of exact ties -- see
docs/PHASE_7_REPORT.md's "Revised (RNG fix)" section) flips the final delegate-in hub
identity in 21/30 seeds (70%) on seed pool 30200-30229. This script asks the natural
follow-up, framework-agnostic question: does that sensitivity fall off with injection
depth? I.e., is the system's macroscopic outcome most sensitive to perturbations near
round 0, and progressively less sensitive to an equally "tiny" perturbation (one
reversed tie, nothing else touched) injected at 25%, 50%, or 75% of the way through the
same 800-round episode?

Method: for each of 30 fresh, disjoint seeds (40000-40029; verified disjoint by grep
against every `range(...)`-based seed pool used anywhere in experiments/**/*.py -- the
highest prior endpoint found was 30230, from run_path_dependence_test.py's
range(30200, 30230)), run the identical fully-controlled episode (randomized robot IDs,
shuffled processing order, randomized tie-break, all three Section 20 controls
simultaneously -- byte-identical machinery to run_path_dependence_test.py) once
unperturbed (baseline), then FOUR more times, each with a single perturbation injected
at the first reversible exact-EU tie occurring at or after a target round threshold:

  depth 0.00  (>= round 0,   i.e. the original "first tie in the episode" test)
  depth 0.25  (>= round 200)
  depth 0.50  (>= round 400)
  depth 0.75  (>= round 600)

(DECISIONS_PER_ROBOT = 800, same as every other phase_f/m/n/o script.) At most one
perturbation is applied per run; if no reversible tie exists at or after the target
round for a given seed/depth combination, that (seed, depth) cell is recorded as
"no perturbation point found" and excluded from that depth's flip-rate denominator,
reported explicitly.

For every valid (seed, depth) cell, record whether the final delegate-in hub identity
differs between the depth-specific perturbed run and the SAME seed's baseline run, and
report flip rate as a function of injection depth, with 95% percentile bootstrap CIs
(numpy only) over the seed-level flip indicators.

Usage: python -m experiments.phase_o.run_injection_depth_test
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetSimulator
from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.reasoning.contracts import DecisionResult
from src.communication.messaging import MessageBus
from src.delegation.policy import DelegationPolicy
from src.delegation.peer_memory import PeerMemory

from experiments.phase_f.run_specialization_revised import (
    N_ROBOTS, DECISIONS_PER_ROBOT, CONFIG, N_WINDOWS, LAMBDA_LATENCY, analyze,
)
from experiments.phase_m.run_large_n_resolution import bootstrap_ci

SEEDS = list(range(40000, 40030))  # 30 fresh seeds, disjoint from every prior pool (max prior endpoint 30230)
DEPTHS = [0.00, 0.25, 0.50, 0.75]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_episode_with_depth_injection(seed: int, target_round: int | None):
    """Fully-controlled episode (randomized robot IDs, shuffled processing order,
    randomized tie-break), with an optional single perturbation: at the FIRST delegate
    decision whose round_idx >= target_round AND whose expected-utility comparison among
    candidate peers is an exact tie that genuinely flips under reversed evaluation
    order, reverse that one decision's peer-evaluation order. If target_round is None,
    no perturbation is injected (baseline run). Everything else is byte-identical to
    run_path_dependence_test.run_episode_with_perturbation.

    Returns (activity, perturbation_info).
    """
    sim = FleetSimulator(CONFIG, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    policy = DelegationPolicy(lambda_latency=LAMBDA_LATENCY)
    bus = MessageBus()
    rng_ctrl = np.random.default_rng(seed + 900000)

    id_rng = np.random.default_rng(seed + 800000)
    perm = id_rng.permutation(N_ROBOTS)
    robot_id_map = {raw: int(perm[raw]) for raw in range(N_ROBOTS)}

    order_rng = np.random.default_rng(seed + 810000)
    tie_break_rng = np.random.default_rng(seed + 950000)

    peer_memories = [PeerMemory() for _ in range(N_ROBOTS)]

    window_len = DECISIONS_PER_ROBOT // N_WINDOWS
    activity = np.zeros((N_WINDOWS, N_ROBOTS, 4), dtype=int)

    perturbation_applied = False
    perturbation_round = None

    for round_idx in range(DECISIONS_PER_ROBOT):
        window = min(round_idx // window_len, N_WINDOWS - 1)
        peer_loads = {rid: 0 for rid in range(N_ROBOTS)}
        order = list(range(N_ROBOTS))
        order_rng.shuffle(order)
        for raw_rid in order:
            rid = robot_id_map[raw_rid]
            obs = sim.make_observation(raw_rid, det_backend.estimate)
            candidates = obs.candidates
            true_costs = [c.true_cost for c in candidates]
            min_cost = min(true_costs)
            det_idx = int(np.argmin(obs.det_estimate_costs))

            peer_id_order = list(range(N_ROBOTS))
            tie_break_rng.shuffle(peer_id_order)
            action, peer_id = policy.decide(raw_rid, obs, peer_memories[raw_rid], peer_id_order, peer_loads)

            if (target_round is not None and not perturbation_applied
                    and round_idx >= target_round and action == "delegate"):
                reversed_order = list(reversed(peer_id_order))
                alt_action, alt_peer_id = policy.decide(raw_rid, obs, peer_memories[raw_rid], reversed_order, peer_loads)
                if alt_action == "delegate" and alt_peer_id != peer_id:
                    perturbation_applied = True
                    perturbation_round = round_idx
                    action, peer_id = alt_action, alt_peer_id

            if action == "skip":
                idx, latency = det_idx, obs.det_estimate_latency
                activity[window, rid, 3] += 1
            elif action == "self":
                idx_r, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[raw_rid])
                idx, latency = idx_r, obs.det_estimate_latency + r_latency
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
                activity[window, robot_id_map[peer_id], 2] += 1

            sim.apply_decision(raw_rid, obs, DecisionResult(chosen_index=idx, latency=latency, used_reasoning=(action != "skip")))

    perturbation_info = {"applied": perturbation_applied, "round_idx": perturbation_round}
    return activity, perturbation_info


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_seed = {}
    depth_flip_lists = {d: [] for d in DEPTHS}
    depth_no_point = {d: 0 for d in DEPTHS}

    for seed in SEEDS:
        activity_base, _ = run_episode_with_depth_injection(seed, target_round=None)
        a_base = analyze(activity_base)
        baseline_hub = a_base["max_delegate_in_robot"]

        seed_entry = {"baseline_hub": baseline_hub, "depths": {}}
        for depth in DEPTHS:
            target_round = int(depth * DECISIONS_PER_ROBOT)
            activity_pert, pinfo = run_episode_with_depth_injection(seed, target_round=target_round)
            a_pert = analyze(activity_pert)
            cell = {
                "perturbation": pinfo,
                "perturbed_hub": a_pert["max_delegate_in_robot"],
                "perturbed_gini": a_pert["delegate_in_gini"],
            }
            if pinfo["applied"]:
                flipped = baseline_hub != a_pert["max_delegate_in_robot"]
                cell["hub_flipped"] = flipped
                depth_flip_lists[depth].append(flipped)
            else:
                cell["hub_flipped"] = None
                depth_no_point[depth] += 1
            seed_entry["depths"][str(depth)] = cell
        per_seed[str(seed)] = seed_entry

    depth_summary = {}
    for depth in DEPTHS:
        flips = depth_flip_lists[depth]
        n_valid = len(flips)
        flip_rate = float(np.mean(flips)) if flips else None
        ci = bootstrap_ci([float(f) for f in flips], seed=int(depth * 1000) + 7) if flips else None
        depth_summary[str(depth)] = {
            "target_round": int(depth * DECISIONS_PER_ROBOT),
            "n_seeds_total": len(SEEDS),
            "n_valid_perturbation_points": n_valid,
            "n_no_perturbation_point": depth_no_point[depth],
            "flip_rate": flip_rate,
            "n_flips": int(sum(flips)) if flips else 0,
            "flip_rate_bootstrap_ci": ci,
        }

    summary = {
        "seed_pool": {"start": SEEDS[0], "end": SEEDS[-1], "count": len(SEEDS)},
        "depths_tested": DEPTHS,
        "decisions_per_robot": DECISIONS_PER_ROBOT,
        "by_depth": depth_summary,
    }

    with open(os.path.join(RESULTS_DIR, "injection_depth_per_seed.json"), "w") as f:
        json.dump(per_seed, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "injection_depth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
