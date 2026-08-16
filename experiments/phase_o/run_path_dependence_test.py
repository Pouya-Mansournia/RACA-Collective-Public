"""Phase O -- path-dependence / sensitivity-to-early-conditions test for the residual
delegate-in concentration (Gini ~0.478, n=50, 95% CI [0.446, 0.512], non-overlapping with
the null model's ~0.091, see docs/PHASE_7_REPORT.md "Definitive resolution" and
"Mechanism investigation" sections). Five prior mechanistic hypotheses (early-luck
rich-get-richer, opportunity-count privilege, spatial privilege, whole-episode
success-rate reinforcement, unbounded-memory accumulation) were tested via linear/rank
CORRELATION and ruled out. This script tests a methodologically DIFFERENT hypothesis --
genuine dynamical path dependence -- that would not necessarily show up as a simple
correlation with an early-window statistic: does a single, tiny, controlled perturbation
injected very early in an otherwise-identical episode change WHO ends up as the final
delegate-in hub?

Method: for each seed, run the fully-controlled episode (all three Section 20 controls
simultaneously: randomized robot IDs, shuffled processing order, randomized tie-break --
identical to experiments/phase_m/run_large_n_resolution.run_episode_full_control) once
unperturbed. Then rerun the SAME seed with byte-identical setup, except: scan forward
through the episode for the first "delegate" decision that is an exact expected-utility
TIE among two or more candidate peers (DelegationPolicy.decide() uses strict `>` so ties
are common early, before any peer has an observed track record -- see
docs/PHASE_7_REPORT.md's "Revised (RNG fix)" tie-breaking diagnosis). At that one decision,
reverse the (already-tie-break-RNG-shuffled) peer evaluation order before calling
decide() a second time; if this actually changes which peer is chosen (i.e. the tie was
real, not merely detected), that is the injected perturbation -- apply it and let the
rest of the episode run through completely untouched machinery (same seed, same RNG
streams, same everything else). No other decision in the episode is touched. If no tied
delegate decision that actually flips the outcome exists in the entire episode for a
given seed, that seed is recorded as "no perturbation point found" and excluded from the
flip-rate denominator (reported explicitly, not silently dropped).

For every seed with a valid perturbation point, record:
  - the final delegate-in hub identity (argmax delegate-in count) in the unperturbed run,
  - the final delegate-in hub identity in the perturbed run,
  - whether the hub identity differs (a "flip"),
  - the delegate-in Gini in each run and the shift between them.

Fresh, disjoint seed pool: 30200-30229 (30 seeds). Verified disjoint by grep against
every `range(...)`-based seed pool used anywhere in experiments/**/*.py: the highest
prior endpoint found was 30150 (experiments/phase_n/run_mechanism_investigation.py's
range(30100, 30150)); every other pool in this repository ends at or below 25915.

Usage: python -m experiments.phase_o.run_path_dependence_test
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
    N_ROBOTS, DECISIONS_PER_ROBOT, CONFIG, N_WINDOWS, LAMBDA_LATENCY, analyze, gini,
)
from experiments.phase_m.run_large_n_resolution import bootstrap_ci

SEEDS = list(range(30200, 30230))  # 30 fresh seeds, disjoint from every prior pool (max prior endpoint 30150)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_episode_with_perturbation(seed: int, inject_perturbation: bool):
    """Fully-controlled episode (randomized robot IDs, shuffled processing order,
    randomized tie-break -- identical to phase_m's run_episode_full_control), with an
    optional single early perturbation: at the FIRST delegate decision in the episode
    whose expected-utility comparison among candidate peers is an exact tie AND whose
    outcome actually changes when the peer-evaluation order is reversed, reverse that
    one decision's order. Everything before and after that single decision is untouched.

    Returns (activity, perturbation_info) where perturbation_info records whether/where
    the perturbation was applied.
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
    perturbation_original_peer = None
    perturbation_flipped_peer = None

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

            if inject_perturbation and not perturbation_applied and action == "delegate":
                reversed_order = list(reversed(peer_id_order))
                alt_action, alt_peer_id = policy.decide(raw_rid, obs, peer_memories[raw_rid], reversed_order, peer_loads)
                if alt_action == "delegate" and alt_peer_id != peer_id:
                    # genuine tie that flips under reversed evaluation order: this is the
                    # single, tiny, controlled early perturbation
                    perturbation_applied = True
                    perturbation_round = round_idx
                    perturbation_original_peer = peer_id
                    perturbation_flipped_peer = alt_peer_id
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

    perturbation_info = {
        "applied": perturbation_applied,
        "round_idx": perturbation_round,
        "original_peer_raw_id": perturbation_original_peer,
        "flipped_peer_raw_id": perturbation_flipped_peer,
    }
    return activity, perturbation_info


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_seed = {}
    flips = []
    gini_shifts = []
    gini_baseline_vals = []
    gini_perturbed_vals = []
    n_no_perturbation_point = 0

    for seed in SEEDS:
        activity_base, _ = run_episode_with_perturbation(seed, inject_perturbation=False)
        a_base = analyze(activity_base)

        activity_pert, pinfo = run_episode_with_perturbation(seed, inject_perturbation=True)
        a_pert = analyze(activity_pert)

        entry = {
            "perturbation": pinfo,
            "baseline_hub": a_base["max_delegate_in_robot"],
            "perturbed_hub": a_pert["max_delegate_in_robot"],
            "baseline_gini": a_base["delegate_in_gini"],
            "perturbed_gini": a_pert["delegate_in_gini"],
        }

        if pinfo["applied"]:
            flipped = a_base["max_delegate_in_robot"] != a_pert["max_delegate_in_robot"]
            entry["hub_flipped"] = flipped
            flips.append(flipped)
            gini_shifts.append(abs(a_pert["delegate_in_gini"] - a_base["delegate_in_gini"]))
            gini_baseline_vals.append(a_base["delegate_in_gini"])
            gini_perturbed_vals.append(a_pert["delegate_in_gini"])
        else:
            entry["hub_flipped"] = None
            n_no_perturbation_point += 1

        per_seed[str(seed)] = entry

    n_valid = len(flips)
    flip_rate = float(np.mean(flips)) if flips else None
    mean_abs_gini_shift = float(np.mean(gini_shifts)) if gini_shifts else None

    summary = {
        "seed_pool": {"start": SEEDS[0], "end": SEEDS[-1], "count": len(SEEDS)},
        "n_seeds_total": len(SEEDS),
        "n_seeds_with_valid_perturbation_point": n_valid,
        "n_seeds_no_perturbation_point_found": n_no_perturbation_point,
        "hub_flip_rate_among_valid_seeds": flip_rate,
        "n_hub_flips": int(sum(flips)) if flips else 0,
        "mean_abs_gini_shift": mean_abs_gini_shift,
        "gini_shift_ci": bootstrap_ci(gini_shifts, seed=3) if gini_shifts else None,
        "baseline_gini_ci": bootstrap_ci(gini_baseline_vals, seed=4) if gini_baseline_vals else None,
        "perturbed_gini_ci": bootstrap_ci(gini_perturbed_vals, seed=5) if gini_perturbed_vals else None,
    }

    with open(os.path.join(RESULTS_DIR, "path_dependence_per_seed.json"), "w") as f:
        json.dump(per_seed, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "path_dependence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
