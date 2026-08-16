"""Phase N -- mechanism investigation for the residual delegate-in concentration
(Gini ~0.478, n=50, 95% CI [0.446, 0.512]) confirmed real (vs a null random-choice
model's Gini ~0.091) by experiments/phase_m/run_large_n_resolution.py, whose one tested
mechanistic hypothesis ("rich-get-richer from early lucky successes", first 10% of the
episode) was directly ruled out (Pearson r=0.086). This script tests further, more
precisely stated hypotheses for WHAT actually produces the concentration, per the
investigation requested in docs/PHASE_7_REPORT.md "Mechanism investigation" section.

Hypotheses tested, in order:

H1 -- decision-opportunity-count: does a robot's number of DELEGATION-CANDIDACY
  opportunities (rounds in which it was under peer_capacity, i.e. eligible to receive a
  delegation, when some other robot was choosing a peer) correlate with its final
  delegate-in share? All three Section 20 artifact controls are active (randomized IDs,
  shuffled processing order, randomized tie-break), same as Phase M.

H2 -- spatial/positional privilege: analytical, not simulated. src/environment/simulator.py's
  FleetSimulator has no per-robot spatial position at all -- candidates for a robot's
  decision are drawn entirely from that robot's own private RNG stream
  (`self.robot_rngs[robot_id]`), not from proximity to a station or to other robots.
  There is no code path in which one robot is placed nearer a "busy station" or given
  first claim in a round-robin queue that depends on anything but a random RNG stream
  spawned identically for every robot (`SeedSequence(seed).spawn(num_robots)`, Phase
  6/7's RNG fix). This hypothesis is therefore ruled out by construction, not simulation
  (see the writeup); it is recorded here rather than run because there is no station- or
  position-dependent code path to ablate.

H3 -- whole-episode cumulative success-rate reinforcement: does a robot's FINAL,
  whole-episode delegate-in success_rate (as recorded in its delegators' PeerMemory,
  cumulative over all rounds, not just the first 10%) correlate with its final
  delegate-in share? This is the untested alternative flagged at the end of Phase M:
  maybe persistence reflects average quality accumulated gradually across the WHOLE
  episode rather than a first-decile lucky break.

H4 -- unbounded-memory-accumulation ablation: PeerMemory's success_rate is a cumulative,
  unbounded running average over the whole episode (src/delegation/peer_memory.py) -- one
  early success or failure never fully leaves the estimate, it just gets diluted. This
  ablation reruns the identical fully-controlled setup with WindowedPeerMemory
  (src/delegation/windowed_peer_memory.py), which only remembers each peer's most recent
  `window` responses, so old evidence eventually ages out completely. If unbounded
  accumulation is what lets an early quality edge compound into permanent concentration,
  bounding the window should measurably shrink Gini toward the null model's ~0.091. If
  Gini is essentially unchanged, unbounded accumulation is not the mechanism.

All four hypotheses use a fresh 50-seed pool, 30100-30149, verified disjoint from every
`range(...)`-based seed pool used anywhere in experiments/**/*.py (highest previously
used endpoint found by grep: 30049, from experiments/phase_m/run_large_n_resolution.py;
every other pool used in this repository ends at or below 25915).

Usage: python -m experiments.phase_n.run_mechanism_investigation
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
from src.delegation.windowed_peer_memory import WindowedPeerMemory
from src.memory.conditions import MemoryConditionConfig, make_peer_memories

from experiments.phase_f.run_specialization_revised import (
    N_ROBOTS, DECISIONS_PER_ROBOT, CONFIG, LAMBDA_LATENCY, analyze,
)
from experiments.phase_m.run_large_n_resolution import bootstrap_ci, pearson, spearman

SEEDS = list(range(30100, 30150))  # 50 fresh seeds, disjoint from 30000-30049 and every older pool
WINDOW = 50  # H4 ablation: number of most-recent responses a peer's stats remember

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_episode_mechanism(seed: int, peer_memory_factory, track_opportunity: bool = False,
                           track_full_success: bool = False):
    """Same fully-controlled episode as experiments/phase_m's run_episode_full_control
    (randomized robot IDs + shuffled processing order + randomized tie-break, all three
    simultaneously), generalized to accept any peer-memory factory (real PeerMemory or
    WindowedPeerMemory for the H4 ablation) and optionally track:
      - opportunity_counts[mapped_robot_id]: number of times this robot was under
        peer_capacity (eligible to be delegated to) at the moment some OTHER robot made
        a delegate/self/skip decision this round (H1).
      - final cumulative success_rate the delegators' PeerMemory holds for each robot at
        the END of the episode (H3), read directly off PeerStats after the run.
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

    peer_memories = [peer_memory_factory() for _ in range(N_ROBOTS)]

    n_windows = 4
    window_len = DECISIONS_PER_ROBOT // n_windows
    activity = np.zeros((n_windows, N_ROBOTS, 4), dtype=int)

    opportunity_counts = np.zeros(N_ROBOTS, dtype=int) if track_opportunity else None

    for round_idx in range(DECISIONS_PER_ROBOT):
        window = min(round_idx // window_len, n_windows - 1)
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

            if track_opportunity:
                for other_raw in range(N_ROBOTS):
                    if other_raw == raw_rid:
                        continue
                    if peer_loads.get(other_raw, 0) < policy.peer_capacity:
                        opportunity_counts[robot_id_map[other_raw]] += 1

            peer_id_order = list(range(N_ROBOTS))
            tie_break_rng.shuffle(peer_id_order)
            action, peer_id = policy.decide(raw_rid, obs, peer_memories[raw_rid], peer_id_order, peer_loads)

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

    final_success_rate = None
    if track_full_success:
        # pooled, whole-episode cumulative success_rate each delegator ended up recording
        # for each (raw) peer id, mapped to the peer's displayed identity, weighted by
        # that delegator's own number of responses from that peer (so a peer's overall
        # "reputation" this episode is the pooled success rate across everyone who used it)
        pooled_success = {r: [0, 0] for r in range(N_ROBOTS)}  # mapped_id -> [n_success, n_responses]
        for raw_rid in range(N_ROBOTS):
            for raw_peer_id, stats in peer_memories[raw_rid].stats.items():
                mapped = robot_id_map[raw_peer_id]
                pooled_success[mapped][0] += stats.n_success
                pooled_success[mapped][1] += stats.n_responses
        final_success_rate = {
            r: (pooled_success[r][0] / pooled_success[r][1] if pooled_success[r][1] > 0 else None)
            for r in range(N_ROBOTS)
        }
        final_n_responses = {r: pooled_success[r][1] for r in range(N_ROBOTS)}
    else:
        final_n_responses = None

    return activity, opportunity_counts, final_success_rate, final_n_responses


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- H1 + H3: real (unbounded) PeerMemory, tracking opportunity counts and final
    # cumulative success rate, on the same run as the headline Gini/persistence stats.
    real_stats = {"gini": [], "hhi": [], "entropy": [], "persistence": []}
    opportunity_vals, share_vals_h1, delegate_in_raw_h1 = [], [], []
    success_rate_vals, share_vals_h3, n_responses_h3 = [], [], []

    decisions_seen_as_other = DECISIONS_PER_ROBOT * (N_ROBOTS - 1)  # constant per robot: rounds * (N-1 other deciders)

    for seed in SEEDS:
        activity, opp_counts, final_success, final_n_resp = run_episode_mechanism(
            seed, peer_memory_factory=PeerMemory,
            track_opportunity=True, track_full_success=True,
        )
        a = analyze(activity)
        real_stats["gini"].append(a["delegate_in_gini"])
        real_stats["hhi"].append(a["delegate_in_hhi"])
        real_stats["entropy"].append(a["mean_entropy"])
        real_stats["persistence"].append(a["mean_role_persistence_rank_corr"])

        total_in = sum(a["delegate_in_counts"])
        if total_in > 0:
            for r in range(N_ROBOTS):
                share = a["delegate_in_counts"][r] / total_in
                opportunity_vals.append(int(opp_counts[r]))
                share_vals_h1.append(share)
                delegate_in_raw_h1.append(int(a["delegate_in_counts"][r]))
                if final_success[r] is not None:
                    success_rate_vals.append(final_success[r])
                    share_vals_h3.append(share)
                    n_responses_h3.append(int(final_n_resp[r]))

    real_ci = {k: bootstrap_ci(v, seed=1) for k, v in real_stats.items()}

    h1_pearson = pearson(opportunity_vals, share_vals_h1)
    h1_spearman = spearman(opportunity_vals, share_vals_h1)
    h3_pearson = pearson(success_rate_vals, share_vals_h3)
    h3_spearman = spearman(success_rate_vals, share_vals_h3)

    # H1 mechanism diagnostic: "opportunity count" as tracked here is depressed within a
    # round the moment a robot has already received peer_capacity (=2) delegations that
    # round, because it then reads as unavailable to every later decider in that same
    # round -- i.e. becoming a hub MECHANICALLY suppresses this specific metric, rather
    # than low opportunity causing hub status. Test this directly: "unavailable_count"
    # (decisions_seen_as_other - opportunity_count) should correlate strongly POSITIVELY
    # with raw delegate-in count if this capacity-saturation explanation is right.
    unavailable_vals = [decisions_seen_as_other - o for o in opportunity_vals]
    h1_mechanism_pearson = pearson(unavailable_vals, delegate_in_raw_h1)
    h1_mechanism_spearman = spearman(unavailable_vals, delegate_in_raw_h1)
    h1_raw_pearson = pearson(opportunity_vals, delegate_in_raw_h1)

    # H3 outlier check: a robot-seed sample's success_rate is a noisy estimate when it is
    # pooled from very few responses (e.g. a robot that only ever received 1-2 delegate-in
    # events this episode can show success_rate 0.0 or 1.0 by pure chance, and it also
    # necessarily has a tiny delegate-in share, potentially inflating a spurious Pearson
    # correlation through a handful of extreme, low-count points). Recompute Pearson/
    # Spearman restricted to samples with at least MIN_RESPONSES pooled responses, and
    # separately report how many low-count points were excluded.
    MIN_RESPONSES_H3 = 10
    reliable_mask = [n >= MIN_RESPONSES_H3 for n in n_responses_h3]
    n_excluded_low_count = int(len(reliable_mask) - sum(reliable_mask))
    success_rate_filtered = [sr for sr, keep in zip(success_rate_vals, reliable_mask) if keep]
    share_filtered = [sh for sh, keep in zip(share_vals_h3, reliable_mask) if keep]
    n_responses_filtered = [n for n, keep in zip(n_responses_h3, reliable_mask) if keep]
    h3_pearson_filtered = pearson(success_rate_filtered, share_filtered)
    h3_spearman_filtered = spearman(success_rate_filtered, share_filtered)
    h3_min_responses_threshold = MIN_RESPONSES_H3

    # ---- H4: windowed-memory ablation, same 50 seeds, same controls, only the peer
    # memory implementation differs (bounded window vs unbounded cumulative).
    windowed_stats = {"gini": [], "hhi": [], "entropy": [], "persistence": []}
    for seed in SEEDS:
        activity, _, _, _ = run_episode_mechanism(
            seed, peer_memory_factory=lambda: WindowedPeerMemory(window=WINDOW),
            track_opportunity=False, track_full_success=False,
        )
        a = analyze(activity)
        windowed_stats["gini"].append(a["delegate_in_gini"])
        windowed_stats["hhi"].append(a["delegate_in_hhi"])
        windowed_stats["entropy"].append(a["mean_entropy"])
        windowed_stats["persistence"].append(a["mean_role_persistence_rank_corr"])

    windowed_ci = {k: bootstrap_ci(v, seed=2) for k, v in windowed_stats.items()}

    real_gini_ci = real_ci["gini"]
    windowed_gini_ci = windowed_ci["gini"]
    h4_ci_overlap = not (real_gini_ci["ci_lo"] > windowed_gini_ci["ci_hi"]
                          or windowed_gini_ci["ci_lo"] > real_gini_ci["ci_hi"])

    summary = {
        "seed_pool": {"start": SEEDS[0], "end": SEEDS[-1], "count": len(SEEDS)},
        "H1_opportunity_count": {
            "description": "correlation of per-robot delegation-candidacy opportunity count "
                            "(rounds under peer_capacity when another robot decided) vs final delegate-in share",
            "n_samples": len(opportunity_vals),
            "pearson_r": h1_pearson,
            "spearman_r": h1_spearman,
            "mechanism_diagnostic": {
                "description": "tests whether the negative H1 correlation is a mechanical artifact of "
                                "the peer_capacity=2 constraint: once a robot has already received "
                                "peer_capacity delegations within a round, it reads as 'unavailable' to "
                                "every later decider that same round, mechanically depressing its own "
                                "opportunity_count in direct proportion to how much delegate-in traffic "
                                "it is already receiving. unavailable_count = decisions_seen_as_other - "
                                "opportunity_count; if this is the explanation, unavailable_count should "
                                "correlate strongly POSITIVELY with raw delegate-in count.",
                "unavailable_count_vs_raw_delegate_in_pearson": h1_mechanism_pearson,
                "unavailable_count_vs_raw_delegate_in_spearman": h1_mechanism_spearman,
                "opportunity_count_vs_raw_delegate_in_pearson": h1_raw_pearson,
            },
        },
        "H2_spatial_positional": {
            "description": "ruled out by construction (analytical, not simulated): "
                            "FleetSimulator has no per-robot spatial position; every robot's candidate "
                            "set is drawn purely from its own private RNG stream, not proximity to a "
                            "station or another robot. See src/environment/simulator.py.",
            "tested_by_simulation": False,
        },
        "H3_whole_episode_success_rate": {
            "description": "correlation of each robot's pooled, whole-episode cumulative delegate-in "
                            "success_rate vs final delegate-in share (contrast with Phase M's first-10%-only check)",
            "n_samples": len(success_rate_vals),
            "pearson_r": h3_pearson,
            "spearman_r": h3_spearman,
            "outlier_check": {
                "description": f"recomputed excluding robot-seed samples with fewer than "
                                f"{h3_min_responses_threshold} pooled delegate-in responses this episode "
                                "(unreliable success_rate estimate)",
                "n_excluded_low_count": n_excluded_low_count,
                "n_samples_filtered": len(success_rate_filtered),
                "pearson_r_filtered": h3_pearson_filtered,
                "spearman_r_filtered": h3_spearman_filtered,
            },
        },
        "H4_windowed_memory_ablation": {
            "description": f"unbounded PeerMemory vs WindowedPeerMemory(window={WINDOW}), same seeds/controls",
            "real_unbounded_gini_ci": real_gini_ci,
            "windowed_gini_ci": windowed_gini_ci,
            "gini_ci_overlap": h4_ci_overlap,
            "real_unbounded_full": real_ci,
            "windowed_full": windowed_ci,
        },
    }

    with open(os.path.join(RESULTS_DIR, "mechanism_investigation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
