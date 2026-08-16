"""Phase 2: counterfactual outcome dataset generation.

For each sampled decision state x, we evaluate BOTH the deterministic and the
expensive reasoning backend on the SAME candidate set (same true costs), each
with its own independent noise draw, without letting an online router choose
between them and without mutating simulator state (battery, recent latency
history) based on the counterfactual outcomes. Only one of the two branches
(matching the actually-deployed policy) is ever applied to update state in
the online experiments; this module is for offline analysis only.

Units: cost is in the same "distance" units as the simulator's true costs.
Latency is in seconds. We combine them into a single utility via a latency
price `lambda_latency` (cost units per second), the same constant used by
the Phase 1 heuristic router, so that Oracle/B0/B1/B2 utilities are directly
comparable and mean the same thing across baselines.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from src.environment.simulator import FleetSimulator, FleetConfig
from src.reasoning.backends import DeterministicBackend, ExpensiveReasoningBackend
from src.routing.baselines import AlwaysDeterministic
from src.routing.heuristic import HeuristicRouter


@dataclasses.dataclass
class CounterfactualRecord:
    ambiguity: float
    urgency: float
    candidate_count: int
    cost_margin: float
    battery_soc: float
    recent_latency_mean: float
    is_hard: bool
    regret_det: float
    latency_det: float
    regret_reasoning: float
    latency_reasoning: float
    heuristic_used_reasoning: bool


def utility(regret: float, latency: float, lambda_latency: float) -> float:
    return -regret - lambda_latency * latency


def generate_dataset(config: FleetConfig, seed: int, n_samples: int,
                      lambda_latency: float = 0.15) -> list[CounterfactualRecord]:
    sim = FleetSimulator(config, seed=seed)
    det_backend = DeterministicBackend()
    reasoning_backend = ExpensiveReasoningBackend()
    heuristic = HeuristicRouter(det_backend=det_backend, reasoning_backend=reasoning_backend)
    records = []
    made = 0
    robot_cycle = 0
    while made < n_samples:
        rid = robot_cycle % config.num_robots
        robot_cycle += 1
        obs = sim.make_observation(rid, det_backend.estimate)
        candidates = obs.candidates
        true_costs = [c.true_cost for c in candidates]
        min_cost = min(true_costs)

        det_idx = int(np.argmin(obs.det_estimate_costs))
        regret_det = candidates[det_idx].true_cost - min_cost
        latency_det = obs.det_estimate_latency

        r_idx, r_latency = reasoning_backend.decide(candidates, sim.robot_rngs[rid])
        regret_reasoning = candidates[r_idx].true_cost - min_cost
        latency_reasoning_total = obs.det_estimate_latency + r_latency

        heuristic_gap = heuristic.utility_gap(obs)

        records.append(CounterfactualRecord(
            ambiguity=obs.ambiguity,
            urgency=obs.urgency,
            candidate_count=obs.candidate_count,
            cost_margin=obs.cost_margin,
            battery_soc=obs.battery_soc,
            recent_latency_mean=obs.recent_latency_mean,
            is_hard=obs.is_hard,
            regret_det=regret_det,
            latency_det=latency_det,
            regret_reasoning=regret_reasoning,
            latency_reasoning=latency_reasoning_total,
            heuristic_used_reasoning=heuristic_gap > 0,
        ))
        # advance simulator state using the deterministic (non-counterfactual) outcome
        # so battery/recent-latency evolve realistically without letting the
        # counterfactual evaluation leak into the online trajectory.
        from src.reasoning.contracts import DecisionResult
        sim.apply_decision(rid, obs, DecisionResult(chosen_index=det_idx, latency=latency_det, used_reasoning=False))
        made += 1
    return records
