"""Lightweight pure-Python warehouse-fleet simulator.

No ROS2/Gazebo. Each simulated "step" is a task-assignment decision for one
robot: the robot is offered a set of candidate tasks (stations to service)
with unknown true costs, must pick one via a routing policy + reasoning
backend, and the outcome (true cost incurred) drains its battery. Running
out of battery on a task counts as a task failure and resets the robot's
battery (representing a recharge trip), which is how "task completion rate"
becomes a meaningful, cost-sensitive metric.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from src.reasoning.contracts import Candidate, Observation, DecisionResult


@dataclasses.dataclass
class FleetConfig:
    num_robots: int = 10
    num_stations: int = 8
    grid_size: float = 50.0
    min_candidates: int = 2
    max_candidates: int = 6
    decisions_per_robot: int = 60
    battery_drain_scale: float = 0.01
    cost_noise_scale: float = 1.0  # scale of intrinsic candidate cost spread
    # --- difficulty axis (Phase 2 revised cost regime) ---
    # Each decision is independently drawn "hard" or "easy". Hard decisions keep the
    # per-candidate cost spread AROUND ITS OWN MEAN wide (stakes are big: picking the
    # wrong candidate is expensive), easy decisions collapse candidates to near-identical
    # cost (nothing to gain from extra reasoning). The additive per-candidate noise that
    # creates near-ties scales with the same multiplier as the spread, so the *relative*
    # chance of a near-tie is similar at both difficulty levels, but the backends' cost
    # ESTIMATION noise (fixed std, see reasoning/backends.py) does not scale with
    # difficulty. That fixed-magnitude estimation noise is what lets a lower-noise
    # backend (reasoning) avoid mistakes on hard decisions that a higher-noise backend
    # (deterministic) sometimes makes, and those avoided mistakes are worth a lot more
    # in absolute terms on hard decisions than on easy ones.
    difficulty_hard_fraction: float = 0.2
    hard_spread_multiplier: float = 5.0
    easy_spread_multiplier: float = 0.2


class FleetSimulator:
    """Generates decision states and applies routing outcomes.

    Determinism: all randomness flows from a single seed via
    `np.random.default_rng(seed)`, and each robot gets its own child
    generator derived from that seed plus its robot id, so results are
    exactly reproducible for a given (config, seed).
    """

    def __init__(self, config: FleetConfig, seed: int):
        self.config = config
        self.seed = seed
        self._root_rng = np.random.default_rng(seed)
        # station positions fixed per seed (part of "geometry")
        self.stations = self._root_rng.uniform(0, config.grid_size, size=(config.num_stations, 2))
        # Per-robot RNG streams: previously constructed as
        # `default_rng(root_rng.integers(...) + rid)`, which added the robot's integer ID
        # directly to a shared base draw before seeding. That created a STRUCTURED,
        # ID-correlated offset between adjacent robots' streams (robot 0's stream was
        # deterministically base_draw+0, robot 1's base_draw+1, ...), which is exactly the
        # kind of "fixed random seed" / "unequal initial state" artifact
        # RACA-Collective.txt Section 20 warns about (see Phase 6/7 "Revised (RNG fix)"
        # sections for the diagnosis and rerun). Fixed here by spawning independent,
        # non-adjacent child streams from a single SeedSequence: each robot's stream is
        # statistically independent of every other robot's, and no robot's stream is a
        # deterministic small-integer offset of another's.
        self._seed_sequence = np.random.SeedSequence(seed)
        child_sequences = self._seed_sequence.spawn(config.num_robots)
        self.robot_rngs = [np.random.default_rng(child_sequences[rid]) for rid in range(config.num_robots)]
        self.battery_soc = [1.0 for _ in range(config.num_robots)]
        self.recent_latency = [[] for _ in range(config.num_robots)]

    def _make_candidates(self, rng: np.random.Generator) -> tuple[list[Candidate], bool]:
        cfg = self.config
        k = int(rng.integers(cfg.min_candidates, cfg.max_candidates + 1))
        idx = rng.choice(cfg.num_stations, size=min(k, cfg.num_stations), replace=False)
        pos_a = self.stations[idx]
        idx_b = rng.choice(cfg.num_stations, size=len(idx))
        pos_b = self.stations[idx_b]
        is_hard = bool(rng.random() < cfg.difficulty_hard_fraction)
        spread = cfg.hard_spread_multiplier if is_hard else cfg.easy_spread_multiplier
        base_costs = np.linalg.norm(pos_a - pos_b, axis=1) * cfg.cost_noise_scale
        mean_cost = float(np.mean(base_costs)) if len(base_costs) else 0.0
        true_costs = mean_cost + (base_costs - mean_cost) * spread
        true_costs = true_costs + rng.normal(0, 0.15 * cfg.cost_noise_scale * spread, size=len(true_costs))
        true_costs = np.clip(true_costs, 0.05, None)
        return [Candidate(id=int(i), true_cost=float(c)) for i, c in enumerate(true_costs)], is_hard

    def make_observation(self, robot_id: int, det_estimate_fn: Callable) -> Observation:
        rng = self.robot_rngs[robot_id]
        candidates, is_hard = self._make_candidates(rng)
        est_costs, est_latency = det_estimate_fn(candidates, rng)
        sorted_costs = sorted(est_costs)
        margin = (sorted_costs[1] - sorted_costs[0]) if len(sorted_costs) > 1 else 0.0
        ambiguity = float(np.clip(1.0 - margin / max(sorted_costs[0], 1e-6) / 1.0, 0.0, 1.0)) \
            if len(sorted_costs) > 1 else 0.0
        urgency = float(np.clip(1.0 - self.battery_soc[robot_id], 0.0, 1.0))
        recent = self.recent_latency[robot_id][-10:]
        recent_latency_mean = float(np.mean(recent)) if recent else 0.0
        return Observation(
            robot_id=robot_id,
            candidates=candidates,
            det_estimate_costs=est_costs,
            det_estimate_latency=est_latency,
            ambiguity=ambiguity,
            urgency=urgency,
            candidate_count=len(candidates),
            cost_margin=margin,
            battery_soc=self.battery_soc[robot_id],
            recent_latency_mean=recent_latency_mean,
            is_hard=is_hard,
        )

    def apply_decision(self, robot_id: int, obs: Observation, result: DecisionResult) -> dict:
        chosen = obs.candidates[result.chosen_index]
        true_costs = [c.true_cost for c in obs.candidates]
        min_cost = min(true_costs)
        regret = chosen.true_cost - min_cost
        drain = chosen.true_cost * self.config.battery_drain_scale
        self.battery_soc[robot_id] -= drain
        failed = self.battery_soc[robot_id] < 0.0
        if failed:
            self.battery_soc[robot_id] = 1.0
        self.recent_latency[robot_id].append(result.latency)
        return {
            "true_cost": chosen.true_cost,
            "min_cost": min_cost,
            "regret": regret,
            "latency": result.latency,
            "used_reasoning": result.used_reasoning,
            "failed": failed,
            "battery_soc": self.battery_soc[robot_id],
        }

    def run(self, router) -> list[dict]:
        """Runs decisions_per_robot decisions for every robot, round-robin."""
        records = []
        for _ in range(self.config.decisions_per_robot):
            for rid in range(self.config.num_robots):
                obs = self.make_observation(rid, router.det_estimate)
                result = router.decide(obs, self.robot_rngs[rid])
                rec = self.apply_decision(rid, obs, result)
                rec["robot_id"] = rid
                records.append(rec)
        return records
