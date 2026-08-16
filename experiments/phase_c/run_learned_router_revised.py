"""Phase 3 revised: train B3 on the counterfactual dataset under the new difficulty-axis
cost regime (see docs/PHASE_2_REPORT.md addendum), fresh seed pools disjoint from the
original 3000s/3100s/3200s/3300s pools (shifted +10000).

Seed pools:
  development seeds : 13000-13009 (10)
  calibration seeds : 13100-13109 (10)
  validation seeds  : 13200-13209 (10)
  held-out test     : 13300-13319 (20)

Usage: python -m experiments.phase_c.run_learned_router_revised
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.simulator import FleetConfig
from src.reasoning.counterfactual import generate_dataset, utility
from src.reasoning.value_estimator import LogisticValueEstimator, featurize, label
from src.routing.oracle import oracle_utility
from src.evaluation.metrics import bootstrap_ci

DEV_SEEDS = list(range(13000, 13010))
CAL_SEEDS = list(range(13100, 13110))
VAL_SEEDS = list(range(13200, 13210))
TEST_SEEDS = list(range(13300, 13320))
LAMBDA_LATENCY = 0.15
N_SAMPLES_PER_SEED = 300

CONFIGS = {
    "small_dense": FleetConfig(num_robots=8, num_stations=6, decisions_per_robot=80,
                                min_candidates=2, max_candidates=4, grid_size=20.0),
    "large_sparse": FleetConfig(num_robots=25, num_stations=20, decisions_per_robot=80,
                                 min_candidates=3, max_candidates=8, grid_size=80.0),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_dataset(cfg, seeds, n_per_seed=N_SAMPLES_PER_SEED):
    records = []
    for seed in seeds:
        records.extend(generate_dataset(cfg, seed=seed, n_samples=n_per_seed, lambda_latency=LAMBDA_LATENCY))
    return records


def to_xy(records):
    X = np.stack([featurize(r) for r in records])
    y = np.array([label(r, LAMBDA_LATENCY) for r in records], dtype=float)
    return X, y


def train_estimator(cfg):
    dev_records = build_dataset(cfg, DEV_SEEDS)
    X, y = to_xy(dev_records)
    est = LogisticValueEstimator().fit(X, y, lr=0.3, epochs=800, seed=0)
    return est, dev_records


def calibrate_tau(est, cal_records):
    X = np.stack([featurize(r) for r in cal_records])
    p = est.predict_proba(X)
    best_tau, best_u = 0.5, -np.inf
    for tau in np.linspace(0.05, 0.95, 19):
        utils = []
        for r, pi in zip(cal_records, p):
            if pi > tau:
                utils.append(utility(r.regret_reasoning, r.latency_reasoning, LAMBDA_LATENCY))
            else:
                utils.append(utility(r.regret_det, r.latency_det, LAMBDA_LATENCY))
        mean_u = float(np.mean(utils))
        if mean_u > best_u:
            best_u, best_tau = mean_u, float(tau)
    return best_tau, best_u


def policy_utilities(records, est, tau):
    X = np.stack([featurize(r) for r in records])
    p = est.predict_proba(X)
    u_b0 = [utility(r.regret_det, r.latency_det, LAMBDA_LATENCY) for r in records]
    u_b1 = [utility(r.regret_reasoning, r.latency_reasoning, LAMBDA_LATENCY) for r in records]
    u_b2 = [
        utility(r.regret_reasoning, r.latency_reasoning, LAMBDA_LATENCY) if r.heuristic_used_reasoning
        else utility(r.regret_det, r.latency_det, LAMBDA_LATENCY)
        for r in records
    ]
    u_b3 = [
        utility(r.regret_reasoning, r.latency_reasoning, LAMBDA_LATENCY) if pi > tau
        else utility(r.regret_det, r.latency_det, LAMBDA_LATENCY)
        for r, pi in zip(records, p)
    ]
    oracle_pairs = [oracle_utility(r, LAMBDA_LATENCY) for r in records]
    u_oracle = [o[0] for o in oracle_pairs]
    b3_escalation_rate = float(np.mean(p > tau))
    return {"B0": u_b0, "B1": u_b1, "B2": u_b2, "B3": u_b3, "Oracle": u_oracle}, b3_escalation_rate


def summarize_utils(utils, boot_seed_offset=0):
    out = {}
    for i, (name, vals) in enumerate(utils.items()):
        point, lo, hi = bootstrap_ci(vals, seed=boot_seed_offset + i)
        out[f"utility_{name}"] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
    for name in ["B0", "B1", "B2", "B3"]:
        regret_vs_oracle = [o - x for o, x in zip(utils["Oracle"], utils[name])]
        point, lo, hi = bootstrap_ci(regret_vs_oracle, seed=boot_seed_offset + 100)
        out[f"regret_vs_oracle_{name}"] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
    return out


def run_same_distribution(cfg_name):
    cfg = CONFIGS[cfg_name]
    est, dev_records = train_estimator(cfg)
    cal_records = build_dataset(cfg, CAL_SEEDS)
    tau, cal_u = calibrate_tau(est, cal_records)
    val_records = build_dataset(cfg, VAL_SEEDS)
    val_utils, val_esc = policy_utilities(val_records, est, tau)
    val_summary = summarize_utils(val_utils, boot_seed_offset=10)

    test_records = build_dataset(cfg, TEST_SEEDS)
    test_utils, test_esc = policy_utilities(test_records, est, tau)
    test_summary = summarize_utils(test_utils, boot_seed_offset=20)

    return {
        "config": cfg_name,
        "tau": tau,
        "weights": est.weights.tolist(),
        "bias": est.bias,
        "feature_mean": est.mean_.tolist(),
        "feature_std": est.std_.tolist(),
        "validation": {**val_summary, "b3_escalation_rate": val_esc},
        "held_out_test": {**test_summary, "b3_escalation_rate": test_esc},
    }, est, tau


def run_distribution_shift(train_cfg_name, test_cfg_name, est, tau):
    test_cfg = CONFIGS[test_cfg_name]
    test_records = build_dataset(test_cfg, TEST_SEEDS)
    utils, esc = policy_utilities(test_records, est, tau)
    summary = summarize_utils(utils, boot_seed_offset=30)
    return {"trained_on": train_cfg_name, "evaluated_on": test_cfg_name,
            "b3_escalation_rate": esc, **summary}


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    estimators = {}
    for cfg_name in CONFIGS:
        result, est, tau = run_same_distribution(cfg_name)
        results[cfg_name] = result
        estimators[cfg_name] = (est, tau)

    shift_results = []
    shift_results.append(run_distribution_shift("small_dense", "large_sparse", *estimators["small_dense"]))
    shift_results.append(run_distribution_shift("large_sparse", "small_dense", *estimators["large_sparse"]))
    results["distribution_shift"] = shift_results

    with open(os.path.join(os.path.dirname(__file__), "summary_revised.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
