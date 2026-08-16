"""Evaluation metrics and a manual percentile bootstrap CI (no scipy)."""
from __future__ import annotations

import numpy as np


def task_completion_rate(records: list[dict]) -> float:
    if not records:
        return float("nan")
    failed = sum(1 for r in records if r["failed"])
    return 1.0 - failed / len(records)


def mean_cost(records: list[dict]) -> float:
    return float(np.mean([r["true_cost"] for r in records])) if records else float("nan")


def mean_regret(records: list[dict]) -> float:
    return float(np.mean([r["regret"] for r in records])) if records else float("nan")


def mean_latency(records: list[dict]) -> float:
    return float(np.mean([r["latency"] for r in records])) if records else float("nan")


def reasoning_call_rate(records: list[dict]) -> float:
    if not records:
        return float("nan")
    return sum(1 for r in records if r["used_reasoning"]) / len(records)


def bootstrap_ci(values: list[float], stat_fn=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Manual percentile bootstrap CI. Returns (point_estimate, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(stat_fn(arr))
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boots[i] = stat_fn(sample)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def summarize(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "completion_rate": task_completion_rate(records),
        "mean_cost": mean_cost(records),
        "mean_regret": mean_regret(records),
        "mean_latency": mean_latency(records),
        "reasoning_call_rate": reasoning_call_rate(records),
    }
