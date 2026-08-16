"""Offline oracle: NOT deployable online. Given counterfactual outcomes for
both backends on the same decision state, picks whichever backend actually
yielded higher utility. Establishes an empirical upper bound only.
"""
from __future__ import annotations

from src.reasoning.counterfactual import CounterfactualRecord, utility


def oracle_utility(record: CounterfactualRecord, lambda_latency: float = 0.15) -> tuple[float, bool]:
    u_det = utility(record.regret_det, record.latency_det, lambda_latency)
    u_reasoning = utility(record.regret_reasoning, record.latency_reasoning, lambda_latency)
    if u_reasoning > u_det:
        return u_reasoning, True
    return u_det, False
