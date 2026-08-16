import numpy as np

from experiments.phase_m.run_large_n_resolution import (
    bootstrap_ci, pearson, spearman, run_episode_full_control,
)
from src.delegation.policy import DelegationPolicy
from src.delegation.null_policy import NullDelegationPolicy


def test_bootstrap_ci_contains_true_mean_for_low_variance_data():
    values = [0.5, 0.51, 0.49, 0.50, 0.52, 0.48]
    result = bootstrap_ci(values, n_boot=2000, seed=0)
    assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]
    assert result["n_seeds"] == 6


def test_bootstrap_ci_widens_with_more_variance():
    tight = bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5], n_boot=2000, seed=0)
    wide = bootstrap_ci([0.1, 0.9, 0.2, 0.8, 0.5], n_boot=2000, seed=0)
    assert (wide["ci_hi"] - wide["ci_lo"]) > (tight["ci_hi"] - tight["ci_lo"])


def test_pearson_perfect_correlation():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert abs(pearson(x, y) - 1.0) < 1e-9


def test_spearman_monotonic_nonlinear():
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]  # nonlinear but monotonic -> spearman 1.0, pearson < 1.0
    assert abs(spearman(x, y) - 1.0) < 1e-9
    assert pearson(x, y) < 1.0


def test_correlation_handles_constant_input():
    # pearson is well-defined as 0.0 for a constant series (guarded explicitly in
    # pearson()); spearman is computed as pearson-of-ranks and is not meaningfully
    # defined when all ranks tie (argsort of a constant array is an arbitrary but
    # stable permutation), so it is not asserted here.
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_null_model_run_produces_lower_concentration_than_real_policy():
    """Smoke test on a small fleet: the null-model policy (uniform random peer choice)
    should not systematically out-concentrate the real (capability/history-based) policy.
    Run on a handful of seeds to keep the test fast; the full 50-seed comparison with
    bootstrap CIs is the actual experiment (experiments/phase_m/results/)."""
    from experiments.phase_f.run_specialization_revised import analyze

    real_ginis = []
    null_ginis = []
    for seed in range(90000, 90003):
        real_policy = DelegationPolicy(lambda_latency=0.15)
        activity_real, _ = run_episode_full_control(seed, real_policy, track_early_luck=False)
        real_ginis.append(analyze(activity_real)["delegate_in_gini"])

        null_policy = NullDelegationPolicy(lambda_latency=0.15, null_rng_seed=seed + 1)
        activity_null, _ = run_episode_full_control(seed, null_policy, track_early_luck=False)
        null_ginis.append(analyze(activity_null)["delegate_in_gini"])

    assert np.mean(real_ginis) >= np.mean(null_ginis)


def test_run_episode_full_control_tracks_early_luck_events():
    policy = DelegationPolicy(lambda_latency=0.15)
    activity, delegate_events = run_episode_full_control(90010, policy, track_early_luck=True)
    assert delegate_events is not None
    total_events = sum(len(v) for v in delegate_events.values())
    total_delegate_in_from_activity = int(activity[:, :, 2].sum())
    assert total_events == total_delegate_in_from_activity
