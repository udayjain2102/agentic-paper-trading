import math

import pytest

from rhagent.gate.stats import bonferroni, deflated_sharpe, norm_cdf, norm_ppf


def test_norm_cdf_known_values():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(norm_cdf(1.959964) - 0.975) < 1e-4
    assert abs(norm_cdf(-1.959964) - 0.025) < 1e-4


def test_norm_ppf_known_values_and_roundtrip():
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-4
    assert abs(norm_ppf(0.5)) < 1e-9
    assert abs(norm_ppf(norm_cdf(0.7)) - 0.7) < 1e-6


def test_norm_ppf_rejects_boundaries():
    with pytest.raises(ValueError):
        norm_ppf(0.0)
    with pytest.raises(ValueError):
        norm_ppf(1.0)


def test_bonferroni_strong_signal_passes():
    p, thr, passed = bonferroni(0.5, 100, 10, alpha=0.05)
    assert thr == 0.05 / 10
    assert p < thr
    assert passed is True


def test_bonferroni_weak_signal_fails():
    p, thr, passed = bonferroni(0.05, 30, 50, alpha=0.05)
    assert p > thr
    assert passed is False


def test_deflated_sharpe_strong_few_trials_near_one():
    dsr = deflated_sharpe(0.5, 100, 0.0, 0.0, 3, 0.01)
    assert dsr > 0.95


def test_deflated_sharpe_weak_many_trials_near_zero():
    dsr = deflated_sharpe(0.05, 30, 0.0, 0.0, 100, 0.04)
    assert dsr < 0.5


def test_deflated_sharpe_zero_variance_uses_zero_benchmark():
    # var_trials=0 -> sr0=0 -> just PSR vs 0; a solid SR should score high
    assert deflated_sharpe(0.3, 50, 0.0, 0.0, 100, 0.0) > 0.9


def test_deflated_sharpe_degenerate_returns_zero():
    assert deflated_sharpe(0.5, 1, 0.0, 0.0, 10, 0.01) == 0.0    # n_eff <= 1
    # denom <= 0 (extreme skew) -> 0.0
    assert deflated_sharpe(0.5, 50, 100.0, 0.0, 10, 0.01) == 0.0
