import numpy as np, pandas as pd
from rhagent.evaluate_robust import (
    fold_sharpe, bootstrap_sharpe_ci, deflated_sharpe, _baseline_by_group,
)

def _net(mean=0.001, sd=0.01, n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(mean, sd, n), index=idx)

def test_bootstrap_ci_brackets_point_sharpe():
    net = _net()
    ann = float(net.mean() / net.std() * np.sqrt(252))
    lo, hi = bootstrap_sharpe_ci(net, n=500, seed=1)
    assert lo <= ann <= hi
    assert lo < hi

def test_fold_sharpe_returns_mean_and_std():
    net = _net()
    m, s = fold_sharpe(net, fold=60, step=30)
    assert np.isfinite(m) and s >= 0.0

def test_deflated_sharpe_in_unit_interval_and_penalizes_trials():
    net = _net(mean=0.002)
    sr = float(net.mean() / net.std() * np.sqrt(252))
    d_few = deflated_sharpe(sr, [sr, 0.0], net)
    d_many = deflated_sharpe(sr, [sr] + [0.0] * 50, net)
    assert 0.0 <= d_many <= d_few <= 1.0  # more trials => harder to clear

def test_baseline_by_group_is_per_engine_and_symbols():
    rows = [
        # group A: mean_reversion on NVDA,SPY -> baseline is the best 'none' (0.5)
        {"engine": "mean_reversion", "symbols": ["NVDA", "SPY"], "overlay": "none", "point_sharpe": 0.5},
        {"engine": "mean_reversion", "symbols": ["NVDA", "SPY"], "overlay": "none", "point_sharpe": 0.2},
        {"engine": "mean_reversion", "symbols": ["NVDA", "SPY"], "overlay": "bucket", "point_sharpe": 5.0},
        # group B: agent on AAPL -> a wildly high 'none' Sharpe must not leak into group A
        {"engine": "agent", "symbols": ["AAPL"], "overlay": "none", "point_sharpe": 9.0},
        # group C: no 'none' run at all -> no baseline entry
        {"engine": "agent", "symbols": ["AMD"], "overlay": "bucket", "point_sharpe": 1.0},
    ]
    baselines = _baseline_by_group(rows)
    assert baselines[("mean_reversion", ("NVDA", "SPY"))] == 0.5
    assert baselines[("agent", ("AAPL",))] == 9.0
    assert ("agent", ("AMD",)) not in baselines

def test_deflated_sharpe_lone_run_not_maximally_significant():
    net = _net(mean=0.002)
    sr = float(net.mean() / net.std() * np.sqrt(252))
    d = deflated_sharpe(sr, [sr], net)
    assert np.isfinite(d)
    assert d < 1.0
