import numpy as np, pandas as pd
from rhagent.evaluate_robust import fold_sharpe, bootstrap_sharpe_ci, deflated_sharpe

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
