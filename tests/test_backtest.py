import numpy as np
import pandas as pd

from rhagent.backtest import net_returns, result_from_returns, run


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_always_long_earns_the_buy_and_hold_return():
    prices = [100, 110, 121]  # +10% then +10%
    bars = _bars(prices)
    pos = pd.Series([1, 1, 1], index=bars.index)
    res = run(bars, pos, cost_bps=0.0)
    # Position on the last day is dropped (no forward return): two +10% steps.
    assert res.total_return == pytest_approx(0.21)
    assert res.n_days == 2


def test_flat_earns_nothing():
    bars = _bars([100, 110, 121])
    pos = pd.Series([0, 0, 0], index=bars.index)
    res = run(bars, pos, cost_bps=0.0)
    assert res.total_return == 0.0


def test_costs_reduce_return():
    bars = _bars([100, 110, 121])
    pos = pd.Series([1, 1, 1], index=bars.index)
    gross = run(bars, pos, cost_bps=0.0).total_return
    netted = run(bars, pos, cost_bps=50.0).total_return
    assert netted < gross


def pytest_approx(x):
    import pytest

    return pytest.approx(x, rel=1e-6)
