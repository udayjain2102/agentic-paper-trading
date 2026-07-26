"""Guards against evaluate.py / evaluate_robust.py emitting confident numbers
on thin samples again."""

import pandas as pd
import pytest

from rhagent.evaluate import aggregate
from rhagent.evaluate_robust import bootstrap_sharpe_ci, deflated_sharpe, fold_sharpe


def test_zero_trades_returns_none_not_fake_zero():
    trades = pd.DataFrame(columns=["pnl_abs", "outcome", "holding_bars"])
    net = pd.Series([0.0, 0.01], index=pd.date_range("2026-01-01", periods=2))
    a = aggregate(trades, net)
    assert a["n_trades"] == 0
    for key in ("win_rate", "avg_win", "avg_loss", "profit_factor", "avg_holding_bars"):
        assert a[key] is None, f"{key} should be None at n_trades=0, got {a[key]!r}"


def test_six_return_days_sharpe_is_not_a_bare_number():
    # the exact shape from the audit: five 0.0% days, one +0.0516% day ->
    # naive annualized sharpe comes out to +6.48. That must not survive.
    net = pd.Series(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.000516],
        index=pd.date_range("2026-01-01", periods=6),
    )
    trades = pd.DataFrame(columns=["pnl_abs", "outcome", "holding_bars"])
    a = aggregate(trades, net)
    assert a["n_return_days"] == 6
    assert a["sharpe"] is None

    lo, hi = bootstrap_sharpe_ci(net)
    assert lo is None and hi is None
    fm, fs = fold_sharpe(net)
    assert fm is None and fs is None


def test_deflated_sharpe_none_below_three_observations():
    net = pd.Series([0.01, 0.02], index=pd.date_range("2026-01-01", periods=2))
    assert deflated_sharpe(1.0, [1.0], net) is None
