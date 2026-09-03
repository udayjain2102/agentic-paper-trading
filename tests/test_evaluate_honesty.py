"""Guards against evaluate.py / evaluate_robust.py emitting confident numbers
on thin samples again."""

import json

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


def _write_thin_run(run_dir, *, engine="mean_reversion"):
    """0 trades, 6 return-days -- exactly the shape that made sharpe None
    and crashed a bare `:.2f}` format spec in production (GH Actions run
    30246319323)."""
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(json.dumps({
        "run_id": run_dir.name, "engine": engine, "symbols": ["A"],
        "start": "2026-07-01", "end": "2026-07-08", "notional": 10_000.0,
    }))
    run_dir.joinpath("trades.jsonl").write_text("")
    idx = pd.date_range("2026-07-01", periods=6, freq="D")
    pd.DataFrame({"date": idx, "net": [0.0] * 6}).to_csv(
        run_dir / "returns.csv", index=False
    )


def test_forward_report_does_not_crash_on_a_thin_sample(tmp_path):
    """The bug that took down the daily run: forward._report() f-string
    formatting a None sharpe with `:.2f`. Exercises the real render path
    against a record with 0 trades and 6 return-days (below both
    MIN_TRADES_FOR_RATE_STATS and MIN_RETURN_DAYS_FOR_SHARPE)."""
    from rhagent import forward

    run_dir = tmp_path / "forward" / "mean_reversion"
    _write_thin_run(run_dir)
    forward._report(run_dir)  # must not raise


def test_papertrade_print_report_does_not_crash_on_a_thin_sample(tmp_path, capsys):
    from rhagent import papertrade

    run_dir = tmp_path / "papertrade" / "run1"
    _write_thin_run(run_dir, engine="scripted")
    papertrade._print_report(run_dir)  # must not raise
    out = capsys.readouterr().out
    assert "n/a (needs" in out

