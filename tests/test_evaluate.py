import json
from pathlib import Path

import pandas as pd
import pytest

from rhagent.evaluate import (
    _bucket_labels,
    aggregate,
    compare_runs,
    failure_buckets,
    load_run,
    spy_benchmark,
)


def _trade(tid, pnl_abs, pnl_pct, outcome, vol=0.01, gap=0.0, holding=3,
           symbol="A", side="long"):
    return {
        "trade_id": tid, "run_id": tid.split("#")[0], "symbol": symbol,
        "side": side, "entry_ts": "2026-01-02", "entry_price": 100.0,
        "entry_reason": "r", "exit_ts": "2026-01-05", "exit_price": 101.0,
        "exit_reason": "r", "qty": 1.0, "pnl_abs": pnl_abs, "pnl_pct": pnl_pct,
        "holding_bars": holding, "outcome": outcome,
        "entry_features": {"vol20": vol, "gap": gap, "trend5": 0.0},
    }


def _write_run(run_dir: Path, trades, nets, engine="scripted"):
    run_dir.mkdir(parents=True)
    rid = run_dir.name
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": rid, "engine": engine, "symbols": ["A"],
        "start": "2026-01-01", "end": "2026-01-10",
        "cost_bps": 1.0, "notional": 10000.0, "created_ts": "2026-07-11T00:00:00Z",
    }))
    with (run_dir / "trades.jsonl").open("w") as fh:
        for t in trades:
            fh.write(json.dumps(t) + "\n")
    idx = pd.date_range("2026-01-01", periods=len(nets), freq="D")
    pd.DataFrame({"date": idx, "net": nets}).to_csv(run_dir / "returns.csv", index=False)


@pytest.fixture
def run_dir(tmp_path):
    rid = "2026-07-11T00-00-00Z-aaaaaaaa"
    trades = [
        _trade(f"{rid}#0001", 200.0, 0.02, "win", vol=0.005, gap=0.01, holding=2),
        _trade(f"{rid}#0002", -100.0, -0.01, "loss", vol=0.02, gap=-0.01, holding=8),
        _trade(f"{rid}#0003", -300.0, -0.03, "loss", vol=0.03, gap=-0.02, holding=10,
               symbol="B", side="short"),
        _trade(f"{rid}#0004", 100.0, 0.01, "win", vol=0.01, gap=0.0, holding=1),
    ]
    d = tmp_path / rid
    _write_run(d, trades, [0.0, 0.01, -0.005, 0.02])
    return d


def test_load_run_expands_features(run_dir):
    meta, trades, net = load_run(run_dir)
    assert meta["engine"] == "scripted"
    assert len(trades) == 4
    assert {"feat_vol20", "feat_gap", "feat_trend5"} <= set(trades.columns)
    assert len(net) == 4


def test_aggregate_stats(run_dir):
    _, trades, net = load_run(run_dir)
    a = aggregate(trades, net)
    assert a["n_trades"] == 4
    assert abs(a["win_rate"] - 0.5) < 1e-12
    assert abs(a["avg_win"] - 150.0) < 1e-9
    assert abs(a["avg_loss"] - (-200.0)) < 1e-9
    assert abs(a["profit_factor"] - (300.0 / 400.0)) < 1e-12
    assert abs(a["avg_holding_bars"] - 5.25) < 1e-12
    # return metrics come from backtest.result_from_returns on net
    assert a["total_return"] == pytest.approx((1.01 * 0.995 * 1.02) - 1)


def test_failure_buckets_loss_share(run_dir):
    _, trades, _ = load_run(run_dir)
    b = failure_buckets(trades)
    assert list(b.columns) == ["dimension", "bucket", "n_trades", "win_rate",
                               "loss_share"]
    sym = b[b.dimension == "symbol"].set_index("bucket")
    # total loss 400: A lost 100 (25%), B lost 300 (75%)
    assert abs(sym.loc["B", "loss_share"] - 0.75) < 1e-12
    assert abs(sym.loc["A", "loss_share"] - 0.25) < 1e-12
    side = b[b.dimension == "side"].set_index("bucket")
    assert abs(side.loc["short", "loss_share"] - 0.75) < 1e-12
    # sorted by loss_share descending
    assert b["loss_share"].is_monotonic_decreasing


def test_failure_buckets_no_losses_is_all_zero_share(tmp_path):
    rid = "2026-07-11T00-00-00Z-bbbbbbbb"
    trades = [_trade(f"{rid}#0001", 100.0, 0.01, "win")]
    d = tmp_path / rid
    _write_run(d, trades, [0.01])
    _, tdf, _ = load_run(d)
    b = failure_buckets(tdf)
    assert (b["loss_share"] == 0.0).all()


def test_compare_runs(tmp_path, run_dir):
    # run_dir fixture lives in tmp_path; add a second run
    rid2 = "2026-07-12T00-00-00Z-cccccccc"
    _write_run(tmp_path / rid2, [_trade(f"{rid2}#0001", 50.0, 0.005, "win")],
               [0.005], engine="mean_reversion")
    df = compare_runs(tmp_path)
    assert list(df["run_id"]) == [run_dir.name, rid2]
    assert list(df["engine"]) == ["scripted", "mean_reversion"]
    assert {"n_trades", "win_rate", "profit_factor", "total_return",
            "sharpe", "max_drawdown"} <= set(df.columns)


def test_compare_runs_pnl_comes_from_return_curve_not_trade_sum(tmp_path):
    rid = "2026-07-12T00-00-00Z-dddddddd"
    # Trade ledger sums to +$10,000, but the account return curve is +1%.
    # The dashboard/comparison should report the return-derived account P&L.
    _write_run(tmp_path / rid, [_trade(f"{rid}#0001", 10000.0, 1.0, "win")],
               [0.01], engine="mean_reversion")
    df = compare_runs(tmp_path)
    assert df.loc[0, "net_pnl"] == pytest.approx(100.0)


def test_bucket_labels_dow_and_near_high(run_dir):
    _, trades, _ = load_run(run_dir)
    trades["feat_dow"] = [0.0, 4.0, 6.0, 2.0]  # Mon, Fri, out-of-range->dropped, Wed
    trades["feat_dist_high20"] = [-0.001, -0.02, -0.2, -0.004]  # at_high, mid, far_below, at_high
    labels = _bucket_labels(trades)
    assert list(labels["dow"]) == ["Mon", "Fri", None, "Wed"]
    assert list(labels["near_high"]) == ["at_high", "mid", "far_below", "at_high"]


def test_bucket_labels_nan_features_drop_out(run_dir):
    # Trades concatenated from a pre-feature ledger carry NaN in the new
    # feat_ columns; they must drop out of those dimensions, not crash.
    _, trades, _ = load_run(run_dir)
    trades["feat_dow"] = [0.0, float("nan"), 4.0, float("nan")]
    trades["feat_dist_high20"] = [float("nan"), -0.02, float("nan"), -0.001]
    labels = _bucket_labels(trades)
    assert list(labels["dow"]) == ["Mon", None, "Fri", None]
    nh = list(labels["near_high"])
    assert [nh[1], nh[3]] == ["mid", "at_high"]
    assert pd.isna(nh[0]) and pd.isna(nh[2])
    # groupby drops the NaN/None rows from the dimension entirely
    grouped = trades.groupby(labels["dow"]).size()
    assert grouped.sum() == 2


def test_bucket_labels_skips_dow_and_near_high_when_missing(run_dir):
    _, trades, _ = load_run(run_dir)
    labels = _bucket_labels(trades)
    assert "dow" not in labels
    assert "near_high" not in labels


def test_load_run_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope")


def test_spy_benchmark_matches_strategy_window(tmp_path):
    # SPY cache spans more days than the strategy window; the benchmark must
    # be computed only over the strategy's own date range, not the full cache.
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    closes = [100.0 + i for i in range(10)]
    pd.DataFrame({"date": dates, "close": closes}).to_csv(tmp_path / "SPY.csv", index=False)

    strat_dates = dates[2:6]  # 2026-01-03 .. 2026-01-06, a strict subset
    out = spy_benchmark(strat_dates, cache_dir=tmp_path)

    assert out["start"] == str(strat_dates.min().date())
    assert out["end"] == str(strat_dates.max().date())
    assert out["return"] == pytest.approx(105.0 / 102.0 - 1.0)


def test_spy_benchmark_never_extends_past_available_cache(tmp_path):
    # Off-by-window failure mode: if the SPY cache lags the strategy's
    # window, the benchmark's reported end date must trail with it rather
    # than silently comparing against a shorter (or wrong) SPY window.
    cache_dates = pd.date_range("2026-01-01", periods=5, freq="D")
    pd.DataFrame({"date": cache_dates, "close": [100.0, 101.0, 102.0, 103.0, 104.0]}).to_csv(
        tmp_path / "SPY.csv", index=False
    )

    strat_dates = pd.date_range("2026-01-01", periods=8, freq="D")  # 3 days beyond the cache
    out = spy_benchmark(strat_dates, cache_dir=tmp_path)

    assert out["end"] == str(cache_dates.max().date())
    assert pd.Timestamp(out["end"]) <= strat_dates.max()
