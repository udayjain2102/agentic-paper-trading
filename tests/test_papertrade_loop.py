# tests/test_papertrade_loop.py
import json

import pandas as pd
import pytest

from rhagent.engine import Decision
from rhagent.papertrade import PaperTrader


def _bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


class FakeSource:
    def __init__(self, frames):  # dict[str, DataFrame]
        self._frames = frames

    def bars(self):
        return self._frames


class ScriptedEngine:
    """Emits a fixed target sequence per symbol — fully deterministic."""

    name = "scripted"

    def __init__(self, script):  # dict[str, list[float]]
        self.script = script

    def decide(self, symbol, history, current_pos):
        t = len(history) - 1
        target = float(self.script[symbol][t])
        return Decision(target=target, reason=f"scripted[{t}]={target:+.0f}")


def _run(script, closes, tmp_path, cost_bps=0.0):
    trader = PaperTrader(
        engine=ScriptedEngine(script),
        source=FakeSource({s: _bars(closes[s]) for s in closes}),
        cost_bps=cost_bps,
        notional=10_000.0,
        out_dir=tmp_path,
        run_id="2026-07-11T00-00-00Z-deadbeef",
    )
    run_dir = trader.run()
    trades = [json.loads(l) for l in (run_dir / "trades.jsonl").read_text().splitlines()]
    return run_dir, trades


def test_open_then_close_produces_one_trade(tmp_path):
    # flat, long, long, flat  on closes 100,110,121,133.1
    _, trades = _run({"A": [0, 1, 1, 0]}, {"A": [100.0, 110.0, 121.0, 133.1]}, tmp_path)
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "long"
    assert t["entry_price"] == 110.0 and t["exit_price"] == 133.1
    assert t["holding_bars"] == 2
    assert abs(t["pnl_pct"] - (133.1 / 110.0 - 1)) < 1e-12
    assert t["outcome"] == "win"


def test_flip_splits_into_two_trades(tmp_path):
    _, trades = _run({"A": [1, 1, -1, -1]}, {"A": [100.0, 110.0, 120.0, 90.0]}, tmp_path)
    assert len(trades) == 2
    first, second = trades
    assert first["side"] == "long" and first["exit_price"] == 120.0
    assert second["side"] == "short" and second["entry_price"] == 120.0
    # short from 120 to 90: +25%
    assert abs(second["pnl_pct"] - 0.25) < 1e-12


def test_end_of_data_force_closes(tmp_path):
    _, trades = _run({"A": [0, 1, 1]}, {"A": [100.0, 100.0, 105.0]}, tmp_path)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "end_of_data"
    assert trades[0]["exit_price"] == 105.0


def test_trade_ids_unique_monotonic_and_parse_to_run(tmp_path):
    _, trades = _run(
        {"A": [1, 0, 1, 0]}, {"A": [100.0, 101.0, 102.0, 103.0]}, tmp_path
    )
    ids = [t["trade_id"] for t in trades]
    assert ids == ["2026-07-11T00-00-00Z-deadbeef#0001",
                   "2026-07-11T00-00-00Z-deadbeef#0002"]
    assert all(t["run_id"] == "2026-07-11T00-00-00Z-deadbeef" for t in trades)


def test_cost_bps_charged_on_round_trip(tmp_path):
    _, trades = _run({"A": [0, 1, 0]}, {"A": [100.0, 100.0, 100.0]}, tmp_path,
                     cost_bps=10.0)
    t = trades[0]
    # flat prices, 10 bps each way -> -20 bps
    assert abs(t["pnl_pct"] - (-0.002)) < 1e-12
    assert t["outcome"] == "loss"
    assert abs(t["pnl_abs"] - (-20.0)) < 1e-9


def test_run_json_and_returns_csv_written(tmp_path):
    run_dir, _ = _run({"A": [0, 1, 0]}, {"A": [100.0, 110.0, 121.0]}, tmp_path)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["run_id"] == "2026-07-11T00-00-00Z-deadbeef"
    assert meta["engine"] == "scripted"
    assert meta["symbols"] == ["A"]
    rets = pd.read_csv(run_dir / "returns.csv", parse_dates=["date"])
    assert list(rets.columns) == ["date", "net"]
    assert len(rets) == 3
    # held +1 from day1 close 110 to day2 close 121 -> 10% on day2
    assert abs(rets["net"].iloc[2] - 0.10) < 1e-12


def test_two_symbols_equal_weight_returns(tmp_path):
    run_dir, trades = _run(
        {"A": [1, 1, 1], "B": [0, 0, 0]},
        {"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 50.0]},
        tmp_path,
    )
    rets = pd.read_csv(run_dir / "returns.csv")
    # A earns 10% on day1; B flat -> equal-weight 5%
    assert abs(rets["net"].iloc[1] - 0.05) < 1e-12
    # A's trade should book its 1/N notional slice, not the full $10,000,
    # else gross win/loss (sum of pnl_abs) drift away from net portfolio P&L.
    a_trade = next(t for t in trades if t["symbol"] == "A")
    assert abs(a_trade["pnl_abs"] - 1050.0) < 1e-9  # (10000/2) * 21% (100->121)


def test_entry_features_present_on_trades(tmp_path):
    _, trades = _run({"A": [0, 1, 0]}, {"A": [100.0, 110.0, 121.0]}, tmp_path)
    assert set(trades[0]["entry_features"]) == {"vol20", "gap", "trend5", "dow", "dist_high20", "dist_low20", "ret1", "ret5", "ret20", "zscore20", "rsi14", "vol_ratio"}


def test_empty_symbols_raises():
    with pytest.raises(ValueError, match="no symbols"):
        PaperTrader(engine=ScriptedEngine({}), source=FakeSource({})).run()


def test_diverging_symbol_indices_raise():
    frames = {"A": _bars([100.0, 101.0, 102.0, 103.0]), "B": _bars([50.0, 51.0, 52.0])}
    script = {"A": [0, 1, 1, 0], "B": [0, 1, 0]}
    with pytest.raises(ValueError, match="indices differ"):
        PaperTrader(engine=ScriptedEngine(script), source=FakeSource(frames)).run()


def test_determinism_same_inputs_same_ledger(tmp_path):
    script = {"A": [0, 1, 1, 0, -1, -1]}
    closes = {"A": [100.0, 101.0, 99.0, 102.0, 103.0, 101.0]}
    _, t1 = _run(script, closes, tmp_path / "r1")
    _, t2 = _run(script, closes, tmp_path / "r2")
    assert t1 == t2
