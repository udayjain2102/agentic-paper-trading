import pandas as pd

from rhagent.strategies.momentum import Momentum
from rhagent.strategy_runner import target_orders


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_buys_when_signal_long_and_not_held():
    bars = {"AAPL": _bars([100 + i for i in range(50)])}  # uptrend -> long
    orders = target_orders(Momentum(lookback=40), bars, held=set(), notional_usd=250)
    assert orders == [("AAPL", "buy", 250)]


def test_sells_when_signal_flat_and_held():
    bars = {"AAPL": _bars([100 - i for i in range(50)])}  # downtrend -> flat
    orders = target_orders(Momentum(lookback=40), bars, held={"AAPL"}, notional_usd=250)
    assert orders == [("AAPL", "sell", 250)]


def test_no_order_when_already_in_desired_state():
    bars = {"AAPL": _bars([100 + i for i in range(50)])}  # long, already held
    orders = target_orders(Momentum(lookback=40), bars, held={"AAPL"}, notional_usd=250)
    assert orders == []
