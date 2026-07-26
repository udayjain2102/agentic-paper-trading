import pandas as pd

from rhagent.strategies.momentum import Momentum
from rhagent.strategy_runner import orders_from_positions, target_orders


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


def test_sell_liquidates_actual_held_value_not_fixed_notional():
    bars = {"AAPL": _bars([100 - i for i in range(50)])}  # downtrend -> flat
    orders = target_orders(
        Momentum(lookback=40),
        bars,
        held={"AAPL"},
        notional_usd=250,
        held_values={"AAPL": 180.0},
    )
    assert orders == [("AAPL", "sell", 180.0)]





# --- orders_from_positions: same diff, from a precomputed position series
# (the shape forward._positions produces) instead of a strategy + bars -----

def test_positions_buys_when_target_long_and_not_held():
    pos = {"AAPL": pd.Series([0.0, 1.0])}
    orders = orders_from_positions(pos, held=set(), notional_usd=250)
    assert orders == [("AAPL", "buy", 250)]


def test_positions_sells_liquidating_held_value_when_target_flips_flat():
    pos = {"AAPL": pd.Series([1.0, 0.0])}
    orders = orders_from_positions(
        pos, held={"AAPL"}, notional_usd=250, held_values={"AAPL": 180.0}
    )
    assert orders == [("AAPL", "sell", 180.0)]


def test_positions_no_order_when_already_in_desired_state():
    pos = {"AAPL": pd.Series([1.0, 1.0])}
    orders = orders_from_positions(pos, held={"AAPL"}, notional_usd=250)
    assert orders == []
