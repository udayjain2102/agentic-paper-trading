import pandas as pd

from rhagent.strategies.momentum import Momentum
from rhagent.strategies.pairs import Pairs
from rhagent.strategy_runner import pairs_target_orders, target_orders


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


def test_pairs_buys_cheap_leg_and_leaves_other_flat():
    # A dips relative to B -> long-only signal longs A, clamps B's short to flat.
    a = [100] * 20 + [90]
    b = [100] * 21
    orders = pairs_target_orders(
        Pairs(lookback=20, entry=1.0),
        _bars(a),
        _bars(b),
        "A",
        "B",
        held=set(),
        notional_usd=250,
    )
    assert orders == [("A", "buy", 250)]


def test_pairs_sell_liquidates_held_value_when_signal_flat():
    # A was held long from a prior cheap signal; now flat -> sell at held value.
    a = [100] * 21
    b = [100] * 21
    orders = pairs_target_orders(
        Pairs(lookback=20, entry=1.0),
        _bars(a),
        _bars(b),
        "A",
        "B",
        held={"A"},
        notional_usd=250,
        held_values={"A": 175.0},
    )
    assert orders == [("A", "sell", 175.0)]
