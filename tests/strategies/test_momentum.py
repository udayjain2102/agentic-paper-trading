import pandas as pd

from rhagent.strategies.momentum import Momentum


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_long_when_trailing_return_positive():
    prices = [100 + i for i in range(50)]  # steady uptrend
    pos = Momentum(lookback=40).positions(_bars(prices))
    assert pos.iloc[-1] == 1


def test_flat_when_trailing_return_negative():
    prices = [100 - i for i in range(50)]  # steady downtrend
    pos = Momentum(lookback=40).positions(_bars(prices))
    assert pos.iloc[-1] == 0  # long-only clamps the short signal to flat


def test_warmup_is_flat():
    prices = list(range(1, 10))
    pos = Momentum(lookback=40).positions(_bars(prices))
    assert (pos == 0).all()
