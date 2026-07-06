import numpy as np
import pandas as pd

from rhagent.strategies.mean_reversion import MeanReversion


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_goes_long_after_a_sharp_drop():
    # Flat then a big dip -> z drops below -entry -> long.
    prices = [100] * 20 + [90]
    s = MeanReversion(lookback=20, entry=1.0, exit=0.0)
    pos = s.positions(_bars(prices))
    assert pos.iloc[-1] == 1


def test_warmup_is_flat():
    prices = list(range(1, 10))  # fewer than lookback
    s = MeanReversion(lookback=20)
    pos = s.positions(_bars(prices))
    assert (pos == 0).all()


def test_no_lookahead_appending_future_bars_does_not_change_past():
    prices = [100] * 20 + [90]
    s = MeanReversion(lookback=20, entry=1.0, exit=0.0)
    short = s.positions(_bars(prices))
    long = s.positions(_bars(prices + [80, 120]))
    # positions for the original dates are unchanged by future bars.
    assert list(short.values) == list(long.iloc[: len(short)].values)
