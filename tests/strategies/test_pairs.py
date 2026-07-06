import numpy as np
import pandas as pd

from rhagent.strategies.pairs import Pairs


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_long_a_when_a_cheap_relative_to_b():
    # A and B move together, then A dips -> spread negative -> long A leg.
    a = [100] * 20 + [90]
    b = [100] * 21
    pa, pb = Pairs(lookback=20, entry=1.0, allow_short=True).positions_pair(
        _bars(a), _bars(b)
    )
    assert pa.iloc[-1] == 1
    assert pb.iloc[-1] == -1


def test_long_only_clamps_short_leg_to_flat():
    a = [100] * 20 + [90]
    b = [100] * 21
    pa, pb = Pairs(lookback=20, entry=1.0, allow_short=False).positions_pair(
        _bars(a), _bars(b)
    )
    assert pa.iloc[-1] == 1
    assert pb.iloc[-1] == 0  # short leg clamped
