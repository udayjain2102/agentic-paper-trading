import numpy as np
import pandas as pd

from rhagent.strategies.linreg import LinReg


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_predicts_up_in_a_persistent_uptrend():
    # Compounding uptrend: positive-return autocorrelation -> predicts long.
    prices = [100 * (1.01 ** i) for i in range(80)]
    pos = LinReg(min_train=40).positions(_bars(prices))
    assert pos.iloc[-1] == 1


def test_warmup_is_flat():
    prices = [100 * (1.01 ** i) for i in range(30)]  # below min_train
    pos = LinReg(min_train=40).positions(_bars(prices))
    assert (pos == 0).all()


def test_no_lookahead_appending_future_bars_does_not_change_past():
    prices = [100 * (1.01 ** i) for i in range(80)]
    s = LinReg(min_train=40)
    short = s.positions(_bars(prices))
    longer = s.positions(_bars(prices + [200, 150, 300]))
    assert list(short.values) == list(longer.iloc[: len(short)].values)
