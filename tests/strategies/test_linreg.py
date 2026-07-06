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


def _noisy_mean_reverting_prices(n=120, seed=0):
    # Mean-reverting series (negative autocorrelation in returns) so that the
    # rolling OLS prediction sign genuinely flips over time, unlike a smooth
    # compounding uptrend where the sign stays positive regardless of how the
    # betas are corrupted. This is what gives the no-lookahead prefix check
    # its teeth: a future-leaking fit would shift the fitted betas and change
    # some of the earlier predicted signs.
    rng = np.random.default_rng(seed)
    prev_ret = 0.0
    close = [100.0]
    for _ in range(n):
        r = rng.normal(0, 0.01) - 0.5 * prev_ret
        close.append(close[-1] * (1 + r))
        prev_ret = r
    return close[1:]


def test_no_lookahead_appending_future_bars_does_not_change_past():
    prices = _noisy_mean_reverting_prices(n=120, seed=0)
    s = LinReg(min_train=40)
    short = s.positions(_bars(prices))
    longer = s.positions(_bars(prices + [95.0, 130.0, 80.0]))

    # Sanity: the fixture must actually produce varying signals, otherwise
    # the prefix-equality assertion below would be trivially satisfied even
    # by an implementation that leaks the future (e.g. fits OLS once on the
    # entire series) as long as predicted signs never change.
    assert short.nunique() > 1

    assert list(short.values) == list(longer.iloc[: len(short)].values)
