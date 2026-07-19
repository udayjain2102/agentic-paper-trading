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


def test_stop_exits_and_blocks_reentry_until_z_resets():
    # Dip -> long at 90, then keep falling: with a 5% stop we exit and stay
    # flat (z still < -entry, so re-entry is latched off).
    prices = [100] * 20 + [90, 84, 80, 79]
    s = MeanReversion(lookback=20, entry=1.0, exit=0.0, stop=0.05)
    pos = s.positions(_bars(prices))
    assert pos.iloc[20] == 1  # entered on the dip
    assert (pos.iloc[21:] == 0).all()  # stopped out at 84, no re-entry

    # Without a stop the same path stays long the whole way down.
    no_stop = MeanReversion(lookback=20, entry=1.0, exit=0.0)
    assert (no_stop.positions(_bars(prices)).iloc[20:] == 1).all()


def test_stop_default_off_matches_previous_behavior():
    prices = list(100 + 10 * np.sin(np.arange(60) / 3))
    base = MeanReversion(lookback=20, entry=1.0, exit=0.0)
    explicit_off = MeanReversion(lookback=20, entry=1.0, exit=0.0, stop=None)
    assert list(base.positions(_bars(prices))) == list(
        explicit_off.positions(_bars(prices))
    )
