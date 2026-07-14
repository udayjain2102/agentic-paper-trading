"""The forward tick must only record fully-realized universe days.

net_returns records a day's return at its entry date, so a day is trustworthy
only once the next trading bar exists for every leg. _net_series must drop any
trailing (or internal) day where some name is not yet realized, otherwise the
tick bakes in a thin partial-day mean.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward


def _cfg(universe):
    return SimpleNamespace(strategy=SimpleNamespace(
        name="mean_reversion", params={}, universe=universe, overlay="none"))


def test_partial_final_day_is_dropped():
    # AAA runs one bar longer than BBB. That extra bar lets AAA realize a return
    # for a date BBB cannot -> the basket is only partially realized there, so
    # _net_series must exclude it.
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    aaa = pd.DataFrame({"open": close, "close": close})
    bbb = aaa.iloc[:-1].copy()  # BBB is missing the final bar

    net = forward._net_series(_cfg(["AAA", "BBB"]), "mean_reversion",
                              {"AAA": aaa, "BBB": bbb}, 1.0, Path("/tmp"))

    # BBB's last realizable entry is one day earlier than AAA's; the last date
    # where only AAA is realized must not appear.
    from rhagent.backtest import net_returns
    from rhagent.strategies import build
    strat = build("mean_reversion", {})
    aaa_last = net_returns(aaa, strat.positions(aaa), 1.0).index.max()
    bbb_last = net_returns(bbb, strat.positions(bbb), 1.0).index.max()
    assert aaa_last > bbb_last                     # AAA really does realize later
    assert net.index.max() == bbb_last             # basket stops at full coverage
    assert aaa_last not in net.index               # the partial day is gone


def test_full_coverage_unchanged():
    # When every leg is realized on every date, no day is dropped.
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    rng = np.random.default_rng(1)
    def frame(seed):
        c = pd.Series(100 * np.exp(np.cumsum(
            np.random.default_rng(seed).normal(0, 0.01, len(idx)))), index=idx)
        return pd.DataFrame({"open": c, "close": c})
    bars = {"AAA": frame(1), "BBB": frame(2)}

    net = forward._net_series(_cfg(["AAA", "BBB"]), "mean_reversion", bars, 1.0, Path("/tmp"))
    from rhagent.backtest import net_returns
    from rhagent.strategies import build
    strat = build("mean_reversion", {})
    # both legs realize the same span -> that whole span survives
    span = net_returns(bars["AAA"], strat.positions(bars["AAA"]), 1.0).index
    assert net.index.equals(span)
