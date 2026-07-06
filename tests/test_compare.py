import numpy as np
import pandas as pd

from rhagent.compare import best_pair, evaluate


def _bars(prices, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def _universe():
    rng = np.random.default_rng(0)
    up = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    down = 100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 120))
    return {
        "AAPL": _bars(up),
        "MSFT": _bars(up * 1.01),  # highly correlated with AAPL
        "NVDA": _bars(down),
        "SPY": _bars(up * 0.5 + 50),
    }


def test_evaluate_returns_one_row_per_strategy_sorted_by_return():
    rows = evaluate(_universe(), cost_bps=1.0)
    names = [name for name, _ in rows]
    assert set(names) == {"mean_reversion", "momentum", "linreg", "pairs"}
    returns = [res.total_return for _, res in rows]
    assert returns == sorted(returns, reverse=True)  # descending


def test_best_pair_picks_the_two_most_correlated():
    pair = best_pair(_universe())
    assert set(pair) == {"AAPL", "MSFT"}
