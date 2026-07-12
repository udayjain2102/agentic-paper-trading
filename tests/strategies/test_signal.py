import numpy as np
import pandas as pd

from rhagent.strategies import build


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_mean_reversion_signal_high_after_drop():
    # 30 flat bars then a sharp drop -> z very negative -> signal (-z) high positive
    closes = [100.0] * 30 + [80.0]
    sig = build("mean_reversion", {}).signal(_bars(closes))
    assert sig.iloc[-1] > 0
    assert not np.isnan(sig.iloc[-1])


def test_momentum_signal_is_trailing_return():
    closes = [100.0 * (1.01 ** i) for i in range(60)]  # steady uptrend
    sig = build("momentum", {}).signal(_bars(closes))
    # trailing 40-bar return, positive in an uptrend
    assert sig.iloc[-1] > 0
    expected = closes[-1] / closes[-41] - 1.0
    assert abs(sig.iloc[-1] - expected) < 1e-9


def test_linreg_signal_matches_position_sign_where_in_position():
    rng = np.random.default_rng(0)
    closes = list(100 + np.cumsum(rng.normal(0, 1, 120)))
    strat = build("linreg", {})
    sig = strat.signal(_bars(closes))
    pos = strat.positions(_bars(closes))
    # where the strategy holds (pos != 0), the signal sign must match the position
    held = pos[pos != 0]
    assert len(held) > 0
    for t in held.index:
        assert np.sign(sig[t]) == pos[t]


def test_signal_nan_free_after_warmup():
    sig = build("mean_reversion", {}).signal(_bars([100.0 + i for i in range(40)]))
    assert not sig.iloc[30:].isna().any()


def test_signal_no_lookahead_recomputation_invariance():
    closes = [100.0 + (i % 7) - 3 for i in range(60)]
    strat = build("mean_reversion", {})
    full = strat.signal(_bars(closes))
    truncated = strat.signal(_bars(closes[:45]))
    # signal at day 44 is identical whether or not later bars exist
    assert abs(full.iloc[44] - truncated.iloc[44]) < 1e-12


def test_base_signal_not_implemented():
    from rhagent.strategies.base import Strategy
    import pytest
    with pytest.raises(NotImplementedError):
        Strategy().signal(_bars([1.0, 2.0]))
