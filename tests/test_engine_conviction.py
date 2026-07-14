import numpy as np, pandas as pd
from rhagent.engine import StrategyEngine, Decision
from rhagent.strategies import build

def _bars(n=60):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    return pd.DataFrame({"open": close, "close": close}, index=idx)

def test_decision_has_conviction_field_default_none():
    d = Decision(target=1.0, reason="x")
    assert d.conviction is None

def test_strategy_engine_sets_conviction_from_signal():
    eng = StrategyEngine(build("mean_reversion", {}))
    bars = _bars()
    d = eng.decide("NVDA", bars, 0.0)
    strat = build("mean_reversion", {})
    expected = float(strat.signal(bars).iloc[-1])
    assert (d.conviction == expected) or (np.isnan(d.conviction) and np.isnan(expected))
