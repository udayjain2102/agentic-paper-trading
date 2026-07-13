import numpy as np, pandas as pd
from rhagent.overlay import Overlay
from rhagent.papertrade import PaperTrader, HistoricalSource
from rhagent.engine import StrategyEngine, Decision
from rhagent.strategies import build

class _SpyOverlay:
    """Records the max exit_ts it is ever shown, per bar timestamp seen."""
    name = "spy"
    def __init__(self):
        self.violations = []
    def adjust(self, symbol, history, decision, closed_trades):
        today = history.index[-1]
        if len(closed_trades):
            exits = pd.to_datetime(closed_trades["exit_ts"])
            if (exits >= today).any():
                self.violations.append((symbol, str(today)))
        return decision.target

def test_overlay_never_sees_future_or_same_day_close(tmp_path):
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    close = pd.Series(100 + np.sin(np.arange(120) / 3.0) * 5, index=idx)
    df = pd.DataFrame({"open": close, "close": close}, index=idx)

    class _Src:
        def bars(self): return {"NVDA": df}

    spy = _SpyOverlay()
    trader = PaperTrader(engine=StrategyEngine(build("mean_reversion", {})),
                         source=_Src(), out_dir=str(tmp_path), overlay=spy)
    trader.run()
    assert spy.violations == [], f"overlay saw non-past closes: {spy.violations[:5]}"


def test_overlay_never_sees_same_bar_close_across_symbols(tmp_path):
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    symbols = ["AAA", "BBB", "CCC"]

    def _mk(phase):
        close = pd.Series(100 + np.sin((np.arange(120) + phase) / 3.0) * 5, index=idx)
        return pd.DataFrame({"open": close, "close": close}, index=idx)

    frames = {sym: _mk(i * 3) for i, sym in enumerate(symbols)}

    class _Src:
        def bars(self): return frames

    spy = _SpyOverlay()
    trader = PaperTrader(engine=StrategyEngine(build("mean_reversion", {})),
                         source=_Src(), out_dir=str(tmp_path), overlay=spy)
    trader.run()
    assert spy.violations == [], f"overlay saw non-past closes: {spy.violations[:5]}"
