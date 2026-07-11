import pandas as pd

from rhagent.engine import Decision, StrategyEngine
from rhagent.strategies.base import Strategy


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


class AlwaysLong(Strategy):
    name = "always_long"

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=bars.index, dtype=int)


def test_decision_is_frozen():
    d = Decision(target=1.0, reason="x")
    assert d.target == 1.0 and d.reason == "x"


def test_strategy_engine_takes_last_position_as_target():
    eng = StrategyEngine(AlwaysLong())
    hist = _bars([100.0, 101.0, 102.0])
    d = eng.decide("AAPL", hist, current_pos=0.0)
    assert d.target == 1.0


def test_strategy_engine_reason_names_strategy_and_close():
    eng = StrategyEngine(AlwaysLong())
    hist = _bars([100.0, 250.5])
    d = eng.decide("AAPL", hist, current_pos=0.0)
    assert "always_long" in d.reason
    assert "250.50" in d.reason


def test_strategy_engine_exposes_strategy_name():
    assert StrategyEngine(AlwaysLong()).name == "always_long"


def test_strategy_engine_no_lookahead_only_sees_history():
    # Target on a 2-bar history must equal the strategy's value at that bar,
    # regardless of what later bars would have said.
    class LastCloseSign(Strategy):
        name = "last_close_sign"

        def positions(self, bars: pd.DataFrame) -> pd.Series:
            sign = 1 if bars["close"].iloc[-1] >= 100 else 0
            return pd.Series(sign, index=bars.index, dtype=int)

    eng = StrategyEngine(LastCloseSign())
    assert eng.decide("A", _bars([100.0, 99.0]), 0.0).target == 0.0
    assert eng.decide("A", _bars([100.0, 99.0, 101.0]), 0.0).target == 1.0
