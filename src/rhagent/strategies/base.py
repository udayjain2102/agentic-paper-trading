"""The common contract every strategy implements.

A strategy maps a DataFrame of daily bars to a target-position series with
values in {-1, 0, +1}, obeying the no-lookahead invariant (the position at day
t uses only data up to and including day t). The backtest engine applies that
position to the day t -> t+1 return.
"""

from __future__ import annotations

import pandas as pd


def clamp_short(pos: pd.Series, allow_short: bool) -> pd.Series:
    """Long-only guard: map short signals (-1) to flat (0) unless shorting is on."""
    if allow_short:
        return pos
    return pos.clip(lower=0)


class Strategy:
    name: str = "base"

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def target(self, bars: pd.DataFrame) -> float:
        """Today's target position (last row only). Default recomputes the whole
        series; subclasses whose last value is independent of the earlier ones
        can override with a cheaper single-step computation."""
        return float(self.positions(bars).iloc[-1])

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Continuous score aligned to bars.index; higher = more bullish on the
        forward return. No lookahead: the value at day t uses only bars up to t.
        Subclasses that support IC evaluation override this."""
        raise NotImplementedError
