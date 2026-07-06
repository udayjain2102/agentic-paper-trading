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
