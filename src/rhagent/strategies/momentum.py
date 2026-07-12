"""Trend following: long when the trailing ``lookback``-day return is positive.

Signal is +1 (up-trend), -1 (down-trend), 0 (warmup). Long-only clamps -1 to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, clamp_short


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, lookback=40, allow_short=False):
        self.lookback = lookback
        self.allow_short = allow_short

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        trailing = close.pct_change(self.lookback)
        pos = pd.Series(np.sign(trailing), index=close.index)
        pos = pos.fillna(0).astype(int)
        return clamp_short(pos, self.allow_short)

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        return close.pct_change(self.lookback)
