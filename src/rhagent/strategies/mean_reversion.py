"""Z-score mean reversion: buy statistically-cheap dips, exit on reversion.

z = (close - rolling_mean) / rolling_std over ``lookback`` days. Enter long when
z < -entry; exit to flat when z >= -exit. Hysteresis (entry != exit) avoids
churning around the threshold. Long-only unless allow_short.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, clamp_short


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, lookback=20, entry=1.0, exit=0.0, allow_short=False):
        self.lookback = lookback
        self.entry = entry
        self.exit = exit
        self.allow_short = allow_short

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        z = (close - mean) / std

        pos = pd.Series(0, index=close.index, dtype=int)
        holding = 0  # +1 long, -1 short, 0 flat
        for t in close.index:
            zt = z[t]
            if pd.isna(zt):
                pos[t] = 0
                continue
            if holding == 0:
                if zt < -self.entry:
                    holding = 1
                elif zt > self.entry:
                    holding = -1
            elif holding == 1 and zt >= -self.exit:
                holding = 0
            elif holding == -1 and zt <= self.exit:
                holding = 0
            pos[t] = holding
        return clamp_short(pos, self.allow_short)
