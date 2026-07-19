"""Z-score mean reversion: buy statistically-cheap dips, exit on reversion.

z = (close - rolling_mean) / rolling_std over ``lookback`` days. Enter long when
z < -entry; exit to flat when z >= -exit. Hysteresis (entry != exit) avoids
churning around the threshold. Long-only unless allow_short.

Optional hard stop (``stop``, fraction, e.g. 0.04 = 4% adverse move vs entry
price) caps the loss on any one trade; after a stop-out, re-entry is blocked
until z recovers inside [-entry, entry]. Default OFF: on the 400-day/65-symbol
cache every stop level tested (3-20%, and time stops of 5-15 bars) reduced
total return, profit factor, and Sharpe vs no stop — mean reversion's losers
mostly revert, so stops realize losses at max pain. avg_loss > avg_win is
intrinsic here and paid for by the ~70% win rate.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, clamp_short


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, lookback=20, entry=1.0, exit=0.0, allow_short=False, stop=None):
        self.lookback = lookback
        self.entry = entry
        self.exit = exit
        self.allow_short = allow_short
        self.stop = stop  # max adverse move vs entry price; 0/None disables

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        z = (close - mean) / std

        pos = pd.Series(0, index=close.index, dtype=int)
        holding = 0  # +1 long, -1 short, 0 flat
        entry_px = 0.0
        stopped = False  # stop-out latch: no re-entry until z resets past -entry
        for t in close.index:
            zt = z[t]
            if pd.isna(zt):
                pos[t] = 0
                continue
            px = close[t]
            if holding == 0:
                if stopped and abs(zt) <= self.entry:
                    stopped = False
                if not stopped:
                    if zt < -self.entry:
                        holding = 1
                        entry_px = px
                    elif zt > self.entry:
                        holding = -1
                        entry_px = px
            elif self.stop and holding * (px / entry_px - 1.0) <= -self.stop:
                holding = 0
                stopped = True
            elif holding == 1 and zt >= -self.exit:
                holding = 0
            elif holding == -1 and zt <= self.exit:
                holding = 0
            pos[t] = holding
        return clamp_short(pos, self.allow_short)

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        z = (close - mean) / std
        return -z  # cheap dips (z << 0) score high
