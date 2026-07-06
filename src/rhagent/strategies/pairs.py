"""Pairs trading: trade the mean-reverting spread between two correlated names.

spread = log(close_a) - log(close_b), z-scored over ``lookback``. When A is rich
vs B (z > entry): short A / long B. When A is cheap (z < -entry): long A / short
B. Long-only clamps each short leg to flat (so only the cheap leg trades).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import clamp_short


class Pairs:
    name = "pairs"

    def __init__(self, lookback=20, entry=1.0, allow_short=False):
        self.lookback = lookback
        self.entry = entry
        self.allow_short = allow_short

    def positions_pair(self, bars_a, bars_b):
        a = bars_a["close"].astype(float)
        b = bars_b["close"].astype(float)
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]

        spread = np.log(a) - np.log(b)
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        z = (spread - mean) / std

        pos_a = pd.Series(0, index=idx, dtype=int)
        for t in idx:
            zt = z[t]
            if pd.isna(zt):
                continue
            if zt < -self.entry:
                pos_a[t] = 1
            elif zt > self.entry:
                pos_a[t] = -1
        pos_b = -pos_a
        return (
            clamp_short(pos_a, self.allow_short),
            clamp_short(pos_b, self.allow_short),
        )
