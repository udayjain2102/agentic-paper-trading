"""Linear-regression signal: predict next-day return via rolling OLS.

Features known at day t: [1, ret_lag1, ret_lag2, ma_ratio]. Target: next-day
return. At each day t we fit OLS on rows whose target is already realized
(strictly before t) and predict day t's next-day return. Long when the
prediction is positive. The expanding train window uses only past data, so
there is no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, clamp_short


class LinReg(Strategy):
    name = "linreg"

    def __init__(self, min_train=40, allow_short=False):
        self.min_train = min_train
        self.allow_short = allow_short

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        ret = close.pct_change()
        feats = pd.DataFrame(
            {
                "bias": 1.0,
                "ret_lag1": ret,
                "ret_lag2": ret.shift(1),
                "ma_ratio": close / close.rolling(10).mean() - 1.0,
            }
        )
        target = ret.shift(-1)  # next-day return, realized at the following day

        cols = ["bias", "ret_lag1", "ret_lag2", "ma_ratio"]
        pos = pd.Series(0, index=close.index, dtype=int)
        n = len(close)
        for i in range(n):
            # Rows usable for training at decision day i: target realized, i.e.
            # index j with j <= i-1 and all feature/target values present.
            train = feats.iloc[:i].copy()
            train["y"] = target.iloc[:i]
            train = train.dropna()
            x_now = feats.iloc[i][cols]
            if len(train) < self.min_train or x_now.isna().any():
                continue
            X = train[cols].to_numpy()
            y = train["y"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = float(x_now.to_numpy() @ beta)
            pos.iloc[i] = int(np.sign(pred))
        return clamp_short(pos, self.allow_short)
