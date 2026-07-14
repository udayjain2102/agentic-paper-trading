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

    _COLS = ["bias", "ret_lag1", "ret_lag2", "ma_ratio"]

    def _features(self, bars: pd.DataFrame):
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
        return close, feats, target

    def _fit_predict(self, feats: pd.DataFrame, target: pd.Series, i: int) -> float:
        """OLS on rows with realized targets (strictly before i); predict row i.
        Returns NaN when there is too little training data or features are NaN."""
        train = feats.iloc[:i].copy()
        train["y"] = target.iloc[:i]
        train = train.dropna()
        x_now = feats.iloc[i][self._COLS]
        if len(train) < self.min_train or x_now.isna().any():
            return float("nan")
        beta, *_ = np.linalg.lstsq(
            train[self._COLS].to_numpy(), train["y"].to_numpy(), rcond=None
        )
        return float(x_now.to_numpy() @ beta)

    def target(self, bars: pd.DataFrame) -> float:
        """Single-step: fit and predict only the last row (O(n) not O(n^2))."""
        close, feats, tgt = self._features(bars)
        if len(close) == 0:
            return 0.0
        pred = self._fit_predict(feats, tgt, len(close) - 1)
        pos = 0 if pd.isna(pred) else int(np.sign(pred))
        return float(clamp_short(pd.Series([pos]), self.allow_short).iloc[0])

    def _predictions(self, bars: pd.DataFrame) -> pd.Series:
        close, feats, target = self._features(bars)
        pred = pd.Series(np.nan, index=close.index, dtype=float)
        for i in range(len(close)):
            pred.iloc[i] = self._fit_predict(feats, target, i)
        return pred

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        pred = self._predictions(bars)
        pos = np.sign(pred).fillna(0).astype(int)
        return clamp_short(pos, self.allow_short)

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        return self._predictions(bars)
