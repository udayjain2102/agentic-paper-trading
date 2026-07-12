"""Cross-sectional Information Coefficient math.

Pure functions over panels ([dates x symbols] DataFrames). rank_ic_one is the
Spearman rank correlation between a day's signal cross-section and its forward
returns; because it ranks, it is invariant to a common additive shift applied to
every name's return that day (it removes the equal-weighted common cross-sectional
mean — no separate demeaning step is needed for that). This is NOT the same as
being market-neutral: rank-IC does not remove differential beta exposure, so a
signal that merely proxies market beta (e.g. loads more on high-beta names) can
still earn a positive rank-IC. Real beta-neutralization (residualizing against
factor/market beta) is deferred to a later sub-project. ICIR is the consistency
of the daily IC series, and the decay curve reports how IC fades across forward
horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    return close.shift(-h) / close - 1.0


def rank_ic_one(sig_row: pd.Series, ret_row: pd.Series, min_names: int = 10) -> float:
    df = pd.DataFrame({"s": sig_row, "r": ret_row}).dropna()
    if len(df) < min_names:
        return float("nan")
    sr = df["s"].rank()
    rr = df["r"].rank()
    if sr.std(ddof=0) == 0 or rr.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(sr.to_numpy(), rr.to_numpy())[0, 1])


def ic_series(
    signal_panel: pd.DataFrame, close_panel: pd.DataFrame, h: int, min_names: int = 10
) -> pd.Series:
    fwd = forward_returns(close_panel, h)
    rows: dict = {}
    for t in signal_panel.index.intersection(fwd.index):
        ic = rank_ic_one(signal_panel.loc[t], fwd.loc[t], min_names)
        if not np.isnan(ic):
            rows[t] = ic
    return pd.Series(rows, dtype=float)


def icir(ic: pd.Series) -> float:
    ic = ic.dropna()
    if len(ic) == 0:
        return 0.0
    sd = ic.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(ic.mean() / sd)


def ic_decay(
    signal_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    horizons=(1, 5, 10, 20, 50),
    min_names: int = 10,
) -> dict:
    out: dict = {}
    for h in horizons:
        s = ic_series(signal_panel, close_panel, h, min_names)
        out[h] = float(s.mean()) if len(s) else float("nan")
    return out


def half_life(decay: dict):
    horizons = sorted(decay)
    if not horizons:
        return None
    base = decay[horizons[0]]
    if base is None or np.isnan(base) or base == 0:
        return None
    target = abs(base) / 2.0
    for h in horizons:
        v = decay[h]
        if not np.isnan(v) and abs(v) <= target:
            return h
    return f">{horizons[-1]}"
