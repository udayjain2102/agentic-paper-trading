"""Lookahead-free entry-time features, shared by the ledger writer and overlays."""

from __future__ import annotations

import pandas as pd


def entry_features(history: pd.DataFrame) -> dict:
    """Cheap lookahead-free scalars at entry, used for failure bucketing."""
    close = history["close"].astype(float)
    rets = close.pct_change().dropna()

    vol20 = float(rets.tail(20).std()) if len(rets) >= 2 else 0.0
    if pd.isna(vol20):
        vol20 = 0.0

    gap = 0.0
    if len(close) >= 2 and "open" in history:
        gap = float(history["open"].iloc[-1] / close.iloc[-2] - 1.0)

    trend5 = 0.0
    if len(close) >= 6:
        diff = float(close.iloc[-1] - close.iloc[-6])
        trend5 = 0.0 if diff == 0 else (1.0 if diff > 0 else -1.0)

    try:
        dow = float(history.index[-1].dayofweek)
    except (AttributeError, TypeError):
        dow = 0.0

    dist_high20 = 0.0
    dist_low20 = 0.0
    ret1 = 0.0
    if len(close) >= 2:
        last20 = close.tail(20)
        dist_high20 = float(close.iloc[-1] / last20.max() - 1.0)
        dist_low20 = float(close.iloc[-1] / last20.min() - 1.0)
        ret1 = float(close.iloc[-1] / close.iloc[-2] - 1.0)
    if pd.isna(dist_high20):
        dist_high20 = 0.0
    if pd.isna(dist_low20):
        dist_low20 = 0.0
    if pd.isna(ret1):
        ret1 = 0.0

    # Multi-horizon momentum (Kakushadze/GTJA momentum family), lookahead-free.
    ret5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0

    # Mean-reversion z-score of the last close vs its trailing 20-day window.
    zscore20 = 0.0
    if len(close) >= 20:
        w = close.tail(20)
        sd = float(w.std())
        if sd > 0:
            zscore20 = float((close.iloc[-1] - w.mean()) / sd)

    # RSI(14), simple-average variant. Neutral 50 default on short history.
    # ponytail: simple mean of gains/losses, not Wilder's EMA smoothing — fine
    # as a bucketing feature; swap to Wilder if it ever drives a live signal.
    rsi14 = 50.0
    if len(close) >= 15:
        d = close.diff().dropna()
        up = float(d.clip(lower=0.0).tail(14).mean())
        dn = float((-d.clip(upper=0.0)).tail(14).mean())
        if dn == 0:
            rsi14 = 100.0
        elif up == 0:
            rsi14 = 0.0
        else:
            rsi14 = 100.0 - 100.0 / (1.0 + up / dn)

    # Volume surge: last bar's volume vs trailing 20-day average. Neutral 1.0
    # default when volume is absent (e.g. synthetic fixtures) or history short.
    vol_ratio = 1.0
    if "volume" in history and len(history) >= 2:
        v = history["volume"].astype(float)
        avg = float(v.tail(20).mean())
        if avg > 0:
            vol_ratio = float(v.iloc[-1] / avg)

    return {
        "vol20": vol20, "gap": gap, "trend5": trend5,
        "dow": dow, "dist_high20": dist_high20, "dist_low20": dist_low20, "ret1": ret1,
        "ret5": ret5, "ret20": ret20, "zscore20": zscore20,
        "rsi14": rsi14, "vol_ratio": vol_ratio,
    }


def flatten_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Flatten a trades frame's nested `entry_features` dict column into
    `feat_*` columns, matching evaluate.load_run exactly. No-op (returns as-is)
    if empty, already flattened, or lacking an `entry_features` column."""
    if len(trades) == 0 or "entry_features" not in trades.columns:
        return trades
    trades = trades.copy()
    feats = pd.json_normalize(trades.pop("entry_features")).add_prefix("feat_")
    return pd.concat([trades.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
