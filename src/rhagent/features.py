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

    return {"vol20": vol20, "gap": gap, "trend5": trend5}


def flatten_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Flatten a trades frame's nested `entry_features` dict column into
    `feat_*` columns, matching evaluate.load_run exactly. No-op (returns as-is)
    if empty, already flattened, or lacking an `entry_features` column."""
    if len(trades) == 0 or "entry_features" not in trades.columns:
        return trades
    trades = trades.copy()
    feats = pd.json_normalize(trades.pop("entry_features")).add_prefix("feat_")
    return pd.concat([trades.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
