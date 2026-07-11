"""Event-driven paper-trading harness.

Steps a DecisionEngine through bars one day at a time, turns position changes
into discrete ID-stamped trades, and writes an append-only ledger under
journal/papertrade/{run_id}/. Two seams keep it world-model-ready: bars come
from a MarketSource and orders are priced by a FillModel — swap either without
touching the loop. The vectorized backtest.py is untouched and remains the
fast ranking path.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd

from .data import get_bars


class MarketSource(Protocol):
    def bars(self) -> dict[str, pd.DataFrame]: ...


class FillModel(Protocol):
    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float: ...


class HistoricalSource:
    """Real cached history via data.get_bars (offline once cached)."""

    def __init__(self, symbols, start: str, end: str, cache_dir="data") -> None:
        self.symbols = list(symbols)
        self.start, self.end, self.cache_dir = start, end, cache_dir

    def bars(self) -> dict[str, pd.DataFrame]:
        return get_bars(self.symbols, self.start, self.end, cache_dir=self.cache_dir)


class CloseFill:
    """Perfect fill at the bar's close. cost_bps is charged by the loop."""

    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float:
        return float(bar["close"])


def new_run_id(now: datetime | None = None, suffix: str | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    suffix = suffix or secrets.token_hex(4)
    return f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}-{suffix}"


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
