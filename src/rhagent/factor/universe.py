"""The evaluation universe and its aligned price panels.

A fixed list of liquid large-cap individual stocks (ETFs excluded — an ETF is
the market itself, not a cross-sectional member). load_universe fetches daily
bars cache-first via data.get_bars and returns both the per-symbol OHLCV frames
(for signal computation) and a [dates x symbols] close panel inner-joined to the
common trading calendar (for IC / forward returns).
"""

from __future__ import annotations

import pandas as pd

from ..data import get_bars

UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "CRM", "ADBE", "NFLX", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK",
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "GILD", "BMY", "MDT", "HD", "LOW", "MCD", "SBUX", "NKE", "COST", "WMT",
    "PG", "KO", "PEP", "PM", "DIS", "CMCSA", "VZ", "T", "XOM", "CVX",
    "CAT", "BA", "GE", "HON", "UNP", "LIN",
]


def load_universe(
    symbols, start, end, cache_dir="data", min_bars: int = 60
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    bars_by_symbol = get_bars(symbols, start, end, cache_dir=cache_dir)
    bars_by_symbol = {s: b for s, b in bars_by_symbol.items() if len(b) >= min_bars}
    if not bars_by_symbol:
        raise ValueError(f"no symbols with >= {min_bars} min_bars in the universe")
    close = pd.DataFrame(
        {s: b["close"].astype(float) for s, b in bars_by_symbol.items()}
    ).dropna(how="any")  # inner-join to the common calendar
    return bars_by_symbol, close
