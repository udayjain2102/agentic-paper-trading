"""Assemble a [dates x symbols] signal panel from a strategy."""

from __future__ import annotations

import pandas as pd

from ..strategies.base import Strategy


def signal_panel(
    strat: Strategy, bars_by_symbol: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.DataFrame:
    cols = {s: strat.signal(bars).reindex(index) for s, bars in bars_by_symbol.items()}
    return pd.DataFrame(cols, index=index)
