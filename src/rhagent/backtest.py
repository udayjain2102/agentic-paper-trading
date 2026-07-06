"""Offline backtest engine.

Turns a target-position series into a net-return series and summary metrics.
Mechanics: the position held on day t earns the return from day t to t+1, so the
final day (which has no forward return) is dropped. A per-trade cost in basis
points is charged on turnover (absolute change in position).

This module does no I/O and knows nothing about strategies — it just scores a
positions series against prices. Ranking uses ``total_return``; the other
metrics are reported for context only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ANNUALIZATION = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    total_return: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    n_days: int


def net_returns(bars: pd.DataFrame, positions: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    close = bars["close"].astype(float)
    fwd = close.pct_change().shift(-1)  # return from t to t+1, indexed at t
    pos = positions.reindex(close.index).fillna(0).astype(float)

    turnover = pos.diff().abs()
    if len(pos):
        turnover.iloc[0] = abs(pos.iloc[0])
    cost = turnover * (cost_bps / 1e4)

    net = pos * fwd - cost
    return net[fwd.notna()].fillna(0.0)


def result_from_returns(net: pd.Series) -> BacktestResult:
    if len(net) == 0:
        empty = pd.Series(dtype=float)
        return BacktestResult(empty, 0.0, 0.0, 0.0, 0.0, 0)

    equity = (1.0 + net).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    std = net.std()
    sharpe = (
        float(net.mean() / std * np.sqrt(_ANNUALIZATION))
        if std and not pd.isna(std) and std > 0
        else 0.0
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    active = net[net != 0]
    hit_rate = float((active > 0).mean()) if len(active) else 0.0

    return BacktestResult(
        equity=equity,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        n_days=int(len(net)),
    )


def run(bars: pd.DataFrame, positions: pd.Series, cost_bps: float = 1.0) -> BacktestResult:
    return result_from_returns(net_returns(bars, positions, cost_bps))
