"""Run every strategy over the universe, rank by total return, pick the winner.

    python -m rhagent.compare

The three single-symbol strategies are evaluated per symbol and equal-weighted
into one net-return series each; pairs is evaluated on the most-correlated pair.
Ranking is by total return; Sharpe, max drawdown, and hit-rate are shown for
context. The top row is the winner, printed with a ready-to-paste config block.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

from .backtest import BacktestResult, net_returns, result_from_returns
from .data import get_bars
from .strategies import REGISTRY, build
from .strategies.pairs import Pairs

UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY"]


def best_pair(bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[str, str]:
    closes = pd.DataFrame(
        {s: b["close"] for s, b in bars_by_symbol.items()}
    ).dropna()
    corr = closes.pct_change().dropna().corr()
    best, best_val = None, -2.0
    syms = list(corr.columns)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = corr.iloc[i, j]
            if c > best_val:
                best_val, best = c, (syms[i], syms[j])
    return best


def _aggregate(nets: list[pd.Series]) -> BacktestResult:
    combined = pd.concat(nets, axis=1).mean(axis=1).dropna()
    return result_from_returns(combined)


def evaluate(
    bars_by_symbol: dict[str, pd.DataFrame], cost_bps: float = 1.0
) -> list[tuple[str, BacktestResult]]:
    rows: list[tuple[str, BacktestResult]] = []

    for name in REGISTRY:
        strat = build(name, {})
        nets = [
            net_returns(bars, strat.positions(bars), cost_bps)
            for bars in bars_by_symbol.values()
        ]
        rows.append((name, _aggregate(nets)))

    a, b = best_pair(bars_by_symbol)
    pa, pb = Pairs().positions_pair(bars_by_symbol[a], bars_by_symbol[b])
    pair_nets = [
        net_returns(bars_by_symbol[a], pa, cost_bps),
        net_returns(bars_by_symbol[b], pb, cost_bps),
    ]
    rows.append(("pairs", _aggregate(pair_nets)))

    rows.sort(key=lambda r: r[1].total_return, reverse=True)
    return rows


def main() -> int:
    end = date.today()
    start = end - timedelta(days=400)
    bars = get_bars(UNIVERSE, start.isoformat(), end.isoformat())

    rows = evaluate(bars)
    print(f"{'strategy':<16}{'total_ret':>12}{'sharpe':>10}{'max_dd':>10}{'hit':>8}")
    for name, res in rows:
        print(
            f"{name:<16}{res.total_return:>11.2%}{res.sharpe:>10.2f}"
            f"{res.max_drawdown:>10.2%}{res.hit_rate:>8.2%}"
        )

    winner, wres = rows[0]
    print(f"\nWinner (by total return): {winner} ({wres.total_return:.2%})")
    if winner == "pairs":
        a, b = best_pair(bars)
        print(f"Chosen pair: {a}/{b}. Long-only trades only the cheap leg.")
        print("Add this to config.yaml:\n")
        print("strategy:")
        print("  name: pairs")
        print("  params: {}")
        print(f"  universe: [{a}, {b}]")
    else:
        print("Add this to config.yaml:\n")
        print("strategy:")
        print(f"  name: {winner}")
        print("  params: {}")
        print(f"  universe: {UNIVERSE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
