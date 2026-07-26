"""Pure order generation for strategy mode.

Given a strategy and recent bars per symbol, compute each symbol's latest target
position and diff it against what's currently held to produce buy/sell orders.
No I/O: the runner feeds the returned tuples through OrderExecutor so the same
guardrails apply as in the LLM path.

On an exit (target flips from held to flat), the sell liquidates the actual
held position value (via ``held_values``) rather than the fixed per-trade
notional, so a smaller-than-notional holding isn't over-sold into a short.
"""

from __future__ import annotations

import pandas as pd


def _order_for(
    symbol: str,
    target: int,
    is_held: bool,
    notional_usd: float,
    held_values: dict[str, float] | None = None,
) -> tuple[str, str, float] | None:
    if target == 1 and not is_held:
        return (symbol, "buy", notional_usd)
    if target == 0 and is_held:
        sell_notional = (held_values or {}).get(symbol, notional_usd)
        return (symbol, "sell", sell_notional)
    return None


def target_orders(
    strategy,
    bars_by_symbol: dict[str, pd.DataFrame],
    held: set[str],
    notional_usd: float,
    held_values: dict[str, float] | None = None,
) -> list[tuple[str, str, float]]:
    orders: list[tuple[str, str, float]] = []
    for symbol, bars in bars_by_symbol.items():
        pos = strategy.positions(bars)
        if len(pos) == 0:
            continue
        order = _order_for(symbol, int(pos.iloc[-1]), symbol in held, notional_usd, held_values)
        if order is not None:
            orders.append(order)
    return orders


def orders_from_positions(
    pos_by_symbol: dict[str, pd.Series],
    held: set[str],
    notional_usd: float,
    held_values: dict[str, float] | None = None,
) -> list[tuple[str, str, float]]:
    """Same diff as ``target_orders``, but from an already-computed target
    position series per symbol (e.g. ``forward._positions``'s output) instead
    of a strategy object -- so the daily paper run trades exactly the series
    the forward record itself is built from, overlay included.
    """
    orders: list[tuple[str, str, float]] = []
    for symbol, pos in pos_by_symbol.items():
        if len(pos) == 0:
            continue
        order = _order_for(symbol, int(pos.iloc[-1]), symbol in held, notional_usd, held_values)
        if order is not None:
            orders.append(order)
    return orders
