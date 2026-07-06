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
        target = int(pos.iloc[-1])
        is_held = symbol in held
        if target == 1 and not is_held:
            orders.append((symbol, "buy", notional_usd))
        elif target == 0 and is_held:
            sell_notional = (held_values or {}).get(symbol, notional_usd)
            orders.append((symbol, "sell", sell_notional))
    return orders
