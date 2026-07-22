"""Pure, side-effect-free safety checks.

This module is the safety core of the manual-invocation agent path
(runner.py / agent.py / executor.py) — it is not on the scheduled paper-run
path (scripts/paper_cron.sh), which never calls the LLM order-placement loop.
It performs no I/O, talks to no network, and holds no state of its own —
every input is passed in, every output is a decision. That makes it trivially
testable and impossible to surprise.

The agent never calls the broker's order API directly; every proposed order
flows through ``validate_order`` first (see ``executor.py``). The model cannot
talk its way past a hard cap because the cap is enforced here, in code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Optional, Tuple

# A plain US equity ticker: 1-5 uppercase letters. Rejects crypto pairs
# ("BTC-USD"), option symbols, and anything with digits or punctuation.
_EQUITY_SYMBOL = re.compile(r"^[A-Z]{1,5}$")

_SIDES = {"buy", "sell"}


@dataclass(frozen=True)
class Limits:
    per_trade_max_usd: float
    total_deployed_max_usd: float
    max_new_positions_per_run: int
    max_orders_per_run: int
    daily_loss_limit_usd: float


@dataclass(frozen=True)
class Account:
    """Live account snapshot, read fresh from the broker each run."""

    buying_power_usd: float
    total_position_value_usd: float
    positions: FrozenSet[str]
    realized_pnl_today_usd: float
    # Optional symbol -> current position value, for callers (e.g. strategy
    # mode) that need to liquidate the actual held amount rather than a fixed
    # notional. Additive field with a safe default so existing construction
    # sites are unaffected.
    position_values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    notional_usd: float


@dataclass
class RunState:
    """Mutable counters scoped to a single run (one cron tick)."""

    orders_placed: int = 0
    new_positions: int = 0


def check_halted(
    account: Account,
    limits: Limits,
    halt_file_present: bool,
) -> Tuple[bool, Optional[str]]:
    """Top-of-run gate. If this returns halted, the run does nothing."""
    if halt_file_present:
        return True, "HALT file present — trading halted by operator."
    if account.realized_pnl_today_usd <= -limits.daily_loss_limit_usd:
        return (
            True,
            f"Daily loss limit hit: realized P&L "
            f"${account.realized_pnl_today_usd:.2f} <= "
            f"-${limits.daily_loss_limit_usd:.2f}.",
        )
    return False, None


def validate_order(
    order: Order,
    account: Account,
    run_state: RunState,
    limits: Limits,
) -> Tuple[bool, Optional[str]]:
    """Return (ok, reason). ``ok`` is True only if every check passes.

    ``reason`` is a human-readable rejection message when ok is False, and None
    when ok is True. Pure function — does not mutate ``run_state``; the caller
    updates counters only after an order is actually accepted and acted on.
    """
    # 1. Input sanity.
    if order.side not in _SIDES:
        return False, f"Invalid side {order.side!r}; expected 'buy' or 'sell'."
    if not _EQUITY_SYMBOL.match(order.symbol or ""):
        return False, (
            f"Invalid symbol {order.symbol!r}; only US equity tickers "
            "(1-5 uppercase letters) are allowed."
        )
    if not order.notional_usd > 0:
        return False, f"Notional must be positive; got {order.notional_usd}."

    # 2. Per-trade dollar cap (applies to buys and sells alike).
    if order.notional_usd > limits.per_trade_max_usd:
        return False, (
            f"Order ${order.notional_usd:.2f} exceeds per-trade cap "
            f"${limits.per_trade_max_usd:.2f}."
        )

    # 3. Per-run order rate limit.
    if run_state.orders_placed >= limits.max_orders_per_run:
        return False, (
            f"Max orders per run reached "
            f"({limits.max_orders_per_run})."
        )

    # Buy-only exposure checks. Sells reduce risk, so they skip these.
    if order.side == "buy":
        if order.notional_usd > account.buying_power_usd:
            return False, (
                f"Insufficient buying power: order ${order.notional_usd:.2f} > "
                f"available ${account.buying_power_usd:.2f}."
            )
        if (
            account.total_position_value_usd + order.notional_usd
            > limits.total_deployed_max_usd
        ):
            return False, (
                f"Order would push deployed capital to "
                f"${account.total_position_value_usd + order.notional_usd:.2f}, "
                f"over cap ${limits.total_deployed_max_usd:.2f}."
            )
        is_new_position = order.symbol not in account.positions
        if (
            is_new_position
            and run_state.new_positions >= limits.max_new_positions_per_run
        ):
            return False, (
                f"Max new positions per run reached "
                f"({limits.max_new_positions_per_run})."
            )

    return True, None
