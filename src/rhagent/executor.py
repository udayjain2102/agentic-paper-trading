"""The single funnel every order passes through.

The agent's ``place_order`` tool calls ``OrderExecutor.execute`` — it never
touches the broker directly. This class:

  1. validates the order against the guardrails (in code),
  2. in dry-run, logs the intended order and places nothing,
  3. in live mode, places the order via the broker and records the fill,
  4. updates the per-run counters and journals every outcome.

Because this is the only path to the broker, the guardrails cannot be bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .broker import Broker
from .guardrails import Account, Limits, Order, RunState, validate_order
from .journal import Journal


@dataclass
class ExecuteResult:
    accepted: bool
    placed: bool
    reason: str | None = None
    broker_order_id: str | None = None

    def as_tool_text(self) -> str:
        if not self.accepted:
            return f"REJECTED: {self.reason}"
        if not self.placed:
            return "ACCEPTED (dry-run): order logged, nothing placed."
        return f"PLACED: broker_order_id={self.broker_order_id}"


class OrderExecutor:
    def __init__(
        self,
        *,
        broker: Broker,
        account: Account,
        limits: Limits,
        run_state: RunState,
        journal: Journal,
        dry_run: bool,
    ) -> None:
        self._broker = broker
        self._account = account
        self._limits = limits
        self._run = run_state
        self._journal = journal
        self._dry_run = dry_run

    def execute(self, symbol: str, side: str, notional_usd: float) -> ExecuteResult:
        order = Order(symbol=symbol, side=side, notional_usd=float(notional_usd))

        ok, reason = validate_order(order, self._account, self._run, self._limits)
        if not ok:
            self._journal.record(
                "order_rejected",
                **_order_fields(order),
                reason=reason,
            )
            return ExecuteResult(accepted=False, placed=False, reason=reason)

        if self._dry_run:
            self._journal.record("order_intended", **_order_fields(order))
            self._advance_counters(order)
            return ExecuteResult(accepted=True, placed=False)

        fill = self._broker.place_order(order)
        self._journal.record(
            "order_placed",
            **_order_fields(order),
            status=fill.status,
            broker_order_id=fill.broker_order_id,
        )
        self._advance_counters(order)
        return ExecuteResult(
            accepted=True, placed=True, broker_order_id=fill.broker_order_id
        )

    def _advance_counters(self, order: Order) -> None:
        self._run.orders_placed += 1
        if order.side == "buy" and order.symbol not in self._account.positions:
            self._run.new_positions += 1


def _order_fields(order: Order) -> Dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "notional_usd": order.notional_usd,
    }
