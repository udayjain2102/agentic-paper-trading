"""Broker abstraction over the Robinhood trading MCP.

This is the ONLY module that touches the broker. Everything else works against
the ``Broker`` protocol, which keeps the rest of the system testable (inject a
``MockBroker``) and keeps all network/auth concerns in one place.

The live ``McpBroker`` talks to the Robinhood MCP over its streamable-HTTP
transport. Because that endpoint requires OAuth and is not wired up until you
authenticate it, the live path is intentionally thin and the tool/field names
are marked as the integration points to confirm against the live server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from .guardrails import Account, Order


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    notional_usd: float
    status: str
    broker_order_id: str


class Broker(Protocol):
    def get_account(self) -> Account: ...
    def get_quote(self, symbol: str) -> float: ...
    def place_order(self, order: Order) -> Fill: ...


class MockBroker:
    """In-memory broker for tests and dry-run development.

    Seeded with a starting account; records any order passed to ``place_order``
    so tests can assert on what would have been sent.
    """

    def __init__(
        self,
        *,
        buying_power_usd: float = 10_000,
        total_position_value_usd: float = 0,
        positions: Dict[str, float] | None = None,
        realized_pnl_today_usd: float = 0,
        quotes: Dict[str, float] | None = None,
    ) -> None:
        self._buying_power = buying_power_usd
        self._deployed = total_position_value_usd
        self._positions = dict(positions or {})
        self._realized_pnl = realized_pnl_today_usd
        self._quotes = dict(quotes or {})
        self.placed: List[Order] = []

    def get_account(self) -> Account:
        return Account(
            buying_power_usd=self._buying_power,
            total_position_value_usd=self._deployed,
            positions=frozenset(self._positions),
            realized_pnl_today_usd=self._realized_pnl,
            position_values=dict(self._positions),
        )

    def get_quote(self, symbol: str) -> float:
        return self._quotes.get(symbol, 0.0)

    def place_order(self, order: Order) -> Fill:
        self.placed.append(order)
        return Fill(
            symbol=order.symbol,
            side=order.side,
            notional_usd=order.notional_usd,
            status="filled",
            broker_order_id=f"mock-{len(self.placed)}",
        )


class McpBroker:
    """Live broker backed by the Robinhood trading MCP.

    Constructed with an already-connected MCP client session. We keep the MCP
    wiring (transport, OAuth) out of this class so it stays a thin, mockable
    adapter — see ``mcp_session`` for how the session is created.

    The tool names below (``get_account`` etc.) are placeholders for the actual
    tools the Robinhood MCP exposes; confirm them against the live server's
    ``list_tools`` once authenticated, then adjust the mapping here only.
    """

    def __init__(self, session) -> None:  # session: mcp.ClientSession
        self._session = session

    def _call(self, tool: str, **arguments) -> dict:
        import anyio

        result = anyio.from_thread.run(
            self._session.call_tool, tool, arguments
        )
        return _structured(result)

    def get_account(self) -> Account:
        data = self._call("get_account")
        positions = data.get("positions", {})
        if isinstance(positions, dict):
            position_values = {k: float(v) for k, v in positions.items()}
        else:
            position_values = {}
        return Account(
            buying_power_usd=float(data["buying_power_usd"]),
            total_position_value_usd=float(data["total_position_value_usd"]),
            positions=frozenset(positions),
            realized_pnl_today_usd=float(data.get("realized_pnl_today_usd", 0)),
            position_values=position_values,
        )

    def get_quote(self, symbol: str) -> float:
        data = self._call("get_quote", symbol=symbol)
        return float(data["price"])

    def place_order(self, order: Order) -> Fill:
        data = self._call(
            "place_order",
            symbol=order.symbol,
            side=order.side,
            notional_usd=order.notional_usd,
        )
        return Fill(
            symbol=order.symbol,
            side=order.side,
            notional_usd=order.notional_usd,
            status=str(data.get("status", "submitted")),
            broker_order_id=str(data.get("order_id", "")),
        )


def _structured(call_tool_result) -> dict:
    """Pull the structured/JSON payload out of an MCP CallToolResult."""
    structured = getattr(call_tool_result, "structuredContent", None)
    if structured:
        return structured
    # Fall back to the first text content block parsed as JSON.
    import json

    for block in getattr(call_tool_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    return {}
