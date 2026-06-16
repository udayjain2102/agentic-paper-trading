"""The decision agent: Claude reasons over the portfolio and proposes trades.

We use a manual agentic loop (not the auto tool-runner) on purpose: it gives us
a code-controlled choke point where the ``place_order`` tool is dispatched to
``OrderExecutor`` — which enforces the guardrails before anything reaches the
broker. The model decides; the code decides what's allowed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import anthropic

from .broker import Broker
from .config import AgentConfig
from .executor import OrderExecutor

SYSTEM_PROMPT = """\
You are an autonomous trading agent for a single US-equities brokerage account.

Each run, you review the account and market data, then decide whether to place
any trades. You may only trade US equities (stocks/ETFs).

Hard rules enforced by the system (you cannot override them):
- Every order is validated against position-size, total-deployment, and
  rate-limit caps. Orders that violate a cap are rejected and not placed.
- The system may be in DRY-RUN mode, in which case your orders are logged but
  not actually placed. Behave identically regardless.

Process for this run:
1. Call get_account to see buying power, deployed capital, positions, and P&L.
2. Use get_quote for any symbols you're considering.
3. Decide. It is completely acceptable to place no trades.
4. For each trade, call place_order with symbol, side ('buy'/'sell'), and a
   dollar notional. Read the tool result: 'REJECTED' means the order did not go
   through — do not retry the same order.

Be conservative. Explain your reasoning briefly before acting.
"""

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_account",
        "description": "Get buying power, total deployed capital, current "
        "positions, and realized P&L for today.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_quote",
        "description": "Get the latest price for a US equity symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "US equity ticker, e.g. AAPL"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "place_order",
        "description": "Propose an order. It is validated against all safety "
        "limits before being placed (or logged, in dry-run). Returns whether it "
        "was REJECTED, ACCEPTED (dry-run), or PLACED.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "US equity ticker"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "notional_usd": {
                    "type": "number",
                    "description": "Dollar amount to trade",
                },
            },
            "required": ["symbol", "side", "notional_usd"],
        },
    },
]


def _dispatch(name: str, args: Dict[str, Any], broker: Broker, executor: OrderExecutor) -> str:
    if name == "get_account":
        a = broker.get_account()
        return (
            f"buying_power_usd={a.buying_power_usd}, "
            f"total_position_value_usd={a.total_position_value_usd}, "
            f"positions={sorted(a.positions)}, "
            f"realized_pnl_today_usd={a.realized_pnl_today_usd}"
        )
    if name == "get_quote":
        price = broker.get_quote(args["symbol"])
        return f"{args['symbol']}={price}"
    if name == "place_order":
        return executor.execute(
            args["symbol"], args["side"], args["notional_usd"]
        ).as_tool_text()
    return f"Unknown tool: {name}"


def run_session(
    *,
    client: anthropic.Anthropic,
    broker: Broker,
    executor: OrderExecutor,
    agent_cfg: AgentConfig,
    max_turns: int = 12,
) -> str:
    """Run the agent loop for one cron tick. Returns Claude's final text."""
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": "Review the account and decide whether to trade right now.",
        }
    ]

    final_text = ""
    for _ in range(max_turns):
        with client.messages.stream(
            model=agent_cfg.model,
            max_tokens=agent_cfg.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": agent_cfg.effort},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        final_text = "".join(
            b.text for b in response.content if b.type == "text"
        )

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                output = _dispatch(block.name, block.input, broker, executor)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return final_text
