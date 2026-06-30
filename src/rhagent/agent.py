"""The decision agent: an LLM reasons over the portfolio and proposes trades.

The model is served by NVIDIA's OpenAI-compatible API (integrate.api.nvidia.com),
driven through the ``openai`` SDK.

We use a manual agentic loop (not an auto tool-runner) on purpose: it gives us
a code-controlled choke point where the ``place_order`` tool is dispatched to
``OrderExecutor`` — which enforces the guardrails before anything reaches the
broker. The model decides; the code decides what's allowed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

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


# OpenAI/NVIDIA tool-calling schema, derived from the single source above.
OPENAI_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
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


def run_scripted_session(
    *,
    broker: Broker,
    executor: OrderExecutor,
    **_ignored: Any,
) -> str:
    """A no-API stand-in for ``run_session``.

    Walks the same tool-dispatch path the real Claude loop would (the
    ``place_order`` calls still go through ``OrderExecutor`` and the guardrails),
    but the "decision" is a fixed script instead of a model call. Lets you
    exercise the full executor/journal/dry-run pipeline with no ANTHROPIC_API_KEY.
    The script deliberately includes one order that should clear the per-trade
    cap and one that should be rejected by it, so you can see both paths.
    """
    acct = _dispatch("get_account", {}, broker, executor)
    lines = [f"[scripted] get_account -> {acct}"]

    plan = [
        ("get_quote", {"symbol": "AAPL"}),
        ("place_order", {"symbol": "AAPL", "side": "buy", "notional_usd": 250}),
        ("place_order", {"symbol": "AAPL", "side": "buy", "notional_usd": 10_000}),
    ]
    for name, args in plan:
        out = _dispatch(name, args, broker, executor)
        lines.append(f"[scripted] {name}({args}) -> {out}")

    return "\n".join(lines)


def run_session(
    *,
    client: OpenAI,
    broker: Broker,
    executor: OrderExecutor,
    agent_cfg: AgentConfig,
    max_turns: int = 12,
) -> str:
    """Run the agent loop for one cron tick. Returns the model's final text."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Review the account and decide whether to trade right now.",
        },
    ]

    final_text = ""
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=agent_cfg.model,
            max_tokens=agent_cfg.max_tokens,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        final_text = msg.content or ""

        if not msg.tool_calls:
            break

        # Echo the assistant turn (with its tool_calls) back into the history.
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            output = _dispatch(call.function.name, args, broker, executor)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                }
            )

    return final_text
