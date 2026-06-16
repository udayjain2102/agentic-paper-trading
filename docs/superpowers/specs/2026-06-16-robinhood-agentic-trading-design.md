# Robinhood Agentic Trading — Design v1

**Date:** 2026-06-16
**Status:** Approved

## Goal

A Python application in which Claude (via the Claude Agent SDK) wakes on a cron
schedule, reviews the portfolio and market data through the Robinhood trading
MCP server, and decides and places US-equity trades — inside hard, code-enforced
safety rails. It defaults to paper (dry-run) mode and only places real orders
when explicitly switched to live.

## Decisions (locked)

- **Role:** Autonomous trading.
- **Runtime:** Scheduled batch (cron). One run = one cron tick: wake, evaluate,
  act, exit.
- **Decision engine:** LLM agent (Claude Agent SDK). Claude reasons over data
  each run and proposes actions.
- **Mandate / universe:** Any US equity (stocks/ETFs). Widest blast radius —
  hard caps are the primary containment.
- **Stack:** Python.
- **Safety posture:** Dry-run by default. Real orders require `LIVE=true`.

## Architecture (single run = one cron tick)

```
cron → runner.py
        ├─ load config + guardrail limits (env / config.yaml)
        ├─ check halt file + daily-loss kill-switch → abort if tripped
        ├─ start Claude Agent SDK session
        │     tools: Robinhood MCP (read quotes/positions/account)
        │            + a custom place_order tool (our wrapper, NOT raw MCP)
        │     system prompt: mandate, limits, output contract
        ├─ Claude reasons → proposes orders
        ├─ guardrail layer validates every proposed order
        │     (caps, symbol sanity, buying power, rate limit)
        ├─ if DRY_RUN: log intended order, place nothing
        │  if LIVE:    place via MCP, record fill
        └─ append run to journal (JSONL) + structured log
```

**Key design choice:** Claude never calls the raw order MCP tool directly. It
calls *our* `place_order` wrapper, which enforces every guardrail in code before
anything reaches Robinhood. The LLM cannot talk its way past a hard cap. Read
tools (quotes, positions, account) are exposed to Claude directly via the MCP.

## Components

- `runner.py` — orchestration of one tick.
- `guardrails.py` — pure, unit-tested validation: per-trade $ cap, total
  deployed cap, max new positions/run, max orders/run, buying-power check,
  daily realized-loss kill-switch, halt-file check.
- `broker.py` — thin wrapper over the Robinhood MCP (quotes, positions,
  place/cancel). The only component that touches the MCP.
- `agent.py` — Agent SDK setup: system prompt, tool registration, model config.
- `journal.py` — append-only JSONL of every decision and outcome (audit trail).
- `config.yaml` + `.env` — limits and the `LIVE` / `DRY_RUN` flag.
- `tests/` — guardrails covered exhaustively; broker mocked.

## Guardrails (configurable, conservative defaults)

- `DRY_RUN=true` by default — must explicitly set `LIVE=true` to place orders.
- Per-trade max dollars.
- Total deployed-capital max dollars.
- Max new positions per run.
- Max orders per run (rate limit).
- Buying-power check before any buy.
- Daily realized-loss kill-switch → halts trading for the rest of the day.
- `HALT` file present → immediate no-op abort.
- Only US equities accepted; anything else rejected.

## Data flow / state

- No database for v1. Positions and buying power are read live from Robinhood
  each run. Local state is limited to:
  - the append-only JSONL journal (audit trail), and
  - a small daily P&L tracker file backing the kill-switch.

## Testing

- `guardrails.py` unit-tested exhaustively — the safety-critical core.
- `broker.py` tested against a mock MCP.
- End-to-end dry-run smoke test asserting zero orders are placed.

## Out of scope for v1

Options, crypto, shorting, backtesting, real-time streaming, web UI. Cron plus
watch-and-learn-on-paper first.

## Operator responsibilities (not the software's)

- Funding the account and flipping `LIVE=true`.
- Compliance: PDT rules, taxes, and Robinhood API terms.
- Monitoring runs and pulling the halt file if needed.
