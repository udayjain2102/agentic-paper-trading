# Robinhood Agentic Trading

An autonomous US-equities trading agent. On a cron schedule, an LLM (Nemotron,
served over NVIDIA's OpenAI-compatible API) reviews the account and market data
through the Robinhood trading MCP, and decides whether to place trades — inside
**hard, code-enforced safety rails**. It defaults to **paper (dry-run) mode**
and only places real orders when you explicitly switch it to live.

> ⚠️ This trades real money when live. Read [Safety](#safety) before enabling it.
> The author of this software does not run it or place trades for you — funding
> the account and flipping `LIVE=true` is your decision, and PDT rules, taxes,
> and Robinhood's API terms are your responsibility.

## How it works

One cron tick = one run:

```
cron → python -m rhagent.runner
        ├─ load config + guardrail limits
        ├─ check HALT file + daily-loss kill-switch → abort if tripped
        ├─ the LLM reasons over account + quotes (read via the RH MCP)
        │     and proposes orders by calling place_order
        ├─ every proposed order is validated in code (guardrails.py)
        ├─ DRY-RUN: log the intended order, place nothing
        │  LIVE:    place via the broker, record the fill
        └─ append every decision to journal/runs.jsonl
```

**The agent never calls the broker's order API directly.** Its `place_order`
tool is dispatched to `OrderExecutor`, which runs the guardrails first. The
model cannot talk its way past a hard cap — the cap is enforced in code.

## Layout

| File | Role |
|------|------|
| `src/rhagent/guardrails.py` | Pure, exhaustively-tested safety checks. The core. |
| `src/rhagent/executor.py` | The single funnel every order passes through. |
| `src/rhagent/broker.py` | The only code that touches the broker (`MockBroker` / `McpBroker`). |
| `src/rhagent/mcp_session.py` | Connects to the Robinhood MCP (streamable HTTP). |
| `src/rhagent/agent.py` | The LLM decision loop (manual agentic loop, Nemotron via NVIDIA's API). |
| `src/rhagent/runner.py` | Orchestrates one cron tick. |
| `src/rhagent/journal.py` | Append-only JSONL audit trail. |
| `config.yaml` | Guardrail limits + model config. |
| `src/rhagent/strategies/` | Rule-based strategies (mean-reversion, momentum, linreg). |
| `src/rhagent/backtest.py` | Offline backtest engine (equity curve + metrics). |
| `src/rhagent/data.py` | Historical bars via the RH MCP, cached to `data/*.csv`. |
| `src/rhagent/compare.py` | Rank all strategies by total return, pick the winner. |
| `src/rhagent/strategy_runner.py` | Turns a strategy's target positions into orders. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Running

```bash
# Dry-run (default). With no Robinhood MCP token set, uses a simulated account.
.venv/bin/python -m rhagent.runner
```

Schedule it (example: every 30 min during market hours, Mon–Fri):

```cron
*/30 13-20 * * 1-5  cd /path/to/project && .venv/bin/python -m rhagent.runner >> cron.log 2>&1
```

### Going live

1. **Authenticate the Robinhood MCP.** In an interactive `claude` session in
   this directory, run `/mcp`, pick `robinhood-trading`, and complete the OAuth
   flow. Then put the resulting bearer token in `.env` as `ROBINHOOD_MCP_TOKEN`.
   (Confirm the MCP's actual read/order tool names against its `list_tools` and
   adjust the mapping in `broker.py` if they differ from the placeholders.)
2. **Watch it on paper first.** Let it run in dry-run and review
   `journal/runs.jsonl` until you trust its behavior.
3. **Flip the switch.** Set `LIVE=true` in your environment. Only the literal
   string `true` enables live trading; anything else stays paper.

## Backtesting & strategy mode

Rank the four strategies over ~1yr of daily bars and pick the best by total return:

```bash
.venv/bin/python -m rhagent.compare
```

It caches price data under `data/` (gitignored). Paste the printed `strategy:`
block into `config.yaml`, then run the winner through the normal guardrails:

```bash
STRATEGY_MODE=true .venv/bin/python -m rhagent.runner
```

Strategy mode is dry-run unless `LIVE=true`, and every order it emits passes
through the same `OrderExecutor`/guardrails as the LLM path.

## How a strategy is graded

**Trade-level grading is the project's judge**: the scorecard (win rate, avg
win/loss, profit factor, Sharpe, max drawdown) and failure buckets
(`rhagent.evaluate`), plus the robust bake-off (fold Sharpe + bootstrap CI +
deflated Sharpe, `rhagent.evaluate_robust`) — all confirmed against the live
forward paper-trade record, not backtests. That forward record, growing
unattended every trading day, is the only thing that can earn the eventual
`LIVE=true` flip; see `md/FINDINGS.md` for the trust ladder.

The IC/ICIR machinery under `factor/`, `search/`, and `gate/` (md/ARCHITECTURE.md
§2) is an offline research tool for *narrowing candidates* before they enter
the bake-off above — it is not a competing grading system, and a strategy does
not need to clear its gates to be promoted.

## Safety

- **Dry-run by default.** `LIVE` must equal `true` to place real orders.
- **Per-trade cap, total-deployed cap, max new positions/run, max orders/run** —
  all in `config.yaml`, all enforced in `guardrails.py`.
- **Daily realized-loss kill switch** halts trading for the day once breached.
- **`HALT` file** — `touch HALT` in the project root to stop all trading
  immediately on the next run.
- **US equities only** — non-equity symbols are rejected.

## Tests

```bash
.venv/bin/python -m pytest
```

The guardrails are covered exhaustively (every rejection path), the broker is
mocked, and an end-to-end dry-run smoke test asserts zero orders are placed.

## Out of scope (v1)

Options, crypto, shorting, real-time streaming, web UI.
