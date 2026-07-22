# Robinhood Agentic Trading

A US-equities **strategy research system**. It runs an unattended paper-trade
loop (GitHub Actions, Mon-Fri) that ticks rule-based strategies and an LLM agent
(Nemotron, via NVIDIA's OpenAI-compatible API) forward one day at a time against
real prices, and scores the resulting track record.

> **It does not place orders.** There is no live order path in this repo: the
> runner/executor stack that once funnelled orders through the guardrails was
> removed as dead code (nothing scheduled ever invoked it). `LIVE=true` no
> longer causes anything to trade, and `guardrails.validate_order` /
> `check_halted` currently have no production callers — they are retained,
> tested, and ready for a future order path, not guarding a live one today.
> Everything this repo does is read-only against market data.

## How it actually runs

The scheduled path is a GitHub Actions cron (`.github/workflows/daily-paper-run.yml`,
Mon-Fri) that runs `scripts/paper_cron.sh`: it refreshes the price cache
(Yahoo's keyless chart API by default; the Robinhood MCP only if
`ROBINHOOD_MCP_URL`/`ROBINHOOD_MCP_TOKEN` secrets are set), ticks the forward
paper-trade record (`rhagent.forward`), and — only if `NVIDIA_API_KEY` is
set — runs one LLM-agent tick. Nothing here places a real order; this is a
paper/dry-run system end to end unless you flip `LIVE=true` yourself.


## Layout

| File | Role |
|------|------|
| `scripts/paper_cron.sh` | **The real scheduled entry point** — refresh, forward tick, optional agent tick. |
| `src/rhagent/refresh.py` | Historical bars: Yahoo by default, RH MCP if secrets are set. Cached to `data/*.csv`. |
| `src/rhagent/forward.py` | Ticks the forward paper-trade record the scheduled run and dashboard read from. |
| `src/rhagent/guardrails.py` | Pure, exhaustively-tested safety checks. |
| `src/rhagent/broker.py` | The only code that touches the broker (`MockBroker` / `McpBroker`). |
| `src/rhagent/mcp_session.py` | Connects to the Robinhood MCP (streamable HTTP). |
| `config.yaml` | Guardrail limits + model config. |
| `src/rhagent/strategies/` | Rule-based strategies (mean-reversion, momentum, linreg). |
| `src/rhagent/backtest.py` | Offline backtest engine (equity curve + metrics). |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in NVIDIA_API_KEY (needed for the LLM agent path)
```

## How a strategy is graded

**Trade-level grading is the project's judge**: the scorecard (win rate, avg
win/loss, profit factor, Sharpe, max drawdown) and failure buckets
(`rhagent.evaluate`), plus the robust bake-off (fold Sharpe + bootstrap CI +
deflated Sharpe, `rhagent.evaluate_robust`) — all confirmed against the live
forward paper-trade record, not backtests. That forward record, growing
unattended every trading day, is the only thing that can earn the eventual
`LIVE=true` flip; see `.md/FINDINGS.md` for the trust ladder.

The IC/ICIR machinery under `factor/`, `search/`, and `gate/` (.md/ARCHITECTURE.md
§2) is an offline research tool for *narrowing candidates* before they enter
the bake-off above — it is not a competing grading system, and a strategy does
not need to clear its gates to be promoted.

## Safety

**The current safety property is that there is no order path at all.** Nothing
in this repo can place a trade; the scheduled run only reads prices and appends
to a paper record.

The guardrail primitives below are implemented and exhaustively tested, but are
**not currently wired to anything** — they were enforced by `executor.py` /
`runner.py`, which were removed as dead code. Treat this list as the contract
any future order path must satisfy, not as protection in force today:

- Per-trade cap, total-deployed cap, max new positions/run, max orders/run —
  defined in `config.yaml`, implemented in `guardrails.validate_order`.
- Daily realized-loss kill switch and `HALT` file — implemented in
  `guardrails.check_halted`, called by nothing.
- US equities only — non-equity symbols are rejected.

If you reintroduce order placement, route it through `guardrails.validate_order`
before `broker.place_order` and re-verify the caps end to end. Do not assume the
`LIVE` flag still gates anything — it does not.

## Tests

```bash
.venv/bin/python -m pytest
```

The guardrails are still covered exhaustively (every rejection path) and the
broker is mocked, even though neither is on a live path today. The dry-run
smoke test was removed along with the runner it exercised.
CI (`.github/workflows/tests.yml`) runs this suite on every push and PR; the
daily paper run also runs it first and fails fast if it doesn't pass.

## Out of scope (v1)

Options, crypto, shorting, real-time streaming, web UI.
