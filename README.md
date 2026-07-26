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
`ROBINHOOD_MCP_URL`/`ROBINHOOD_MCP_TOKEN` secrets are set), ticks three forward
paper-trade records (`rhagent.forward`), and renders the dashboard. Nothing here
places a real order — there is no code path that can.

The three records exist on purpose, on different cost/fill bases:

| Record | Cost | Fill | Why |
|---|---|---|---|
| `mean_reversion` | 1 bp | `close` | The original record, pinned to its seed basis so its curve has no discontinuity. Flattering fills. |
| `mean_reversion_real` | 7 bp | `next_open` | **The honest go-forward number** — a cost and a fill you could actually get. |
| `agent` | config | config | The LLM engine, only when `NVIDIA_API_KEY` is set. |

## Layout

| File | Role |
|------|------|
| `scripts/paper_cron.sh` | **The real scheduled entry point** — refresh, three forward ticks, state push. |
| `src/rhagent/refresh.py` | Historical bars: Yahoo by default, RH MCP if secrets are set. Cached to `data/*.csv`. |
| `src/rhagent/forward.py` | Ticks the forward paper-trade records the scheduled run and dashboard read from. |
| `src/rhagent/engine.py` | The two decision engines: `StrategyEngine` (rules) and `AgentEngine` (LLM). |
| `src/rhagent/backtest.py` | `net_returns` — the one place a position series becomes a return series. |
| `src/rhagent/evaluate.py` | Scorecard, failure buckets, SPY benchmark. |
| `src/rhagent/strategies/` | Rule-based strategies (mean-reversion, momentum, linreg). |
| `src/rhagent/guardrails.py` | Pure, exhaustively-tested safety checks — currently called by nothing (see Safety). |
| `src/rhagent/broker.py` | `MockBroker` / `McpBroker` — no order path reaches them today. |
| `src/rhagent/mcp_session.py` | Connects to the Robinhood MCP (streamable HTTP). Used for data, not orders. |
| `scripts/make_dashboard.py` | Renders the record to a static HTML page, deployed to Vercel by CI. |
| `config.yaml` | Guardrail limits, model config, and the locked-in strategy preset. |

Full detail — every module, flag and known weak point — is in
[`.md/ARCHITECTURE.md`](.md/ARCHITECTURE.md).

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
`LIVE=true` flip.

The IC/ICIR machinery under `factor/`, `search/`, and `gate/`
([`.md/ARCHITECTURE.md`](.md/ARCHITECTURE.md) §8) is an offline research tool
for *narrowing candidates* before they enter the bake-off above — it is not a
competing grading system, and a strategy does not need to clear its gates to be
promoted. Its one real-data verdict so far was `viable: 0`.

## The road to real money

Trust is a ladder, and each rung is evidence, not code:

1. **Forward record growing daily, unattended** — the system runs itself and
   the numbers are honest. (This rung is met; the records are on `paper-state`.)
2. **The record clears a bar defined in advance.** Months, not days, and the
   bar written down *before* the data exists — e.g. "3+ months, positive net of
   costs on the `mean_reversion_real` basis, max drawdown within the backtest's,
   bootstrap CI lower bound above zero." At the current daily mean this takes
   years, not weeks; that is the honest timeline.
3. **The agent beats or matches the rule baseline** on the same cadence —
   otherwise the honest conclusion is to fund the rule, not the agent.
4. **Small real money behind a rebuilt order path**, with the `config.yaml` caps
   tightened further than their defaults. See
   [`.md/going-live.md`](.md/going-live.md) for the runbook.

Longer reasoning behind this ladder: [`.md/AUDIT-2026-07-16.md`](.md/AUDIT-2026-07-16.md).

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
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The guardrails are still covered exhaustively (every rejection path) and the
broker is mocked, even though neither is on a live path today. The dry-run
smoke test was removed along with the runner it exercised.
CI (`.github/workflows/tests.yml`) runs this suite on every push and PR; the
daily paper run also runs it first and fails fast if it doesn't pass.

## Out of scope (v1)

Options, crypto, shorting, real-time streaming. The only UI is the static
dashboard `scripts/make_dashboard.py` generates.
