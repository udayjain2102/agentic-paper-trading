# Robinhood Agentic Trading

A US-equities **strategy research system**. It runs an unattended paper-trade
loop (GitHub Actions, Mon-Fri) that ticks rule-based strategies and an LLM agent
(Nemotron, via NVIDIA's OpenAI-compatible API) forward one day at a time against
real prices, and scores the resulting track record.

> **It does not place orders.** The guardrail funnel runs on every scheduled
> day: `paper_run.run()` calls `guardrails.check_halted` (`paper_run.py:62`),
> then `OrderExecutor.execute()` calls `guardrails.validate_order`
> (`executor.py:60`) before `broker.place_order` (`executor.py:74`). What
> actually stops a real order is that `paper_run.py` hardcodes `MockBroker`
> unconditionally — it has no import of the live broker class and no branch
> that can place a real order — and `McpBroker` (`broker.py:82`), the only
> broker that talks to Robinhood, has zero callers anywhere in `src/`,
> `scripts/*.sh`, or `.github/workflows/*.yml`. `LIVE=true` gates nothing on
> this path: `config.py:47` reads it into `cfg.dry_run`, and no order code
> consults `dry_run` — it only reaches `make_dashboard.py:270` for display.
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

The scorecard (win rate, avg win/loss, profit factor, Sharpe, max drawdown)
and failure buckets (`rhagent.evaluate`), plus the robust bake-off (fold
Sharpe + bootstrap CI + deflated Sharpe, `rhagent.evaluate_robust`), are the
tools used to grade a strategy against backtests. They are **not** yet
confirmed against the live forward paper-trade record: as of this writing
that record (`origin/paper-state`) holds 10, 6, and 2 realized return-days
across its three tracked strategies, and all three `trades.jsonl` files are
0-1 bytes — zero trade records. The forward record is meant to be the thing
that eventually earns a `LIVE=true` flip, but it does not yet have enough
history to confirm or reject anything; see `docs/archive/FINDINGS.md` for the
original trust-ladder proposal.

The IC/ICIR machinery under `factor/`, `search/`, and `gate/` (docs/ARCHITECTURE.md
§2) is an offline research tool for *narrowing candidates* before they enter
the bake-off above — it is not a competing grading system, and a strategy does
not need to clear its gates to be promoted.

## Safety

The guardrail funnel described above (`check_halted` → `validate_order`)
executes on every scheduled run, in code, not just in tests:

- Per-trade cap, total-deployed cap, max new positions/run, max orders/run —
  defined in `config.yaml`, implemented in `guardrails.validate_order`.
- Daily realized-loss kill switch and `HALT` file — implemented in
  `guardrails.check_halted`, called from `paper_run.run()` on every tick.
- US equities only — non-equity symbols are rejected.

The property that keeps this repo from placing a real order today is not the
guardrails — it's that `paper_run.py` hardcodes `MockBroker` unconditionally
and `McpBroker` (the only broker that can reach Robinhood) has no caller
anywhere in this repo. If you ever wire a live broker into `paper_run.py`,
that hardcoding is the thing you'd be removing, and the guardrail funnel is
already upstream of it — verify the caps end to end when you do. Do not
assume the `LIVE` flag gates anything on this path; it does not (see above).

## Tests

```bash
.venv/bin/python -m pytest
```

The guardrails are covered exhaustively (every rejection path), and
`tests/test_paper_run.py` is an end-to-end dry-run smoke test against the real
`paper_run.run()` path: it asserts zero real orders are placed, rejected
orders never reach the broker, the `HALT` file and daily-loss kill switch
both stop execution, and positions persist across runs.
CI (`.github/workflows/tests.yml`) runs this suite on every push and PR; the
daily paper run also runs it first and fails fast if it doesn't pass.

## Out of scope (v1)

Options, crypto, real-time streaming, web UI. The rule-based strategies are
long-only (bake-off winner, `config.yaml`). The LLM agent's `allow_short` is a
config knob (`config.yaml: agent.allow_short`, default `true`) — shorting is
not blanket out of scope there.
