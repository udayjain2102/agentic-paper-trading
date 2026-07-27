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
> this path: `config.py` reads it into `cfg.dry_run` (`is_live()`), and no
> order code consults `dry_run` — it only reaches `make_dashboard.py` for
> display. Everything this repo does is read-only against market data.

## How it actually runs

The scheduled path is a GitHub Actions cron (`.github/workflows/daily-paper-run.yml`,
Mon-Fri) that runs `scripts/paper_cron.sh`: it refreshes the price cache
(Yahoo's keyless chart API by default; the Robinhood MCP only if
`ROBINHOOD_MCP_URL`/`ROBINHOOD_MCP_TOKEN` secrets are set), ticks three forward
paper-trade records (`rhagent.forward`), runs `rhagent.paper_run` through the
guardrail funnel (dry-run, `MockBroker` only), and renders the dashboard.
Nothing here places a real order — there is no code path that can.

The three records exist on purpose, on different cost/fill bases:

| Record | Cost | Fill | Why |
|---|---|---|---|
| `mean_reversion` | 1 bp | `close` | The original record, pinned to its seed basis so its curve has no discontinuity. Flattering fills. |
| `mean_reversion_real` | 7 bp | `next_open` | **The honest go-forward number** — a cost and a fill you could actually get. |
| `agent` | config | config | The LLM engine, only when `NVIDIA_API_KEY` is set. |

## Layout

| File | Role |
|------|------|
| `scripts/paper_cron.sh` | **The real scheduled entry point** — refresh, three forward ticks, guardrail-gated `paper_run` tick, state push. |
| `src/rhagent/refresh.py` | Historical bars: Yahoo by default, RH MCP if secrets are set. Cached to `data/*.csv`. |
| `src/rhagent/forward.py` | Ticks the forward paper-trade records the scheduled run and dashboard read from. |
| `src/rhagent/engine.py` | The two decision engines: `StrategyEngine` (rules) and `AgentEngine` (LLM). |
| `src/rhagent/backtest.py` | `net_returns` — the one place a position series becomes a return series. |
| `src/rhagent/evaluate.py` | Scorecard, failure buckets, SPY benchmark. |
| `src/rhagent/strategies/` | Rule-based strategies (mean-reversion, momentum, linreg). |
| `src/rhagent/guardrails.py` | Pure, exhaustively-tested safety checks, called on every `paper_run` tick (see Safety). |
| `src/rhagent/broker.py` | `MockBroker` / `McpBroker` — `paper_run.py` only ever constructs `MockBroker`; `McpBroker` has no caller. |
| `src/rhagent/mcp_session.py` | Connects to the Robinhood MCP (streamable HTTP). Used for data, not orders. |
| `scripts/make_dashboard.py` | Renders the record to a static HTML page, deployed to Vercel by CI. |
| `config.yaml` | Guardrail limits, model config, and the locked-in strategy preset. |

Full detail — every module, flag and known weak point — is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
history to confirm or reject anything; see
[`docs/archive/AUDIT-2026-07-16.md`](docs/archive/AUDIT-2026-07-16.md) for the
original trust-ladder proposal.

The IC/ICIR machinery under `factor/`, `search/`, and `gate/`
([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2) is an offline research
tool for *narrowing candidates* before they enter the bake-off above — it is
not a competing grading system, and a strategy does not need to clear its
gates to be promoted. Its one real-data verdict so far was `viable: 0`.

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
4. **Small real money behind the existing guardrail-gated order path**
   (`paper_run.py`), with the `config.yaml` caps tightened further than their
   defaults, and `MockBroker` swapped for a real one. See
   [`docs/going-live.md`](docs/going-live.md) for the runbook.

Longer reasoning behind this ladder:
[`docs/archive/AUDIT-2026-07-16.md`](docs/archive/AUDIT-2026-07-16.md).

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
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The guardrails are covered exhaustively (every rejection path), and
`tests/test_paper_run.py` is an end-to-end dry-run smoke test against the real
`paper_run.run()` path: it asserts zero real orders are placed, rejected
orders never reach the broker, the `HALT` file and daily-loss kill switch
both stop execution, and positions persist across runs.
CI (`.github/workflows/tests.yml`) runs this suite on every push and PR; the
daily paper run also runs it first and fails fast if it doesn't pass.

## Out of scope (v1)

Options, crypto, real-time streaming. The only UI is the static dashboard
`scripts/make_dashboard.py` generates. The rule-based strategies are
long-only (bake-off winner, `config.yaml`). The LLM agent's `allow_short` is a
config knob (`config.yaml: agent.allow_short`, default `true`) — shorting is
not blanket out of scope there.
