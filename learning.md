# Learning from Vibe-Trading (HKUDS)

Source: https://github.com/HKUDS/Vibe-Trading
Cloned + run side-by-side with our repo on 2026-07-21. Clone at
`/Users/adijain/vibe-trading-compare` (venv: `.venv-compare`).

## What it is

Open-source AI trading research platform. Natural-language research prompt →
agent → strategy → backtest → paper/live trade.

- Backend: FastAPI, MCP transport for tool integration.
- Frontend: React 19, streaming UI.
- Agent runtime: LangChain/LangGraph loop, reasoning + tool calls + context
  compression.
- Data: 18+ sources (Tushare, OKX, CCXT, yfinance, Futu, mootdx, AKShare) with
  automatic fallback chains, point-in-time (PIT) fundamental data.
- Backtesting: composite across asset classes (A-share, HK, US, crypto,
  futures, forex) with shared capital pools.
- Trading: broker connectors (IBKR, Robinhood, Tiger, Alpaca, OKX, Binance),
  strategy export to Pine Script v6 / TradingView / MT5 / TDX.
- Research tools: 452+ prebuilt quant factors, "shadow account" broker-journal
  bias diagnosis, trade journal analysis, correlation heatmaps.
- Memory: persistent cross-session memory, "self-evolving skills."
- 13+ LLM providers supported.

## What to steal, mapped to our repo

Checked against actual code (`src/rhagent/`), not just the pitch — three real
gaps found:

### 1. Data fallback chain
[`data.py`](src/rhagent/data.py:38) `get_bars()` has exactly one source: the
Robinhood MCP (`mcp_fetch`), cache-first. No fallback. This is the concrete
code behind the NVIDIA rate-limit ceiling (burst-then-~18/min) that caps our
paper-trade run size — one source, one failure mode.

Vibe-Trading pattern: pluggable loaders, auto-fallback across sources on
429/timeout, same cache layer underneath.

Fix shape: `get_bars` already takes a `fetch` param. Swap
`fetch=mcp_fetch` for `fetch=fallback_chain([mcp_fetch, yfinance_fetch])`.
Small diff, no rewrite of the cache logic.

### 2. Bigger factor library
[`features.py`](src/rhagent/features.py:8) has one function
(`entry_features`), a handful of signals. Vibe-Trading ships 452 prebuilt
factors (momentum, value, quality, etc.) — we don't need 452, but 5-10 more
(RSI, vol-adjusted momentum, mean-reversion z-score) is cheap to add and
gives [`strategy_runner.py`](src/rhagent/strategy_runner.py) more signal to
pick from. Direct lever on backtest quality.

### 3. Reflection → enforced action (not just prose)
[`memory.py`](src/rhagent/memory.py:91) `reflect()` writes lessons as prose
into `journal/agent_memory.md`, fed back as prompt context next run —
advisory only. Nothing enforces the agent actually acting on a past lesson.

Vibe-Trading's "self-evolving skills" idea: turn a lesson into a concrete
parameter change instead of hoping the LLM re-reads and obeys its own diary.
E.g. 3 losses on the same symbol → auto-blacklist N days, or auto-tune
`per_trade_max_usd` in [`guardrails.py`](src/rhagent/guardrails.py:26)
`Limits`. This closes the loop from text to enforced code.

## Where we already beat Vibe-Trading

[`guardrails.py`](src/rhagent/guardrails.py:1) — pure, side-effect-free, hard
caps the model cannot talk its way past (`validate_order`, `check_halted`).
Vibe-Trading's pitch is research/prototyping speed, not this kind of
production safety rail. Don't touch this, it's our strong point.

## Explicitly skipped (YAGNI for us)

- Multi-agent "trading committees" — overkill, single agent works for a
  single Robinhood account.
- Pine Script / MT5 / TradingView export — we trade directly via the
  Robinhood MCP, no external platform hop needed.
- 452-factor library at full scale — we're not running a multi-market quant
  shop.
- PIT fundamental data — moot until we actually pull fundamentals (we
  currently only use price bars); revisit if factor set above grows to
  include fundamentals.

## Bottom line

Two real levers (data fallback, factor count) plus one design upgrade
(reflection → enforcement). The rest of Vibe-Trading solves problems we
don't have (multi-broker export, quant-shop scale, multi-market composite
backtesting).

---

# Side-by-side run comparison (2026-07-21)

Both programs were cloned/installed and actually run. This section records the
concrete, evidence-backed differences observed — structural (code) and runtime
(logs). Every claim below was verified by running or reading, not inferred.

## 0. Setup / runnability

| | Ours (rhagent) | Vibe-Trading |
|---|---|---|
| Install | `.venv`, `requirements.txt`, runs offline in dry-run | 64-dep heavy stack (langchain, langgraph, fastapi, react frontend, tushare/ccxt/yfinance, weasyprint, sklearn). Needs pydantic+langchain+fastmcp just to import all CLI paths. |
| Entry point | `python -m rhagent.runner` (one cron tick) | `vibe-trading` CLI REPL + `serve` (FastAPI) + `mcp_server` |
| Runs with no keys? | Yes — dry-run uses a simulated account | Partially — CLI/`--skills`/`provider doctor`/`alpha list` work offline; `run`/`backtest`/`swarm` need a provider key (+ data keys) |

Observed install walls on Vibe: `rich` → `pydantic` → `fastmcp` missing, one
at a time. `swarm-presets` still fails without `fastmcp`. Ours has no such wall.

## 1. Scale

- **Ours: 3,999 LOC** Python, 22 modules, single package `rhagent`.
- **Vibe: 219,741 LOC** Python (`agent/`) + a React 19 frontend.
- ~55x larger. Different category: ours is a focused cron agent; Vibe is a
  full-stack research platform.

## 2. Scope / surface

| Dimension | Ours | Vibe-Trading |
|---|---|---|
| Brokers | 1 (Robinhood MCP) | IBKR, Robinhood, Tiger, Alpaca, OKX, Binance, MT5 |
| Markets | US equities only | A-share, HK, US, India, crypto, futures, forex |
| LLM providers | 1 (NVIDIA Nemotron, OpenAI-compatible) | 13+ (Claude, GPT, DeepSeek, Kimi, Gemini, …) |
| Interface | cron tick → JSONL | CLI REPL, REST API (FastAPI+SSE), MCP server, web UI |
| Agent shape | single manual agentic loop | LangGraph loop + multi-agent "swarm" presets |

## 3. Data layer — the confirmed steal-target

- **Ours** [`data.py`](src/rhagent/data.py:38): ONE source (RH MCP `mcp_fetch`),
  cache-first CSV, **no fallback**. This is the actual code behind our
  NVIDIA/MCP rate-limit ceiling — single source, single failure mode.
- **Vibe** [`agent/src/market_data.py`]: regex source routing (`detect_source`)
  → `FALLBACK_CHAINS` in `backtest/loaders/registry.py`. Symbol pattern picks a
  preferred source (yahoo/tencent/okx/ccxt/mt5/tushare), then degrades down the
  chain on failure. Also does token-budget row-capping (`cap_rows`,
  even-stride downsample, last bar pinned).

This directly validates steal-target #1. Their routing table is worth copying
in spirit (not the China/forex rules we don't need).

## 4. Factor library

- **Ours** [`features.py`](src/rhagent/features.py:8): **1 function**
  (`entry_features`), a handful of signals.
- **Vibe**: **462 registered alphas** (verified via `alpha list`: "Showing 50
  of 462"). Zoos: `alpha101` (Kakushadze), `gtja191`, `qlib158`, `academic`,
  `fundamental`. CLI: `alpha list/show/bench/compare/export-manifest` with
  IC/IR benchmarking over a universe.
- Gap is real but we don't need 462; 5-10 more (RSI, vol-adj momentum,
  z-score reversion) is the right dose. Their `alpha101` set is the cleanest
  source to lift a few from.

## 5. Backtest engine

- **Ours** [`backtest.py`](src/rhagent/backtest.py): 81 lines, single-asset,
  positions→net-returns, metrics = total_return / sharpe / max_drawdown /
  hit_rate. No I/O, dead simple, easy to trust.
- **Vibe** [`agent/backtest/`]: **12 market-specific engines** (china_a,
  crypto, forex, futures_base, global_equity, india_equity, options_portfolio,
  composite, …), plus `metrics.py` with per-source annualization tables
  (252/365/260 trading days, bars-per-day per interval), `validation.py`,
  `optimizers/`, `correlation.py`, `run_card.py`.
- Their `metrics.py` per-source annualization is a subtle correctness thing
  worth noting: our fixed `_ANNUALIZATION = 252` is fine for US daily equities
  (our only case) but would mis-annualize if we ever add crypto/intraday.

## 6. Memory

- **Ours** [`memory.py`](src/rhagent/memory.py): `agent_memory.md`, prose
  reflections, `MAX_ENTRIES=40`, fed back as prompt context — **advisory only**,
  nothing enforces a past lesson.
- **Vibe** [`agent/src/memory/persistent.py`]: `~/.vibe-trading/memory/` with a
  `MEMORY.md` index + per-entry **YAML frontmatter**, typed
  (user/feedback/project/reference), scored retrieval (`METADATA_WEIGHT=2.0`),
  CLI `memory list/show/search/forget`. **This is the exact same pattern as the
  `~/.claude` auto-memory system** — Vibe just made it a first-class trading
  feature with typed entries and a recall query.
- Still advisory in both. Our steal-target #3 (lesson → enforced parameter
  change) is not something Vibe does either — it'd be a genuine improvement
  over both.

## 7. Safety — where the two philosophies diverge (our strong point)

- **Ours** [`guardrails.py`](src/rhagent/guardrails.py): pure, side-effect-free,
  **code-enforced hard caps on orders** — per-trade $, total deployed $, new
  positions/run, orders/run, daily-loss kill switch, HALT file. The LLM
  physically cannot place an order past these. **Verified at runtime**: the live
  NVIDIA agent proposed buying SPY $5000, guardrails rejected it
  (`Order $5000.00 exceeds per-trade cap $250.00`).
- **Vibe** [`agent/src/security/`]: `network.py`, `scanner.py`,
  `workspace_access.py`, `workspace_policy.py` — **sandbox/code-execution
  safety** (Vibe generates and runs strategy code, so it guards the sandbox,
  filesystem, network). It does NOT have order-level financial hard caps like
  ours.
- Different problems: Vibe sandboxes generated code; we cap live orders.
  **Ours is the safer design for actually trading real money** — a code-enforced
  order cap the model can't argue past. Keep it, it's our edge.

## 8. Runtime logs observed

**Ours** (`journal/runs.jsonl`, append-only JSONL audit trail):
```
{"event":"run_start","mode":"DRY-RUN",...}
{"event":"order_rejected","symbol":"SPY","notional_usd":5000.0,
 "reason":"Order $5000.00 exceeds per-trade cap $250.00.",...}
```
Real live-LLM decision → caught by code. Run completed (exit 0) after a
multi-round loop under NVIDIA's burst-then-~18/min token bucket. Final
`run_end` summary (live agent's own words): *"The order to buy $5000 of SPY was
rejected because it exceeded the per-trade cap of $250. Since SPY's price
($540) is higher than the $250 cap, even a single share..."* — the model
correctly reasoned about the guardrail it hit. All output lands in JSONL;
console prints only a final summary.

**Vibe** (rich CLI, run traces): `--skills` (88 skills listed), `provider
doctor` (structured JSON diagnostics: provider/model/keys/packages/timeouts),
`alpha list` (462 factors, rich table), subcommands
`run/serve/provider/data/channels/list/show/chat/memory/alpha/hypothesis`.
Run history via `--list/--show/--trace RUN_ID`. To actually `run` a prompt it
needs a provider key; we did not supply one (would incur cost + is an external
LLM call), so no live agent decision was captured from Vibe.

## 9. Output / export

- **Ours**: `journal/runs.jsonl` + `dashboard.html` (`make_dashboard.py`).
- **Vibe**: run traces, **Pine Script export** (`--pine`), generated strategy
  code (`--code`), **shadow-account HTML/PDF reports** (weasyprint/jinja2),
  correlation heatmaps. Broader, but tied to the research-platform use case.

## 10. Net differences, ranked by relevance to us

1. **Data fallback chain** (their `market_data.py` routing + `FALLBACK_CHAINS`)
   — directly fixes our known rate-limit ceiling. Highest value.
2. **A few more factors** (lift 5-10 from their `alpha101` zoo) — cheap signal.
3. **Per-source annualization** (their `metrics.py`) — only matters if we leave
   US daily equities; note-worthy, not urgent.
4. **Typed memory with recall** (their `persistent.py`) — nicer than our flat
   prose file, but our flat file is fine at current scale; skip unless memory
   grows.
5. Everything else (multi-broker, multi-market, 12 backtest engines, swarm,
   Pine/MT5 export, sandbox security, FastAPI/React) solves problems we don't
   have. Skip.

**They do NOT beat us on:** code-enforced order-level safety. That's ours.

---

# Second run (2026-07-21, re-run after offline changes)

Re-ran both to confirm behavior after adding the fallback chain + factors.

- **Ours**: ran clean (exit 0), 212 tests still green. Notable: this run the
  live agent produced **zero orders** — it saw $5,000 buying power / no
  positions and chose to sit out, whereas the earlier run today tried SPY
  $5,000 on the *same* account state. **Learning: the live NVIDIA agent is
  non-deterministic** — identical inputs, different trade/no-trade decision.
  This is exactly why the code-enforced guardrails matter more than the
  model's judgement: the safety floor is deterministic even when the agent
  isn't. It also argues for our steal-target #3 (lesson → enforced parameter)
  over relying on prose reflection the model may or may not act on.
- **Vibe**: still **0 trades**. No `~/.vibe-trading` store exists — it never
  ran a live agent (no provider key supplied, would be a billable external
  call). Vibe only produces trades on-demand inside a per-prompt run trace and
  persists none standing; there is nothing to diff against our ledger.

## Full trade ledger (ours) — 12 order events / 14 runs

5 accepted (all exactly $250, the per-trade cap: AAPL x3, SPY, NVDA x2),
5 rejected (every $1k-$10k reach blocked by the cap), 2 intended-then-capped
pairs. Newest run: no order. Zero real money moved — all dry-run.

**Vibe ledger: empty.** Not comparable — ours is a standing audit trail, Vibe
is stateless per-prompt generation.

---

# Third pass (2026-07-21): got Vibe to actually trade, using our NVIDIA key

Pointed Vibe's `openai` provider at NVIDIA's OpenAI-compatible endpoint with our
existing `NVIDIA_API_KEY` (`OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1`,
model `nvidia/llama-3.3-nemotron-super-49b-v1.5`). Vibe's agent ran end-to-end
and finally produced trades — but only after clearing four walls.

## What it took (the real story)

1. **LLM key** — reused our NVIDIA key via the OpenAI-compat shim. Auth worked.
2. **Sandbox run-root** — `VIBE_TRADING_ALLOWED_RUN_ROOTS` parser splits on
   **comma**, not colon. First attempt (colon) silently produced one garbage
   path → backtest stayed blocked by Vibe's own `security/` sandbox.
3. **~60 deps** — yfinance, tushare, defusedxml, bottleneck, etc., installed one
   ImportError at a time until the loader/engine chain resolved.
4. **Vibe's agent-generated code was broken** — the LLM wrote a `signal_engine.py`
   that assumed the wrong data schema (`data_map['close']`, treating data_map as
   field-keyed) and returned the wrong type (a `pd.Series`, not the required
   `Dict[str, pd.Series]`). It raised `KeyError: 'close'` and would not run. I
   hand-fixed it to the real engine contract (`data_map` is `Dict[symbol ->
   DataFrame]` with a `close` column; return `{symbol: signal_series}`).

**Sharpest finding:** Vibe's run reported `Status: SUCCESS` and "no trades" both
times, and never surfaced that the code its own agent wrote was non-runnable.
The failure was silent — a green status over broken generated code. Our
guardrail-first design can't emit an order off code that doesn't execute.

## Vibe's 4 executed trades (AAPL SMA-20 momentum, 6mo, $1M backtest)

| # | Entry | Exit | Hold | Return |
|---|---|---|---|---|
| 1 | 2026-02-19 $262.73 | 2026-02-20 $258.84 | 1d  | -1.48% |
| 2 | 2026-02-24 $267.99 | 2026-03-02 $262.28 | 6d  | -2.13% |
| 3 | 2026-04-02 $254.33 | 2026-06-09 $300.13 | 68d | +18.01% |
| 4 | 2026-07-06 $307.51 | 2026-07-21 $326.79 | 15d | +6.27% (closed at end) |

Total +20.9%, Sharpe 2.35, win rate 50% (2W/2L). **Underperformed buy-and-hold
AAPL (+32%) by ~11% excess return** — the momentum rule churned in/out and
missed upside vs just holding.

## Still not apples-to-apples

- **Vibe** = historical *backtest* of a mechanical rule → round-trip trades with
  realized P&L.
- **Ours** = *live LLM* order intents, guardrail-capped, dry-run, never
  executed → no exits, no realized returns.

Both now have "trades" on the table, but they measure different things: Vibe
scores a strategy over the past; ours records what the agent decided to do in
the present, with the safety floor enforced in code.

---

# Improvement shipped: agent decision-P&L tracker (2026-07-21)

Built from the top gap the Vibe comparison exposed — Vibe measures strategy P&L,
we logged order *intents* but never graded them. Written by a Sonnet subagent,
reviewed by an Opus subagent (verdict: SHIP), one root-cause fix applied.

## New: `src/rhagent/decision_pnl.py` (+ `tests/test_decision_pnl.py`)
Scores every accepted dry-run decision (`order_intended` in `journal/runs.jsonl`)
against forward price moves — 1-day and 5-day forward returns, lookahead-free
(same technique as `memory.recent_outcomes`). Offline/deterministic: reads only
cached bars, never fetches during a reporting run. `python -m rhagent.decision_pnl`
prints a per-decision table + aggregate. Purely additive, touches no safety code.

## First real read on our agent's decision quality
```
n_decisions=6  scored_1d=5  hit_1d=100%  avg_1d=+1.60%
               scored_5d=4  hit_5d=50%   avg_5d=+1.11%
```
Small sample, but: the agent's picks were up 5/5 the next day (+1.60% avg), and
still net-positive at 5 days (+1.11%) though the edge decays (one AAPL call went
-7% over 5d). This is exactly the kind of measurement we couldn't do before.

## Process note (subagent workflow)
Sonnet wrote it (215 tests green, real CLI verified). Opus reviewed for lookahead
bias / off-by-one / aggregation / safety → clean SHIP with two nits. Nit #1
(uncached symbols got a range-truncated fetch) turned out to share a root with a
failing test isolation issue; fixed both by making the scorer purely cache-read
(`fetch=lambda: {}`) — more correct than the reviewer's suggested padding.
