# Architecture

How this system decides what to trade, and the math it uses to judge a
strategy.

The project has **two decision brains**, but only one of them runs through
the guardrail funnel today:

1. **The LLM agent** — Nemotron (via NVIDIA's API) reasons over the account
   each forward tick (`rhagent.forward --engine agent`) and computes a target
   position for the paper ledger directly. This path never calls
   `OrderExecutor` or `guardrails.py` — verified: `forward.py` has no
   reference to either. There is no order to guard because it never proposes
   a broker order, only a position size for its own paper track.
2. **The quant strategy pipeline** — the locked strategy (`mean_reversion`,
   `config.yaml`) is what `paper_run.py`'s scheduled tick turns into orders
   and routes through `OrderExecutor → guardrails` (§3) before
   `broker.place_order` — which is always `MockBroker` (see the README's
   "Safety" section).

The code-enforced funnel in §3 is real and exercised daily, but it sits on
the quant-strategy path only, not the LLM agent's.

---

## 1. The model

### 1a. The two decision engines (`engine.py`)

There is no standalone tool-calling `agent.py` loop — that design was replaced.
Both engines satisfy the same `DecisionEngine` Protocol —
`decide(symbol, history, current_pos) -> Decision` — where `Decision` carries a
`target` in `{-1, 0, +1}`, a `reason`, and a `status` (`"ok"` or `"failed"`).
Neither proposes a broker order; they propose a *position*.

- **`StrategyEngine`** wraps a rule-based `Strategy` (§1b) and returns
  `strat.target(history)` with `strat.signal(history).iloc[-1]` as conviction.
- **`AgentEngine`** asks Nemotron (`nvidia/llama-3.3-nemotron-super-49b-v1.5`
  via NVIDIA's OpenAI-compatible API) for a verdict from a compact,
  lookahead-free prompt (`last_close`, `momentum_5d`, `vol_20d`, `current_pos`,
  and the lessons string). `decide_all(symbols, histories, current_pos)` is the
  production entry point: **one model call decides every symbol in the
  universe for a given bar**, not one call per symbol — 65 sequential calls
  per bar against NVIDIA's burst-then-~18/min token bucket is what produced
  the timeouts and ~50-minute ticks before this batching. If the call fails,
  every symbol in that batch is `status="failed"`, holding `current_pos`; if
  it answers but a symbol's verdict is missing or malformed, only that symbol
  fails. `complete(prompt) -> str` is an injectable seam, so tests never hit
  the API.

### 1b. The rule-based strategies (`strategies/`)

Every strategy implements one contract (`strategies/base.py`):

- `positions(bars) -> Series` in `{-1, 0, +1}` — the target position, obeying the
  **no-lookahead invariant**: the position at day *t* uses only data up to and
  including day *t*.
- `signal(bars) -> Series` — a *continuous* score where higher = more bullish on
  the forward return. This is what the factor/IC math evaluates.
- `target(bars) -> float` — just *today's* position (the last value). The base
  default is `positions(bars).iloc[-1]`; a strategy whose last value is
  independent of the earlier ones may override it with a cheaper single-step
  computation. `StrategyEngine.decide` calls `target`.

| Strategy            | Signal                                                       | Position rule                                                                   |   |           |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------- | - | --------- |
| **mean\_reversion** | `-z`, where `z = (close − rollmean)/rollstd` over `lookback` | long when `z < −entry`, exit to flat when `z ≥ −exit` (hysteresis avoids churn) |   |           |
| **momentum**        | `close.pct_change(lookback)` (trailing return)               | `sign(trailing return)`                                                         |   |           |

`momentum` survives only as a cheap test fixture; `linreg` was deleted on
2026-09-03 (git history).

`clamp_short` maps any `-1` to `0` unless shorting is explicitly enabled. **The
rule-based strategies are long-only**: `config.yaml`'s locked preset (mean
reversion) is the long-only bake-off winner, and the paper-trade CLI exposes no
short toggle for it. The LLM agent is different: `AgentEngine.__init__` still
defaults `allow_short=False` (`engine.py:134`), but `config.yaml`'s `agent.allow_short:
true` is passed explicitly at every production `AgentEngine(...)` construction
site (`forward.py`'s `_positions` and `tick_and_reflect`), so the agent runs
with shorting enabled today — a change from earlier, when every production
call site omitted the kwarg and every production decision was long-only
long-or-flat. `config.yaml`'s `agent.use_lessons` (default `false`) is the
matching knob for the lessons text fed into the prompt: when false, `forward.py`
passes `lessons=""` instead of concatenating `memory.read_memory()` +
`learn.lessons_from_runs()`, because that stale lessons text (frozen since
2026-07-15, "avoid holding=long setups") was the other half of the all-flat
cause. A run can sweep the whole cached universe at once with `--symbols all`.

### 1c. The trade setup — the live preset

The shipped configuration (`config.yaml`) is **mean\_reversion with params: {}
(all defaults), the conviction overlay, long-only, over the 65-name universe.**
The empty `params` means every knob below is the strategy default in code, not a
tuned value.

**Entry / exit (strategies/mean\_reversion.py).** A trade is driven entirely by the
z-score `z = (close − 20-day mean) / 20-day std`:

| Knob       | Value                 | Meaning                                                                                                          |
| ---------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `lookback` | 20                    | rolling window for the z-score mean/std                                                                          |
| `entry`    | 1.0                   | **enter long** when `z < −1.0` — price ≥ 1σ below its own 20-day mean (a "statistically cheap" dip)              |
| `exit`     | 0.0                   | **exit to flat** when `z ≥ 0` — price has reverted back to the mean                                              |
| hysteresis | −1.0 enter / 0.0 exit | the gap between the two thresholds is a dead-band, so price wobbling around one level doesn't churn the position |
| direction  | long-only             | short signals are clamped to flat (`clamp_short`); the position is `0` or `+1`                                   |

**There is no per-trade price stop-loss.** Exit is purely mean-reversion: a position
is held until `z` climbs back to `0`. If price keeps falling (`z` goes *more*
negative) the position is **held, not cut** — that is the strategy's thesis and also
its tail risk. The only loss backstop is the portfolio-level kill switch below, and
it exists only on the live-order path, not in the paper forward record.

**Conviction gate (overlay.py, applied on the eval + forward path).** Of the raw
long entries, only the higher-conviction ones are taken: an entry is vetoed unless
`|signal|` (= `−z`) strictly exceeds the **60th percentile** (`pctile = 0.60`) of that
symbol's own `|signal|` over the trailing **120 bars** (`window = 120`).

**Costs / sizing (paper eval).** Turnover (`|Δposition|`) is charged `cost_bps = 1.0`
(1 bp); paper P\&L is scaled to a `notional` of $10,000; the forward record equal-
weights the per-name net returns across the universe.

**Live-execution guardrails (guardrails.py, §3 — apply only when actually placing
orders, LIVE=true; they do not touch the paper record).** These are the real
"when to stop" limits:

| Guardrail                   | Value          | Effect                                                                                   |
| --------------------------- | -------------- | ---------------------------------------------------------------------------------------- |
| `per_trade_max_usd`         | $250           | max dollars committed to any single order                                                |
| `total_deployed_max_usd`    | $2,000         | new buys rejected if total deployed would exceed this                                    |
| `max_new_positions_per_run` | 2              | cap on newly-opened symbols per run                                                      |
| `max_orders_per_run`        | 5              | cap on orders per run                                                                    |
| `daily_loss_limit_usd`      | $200           | **kill switch** — if realized P\&L for the day ≤ −$200, the run halts and places nothing |
| `HALT` file                 | present → halt | operator manual stop                                                                     |

These are deliberately conservative starter values (`config.yaml` says "tighten
before going live").

---

## 2. The math (removed 2026-09-03)

The offline cross-sectional IC/ICIR research pipeline (`factor/`, `search/`,
`gate/`: rank-IC, ICIR, half-life, Bonferroni and deflated-Sharpe gates, a
locked OOS split) was deleted on 2026-09-03. Its one real-data verdict was
`viable: 0`; nothing on the scheduled path imported it; it had not run since
July. The code and its full write-up are in git history before that date.
What remains of the grading math is `backtest.py` (net returns, Sharpe, max
drawdown, hit rate) and `evaluate_robust.py` (fold Sharpe, bootstrap CI,
deflated Sharpe on realized paper-trade returns), described under Loop D.

---

## 3. How it decides (the safety funnel)

Every order — LLM or strategy — passes through the same gauntlet. `guardrails.py`
is pure, does no I/O, holds no state, and is exhaustively tested on every
rejection path.

```javascript
                proposed order (symbol, side, notional)
                              │
                              ▼
            ┌─────────  OrderExecutor.execute  ─────────┐
            │  check_halted (top of run):               │
            │    • HALT file present?      → abort run   │
            │    • daily realized loss ≤ −limit? → abort │
            ├───────────────────────────────────────────┤
            │  validate_order (per order):               │
            │    1. valid side + US-equity ticker        │
            │       (1–5 uppercase letters; rejects      │
            │        crypto/options/anything else)       │
            │    2. notional > 0                          │
            │    3. notional ≤ per_trade_max_usd          │
            │    4. orders_placed < max_orders_per_run    │
            │    buys only:                               │
            │    5. notional ≤ buying_power               │
            │    6. deployed + notional ≤ total_dep_cap   │
            │    7. new-symbol count < max_new_positions  │
            └───────────────────────────────────────────┘
                              │
                 ok? ─── no ──► REJECTED (logged, not placed)
                  │
                 yes
                  ▼
        DRY-RUN: log intent, place nothing   │   LIVE: broker places, record fill
                  │                                     │
                  └──────────► journal/runs.jsonl ◄─────┘  (append-only audit)
```

Design choices worth noting:

- **Dry-run by default.** `LIVE` must equal the literal string `true`; anything
  else stays paper.
- **Sells skip the exposure checks** (5–7) — selling reduces risk.
- `validate_order` is a *pure function* — it doesn't mutate the run counters. The
  executor bumps `orders_placed`/`new_positions` only after an order is actually
  accepted and acted on.
- Limits live in `config.yaml` (per-trade $250, total-deployed $2000, 2 new
  positions/run, 5 orders/run, $200 daily-loss kill switch) — all conservative
  defaults.

The scheduled path today is `paper_run.py`, not a standalone `runner.py`
(that module no longer exists): load config → `check_halted` → turn the
day's target positions into orders → `OrderExecutor` → journal to
`journal/paper_orders.jsonl`. `paper_run.py` hardcodes `MockBroker`
unconditionally, so this path never reaches a real broker regardless of
`LIVE`. See the README's "Safety" section for the verified call chain.

---

## 4. How it improves

Improvement happens **offline, on the quant side**, in three nested loops that
get progressively stricter about "is this edge real?". Nothing here can touch
live trading until it survives all of them and is manually pasted into
`config.yaml`.

### Loops A and B — parameter search and OOS gate (removed 2026-09-03)

Deleted with the `factor/`, `search/` and `gate/` packages (§2). The locked
preset in `config.yaml` was chosen by Loop C's bake-off, not by these.

### Loop C — Event-driven paper-trade & failure analysis (`papertrade.py`, `evaluate.py`)

Where a surviving strategy meets bar-by-bar reality and its *failures* get
diagnosed. `PaperTrader` steps a `DecisionEngine` through history one bar at a
time (never peeking ahead), turning each position change into a discrete,
ID-stamped trade written to an append-only ledger (`journal/papertrade/{run_id}/`).

Two seams keep it extensible without touching the loop: bars come from a
`MarketSource`, fills from a `FillModel`. **These two seams are the world-model
hook** (§5).

The payoff is `evaluate.py`:

- **Aggregate scorecard** — win rate, avg win/loss, profit factor, total return,
  Sharpe, max drawdown, avg holding period. Return metrics reuse
  `backtest.result_from_returns`, so paper-trade and vectorized numbers agree.
- **Failure buckets** — *where do the losses concentrate?* At entry, each trade
  records cheap lookahead-free features (20-day vol, overnight gap, 5-day trend).
  Losses are then attributed across dimensions (vol regime, gap direction,
  holding length, symbol, side) and ranked by **loss share**. This is the
  feedback signal: it tells you the edge dies in, say, high-vol down-gaps, which
  points at the next parameter or filter to change.
- **compare command** — rank every paper-trade run side by side. The same
  numbers render as a self-contained HTML dashboard (`scripts/make_dashboard.py`):
  an all-runs index (per run: trades, won/lost counts, net P\&L, total return,
  Sharpe, max DD) where clicking a run id opens *only* that run's full detail
  (scorecard, equity curve, ledger, failure buckets) — native CSS `:target`, no
  JavaScript.

### Loop D — Learning from losses: decision overlays (`overlay.py`, `evaluate_robust.py`)

Loops A–C *judge* strategies; Loop D lets one **adapt to its own realized
losses** without retraining. A single seam sits between the strategy's raw target
and the position actually taken — `StrategyEngine.decide` produces a target and a
per-bar `conviction` (the continuous `signal()` value), and an `Overlay.adjust`
gets the last say:

```javascript
final_target = overlay.adjust(symbol, history, decision, closed_trades)
```

`closed_trades` is the ledger of trades that closed **strictly before** today's
bar — the seam snapshots it once per bar, before any of that bar's own closes, so
the same **no-lookahead invariant** holds as everywhere else. Return `0` to veto,
a fraction to down-size, or the raw target to pass through. The baseline is an
`IdentityOverlay` (a `--overlay none` run is byte-identical to no overlay at all).
One overlay survives:

- **ConvictionGate** — vetoes entries whose `|conviction|` is below a rolling
  percentile of that symbol's own past convictions (trade-level noise filter).

Two other variants (BucketFilter, a loss-bucket veto; WinProbGate, a logit
win-probability gate) were baked off against it, lost, and were removed in the
2026-07-17 cleanup; they live in git history.

Because these barely-profitable strategies live in the noise, the bake-off is
judged by a **robust evaluator** (`evaluate_robust.py`), not a single Sharpe:
per-fold Sharpe across rolling windows, a **bootstrap 95% CI** on the per-bar net
returns, and a **deflated Sharpe** that penalizes for the number of variants
tried (penalizes for the number of variants tried).
A variant "beats baseline" only if its CI lower bound clears the *same
engine+universe* baseline's Sharpe. This renders as a bake-off panel on the
dashboard. Empirically so far: the conviction gate lifts point Sharpe \~5×
(0.11 → 0.56) but its CI still spans zero — **nothing clears the noise band**,
which is the honest, expected outcome at this data scale.

### Loop E — The forward track record (`forward.py`, `refresh.py`)

Loops A–D score strategies on *history*. Loop E builds the one thing a backtest
cannot: a **genuine out-of-sample record that accrues going forward**. `forward.py`
ticks once per trading day, computing the configured strategy's (conviction-gated)
net return for each newly-realized day and appending it to a single growing record
under `journal/forward/<eval_id>/`, in the same format `evaluate.py` and the
dashboard already read. It is **anchored** at first run — the curve reflects the
go-forward period, not backfilled history — and reuses `backtest.net_returns`, so
forward numbers match the ranking path exactly.

- **Fully-realized-day guard.** `net_returns` records a day's return at its *entry*
  date (the position on day *t* earns *t→t+1*), so a day is trustworthy only once
  the next bar exists for **every** universe name. The tick appends a day only when
  the whole basket has settled it (`df.notna().sum(axis=1) == len(universe)`);
  ticking mid-update would otherwise bake in a thin partial-day mean. A corollary:
  one chronically-missing name would freeze the record — which is why the dead XOM
  listing was dropped (universe is 65 names).
- **Conviction on the forward path.** The bar-by-bar `ConvictionGate` (Loop D) has an
  exact vectorized twin, `overlay.apply_conviction` (proven bit-identical); the
  forward path applies it whenever `strategy.overlay == "conviction"`, so the
  go-forward record uses the same gate the bake-off crowned.
- **Exclusion rule for the agent engine.** `_agent_positions` decides every
  symbol for a bar with one `decide_all` call (§1a) and freezes each verdict
  to `eval_dir/pos_<sym>.csv` (columns `date,pos,status`). A date is dropped
  from the net-return series if **any** symbol's decision for it is not
  `status == "ok"` — a failed call holds the prior position, so that P&L
  belongs to an outage, not a decision, and the net series is the basket
  mean, so one leg's verdict missing isn't a basket decision at all. Rows
  with no status (pre-`status`-column caches, and the seeded anchor bars) are
  legacy/unknown, treated the same as failed.
- **Data refresh.** `get_bars` is cache-first and never refetches, so a live
  loop updates the cache itself. The fetch chain degrades Robinhood MCP →
  Yahoo's keyless v8 chart API → absent; it never fabricates a number. In
  practice CI always lands on Yahoo, because the MCP's OAuth only completes
  inside an interactive Claude session and `ROBINHOOD_MCP_URL`/`TOKEN` are
  unset there. `refresh.py --fetch` merges fresh bars into `data/<SYM>.csv`
  (dedup by date, dropping volume-0 snapshot placeholders); a stale cache is
  used silently, `update_cache` is last-row-wins on a date collision, and a
  symbol the source can't serve is skipped with a stderr line, run still green.
- **Durable cadence.** A weekday GitHub Actions workflow (`daily-paper-run.yml` →
  `scripts/paper_cron.sh`) runs `refresh --fetch` + tick on GitHub's runners, so the
  record grows without a live laptop or Claude session. The cumulative cache and
  record (both gitignored) persist on a dedicated `paper-state` branch. One-time
  setup in `docs/paper-cron-setup.md`.
- **Self-written memory loop (`memory.py`).** The agent's education is no longer
  just the one-sentence `lessons_from_runs()` stats line — before each bar's
  decision it also reads `journal/agent_memory.md`, its own dated reflections.
  After an agent tick appends a new day, `tick_and_reflect` has the model review
  its recent decisions and realized outcomes (`recent_outcomes`) and write 3-5
  falsifiable bullet lessons, appended under a `## <date>` header. Capped at the
  most recent 40 entries. Because `journal/` persists on `paper-state`, this
  memory carries forward across CI runs for free; `run.json` records
  `memory_chars`/`reflected` per run as an audit trail of what education it got.

This record is the evidence the promotion decision (below) waits on: it is what turns
"the backtest looks good" into "it held up out-of-sample," before anyone flips
`LIVE=true`.

---

## The improvement flow, end to end

Loop C's paper-trade bake-off picks a candidate; Loop D's robust evaluator
says whether it beat baseline outside the noise band; Loop E's forward record
is the only evidence that counts. Promotion is a human pasting a config block
and, one day, flipping `LIVE=true` — every guardrail in §3 still stands
between that config and a real order. The system never auto-promotes.

---

## 5. The world model

**Status: the current code ships the seams, not the world model itself.** The
world model is a planned set of extensions, each of which plugs into a seam that
already exists — so none requires rewriting the harness.

The hook is the paper-trade loop's two Protocols (`papertrade.py`): the loop
consumes bars through a `MarketSource` and prices orders through a `FillModel`,
and knows nothing else about where either comes from. Today those are
`HistoricalSource` (real cached bars) and `CloseFill` (perfect fill at the
close). Swapping them turns real-history replay into a full world model. The
roadmap (from the design spec, in order):

1. **Synthetic price paths** — a `MarketSource` that *generates* bar frames
   instead of reading history: start with block-bootstrap of real returns,
   upgrade to GARCH/regime models. Enables Monte-Carlo robustness over thousands
   of scenarios rather than one real path. Must be validated against real-data
   statistics before its evaluations are trusted.
2. **Market impact** — a `FillModel` where the agent's own orders move the fill
   price (slippage/impact), for honest evaluation at size.
3. **Counterfactual replay** — re-run the deterministic loop from a saved state,
   overriding one decision, to branch the timeline ("what if we'd held at trade
   \#7"). Relies on the loop's determinism guarantee.
4. **Agent mental model** — a running belief state maintained inside a future
   `AgentEngine`, shipping with the LLM-through-the-loop integration.

The loop was built event-driven (not vectorized like `backtest.py`) specifically
so these can attach. `backtest.py` stays the fast ranking path; the paper-trade
loop is the slow, honest, world-model-ready path.

---

## 6. Noise reduction

Separating a real edge from noise is the *purpose* of the whole quant side, so
the mechanisms are spread across the layers on purpose — defense in depth against
fooling yourself. Collected in one place:

| Mechanism                           | Where                          | What noise it removes                                                                                                                                                                            |
| ----------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Hysteresis** (entry ≠ exit)       | `mean_reversion.py`            | avoids churning in/out around a single threshold — trade-level noise                                                                                                                             |
| **Turnover cost** (`cost_bps`)      | `backtest.py`, `papertrade.py` | charges every position flip, so a "signal" that only looks good gross gets penalized for thrashing                                                                                               |
| **Failure buckets**                 | `evaluate.py`                  | separates *where* losses concentrate (a regime) from random scatter, so you fix a cause instead of overfitting to individual losers                                                              |
| **ConvictionGate** (overlay)        | `overlay.py`                   | drops entries whose signal is below a rolling percentile of its own history — trades only the high-conviction subset, so coin-flip entries stop diluting the edge                                |
| **Robust bake-off evaluator**       | `evaluate_robust.py`           | judges a paper-trade variant by fold-Sharpe + bootstrap 95% CI + deflated Sharpe, not one number — a variant only "wins" if its CI lower bound clears baseline, so a lucky window can't crown it |
| **Fully-realized-day guard**        | `forward.py`                   | the forward record admits a day only once every universe name has settled the next bar, so a half-updated cache can't inject a thin partial-day mean that misrepresents the basket               |

The throughline: at every layer the system assumes an apparent edge is noise
until it clears a bar, and it makes the bar *higher* the more you searched.

---

## 7. Known weak points

Named here so nobody has to rediscover them.

- **The forward record is tiny.** At this daily mean, distinguishing the
  conviction gate from zero takes years of data, not weeks. No pre-committed
  keep/kill criterion is written down yet.
- **`paper-state` is the only copy** of the track record, with no backup.
  The unauthenticated Vercel trigger endpoint and the on-demand
  `research-run.yml` workflow that could append to it were both deleted.
- **The agent's failure mode is excluded, not silent, but still lossy.** A
  failed `decide_all` call marks every symbol in that batch `status="failed"`
  and holds `current_pos`; the forward record then drops that date entirely
  (§1a's exclusion rule) rather than booking it as a flat verdict. That is
  more honest than silently counting it as a real decision, but a chronic
  outage still starves the record of days without raising anywhere except a
  decisions.jsonl scan.
- **The universe is survivorship-selected**: today's mega-caps, chosen recently.
- **The world model was never built.** `MarketSource`/`FillModel` (§5) are
  the seams it would attach to; synthetic price paths, market impact and
  counterfactual replay are all unimplemented. Do not read the seams as a
  feature.
