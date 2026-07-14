# Architecture

How this system decides what to trade, the math it uses to judge a strategy,
and the loop by which it improves — all behind hard, code-enforced safety rails.

The project has **two decision brains** that share one safety funnel:

1. **The LLM agent** — Claude/Nemotron reasons over the live account each cron
   tick and proposes orders.
2. **The quant strategy pipeline** — rule-based strategies that are searched,
   statistically gated, and paper-traded offline, then run live in "strategy mode".

Both emit the same thing — proposed `(symbol, side, notional)` orders — and both
are forced through the identical `OrderExecutor → guardrails → broker` path. The
model decides; **the code decides what's allowed.**

---

## 1. The model

### 1a. The LLM decision agent (`agent.py`)

A **manual agentic loop** (deliberately *not* an auto tool-runner) served by
NVIDIA's OpenAI-compatible API (`nvidia/llama-3.3-nemotron-super-49b-v1.5`).

Each run seeds the model with a system prompt and one instruction — *"Review the
account and decide whether to trade right now"* — and exposes three tools:

| Tool | What it does |
|------|--------------|
| `get_account` | buying power, deployed capital, positions, realized P&L today |
| `get_quote` | latest price for a symbol |
| `place_order` | propose `(symbol, side, notional_usd)` |

The loop runs up to `max_turns` turns: call the model → if it emitted tool calls,
dispatch each and feed results back → repeat until the model stops calling tools.
The reason it is *manual* is the choke point: `place_order` is never sent to the
broker. It is dispatched to `OrderExecutor`, which runs the guardrails first
(§3). The model can reason however it likes, but a hard cap is enforced in code
it cannot talk past.

`run_scripted_session` is a no-API stand-in that walks the same dispatch path
with a fixed plan (one order that clears the per-trade cap, one that's rejected)
so the full pipeline can be exercised without an API key.

### 1b. The rule-based strategies (`strategies/`)

Every strategy implements one contract (`strategies/base.py`):

- `positions(bars) -> Series` in `{-1, 0, +1}` — the target position, obeying the
  **no-lookahead invariant**: the position at day *t* uses only data up to and
  including day *t*.
- `signal(bars) -> Series` — a *continuous* score where higher = more bullish on
  the forward return. This is what the factor/IC math evaluates.
- `target(bars) -> float` — just *today's* position (the last value). The base
  default is `positions(bars).iloc[-1]`, but a strategy whose last value is
  independent of the earlier ones overrides it with a cheaper single-step
  computation. `linreg` does: instead of refitting an OLS for every past day only
  to keep the last, it fits **one** OLS for the current bar — bit-identical
  output, ~76× faster in the bar-by-bar loop, verified by equivalence checks.
  `StrategyEngine.decide` (§4C) calls `target`, so the paper-trade loop pays the
  single-step cost, not the full-series cost, every bar.

| Strategy | Signal | Position rule |
|----------|--------|---------------|
| **mean_reversion** | `-z`, where `z = (close − rollmean)/rollstd` over `lookback` | long when `z < −entry`, exit to flat when `z ≥ −exit` (hysteresis avoids churn) |
| **momentum** | `close.pct_change(lookback)` (trailing return) | `sign(trailing return)` |
| **linreg** | rolling-OLS prediction of next-day return | long when prediction > 0 |
| **pairs** | z-score of `log(A) − log(B)` spread | long the cheap leg / short the rich leg when `|z| > entry` |

`linreg` is the only one that fits parameters: at each day *t* it runs
`np.linalg.lstsq` on features `[1, ret_lag1, ret_lag2, ma_ratio]` against next-day
return, trained only on rows whose target is already realized (strictly before
*t*), then predicts day *t*. Expanding window, no lookahead.

`clamp_short` maps any `-1` to `0` unless shorting is explicitly enabled. **The
system is long-only: shorting is disabled everywhere by default** (every strategy
*and* `AgentEngine` default `allow_short=False`, and the paper-trade CLI exposes
no short toggle). The `allow_short` parameter still exists in the strategy
classes, so the capability can be re-enabled deliberately in code/config, but no
normal run shorts. A run can sweep the whole cached universe at once with
`--symbols all`.

### 1c. The trade setup — the live preset

The shipped configuration (`config.yaml`) is **`mean_reversion` with `params: {}`
(all defaults), the `conviction` overlay, long-only, over the 65-name universe.**
The empty `params` means every knob below is the strategy default in code, not a
tuned value.

**Entry / exit (`strategies/mean_reversion.py`).** A trade is driven entirely by the
z-score `z = (close − 20-day mean) / 20-day std`:

| Knob | Value | Meaning |
|------|-------|---------|
| `lookback` | 20 | rolling window for the z-score mean/std |
| `entry` | 1.0 | **enter long** when `z < −1.0` — price ≥ 1σ below its own 20-day mean (a "statistically cheap" dip) |
| `exit` | 0.0 | **exit to flat** when `z ≥ 0` — price has reverted back to the mean |
| hysteresis | −1.0 enter / 0.0 exit | the gap between the two thresholds is a dead-band, so price wobbling around one level doesn't churn the position |
| direction | long-only | short signals are clamped to flat (`clamp_short`); the position is `0` or `+1` |

**There is no per-trade price stop-loss.** Exit is purely mean-reversion: a position
is held until `z` climbs back to `0`. If price keeps falling (`z` goes *more*
negative) the position is **held, not cut** — that is the strategy's thesis and also
its tail risk. The only loss backstop is the portfolio-level kill switch below, and
it exists only on the live-order path, not in the paper forward record.

**Conviction gate (`overlay.py`, applied on the eval + forward path).** Of the raw
long entries, only the higher-conviction ones are taken: an entry is vetoed unless
`|signal|` (= `−z`) strictly exceeds the **60th percentile** (`pctile = 0.60`) of that
symbol's own `|signal|` over the trailing **120 bars** (`window = 120`).

**Costs / sizing (paper eval).** Turnover (`|Δposition|`) is charged `cost_bps = 1.0`
(1 bp); paper P&L is scaled to a `notional` of \$10,000; the forward record equal-
weights the per-name net returns across the universe.

**Live-execution guardrails (`guardrails.py`, §3 — apply only when actually placing
orders, `LIVE=true`; they do not touch the paper record).** These are the real
"when to stop" limits:

| Guardrail | Value | Effect |
|-----------|-------|--------|
| `per_trade_max_usd` | \$250 | max dollars committed to any single order |
| `total_deployed_max_usd` | \$2,000 | new buys rejected if total deployed would exceed this |
| `max_new_positions_per_run` | 2 | cap on newly-opened symbols per run |
| `max_orders_per_run` | 5 | cap on orders per run |
| `daily_loss_limit_usd` | \$200 | **kill switch** — if realized P&L for the day ≤ −\$200, the run halts and places nothing |
| `HALT` file | present → halt | operator manual stop |

These are deliberately conservative starter values (`config.yaml` says "tighten
before going live").

---

## 2. The math

This is how the system judges whether a signal is *real* rather than lucky. The
core metric is **cross-sectional Information Coefficient (IC)**, not raw P&L.

### 2a. Information Coefficient (`factor/ic.py`)

For a given forward horizon *h*, the forward return is
`forward_returns = close.shift(-h)/close − 1`.

**Rank-IC on one day** is the Spearman rank correlation between that day's signal
cross-section and its forward returns:

```
rank_ic(t) = corr( rank(signal_t across names), rank(fwd_return_t across names) )
```

Ranking makes it invariant to a common additive shift applied to every name that
day — it removes the equal-weighted cross-sectional mean, so no separate demeaning
step is needed. **Caveat (documented in the code):** rank-IC is *not* market-
neutral. It does not remove differential beta, so a signal that merely proxies
market beta can still earn a positive rank-IC. True beta-neutralization is
deferred.

Two summary statistics fall out of the daily IC series `ic(t)`:

- **ICIR** (Information Coefficient Information Ratio) — the *consistency* of the
  edge: `ICIR = mean(ic) / std(ic)`. This, not total return, is the primary
  ranking and gating metric.
- **IC decay / half-life** — mean IC computed at horizons `(1, 5, 10, 20, 50)`.
  The **half-life** is the first horizon where `|IC|` falls to half its 1-day
  value. A fast-decaying signal is fragile; a slow one is tradeable.

### 2b. Backtest metrics (`backtest.py`)

The vectorized engine turns a positions series into net returns and a scorecard.
The position held on day *t* earns the *t → t+1* return; the final day (no
forward return) is dropped. Turnover (`|Δposition|`) is charged `cost_bps`
basis points.

```
net(t)   = position(t) · fwd_return(t) − turnover(t)·cost_bps/1e4
equity   = cumprod(1 + net)
sharpe   = mean(net)/std(net) · √252          (annualized)
max_dd   = min(equity/cummax(equity) − 1)
hit_rate = fraction of nonzero-position days that are profitable
```

Strategy ranking in `compare.py` uses **total_return**; the rest are context.

### 2c. Multiple-testing correction (`gate/stats.py`)

The danger: search enough configs and one *will* look great by luck. Two
corrections, both implemented pure (no scipy — just `math.erfc` for the normal
CDF and Acklam's rational approximation for its inverse):

- **Bonferroni** — turn ICIR into a t-stat `t = |ICIR|·√n_eff`, get its two-sided
  p-value, and require it to beat `α / n_tested`. The more configs tried, the
  higher the bar every survivor must clear.
- **Deflated Sharpe Ratio** (Bailey & López de Prado) — asks: given that *N*
  configs were tried, and given the *variance of the ICIRs across those trials*,
  how probable is it that an ICIR this high is real rather than the luckiest
  draw? It corrects for both the number of trials and the non-normality
  (skew/kurtosis) of the return stream, returning a probability that must exceed
  `dsr_threshold` (0.95).

The DSR's expected-maximum-under-null term is
`sr0 = √var_trials · [(1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]`, where γ is the
Euler–Mascheroni constant — the expected maximum of *N* draws from the null.

### 2d. The locked split (`factor/split.py`)

The out-of-sample (OOS) slice is fixed up front (default last 25% of dates) and
**must never be read during signal development or search** — it is reserved for
the final gate. `in_sample_mask` additionally trims the boundary so no in-sample
day's *h*-day forward-return window peeks across the cutoff.

---

## 3. How it decides (the safety funnel)

Every order — LLM or strategy — passes through the same gauntlet. `guardrails.py`
is pure, does no I/O, holds no state, and is exhaustively tested on every
rejection path.

```
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

One cron tick (`runner.py`): load config → `check_halted` → build executor →
route to LLM agent *or* strategy mode (`STRATEGY_MODE=true`) → journal.

---

## 4. How it improves

Improvement happens **offline, on the quant side**, in three nested loops that
get progressively stricter about "is this edge real?". Nothing here can touch
live trading until it survives all of them and is manually pasted into
`config.yaml`.

### Loop A — Coarse-to-fine parameter search (`search/`) — *the iteration loop*

Strictly in-sample. This is the literal iteration loop: `run_search` runs up to
`max_rounds` rounds, and **each round rewrites its own search grid based on the
previous round's survivors**:

```
round 0:  score the coarse Cartesian product → apply gates → keep top-k
round r:  refine_grids(around survivors) → score the NEW configs
          → apply gates → keep top-k
stop when: a round produces no survivors, OR the best ICIR stops improving
```

Each strategy has a small parameter grid (`search/space.py`). The loop scores the
product by ICIR, then **refines the grid around the survivors** (`refine_grids`
inserts midpoints between surviving values) so each successive round concentrates
its samples where the edge appears — coarse first, then progressively finer. The
`prev_best.icir` check is the convergence test that ends the iteration.

Four **survival gates** (`search/loop.py`), all on by default:

1. **ICIR floor** — `icir ≥ 0.3`.
2. **Half-life floor** — the edge must persist `≥ 5` days.
3. **Sign stability** — mean IC must be positive in *every* in-sample sub-period
   (an edge that flips sign mid-history is noise).
4. **Parameter robustness** — a config's grid *neighbors* must also clear the
   ICIR floor. A lone lucky setting surrounded by junk cannot survive.

The loop reports `n_tested` — the count of distinct configs scored — which feeds
the multiple-testing correction downstream.

### Loop B — The out-of-sample gate (`gate/`)

The one place the locked OOS slice is read. For each in-sample survivor:

1. Recompute ICIR and half-life on the **OOS** slice.
2. **ICIR-retention** — OOS ICIR must be positive and ≥ 50% of the in-sample
   ICIR (edge held up out of sample).
3. **Decay holds** — OOS half-life ≥ floor.
4. **Bonferroni** pass (§2c), penalized by `n_tested`.
5. **Deflated Sharpe** pass, using the ICIR variance across *all* scored configs.

Only a config that clears **all five** is `viable`. The gate prints a verdict
table with a `reason` for each rejection. (Real-data verdict so far: viable = 0,
but real — the honest outcome of a strict gate on limited data.)

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
- **`compare` command** — rank every paper-trade run side by side. The same
  numbers render as a self-contained HTML dashboard (`scripts/make_dashboard.py`):
  an all-runs index (per run: trades, won/lost counts, net P&L, total return,
  Sharpe, max DD) where clicking a run id opens *only* that run's full detail
  (scorecard, equity curve, ledger, failure buckets) — native CSS `:target`, no
  JavaScript.

### Loop D — Learning from losses: decision overlays (`overlay.py`, `evaluate_robust.py`)

Loops A–C *judge* strategies; Loop D lets one **adapt to its own realized
losses** without retraining. A single seam sits between the strategy's raw target
and the position actually taken — `StrategyEngine.decide` produces a target and a
per-bar `conviction` (the continuous `signal()` value), and an `Overlay.adjust`
gets the last say:

```
final_target = overlay.adjust(symbol, history, decision, closed_trades)
```

`closed_trades` is the ledger of trades that closed **strictly before** today's
bar — the seam snapshots it once per bar, before any of that bar's own closes, so
the same **no-lookahead invariant** holds as everywhere else. Return `0` to veto,
a fraction to down-size, or the raw target to pass through. The baseline is an
`IdentityOverlay` (a `--overlay none` run is byte-identical to no overlay at all).
Three overlays exist as interchangeable, comparable variants:

- **ConvictionGate** — vetoes entries whose `|conviction|` is below a rolling
  percentile of that symbol's own past convictions (trade-level noise filter).
- **BucketFilter** — vetoes/down-sizes entries whose setup bucket (the same
  vol/gap/side buckets as the failure analysis) has been the worst loser among
  closed trades so far. Size is snapped to coarse levels to avoid per-bar churn.
- **WinProbGate** — a numpy-logit (IRLS) win-probability model over closed-trade
  features that vetoes entries below a probability threshold; inert at the default
  ~68% base win rate, so it only bites when the threshold is raised. **ParamTune**
  (a walk-forward parameter re-fit) remains planned — the seam is built for it.

Because these barely-profitable strategies live in the noise, the bake-off is
judged by a **robust evaluator** (`evaluate_robust.py`), not a single Sharpe:
per-fold Sharpe across rolling windows, a **bootstrap 95% CI** on the per-bar net
returns, and a **deflated Sharpe** that penalizes for the number of variants
tried (§2c, same math, applied to realized paper-trade returns instead of ICIR).
A variant "beats baseline" only if its CI lower bound clears the *same
engine+universe* baseline's Sharpe. This renders as a bake-off panel on the
dashboard. Empirically so far: the conviction gate lifts point Sharpe ~5×
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
- **Data refresh.** `get_bars` is cache-first and never refetches, so a live loop
  updates the cache itself. `refresh.py` merges fresh MCP historicals into
  `data/<SYM>.csv` (dedup by date, dropping volume-0 snapshot placeholders) two ways:
  a payload piped in from Claude's interactive MCP session, or `--fetch`, which pulls
  the whole universe headlessly over the MCP (`ROBINHOOD_MCP_URL`/`TOKEN`).
- **Durable cadence.** A weekday GitHub Actions workflow (`daily-paper-run.yml` →
  `scripts/paper_cron.sh`) runs `refresh --fetch` + tick on GitHub's runners, so the
  record grows without a live laptop or Claude session. The cumulative cache and
  record (both gitignored) persist on a dedicated `paper-state` branch. One-time
  setup in `docs/paper-cron-setup.md`.

This record is the evidence the promotion decision (below) waits on: it is what turns
"the backtest looks good" into "it held up out-of-sample," before anyone flips
`LIVE=true`.

---

## The improvement flow, end to end

```
   search (in-sample)          gate (locked OOS)         paper-trade + failure buckets
  ┌──────────────────┐       ┌──────────────────┐       ┌───────────────────────────┐
  │ ICIR ranking     │       │ ICIR retention   │       │ per-trade ledger          │
  │ 4 survival gates │  ───► │ Bonferroni       │  ───► │ aggregate scorecard       │
  │ coarse→fine grid │       │ Deflated Sharpe  │       │ losses by regime bucket   │
  └──────────────────┘       └──────────────────┘       └───────────────────────────┘
          │                          │                             │
    survivors + n_tested      viable configs              "where the edge dies"
                                                                  │
                                                                  ▼
                                              manual: paste winner into config.yaml,
                                              run live via STRATEGY_MODE=true —
                                              through the SAME guardrails as the LLM.
```

The system does **not** auto-promote a strategy to live trading. Every gate can
say "nothing is viable," and the honest answer is usually exactly that.
Promotion is a human pasting a config block and flipping `LIVE=true` — every
guardrail in §3 still stands between that config and a real order.

---

## 5. The world model

**Status: the current code ships the *seams*, not the world model itself.** The
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
   #7"). Relies on the loop's determinism guarantee.
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

| Mechanism | Where | What noise it removes |
|-----------|-------|-----------------------|
| **Rank-IC** (Spearman, not Pearson) | `factor/ic.py` | rank-based → immune to outlier days and monotone rescaling; ranking removes the common cross-sectional mean each day |
| **ICIR over raw IC** | `factor/ic.py` | `mean/std` scores *consistency*, not a lucky-day spike; the `< 0.3 = "likely noise"` band names it outright (`factor/__main__.py`) |
| **Overlapping-window caveat** | `factor/__main__.py` | flags that horizon-h ICs are autocorrelated, so the effective independent sample is ~`days/h` and the ICIR band *overstates* evidence — honest denominator |
| **Sign-stability gate** | `search/loop.py` | rejects edges that flip sign between in-sample sub-periods (real edge is persistent; noise wanders) |
| **Robustness gate** | `search/loop.py` | a config's grid neighbors must also pass — kills lone lucky settings surrounded by junk |
| **Half-life floor** | `search`, `gate` | rejects fast-decaying signals that are mostly microstructure noise |
| **OOS ICIR-retention** | `gate/oos.py` | edge must survive on data it was never fit to, at ≥50% strength |
| **Bonferroni + Deflated Sharpe** | `gate/stats.py` | the core statistical noise filter: penalize every survivor by *how many configs were tried*, so search can't manufacture significance |
| **Hysteresis** (entry ≠ exit) | `mean_reversion.py` | avoids churning in/out around a single threshold — trade-level noise |
| **Turnover cost** (`cost_bps`) | `backtest.py`, `papertrade.py` | charges every position flip, so a "signal" that only looks good gross gets penalized for thrashing |
| **Failure buckets** | `evaluate.py` | separates *where* losses concentrate (a regime) from random scatter, so you fix a cause instead of overfitting to individual losers |
| **ConvictionGate** (overlay) | `overlay.py` | drops entries whose signal is below a rolling percentile of its own history — trades only the high-conviction subset, so coin-flip entries stop diluting the edge |
| **Robust bake-off evaluator** | `evaluate_robust.py` | judges a paper-trade variant by fold-Sharpe + bootstrap 95% CI + deflated Sharpe, not one number — a variant only "wins" if its CI lower bound clears baseline, so a lucky window can't crown it |
| **Fully-realized-day guard** | `forward.py` | the forward record admits a day only once every universe name has settled the next bar, so a half-updated cache can't inject a thin partial-day mean that misrepresents the basket |

The throughline: at every layer the system assumes an apparent edge is noise
until it clears a bar, and it makes the bar *higher* the more you searched.
