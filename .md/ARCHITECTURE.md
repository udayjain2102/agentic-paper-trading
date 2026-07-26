# Architecture

How this system actually works today. Every module, flag and number below was
checked against the code in this commit; where the code and an older doc
disagreed, the code won.

**There is no live order path.** `runner.py`, `executor.py`, `agent.py` and
`journal.py` — the "safety funnel" older docs described as the core — were
deleted as dead code (nothing scheduled ever invoked them). What runs is a
read-only research loop: fetch bars, tick a forward paper record, render a
dashboard. `guardrails.py` and `broker.py` still exist and are tested, but
`validate_order` / `check_halted` / `MockBroker` / `McpBroker` have no
callers — see §4.

---

## 1. What runs, and when

Two GitHub Actions workflows are the only scheduled or triggered entry points.

```
tests.yml            (push, pull_request)
  └─ PYTHONPATH=src python -m pytest -q

daily-paper-run.yml  (cron "17 11 * * 1-5" UTC, + workflow_dispatch)
  ├─ pytest -q                                    (fail fast before ticking)
  ├─ scripts/paper_cron.sh
  │    ├─ git restore data/ journal/ from origin/paper-state
  │    ├─ python -m rhagent.refresh --fetch --days 10
  │    ├─ python -m rhagent.forward --cost-bps 1 --fill-mode close
  │    ├─ python -m rhagent.forward --eval-id mean_reversion_real
  │    │        --cost-bps 7 --fill-mode next_open      (non-fatal)
  │    ├─ python -m rhagent.forward --engine agent --eval-id agent
  │    │        (skipped unless NVIDIA_API_KEY is set; non-fatal)
  │    └─ rsync data/ journal/ → push origin/paper-state
  └─ scripts/make_dashboard.py --out site/index.html → Vercel (skipped if
     VERCEL_TOKEN unset)

research-run.yml     (workflow_dispatch, also fired by the unauthenticated
                      Vercel endpoint deploy_api/trigger-run.js)
  └─ python -m rhagent.papertrade --engine mean_reversion --symbols all
```

`data/` and `journal/` are gitignored on `main` and live on the `paper-state`
branch, which the cron script restores from and pushes back to. That branch is
the only copy of the track record. One-time setup: `.md/paper-cron-setup.md`.

**Three forward records run in parallel**, deliberately on different bases:

| Record dir | Cost | Fill | Why |
|---|---|---|---|
| `journal/forward/mean_reversion` | 1 bp | `close` | The original record. Pinned to its seed basis so its curve has no discontinuity. Flattering fills. |
| `journal/forward/mean_reversion_real` | 7 bp | `next_open` | The honest go-forward number: a cost and a fill you could actually get. |
| `journal/forward/agent` | config | config | The LLM engine on the same universe. |

## 2. The two decision engines (`engine.py`)

Both satisfy the same `DecisionEngine` Protocol —
`decide(symbol, history, current_pos) -> Decision` — where `Decision` carries a
`target` in `{-1, 0, +1}`, a `reason`, and an optional continuous `conviction`.
Neither proposes an order; they propose a *position*.

- **`StrategyEngine`** wraps a rule-based `Strategy` and returns
  `strat.target(history)` with `strat.signal(history).iloc[-1]` as conviction.
- **`AgentEngine`** asks an LLM (`nvidia/llama-3.3-nemotron-super-49b-v1.5` via
  NVIDIA's OpenAI-compatible API) for one JSON verdict per bar from a compact,
  lookahead-free prompt containing only `last_close`, `momentum_5d`, `vol_20d`,
  `current_pos`, and the `learn.py` lessons string. `complete(prompt) -> str` is
  an injectable seam, so tests never hit the API.

### Strategies (`strategies/`)

Every strategy implements `strategies/base.py`:

- `positions(bars) -> Series` in `{-1, 0, +1}`, obeying the **no-lookahead
  invariant**: the position at day *t* uses only data through day *t*.
- `signal(bars) -> Series` — a continuous score, higher = more bullish. This is
  what the factor/IC math evaluates and what the conviction gate thresholds.
- `target(bars) -> float` — just today's position. The base default is
  `positions(bars).iloc[-1]`; `linreg` overrides it with a single-step fit
  instead of refitting an OLS per historical day.

`REGISTRY` (`strategies/__init__.py`) holds three: `mean_reversion`,
`momentum`, `linreg`. `clamp_short` maps `-1` to `0` unless `allow_short` — the
system is long-only by default everywhere.

### The locked-in preset (`config.yaml`)

`mean_reversion`, `params: {}` (all code defaults), `overlay: conviction`,
`cost_bps: 7.0`, `fill_mode: next_open`, over a fixed 65-name mega-cap universe.

Mean reversion on `z = (close − 20d mean) / 20d std`: enter long when
`z < −1.0`, exit flat when `z ≥ 0`. The gap between the two thresholds is a
dead-band that stops a price wobbling at one level from churning the position.

The strategy takes an optional `stop` (fractional adverse move vs entry price,
re-entry blocked until `z` recovers into `[−entry, entry]`) but it is **off by
default**: on the 400-day/65-symbol cache every level tested (3–20%, plus time
stops of 5–15 bars) reduced total return, profit factor and Sharpe. Mean
reversion's losers mostly revert, so a stop realizes the loss at maximum pain.
`avg_loss > avg_win` is intrinsic here and is paid for by the high win rate.
With stops off, the only exit is reversion to `z ≥ 0` — if price keeps falling
the position is **held, not cut**. That is the thesis and also the tail risk.

### The conviction overlay (`overlay.py`)

One seam sits between the raw target and the position taken:

```
final_target = overlay.adjust(symbol, history, decision, closed_trades)
```

`closed_trades` holds only trades that closed *strictly before* today's bar, so
the no-lookahead invariant holds here too. Return `0` to veto, a fraction to
downsize, the raw target to pass through.

`build_overlay` returns exactly two: `IdentityOverlay` (`none`) and
`ConvictionGate` (`conviction`). The gate vetoes an entry unless `|conviction|`
strictly exceeds the 60th percentile (`pctile=0.60`) of that symbol's own past
`|conviction|` over the trailing 120 bars (`window=120`), and passes everything
through during the cold start before 120 bars exist. `apply_conviction` is its
vectorized twin, used on the forward path. BucketFilter and WinProbGate lost
the 2026-07-13/14 bake-off and were deleted (`.md/AUDIT-2026-07-17.md`); they
live in git history.

## 3. Return accounting (`backtest.py`)

`net_returns(bars, positions, cost_bps, fill)` is the single place a position
series becomes a return series — the forward record, the papertrade evaluator
and the vectorized backtest all route through it, so their numbers agree.

```
turnover(t) = |Δposition(t)|          (turnover[0] = |position[0]|)
cost(t)     = turnover(t) · cost_bps/1e4
net(t)      = position(t) · fwd(t) − cost(t)
```

`fill` decides `fwd(t)`, and this is the single biggest honesty knob:

- `close` — `close[t] → close[t+1]`. Assumes you can trade at the very close
  that produced the signal. You cannot. It flatters dip-buying specifically.
- `next_open` — on any day the position *changes*, `open[t+1] → close[t+1]`
  instead, skipping the overnight gap you could not have traded. A day of
  unchanged position still earns the plain close-to-close move.

`result_from_returns` turns that into `BacktestResult`: equity curve
(`cumprod(1+net)`), `total_return`, `sharpe` (`mean/std · √252`),
`max_drawdown`, `hit_rate`.

## 4. What the safety code guards today: nothing

`guardrails.py` is pure (no I/O, no state) and exhaustively tested on every
rejection path. `broker.py` has `MockBroker` and `McpBroker`. Both are real and
both are currently **unreached**:

- The only import of `guardrails` anywhere is `config.py` pulling in `Limits` to
  parse the `limits:` block of `config.yaml`. Nothing calls `validate_order` or
  `check_halted`.
- The only import of `broker` is `data.py` / `refresh.py` borrowing the
  `_structured` helper to unpack MCP tool results. Nothing places an order.
- `mcp_session.py` is used only for *data* fetch, never for orders.
- The `HALT` file and the daily-loss kill switch are implemented in
  `check_halted`, which nobody calls. `LIVE=true` gates nothing.

The limits in `config.yaml` (per-trade $250, total deployed $2,000, ≤2 new
positions/run, ≤5 orders/run, $200 daily-loss kill switch) are therefore the
**contract any future order path must satisfy**, not protection in force. If you
reintroduce order placement, route it through `guardrails.validate_order` before
`broker.place_order`, call `check_halted` at the top of the run, and re-verify
the caps end to end.

## 5. Data (`data.py`, `refresh.py`)

`get_bars` is **cache-first and never refetches** an existing `data/<SYM>.csv`.
The fetch chain degrades Robinhood MCP → Yahoo → absent; it never degrades to a
made-up number. In practice CI always uses Yahoo's keyless v8 chart API, because
the MCP's OAuth only completes inside an interactive Claude session and the
`ROBINHOOD_MCP_URL`/`ROBINHOOD_MCP_TOKEN` secrets are unset.

`refresh.py --fetch` merges fresh bars into the cache, deduping by date and
dropping volume-0 snapshot placeholders. Known sharp edges, all deliberate:

- A stale cache is used silently — `get_bars` never notices it is old.
- `update_cache` is last-row-wins on a date collision, so one bad bar
  overwrites a good one.
- A symbol the source can't serve is skipped with a stderr line; the run
  continues green.

## 6. The forward record (`forward.py`)

`tick()` computes the configured engine's net return for each *newly realized*
day and appends it to `journal/forward/<eval_id>/`, in the format `evaluate.py`
and the dashboard already read. It is **anchored at first run** — the curve is
the go-forward period, not backfilled history — and idempotent within a day.

**Fully-realized-day guard.** `net_returns` records a day's return at its entry
date, so a day is trustworthy only once the next bar exists for *every* universe
name. The tick appends a day only when the whole basket has settled it. The
corollary is a real failure mode: one chronically-missing name freezes the
record forever, with no alert.

`tick_and_reflect` additionally runs the agent's memory loop: after an agent
tick appends a day, the model reviews its recent decisions and realized outcomes
(`memory.recent_outcomes`) and appends dated, falsifiable bullet lessons to
`journal/agent_memory.md`, capped at the 40 most recent entries
(`memory.MAX_ENTRIES`). That file is read back into the next run's prompt.
Because `journal/` persists on `paper-state`, the memory carries across CI runs.

`--report` prints the record; `_report_decision_quality` scores the engine's
decisions against 1-day and 5-day forward moves.

## 7. Bar-by-bar paper trading and grading

### `papertrade.py`

`PaperTrader` steps a `DecisionEngine` through history one bar at a time,
turning each position change into a discrete, ID-stamped trade in an
append-only ledger under `journal/papertrade/<run_id>/`. Two Protocols keep the
loop swappable: bars arrive through a `MarketSource` (`HistoricalSource`) and
orders price through a `FillModel` (`CloseFill` or `NextOpenFill`). At entry
each trade records cheap lookahead-free features (`features.entry_features`:
20-day vol, overnight gap, 5-day trend).

### `evaluate.py` — the judge

- **`aggregate`** — win rate, avg win/loss, profit factor, total return, Sharpe,
  max drawdown, avg holding period. Return metrics reuse
  `backtest.result_from_returns`, so paper-trade and vectorized numbers agree.
- **`failure_buckets`** — attributes losses across vol regime, gap direction,
  holding length, symbol and side, ranked by loss share. This is the feedback
  signal: it names the regime where the edge dies.
- **`spy_benchmark`** — buy-and-hold SPY over the same dates, so a run's return
  is reported against something rather than against zero.
- **`compare_runs`** — ranks every run side by side (`papertrade compare`).

### `evaluate_robust.py` — the bake-off judge

These strategies live in the noise, so a variant is not judged on one Sharpe:
`fold_sharpe` (per-fold Sharpe over rolling 60-bar windows, step 30),
`bootstrap_sharpe_ci` (1000 resamples of the per-bar net returns), and
`deflated_sharpe` penalizing for the number of variants tried. A variant
"beats baseline" only if its CI lower bound clears the *same engine+universe*
baseline's Sharpe. Empirically the conviction gate lifted point Sharpe several
fold but its CI still spans zero — nothing clears the noise band, which is the
honest outcome at this data scale.

## 8. The offline IC research tools — candidate generation, not the judge

`factor/`, `search/` and `gate/` are standalone CLIs
(`python -m rhagent.factor|search|gate`). **No production code imports any of
them.** They have never selected a shipped strategy: the real-data verdict was
`viable: 0` for every strategy, and the config preset is the code default. They
narrow candidates before a bake-off; they do not decide what ships.

### `factor/ic.py` — Information Coefficient

`forward_returns(close, h) = close.shift(-h)/close − 1`. Rank-IC on one day is
the Spearman rank correlation between that day's signal cross-section and its
forward returns. Ranking removes the equal-weighted cross-sectional mean, so no
separate demeaning is needed. **It is not market-neutral** — a signal that
merely proxies market beta can still earn positive rank-IC.

- `icir(ic) = mean(ic)/std(ic)` — consistency of the edge, the primary ranking
  metric. Below 0.3 is flagged as likely noise.
- `ic_decay` / `half_life` — mean IC at horizons `(1, 5, 10, 20, 50)`; the
  half-life is the first horizon where `|IC|` falls to half its 1-day value.

Horizon-*h* ICs overlap and are autocorrelated, so the effective independent
sample is roughly `days/h` — the ICIR confidence band overstates the evidence,
which `factor/__main__.py` says out loud.

### `gate/stats.py` — multiple-testing correction

Pure implementations, no scipy: `norm_cdf` via `math.erfc`, `norm_ppf` via
Acklam's rational approximation.

- **`bonferroni`** — `t = |ICIR|·√n_eff`, two-sided p-value, required to beat
  `α / n_tested`. The more configs tried, the higher the bar.
- **`deflated_sharpe`** (Bailey & López de Prado) — given *N* trials and the
  variance of ICIRs across them, how probable is an ICIR this high under the
  null? Corrects for trial count and for skew/kurtosis of the return stream.
  The expected-maximum-under-null term is
  `sr0 = √var_trials · [(1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]`, γ = Euler–
  Mascheroni.

### `factor/split.py` — the locked split

`oos_cutoff` fixes the out-of-sample slice up front (default last 25% of dates);
it must never be read during development or search. `in_sample_mask` trims the
boundary so no in-sample day's *h*-day forward window peeks across the cutoff.

### `search/` — coarse-to-fine, strictly in-sample

`run_search` runs up to `max_rounds`, and each round rewrites its own grid:
score the Cartesian product by ICIR → apply gates → keep top-k →
`refine_grids` inserts midpoints between surviving values → repeat. It stops
when a round yields no survivors or the best ICIR stops improving. Four
survival `Gates`: ICIR floor (0.3), half-life floor (5 days), sign stability
(mean IC positive in *every* in-sample sub-period), and parameter robustness
(a config's grid `neighbors` must also clear the ICIR floor). It reports
`n_tested`, which feeds the correction above.

### `gate/` — the out-of-sample gate

The one place the locked OOS slice is read. `evaluate_oos` recomputes ICIR and
half-life there; `verdict` requires all five of: `icir_holds` (OOS ICIR positive
and ≥50% of in-sample), `decay_holds`, Bonferroni pass, deflated-Sharpe pass,
and the floors. Only then is a config `viable`.

## 9. Noise reduction, collected

| Mechanism | Where | What noise it removes |
|---|---|---|
| Rank-IC (Spearman, not Pearson) | `factor/ic.py` | outlier days, monotone rescaling, the common cross-sectional mean |
| ICIR over raw IC | `factor/ic.py` | a lucky-day spike; scores consistency |
| Overlapping-window caveat | `factor/__main__.py` | an overstated denominator |
| Sign-stability gate | `search/loop.py` | edges that flip sign between sub-periods |
| Robustness gate | `search/loop.py` | a lone lucky setting surrounded by junk |
| Half-life floor | `search/`, `gate/` | fast-decaying microstructure noise |
| OOS ICIR-retention | `gate/oos.py` | an edge that dies on unseen data |
| Bonferroni + deflated Sharpe | `gate/stats.py` | significance manufactured by searching |
| Hysteresis (entry ≠ exit) | `mean_reversion.py` | churn around one threshold |
| Turnover cost (`cost_bps`) | `backtest.py` | a signal that only looks good gross |
| `next_open` fill | `backtest.py` | the untradable same-close fill |
| Failure buckets | `evaluate.py` | random scatter vs a real losing regime |
| SPY benchmark | `evaluate.py` | a return quoted against zero instead of the market |
| ConvictionGate | `overlay.py` | coin-flip entries diluting the edge |
| Robust bake-off (fold + bootstrap CI + DSR) | `evaluate_robust.py` | one lucky window crowning a variant |
| Fully-realized-day guard | `forward.py` | a thin partial-day mean from a half-updated cache |

The throughline: assume an apparent edge is noise until it clears a bar, and
raise the bar the more you searched.

## 10. Known weak points

Named here so nobody has to rediscover them.

- **The forward record is tiny.** At this daily mean, distinguishing the
  conviction gate from zero takes years of data, not weeks. No pre-committed
  keep/kill criterion is written down yet.
- **`paper-state` is the only copy** of the track record, with no backup, and
  `deploy_api/trigger-run.js` can append to the run archive unauthenticated.
- **The agent's failure mode is silent.** `AgentEngine.decide` catches parse and
  API failures and records "held" — a total model outage is indistinguishable
  from a genuine flat verdict in the record.
- **The universe is survivorship-selected**: today's mega-caps, chosen recently.
- **`--overlay bucket|winprob`** is still an accepted `papertrade` CLI choice
  but `build_overlay` raises `KeyError` on both — leftovers from the deleted
  overlays.
- **The world model was never built.** `MarketSource`/`FillModel` are the seams
  it would attach to; synthetic price paths, market impact and counterfactual
  replay are all unimplemented. Do not read the seams as a feature.
