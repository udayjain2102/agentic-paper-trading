# Cross-Sectional IC/ICIR Evaluator — Design

Date: 2026-07-12
Status: Approved (brainstorming)

Sub-project 1 of 3 in the quant strategy-search framework. The other two —
the search loop, and the out-of-sample gate with multiple-testing correction —
get their own specs and build on this one.

## Problem

We want to know whether a strategy's signal has a *real, consistent* edge or is
just fitting noise. The standard quant answer is the Information Coefficient
(IC): the cross-sectional correlation, each period, between a signal and the
forward returns it is supposed to predict, and its consistency ICIR =
mean(IC)/std(IC). Our existing strategies emit discrete `{-1,0,+1}` positions,
which cannot be correlated meaningfully, and we only trade 2 symbols, on which a
cross-sectional correlation is undefined. This sub-project builds the missing
foundation: a continuous per-name signal, a broad universe, and a correct
cross-sectional IC/ICIR + decay computation, with the out-of-sample slice locked
away from the start.

## Goals

- Add a continuous `signal(bars) -> Series` to each strategy (higher = more
  bullish on forward return), without disturbing the existing `positions()`
  path or the trade harness.
- Load a ~60-name large-cap universe of daily bars, aligned to a common
  calendar, and compute forward returns.
- Lock an out-of-sample slice (most recent ~25% by date) that this sub-project
  never reads — reserved for the sub-project-3 gate.
- Compute, on in-sample data only: cross-sectional rank-IC per day,
  market-neutralized; ICIR; and the IC decay curve + half-life over horizons
  {1, 5, 10, 20, 50}.
- A CLI that prints a strategy's in-sample ICIR (with interpretation bands) and
  decay curve.

## Non-goals

- The search loop (generate/refine variants) — sub-project 2.
- The OOS gate and multiple-testing correction — sub-project 3. This
  sub-project only *reserves* the OOS slice; it does not test on it.
- Changing `positions()`, `backtest.py`, `papertrade.py`, or the trade harness.
- Portfolio construction / P&L from signals. IC measures predictive skill, not
  returns; the harness already handles P&L.

## Architecture

New package `src/rhagent/factor/`. Nothing existing is modified except adding a
`signal()` method to `strategies/base.py` and the three strategy classes.

```
universe.py  ── symbol list + aligned bar/forward-return panels
signals.py   ── build a [dates × symbols] signal panel from a Strategy
split.py     ── the locked in-sample / out-of-sample date split
ic.py        ── cross-sectional rank-IC, ICIR, decay/half-life  (pure)
__main__.py  ── CLI: evaluate a strategy's in-sample ICIR + decay
```

### 1. Signal contract — `strategies/base.py` + 3 strategies

Add to `Strategy`:

```python
def signal(self, bars: pd.DataFrame) -> pd.Series:
    """Continuous score aligned to bars.index; higher = more bullish on the
    forward return. No lookahead: value at day t uses only bars up to t.
    Default: NotImplementedError (subclasses that support IC override it)."""
    raise NotImplementedError
```

Implementations (the continuous value *behind* each strategy's existing
discrete position — same computation, returned before the sign/clamp):

- **MeanReversion**: `signal = -z` where `z = (close - roll_mean)/roll_std` over
  `lookback`. Cheap dips (z ≪ 0) score high. Warmup NaNs preserved (dropped per
  day in IC).
- **Momentum**: `signal = close.pct_change(lookback)` — the trailing return whose
  sign the strategy already trades.
- **LinReg**: `signal = pred`, the raw rolling-OLS predicted next-day return
  (same fit as `positions()`, returning `pred` instead of `sign(pred)`). Days
  before `min_train` are NaN.

`positions()` is untouched. `signal()` returns a float Series indexed like
`bars`, with NaN during warmup.

### 2. Universe — `universe.py`

- `UNIVERSE`: a fixed list of ~60 liquid large-cap **individual stocks** (S&P 100
  subset; ETFs like SPY excluded — an ETF is the market, not a cross-sectional
  member). Concrete list is committed in this module.
- `load_universe(symbols, start, end, cache_dir="data", min_bars=60) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]`:
  fetch via `data.get_bars` (batched MCP call, cache-first), drop symbols with
  fewer than `min_bars` rows, and return `(bars_by_symbol, close_panel)` where
  `close_panel` is the `[dates × symbols]` close DataFrame inner-joined to the
  common trading calendar. `bars_by_symbol` is the per-symbol OHLCV frames that
  `signal_panel` feeds to `strat.signal`.
- `forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame`: `close.shift(-h)
  / close - 1`, indexed at t (the return realized from t to t+h).

### 2b. Signal panel — `signals.py`

- `signal_panel(strat: Strategy, bars_by_symbol: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame`:
  call `strat.signal(bars)` for each symbol, reindex each result to `index`, and
  assemble a `[dates × symbols]` signal panel aligned to `close_panel`. A symbol
  whose `signal()` is all-NaN over `index` is kept (its NaNs drop out per-day in
  IC). Pure over its inputs; no I/O.

### 3. Locked split — `split.py`

- `oos_cutoff(dates: pd.DatetimeIndex, oos_frac: float = 0.25) -> pd.Timestamp`:
  the date at the `1 - oos_frac` quantile of the sorted unique dates.
- `in_sample_mask(dates, cutoff, horizon) -> pd.Series`: True where
  `date < cutoff` **and** `date + horizon bars < cutoff` — the second clause
  prevents an in-sample day's forward return from peeking across the boundary
  into OOS. IC in this sub-project is computed strictly under this mask.
- The OOS slice (`date >= cutoff`) is never read here. A module constant
  documents that reading it is reserved for sub-project 3.

### 4. IC / ICIR / decay — `ic.py` (pure, no I/O)

Locked definitions (the tests encode these):

- **Rank-IC one period.** For day t and horizon h: take names present in both
  the signal row `S[t, :]` and the forward-return row `R[t, :]`. Require at
  least `min_names` (default 10) such names, else IC_t is NaN (excluded).
  **Market-neutralize** the forward returns by subtracting the cross-sectional
  mean across names that day. Rank both vectors (`pandas.rank`) and take the
  Pearson correlation of the ranks — this is Spearman rank-IC, computed without
  scipy.
- **IC series.** `ic_series(signal_panel, close_panel, h, min_names) -> pd.Series`
  of IC_t over all valid days (NaN days dropped).
- **ICIR.** `mean(IC) / std(IC)` over the series (population std; 0.0 if std is
  0 or the series is empty). Interpretation bands reported by the CLI:
  `>0.5 strong`, `0.3–0.5 moderate`, `<0.3 likely noise`.
- **Decay + half-life.** `ic_decay(signal_panel, close_panel, horizons) -> dict`
  of `h -> mean IC_h` for `h ∈ {1, 5, 10, 20, 50}`. `half_life` = the smallest
  horizon at which `|mean IC_h| <= 0.5 * |mean IC_1|` (report `>50` if it never
  falls that far within the tested horizons; `None` if IC_1 ≈ 0).

All IC computations here receive only the in-sample-masked panels; `ic.py`
itself is agnostic to the split (it correlates whatever panels it is given).

### 5. CLI — `python -m rhagent.factor`

```
python -m rhagent.factor --strategy momentum [--horizon 5] [--oos-frac 0.25]
                         [--days 400] [--min-names 10]
    → load universe, build the strategy's signal panel, apply the locked
      in-sample mask, and print:
        universe size, n in-sample days,
        ICIR at horizons 1/5/10/20/50 with band labels,
        the IC decay curve and half-life.
```

`--strategy` maps through the existing `REGISTRY`/`build`. Only strategies that
implement `signal()` are eligible; an unknown or signal-less strategy errors
clearly.

## Error handling

Boundary validation only, matching the existing "fail with a clear error"
convention: unknown strategy, a strategy without `signal()`, an empty universe
after history-length filtering, or a horizon/oos-frac that leaves no in-sample
days → a clear `ValueError`/`KeyError` and abort. No defensive handling of
impossible internal states.

## Testing (TDD)

- **signal() implementations** — on a crafted price series, each strategy's
  `signal()` has the expected sign/ordering (e.g. mean_reversion signal is high
  after a sharp drop; its `sign` matches the existing `positions()` where the
  strategy is in a position) and is NaN-free after warmup.
- **forward_returns** — a known price panel yields the expected h-step returns,
  correctly NaN in the last h rows.
- **oos split** — the cutoff sits at the right quantile; `in_sample_mask`
  excludes days whose forward window crosses the cutoff; the OOS slice is never
  included.
- **rank-IC** — a signal perfectly rank-correlated with (neutralized) forward
  returns gives IC = +1; a reversed signal gives −1; an orthogonal one ≈ 0. A
  day with fewer than `min_names` names yields NaN and is dropped.
- **market-neutralization** — adding a constant to every name's forward return
  that day does not change IC (the cross-sectional mean is removed).
- **ICIR** — a fixed IC series yields the expected mean/std ratio; empty or
  zero-variance series yields 0.0.
- **decay/half-life** — a synthetic decaying IC profile yields the expected
  half-life; a non-decaying one reports `>50`.
- **no-lookahead** — `signal()` truncated at day t equals the full-series
  `signal()` value at t for the same bars (recomputation invariance).

## Increments

1. `signal()` on base + the 3 strategies + tests.
2. `universe.py` (list, `load_panels`, `forward_returns`) + tests (with a fake
   `fetch`/seeded cache, like `test_data.py`).
3. `split.py` (`oos_cutoff`, `in_sample_mask`) + tests.
4. `ic.py` (rank-IC, ICIR, decay/half-life) + tests.
5. `signals.py` (`signal_panel`) + CLI `__main__.py` + a smoke test on cached
   data.

## Risks / honest notes

- **Small sample.** ~1 year of history ≈ ~140–190 in-sample daily IC points
  after the OOS holdout and horizon trim. Enough for a rough ICIR, but with wide
  error bars. The CLI reports it as an estimate; strong claims wait for more
  history (sub-project 3's correction is what guards against over-reading it).
- **Rank-IC removes the common shift, not beta exposure.** Because rank-IC ranks
  the cross-section, it is already invariant to a common additive shift in that
  day's returns (equivalent to removing the equal-weighted common mean) — no
  separate demeaning step changes it. That is a narrower guarantee than "market
  neutral": it does NOT remove differential beta exposure, so a signal that
  merely proxies market beta (loads more on high-beta names) can still earn a
  positive rank-IC. Real beta-residualization is deferred to a later
  sub-project.
- **Universe is fixed today.** Using a current large-cap list over a ~1-year
  window has negligible survivorship bias; if the window is later extended to
  many years, point-in-time constituents become necessary — noted for the
  future, not handled now.
- **Signal ≠ tradability.** A high ICIR means predictive skill, not profit after
  costs. P&L stays with the trade harness; this evaluator is deliberately about
  signal quality only.
