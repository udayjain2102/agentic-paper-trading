# Backtest & Strategy Selection — Design

**Date:** 2026-07-06
**Status:** Approved (brainstorming), pending implementation plan

## Motivation

The existing bot (`src/rhagent/`) is an LLM-agentic equities trader with hard,
code-enforced guardrails, but it has no explicit numeric strategy and no way to
evaluate one. The goal is to take quantitative techniques from the "Quant Bible"
(MIT Sloan Business Club interview guide) and turn them into explicit, rule-based
strategies, backtest all of them on historical data, rank by **total return**,
and wire the winner into the bot so it trades under the existing safety rails.

## Decisions (locked)

- **Candidate strategies:** run all four; the highest total return wins.
  1. Mean-reversion (z-score)
  2. Momentum / trend
  3. Linear-regression signal
  4. Pairs trading
- **Ranking metric:** total cumulative return. (Sharpe, max drawdown, hit-rate
  reported alongside for context but do not decide the winner.)
- **Data source:** Robinhood MCP `get_equity_historicals` (daily, split-adjusted),
  cached to local CSV so backtests are reproducible and offline.
- **Universe / period:** a handful of large-caps (AAPL, MSFT, NVDA, SPY) over the
  last ~1 year of daily bars.
- **Code:** written by Sonnet 5 subagents. Plan written by Opus.
- **Long-only by default** (matches the current equities bot). Shorting is a flag,
  default off. A short signal (`-1`) is treated as flat when shorting is off.

## Architecture

The backtest capability is a pure, offline module alongside the existing bot. It
reuses the bot's safety funnel for the live "use the winner" path but does not
involve the LLM — the strategies are deterministic rules.

```
src/rhagent/
  strategies/
    base.py            # Strategy interface
    mean_reversion.py
    momentum.py
    linreg.py
    pairs.py
  data.py              # Fetch OHLCV via RH MCP, cache to data/*.csv
  backtest.py          # Engine: strategy over bars -> equity curve + metrics
  compare.py           # Run ALL strategies over universe, rank, pick winner
```

New CLI: `python -m rhagent.compare` — fetches (or loads cached) ~1yr daily bars,
runs all four strategies, prints a ranking table sorted by total return, declares
the winner.

`data/` is gitignored (cached CSVs).

## Components

### `strategies/base.py` — Strategy interface

- What it does: defines the common contract. A strategy takes a DataFrame of daily
  OHLCV bars and returns a target-position series: `+1` (long), `0` (flat), `-1`
  (short), one value per day.
- Interface: `class Strategy: def positions(self, bars: pd.DataFrame) -> pd.Series`.
  Each concrete strategy takes its parameters in `__init__`.
- Depends on: pandas/numpy only.

**No-lookahead invariant (applies to every strategy):** the position for day *t*
is computed only from data up to and including day *t*, and is applied to the
return from day *t* to day *t+1*. No signal may use future bars.

### The four strategies

- **`mean_reversion.py`** — rolling mean & std of close over lookback `N`
  (default 20). `z = (price - mean) / std`. Long when `z < -entry` (default 1.0),
  exit to flat when `z >= -exit` (default 0.0).
- **`momentum.py`** — trailing return over lookback `N` (default 40). Long if
  trailing return > 0, else flat.
- **`linreg.py`** — OLS of next-day return on features (lagged returns, moving-avg
  ratio), fit on a rolling/expanding window. Long when predicted next-day return
  > 0. Rolling fit uses only past data (no lookahead).
- **`pairs.py`** — takes two tickers; z-score of the price spread over lookback
  `N`. Long the underperformer / short the outperformer when the spread diverges
  past `entry`. `compare.py` auto-selects the highest-correlation pair in the
  universe and reports which pair was chosen. With shorting off, only the long
  leg trades; the output flags this.

### `data.py` — data fetch + cache

- What it does: returns a per-symbol DataFrame of daily OHLCV bars for a date
  range. On cache miss, calls the RH MCP `get_equity_historicals` (interval
  `day`, `adjustment_type=split`, up to 10 symbols/call) and writes `data/<SYMBOL>.csv`;
  on cache hit, reads the CSV.
- Interface: `get_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]`.
- Depends on: the MCP client wiring used elsewhere in the bot; pandas.
- Tested against a saved MCP response fixture — tests never hit the network.

### `backtest.py` — engine

- What it does: turns a position series + price series into an equity curve and
  metrics. Daily rebalance; position from day *t* earns the day *t* -> *t+1*
  return. Applies a per-trade cost (default a few bps) on position changes.
- Interface: `run(bars, positions, cost_bps=...) -> BacktestResult` where the
  result carries the equity curve and metrics: **total_return** (ranking metric),
  sharpe, max_drawdown, hit_rate.
- Depends on: pandas/numpy.

### `compare.py` — comparison & selection

- What it does: for each strategy, runs it over every symbol in the universe,
  aggregates equal-weight into one equity curve per strategy, and prints a table
  sorted by total return with all metrics. Declares the top strategy the winner.
- Interface: CLI `python -m rhagent.compare`.
- Output includes the winner's name and parameters, ready to paste into config.

## "The one we use" — integration with the live bot

- The winning strategy + its parameters are recorded in `config.yaml` under a new
  `strategy:` block (`name`, `params`).
- The runner gains a **strategy mode** (selected by config/flag; LLM-agent mode is
  the other and stays unchanged). In strategy mode the runner:
  1. loads recent daily bars for the configured universe via `data.py`,
  2. computes the winning strategy's target position for today,
  3. translates position changes into buy/sell orders,
  4. sends every order through the **existing `OrderExecutor` + guardrails** —
     the same funnel the LLM path uses. No order bypasses the guardrails.
- This is the minimal safe integration: deterministic strategy, existing safety
  rails, no new order path.

## Testing

- Each strategy: unit tests on tiny hand-built price series with known-correct
  signals, plus a **lookahead-guard test** (appending future bars must not change
  earlier positions).
- Engine: a known return series produces a known equity curve and known metrics.
- `data.py`: tested against a saved MCP response fixture (no network).
- Integration (strategy mode): a smoke test asserting orders flow through the
  executor and that a dry-run places nothing (mirrors the existing runner smoke
  test).
- All existing tests continue to pass.

## Dependencies

- Add `pandas` and `numpy` to `requirements.txt`.
- `data/` added to `.gitignore`.

## Out of scope

- Intraday / minute-bar strategies (daily bars only).
- Parameter optimization / walk-forward tuning (fixed sensible defaults; a future
  step).
- Options, crypto, live streaming, web UI (unchanged from bot v1 scope).
- Shorting is implemented as an off-by-default flag but not the focus.
