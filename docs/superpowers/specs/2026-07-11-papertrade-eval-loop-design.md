# Paper-Trading & Evaluation Loop — Design

Date: 2026-07-11
Status: Approved (brainstorming)

## Problem

We want to paper-trade the trading logic over historical data, see per-trade what
went right and what went wrong, and iterate on the model many times. Two decision
engines must eventually run through the same harness: the rule-based strategies
(now) and the Claude LLM agent (later). Every trade must carry a **trade ID** so
any result can be traced back to exactly what was done and why.

The existing `backtest.py` is vectorized: it scores a whole positions series at
once and has no concept of an individual trade, so there is nothing to attach a
trade ID or an entry/exit reason to. This work adds an **event-driven** layer
alongside it — `backtest.py` is left untouched and still serves fast strategy
ranking (`compare.py`).

## Goals

- Step through historical bars one at a time, ask a decision engine what to hold
  and why, and turn each change into discrete, ID-stamped trades.
- Persist a traceable per-trade ledger (run ID + trade ID + reason + entry/exit
  + P&L).
- Produce four evaluation outputs per run: per-trade ledger, aggregate stats,
  failure buckets, run-to-run comparison.
- Support both decision engines through one interface (strategies now, agent
  later).
- Keep each iteration fast and free (runs offline on cached historical bars).

## Non-goals

- Real-time / live paper trading (the existing live runner already covers
  forward dry-run; not part of this loop).
- Wiring the Claude agent into the harness now — the interface is designed for
  it, but the `AgentEngine` adapter is a later increment.
- Changing `backtest.py`, `compare.py`, or the live `runner.py`/`executor.py`
  order path.

## Architecture

Four new modules under `src/rhagent/`, plus a ledger directory. Nothing existing
is modified except adding a strategy-signal-snapshot helper if needed.

```
papertrade engine  ──drives──▶  DecisionEngine  ──wraps──▶  Strategy (now)
      │                                          └──wraps──▶  Agent (later)
      │
      ▼
  trade ledger (journal/papertrade/{run_id}/)
      │
      ▼
  evaluate ──▶ per-trade ledger | aggregate stats | failure buckets | run compare
```

### 1. DecisionEngine interface — new `src/rhagent/engine.py`

The single contract both decision sources implement.

```python
@dataclass(frozen=True)
class Decision:
    target: float          # desired position in {-1, 0, +1}
    reason: str            # human-readable why (signal snapshot or agent text)

class DecisionEngine(Protocol):
    name: str
    def decide(self, symbol: str, history: pd.DataFrame, current_pos: float) -> Decision: ...
```

- `history` contains only bars up to and including the current day t
  (no-lookahead invariant, same as `base.py`).
- `current_pos` is what the harness currently holds for that symbol, so an engine
  can be stateful/hysteretic if it wants.

**StrategyEngine(strat: Strategy)** — adapts an existing `Strategy`. On each call
it computes `strat.positions(history).iloc[-1]` as the target and builds `reason`
from a small signal snapshot (strategy name + the deciding indicator values on
day t). Reuses the `base.py` contract; strategies are not rewritten.

**AgentEngine** — later increment. Same shape, wraps the Claude decision loop.
Out of scope for the first build but the interface must not need changes to add
it.

### 2. Paper-trade engine — `papertrade.py`

Event loop, not vectorized.

- Inputs: a `DecisionEngine`, a `dict[symbol -> bars DataFrame]`, `cost_bps`,
  and a per-symbol notional (equal-weight, fixed capital per symbol).
- For each day t (chronological), for each symbol: call `decide` with
  `history[:t+1]`; diff `target` against the currently held position; the
  difference is an order, "filled" at `close[t]`.
- Fill convention: **fill at the close of day t**. P&L on a position accrues from
  its entry close to its exit close. This is consistent and lookahead-free
  (day t's close is known at end of day t). Turnover is charged `cost_bps` per
  unit traded, matching `backtest.py`'s cost model.
- Position lifecycle → trades: opening from flat starts a trade; reducing/closing
  to flat finishes it; a flip (e.g. +1 → -1) finishes the current trade and opens
  a new one at the same bar. Each finished trade is emitted as one record.
- Deterministic: same inputs → same ledger.

### 3. Trade ledger + ID scheme

Written under `journal/papertrade/{run_id}/`:

- `run.json` — run metadata (config snapshot): run_id, engine name, symbols,
  date range, cost_bps, per-symbol notional, git commit (if available), created
  timestamp. This snapshot is what makes runs comparable.
- `trades.jsonl` — one finished trade per line, append-only (same spirit as
  `journal.py`).

IDs:

- **run_id**: `YYYY-MM-DDTHH-MM-SSZ-<8 hex>` (UTC timestamp + short random
  suffix), e.g. `2026-07-11T14-22-03Z-a1b2c3d4`.
- **trade_id**: `{run_id}#{seq:04d}`, seq monotonic within the run — traces a
  trade back to its exact run and config.

Trade record fields:

```
trade_id, run_id, symbol, side            # side ∈ {long, short}
entry_ts, entry_price, entry_reason
exit_ts,  exit_price,  exit_reason
qty                                        # signed units / notional
pnl_abs, pnl_pct, holding_bars
outcome                                    # win | loss | flat
entry_features { vol20, gap, trend, ... }  # snapshot at entry, for bucketing
```

`entry_features` are cheap, lookahead-free scalars computed from `history` at
entry: e.g. 20-bar realized volatility, overnight gap (open/prev close − 1),
short trend sign. Kept intentionally small; extend later as buckets demand.

### 4. Evaluation — `evaluate.py`

Reads a run's ledger and produces the four outputs. Pure functions over the
loaded records so they are unit-testable without running a paper-trade.

- **Per-trade ledger** — load `trades.jsonl` into a DataFrame; pretty-print the
  full table (all record fields).
- **Aggregate stats** — win rate, avg win, avg loss, profit factor
  (gross win / gross loss), total return, Sharpe (on the per-trade or daily
  return series), max drawdown, avg holding_bars, n_trades.
- **Failure buckets** — take losing trades, group by binned `entry_features`
  (volatility regime low/med/high, gap up/down/flat, holding short/long, symbol,
  side). Report each bucket's share of total loss and its win rate, ranked by
  loss contribution → shows where losses concentrate.
- **Run comparison** — scan `journal/papertrade/*/run.json` + recomputed
  aggregates; print a table across runs (one row per run_id with its aggregate
  stats and a diff of key metrics vs the previous run) → answers "did this
  tuning change actually help".

### 5. CLI — `python -m rhagent.papertrade`

```
python -m rhagent.papertrade --engine mean_reversion --symbols NVDA,SPY --days 400
    → run the loop, write the ledger, print aggregate stats + failure buckets

python -m rhagent.papertrade compare
    → run-to-run comparison table across all recorded runs
```

`--engine` maps to the strategy registry via a `StrategyEngine` wrapper.
`--symbols` and `--days` mirror `compare.py`'s data loading (`get_bars`, cached
to `data/*.csv`).

## Error handling

- Boundary validation only: unknown engine name, empty/short bar history, no
  symbols, malformed ledger file → clear error and abort. No defensive handling
  for impossible internal states.
- Fail fast, consistent with the existing "fail with a clear error" convention
  (e.g. the pairs-universe check in `strategy_runner.py`).

## Testing (TDD)

Write tests first for each unit:

- **Trade extraction** — a deterministic dummy `DecisionEngine` producing a known
  target sequence yields the expected trades: open-from-flat, partial reduce,
  full close, and flip (+1 → −1) splitting into two trades.
- **P&L math** — entry/exit prices and cost_bps produce the expected pnl_abs /
  pnl_pct / holding_bars for long and short.
- **Aggregate math** — a fixed set of trade records yields the expected win rate,
  profit factor, avg win/loss, etc.
- **Bucket grouping** — trades with known `entry_features` land in the expected
  buckets with the expected loss shares.
- **ID scheme** — trade_ids are unique, monotonic, and parse back to their run_id.
- **Determinism** — same inputs → byte-identical ledger (modulo timestamps).

## Increments

1. `engine.py` (Decision + DecisionEngine + StrategyEngine) + tests.
2. `papertrade.py` event loop + trade extraction + ledger writer + tests.
3. `evaluate.py` four outputs + tests.
4. CLI wiring (`__main__` in `papertrade.py`) + smoke test.
5. (later, separate spec) `AgentEngine` to run the Claude agent through the loop.
