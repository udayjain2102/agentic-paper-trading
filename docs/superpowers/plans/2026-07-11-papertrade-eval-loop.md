# Paper-Trading & Evaluation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An event-driven paper-trading harness that steps a swappable DecisionEngine through historical bars, stamps every trade with a traceable trade ID + reason, and evaluates runs (per-trade ledger, aggregate stats, failure buckets, run-to-run comparison).

**Architecture:** Three new modules in `src/rhagent/`: `engine.py` (Decision/DecisionEngine/StrategyEngine), `papertrade.py` (MarketSource + FillModel seams, event loop, ledger writer, CLI), `evaluate.py` (pure functions over the ledger). Nothing existing is modified; `backtest.py` stays for fast vectorized ranking and is reused for return metrics.

**Tech Stack:** Python 3, pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-11-papertrade-eval-loop-design.md`

## Global Constraints

- Positions/targets are in `{-1, 0, +1}` (units of one per-symbol notional), matching `strategies/base.py`.
- No-lookahead invariant: a decision on day t sees only bars up to and including t; fills happen at `close[t]`.
- Turnover cost: `cost_bps` per unit traded (default 1.0), same model as `backtest.py`.
- Ledger layout: `journal/papertrade/{run_id}/` containing `run.json`, `trades.jsonl`, `returns.csv`.
- ID formats: run_id `YYYY-MM-DDTHH-MM-SSZ-<8 hex>`; trade_id `{run_id}#{seq:04d}` with seq starting at 1.
- Determinism: same inputs (and injected run_id/clock) → identical ledger.
- Error handling at boundaries only: unknown engine, empty history, no symbols, malformed ledger → clear exception. No defensive handling of impossible internal states.
- Style: follow existing module conventions (`from __future__ import annotations`, module docstring explaining role, small pure functions).
- Tests live under `tests/`, run with `.venv/bin/python -m pytest`.

---

### Task 1: DecisionEngine interface (`engine.py`)

**Files:**
- Create: `src/rhagent/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `rhagent.strategies.base.Strategy` (existing: `.name: str`, `.positions(bars: pd.DataFrame) -> pd.Series`).
- Produces (used by Tasks 2–4):
  - `Decision(target: float, reason: str)` — frozen dataclass.
  - `DecisionEngine` Protocol: `.name: str`, `.decide(symbol: str, history: pd.DataFrame, current_pos: float) -> Decision`.
  - `StrategyEngine(strat: Strategy)` — adapter; `.name` == strategy name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py
import pandas as pd

from rhagent.engine import Decision, StrategyEngine
from rhagent.strategies.base import Strategy


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


class AlwaysLong(Strategy):
    name = "always_long"

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=bars.index, dtype=int)


def test_decision_is_frozen():
    d = Decision(target=1.0, reason="x")
    assert d.target == 1.0 and d.reason == "x"


def test_strategy_engine_takes_last_position_as_target():
    eng = StrategyEngine(AlwaysLong())
    hist = _bars([100.0, 101.0, 102.0])
    d = eng.decide("AAPL", hist, current_pos=0.0)
    assert d.target == 1.0


def test_strategy_engine_reason_names_strategy_and_close():
    eng = StrategyEngine(AlwaysLong())
    hist = _bars([100.0, 250.5])
    d = eng.decide("AAPL", hist, current_pos=0.0)
    assert "always_long" in d.reason
    assert "250.50" in d.reason


def test_strategy_engine_exposes_strategy_name():
    assert StrategyEngine(AlwaysLong()).name == "always_long"


def test_strategy_engine_no_lookahead_only_sees_history():
    # Target on a 2-bar history must equal the strategy's value at that bar,
    # regardless of what later bars would have said.
    class LastCloseSign(Strategy):
        name = "last_close_sign"

        def positions(self, bars: pd.DataFrame) -> pd.Series:
            sign = 1 if bars["close"].iloc[-1] >= 100 else 0
            return pd.Series(sign, index=bars.index, dtype=int)

    eng = StrategyEngine(LastCloseSign())
    assert eng.decide("A", _bars([100.0, 99.0]), 0.0).target == 0.0
    assert eng.decide("A", _bars([100.0, 99.0, 101.0]), 0.0).target == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'rhagent.engine'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/engine.py
"""The decision seam between the paper-trade loop and whatever decides.

A DecisionEngine answers one question per bar: given the history up to and
including today and what we currently hold, what should the position be and
why. StrategyEngine adapts the existing rule-based strategies; an AgentEngine
wrapping the Claude loop plugs into the same protocol later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .strategies.base import Strategy


@dataclass(frozen=True)
class Decision:
    target: float  # desired position in {-1, 0, +1}
    reason: str    # human-readable why


class DecisionEngine(Protocol):
    name: str

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision: ...


class StrategyEngine:
    """Adapt a vectorized Strategy: the last value of positions(history) is
    the target for today. history must contain only bars up to today."""

    def __init__(self, strat: Strategy) -> None:
        self.strat = strat
        self.name = strat.name

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision:
        target = float(self.strat.positions(history).iloc[-1])
        close = float(history["close"].iloc[-1])
        reason = f"{self.name}: target={target:+.0f} close={close:.2f}"
        return Decision(target=target, reason=reason)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/engine.py tests/test_engine.py
git commit -m "feat: DecisionEngine interface + StrategyEngine adapter"
```

---

### Task 2: Paper-trade seams + helpers (`papertrade.py` part 1)

**Files:**
- Create: `src/rhagent/papertrade.py`
- Test: `tests/test_papertrade_helpers.py`

**Interfaces:**
- Consumes: `rhagent.data.get_bars(symbols, start, end, *, fetch=None, cache_dir="data") -> dict[str, pd.DataFrame]`.
- Produces (used by Tasks 3–5):
  - `MarketSource` Protocol: `.bars() -> dict[str, pd.DataFrame]`.
  - `FillModel` Protocol: `.fill(symbol: str, delta: float, bar: pd.Series) -> float`.
  - `HistoricalSource(symbols: list[str], start: str, end: str, cache_dir="data")`.
  - `CloseFill()` — fills at `bar["close"]`.
  - `new_run_id(now: datetime | None = None, suffix: str | None = None) -> str`.
  - `entry_features(history: pd.DataFrame) -> dict` with keys `vol20`, `gap`, `trend5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_papertrade_helpers.py
import re
from datetime import datetime, timezone

import pandas as pd

from rhagent.papertrade import CloseFill, HistoricalSource, entry_features, new_run_id


def _bars(closes, opens=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    opens = opens or closes
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_run_id_format_and_determinism():
    now = datetime(2026, 7, 11, 14, 22, 3, tzinfo=timezone.utc)
    rid = new_run_id(now=now, suffix="a1b2c3d4")
    assert rid == "2026-07-11T14-22-03Z-a1b2c3d4"


def test_run_id_random_suffix_matches_format():
    rid = new_run_id()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z-[0-9a-f]{8}", rid)


def test_close_fill_fills_at_close():
    bar = pd.Series({"open": 10.0, "close": 12.5})
    assert CloseFill().fill("AAPL", 1.0, bar) == 12.5


def test_historical_source_reads_cached_csv(tmp_path):
    df = _bars([1.0, 2.0])
    df.to_csv(tmp_path / "AAPL.csv")
    src = HistoricalSource(["AAPL"], "2026-01-01", "2026-01-02", cache_dir=tmp_path)
    out = src.bars()
    assert list(out) == ["AAPL"]
    assert out["AAPL"]["close"].tolist() == [1.0, 2.0]


def test_entry_features_keys_and_values():
    closes = [100.0] * 25
    opens = list(closes)
    opens[-1] = 102.0  # 2% gap up vs prev close 100
    hist = _bars(closes, opens)
    f = entry_features(hist)
    assert set(f) == {"vol20", "gap", "trend5"}
    assert f["vol20"] == 0.0          # flat closes -> zero vol
    assert abs(f["gap"] - 0.02) < 1e-9
    assert f["trend5"] == 0.0         # flat -> no trend


def test_entry_features_trend_sign():
    hist = _bars([100, 100, 100, 100, 100, 101, 102, 103, 104, 105])
    assert entry_features(hist)["trend5"] == 1.0


def test_entry_features_short_history_is_nan_free():
    f = entry_features(_bars([100.0, 101.0]))
    assert all(v == v for v in f.values())  # no NaNs leak into the ledger
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_papertrade_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rhagent.papertrade'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/papertrade.py
"""Event-driven paper-trading harness.

Steps a DecisionEngine through bars one day at a time, turns position changes
into discrete ID-stamped trades, and writes an append-only ledger under
journal/papertrade/{run_id}/. Two seams keep it world-model-ready: bars come
from a MarketSource and orders are priced by a FillModel — swap either without
touching the loop. The vectorized backtest.py is untouched and remains the
fast ranking path.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd

from .data import get_bars


class MarketSource(Protocol):
    def bars(self) -> dict[str, pd.DataFrame]: ...


class FillModel(Protocol):
    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float: ...


class HistoricalSource:
    """Real cached history via data.get_bars (offline once cached)."""

    def __init__(self, symbols, start: str, end: str, cache_dir="data") -> None:
        self.symbols = list(symbols)
        self.start, self.end, self.cache_dir = start, end, cache_dir

    def bars(self) -> dict[str, pd.DataFrame]:
        return get_bars(self.symbols, self.start, self.end, cache_dir=self.cache_dir)


class CloseFill:
    """Perfect fill at the bar's close. cost_bps is charged by the loop."""

    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float:
        return float(bar["close"])


def new_run_id(now: datetime | None = None, suffix: str | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    suffix = suffix or secrets.token_hex(4)
    return f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}-{suffix}"


def entry_features(history: pd.DataFrame) -> dict:
    """Cheap lookahead-free scalars at entry, used for failure bucketing."""
    close = history["close"].astype(float)
    rets = close.pct_change().dropna()

    vol20 = float(rets.tail(20).std()) if len(rets) >= 2 else 0.0
    if pd.isna(vol20):
        vol20 = 0.0

    gap = 0.0
    if len(close) >= 2 and "open" in history:
        gap = float(history["open"].iloc[-1] / close.iloc[-2] - 1.0)

    trend5 = 0.0
    if len(close) >= 6:
        diff = float(close.iloc[-1] - close.iloc[-6])
        trend5 = 0.0 if diff == 0 else (1.0 if diff > 0 else -1.0)

    return {"vol20": vol20, "gap": gap, "trend5": trend5}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_papertrade_helpers.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/papertrade.py tests/test_papertrade_helpers.py
git commit -m "feat: papertrade seams (MarketSource/FillModel) + run-id and entry-feature helpers"
```

---

### Task 3: PaperTrader event loop + ledger (`papertrade.py` part 2)

**Files:**
- Modify: `src/rhagent/papertrade.py` (append to Task 2's file)
- Test: `tests/test_papertrade_loop.py`

**Interfaces:**
- Consumes: `Decision`, `DecisionEngine` (Task 1); `MarketSource`, `FillModel`, `CloseFill`, `new_run_id`, `entry_features` (Task 2).
- Produces (used by Tasks 4–5):
  - `PaperTrader(engine, source, fill=None, cost_bps=1.0, notional=10_000.0, out_dir="journal/papertrade", run_id=None)`.
  - `.run() -> pathlib.Path` — executes the loop, writes `run.json`, `trades.jsonl`, `returns.csv` under `{out_dir}/{run_id}/`, returns that directory.
  - Trade record dict fields (one JSON object per `trades.jsonl` line):
    `trade_id, run_id, symbol, side ("long"|"short"), entry_ts, entry_price,
    entry_reason, exit_ts, exit_price, exit_reason, qty, pnl_abs, pnl_pct,
    holding_bars, outcome ("win"|"loss"|"flat"), entry_features (dict)`.
  - `returns.csv` columns: `date, net` — daily equal-weight portfolio net return.
  - `run.json` fields: `run_id, engine, symbols, start, end, cost_bps, notional, created_ts`.

**Trade-lifecycle rule (locked in):** a trade is a period of constant nonzero
position. Any change of position closes the open trade (if any) at the fill
price and, if the new target is nonzero, opens a new trade at the same bar.
A flip (+1 → −1) therefore produces two trades; a "partial reduce" is a
close-and-reopen at the smaller size. Trades still open at the end of data are
force-closed at the last close with `exit_reason="end_of_data"`.

**P&L rule (locked in):** for a trade of signed position `q` (e.g. +1 or −1)
entered at `pe` and exited at `px`:
`pnl_pct = sign(q) * (px / pe - 1) - (2 * |q| * cost_bps / 1e4)` and
`pnl_abs = notional * pnl_pct`. `holding_bars` = number of bars from entry to
exit (exit index − entry index). `outcome` = win if `pnl_abs > 0`, loss if
`< 0`, else flat.

**Daily return rule (locked in):** for each symbol, day t contributes
`pos_{t-1} * (close_t / close_{t-1} - 1) - |pos_t - pos_{t-1}| * cost_bps / 1e4`;
day 0 contributes `-|pos_0| * cost_bps / 1e4`. The portfolio `net` is the
equal-weight mean across symbols. Same cost model as `backtest.net_returns`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_papertrade_loop.py
import json

import pandas as pd
import pytest

from rhagent.engine import Decision
from rhagent.papertrade import PaperTrader


def _bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


class FakeSource:
    def __init__(self, frames):  # dict[str, DataFrame]
        self._frames = frames

    def bars(self):
        return self._frames


class ScriptedEngine:
    """Emits a fixed target sequence per symbol — fully deterministic."""

    name = "scripted"

    def __init__(self, script):  # dict[str, list[float]]
        self.script = script

    def decide(self, symbol, history, current_pos):
        t = len(history) - 1
        target = float(self.script[symbol][t])
        return Decision(target=target, reason=f"scripted[{t}]={target:+.0f}")


def _run(script, closes, tmp_path, cost_bps=0.0):
    trader = PaperTrader(
        engine=ScriptedEngine(script),
        source=FakeSource({s: _bars(closes[s]) for s in closes}),
        cost_bps=cost_bps,
        notional=10_000.0,
        out_dir=tmp_path,
        run_id="2026-07-11T00-00-00Z-deadbeef",
    )
    run_dir = trader.run()
    trades = [json.loads(l) for l in (run_dir / "trades.jsonl").read_text().splitlines()]
    return run_dir, trades


def test_open_then_close_produces_one_trade(tmp_path):
    # flat, long, long, flat  on closes 100,110,121,133.1
    _, trades = _run({"A": [0, 1, 1, 0]}, {"A": [100.0, 110.0, 121.0, 133.1]}, tmp_path)
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "long"
    assert t["entry_price"] == 110.0 and t["exit_price"] == 133.1
    assert t["holding_bars"] == 2
    assert abs(t["pnl_pct"] - (133.1 / 110.0 - 1)) < 1e-12
    assert t["outcome"] == "win"


def test_flip_splits_into_two_trades(tmp_path):
    _, trades = _run({"A": [1, 1, -1, -1]}, {"A": [100.0, 110.0, 120.0, 90.0]}, tmp_path)
    assert len(trades) == 2
    first, second = trades
    assert first["side"] == "long" and first["exit_price"] == 120.0
    assert second["side"] == "short" and second["entry_price"] == 120.0
    # short from 120 to 90: +25%
    assert abs(second["pnl_pct"] - 0.25) < 1e-12


def test_end_of_data_force_closes(tmp_path):
    _, trades = _run({"A": [0, 1, 1]}, {"A": [100.0, 100.0, 105.0]}, tmp_path)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "end_of_data"
    assert trades[0]["exit_price"] == 105.0


def test_trade_ids_unique_monotonic_and_parse_to_run(tmp_path):
    _, trades = _run(
        {"A": [1, 0, 1, 0]}, {"A": [100.0, 101.0, 102.0, 103.0]}, tmp_path
    )
    ids = [t["trade_id"] for t in trades]
    assert ids == ["2026-07-11T00-00-00Z-deadbeef#0001",
                   "2026-07-11T00-00-00Z-deadbeef#0002"]
    assert all(t["run_id"] == "2026-07-11T00-00-00Z-deadbeef" for t in trades)


def test_cost_bps_charged_on_round_trip(tmp_path):
    _, trades = _run({"A": [0, 1, 0]}, {"A": [100.0, 100.0, 100.0]}, tmp_path,
                     cost_bps=10.0)
    t = trades[0]
    # flat prices, 10 bps each way -> -20 bps
    assert abs(t["pnl_pct"] - (-0.002)) < 1e-12
    assert t["outcome"] == "loss"
    assert abs(t["pnl_abs"] - (-20.0)) < 1e-9


def test_run_json_and_returns_csv_written(tmp_path):
    run_dir, _ = _run({"A": [0, 1, 0]}, {"A": [100.0, 110.0, 121.0]}, tmp_path)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["run_id"] == "2026-07-11T00-00-00Z-deadbeef"
    assert meta["engine"] == "scripted"
    assert meta["symbols"] == ["A"]
    rets = pd.read_csv(run_dir / "returns.csv", parse_dates=["date"])
    assert list(rets.columns) == ["date", "net"]
    assert len(rets) == 3
    # held +1 from day1 close 110 to day2 close 121 -> 10% on day2
    assert abs(rets["net"].iloc[2] - 0.10) < 1e-12


def test_two_symbols_equal_weight_returns(tmp_path):
    run_dir, trades = _run(
        {"A": [1, 1, 1], "B": [0, 0, 0]},
        {"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 50.0]},
        tmp_path,
    )
    rets = pd.read_csv(run_dir / "returns.csv")
    # A earns 10% on day1; B flat -> equal-weight 5%
    assert abs(rets["net"].iloc[1] - 0.05) < 1e-12


def test_entry_features_present_on_trades(tmp_path):
    _, trades = _run({"A": [0, 1, 0]}, {"A": [100.0, 110.0, 121.0]}, tmp_path)
    assert set(trades[0]["entry_features"]) == {"vol20", "gap", "trend5"}


def test_empty_symbols_raises():
    with pytest.raises(ValueError, match="no symbols"):
        PaperTrader(engine=ScriptedEngine({}), source=FakeSource({})).run()


def test_determinism_same_inputs_same_ledger(tmp_path):
    script = {"A": [0, 1, 1, 0, -1, -1]}
    closes = {"A": [100.0, 101.0, 99.0, 102.0, 103.0, 101.0]}
    _, t1 = _run(script, closes, tmp_path / "r1")
    _, t2 = _run(script, closes, tmp_path / "r2")
    assert t1 == t2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_papertrade_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'PaperTrader'`

- [ ] **Step 3: Write the implementation (append to `src/rhagent/papertrade.py`)**

```python
# append to src/rhagent/papertrade.py
import json
from pathlib import Path

from .engine import DecisionEngine


class PaperTrader:
    """Drive a DecisionEngine bar-by-bar and write the trade ledger.

    A trade is a period of constant nonzero position: any target change closes
    the open trade at the fill price and, if the new target is nonzero, opens
    a new one at the same bar. Open trades at end-of-data are force-closed.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        source: MarketSource,
        fill: FillModel | None = None,
        cost_bps: float = 1.0,
        notional: float = 10_000.0,
        out_dir: str | Path = "journal/papertrade",
        run_id: str | None = None,
    ) -> None:
        self.engine = engine
        self.source = source
        self.fill = fill or CloseFill()
        self.cost_bps = cost_bps
        self.notional = notional
        self.out_dir = Path(out_dir)
        self.run_id = run_id or new_run_id()

    def run(self) -> Path:
        frames = self.source.bars()
        if not frames:
            raise ValueError("no symbols: MarketSource returned no bar frames")
        for s, df in frames.items():
            if len(df) < 2:
                raise ValueError(f"history too short for {s}: {len(df)} bars")

        symbols = sorted(frames)
        index = frames[symbols[0]].index
        pos: dict[str, float] = {s: 0.0 for s in symbols}
        open_trades: dict[str, dict] = {}
        trades: list[dict] = []
        daily_net: list[float] = []
        seq = 0

        def close_trade(sym: str, ts, price: float, reason: str, bar_i: int) -> None:
            tr = open_trades.pop(sym)
            q = tr["qty"]
            sign = 1.0 if q > 0 else -1.0
            pnl_pct = sign * (price / tr["entry_price"] - 1.0) - (
                2.0 * abs(q) * self.cost_bps / 1e4
            )
            tr.update(
                exit_ts=str(ts),
                exit_price=price,
                exit_reason=reason,
                pnl_pct=pnl_pct,
                pnl_abs=self.notional * pnl_pct,
                holding_bars=bar_i - tr.pop("_entry_i"),
                outcome=(
                    "win" if pnl_pct > 0 else "loss" if pnl_pct < 0 else "flat"
                ),
            )
            trades.append(tr)

        for i, ts in enumerate(index):
            net_today = []
            for sym in symbols:
                bars = frames[sym]
                history = bars.iloc[: i + 1]
                bar = bars.iloc[i]
                prev = pos[sym]

                d = self.engine.decide(sym, history, prev)
                target = d.target

                # accrue yesterday's position over today's move
                ret = 0.0
                if i > 0:
                    ret = prev * (
                        float(bar["close"]) / float(bars["close"].iloc[i - 1]) - 1.0
                    )
                turnover = abs(target - prev)
                net_today.append(ret - turnover * self.cost_bps / 1e4)

                if target != prev:
                    price = self.fill.fill(sym, target - prev, bar)
                    if prev != 0.0:
                        close_trade(sym, ts, price, d.reason, i)
                    if target != 0.0:
                        seq += 1
                        open_trades[sym] = {
                            "trade_id": f"{self.run_id}#{seq:04d}",
                            "run_id": self.run_id,
                            "symbol": sym,
                            "side": "long" if target > 0 else "short",
                            "qty": target,
                            "entry_ts": str(ts),
                            "entry_price": price,
                            "entry_reason": d.reason,
                            "entry_features": entry_features(history),
                            "_entry_i": i,
                        }
                    pos[sym] = target
            daily_net.append(sum(net_today) / len(symbols))

        # force-close whatever is still open at the last bar
        last_i = len(index) - 1
        for sym in sorted(open_trades):
            last_close = float(frames[sym]["close"].iloc[-1])
            close_trade(sym, index[-1], last_close, "end_of_data", last_i)

        return self._write(symbols, index, trades, daily_net)

    def _write(self, symbols, index, trades, daily_net) -> Path:
        run_dir = self.out_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": self.run_id,
            "engine": self.engine.name,
            "symbols": symbols,
            "start": str(index[0]),
            "end": str(index[-1]),
            "cost_bps": self.cost_bps,
            "notional": self.notional,
            "created_ts": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

        with (run_dir / "trades.jsonl").open("w", encoding="utf-8") as fh:
            for tr in trades:
                fh.write(json.dumps(tr, sort_keys=True) + "\n")

        pd.DataFrame({"date": index, "net": daily_net}).to_csv(
            run_dir / "returns.csv", index=False
        )
        return run_dir
```

Note: trades close in bar order because each symbol pass is chronological;
end-of-data closes iterate `sorted(open_trades)` for determinism. `qty` stays
in the record (signed target units); `_entry_i` is internal and popped before
writing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_papertrade_loop.py tests/test_papertrade_helpers.py tests/test_engine.py -v`
Expected: all pass (10 + 7 + 5)

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/papertrade.py tests/test_papertrade_loop.py
git commit -m "feat: PaperTrader event loop with ID-stamped trade ledger"
```

---

### Task 4: Evaluation (`evaluate.py`)

**Files:**
- Create: `src/rhagent/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: ledger files written by Task 3 (`run.json`, `trades.jsonl`,
  `returns.csv`); `rhagent.backtest.result_from_returns(net: pd.Series) -> BacktestResult` (existing — reused for total_return/sharpe/max_drawdown).
- Produces (used by Task 5):
  - `load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.Series]` — (meta, trades DataFrame with `entry_features` expanded to `feat_vol20`/`feat_gap`/`feat_trend5` columns, daily net Series indexed by date).
  - `aggregate(trades: pd.DataFrame, net: pd.Series) -> dict` with keys:
    `n_trades, win_rate, avg_win, avg_loss, profit_factor, total_return,
    sharpe, max_drawdown, avg_holding_bars`.
  - `failure_buckets(trades: pd.DataFrame) -> pd.DataFrame` with columns
    `dimension, bucket, n_trades, win_rate, loss_share`, sorted by
    `loss_share` descending.
  - `compare_runs(base_dir: Path) -> pd.DataFrame` — one row per run
    (columns: `run_id, engine, n_trades, win_rate, profit_factor,
    total_return, sharpe, max_drawdown`), sorted by run_id ascending.

**Bucket rules (locked in):** losing-trade analysis over these dimensions —
`vol` (terciles of `feat_vol20` across all trades: low/med/high), `gap`
(`feat_gap` < −0.005 down, > 0.005 up, else flat), `holding`
(`holding_bars` < 5 short else long), `symbol`, `side`. For each (dimension,
bucket): `n_trades` = all trades in bucket, `win_rate` = share of those with
outcome win, `loss_share` = bucket's summed negative `pnl_abs` / total
negative `pnl_abs` (0.0 if there are no losses anywhere).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
import json
from pathlib import Path

import pandas as pd
import pytest

from rhagent.evaluate import aggregate, compare_runs, failure_buckets, load_run


def _trade(tid, pnl_abs, pnl_pct, outcome, vol=0.01, gap=0.0, holding=3,
           symbol="A", side="long"):
    return {
        "trade_id": tid, "run_id": tid.split("#")[0], "symbol": symbol,
        "side": side, "entry_ts": "2026-01-02", "entry_price": 100.0,
        "entry_reason": "r", "exit_ts": "2026-01-05", "exit_price": 101.0,
        "exit_reason": "r", "qty": 1.0, "pnl_abs": pnl_abs, "pnl_pct": pnl_pct,
        "holding_bars": holding, "outcome": outcome,
        "entry_features": {"vol20": vol, "gap": gap, "trend5": 0.0},
    }


def _write_run(run_dir: Path, trades, nets, engine="scripted"):
    run_dir.mkdir(parents=True)
    rid = run_dir.name
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": rid, "engine": engine, "symbols": ["A"],
        "start": "2026-01-01", "end": "2026-01-10",
        "cost_bps": 1.0, "notional": 10000.0, "created_ts": "2026-07-11T00:00:00Z",
    }))
    with (run_dir / "trades.jsonl").open("w") as fh:
        for t in trades:
            fh.write(json.dumps(t) + "\n")
    idx = pd.date_range("2026-01-01", periods=len(nets), freq="D")
    pd.DataFrame({"date": idx, "net": nets}).to_csv(run_dir / "returns.csv", index=False)


@pytest.fixture
def run_dir(tmp_path):
    rid = "2026-07-11T00-00-00Z-aaaaaaaa"
    trades = [
        _trade(f"{rid}#0001", 200.0, 0.02, "win", vol=0.005, gap=0.01, holding=2),
        _trade(f"{rid}#0002", -100.0, -0.01, "loss", vol=0.02, gap=-0.01, holding=8),
        _trade(f"{rid}#0003", -300.0, -0.03, "loss", vol=0.03, gap=-0.02, holding=10,
               symbol="B", side="short"),
        _trade(f"{rid}#0004", 100.0, 0.01, "win", vol=0.01, gap=0.0, holding=1),
    ]
    d = tmp_path / rid
    _write_run(d, trades, [0.0, 0.01, -0.005, 0.02])
    return d


def test_load_run_expands_features(run_dir):
    meta, trades, net = load_run(run_dir)
    assert meta["engine"] == "scripted"
    assert len(trades) == 4
    assert {"feat_vol20", "feat_gap", "feat_trend5"} <= set(trades.columns)
    assert len(net) == 4


def test_aggregate_stats(run_dir):
    _, trades, net = load_run(run_dir)
    a = aggregate(trades, net)
    assert a["n_trades"] == 4
    assert abs(a["win_rate"] - 0.5) < 1e-12
    assert abs(a["avg_win"] - 150.0) < 1e-9
    assert abs(a["avg_loss"] - (-200.0)) < 1e-9
    assert abs(a["profit_factor"] - (300.0 / 400.0)) < 1e-12
    assert abs(a["avg_holding_bars"] - 5.25) < 1e-12
    # return metrics come from backtest.result_from_returns on net
    assert a["total_return"] == pytest.approx((1.01 * 0.995 * 1.02) - 1)


def test_failure_buckets_loss_share(run_dir):
    _, trades, _ = load_run(run_dir)
    b = failure_buckets(trades)
    assert list(b.columns) == ["dimension", "bucket", "n_trades", "win_rate",
                               "loss_share"]
    sym = b[b.dimension == "symbol"].set_index("bucket")
    # total loss 400: A lost 100 (25%), B lost 300 (75%)
    assert abs(sym.loc["B", "loss_share"] - 0.75) < 1e-12
    assert abs(sym.loc["A", "loss_share"] - 0.25) < 1e-12
    side = b[b.dimension == "side"].set_index("bucket")
    assert abs(side.loc["short", "loss_share"] - 0.75) < 1e-12
    # sorted by loss_share descending
    assert b["loss_share"].is_monotonic_decreasing


def test_failure_buckets_no_losses_is_all_zero_share(tmp_path):
    rid = "2026-07-11T00-00-00Z-bbbbbbbb"
    trades = [_trade(f"{rid}#0001", 100.0, 0.01, "win")]
    d = tmp_path / rid
    _write_run(d, trades, [0.01])
    _, tdf, _ = load_run(d)
    b = failure_buckets(tdf)
    assert (b["loss_share"] == 0.0).all()


def test_compare_runs(tmp_path, run_dir):
    # run_dir fixture lives in tmp_path; add a second run
    rid2 = "2026-07-12T00-00-00Z-cccccccc"
    _write_run(tmp_path / rid2, [_trade(f"{rid2}#0001", 50.0, 0.005, "win")],
               [0.005], engine="mean_reversion")
    df = compare_runs(tmp_path)
    assert list(df["run_id"]) == [run_dir.name, rid2]
    assert list(df["engine"]) == ["scripted", "mean_reversion"]
    assert {"n_trades", "win_rate", "profit_factor", "total_return",
            "sharpe", "max_drawdown"} <= set(df.columns)


def test_load_run_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rhagent.evaluate'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/evaluate.py
"""Evaluation over paper-trade ledgers.

Pure functions over the files PaperTrader writes: the per-trade ledger, the
aggregate scorecard, failure buckets (where do losses concentrate), and the
run-to-run comparison. Return metrics reuse backtest.result_from_returns so
the numbers match the vectorized path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import result_from_returns


def load_run(run_dir: str | Path) -> tuple[dict, pd.DataFrame, pd.Series]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"not a run directory (no run.json): {run_dir}")
    meta = json.loads(meta_path.read_text())

    records = [
        json.loads(line)
        for line in (run_dir / "trades.jsonl").read_text().splitlines()
        if line.strip()
    ]
    trades = pd.DataFrame(records)
    if len(trades):
        feats = pd.json_normalize(trades.pop("entry_features")).add_prefix("feat_")
        trades = pd.concat([trades, feats], axis=1)

    rets = pd.read_csv(run_dir / "returns.csv", parse_dates=["date"])
    net = rets.set_index("date")["net"]
    return meta, trades, net


def aggregate(trades: pd.DataFrame, net: pd.Series) -> dict:
    res = result_from_returns(net.astype(float))
    if len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "total_return": res.total_return,
            "sharpe": res.sharpe, "max_drawdown": res.max_drawdown,
            "avg_holding_bars": 0.0,
        }
    pnl = trades["pnl_abs"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["outcome"] == "win").mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "total_return": res.total_return,
        "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown,
        "avg_holding_bars": float(trades["holding_bars"].mean()),
    }


def _bucket_labels(trades: pd.DataFrame) -> dict[str, pd.Series]:
    vol = trades["feat_vol20"].astype(float)
    try:
        vol_bucket = pd.qcut(vol, 3, labels=["low", "med", "high"], duplicates="drop")
    except ValueError:  # too few distinct values to cut
        vol_bucket = pd.Series("all", index=trades.index)
    gap = trades["feat_gap"].astype(float)
    gap_bucket = pd.Series("flat", index=trades.index)
    gap_bucket[gap < -0.005] = "down"
    gap_bucket[gap > 0.005] = "up"
    holding = pd.Series(
        ["short" if h < 5 else "long" for h in trades["holding_bars"]],
        index=trades.index,
    )
    return {
        "vol": vol_bucket.astype(str),
        "gap": gap_bucket,
        "holding": holding,
        "symbol": trades["symbol"],
        "side": trades["side"],
    }


def failure_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["dimension", "bucket", "n_trades", "win_rate", "loss_share"]
    if len(trades) == 0:
        return pd.DataFrame(columns=cols)

    pnl = trades["pnl_abs"].astype(float)
    total_loss = float(-pnl[pnl < 0].sum())

    rows = []
    for dim, labels in _bucket_labels(trades).items():
        for bucket, idx in trades.groupby(labels).groups.items():
            sub = trades.loc[idx]
            sub_pnl = sub["pnl_abs"].astype(float)
            bucket_loss = float(-sub_pnl[sub_pnl < 0].sum())
            rows.append({
                "dimension": dim,
                "bucket": str(bucket),
                "n_trades": int(len(sub)),
                "win_rate": float((sub["outcome"] == "win").mean()),
                "loss_share": bucket_loss / total_loss if total_loss > 0 else 0.0,
            })
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("loss_share", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def compare_runs(base_dir: str | Path) -> pd.DataFrame:
    base_dir = Path(base_dir)
    rows = []
    for meta_path in sorted(base_dir.glob("*/run.json")):
        meta, trades, net = load_run(meta_path.parent)
        a = aggregate(trades, net)
        rows.append({
            "run_id": meta["run_id"],
            "engine": meta["engine"],
            "n_trades": a["n_trades"],
            "win_rate": a["win_rate"],
            "profit_factor": a["profit_factor"],
            "total_return": a["total_return"],
            "sharpe": a["sharpe"],
            "max_drawdown": a["max_drawdown"],
        })
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/evaluate.py tests/test_evaluate.py
git commit -m "feat: ledger evaluation - aggregate stats, failure buckets, run comparison"
```

---

### Task 5: CLI (`python -m rhagent.papertrade`) + smoke test

**Files:**
- Modify: `src/rhagent/papertrade.py` (append `main()` + `__main__` guard)
- Test: `tests/test_papertrade_cli.py`

**Interfaces:**
- Consumes: `REGISTRY`/`build` from `rhagent.strategies`; `StrategyEngine`
  (Task 1); `PaperTrader`, `HistoricalSource` (Tasks 2–3); `load_run`,
  `aggregate`, `failure_buckets`, `compare_runs` (Task 4).
- Produces: `main(argv: list[str] | None = None) -> int`. Usage:
  - `python -m rhagent.papertrade --engine mean_reversion --symbols NVDA,SPY --days 400 [--cost-bps 1.0] [--out-dir journal/papertrade] [--cache-dir data]` → runs, prints run_id + aggregate stats + failure buckets, returns 0.
  - `python -m rhagent.papertrade compare [--out-dir journal/papertrade]` → prints the run-to-run table, returns 0.
  - Unknown `--engine` → clear error, exit 2 (argparse convention).

Note: v1 engines are the single-symbol strategies in `REGISTRY`
(`mean_reversion`, `momentum`, `linreg`). The two-symbol `pairs` strategy is
not exposed here — it needs a two-leg engine adapter, a later increment.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_papertrade_cli.py
import pandas as pd
import pytest

from rhagent.papertrade import main


def _seed_cache(cache_dir, symbol, closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_runs_and_writes_ledger(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    # a wave so mean_reversion actually trades
    closes = [100 + 10 * ((i % 10) - 5) for i in range(80)]
    _seed_cache(cache, "AAPL", [float(c) for c in closes])

    rc = main([
        "--engine", "mean_reversion", "--symbols", "AAPL", "--days", "80",
        "--out-dir", str(tmp_path / "runs"), "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id" in out
    assert "win_rate" in out
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "trades.jsonl").exists()


def test_cli_compare_lists_runs(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    _seed_cache(cache, "AAPL", [100.0 + (i % 7) for i in range(60)])
    for _ in range(2):
        main(["--engine", "momentum", "--symbols", "AAPL", "--days", "60",
              "--out-dir", str(tmp_path / "runs"), "--cache-dir", str(cache)])
    capsys.readouterr()

    rc = main(["compare", "--out-dir", str(tmp_path / "runs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("momentum") == 2


def test_cli_unknown_engine_exits_with_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--engine", "nope", "--symbols", "AAPL",
              "--out-dir", str(tmp_path / "runs")])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_papertrade_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write the implementation (append to `src/rhagent/papertrade.py`)**

```python
# append to src/rhagent/papertrade.py
import argparse
import sys
from datetime import date, timedelta


def _print_report(run_dir: Path) -> None:
    from .evaluate import aggregate, failure_buckets, load_run

    meta, trades, net = load_run(run_dir)
    print(f"run_id: {meta['run_id']}")
    print(f"engine: {meta['engine']}  symbols: {','.join(meta['symbols'])}")

    a = aggregate(trades, net)
    print("\naggregate stats")
    for k, v in a.items():
        print(f"  {k:<18}{v:>12.4f}" if isinstance(v, float) else f"  {k:<18}{v:>12}")

    b = failure_buckets(trades)
    print("\nfailure buckets (by loss share)")
    if len(b) == 0:
        print("  no trades")
    else:
        print(b.to_string(index=False, float_format=lambda x: f"{x:.2%}"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv[:1] == ["compare"]:
        p = argparse.ArgumentParser(prog="rhagent.papertrade compare")
        p.add_argument("--out-dir", default="journal/papertrade")
        args = p.parse_args(argv[1:])
        from .evaluate import compare_runs

        df = compare_runs(args.out_dir)
        if len(df) == 0:
            print(f"no runs found under {args.out_dir}")
        else:
            print(df.to_string(index=False))
        return 0

    from .strategies import REGISTRY, build

    p = argparse.ArgumentParser(prog="rhagent.papertrade")
    p.add_argument("--engine", required=True, choices=sorted(REGISTRY))
    p.add_argument("--symbols", required=True, help="comma-separated, e.g. NVDA,SPY")
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--out-dir", default="journal/papertrade")
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        p.error("no symbols given")

    end = date.today()
    start = end - timedelta(days=args.days)
    source = HistoricalSource(
        symbols, start.isoformat(), end.isoformat(), cache_dir=args.cache_dir
    )
    engine = __import__("rhagent.engine", fromlist=["StrategyEngine"]).StrategyEngine(
        build(args.engine, {})
    )

    trader = PaperTrader(
        engine=engine, source=source, cost_bps=args.cost_bps,
        out_dir=args.out_dir,
    )
    run_dir = trader.run()
    _print_report(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: use a plain `from .engine import StrategyEngine` at module top instead of
the inline `__import__` if no circular-import issue arises (there shouldn't be:
`engine.py` does not import `papertrade.py`). Prefer the top-level import.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (existing suite + ~25 new)

- [ ] **Step 5: End-to-end smoke against real cached data**

Run: `.venv/bin/python -m rhagent.papertrade --engine mean_reversion --symbols NVDA,SPY --days 400`
Expected: prints run_id, aggregate stats table, failure buckets; a new
directory appears under `journal/papertrade/`.

Then: `.venv/bin/python -m rhagent.papertrade compare`
Expected: one-row table showing the run just recorded.

- [ ] **Step 6: Commit**

```bash
git add src/rhagent/papertrade.py tests/test_papertrade_cli.py
git commit -m "feat: papertrade CLI - run engines over history, print eval report, compare runs"
```

---

## Self-Review Notes

- Spec coverage: DecisionEngine (T1), event loop + seams + ledger + IDs (T2–3), four eval outputs (T4 + `_print_report`/`compare` in T5), CLI (T5), error boundaries (T3 `ValueError`s, T4 `FileNotFoundError`, T5 argparse), determinism (T3 test). World-model extensions and AgentEngine are explicitly future specs — no tasks here by design.
- `journal/papertrade/` output dirs: `.gitignore` already ignores journal artifacts? Check during T3; if `journal/` is tracked (runs.jsonl committed?), add `journal/papertrade/` to `.gitignore` in T3's commit.
- Types consistent: `Decision.target: float`, `decide(symbol, history, current_pos)`, trade-record field names identical in T3 writer and T4 tests.
