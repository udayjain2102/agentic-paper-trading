# Backtest & Strategy Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four rule-based strategies (mean-reversion, momentum, linear-regression, pairs) plus an offline backtest engine that ranks them by total return and picks a winner, then wire the winner into the runner behind the existing guardrails.

**Architecture:** A pure, offline `strategies/` package + `backtest.py` engine + `data.py` (RH-MCP fetch with CSV cache) + `compare.py` CLI ranks all strategies. The winner is recorded in `config.yaml`; a new strategy mode in `runner.py` computes today's target positions and emits orders through the existing `OrderExecutor` — no order bypasses `guardrails.py`.

**Tech Stack:** Python 3.14, pandas, numpy, PyYAML, pytest. Existing bot modules: `guardrails.py`, `executor.py`, `broker.py`, `runner.py`, `config.py`, `journal.py`.

## Global Constraints

- All strategy `positions()` outputs MUST obey the **no-lookahead invariant**: the position at day *t* uses only data up to and including day *t*. The engine applies that position to the day *t*→*t+1* return.
- Positions are a `pandas.Series` aligned to the bars index, values in `{-1, 0, +1}`.
- **Long-only by default.** Every strategy takes `allow_short: bool = False`; when False, a `-1` signal is clamped to `0`.
- The ranking metric is **total return**. Sharpe, max-drawdown, and hit-rate are reported but never decide the winner.
- Bars DataFrames are indexed by a `DatetimeIndex` named `date`, sorted ascending, with at least a float `close` column.
- No order path may bypass `OrderExecutor`/`validate_order`.
- Follow existing style: `from __future__ import annotations`, module docstrings, small focused files, tests under `tests/`.

---

### Task 1: Dependencies + strategy package skeleton + `Strategy` base

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `src/rhagent/strategies/__init__.py`
- Create: `src/rhagent/strategies/base.py`
- Test: `tests/strategies/__init__.py`, `tests/strategies/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Strategy` with class attr `name: str` and method `positions(self, bars: pd.DataFrame) -> pd.Series`.
  - `def clamp_short(pos: pd.Series, allow_short: bool) -> pd.Series` — maps `-1`→`0` when `allow_short` is False.

- [ ] **Step 1: Add deps and gitignore entry**

In `requirements.txt`, add after the `mcp>=1.2` line (before `# Dev`):
```
# Backtesting: strategies + engine work on price DataFrames.
pandas>=2.0
numpy>=1.26
```
In `.gitignore`, add a line:
```
data/
```

- [ ] **Step 2: Install deps**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: pandas + numpy install successfully.

- [ ] **Step 3: Write the failing test**

Create `tests/strategies/__init__.py` (empty). Create `tests/strategies/test_base.py`:
```python
import pandas as pd

from rhagent.strategies.base import Strategy, clamp_short


def test_clamp_short_zeros_negatives_when_long_only():
    pos = pd.Series([-1, 0, 1])
    out = clamp_short(pos, allow_short=False)
    assert list(out) == [0, 0, 1]


def test_clamp_short_keeps_negatives_when_allowed():
    pos = pd.Series([-1, 0, 1])
    out = clamp_short(pos, allow_short=True)
    assert list(out) == [-1, 0, 1]


def test_base_strategy_positions_not_implemented():
    s = Strategy()
    try:
        s.positions(pd.DataFrame({"close": [1.0]}))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/strategies/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: rhagent.strategies.base`.

- [ ] **Step 5: Implement the package + base**

Create `src/rhagent/strategies/__init__.py`:
```python
"""Rule-based trading strategies derived from the Quant Bible.

Each strategy is a pure function of a price-bar DataFrame; it produces a target
position series and never performs I/O. See ``base.Strategy``.
"""
```
Create `src/rhagent/strategies/base.py`:
```python
"""The common contract every strategy implements.

A strategy maps a DataFrame of daily bars to a target-position series with
values in {-1, 0, +1}, obeying the no-lookahead invariant (the position at day
t uses only data up to and including day t). The backtest engine applies that
position to the day t -> t+1 return.
"""

from __future__ import annotations

import pandas as pd


def clamp_short(pos: pd.Series, allow_short: bool) -> pd.Series:
    """Long-only guard: map short signals (-1) to flat (0) unless shorting is on."""
    if allow_short:
        return pos
    return pos.clip(lower=0)


class Strategy:
    name: str = "base"

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/strategies/test_base.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore src/rhagent/strategies/ tests/strategies/
git commit -m "feat: add strategy package skeleton, base contract, and backtest deps"
```

---

### Task 2: Mean-reversion strategy

**Files:**
- Create: `src/rhagent/strategies/mean_reversion.py`
- Test: `tests/strategies/test_mean_reversion.py`

**Interfaces:**
- Consumes: `Strategy`, `clamp_short` from `base`.
- Produces: `class MeanReversion(Strategy)` — `__init__(self, lookback=20, entry=1.0, exit=0.0, allow_short=False)`, `name = "mean_reversion"`, `positions(bars) -> pd.Series`.

- [ ] **Step 1: Write the failing test**

Create `tests/strategies/test_mean_reversion.py`:
```python
import numpy as np
import pandas as pd

from rhagent.strategies.mean_reversion import MeanReversion


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_goes_long_after_a_sharp_drop():
    # Flat then a big dip -> z drops below -entry -> long.
    prices = [100] * 20 + [90]
    s = MeanReversion(lookback=20, entry=1.0, exit=0.0)
    pos = s.positions(_bars(prices))
    assert pos.iloc[-1] == 1


def test_warmup_is_flat():
    prices = list(range(1, 10))  # fewer than lookback
    s = MeanReversion(lookback=20)
    pos = s.positions(_bars(prices))
    assert (pos == 0).all()


def test_no_lookahead_appending_future_bars_does_not_change_past():
    prices = [100] * 20 + [90]
    s = MeanReversion(lookback=20, entry=1.0, exit=0.0)
    short = s.positions(_bars(prices))
    long = s.positions(_bars(prices + [80, 120]))
    # positions for the original dates are unchanged by future bars.
    assert list(short.values) == list(long.iloc[: len(short)].values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/strategies/test_mean_reversion.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/rhagent/strategies/mean_reversion.py`:
```python
"""Z-score mean reversion: buy statistically-cheap dips, exit on reversion.

z = (close - rolling_mean) / rolling_std over ``lookback`` days. Enter long when
z < -entry; exit to flat when z >= -exit. Hysteresis (entry != exit) avoids
churning around the threshold. Long-only unless allow_short.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, clamp_short


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, lookback=20, entry=1.0, exit=0.0, allow_short=False):
        self.lookback = lookback
        self.entry = entry
        self.exit = exit
        self.allow_short = allow_short

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        z = (close - mean) / std

        pos = pd.Series(0, index=close.index, dtype=int)
        holding = 0  # +1 long, -1 short, 0 flat
        for t in close.index:
            zt = z[t]
            if pd.isna(zt):
                pos[t] = 0
                continue
            if holding == 0:
                if zt < -self.entry:
                    holding = 1
                elif zt > self.entry:
                    holding = -1
            elif holding == 1 and zt >= -self.exit:
                holding = 0
            elif holding == -1 and zt <= self.exit:
                holding = 0
            pos[t] = holding
        return clamp_short(pos, self.allow_short)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/strategies/test_mean_reversion.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/strategies/mean_reversion.py tests/strategies/test_mean_reversion.py
git commit -m "feat: add z-score mean-reversion strategy"
```

---

### Task 3: Momentum strategy

**Files:**
- Create: `src/rhagent/strategies/momentum.py`
- Test: `tests/strategies/test_momentum.py`

**Interfaces:**
- Consumes: `Strategy`, `clamp_short`.
- Produces: `class Momentum(Strategy)` — `__init__(self, lookback=40, allow_short=False)`, `name = "momentum"`, `positions(bars) -> pd.Series`.

- [ ] **Step 1: Write the failing test**

Create `tests/strategies/test_momentum.py`:
```python
import pandas as pd

from rhagent.strategies.momentum import Momentum


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_long_when_trailing_return_positive():
    prices = [100 + i for i in range(50)]  # steady uptrend
    pos = Momentum(lookback=40).positions(_bars(prices))
    assert pos.iloc[-1] == 1


def test_flat_when_trailing_return_negative():
    prices = [100 - i for i in range(50)]  # steady downtrend
    pos = Momentum(lookback=40).positions(_bars(prices))
    assert pos.iloc[-1] == 0  # long-only clamps the short signal to flat


def test_warmup_is_flat():
    prices = list(range(1, 10))
    pos = Momentum(lookback=40).positions(_bars(prices))
    assert (pos == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/strategies/test_momentum.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/rhagent/strategies/momentum.py`:
```python
"""Trend following: long when the trailing ``lookback``-day return is positive.

Signal is +1 (up-trend), -1 (down-trend), 0 (warmup). Long-only clamps -1 to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, clamp_short


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, lookback=40, allow_short=False):
        self.lookback = lookback
        self.allow_short = allow_short

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        trailing = close.pct_change(self.lookback)
        pos = pd.Series(np.sign(trailing), index=close.index)
        pos = pos.fillna(0).astype(int)
        return clamp_short(pos, self.allow_short)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/strategies/test_momentum.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/strategies/momentum.py tests/strategies/test_momentum.py
git commit -m "feat: add momentum strategy"
```

---

### Task 4: Linear-regression signal strategy

**Files:**
- Create: `src/rhagent/strategies/linreg.py`
- Test: `tests/strategies/test_linreg.py`

**Interfaces:**
- Consumes: `Strategy`, `clamp_short`.
- Produces: `class LinReg(Strategy)` — `__init__(self, min_train=40, allow_short=False)`, `name = "linreg"`, `positions(bars) -> pd.Series`.

**Design note:** features known at day *t* are `[1, ret_lag1, ret_lag2, ma_ratio]` where `ret_lag1 = close.pct_change()`, `ret_lag2 = close.pct_change().shift(1)`, `ma_ratio = close / close.rolling(10).mean() - 1`. Target is next-day return `close.pct_change().shift(-1)`. At decision day *t*, train OLS on all rows *s* whose target is already realized (`s <= t-1`), then predict the target from features at *t*. Long if prediction > 0.

- [ ] **Step 1: Write the failing test**

Create `tests/strategies/test_linreg.py`:
```python
import numpy as np
import pandas as pd

from rhagent.strategies.linreg import LinReg


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_predicts_up_in_a_persistent_uptrend():
    # Compounding uptrend: positive-return autocorrelation -> predicts long.
    prices = [100 * (1.01 ** i) for i in range(80)]
    pos = LinReg(min_train=40).positions(_bars(prices))
    assert pos.iloc[-1] == 1


def test_warmup_is_flat():
    prices = [100 * (1.01 ** i) for i in range(30)]  # below min_train
    pos = LinReg(min_train=40).positions(_bars(prices))
    assert (pos == 0).all()


def test_no_lookahead_appending_future_bars_does_not_change_past():
    prices = [100 * (1.01 ** i) for i in range(80)]
    s = LinReg(min_train=40)
    short = s.positions(_bars(prices))
    longer = s.positions(_bars(prices + [200, 150, 300]))
    assert list(short.values) == list(longer.iloc[: len(short)].values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/strategies/test_linreg.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/rhagent/strategies/linreg.py`:
```python
"""Linear-regression signal: predict next-day return via rolling OLS.

Features known at day t: [1, ret_lag1, ret_lag2, ma_ratio]. Target: next-day
return. At each day t we fit OLS on rows whose target is already realized
(strictly before t) and predict day t's next-day return. Long when the
prediction is positive. The expanding train window uses only past data, so
there is no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, clamp_short


class LinReg(Strategy):
    name = "linreg"

    def __init__(self, min_train=40, allow_short=False):
        self.min_train = min_train
        self.allow_short = allow_short

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        ret = close.pct_change()
        feats = pd.DataFrame(
            {
                "bias": 1.0,
                "ret_lag1": ret,
                "ret_lag2": ret.shift(1),
                "ma_ratio": close / close.rolling(10).mean() - 1.0,
            }
        )
        target = ret.shift(-1)  # next-day return, realized at the following day

        cols = ["bias", "ret_lag1", "ret_lag2", "ma_ratio"]
        pos = pd.Series(0, index=close.index, dtype=int)
        n = len(close)
        for i in range(n):
            # Rows usable for training at decision day i: target realized, i.e.
            # index j with j <= i-1 and all feature/target values present.
            train = feats.iloc[: i].copy()
            train["y"] = target.iloc[: i]
            train = train.dropna()
            x_now = feats.iloc[i][cols]
            if len(train) < self.min_train or x_now.isna().any():
                continue
            X = train[cols].to_numpy()
            y = train["y"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = float(x_now.to_numpy() @ beta)
            pos.iloc[i] = int(np.sign(pred))
        return clamp_short(pos, self.allow_short)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/strategies/test_linreg.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/strategies/linreg.py tests/strategies/test_linreg.py
git commit -m "feat: add rolling linear-regression signal strategy"
```

---

### Task 5: Pairs strategy

**Files:**
- Create: `src/rhagent/strategies/pairs.py`
- Test: `tests/strategies/test_pairs.py`

**Interfaces:**
- Consumes: `clamp_short` from `base`.
- Produces: `class Pairs` — `__init__(self, lookback=20, entry=1.0, allow_short=False)`, `name = "pairs"`, method `positions_pair(self, bars_a: pd.DataFrame, bars_b: pd.DataFrame) -> tuple[pd.Series, pd.Series]` returning `(pos_a, pos_b)`.

**Design note:** Pairs is two-symbol, so it does NOT implement the single-symbol `Strategy.positions`. `compare.py` (Task 7) and nothing else consumes `positions_pair`. `spread = log(close_a) - log(close_b)`, z-scored over `lookback`. When `z > entry` (A rich vs B): short A / long B → `pos_a=-1, pos_b=+1`. When `z < -entry`: `pos_a=+1, pos_b=-1`. Else flat. `clamp_short` applied to each leg.

- [ ] **Step 1: Write the failing test**

Create `tests/strategies/test_pairs.py`:
```python
import numpy as np
import pandas as pd

from rhagent.strategies.pairs import Pairs


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_long_a_when_a_cheap_relative_to_b():
    # A and B move together, then A dips -> spread negative -> long A leg.
    a = [100] * 20 + [90]
    b = [100] * 21
    pa, pb = Pairs(lookback=20, entry=1.0, allow_short=True).positions_pair(
        _bars(a), _bars(b)
    )
    assert pa.iloc[-1] == 1
    assert pb.iloc[-1] == -1


def test_long_only_clamps_short_leg_to_flat():
    a = [100] * 20 + [90]
    b = [100] * 21
    pa, pb = Pairs(lookback=20, entry=1.0, allow_short=False).positions_pair(
        _bars(a), _bars(b)
    )
    assert pa.iloc[-1] == 1
    assert pb.iloc[-1] == 0  # short leg clamped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/strategies/test_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/rhagent/strategies/pairs.py`:
```python
"""Pairs trading: trade the mean-reverting spread between two correlated names.

spread = log(close_a) - log(close_b), z-scored over ``lookback``. When A is rich
vs B (z > entry): short A / long B. When A is cheap (z < -entry): long A / short
B. Long-only clamps each short leg to flat (so only the cheap leg trades).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import clamp_short


class Pairs:
    name = "pairs"

    def __init__(self, lookback=20, entry=1.0, allow_short=False):
        self.lookback = lookback
        self.entry = entry
        self.allow_short = allow_short

    def positions_pair(self, bars_a, bars_b):
        a = bars_a["close"].astype(float)
        b = bars_b["close"].astype(float)
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]

        spread = np.log(a) - np.log(b)
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        z = (spread - mean) / std

        pos_a = pd.Series(0, index=idx, dtype=int)
        for t in idx:
            zt = z[t]
            if pd.isna(zt):
                continue
            if zt < -self.entry:
                pos_a[t] = 1
            elif zt > self.entry:
                pos_a[t] = -1
        pos_b = -pos_a
        return (
            clamp_short(pos_a, self.allow_short),
            clamp_short(pos_b, self.allow_short),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/strategies/test_pairs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/strategies/pairs.py tests/strategies/test_pairs.py
git commit -m "feat: add pairs-trading strategy"
```

---

### Task 6: Backtest engine

**Files:**
- Create: `src/rhagent/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (works on any bars + positions).
- Produces:
  - `@dataclass BacktestResult` with fields `equity: pd.Series`, `total_return: float`, `sharpe: float`, `max_drawdown: float`, `hit_rate: float`, `n_days: int`.
  - `def net_returns(bars: pd.DataFrame, positions: pd.Series, cost_bps: float = 1.0) -> pd.Series` — per-day net strategy return (position at t earns t→t+1 return, minus turnover cost); last day (no forward return) dropped.
  - `def result_from_returns(net: pd.Series) -> BacktestResult`.
  - `def run(bars: pd.DataFrame, positions: pd.Series, cost_bps: float = 1.0) -> BacktestResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd

from rhagent.backtest import net_returns, result_from_returns, run


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_always_long_earns_the_buy_and_hold_return():
    prices = [100, 110, 121]  # +10% then +10%
    bars = _bars(prices)
    pos = pd.Series([1, 1, 1], index=bars.index)
    res = run(bars, pos, cost_bps=0.0)
    # Position on the last day is dropped (no forward return): two +10% steps.
    assert res.total_return == pytest_approx(0.21)
    assert res.n_days == 2


def test_flat_earns_nothing():
    bars = _bars([100, 110, 121])
    pos = pd.Series([0, 0, 0], index=bars.index)
    res = run(bars, pos, cost_bps=0.0)
    assert res.total_return == 0.0


def test_costs_reduce_return():
    bars = _bars([100, 110, 121])
    pos = pd.Series([1, 1, 1], index=bars.index)
    gross = run(bars, pos, cost_bps=0.0).total_return
    netted = run(bars, pos, cost_bps=50.0).total_return
    assert netted < gross


def pytest_approx(x):
    import pytest

    return pytest.approx(x, rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: rhagent.backtest`.

- [ ] **Step 3: Implement**

Create `src/rhagent/backtest.py`:
```python
"""Offline backtest engine.

Turns a target-position series into a net-return series and summary metrics.
Mechanics: the position held on day t earns the return from day t to t+1, so the
final day (which has no forward return) is dropped. A per-trade cost in basis
points is charged on turnover (absolute change in position).

This module does no I/O and knows nothing about strategies — it just scores a
positions series against prices. Ranking uses ``total_return``; the other
metrics are reported for context only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ANNUALIZATION = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    total_return: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    n_days: int


def net_returns(bars: pd.DataFrame, positions: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    close = bars["close"].astype(float)
    fwd = close.pct_change().shift(-1)  # return from t to t+1, indexed at t
    pos = positions.reindex(close.index).fillna(0).astype(float)

    turnover = pos.diff().abs()
    if len(pos):
        turnover.iloc[0] = abs(pos.iloc[0])
    cost = turnover * (cost_bps / 1e4)

    net = pos * fwd - cost
    return net[fwd.notna()].fillna(0.0)


def result_from_returns(net: pd.Series) -> BacktestResult:
    if len(net) == 0:
        empty = pd.Series(dtype=float)
        return BacktestResult(empty, 0.0, 0.0, 0.0, 0.0, 0)

    equity = (1.0 + net).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    std = net.std()
    sharpe = (
        float(net.mean() / std * np.sqrt(_ANNUALIZATION))
        if std and not pd.isna(std) and std > 0
        else 0.0
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    active = net[net != 0]
    hit_rate = float((active > 0).mean()) if len(active) else 0.0

    return BacktestResult(
        equity=equity,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        n_days=int(len(net)),
    )


def run(bars: pd.DataFrame, positions: pd.Series, cost_bps: float = 1.0) -> BacktestResult:
    return result_from_returns(net_returns(bars, positions, cost_bps))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/backtest.py tests/test_backtest.py
git commit -m "feat: add offline backtest engine"
```

---

### Task 7: Data fetch + CSV cache

**Files:**
- Create: `src/rhagent/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `def rows_to_df(rows: list[dict]) -> pd.DataFrame` — normalized rows (`{"date","open","high","low","close","volume"}`) → DatetimeIndex-named-`date` DataFrame sorted ascending.
  - `def get_bars(symbols, start, end, *, fetch=None, cache_dir="data") -> dict[str, pd.DataFrame]` — cache-first; on miss calls `fetch(missing, start, end) -> dict[str, list[dict]]`, writes `<cache_dir>/<SYMBOL>.csv`, returns per-symbol DataFrames. `fetch` defaults to `mcp_fetch`.
  - `def mcp_fetch(symbols, start, end) -> dict[str, list[dict]]` — thin RH-MCP integration point (see note).

**Integration note (mirrors `broker.py`):** `mcp_fetch` calls the RH MCP `get_equity_historicals` (interval `day`, `adjustment_type=split`) and normalizes each bar to the row dict shape above. The exact upstream field names (e.g. `begins_at`, `close_price`) must be confirmed against the live server and adjusted here only. Unit tests never call `mcp_fetch`; they inject a fake `fetch` or pre-seed the cache.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:
```python
import pandas as pd

from rhagent.data import get_bars, rows_to_df


FIXTURE = {
    "AAPL": [
        {"date": "2025-01-03", "open": 1, "high": 2, "low": 1, "close": 191.0, "volume": 10},
        {"date": "2025-01-02", "open": 1, "high": 2, "low": 1, "close": 190.0, "volume": 10},
    ]
}


def test_rows_to_df_sorts_and_indexes_by_date():
    df = rows_to_df(FIXTURE["AAPL"])
    assert df.index.name == "date"
    assert list(df["close"]) == [190.0, 191.0]  # sorted ascending by date


def test_get_bars_fetches_then_caches(tmp_path):
    calls = []

    def fake_fetch(symbols, start, end):
        calls.append(list(symbols))
        return FIXTURE

    out = get_bars(["AAPL"], "2025-01-01", "2025-02-01", fetch=fake_fetch, cache_dir=tmp_path)
    assert out["AAPL"]["close"].iloc[-1] == 191.0
    assert (tmp_path / "AAPL.csv").exists()

    # Second call is served from cache — fetch not invoked again.
    out2 = get_bars(["AAPL"], "2025-01-01", "2025-02-01", fetch=fake_fetch, cache_dir=tmp_path)
    assert out2["AAPL"]["close"].iloc[-1] == 191.0
    assert calls == [["AAPL"]]  # only the first call fetched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: rhagent.data`.

- [ ] **Step 3: Implement**

Create `src/rhagent/data.py`:
```python
"""Historical price data: fetch from the Robinhood MCP, cache to CSV.

Cache-first: if ``<cache_dir>/<SYMBOL>.csv`` exists it is read; otherwise bars are
fetched, normalized, and written. This keeps backtests reproducible and offline,
and confines the live-MCP shape to ``mcp_fetch`` (a thin integration point, like
``McpBroker``). Tests inject a fake ``fetch`` or pre-seed the cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_COLUMNS = ["open", "high", "low", "close", "volume"]


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "date"
    for col in _COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index.name = "date"
    return df


def get_bars(symbols, start, end, *, fetch=None, cache_dir="data") -> dict[str, pd.DataFrame]:
    fetch = fetch or mcp_fetch
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, pd.DataFrame] = {}
    missing = []
    for s in symbols:
        path = cache_dir / f"{s}.csv"
        if path.exists():
            out[s] = _read_csv(path)
        else:
            missing.append(s)

    if missing:
        fetched = fetch(missing, start, end)
        for s, rows in fetched.items():
            df = rows_to_df(rows)
            df.to_csv(cache_dir / f"{s}.csv")
            out[s] = df
    return out


def mcp_fetch(symbols, start, end) -> dict[str, list[dict]]:
    """Fetch daily bars from the RH MCP. Integration point — confirm field names.

    Requires a configured MCP session (ROBINHOOD_MCP_TOKEN). Raises if unavailable
    so that offline runs rely on the CSV cache instead.
    """
    from .config import load
    from .mcp_session import mcp_session

    cfg = load()
    with mcp_session(cfg.mcp_url, cfg.mcp_token) as session:
        import anyio

        result = anyio.from_thread.run(
            session.call_tool,
            "get_equity_historicals",
            {
                "symbols": list(symbols),
                "start_time": f"{start}T00:00:00Z",
                "end_time": f"{end}T00:00:00Z",
                "interval": "day",
                "adjustment_type": "split",
            },
        )
    from .broker import _structured

    data = _structured(result)
    return _normalize(data, symbols)


def _normalize(data: dict, symbols) -> dict[str, list[dict]]:
    """Map the RH historicals payload to per-symbol normalized row lists.

    NOTE: upstream field names are placeholders — confirm against the live server
    and adjust here only. Expected upstream: a list of per-symbol result objects,
    each with a list of bars carrying a begin timestamp and OHLCV prices.
    """
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    results = data.get("results") or data.get("data") or []
    for entry in results:
        sym = entry.get("symbol")
        if sym not in out:
            continue
        for bar in entry.get("historicals", []) or entry.get("bars", []):
            out[sym].append(
                {
                    "date": (bar.get("begins_at") or bar.get("date"))[:10],
                    "open": float(bar.get("open_price", bar.get("open", 0)) or 0),
                    "high": float(bar.get("high_price", bar.get("high", 0)) or 0),
                    "low": float(bar.get("low_price", bar.get("low", 0)) or 0),
                    "close": float(bar.get("close_price", bar.get("close", 0)) or 0),
                    "volume": float(bar.get("volume", 0) or 0),
                }
            )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/data.py tests/test_data.py
git commit -m "feat: add historical data fetch with CSV cache"
```

---

### Task 8: Strategy registry + comparison CLI

**Files:**
- Modify: `src/rhagent/strategies/__init__.py`
- Create: `src/rhagent/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `MeanReversion`, `Momentum`, `LinReg`, `Pairs`; `data.get_bars`; `backtest.net_returns`, `result_from_returns`, `BacktestResult`.
- Produces (in `strategies/__init__.py`):
  - `REGISTRY: dict[str, type]` mapping `"mean_reversion"|"momentum"|"linreg"` → class (single-symbol strategies only).
  - `def build(name: str, params: dict) -> Strategy`.
- Produces (in `compare.py`):
  - `def evaluate(bars_by_symbol: dict[str, pd.DataFrame], cost_bps: float = 1.0) -> list[tuple[str, BacktestResult]]` — one aggregated result per strategy (the three single-symbol strategies equal-weighted across symbols, plus the best-correlated pair), sorted by `total_return` descending.
  - `def best_pair(bars_by_symbol) -> tuple[str, str]`.
  - `def main() -> int` — CLI entry; `python -m rhagent.compare`.
  - `UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare.py`:
```python
import numpy as np
import pandas as pd

from rhagent.compare import best_pair, evaluate


def _bars(prices, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def _universe():
    rng = np.random.default_rng(0)
    up = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    down = 100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 120))
    return {
        "AAPL": _bars(up),
        "MSFT": _bars(up * 1.01),  # highly correlated with AAPL
        "NVDA": _bars(down),
        "SPY": _bars(up * 0.5 + 50),
    }


def test_evaluate_returns_one_row_per_strategy_sorted_by_return():
    rows = evaluate(_universe(), cost_bps=1.0)
    names = [name for name, _ in rows]
    assert set(names) == {"mean_reversion", "momentum", "linreg", "pairs"}
    returns = [res.total_return for _, res in rows]
    assert returns == sorted(returns, reverse=True)  # descending


def test_best_pair_picks_the_two_most_correlated():
    pair = best_pair(_universe())
    assert set(pair) == {"AAPL", "MSFT"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: rhagent.compare`.

- [ ] **Step 3: Add the registry**

Append to `src/rhagent/strategies/__init__.py`:
```python
from .base import Strategy
from .linreg import LinReg
from .mean_reversion import MeanReversion
from .momentum import Momentum

# Single-symbol strategies only. Pairs is two-symbol and handled separately.
REGISTRY: dict[str, type] = {
    MeanReversion.name: MeanReversion,
    Momentum.name: Momentum,
    LinReg.name: LinReg,
}


def build(name: str, params: dict) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
```

- [ ] **Step 4: Implement `compare.py`**

Create `src/rhagent/compare.py`:
```python
"""Run every strategy over the universe, rank by total return, pick the winner.

    python -m rhagent.compare

The three single-symbol strategies are evaluated per symbol and equal-weighted
into one net-return series each; pairs is evaluated on the most-correlated pair.
Ranking is by total return; Sharpe, max drawdown, and hit-rate are shown for
context. The top row is the winner, printed with a ready-to-paste config block.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

from .backtest import BacktestResult, net_returns, result_from_returns
from .data import get_bars
from .strategies import REGISTRY, build
from .strategies.pairs import Pairs

UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY"]


def best_pair(bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[str, str]:
    closes = pd.DataFrame(
        {s: b["close"] for s, b in bars_by_symbol.items()}
    ).dropna()
    corr = closes.pct_change().dropna().corr()
    best, best_val = None, -2.0
    syms = list(corr.columns)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = corr.iloc[i, j]
            if c > best_val:
                best_val, best = c, (syms[i], syms[j])
    return best


def _aggregate(nets: list[pd.Series]) -> BacktestResult:
    combined = pd.concat(nets, axis=1).mean(axis=1).dropna()
    return result_from_returns(combined)


def evaluate(
    bars_by_symbol: dict[str, pd.DataFrame], cost_bps: float = 1.0
) -> list[tuple[str, BacktestResult]]:
    rows: list[tuple[str, BacktestResult]] = []

    for name in REGISTRY:
        strat = build(name, {})
        nets = [
            net_returns(bars, strat.positions(bars), cost_bps)
            for bars in bars_by_symbol.values()
        ]
        rows.append((name, _aggregate(nets)))

    a, b = best_pair(bars_by_symbol)
    pa, pb = Pairs().positions_pair(bars_by_symbol[a], bars_by_symbol[b])
    pair_nets = [
        net_returns(bars_by_symbol[a], pa, cost_bps),
        net_returns(bars_by_symbol[b], pb, cost_bps),
    ]
    rows.append(("pairs", _aggregate(pair_nets)))

    rows.sort(key=lambda r: r[1].total_return, reverse=True)
    return rows


def main() -> int:
    end = date.today()
    start = end - timedelta(days=400)
    bars = get_bars(UNIVERSE, start.isoformat(), end.isoformat())

    rows = evaluate(bars)
    print(f"{'strategy':<16}{'total_ret':>12}{'sharpe':>10}{'max_dd':>10}{'hit':>8}")
    for name, res in rows:
        print(
            f"{name:<16}{res.total_return:>11.2%}{res.sharpe:>10.2f}"
            f"{res.max_drawdown:>10.2%}{res.hit_rate:>8.2%}"
        )

    winner, wres = rows[0]
    print(f"\nWinner (by total return): {winner} ({wres.total_return:.2%})")
    if winner == "pairs":
        a, b = best_pair(bars)
        print(f"Chosen pair: {a}/{b}. Long-only trades only the cheap leg.")
        print("Note: pairs is not yet supported in live strategy mode.")
    else:
        print("Add this to config.yaml:\n")
        print("strategy:")
        print(f"  name: {winner}")
        print("  params: {}")
        print(f"  universe: {UNIVERSE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compare.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/rhagent/strategies/__init__.py src/rhagent/compare.py tests/test_compare.py
git commit -m "feat: add strategy registry and comparison CLI"
```

---

### Task 9: Pure order-generation for strategy mode

**Files:**
- Create: `src/rhagent/strategy_runner.py`
- Test: `tests/test_strategy_runner.py`

**Interfaces:**
- Consumes: `Strategy` (via `strategies.build`); bars dicts from `data`.
- Produces:
  - `def target_orders(strategy, bars_by_symbol: dict[str, pd.DataFrame], held: set[str], notional_usd: float) -> list[tuple[str, str, float]]` — computes each symbol's latest target position, then: target `1` and not held → `(symbol, "buy", notional_usd)`; target `0` and held → `(symbol, "sell", notional_usd)`; otherwise no order. Returns a list of `(symbol, side, notional_usd)`.

**Design note:** This is the pure, testable core of strategy mode. The runner (Task 10) fetches bars, calls `target_orders`, and feeds each tuple to `OrderExecutor.execute`. Keeping it pure means no broker/journal is needed to test the decision logic.

- [ ] **Step 1: Write the failing test**

Create `tests/test_strategy_runner.py`:
```python
import pandas as pd

from rhagent.strategies.momentum import Momentum
from rhagent.strategy_runner import target_orders


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D", name="date")
    return pd.DataFrame({"close": [float(p) for p in prices]}, index=idx)


def test_buys_when_signal_long_and_not_held():
    bars = {"AAPL": _bars([100 + i for i in range(50)])}  # uptrend -> long
    orders = target_orders(Momentum(lookback=40), bars, held=set(), notional_usd=250)
    assert orders == [("AAPL", "buy", 250)]


def test_sells_when_signal_flat_and_held():
    bars = {"AAPL": _bars([100 - i for i in range(50)])}  # downtrend -> flat
    orders = target_orders(Momentum(lookback=40), bars, held={"AAPL"}, notional_usd=250)
    assert orders == [("AAPL", "sell", 250)]


def test_no_order_when_already_in_desired_state():
    bars = {"AAPL": _bars([100 + i for i in range(50)])}  # long, already held
    orders = target_orders(Momentum(lookback=40), bars, held={"AAPL"}, notional_usd=250)
    assert orders == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_strategy_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: rhagent.strategy_runner`.

- [ ] **Step 3: Implement**

Create `src/rhagent/strategy_runner.py`:
```python
"""Pure order generation for strategy mode.

Given a strategy and recent bars per symbol, compute each symbol's latest target
position and diff it against what's currently held to produce buy/sell orders.
No I/O: the runner feeds the returned tuples through OrderExecutor so the same
guardrails apply as in the LLM path.
"""

from __future__ import annotations

import pandas as pd


def target_orders(
    strategy, bars_by_symbol: dict[str, pd.DataFrame], held: set[str], notional_usd: float
) -> list[tuple[str, str, float]]:
    orders: list[tuple[str, str, float]] = []
    for symbol, bars in bars_by_symbol.items():
        pos = strategy.positions(bars)
        if len(pos) == 0:
            continue
        target = int(pos.iloc[-1])
        is_held = symbol in held
        if target == 1 and not is_held:
            orders.append((symbol, "buy", notional_usd))
        elif target == 0 and is_held:
            orders.append((symbol, "sell", notional_usd))
    return orders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_strategy_runner.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/strategy_runner.py tests/test_strategy_runner.py
git commit -m "feat: add pure order-generation for strategy mode"
```

---

### Task 10: Config `StrategyConfig` + runner strategy mode

**Files:**
- Modify: `src/rhagent/config.py`
- Modify: `config.yaml`
- Modify: `src/rhagent/runner.py`
- Test: `tests/test_runner_strategy_mode.py`

**Interfaces:**
- Consumes: `data.get_bars`, `strategies.build`, `strategy_runner.target_orders`, `OrderExecutor`.
- Produces:
  - `config.py`: `@dataclass(frozen=True) StrategyConfig` with `name: str`, `params: dict`, `universe: list`; `Config.strategy: StrategyConfig | None = None`; `load()` populates it from an optional `strategy:` block.
  - `runner.py`: `def run_strategy_mode(cfg, broker, executor, journal, *, fetch=None) -> str`; `run()` dispatches to it when `STRATEGY_MODE=true`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_strategy_mode.py`:
```python
import pandas as pd

from rhagent.broker import MockBroker
from rhagent.config import StrategyConfig
from rhagent.executor import OrderExecutor
from rhagent.guardrails import Limits, RunState
from rhagent.journal import Journal
from rhagent import runner


def _limits():
    return Limits(
        per_trade_max_usd=250,
        total_deployed_max_usd=2000,
        max_new_positions_per_run=2,
        max_orders_per_run=5,
        daily_loss_limit_usd=200,
    )


class _Cfg:
    def __init__(self, strategy):
        self.strategy = strategy
        self.dry_run = True


def test_strategy_mode_dry_run_places_nothing(tmp_path):
    broker = MockBroker(quotes={"AAPL": 100.0})
    journal = Journal(tmp_path / "runs.jsonl")
    ex = OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=_limits(),
        run_state=RunState(),
        journal=journal,
        dry_run=True,
    )
    cfg = _Cfg(StrategyConfig(name="momentum", params={"lookback": 40}, universe=["AAPL"]))

    def fake_fetch(symbols, start, end):
        return {
            "AAPL": [
                {"date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                 "open": 0, "high": 0, "low": 0, "close": 100 + i, "volume": 0}
                for i in range(50)
            ]
        }

    summary = runner.run_strategy_mode(
        cfg, broker, ex, journal, fetch=fake_fetch
    )
    assert broker.placed == []  # dry-run: nothing reaches the broker
    assert "AAPL" in summary  # the buy was proposed and logged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner_strategy_mode.py -v`
Expected: FAIL with `AttributeError: module 'rhagent.runner' has no attribute 'run_strategy_mode'` (and `ImportError` for `StrategyConfig`).

- [ ] **Step 3: Add `StrategyConfig` to config.py**

In `src/rhagent/config.py`, add after the `AgentConfig` dataclass:
```python
@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict
    universe: list
```
Add a field to `Config` (append after `nvidia_base_url`):
```python
    strategy: "StrategyConfig | None" = None
```
In `load()`, before the `return Config(...)`, add:
```python
    strategy = (
        StrategyConfig(**raw["strategy"]) if raw.get("strategy") else None
    )
```
and pass `strategy=strategy` into the `Config(...)` constructor.

- [ ] **Step 4: Add a commented strategy block to config.yaml**

Append to `config.yaml`:
```yaml

# Winning strategy, filled in by `python -m rhagent.compare`. When present and
# STRATEGY_MODE=true, the runner trades this strategy through the same guardrails
# as the LLM path. Leave commented to keep the default LLM-agent behavior.
# strategy:
#   name: momentum
#   params: {}
#   universe: [AAPL, MSFT, NVDA, SPY]
```

- [ ] **Step 5: Add `run_strategy_mode` and dispatch in runner.py**

In `src/rhagent/runner.py`, add this function above `run()`:
```python
def run_strategy_mode(cfg, broker, executor, journal, *, fetch=None) -> str:
    """Trade the configured winning strategy through the executor/guardrails."""
    from datetime import date, timedelta

    from .data import get_bars
    from .strategies import build
    from .strategy_runner import target_orders

    sc = cfg.strategy
    strategy = build(sc.name, sc.params)
    end = date.today()
    start = end - timedelta(days=200)
    bars = get_bars(sc.universe, start.isoformat(), end.isoformat(), fetch=fetch)

    held = set(broker.get_account().positions)
    per_trade = getattr(cfg, "limits", None)
    notional = per_trade.per_trade_max_usd if per_trade else 250
    orders = target_orders(strategy, bars, held, notional)

    lines = [f"[strategy:{sc.name}] {len(orders)} order(s) proposed"]
    for symbol, side, amount in orders:
        result = executor.execute(symbol, side, amount)
        lines.append(f"{symbol} {side} {amount} -> {result.as_tool_text()}")
    journal.record("strategy_run", name=sc.name, n_orders=len(orders))
    return "\n".join(lines)
```
Then in `run()`, after the `executor = OrderExecutor(...)` block and before the `if os.environ.get("MOCK_AGENT"...)` block, add:
```python
        if os.environ.get("STRATEGY_MODE", "").strip().lower() == "true":
            if cfg.strategy is None:
                raise SystemExit(
                    "STRATEGY_MODE=true but no `strategy:` block in config.yaml. "
                    "Run `python -m rhagent.compare` to pick one."
                )
            summary = run_strategy_mode(cfg, broker, executor, journal)
            journal.record("run_end", mode=mode, summary=summary)
            print(f"[{mode}] Run complete.\n{summary}")
            return 0
```

- [ ] **Step 6: Run the new test + full suite**

Run: `.venv/bin/python -m pytest tests/test_runner_strategy_mode.py -v`
Expected: PASS (1 test).
Run: `.venv/bin/python -m pytest`
Expected: all tests PASS (existing + new).

- [ ] **Step 7: Commit**

```bash
git add src/rhagent/config.py config.yaml src/rhagent/runner.py tests/test_runner_strategy_mode.py
git commit -m "feat: wire winning strategy into runner behind guardrails"
```

---

### Task 11: Docs — README update

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the layout table and add a Backtesting section**

In `README.md`, add these rows to the Layout table (after the `config.yaml` row):
```markdown
| `src/rhagent/strategies/` | Rule-based strategies (mean-reversion, momentum, linreg, pairs). |
| `src/rhagent/backtest.py` | Offline backtest engine (equity curve + metrics). |
| `src/rhagent/data.py` | Historical bars via the RH MCP, cached to `data/*.csv`. |
| `src/rhagent/compare.py` | Rank all strategies by total return, pick the winner. |
| `src/rhagent/strategy_runner.py` | Turns a strategy's target positions into orders. |
```
Add a new section before `## Safety`:
```markdown
## Backtesting & strategy mode

Rank the four strategies over ~1yr of daily bars and pick the best by total return:

```bash
.venv/bin/python -m rhagent.compare
```

It caches price data under `data/` (gitignored). Paste the printed `strategy:`
block into `config.yaml`, then run the winner through the normal guardrails:

```bash
STRATEGY_MODE=true .venv/bin/python -m rhagent.runner
```

Strategy mode is dry-run unless `LIVE=true`, and every order it emits passes
through the same `OrderExecutor`/guardrails as the LLM path.
```
Also update the "Out of scope (v1)" line to remove `backtesting,`.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document backtesting and strategy mode"
```

---

## Self-Review

**Spec coverage:**
- Four strategies → Tasks 2–5. ✓
- No-lookahead invariant → tested in Tasks 2, 4 (mean-reversion, linreg); momentum/pairs are stateless-by-construction. ✓
- Ranking by total return → Task 8 `evaluate` sorts by `total_return`; other metrics reported. ✓
- Data from RH MCP, cached to CSV → Task 7. ✓
- Universe AAPL/MSFT/NVDA/SPY, ~1yr → Task 8 `UNIVERSE` + `compare.main` 400-day range. ✓
- Long-only default, shorting flag → `clamp_short` (Task 1), used by every strategy. ✓
- Winner recorded in config + runner strategy mode through guardrails → Tasks 9–10. ✓
- Testing (unit per strategy, engine, data fixture, integration smoke) → Tasks 2–10. ✓
- Deps pandas/numpy, `data/` gitignored → Task 1. ✓
- Docs → Task 11. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The `_normalize` field names in Task 7 are explicitly flagged as an integration point to confirm (consistent with the existing `broker.py` pattern), not a placeholder gap.

**Type consistency:** `positions() -> pd.Series` everywhere; `net_returns`/`result_from_returns`/`run` signatures match between Task 6 and their consumers in Task 8; `target_orders` signature matches between Task 9 and its caller in Task 10; `StrategyConfig(name, params, universe)` consistent between Task 10 definition and Task 8's printed config block.
