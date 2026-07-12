# Cross-Sectional IC/ICIR Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a strategy's signal has a real, consistent cross-sectional edge — a continuous per-name `signal()`, a ~60-name universe, and correct market-relative rank-IC / ICIR / decay, computed on an in-sample slice with the out-of-sample slice locked away.

**Architecture:** New package `src/rhagent/factor/`. Strategies gain a continuous `signal(bars) -> Series` alongside their existing discrete `positions()`. `universe.py` loads aligned bar/close panels; `ic.py` holds the pure IC math (forward returns, rank-IC, ICIR, decay); `split.py` owns the locked in-sample/out-of-sample boundary; `signals.py` builds the signal panel; `__main__.py` is the CLI.

**Tech Stack:** Python 3, pandas, numpy, pytest. No new dependencies (rank-IC is computed via `pandas.rank` + `numpy.corrcoef`, so no scipy).

**Spec:** `docs/superpowers/specs/2026-07-12-cross-sectional-ic-evaluator-design.md`

## Global Constraints

- `signal(bars) -> pd.Series`: continuous, higher = more bullish on forward return; aligned to `bars.index`; NaN during warmup; no lookahead (value at t uses only bars up to t).
- Rank-IC = Pearson correlation of the cross-sectional **ranks** of signal vs forward return (Spearman), computed without scipy.
- Rank-IC is inherently invariant to a common additive shift in returns, so it is already market-relative — no separate demeaning step is needed; this property is asserted by a test. (This reconciles the spec's "market-neutralize" intent: under rank-IC, demeaning is a mathematical no-op.)
- `min_names` default 10: a day with fewer valid (signal, return) name-pairs yields NaN IC and is dropped from the series.
- `ICIR = mean(IC) / std(IC)` with population std (`ddof=0`); returns `0.0` if the series is empty or has zero variance.
- ICIR interpretation bands (CLI labels): `>0.5` strong, `0.3–0.5` moderate, `<0.3` likely noise.
- Decay horizons: `(1, 5, 10, 20, 50)`. `half_life` = the smallest horizon where `|mean IC_h| <= 0.5 * |mean IC_1|`; `">50"` if it never falls that far; `None` if `mean IC_1` is ~0.
- OOS: `cutoff` = the date at the `1 - oos_frac` quantile of the sorted unique dates (default `oos_frac=0.25`). In-sample day t is valid only if `t < cutoff` AND the date `horizon` bars ahead is `< cutoff`. The OOS slice (`date >= cutoff`) is never read in this sub-project.
- `forward_returns` lives in `ic.py` (pure IC math), not `universe.py`, to avoid a `universe`↔`ic` import cycle. (Spec listed it under `universe.py`; moved for layering.)
- `positions()` behavior is unchanged. `LinReg` is refactored to share a `_predictions()` helper between `positions()` and `signal()` (behavior-preserving; existing linreg tests must still pass).
- Style: `from __future__ import annotations`, module docstrings, small focused files, matching existing `src/rhagent/` conventions. Tests under `tests/`, run with `.venv/bin/python -m pytest`.

---

### Task 1: Continuous `signal()` on strategies

**Files:**
- Modify: `src/rhagent/strategies/base.py`, `src/rhagent/strategies/mean_reversion.py`, `src/rhagent/strategies/momentum.py`, `src/rhagent/strategies/linreg.py`
- Test: `tests/strategies/test_signal.py`

**Interfaces:**
- Consumes: existing `Strategy`, `MeanReversion`, `Momentum`, `LinReg`.
- Produces: `Strategy.signal(bars: pd.DataFrame) -> pd.Series` (default raises `NotImplementedError`), overridden by the three strategies. Higher value = more bullish; NaN during warmup.

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/test_signal.py
import numpy as np
import pandas as pd

from rhagent.strategies import build


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_mean_reversion_signal_high_after_drop():
    # 30 flat bars then a sharp drop -> z very negative -> signal (-z) high positive
    closes = [100.0] * 30 + [80.0]
    sig = build("mean_reversion", {}).signal(_bars(closes))
    assert sig.iloc[-1] > 0
    assert not np.isnan(sig.iloc[-1])


def test_momentum_signal_is_trailing_return():
    closes = [100.0 * (1.01 ** i) for i in range(60)]  # steady uptrend
    sig = build("momentum", {}).signal(_bars(closes))
    # trailing 40-bar return, positive in an uptrend
    assert sig.iloc[-1] > 0
    expected = closes[-1] / closes[-41] - 1.0
    assert abs(sig.iloc[-1] - expected) < 1e-9


def test_linreg_signal_matches_position_sign_where_in_position():
    rng = np.random.default_rng(0)
    closes = list(100 + np.cumsum(rng.normal(0, 1, 120)))
    strat = build("linreg", {})
    sig = strat.signal(_bars(closes))
    pos = strat.positions(_bars(closes))
    # where the strategy holds (pos != 0), the signal sign must match the position
    held = pos[pos != 0]
    assert len(held) > 0
    for t in held.index:
        assert np.sign(sig[t]) == pos[t]


def test_signal_nan_free_after_warmup():
    sig = build("mean_reversion", {}).signal(_bars([100.0 + i for i in range(40)]))
    assert not sig.iloc[30:].isna().any()


def test_signal_no_lookahead_recomputation_invariance():
    closes = [100.0 + (i % 7) - 3 for i in range(60)]
    strat = build("mean_reversion", {})
    full = strat.signal(_bars(closes))
    truncated = strat.signal(_bars(closes[:45]))
    # signal at day 44 is identical whether or not later bars exist
    assert abs(full.iloc[44] - truncated.iloc[44]) < 1e-12


def test_base_signal_not_implemented():
    from rhagent.strategies.base import Strategy
    import pytest
    with pytest.raises(NotImplementedError):
        Strategy().signal(_bars([1.0, 2.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/strategies/test_signal.py -v`
Expected: FAIL — `AttributeError: 'MeanReversion' object has no attribute 'signal'` (or `NotImplementedError` from the base once added but before overrides).

- [ ] **Step 3: Add `signal()` to the base class**

In `src/rhagent/strategies/base.py`, add to `class Strategy` (after `positions`):

```python
    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Continuous score aligned to bars.index; higher = more bullish on the
        forward return. No lookahead: the value at day t uses only bars up to t.
        Subclasses that support IC evaluation override this."""
        raise NotImplementedError
```

- [ ] **Step 4: Implement `signal()` on `MeanReversion`**

In `src/rhagent/strategies/mean_reversion.py`, add:

```python
    def signal(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        z = (close - mean) / std
        return -z  # cheap dips (z << 0) score high
```

- [ ] **Step 5: Implement `signal()` on `Momentum`**

In `src/rhagent/strategies/momentum.py`, add:

```python
    def signal(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        return close.pct_change(self.lookback)
```

- [ ] **Step 6: Refactor `LinReg` to share predictions between `positions()` and `signal()`**

Replace the body of `LinReg.positions` in `src/rhagent/strategies/linreg.py` with a shared helper, preserving behavior:

```python
    def _predictions(self, bars: pd.DataFrame) -> pd.Series:
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
        target = ret.shift(-1)
        cols = ["bias", "ret_lag1", "ret_lag2", "ma_ratio"]
        pred = pd.Series(np.nan, index=close.index, dtype=float)
        n = len(close)
        for i in range(n):
            train = feats.iloc[:i].copy()
            train["y"] = target.iloc[:i]
            train = train.dropna()
            x_now = feats.iloc[i][cols]
            if len(train) < self.min_train or x_now.isna().any():
                continue
            X = train[cols].to_numpy()
            y = train["y"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred.iloc[i] = float(x_now.to_numpy() @ beta)
        return pred

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        pred = self._predictions(bars)
        pos = np.sign(pred).fillna(0).astype(int)
        return clamp_short(pos, self.allow_short)

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        return self._predictions(bars)
```

- [ ] **Step 7: Run the new + existing strategy tests**

Run: `.venv/bin/python -m pytest tests/strategies/ -v`
Expected: PASS — the new `test_signal.py` (6 tests) plus the existing `test_linreg.py` (confirms the refactor preserved `positions()` behavior) all green.

- [ ] **Step 8: Commit**

```bash
git add src/rhagent/strategies/ tests/strategies/test_signal.py
git commit -m "feat: continuous signal() on strategies for IC evaluation"
```

---

### Task 2: Universe loader (`factor/universe.py`)

**Files:**
- Create: `src/rhagent/factor/__init__.py`, `src/rhagent/factor/universe.py`
- Test: `tests/factor/__init__.py`, `tests/factor/test_universe.py`

**Interfaces:**
- Consumes: `rhagent.data.get_bars(symbols, start, end, *, fetch=None, cache_dir="data") -> dict[str, pd.DataFrame]`.
- Produces:
  - `UNIVERSE: list[str]` — ~60 liquid large-cap tickers.
  - `load_universe(symbols, start, end, cache_dir="data", min_bars=60) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]` — `(bars_by_symbol, close_panel)`; `close_panel` is `[dates × symbols]` inner-joined to the common calendar; symbols with `< min_bars` rows dropped; empty result raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/factor/test_universe.py
import pandas as pd
import pytest

from rhagent.factor.universe import UNIVERSE, load_universe


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_universe_is_deduped_nonempty_no_etfs():
    assert len(UNIVERSE) >= 50
    assert len(UNIVERSE) == len(set(UNIVERSE))
    assert "SPY" not in UNIVERSE  # ETFs excluded from the cross-section


def test_load_universe_builds_aligned_close_panel(tmp_path):
    _seed(tmp_path, "AAA", [float(i) for i in range(1, 11)])
    _seed(tmp_path, "BBB", [float(i) * 2 for i in range(1, 11)])
    bars, close = load_universe(["AAA", "BBB"], "2026-01-01", "2026-01-10",
                                cache_dir=tmp_path, min_bars=5)
    assert set(bars) == {"AAA", "BBB"}
    assert list(close.columns) == ["AAA", "BBB"]
    assert len(close) == 10
    assert close["BBB"].iloc[-1] == 20.0


def test_load_universe_drops_short_history(tmp_path):
    _seed(tmp_path, "AAA", [float(i) for i in range(1, 11)])  # 10 bars
    _seed(tmp_path, "SHORT", [1.0, 2.0, 3.0])                 # 3 bars
    bars, close = load_universe(["AAA", "SHORT"], "2026-01-01", "2026-01-10",
                                cache_dir=tmp_path, min_bars=5)
    assert set(bars) == {"AAA"}
    assert list(close.columns) == ["AAA"]


def test_load_universe_inner_joins_on_common_dates(tmp_path):
    _seed(tmp_path, "AAA", [float(i) for i in range(1, 11)])   # days 1..10
    # BBB shifted to start later so only some dates overlap
    idx = pd.date_range("2026-01-04", periods=10, freq="D", name="date")
    pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [float(i) for i in range(1, 11)],
         "volume": 1e6},
        index=idx,
    ).to_csv(tmp_path / "BBB.csv")
    _, close = load_universe(["AAA", "BBB"], "2026-01-01", "2026-01-14",
                             cache_dir=tmp_path, min_bars=5)
    # inner join keeps only the overlapping dates (2026-01-04 .. 2026-01-10)
    assert close.index.min() == pd.Timestamp("2026-01-04")
    assert close.index.max() == pd.Timestamp("2026-01-10")
    assert not close.isna().any().any()


def test_load_universe_empty_raises(tmp_path):
    _seed(tmp_path, "SHORT", [1.0, 2.0])
    with pytest.raises(ValueError, match="min_bars"):
        load_universe(["SHORT"], "2026-01-01", "2026-01-02",
                      cache_dir=tmp_path, min_bars=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/factor/test_universe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.factor'`

- [ ] **Step 3: Create the package and implementation**

```python
# src/rhagent/factor/__init__.py
"""Cross-sectional factor / IC evaluation for trading signals."""
```

```python
# src/rhagent/factor/universe.py
"""The evaluation universe and its aligned price panels.

A fixed list of liquid large-cap individual stocks (ETFs excluded — an ETF is
the market itself, not a cross-sectional member). load_universe fetches daily
bars cache-first via data.get_bars and returns both the per-symbol OHLCV frames
(for signal computation) and a [dates x symbols] close panel inner-joined to the
common trading calendar (for IC / forward returns).
"""

from __future__ import annotations

import pandas as pd

from ..data import get_bars

UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "CRM", "ADBE", "NFLX", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK",
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "GILD", "BMY", "MDT", "HD", "LOW", "MCD", "SBUX", "NKE", "COST", "WMT",
    "PG", "KO", "PEP", "PM", "DIS", "CMCSA", "VZ", "T", "XOM", "CVX",
    "CAT", "BA", "GE", "HON", "UNP", "LIN",
]


def load_universe(
    symbols, start, end, cache_dir="data", min_bars: int = 60
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    bars_by_symbol = get_bars(symbols, start, end, cache_dir=cache_dir)
    bars_by_symbol = {s: b for s, b in bars_by_symbol.items() if len(b) >= min_bars}
    if not bars_by_symbol:
        raise ValueError(f"no symbols with >= {min_bars} bars in the universe")
    close = pd.DataFrame(
        {s: b["close"].astype(float) for s, b in bars_by_symbol.items()}
    ).dropna(how="any")  # inner-join to the common calendar
    return bars_by_symbol, close
```

- [ ] **Step 4: Create the test package init and run**

```python
# tests/factor/__init__.py
```

Run: `.venv/bin/python -m pytest tests/factor/test_universe.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/factor/__init__.py src/rhagent/factor/universe.py tests/factor/
git commit -m "feat: factor universe loader with aligned close panel"
```

---

### Task 3: Locked out-of-sample split (`factor/split.py`)

**Files:**
- Create: `src/rhagent/factor/split.py`
- Test: `tests/factor/test_split.py`

**Interfaces:**
- Produces:
  - `oos_cutoff(dates, oos_frac=0.25) -> pd.Timestamp` — the first out-of-sample date (at the `1 - oos_frac` quantile of sorted unique dates).
  - `in_sample_mask(index, cutoff, horizon) -> pd.Series[bool]` — indexed by `index`; True where the day AND the day `horizon` bars ahead are both `< cutoff`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/factor/test_split.py
import pandas as pd

from rhagent.factor.split import in_sample_mask, oos_cutoff


def _idx(n):
    return pd.date_range("2026-01-01", periods=n, freq="D", name="date")


def test_oos_cutoff_at_quantile():
    idx = _idx(100)
    cut = oos_cutoff(idx, oos_frac=0.25)
    # 75% in-sample -> cutoff is the 76th day (index 75)
    assert cut == idx[75]


def test_in_sample_mask_excludes_forward_window_crossing_boundary():
    idx = _idx(100)
    cut = oos_cutoff(idx, 0.25)  # idx[75]
    mask = in_sample_mask(idx, cut, horizon=5)
    # day at index 70: 70+5=75 -> idx[75] == cutoff, NOT < cutoff -> excluded
    assert mask.iloc[70] == False
    # day at index 69: 69+5=74 -> idx[74] < cutoff -> included
    assert mask.iloc[69] == True


def test_in_sample_mask_all_oos_days_false():
    idx = _idx(100)
    cut = oos_cutoff(idx, 0.25)
    mask = in_sample_mask(idx, cut, horizon=5)
    assert not mask[idx >= cut].any()


def test_in_sample_mask_horizon_one():
    idx = _idx(10)
    cut = idx[8]  # last day is OOS
    mask = in_sample_mask(idx, cut, horizon=1)
    # index 7: 7+1=8 == cutoff -> excluded; index 6: 6+1=7 < cutoff -> included
    assert mask.iloc[6] == True and mask.iloc[7] == False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/factor/test_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.factor.split'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/factor/split.py
"""The locked in-sample / out-of-sample date boundary.

The out-of-sample slice is fixed up front and must never be read during signal
development or the search loop — it is reserved for the final gate. in_sample_mask
also trims the boundary so that no in-sample day's forward-return window peeks
across the cutoff into out-of-sample data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def oos_cutoff(dates, oos_frac: float = 0.25) -> pd.Timestamp:
    uniq = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    if len(uniq) == 0:
        raise ValueError("no dates to split")
    idx = int(np.floor(len(uniq) * (1.0 - oos_frac)))
    idx = min(max(idx, 1), len(uniq) - 1)
    return uniq[idx]


def in_sample_mask(index, cutoff, horizon: int) -> pd.Series:
    index = pd.DatetimeIndex(index)
    n = len(index)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        j = i + horizon
        ok[i] = j < n and index[j] < cutoff
    return pd.Series(ok, index=index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/factor/test_split.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/factor/split.py tests/factor/test_split.py
git commit -m "feat: locked out-of-sample date split with boundary trim"
```

---

### Task 4: IC math (`factor/ic.py`)

**Files:**
- Create: `src/rhagent/factor/ic.py`
- Test: `tests/factor/test_ic.py`

**Interfaces:**
- Produces:
  - `forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame` — `close.shift(-h)/close - 1`.
  - `rank_ic_one(sig_row: pd.Series, ret_row: pd.Series, min_names=10) -> float` — Spearman rank-IC for one day; NaN if fewer than `min_names` valid pairs or zero rank variance.
  - `ic_series(signal_panel: pd.DataFrame, close_panel: pd.DataFrame, h: int, min_names=10) -> pd.Series` — IC per day (NaN days dropped).
  - `icir(ic: pd.Series) -> float` — `mean/std(ddof=0)`; 0.0 if empty/zero-variance.
  - `ic_decay(signal_panel, close_panel, horizons=(1,5,10,20,50), min_names=10) -> dict[int, float]` — mean IC per horizon.
  - `half_life(decay: dict[int, float]) -> int | str | None` — smallest horizon at which `|mean IC|` halves; `">50"` if never; `None` if base ≈ 0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/factor/test_ic.py
import numpy as np
import pandas as pd

from rhagent.factor.ic import (
    forward_returns, ic_decay, ic_series, icir, half_life, rank_ic_one,
)


def _panel(rows: dict, cols: int):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", name="date")
    return pd.DataFrame(list(rows.values()), index=idx,
                        columns=[f"S{i}" for i in range(cols)])


def test_forward_returns():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]})
    fwd = forward_returns(close, 1)
    assert abs(fwd["A"].iloc[0] - 0.1) < 1e-12
    assert pd.isna(fwd["A"].iloc[-1])  # last row has no forward value


def test_rank_ic_perfect_and_reversed():
    n = 12
    sig = pd.Series(range(n), index=[f"S{i}" for i in range(n)], dtype=float)
    ret = pd.Series(range(n), index=[f"S{i}" for i in range(n)], dtype=float)
    assert abs(rank_ic_one(sig, ret) - 1.0) < 1e-9
    assert abs(rank_ic_one(sig, ret[::-1].reset_index(drop=True)
                           .set_axis(sig.index)) - (-1.0)) < 1e-9


def test_rank_ic_too_few_names_is_nan():
    sig = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    ret = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    assert np.isnan(rank_ic_one(sig, ret, min_names=10))


def test_rank_ic_invariant_to_common_return_shift():
    # market-relative property: adding a constant to every name's return
    # that day does not change rank-IC (rank is shift-invariant)
    n = 12
    idx = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(1)
    sig = pd.Series(rng.normal(size=n), index=idx)
    ret = pd.Series(rng.normal(size=n), index=idx)
    base = rank_ic_one(sig, ret)
    shifted = rank_ic_one(sig, ret + 0.05)
    assert abs(base - shifted) < 1e-12


def test_ic_series_and_icir():
    # 3 days, 12 names; day-by-day signal perfectly ranks next-day return
    n = 12
    cols = [f"S{i}" for i in range(n)]
    idx = pd.date_range("2026-01-01", periods=4, freq="D", name="date")
    # close chosen so 1-day forward return ordering matches the signal ordering
    sig = pd.DataFrame([list(range(n))] * 4, index=idx, columns=cols, dtype=float)
    # each name grows at a distinct rate -> forward-return rank == name index
    rates = [1.0 + i / 100 for i in range(n)]
    close = pd.DataFrame(
        [[r ** t for r in rates] for t in range(4)], index=idx, columns=cols
    )
    ic = ic_series(sig, close, h=1, min_names=10)
    assert len(ic) >= 2
    assert (ic > 0.99).all()
    assert icir(ic) > 0  # positive and finite


def test_icir_empty_and_zero_variance():
    assert icir(pd.Series(dtype=float)) == 0.0
    assert icir(pd.Series([0.3, 0.3, 0.3])) == 0.0


def test_ic_decay_and_half_life():
    decay = {1: 0.10, 5: 0.08, 10: 0.04, 20: 0.02, 50: 0.01}
    assert half_life(decay) == 10  # first horizon where |IC| <= 0.05
    assert half_life({1: 0.10, 5: 0.09, 10: 0.09, 20: 0.09, 50: 0.09}) == ">50"
    assert half_life({1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/factor/test_ic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.factor.ic'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/factor/ic.py
"""Cross-sectional Information Coefficient math.

Pure functions over panels ([dates x symbols] DataFrames). rank_ic_one is the
Spearman rank correlation between a day's signal cross-section and its forward
returns; because it ranks, it is inherently invariant to a common additive shift
in returns (already market-relative — no separate demeaning is needed). ICIR is
the consistency of the daily IC series, and the decay curve reports how IC fades
across forward horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    return close.shift(-h) / close - 1.0


def rank_ic_one(sig_row: pd.Series, ret_row: pd.Series, min_names: int = 10) -> float:
    df = pd.DataFrame({"s": sig_row, "r": ret_row}).dropna()
    if len(df) < min_names:
        return float("nan")
    sr = df["s"].rank()
    rr = df["r"].rank()
    if sr.std(ddof=0) == 0 or rr.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(sr.to_numpy(), rr.to_numpy())[0, 1])


def ic_series(
    signal_panel: pd.DataFrame, close_panel: pd.DataFrame, h: int, min_names: int = 10
) -> pd.Series:
    fwd = forward_returns(close_panel, h)
    rows: dict = {}
    for t in signal_panel.index.intersection(fwd.index):
        ic = rank_ic_one(signal_panel.loc[t], fwd.loc[t], min_names)
        if not np.isnan(ic):
            rows[t] = ic
    return pd.Series(rows, dtype=float)


def icir(ic: pd.Series) -> float:
    ic = ic.dropna()
    if len(ic) == 0:
        return 0.0
    sd = ic.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(ic.mean() / sd)


def ic_decay(
    signal_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    horizons=(1, 5, 10, 20, 50),
    min_names: int = 10,
) -> dict:
    out: dict = {}
    for h in horizons:
        s = ic_series(signal_panel, close_panel, h, min_names)
        out[h] = float(s.mean()) if len(s) else float("nan")
    return out


def half_life(decay: dict):
    horizons = sorted(decay)
    if not horizons:
        return None
    base = decay[horizons[0]]
    if base is None or np.isnan(base) or base == 0:
        return None
    target = abs(base) / 2.0
    for h in horizons:
        v = decay[h]
        if not np.isnan(v) and abs(v) <= target:
            return h
    return f">{horizons[-1]}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/factor/test_ic.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/factor/ic.py tests/factor/test_ic.py
git commit -m "feat: cross-sectional rank-IC, ICIR, and decay math"
```

---

### Task 5: Signal panel + CLI (`factor/signals.py`, `factor/__main__.py`)

**Files:**
- Create: `src/rhagent/factor/signals.py`, `src/rhagent/factor/__main__.py`
- Test: `tests/factor/test_signals.py`, `tests/factor/test_factor_cli.py`

**Interfaces:**
- Consumes: `Strategy.signal` (Task 1); `UNIVERSE`, `load_universe` (Task 2); `oos_cutoff`, `in_sample_mask` (Task 3); `ic_series`, `icir`, `ic_decay`, `half_life` (Task 4); `rhagent.strategies.REGISTRY`/`build`.
- Produces:
  - `signal_panel(strat, bars_by_symbol, index) -> pd.DataFrame` — `[dates × symbols]` signal panel aligned to `index`.
  - `main(argv: list[str] | None = None) -> int` — the `python -m rhagent.factor` CLI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/factor/test_signals.py
import pandas as pd

from rhagent.factor.signals import signal_panel
from rhagent.strategies import build


def _bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_signal_panel_shape_and_alignment():
    bars = {"AAA": _bars([100.0 + i for i in range(50)]),
            "BBB": _bars([200.0 - i for i in range(50)])}
    idx = bars["AAA"].index
    panel = signal_panel(build("momentum", {}), bars, idx)
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.index.equals(idx)
    # momentum(40) signal is defined (non-NaN) once past warmup
    assert not panel.iloc[-1].isna().any()
```

```python
# tests/factor/test_factor_cli.py
import pandas as pd
import pytest

from rhagent.factor.__main__ import main


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_reports_icir(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    # 5 symbols, 120 bars each, distinct trends so signals vary across names
    for k in range(5):
        closes = [100.0 + k + (0.5 * k + 1) * i for i in range(120)]
        _seed(cache, f"S{k}", closes)

    rc = main([
        "--strategy", "momentum",
        "--symbols", "S0,S1,S2,S3,S4",
        "--horizon", "5", "--min-names", "3", "--days", "200",
        "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ICIR" in out
    assert "decay" in out.lower()


def test_cli_unknown_strategy_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--strategy", "nope", "--symbols", "S0", "--cache-dir", str(tmp_path)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/factor/test_signals.py tests/factor/test_factor_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.factor.signals'`

- [ ] **Step 3: Write `signals.py`**

```python
# src/rhagent/factor/signals.py
"""Assemble a [dates x symbols] signal panel from a strategy."""

from __future__ import annotations

import pandas as pd

from ..strategies.base import Strategy


def signal_panel(
    strat: Strategy, bars_by_symbol: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.DataFrame:
    cols = {s: strat.signal(bars).reindex(index) for s, bars in bars_by_symbol.items()}
    return pd.DataFrame(cols, index=index)
```

- [ ] **Step 4: Write the CLI `__main__.py`**

```python
# src/rhagent/factor/__main__.py
"""CLI: evaluate a strategy's in-sample cross-sectional ICIR and decay.

    python -m rhagent.factor --strategy momentum [--horizon 5] [--oos-frac 0.25]
                             [--days 400] [--min-names 10] [--symbols A,B,...]

Loads the universe, builds the strategy's signal panel, restricts to the locked
in-sample slice, and prints ICIR (with interpretation bands) and the IC decay
curve. The out-of-sample slice is never touched here.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from ..strategies import REGISTRY, build
from .ic import ic_decay, ic_series, half_life, icir
from .signals import signal_panel
from .split import in_sample_mask, oos_cutoff
from .universe import UNIVERSE, load_universe


def _band(x: float) -> str:
    a = abs(x)
    if a > 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    return "likely noise"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="rhagent.factor")
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--oos-frac", type=float, default=0.25)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--min-names", type=int, default=10)
    p.add_argument("--symbols", help="comma-separated override of the default universe")
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else UNIVERSE
    )
    strat = build(args.strategy, {})

    end = date.today()
    start = end - timedelta(days=args.days)
    bars_by_symbol, close = load_universe(
        symbols, start.isoformat(), end.isoformat(), cache_dir=args.cache_dir
    )
    try:
        panel = signal_panel(strat, bars_by_symbol, close.index)
    except NotImplementedError:
        p.error(f"strategy {args.strategy!r} does not implement signal()")

    cutoff = oos_cutoff(close.index, args.oos_frac)
    mask = in_sample_mask(close.index, cutoff, args.horizon)
    is_days = close.index[mask.to_numpy()]
    if len(is_days) == 0:
        p.error("no in-sample days after applying the out-of-sample split")

    sig_is = panel.loc[is_days]
    close_is = close.loc[close.index < cutoff]

    ic = ic_series(sig_is, close_is, args.horizon, args.min_names)
    score = icir(ic)
    decay = ic_decay(sig_is, close_is, min_names=args.min_names)
    hl = half_life(decay)

    print(f"strategy: {args.strategy}   universe: {len(bars_by_symbol)} names")
    print(f"in-sample days: {len(is_days)}   IC observations: {len(ic)}")
    print(f"\nICIR (h={args.horizon}): {score:+.3f}  [{_band(score)}]")
    print(f"mean IC (h={args.horizon}): {ic.mean() if len(ic) else float('nan'):+.4f}")
    print("\nIC decay (mean IC by horizon):")
    for h, v in decay.items():
        print(f"  h={h:<3} {v:+.4f}")
    print(f"half-life: {hl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (existing suite + the new factor tests).

- [ ] **Step 6: End-to-end smoke on cached data**

Run: `PYTHONPATH=src .venv/bin/python -m rhagent.factor --strategy momentum --symbols AAPL,MSFT,NVDA --min-names 2 --days 400`
Expected: prints universe size, in-sample day count, ICIR with a band label, and the IC decay curve + half-life. (Only 3 cached names, so `--min-names 2`; the number is thin — this only proves the pipeline runs end-to-end, per the spec's small-sample caveat.)

- [ ] **Step 7: Commit**

```bash
git add src/rhagent/factor/signals.py src/rhagent/factor/__main__.py tests/factor/test_signals.py tests/factor/test_factor_cli.py
git commit -m "feat: factor CLI - in-sample ICIR + IC decay for a strategy signal"
```

---

## Self-Review Notes

- **Spec coverage:** signal contract + 3 impls (T1); universe/`load_universe`/`forward_returns` (T2 loader + T4 `forward_returns`); locked split with boundary trim (T3); rank-IC/ICIR/decay/half-life (T4); signal panel + CLI + smoke (T5). All spec sections map to a task.
- **Two intentional reconciliations with the spec, surfaced for the reviewer:**
  1. `forward_returns` is implemented in `ic.py` (not `universe.py`) to avoid a `universe`↔`ic` import cycle — it is pure IC math. `load_universe` returns the close panel; IC math owns forward returns.
  2. The spec says "market-neutralize by subtracting the cross-sectional mean." Under **rank-IC** that subtraction is a mathematical no-op (rank is invariant to a common additive shift), so no explicit demeaning step exists; the market-relative property is instead **asserted by test** (`test_rank_ic_invariant_to_common_return_shift`). This is the correct implementation of the spec's intent, not a divergence from it.
- **Type consistency:** `signal(bars) -> pd.Series`, `load_universe(...) -> (dict, DataFrame)`, `ic_series(signal_panel, close_panel, h, min_names) -> Series`, `icir(Series) -> float`, `ic_decay(...) -> dict`, `half_life(dict) -> int|str|None`, `signal_panel(strat, bars_by_symbol, index) -> DataFrame`, `main(argv) -> int` — consistent across tasks and the CLI.
- **No-lookahead** is enforced at three layers: `signal()` uses only trailing data (T1 test), `in_sample_mask` trims the forward window at the boundary (T3), and `close_is` is restricted to `< cutoff` in the CLI so no OOS close is even available to in-sample forward returns.
