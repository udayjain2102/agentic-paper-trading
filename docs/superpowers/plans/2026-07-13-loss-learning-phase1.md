# Loss-Learning Bake-Off — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a walk-forward decision-overlay seam to the paper-trade harness with two learning variants (ConvictionGate, BucketFilter) and a statistically honest Sharpe evaluator, so learning approaches can be compared on the existing dashboard.

**Architecture:** One `Overlay` protocol plugs into the single decide→position point in `PaperTrader.run()`. The baseline is an identity overlay; each variant is one `Overlay` implementation that sees only trades closed on prior bars (walk-forward is free at that call site). A separate `evaluate_robust` module ranks runs by fold-Sharpe + bootstrap CI + deflated Sharpe, rendered as a dashboard panel.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest. No scikit-learn or scipy (not in repo; not needed in Phase 1).

## Global Constraints

- **No new dependencies.** Only `numpy` and `pandas` (already present). Normal-CDF via `math.erf`.
- **No lookahead, ever.** An overlay may read only `closed_trades` (trades with `exit_ts` strictly before the current bar) and `history` (bars up to and including today).
- **Backward compatibility.** The existing 171-test suite must still pass. The `Decision` dataclass change is additive (defaulted field); `entry_features` extraction is a pure move with a re-export.
- **Repo conventions:** src-layout under `src/rhagent/`; run tools with `PYTHONPATH=src`; venv at `.venv/bin/python`; tests in `tests/`, assert-based, no fixtures unless already used.
- **Entry-time features are exactly three:** `vol20`, `gap`, `trend5` (stored as `feat_vol20`, `feat_gap`, `feat_trend5`). Plus `side`, `symbol`. `holding_bars`/`outcome`/`pnl_*` are exit-time — usable only as labels, never as overlay inputs.

---

### Task 1: Extract `entry_features` into a shared module

Pure refactor so both the ledger writer and overlays compute identical entry features. No behavior change.

**Files:**
- Create: `src/rhagent/features.py`
- Modify: `src/rhagent/papertrade.py` (remove local `entry_features`, import from `features`)
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `features.entry_features(history: pd.DataFrame) -> dict` returning `{"vol20": float, "gap": float, "trend5": float}`, lookahead-free, computed from bars up to and including the last row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import pandas as pd
from rhagent.features import entry_features

def _bars(closes, opens=None):
    opens = opens or closes
    return pd.DataFrame({"open": opens, "close": closes})

def test_entry_features_shape_and_lookahead_free():
    bars = _bars([10, 11, 12, 13, 14, 15, 16])
    f = entry_features(bars)
    assert set(f) == {"vol20", "gap", "trend5"}
    # trend5 = sign(close[-1] - close[-6]) = sign(16 - 11) = +1
    assert f["trend5"] == 1.0
    # dropping the last bar changes the features (only past data used, no peeking ahead)
    assert entry_features(bars.iloc[:-1])["trend5"] == 1.0  # sign(15-10)

def test_entry_features_short_history_defaults_zero():
    f = entry_features(_bars([10]))
    assert f == {"vol20": 0.0, "gap": 0.0, "trend5": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.features'`

- [ ] **Step 3: Move the implementation**

Copy the body of the current `entry_features` from `papertrade.py` (lines ~59-77) verbatim into the new module:

```python
# src/rhagent/features.py
"""Lookahead-free entry-time features, shared by the ledger writer and overlays."""

from __future__ import annotations

import pandas as pd


def entry_features(history: pd.DataFrame) -> dict:
    """Cheap lookahead-free scalars at entry, used for failure bucketing.

    All values use only bars up to and including the last row of `history`.
    """
    close = history["close"].astype(float)
    rets = close.pct_change().dropna()
    vol20 = float(rets.tail(20).std()) if len(rets) >= 2 else 0.0
    if pd.isna(vol20):
        vol20 = 0.0
    if len(close) >= 2 and "open" in history:
        gap = float(history["open"].astype(float).iloc[-1] / close.iloc[-2] - 1.0)
    else:
        gap = 0.0
    trend5 = 0.0
    if len(close) >= 6:
        diff = close.iloc[-1] - close.iloc[-6]
        trend5 = float((diff > 0) - (diff < 0))
    return {"vol20": vol20, "gap": gap, "trend5": trend5}
```

Note: reproduce the exact logic that currently exists in `papertrade.entry_features`. If the current code differs from the above (e.g. different `gap`/`trend5` computation), copy the CURRENT code — this task must not change behavior. Verify by reading `papertrade.py` first.

- [ ] **Step 4: Replace the original with a re-export**

In `papertrade.py`, delete the local `def entry_features(...)` and add near the top imports:

```python
from .features import entry_features  # noqa: F401  (re-exported for callers/tests)
```

Leave every internal call to `entry_features(...)` unchanged — it now resolves to the imported name.

- [ ] **Step 5: Run new + existing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_features.py -v && PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: new tests PASS; full suite still **171 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/rhagent/features.py src/rhagent/papertrade.py tests/test_features.py
git commit -m "refactor: extract entry_features into shared features module"
```

---

### Task 2: Add `conviction` to `Decision` and populate it in `StrategyEngine`

**Files:**
- Modify: `src/rhagent/engine.py` (Decision dataclass ~21-24; StrategyEngine.decide ~43-49)
- Test: `tests/test_engine_conviction.py`

**Interfaces:**
- Produces: `Decision(target: float, reason: str, conviction: float | None = None)`. `StrategyEngine.decide` sets `conviction = float(strat.signal(history).iloc[-1])` (may be NaN during warmup). `AgentEngine` leaves it `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_conviction.py
import numpy as np, pandas as pd
from rhagent.engine import StrategyEngine, Decision
from rhagent.strategies import build

def _bars(n=60):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    return pd.DataFrame({"open": close, "close": close}, index=idx)

def test_decision_has_conviction_field_default_none():
    d = Decision(target=1.0, reason="x")
    assert d.conviction is None

def test_strategy_engine_sets_conviction_from_signal():
    eng = StrategyEngine(build("mean_reversion", {}))
    bars = _bars()
    d = eng.decide("NVDA", bars, 0.0)
    strat = build("mean_reversion", {})
    expected = float(strat.signal(bars).iloc[-1])
    assert (d.conviction == expected) or (np.isnan(d.conviction) and np.isnan(expected))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_engine_conviction.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'conviction'` OR conviction attribute missing.

- [ ] **Step 3: Add the field**

In `engine.py`, extend the dataclass:

```python
@dataclass(frozen=True)
class Decision:
    target: float
    reason: str
    conviction: float | None = None
```

- [ ] **Step 4: Populate it in StrategyEngine.decide**

Change the return in `StrategyEngine.decide` to compute and pass conviction:

```python
def decide(self, symbol: str, history: pd.DataFrame, current_pos: float) -> Decision:
    target = float(self.strat.positions(history).iloc[-1])
    close = float(history["close"].iloc[-1])
    try:
        conviction = float(self.strat.signal(history).iloc[-1])
    except (NotImplementedError, KeyError, IndexError):
        conviction = None
    reason = f"{self.name}: target={target:+.0f} close={close:.2f}"
    return Decision(target=target, reason=reason, conviction=conviction)
```

Leave `AgentEngine.decide` unchanged (it constructs `Decision(target=..., reason=...)`; conviction defaults to `None`).

- [ ] **Step 5: Run new + existing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_engine_conviction.py -q && PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: new PASS; full suite **171 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/rhagent/engine.py tests/test_engine_conviction.py
git commit -m "feat: expose per-bar conviction on Decision"
```

---

### Task 3: Overlay protocol, IdentityOverlay, seam wiring, and `--overlay` CLI

Wire the seam with the no-op overlay so a `--overlay none` run is byte-identical to today.

**Files:**
- Create: `src/rhagent/overlay.py`
- Modify: `src/rhagent/papertrade.py` (PaperTrader.__init__, run() seam ~156-157, main() argparse + engine construction)
- Test: `tests/test_overlay_seam.py`

**Interfaces:**
- Produces:
  - `overlay.Overlay` Protocol: `name: str`; `adjust(symbol: str, history: pd.DataFrame, decision: Decision, closed_trades: pd.DataFrame) -> float`.
  - `overlay.IdentityOverlay` (name `"none"`): returns `decision.target`.
  - `overlay.build_overlay(name: str) -> Overlay` mapping `"none" -> IdentityOverlay()` (extended in later tasks).
  - `PaperTrader(..., overlay: Overlay = IdentityOverlay())`.
  - CLI flag `--overlay {none}` (choices grow in later tasks), default `none`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_overlay_seam.py
import pandas as pd
from rhagent.overlay import IdentityOverlay, build_overlay
from rhagent.engine import Decision

def test_identity_overlay_passes_target_through():
    ov = IdentityOverlay()
    d = Decision(target=1.0, reason="x", conviction=0.5)
    out = ov.adjust("NVDA", pd.DataFrame({"close": [1, 2]}), d, pd.DataFrame())
    assert out == 1.0
    assert ov.name == "none"

def test_build_overlay_none():
    assert build_overlay("none").name == "none"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_seam.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.overlay'`

- [ ] **Step 3: Create the overlay module**

```python
# src/rhagent/overlay.py
"""Decision overlays: a walk-forward layer between a strategy's raw target and
the position actually taken. Each overlay sees only trades that closed on prior
bars (`closed_trades`), so no overlay can peek at the future.

    final_target = overlay.adjust(symbol, history, decision, closed_trades)

Return semantics: 0.0 vetoes the trade, a fraction downsizes it, and a value
equal to decision.target passes it through unchanged.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .engine import Decision


class Overlay(Protocol):
    name: str
    def adjust(
        self,
        symbol: str,
        history: pd.DataFrame,
        decision: Decision,
        closed_trades: pd.DataFrame,
    ) -> float: ...


class IdentityOverlay:
    name = "none"

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        return decision.target


def build_overlay(name: str) -> Overlay:
    if name == "none":
        return IdentityOverlay()
    raise KeyError(f"unknown overlay {name!r}")
```

- [ ] **Step 4: Wire the seam in PaperTrader**

In `papertrade.py`:

1. Add the import: `from .overlay import Overlay, IdentityOverlay, build_overlay`.
2. In `PaperTrader.__init__`, add a parameter `overlay: Overlay | None = None` and store `self.overlay = overlay or IdentityOverlay()`.
3. In `run()`, at the seam (currently `d = self.engine.decide(...); target = d.target`), replace with:

```python
d = self.engine.decide(sym, history, prev)
target = self.overlay.adjust(sym, history, d, pd.DataFrame(trades))
```

`trades` is the loop's list of already-closed trade dicts; at this point it contains only trades closed on prior bars (closes happen later in the same iteration), so `closed_trades` is walk-forward-clean. Leave the existing final-bar guard and open/close logic below it unchanged — they already operate on `target`, including fractional values.

- [ ] **Step 5: Add the CLI flag and pass the overlay**

In `papertrade.main()` argparse block, add:

```python
p.add_argument("--overlay", default="none", choices=["none"],
               help="decision overlay applied to each target (learning variant)")
```

After building `engine` and before constructing `PaperTrader`, build the overlay and pass it:

```python
overlay = build_overlay(args.overlay)
trader = PaperTrader(
    engine=engine, source=source, cost_bps=args.cost_bps,
    out_dir=args.out_dir, overlay=overlay,
)
```

Also record it in the run metadata so the dashboard/evaluator can label variants: find where `run.json` is written and add `"overlay": args.overlay` to that dict (search for `"engine":` in the metadata-writing code and add the key alongside it).

- [ ] **Step 6: Verify identity — a `--overlay none` run matches baseline**

Run a quick equivalence check against a real cached symbol set:

```bash
PYTHONPATH=src .venv/bin/python -m rhagent.papertrade --engine mean_reversion --symbols NVDA,SPY --overlay none --days 400 --out-dir /tmp/ov_none
```
Expected: run completes; the printed n_trades/win_rate match a plain baseline run of the same command without `--overlay`. Then run tests:

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_seam.py -q && PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: new PASS; full suite **171 passed**.

- [ ] **Step 7: Commit**

```bash
git add src/rhagent/overlay.py src/rhagent/papertrade.py tests/test_overlay_seam.py
git commit -m "feat: decision-overlay seam with no-op IdentityOverlay and --overlay CLI"
```

---

### Task 4: ConvictionGate overlay

Kill coin-flip trades: only enter when `|conviction|` clears a rolling percentile of the symbol's own past convictions. Stateful (accumulates past convictions per symbol) — walk-forward safe because it only ever sees convictions from bars already processed.

**Files:**
- Modify: `src/rhagent/overlay.py` (add class + register in `build_overlay`, extend CLI choices in `papertrade.py`)
- Test: `tests/test_overlay_conviction.py`

**Interfaces:**
- Consumes: `Decision.conviction` (Task 2), `build_overlay` (Task 3).
- Produces: `overlay.ConvictionGate(pctile: float = 0.60, window: int = 120)` with `name = "conviction"`; registered as `build_overlay("conviction")`; CLI choice `"conviction"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_overlay_conviction.py
import numpy as np, pandas as pd
from rhagent.overlay import ConvictionGate
from rhagent.engine import Decision

def _feed(gate, sym, convictions, target=1.0):
    """Feed a sequence of convictions; return the list of adjusted targets."""
    out = []
    for c in convictions:
        d = Decision(target=target, reason="x", conviction=c)
        out.append(gate.adjust(sym, pd.DataFrame({"close": [1]}), d, pd.DataFrame()))
    return out

def test_low_conviction_vetoed_high_passes():
    gate = ConvictionGate(pctile=0.60, window=50)
    # 40 small |conviction| then measure a small vs a large one
    outs = _feed(gate, "NVDA", [0.1] * 40)
    assert all(o in (0.0, 1.0) for o in outs)  # cold start passes, then gating begins
    small = gate.adjust("NVDA", pd.DataFrame({"close": [1]}),
                        Decision(target=1.0, reason="x", conviction=0.1), pd.DataFrame())
    big = gate.adjust("NVDA", pd.DataFrame({"close": [1]}),
                      Decision(target=1.0, reason="x", conviction=5.0), pd.DataFrame())
    assert small == 0.0   # below the 60th pctile of past |conviction|
    assert big == 1.0     # well above threshold

def test_none_conviction_passes_through():
    gate = ConvictionGate()
    d = Decision(target=-1.0, reason="x", conviction=None)
    assert gate.adjust("NVDA", pd.DataFrame({"close": [1]}), d, pd.DataFrame()) == -1.0

def test_nan_conviction_passes_through():
    gate = ConvictionGate()
    d = Decision(target=1.0, reason="x", conviction=float("nan"))
    assert gate.adjust("NVDA", pd.DataFrame({"close": [1]}), d, pd.DataFrame()) == 1.0

def test_cold_start_passes_before_window_fills():
    gate = ConvictionGate(window=30)
    outs = _feed(gate, "NVDA", [0.1] * 10)  # fewer than window
    assert outs == [1.0] * 10  # not enough history to gate yet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_conviction.py -v`
Expected: FAIL — `ImportError: cannot import name 'ConvictionGate'`

- [ ] **Step 3: Implement ConvictionGate**

Add to `overlay.py`:

```python
import math
from collections import defaultdict

import numpy as np


class ConvictionGate:
    """Veto entries whose |conviction| is below a rolling percentile of the
    symbol's own past |conviction|. Stateful and walk-forward: only convictions
    from bars already seen inform the threshold."""

    name = "conviction"

    def __init__(self, pctile: float = 0.60, window: int = 120) -> None:
        self.pctile = pctile
        self.window = window
        self._hist: dict[str, list[float]] = defaultdict(list)

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        c = decision.conviction
        if c is None or (isinstance(c, float) and math.isnan(c)):
            return decision.target
        past = self._hist[symbol]
        # threshold from history BEFORE recording today's value (no self-inclusion)
        if len(past) < self.window:
            result = decision.target  # cold start: not enough history to gate
        else:
            thresh = float(np.percentile(np.abs(past[-self.window:]), self.pctile * 100))
            result = decision.target if abs(c) >= thresh else 0.0
        past.append(abs(c))
        return result
```

- [ ] **Step 4: Register in build_overlay and CLI**

In `overlay.py`, extend `build_overlay`:

```python
def build_overlay(name: str) -> Overlay:
    if name == "none":
        return IdentityOverlay()
    if name == "conviction":
        return ConvictionGate()
    raise KeyError(f"unknown overlay {name!r}")
```

In `papertrade.py`, extend the CLI choices: `choices=["none", "conviction"]`.

- [ ] **Step 5: Run new + existing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_conviction.py -q && PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: new PASS; full suite **171 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/rhagent/overlay.py src/rhagent/papertrade.py tests/test_overlay_conviction.py
git commit -m "feat: ConvictionGate overlay (rolling-percentile signal filter)"
```

---

### Task 5: BucketFilter overlay + walk-forward leak test

Veto entries whose setup bucket has been bleeding, using `evaluate.failure_buckets` on `closed_trades`. Candidate bucket labels are derived from features recomputed from `history` via the shared helper, and vol terciles derived from `closed_trades`.

**Files:**
- Modify: `src/rhagent/overlay.py` (add class + helper, register, CLI)
- Test: `tests/test_overlay_bucket.py`, and a leak test in `tests/test_overlay_walkforward.py`

**Interfaces:**
- Consumes: `features.entry_features` (Task 1), `evaluate.failure_buckets`, `build_overlay`/CLI (Task 3).
- Produces: `overlay.BucketFilter(veto_share=0.25, veto_wr=0.40, min_n=20, min_size=0.3)` with `name = "bucket"`; registered `build_overlay("bucket")`; CLI choice `"bucket"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_overlay_bucket.py
import numpy as np, pandas as pd
from rhagent.overlay import BucketFilter
from rhagent.engine import Decision

def _closed(side, feat_vol20, feat_gap, pnl, n):
    """Build n identical closed trades in one bucket."""
    return pd.DataFrame({
        "symbol": ["NVDA"] * n, "side": [side] * n,
        "feat_vol20": [feat_vol20] * n, "feat_gap": [feat_gap] * n,
        "feat_trend5": [0.0] * n, "holding_bars": [3] * n,
        "pnl_abs": [pnl] * n, "outcome": ["loss" if pnl < 0 else "win"] * n,
    })

def _hist(vol20=0.02, gap=0.0):
    # a history whose entry_features produce roughly vol20/gap is hard to force;
    # BucketFilter recomputes features from history, so build a matching history.
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    close = pd.Series(np.linspace(100, 100 * (1 + gap), 30), index=idx)
    return pd.DataFrame({"open": close, "close": close}, index=idx)

def test_bleeding_short_bucket_vetoed():
    # 40 losing shorts => side=short bucket has 100% loss share, 0% win rate
    closed = _closed("short", 0.02, 0.0, pnl=-100.0, n=40)
    bf = BucketFilter(min_n=20)
    d = Decision(target=-1.0, reason="x", conviction=None)  # candidate is a short
    out = bf.adjust("NVDA", _hist(), d, closed)
    assert out == 0.0

def test_clean_bucket_passes():
    closed = _closed("long", 0.02, 0.0, pnl=+100.0, n=40)  # all winners
    bf = BucketFilter(min_n=20)
    d = Decision(target=1.0, reason="x", conviction=None)
    out = bf.adjust("NVDA", _hist(), d, closed)
    assert out == 1.0

def test_cold_start_passes():
    bf = BucketFilter(min_n=20)
    d = Decision(target=1.0, reason="x", conviction=None)
    assert bf.adjust("NVDA", _hist(), d, pd.DataFrame()) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_bucket.py -v`
Expected: FAIL — `ImportError: cannot import name 'BucketFilter'`

- [ ] **Step 3: Implement BucketFilter**

Add to `overlay.py`:

```python
from .features import entry_features
from .evaluate import failure_buckets


class BucketFilter:
    """Veto/down-size entries whose setup bucket has been the worst loser in
    closed trades so far. Deterministic and inspectable."""

    name = "bucket"

    def __init__(self, veto_share=0.25, veto_wr=0.40, min_n=20, min_size=0.3) -> None:
        self.veto_share = veto_share
        self.veto_wr = veto_wr
        self.min_n = min_n
        self.min_size = min_size

    def _candidate_labels(self, history, side, closed) -> dict:
        """The candidate's bucket in each dimension (vol/gap/side)."""
        f = entry_features(history)
        # vol tercile boundaries from the population of closed trades
        vol_lab = "all"
        vols = closed["feat_vol20"].astype(float)
        if vols.nunique() >= 3:
            lo, hi = np.percentile(vols, [33.333, 66.667])
            v = f["vol20"]
            vol_lab = "low" if v <= lo else ("high" if v > hi else "med")
        gap = f["gap"]
        gap_lab = "flat"
        if gap < -0.005:
            gap_lab = "down"
        elif gap > 0.005:
            gap_lab = "up"
        return {"vol": vol_lab, "gap": gap_lab, "side": side}

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        target = decision.target
        if target == 0.0 or len(closed_trades) < self.min_n:
            return target
        fb = failure_buckets(closed_trades)
        if len(fb) == 0:
            return target
        side = "long" if target > 0 else "short"
        labels = self._candidate_labels(history, side, closed_trades)
        worst_share = 0.0
        for dim, bucket in labels.items():
            row = fb[(fb["dimension"] == dim) & (fb["bucket"] == str(bucket))]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            if r["n_trades"] >= self.min_n and r["loss_share"] >= self.veto_share \
                    and r["win_rate"] <= self.veto_wr:
                return 0.0  # veto: this bucket is bleeding
            worst_share = max(worst_share, float(r["loss_share"]))
        # soft down-size proportional to the worst bucket's loss share
        size = max(self.min_size, 1.0 - worst_share)
        return target * size
```

- [ ] **Step 4: Register and CLI**

Extend `build_overlay`:

```python
    if name == "bucket":
        return BucketFilter()
```
Extend CLI choices in `papertrade.py`: `choices=["none", "conviction", "bucket"]`.

- [ ] **Step 5: Run BucketFilter tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_bucket.py -q`
Expected: PASS. If `failure_buckets` requires columns not present in the synthetic frame, add the missing columns to `_closed(...)` in the test to match what `failure_buckets`/`_bucket_labels` reads (`feat_vol20`, `feat_gap`, `holding_bars`, `symbol`, `side`, `pnl_abs`, `outcome`).

- [ ] **Step 6: Write the walk-forward leak test**

This is the critical correctness guard: prove no overlay ever sees a same-day or future close.

```python
# tests/test_overlay_walkforward.py
import numpy as np, pandas as pd
from rhagent.overlay import Overlay
from rhagent.papertrade import PaperTrader, HistoricalSource
from rhagent.engine import StrategyEngine, Decision
from rhagent.strategies import build

class _SpyOverlay:
    """Records the max exit_ts it is ever shown, per bar timestamp seen."""
    name = "spy"
    def __init__(self):
        self.violations = []
    def adjust(self, symbol, history, decision, closed_trades):
        today = history.index[-1]
        if len(closed_trades):
            exits = pd.to_datetime(closed_trades["exit_ts"])
            if (exits >= today).any():
                self.violations.append((symbol, str(today)))
        return decision.target

def test_overlay_never_sees_future_or_same_day_close(tmp_path):
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    close = pd.Series(100 + np.sin(np.arange(120) / 3.0) * 5, index=idx)
    df = pd.DataFrame({"open": close, "close": close}, index=idx)

    class _Src:
        def bars(self): return {"NVDA": df}

    spy = _SpyOverlay()
    trader = PaperTrader(engine=StrategyEngine(build("mean_reversion", {})),
                         source=_Src(), out_dir=str(tmp_path), overlay=spy)
    trader.run()
    assert spy.violations == [], f"overlay saw non-past closes: {spy.violations[:5]}"
```

- [ ] **Step 7: Run the leak test + full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_walkforward.py -v && PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: leak test PASS (no violations); full suite passes (now 171 + the new tests).

- [ ] **Step 8: Commit**

```bash
git add src/rhagent/overlay.py src/rhagent/papertrade.py tests/test_overlay_bucket.py tests/test_overlay_walkforward.py
git commit -m "feat: BucketFilter overlay + walk-forward leak test"
```

---

### Task 6: Robust evaluator (`evaluate_robust.py`)

Fold-Sharpe, bootstrap CI, deflated Sharpe. Uses only numpy/pandas + `math.erf`.

**Files:**
- Create: `src/rhagent/evaluate_robust.py`
- Test: `tests/test_evaluate_robust.py`

**Interfaces:**
- Consumes: run dirs under a base-dir (each has `returns.csv` and `run.json`), reusing `evaluate.load_run` and `backtest.result_from_returns`.
- Produces:
  - `evaluate_robust.fold_sharpe(net: pd.Series, fold: int = 60, step: int = 30) -> tuple[float, float]` → (mean, std) of per-fold annualized Sharpe.
  - `evaluate_robust.bootstrap_sharpe_ci(net: pd.Series, n: int = 1000, seed: int = 0) -> tuple[float, float]` → (lo, hi) 95% percentile CI.
  - `evaluate_robust.deflated_sharpe(observed_sr: float, all_srs: list[float], net: pd.Series) -> float` → probability in [0,1].
  - `evaluate_robust.robust_table(base_dir) -> pd.DataFrame` with columns: `run_id, engine, overlay, point_sharpe, fold_mean, fold_std, ci_lo, ci_hi, deflated, beats_baseline`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate_robust.py
import numpy as np, pandas as pd
from rhagent.evaluate_robust import fold_sharpe, bootstrap_sharpe_ci, deflated_sharpe

def _net(mean=0.001, sd=0.01, n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(mean, sd, n), index=idx)

def test_bootstrap_ci_brackets_point_sharpe():
    net = _net()
    ann = float(net.mean() / net.std() * np.sqrt(252))
    lo, hi = bootstrap_sharpe_ci(net, n=500, seed=1)
    assert lo <= ann <= hi
    assert lo < hi

def test_fold_sharpe_returns_mean_and_std():
    net = _net()
    m, s = fold_sharpe(net, fold=60, step=30)
    assert np.isfinite(m) and s >= 0.0

def test_deflated_sharpe_in_unit_interval_and_penalizes_trials():
    net = _net(mean=0.002)
    sr = float(net.mean() / net.std() * np.sqrt(252))
    d_few = deflated_sharpe(sr, [sr, 0.0], net)
    d_many = deflated_sharpe(sr, [sr] + [0.0] * 50, net)
    assert 0.0 <= d_many <= d_few <= 1.0  # more trials => harder to clear
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluate_robust.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.evaluate_robust'`

- [ ] **Step 3: Implement the evaluator**

```python
# src/rhagent/evaluate_robust.py
"""Noise-robust ranking of paper-trade runs: fold Sharpe, bootstrap CI, and a
deflated Sharpe that penalizes multiple-testing. numpy/pandas only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import result_from_returns
from .evaluate import load_run

_ANN = 252


def _sharpe(x: np.ndarray) -> float:
    sd = x.std()
    return float(x.mean() / sd * math.sqrt(_ANN)) if sd > 0 else 0.0


def _phi(z: float) -> float:
    """Standard normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def fold_sharpe(net: pd.Series, fold: int = 60, step: int = 30) -> tuple[float, float]:
    x = net.to_numpy(dtype=float)
    srs = [_sharpe(x[i:i + fold]) for i in range(0, max(len(x) - fold + 1, 1), step)]
    srs = [s for s in srs if np.isfinite(s)]
    if not srs:
        return 0.0, 0.0
    return float(np.mean(srs)), float(np.std(srs))


def bootstrap_sharpe_ci(net: pd.Series, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    x = net.to_numpy(dtype=float)
    if len(x) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    srs = np.array([_sharpe(rng.choice(x, size=len(x), replace=True)) for _ in range(n)])
    return float(np.percentile(srs, 2.5)), float(np.percentile(srs, 97.5))


def deflated_sharpe(observed_sr: float, all_srs: list[float], net: pd.Series) -> float:
    """Probabilistic Sharpe vs a benchmark inflated for M trials (Bailey & Lopez
    de Prado). Higher = more likely the Sharpe is real, not multiple-testing luck."""
    x = net.to_numpy(dtype=float)
    T = len(x)
    if T < 3:
        return 0.0
    sd = x.std()
    if sd == 0:
        return 0.0
    z = (x - x.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # non-excess
    M = max(len(all_srs), 1)
    var_sr = float(np.var(all_srs)) if M > 1 else 1.0
    gamma = 0.5772156649  # Euler-Mascheroni
    e_max = (1 - gamma) * _phi_inv(1 - 1.0 / M) + gamma * _phi_inv(1 - 1.0 / (M * math.e))
    sr_benchmark = math.sqrt(var_sr) * e_max
    # daily-scale Sharpe (deflate uses per-observation SR, not annualized)
    sr_daily = observed_sr / math.sqrt(_ANN)
    denom = math.sqrt(max(1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2, 1e-9))
    num = (sr_daily - sr_benchmark / math.sqrt(_ANN)) * math.sqrt(T - 1)
    return float(_phi(num / denom))


def robust_table(base_dir: str | Path) -> pd.DataFrame:
    base_dir = Path(base_dir)
    runs = []
    for meta_path in sorted(base_dir.glob("*/run.json")):
        meta, trades, net = load_run(meta_path.parent)
        res = result_from_returns(net)
        runs.append({
            "run_id": meta["run_id"], "engine": meta.get("engine", ""),
            "overlay": meta.get("overlay", "none"),
            "point_sharpe": res["sharpe"], "net": net,
        })
    if not runs:
        return pd.DataFrame()
    all_srs = [r["point_sharpe"] for r in runs]
    baseline_sr = max((r["point_sharpe"] for r in runs if r["overlay"] == "none"),
                      default=max(all_srs))
    rows = []
    for r in runs:
        fm, fs = fold_sharpe(r["net"])
        lo, hi = bootstrap_sharpe_ci(r["net"])
        d = deflated_sharpe(r["point_sharpe"], all_srs, r["net"])
        rows.append({
            "run_id": r["run_id"], "engine": r["engine"], "overlay": r["overlay"],
            "point_sharpe": r["point_sharpe"], "fold_mean": fm, "fold_std": fs,
            "ci_lo": lo, "ci_hi": hi, "deflated": d,
            "beats_baseline": bool(lo > baseline_sr),
        })
    return pd.DataFrame(rows).sort_values("deflated", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run new + existing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluate_robust.py -q && PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: new PASS; full suite passes. If `result_from_returns` returns a key other than `"sharpe"`, adjust the key name to match `backtest.result_from_returns` (verify by reading it).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/evaluate_robust.py tests/test_evaluate_robust.py
git commit -m "feat: robust evaluator (fold Sharpe + bootstrap CI + deflated Sharpe)"
```

---

### Task 7: Bake-off panel on the dashboard

Render `robust_table` as a panel in the all-runs dashboard.

**Files:**
- Modify: `scripts/make_dashboard.py` (add a panel in `render_all`)
- Test: manual render check (dashboard has no unit-test file; verify by generating HTML and grepping).

**Interfaces:**
- Consumes: `evaluate_robust.robust_table(base_dir)` (Task 6).
- Produces: a `<h2>Bake-off</h2>` panel table in the all-runs HTML, sorted by deflated Sharpe, beats-baseline highlighted.

- [ ] **Step 1: Add a renderer for the robust table**

In `scripts/make_dashboard.py`, add a helper (near `_compare_table`):

```python
def _bakeoff_table(base_dir) -> str:
    from rhagent.evaluate_robust import robust_table
    df = robust_table(base_dir)
    if len(df) == 0:
        return "<p class='muted'>no runs</p>"
    rows = []
    for _, r in df.iterrows():
        win = "beats" if r["beats_baseline"] else ""
        cls = "up" if r["beats_baseline"] else ""
        rows.append(
            f"<tr><td class='mono'>{escape(str(r['run_id']))}</td>"
            f"<td>{escape(str(r['engine']))}</td>"
            f"<td>{escape(str(r['overlay']))}</td>"
            f"<td class='num'>{r['point_sharpe']:.2f}</td>"
            f"<td class='num'>{r['fold_mean']:.2f}±{r['fold_std']:.2f}</td>"
            f"<td class='num'>[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]</td>"
            f"<td class='num'>{r['deflated']:.2f}</td>"
            f"<td class='num {cls}'>{win}</td></tr>"
        )
    return (
        "<table class='grid'><thead><tr><th>run id</th><th>engine</th><th>overlay</th>"
        "<th>point sharpe</th><th>fold mean±sd</th><th>95% CI</th><th>deflated</th>"
        f"<th>vs baseline</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
```

- [ ] **Step 2: Insert the panel into render_all**

In `render_all`, add the panel to the `index` string (after the existing comparison table, before the run sections):

```python
    index += (
        "<h2>Bake-off · robust Sharpe (fold + bootstrap + deflated)</h2>"
        "<p class='sub'>A variant beats baseline only if its 95% CI lower bound "
        "clears the baseline Sharpe.</p>"
        f"<div class='tblscroll'>{_bakeoff_table(base_dir)}</div>"
    )
```

(If `index` is built as a single parenthesized expression, split it so you can append; keep the existing content intact.)

- [ ] **Step 3: Generate and verify**

Run against the real journal:

```bash
PYTHONPATH=src .venv/bin/python scripts/make_dashboard.py --base-dir journal/papertrade --out /tmp/dash.html
grep -c "Bake-off" /tmp/dash.html
grep -c "deflated" /tmp/dash.html
```
Expected: dashboard writes without error; both greps return ≥1.

- [ ] **Step 4: Run full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: full suite passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dashboard.py
git commit -m "feat: bake-off panel (robust Sharpe) on all-runs dashboard"
```

---

### Task 8: Run the Phase-1 bake-off and record the finding

Not code — produce the actual comparison the whole effort is for.

- [ ] **Step 1: Run the three variants on the testbed**

```bash
for ov in none conviction bucket; do
  PYTHONPATH=src .venv/bin/python -m rhagent.papertrade \
    --engine mean_reversion --symbols all --allow-short --overlay $ov \
    --days 400 --out-dir journal/papertrade
done
```
Expected: three runs complete; each prints n_trades (expect conviction/bucket to have FEWER trades than none).

- [ ] **Step 2: Regenerate the dashboard and read the verdict**

```bash
PYTHONPATH=src .venv/bin/python scripts/make_dashboard.py --base-dir journal/papertrade
```
Open `journal/papertrade/dashboard.html`, read the Bake-off panel. Record: does any variant's CI lower bound clear the baseline Sharpe? A "no" is a valid finding (edge not there at this filtering).

- [ ] **Step 3: No commit** — journal outputs are gitignored. Report the bake-off result to the user.

---

## Self-Review

**Spec coverage:**
- §3.1 Overlay protocol → Task 3. §3.2 seam application → Task 3. §3.3 Decision.conviction → Task 2. §4 ConvictionGate → Task 4. §5.1 BucketFilter → Task 5. §6 robust evaluator → Task 6. §6 dashboard panel → Task 7. §7 entry_features extraction → Task 1. §8 CLI `--overlay` → Task 3 (choices grown in 4, 5). §9 phasing (P1 only) → this plan. §10 tests incl. walk-forward leak test → Task 5 Step 6. WinProbGate/ParamTune (§5.2/§5.3) → correctly deferred to Phase 2/3, not in this plan.
- Gap check: `--tune` flag (§8) belongs to ParamTune (Phase 3) — correctly absent here.

**Placeholder scan:** No TBD/TODO. Every code step shows full code. Two steps say "verify the current code / key name matches" (Task 1 Step 3, Task 6 Step 4) — these are correctness confirmations against existing code, not placeholders, and name the exact thing to check.

**Type consistency:** `Overlay.adjust(symbol, history, decision, closed_trades) -> float` used identically in Tasks 3/4/5 and the leak test. `Decision(target, reason, conviction=None)` consistent Tasks 2–5. `build_overlay(name)` grown monotonically (none → +conviction → +bucket). `robust_table` columns match `_bakeoff_table` reads (run_id, engine, overlay, point_sharpe, fold_mean, fold_std, ci_lo, ci_hi, deflated, beats_baseline). `entry_features` returns `{vol20,gap,trend5}` consistently.
