# Loss-Learning Phase 2 — WinProbGate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `WinProbGate`, a decision overlay that fits a small numpy logistic-regression model on past trade outcomes and vetoes candidate entries whose predicted win-probability is too low.

**Architecture:** Same `Overlay` seam as Phase 1 (`overlay.py`). The gate reuses `features.flatten_trades` to read entry features from closed trades (walk-forward), fits a logistic model via IRLS on `outcome=="win"`, refits on a cadence, and vetoes low-P(win) candidates. numpy only — no scikit-learn.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest. No scipy/sklearn.

## Global Constraints

- **No new dependencies.** numpy/pandas only. Logistic regression is hand-rolled (IRLS).
- **No lookahead.** The model is fit only on `closed_trades` (trades closed strictly before today, guaranteed by the seam). Symbol encoding is computed from `closed_trades` only.
- **Long-only.** Shorting is disabled system-wide; every trade is `side=long`, so `side` is NOT a feature. Features are exactly `[feat_vol20, feat_gap, feat_trend5, sym_wr]` plus a bias term.
- **Backward compatible.** Adding `WinProbGate` + registering `"winprob"` must not change `none`/`conviction`/`bucket` behavior. Full suite (194 passed) stays green.
- **Repo conventions:** src-layout, `PYTHONPATH=src .venv/bin/python`, tests assert-based in `tests/`.
- **Reuse:** `features.flatten_trades(df)` already converts the live nested-`entry_features` frame to `feat_vol20/feat_gap/feat_trend5` columns — use it, do not re-normalize by hand.

---

### Task 1: numpy logistic regression (IRLS) helpers

**Files:**
- Modify: `src/rhagent/overlay.py` (add two module-level helpers near the top, after imports)
- Test: `tests/test_logit.py`

**Interfaces:**
- Produces:
  - `overlay._fit_logit(X: np.ndarray, y: np.ndarray, iters: int = 25, l2: float = 1.0) -> np.ndarray` — returns weight vector `beta` of length `X.shape[1]`. `X` includes a bias column. Ridge-regularized IRLS; numerically guarded (clip probabilities, add `l2` to the diagonal so the Hessian is invertible even with separable/degenerate data).
  - `overlay._predict_logit(beta: np.ndarray, X: np.ndarray) -> np.ndarray` — returns `P(y=1)` per row via the logistic sigmoid.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logit.py
import numpy as np
from rhagent.overlay import _fit_logit, _predict_logit

def test_logit_recovers_monotone_separation():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    # P(win) increases in x; label accordingly with noise
    p = 1 / (1 + np.exp(-3 * x))
    y = (rng.uniform(size=n) < p).astype(float)
    X = np.column_stack([np.ones(n), x])          # bias + feature
    beta = _fit_logit(X, y)
    # predictions should be monotone in x: high-x row > low-x row
    lo = _predict_logit(beta, np.array([[1.0, -2.0]]))[0]
    hi = _predict_logit(beta, np.array([[1.0, 2.0]]))[0]
    assert hi > 0.5 > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

def test_logit_handles_separable_without_blowup():
    # perfectly separable data would send unregularized weights to infinity
    X = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    beta = _fit_logit(X, y, l2=1.0)
    assert np.all(np.isfinite(beta))
    p = _predict_logit(beta, X)
    assert np.all((p >= 0.0) & (p <= 1.0))

def test_logit_all_one_class_returns_finite():
    X = np.array([[1.0, 0.3], [1.0, -0.2], [1.0, 0.5]])
    y = np.array([1.0, 1.0, 1.0])
    beta = _fit_logit(X, y)
    assert np.all(np.isfinite(beta))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_logit.py -v`
Expected: FAIL — `ImportError: cannot import name '_fit_logit'`

- [ ] **Step 3: Implement the helpers**

Add to `overlay.py` (module level; `import numpy as np` already present). Standard ridge-regularized IRLS:

```python
def _predict_logit(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.clip(X @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logit(X: np.ndarray, y: np.ndarray, iters: int = 25, l2: float = 1.0) -> np.ndarray:
    """Ridge-regularized logistic regression via IRLS. X includes a bias column.
    l2 on the diagonal keeps the Hessian invertible under separable/degenerate
    data (and when y is all one class)."""
    n, k = X.shape
    beta = np.zeros(k)
    ridge = l2 * np.eye(k)
    for _ in range(iters):
        p = _predict_logit(beta, X)
        w = np.clip(p * (1.0 - p), 1e-6, None)      # IRLS weights, floored
        # Hessian = X^T W X + ridge ; gradient = X^T (y - p) - l2*beta
        H = X.T @ (w[:, None] * X) + ridge
        g = X.T @ (y - p) - l2 * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_logit.py -v`
Expected: PASS (all 3).

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/overlay.py tests/test_logit.py
git commit -m "feat: numpy IRLS logistic regression helpers for WinProbGate"
```

---

### Task 2: WinProbGate overlay

**Files:**
- Modify: `src/rhagent/overlay.py` (add `WinProbGate` class; register in `build_overlay`), `src/rhagent/papertrade.py` (add `"winprob"` to `--overlay` choices)
- Test: `tests/test_overlay_winprob.py`

**Interfaces:**
- Consumes: `_fit_logit`/`_predict_logit` (Task 1); `features.flatten_trades`; `Decision.conviction` NOT needed; `features.entry_features` for the candidate's live features.
- Produces: `overlay.WinProbGate(thresh: float = 0.52, refit_every: int = 20, min_train: int = 50, l2: float = 1.0)` with `name = "winprob"`; registered `build_overlay("winprob")`; CLI choice `"winprob"`.

**Design (verbatim requirements):**
- Feature vector per trade: `[1.0, feat_vol20, feat_gap, feat_trend5, sym_wr]` where `sym_wr` is a **smoothed target-mean encoding** of the symbol: `(wins_for_symbol + a*prior) / (n_for_symbol + a)` with `prior = overall win rate in closed_trades`, `a = 5.0`. Symbols unseen in `closed_trades` get `sym_wr = prior`. This is walk-forward safe (computed only from closed trades).
- Fit label `y = (outcome == "win")` over `flatten_trades(closed_trades)`.
- **Cold start:** if `len(closed_trades) < min_train`, pass through (`return decision.target`).
- **Refit cadence:** refit the model at most every `refit_every` `adjust` calls; cache `(beta, sym_wr_table, prior)` between refits. (A simple call-counter is fine — exact bar alignment is not required.)
- **Veto rule:** build the candidate's feature row from `entry_features(history)` + its symbol's `sym_wr` (or `prior` if unseen); if `P(win) < thresh`, return `0.0`; else return `decision.target`. Gate-only (no upsizing) in Phase 2.
- If `decision.target == 0.0`, return `0.0` (nothing to gate).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_overlay_winprob.py
import numpy as np, pandas as pd
from rhagent.overlay import WinProbGate
from rhagent.engine import Decision

def _closed(vols, gaps, trends, wins, symbol="NVDA"):
    n = len(vols)
    return pd.DataFrame({
        "symbol": [symbol]*n, "side": ["long"]*n,
        "feat_vol20": vols, "feat_gap": gaps, "feat_trend5": trends,
        "holding_bars": [3]*n,
        "outcome": ["win" if w else "loss" for w in wins],
        "pnl_abs": [100.0 if w else -100.0 for w in wins],
    })

def _hist(vol=0.02, gap=0.0):
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    close = pd.Series(np.linspace(100, 100*(1+gap), 30), index=idx)
    return pd.DataFrame({"open": close, "close": close}, index=idx)

def test_cold_start_passes():
    g = WinProbGate(min_train=50)
    d = Decision(target=1.0, reason="x")
    assert g.adjust("NVDA", _hist(), d, _closed([0.02]*10, [0.0]*10, [1.0]*10, [True]*10)) == 1.0

def test_low_winprob_setup_vetoed_high_passes():
    # Losses cluster at high gap; wins at low gap. Model should veto a high-gap candidate.
    rng = np.random.default_rng(0)
    gaps = np.concatenate([rng.uniform(-0.001, 0.001, 60), rng.uniform(0.02, 0.03, 60)])
    wins = np.array([True]*60 + [False]*60)   # low-gap win, high-gap lose
    vols = [0.02]*120; trends = [0.0]*120
    closed = _closed(list(gaps), list(gaps*0+0.0), trends, list(wins))
    # note: gap is the discriminative feature -> put it in feat_gap
    closed["feat_gap"] = list(gaps)
    g = WinProbGate(thresh=0.5, min_train=50, refit_every=1)
    good = g.adjust("NVDA", _hist(gap=0.0), Decision(target=1.0, reason="x"), closed)     # low gap
    bad = g.adjust("NVDA", _hist(gap=0.025), Decision(target=1.0, reason="x"), closed)    # high gap
    assert good == 1.0
    assert bad == 0.0

def test_zero_target_passthrough():
    g = WinProbGate()
    assert g.adjust("NVDA", _hist(), Decision(target=0.0, reason="x"), _closed([0.02]*60,[0.0]*60,[1.0]*60,[True]*30+[False]*30)) == 0.0
```

Note: `_hist(gap=...)` must produce a candidate whose `entry_features(history)["gap"]` reflects the intended gap. If the synthetic history does not yield the intended `feat_gap`, adjust `_hist` so `entry_features` returns the value the test needs (read `features.entry_features` to see how `gap` is computed — it is `open[-1]/close[-2]-1`).

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_winprob.py -v`
Expected: FAIL — `ImportError: cannot import name 'WinProbGate'`

- [ ] **Step 3: Implement WinProbGate**

Add to `overlay.py`:

```python
class WinProbGate:
    """Veto entries whose predicted win-probability is below `thresh`. Fits a
    ridge logistic model on closed trades' entry features (+ a smoothed
    per-symbol win-rate encoding); walk-forward and refit on a cadence."""

    name = "winprob"
    _FEATS = ["feat_vol20", "feat_gap", "feat_trend5"]

    def __init__(self, thresh=0.52, refit_every=20, min_train=50, l2=1.0) -> None:
        self.thresh = thresh
        self.refit_every = refit_every
        self.min_train = min_train
        self.l2 = l2
        self._beta = None
        self._sym_wr: dict[str, float] = {}
        self._prior = 0.5
        self._calls = 0

    def _design(self, feats: pd.DataFrame, sym_wr: np.ndarray) -> np.ndarray:
        base = feats[self._FEATS].to_numpy(dtype=float)
        bias = np.ones((len(feats), 1))
        return np.column_stack([bias, base, sym_wr.reshape(-1, 1)])

    def _refit(self, closed: pd.DataFrame) -> None:
        df = flatten_trades(closed)
        y = (df["outcome"].to_numpy() == "win").astype(float)
        self._prior = float(y.mean()) if len(y) else 0.5
        # smoothed target-mean encoding per symbol
        a = 5.0
        self._sym_wr = {}
        for sym, grp in df.groupby("symbol"):
            w = (grp["outcome"] == "win").sum()
            self._sym_wr[sym] = float((w + a * self._prior) / (len(grp) + a))
        sym_wr = df["symbol"].map(lambda s: self._sym_wr.get(s, self._prior)).to_numpy(float)
        X = self._design(df, sym_wr)
        self._beta = _fit_logit(X, y, l2=self.l2)

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        target = decision.target
        if target == 0.0 or len(closed_trades) < self.min_train:
            return target
        if self._beta is None or self._calls % self.refit_every == 0:
            self._refit(closed_trades)
        self._calls += 1
        f = entry_features(history)
        row = pd.DataFrame([{ "feat_vol20": f["vol20"], "feat_gap": f["gap"],
                              "feat_trend5": f["trend5"] }])
        sym_wr = np.array([self._sym_wr.get(symbol, self._prior)])
        p = float(_predict_logit(self._beta, self._design(row, sym_wr))[0])
        return target if p >= self.thresh else 0.0
```

Ensure `entry_features` and `flatten_trades` are imported in `overlay.py` (from `.features`); Task 1's `_fit_logit`/`_predict_logit` are in the same module.

- [ ] **Step 4: Register + CLI**

In `build_overlay`, add:

```python
    if name == "winprob":
        return WinProbGate()
```

In `papertrade.py`, extend `--overlay` choices to `["none", "conviction", "bucket", "winprob"]`.

- [ ] **Step 5: Run unit tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_overlay_winprob.py -v`
Expected: PASS. If the discriminative-feature test is flaky, verify the candidate's `entry_features` gap matches the intended regime (adjust `_hist`).

- [ ] **Step 6: Walk-forward integration test**

Add to `tests/test_overlay_integration.py` (or a new file) a test that runs a real `PaperTrader.run()` with `overlay=WinProbGate(min_train=5, refit_every=5)` on a synthetic multi-symbol oscillating source (~150 bars, 2-3 symbols, `StrategyEngine(build("mean_reversion", {}))`). Assert `run()` completes without raising and writes a run dir. This proves the live nested-`entry_features` frame flows through `flatten_trades` correctly (the same class of bug Phase 1 hit with BucketFilter).

```python
def test_winprob_runs_through_papertrader(tmp_path):
    import numpy as np, pandas as pd
    from rhagent.overlay import WinProbGate
    from rhagent.papertrade import PaperTrader
    from rhagent.engine import StrategyEngine
    from rhagent.strategies import build
    idx = pd.date_range("2025-01-01", periods=150, freq="D")
    frames = {}
    for k, s in enumerate(["AAA", "BBB"]):
        c = 100 + np.sin(np.arange(150) / 3.0 + k) * 6
        frames[s] = pd.DataFrame({"open": c, "close": c}, index=idx)
    class _Src:
        def bars(self): return frames
    trader = PaperTrader(engine=StrategyEngine(build("mean_reversion", {})),
                         source=_Src(), out_dir=str(tmp_path),
                         overlay=WinProbGate(min_train=5, refit_every=5))
    run_dir = trader.run()
    assert (run_dir / "run.json").exists()
```

- [ ] **Step 7: Run full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: 194 (Phase 1) + Task 1 (3) + Task 2 (4) new = all green.

- [ ] **Step 8: Commit**

```bash
git add src/rhagent/overlay.py src/rhagent/papertrade.py tests/test_overlay_winprob.py tests/test_overlay_integration.py
git commit -m "feat: WinProbGate overlay (numpy-logit win-probability filter)"
```

---

### Task 3: Run the WinProbGate bake-off (controller, not a subagent)

- [ ] Run long-only: `papertrade --engine mean_reversion --symbols all --overlay winprob --days 400 --cache-dir <MAIN>/data --out-dir <MAIN>/journal/papertrade`
- [ ] Regenerate dashboard; read the robust verdict panel: does winprob's Sharpe / CI improve on baseline (1.16) and on conviction (1.82)?
- [ ] Report the finding (journal is gitignored — no commit).

---

## Self-Review

- **Spec coverage:** §5.2 WinProbGate — numpy logit (Task 1), feature set incl. symbol target-mean encoding (Task 2), refit cadence, cold start, gate-only, walk-forward via `flatten_trades` (Task 2 Step 6). `side` correctly dropped (long-only). ✅
- **Placeholder scan:** none; all code shown. The two "verify/adjust `_hist`" notes are correctness confirmations against `entry_features`, not placeholders.
- **Type consistency:** `_fit_logit(X,y,iters,l2)->beta` and `_predict_logit(beta,X)->p` used identically in Tasks 1-2. `WinProbGate(thresh,refit_every,min_train,l2)` matches registration and tests. `flatten_trades`/`entry_features` reused from `features.py` (Phase 1).
