# Loss-Learning Bake-Off — Design Spec

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Author:** brainstormed with the user

## 1. Problem

The trading strategies (`momentum`, `mean_reversion`, `linreg`) are fixed rule-based
functions of price bars. They never adapt to their own results. The only component with
any feedback is the LLM `agent` engine, which receives a one-line summary of where past
losses concentrated (`lessons_from_runs`) as a soft prompt hint — not weight updates, not
per-trade learning, and only for the agent.

The user wants genuine learning-from-losses. Two constraints surfaced during design:

- **Trade-level noise.** Broadening to 66 symbols multiplied trades 30–130× and collapsed
  returns toward zero — most of the added trades are coin-flips, not edge.
- **Verdict-level noise.** With near-zero edge, any bake-off risks crowning whichever
  variant got *lucky* in one window rather than the one that actually learned.

## 2. Goal

Build three different learning mechanisms as interchangeable variants the existing paper-trade
harness can run, plus a noise-reduction layer and a statistically honest evaluator, so we can
**compare them empirically and rank by risk-adjusted return (Sharpe)** on the existing all-runs
dashboard. If nothing clears the noise band, that is itself a valid, valuable finding.

Success metric (chosen by user): **Sharpe ratio**, judged robustly (see §6).

## 3. Core architecture

### 3.1 One seam: the decision overlay

Today a strategy emits a target position (`Decision.target ∈ {-1,0,+1}`) and it is taken as-is
at `papertrade.py:156-157`:

```python
d = self.engine.decide(sym, history, prev)
target = d.target
```

We insert **one** layer between the raw target and the position actually taken. Every learning
approach and the conviction filter is a different implementation of the same `Overlay` protocol,
plugged into this single point. This keeps them directly comparable (same harness, same dashboard)
and builds the plumbing once.

**New file `src/rhagent/overlay.py`:**

```python
class Overlay(Protocol):
    name: str
    def adjust(
        self,
        symbol: str,
        history: pd.DataFrame,      # bars up to and including today
        decision: Decision,          # raw target + conviction (see 3.3)
        closed_trades: pd.DataFrame, # trades that closed STRICTLY before today
    ) -> float:                      # final target in [-1, 1]; 0 = veto, fraction = downsize
        ...
```

Return value semantics: `0.0` vetoes the trade, a fraction (e.g. `0.3`) downsizes it, a value
equal to `decision.target` passes it through unchanged. The baseline is an `IdentityOverlay`
that returns `decision.target`.

### 3.2 Where the overlay is applied

Applied in `PaperTrader.run()` immediately after line 156-157, before the existing final-bar
guard (163-164) and the open/close logic (179+). At that call site the loop's `trades` list
contains only trades that closed on *prior* bars (closes happen at 181-182, after the decide at
156), so **walk-forward isolation is free** — no lookahead is possible by construction. We pass
`pd.DataFrame(trades)` (may be empty early) as `closed_trades`.

`PaperTrader.__init__` gains an `overlay: Overlay = IdentityOverlay()` parameter. The loop
becomes:

```python
d = self.engine.decide(sym, history, prev)
target = self.overlay.adjust(sym, history, d, pd.DataFrame(trades))
# ... existing final-bar guard and open/close logic unchanged, using `target`
```

Fractional targets flow naturally through the existing code: `qty = target` is stored on the
trade, `turnover = abs(target - prev)` and P&L (`sign * (price/entry - 1) ...`) already handle
non-unit sizes.

### 3.3 Exposing conviction without coupling

The conviction gate needs a continuous score, but the overlay must not reach into the strategy.
Solution: add one optional field to the existing frozen dataclass.

```python
@dataclass(frozen=True)
class Decision:
    target: float
    reason: str
    conviction: float | None = None   # NEW: continuous signal strength at this bar
```

`StrategyEngine.decide` sets `conviction = float(self.strat.signal(history).iloc[-1])` (all three
strategies expose `signal()`; it is `NaN` during warmup — passed through as-is). `AgentEngine`
leaves it `None`. Overlays that need conviction treat `None`/`NaN` as "no signal" and no-op.

Adding a defaulted field is backward compatible: every existing `Decision(target=, reason=)`
call still works.

## 4. The conviction gate (trade-level noise)

`ConvictionGate(Overlay)` — kills coin-flip trades. Only enter when `|conviction|` clears a
threshold estimated from *past* bars (no lookahead).

- Threshold = a rolling percentile of `|signal|` over the trailing window (default: 60th
  percentile over the last 120 bars). Recompute from `history["close"]`-derived signal each bar,
  or cache per-symbol.
- If `decision.conviction is None or NaN`: pass through (no-op).
- If `|conviction| < threshold`: return `0.0` (veto). Else return `decision.target`.

This turns linreg's 4,227 near-random trades into a few hundred that lean on the signal. In
Phase 1 it runs as a standalone variant (`--overlay conviction`). Composing it *before* a learner
(gate first, learner on survivors) is a natural extension but is **deferred** — the Phase-1
bake-off compares standalone variants only.

Parameters: `pctile: float = 0.60`, `window: int = 120`.

## 5. The three learning overlays (bake-off contestants)

All consume only `closed_trades` (walk-forward). Entry features available at decision time:
`feat_vol20`, `feat_gap`, `feat_trend5`, `side`, `symbol` (these are the only entry-time fields;
`holding_bars`/`outcome`/`pnl_*` are exit-time and used only as *labels*, never as inputs).

Because the overlay receives `history` and a raw `Decision` (which does not carry the entry
features), each overlay recomputes the same three entry features from `history` using the shared
helper extracted from `papertrade.entry_features` (see §7). This guarantees the features an
overlay scores match the features the ledger stored.

### 5.1 BucketFilter (Phase 1)

Deterministic, fully inspectable. From `closed_trades`, compute loss-share per bucket by reusing
`evaluate.failure_buckets`. For today's candidate, derive its bucket labels (vol/gap/side; skip
`holding` which is unknown at entry) via the same `_bucket_labels` logic on the recomputed entry
features.

- If any of the candidate's buckets has `loss_share >= veto_share` AND `win_rate <= veto_wr` with
  `n_trades >= min_n`: return `0.0` (veto). Otherwise size down proportionally to the worst
  bucket's loss-share, floored at `min_size`, or pass through if no bucket qualifies.
- Cold start (fewer than `min_n` closed trades): pass through.

Parameters: `veto_share=0.25`, `veto_wr=0.40`, `min_n=20`, `min_size=0.3`.

### 5.2 WinProbGate (Phase 2)

A learned gate. Fit a **numpy logistic regression** (IRLS, ~15 lines — no scikit-learn, which is
not in the repo) on `closed_trades`: X = one-hot(`side`) + one-hot(`symbol`, hashed or top-K) +
`[feat_vol20, feat_gap, feat_trend5]`; y = `outcome == "win"`. Predict `P(win)` for the candidate.

- Refit at most every `refit_every` bars (default 20) for speed; cache the fitted weights.
- If `P(win) < thresh` (default 0.52): veto. Else pass through (optionally upsize toward 1.0 when
  `P(win)` is high — deferred; Phase 2 ships gate-only).
- Cold start (fewer than `min_train` closed trades, default 50): pass through.

Symbol encoding: to avoid a 66-wide one-hot exploding a tiny model, encode `symbol` as a single
learned target-mean feature (mean win-rate of that symbol in `closed_trades`, smoothed) rather
than one-hot. This keeps the feature vector small and walk-forward safe.

Parameters: `thresh=0.52`, `refit_every=20`, `min_train=50`.

### 5.3 ParamTune (Phase 3)

Not an overlay — it re-calibrates the strategy itself. Wraps `StrategyEngine`: on a rolling
schedule, re-fit the strategy's own knobs (momentum `lookback`, mean_reversion `entry`/`lookback`)
by grid-search over a small set, scored by trailing-window Sharpe of the resulting positions, and
use the winning params for the next segment. Walk-forward: the trailing window ends strictly
before the segment it configures.

Implemented as a `TunedStrategyEngine(DecisionEngine)` that owns a base strategy class, a param
grid, and re-fits every `resegment` bars. Deferred to Phase 3.

## 6. Robust evaluator (verdict-level noise)

**New file `src/rhagent/evaluate_robust.py`.** Instead of one Sharpe per run:

- **Fold Sharpe.** Split the returns series (`returns.csv`, one equal-weight net value per bar)
  into rolling folds (default 60-bar, 50% overlap). Report mean and standard deviation of
  per-fold Sharpe. A variant that only wins in one fold is exposed.
- **Bootstrap CI.** Resample the per-bar net returns with replacement (N=1000), recompute the
  annualized Sharpe (same formula as `backtest.result_from_returns`: `mean/std * sqrt(252)`) each
  time, report the 95% percentile interval.
- **Deflated Sharpe.** Adjust the observed Sharpe for the number of variants tried (M), penalizing
  multiple testing, using the standard deflated-Sharpe formulation over the M candidate Sharpes.
- **Verdict.** A variant "beats baseline" only if its bootstrap CI lower bound exceeds the
  baseline's point Sharpe. Output a table: variant, point Sharpe, fold mean±sd, 95% CI, deflated
  Sharpe, beats-baseline (bool).

Uses only numpy/pandas (scipy absent — the normal-CDF needed for deflated Sharpe is computed via
`math.erf`).

**Dashboard.** Add one panel to `scripts/make_dashboard.py` (all-runs view) that renders this
comparison table across the current base-dir's runs, sorted by deflated Sharpe, with the
beats-baseline flag highlighted. No change to the per-run sections.

## 7. Shared refactor

Extract the entry-feature computation from `papertrade.entry_features` into a reusable function
(same module or a small `features.py`) so both the ledger writer and the overlays compute
identical features. No behavior change — pure extraction, guarded by the existing tests.

## 8. CLI & run flow

`papertrade` gains `--overlay {none,conviction,bucket,winprob}` (default `none`). Example
Phase-1 bake-off on the chosen testbed (`mean_reversion`):

```
for ov in none conviction bucket; do
  papertrade --engine mean_reversion --symbols all --allow-short --overlay $ov ...
done
evaluate_robust  --base-dir journal/papertrade     # ranks variants
make_dashboard   --base-dir journal/papertrade     # renders bake-off panel
```

ParamTune (Phase 3) is selected via a separate `--tune` flag on the engine rather than `--overlay`,
since it wraps the engine rather than post-processing a target.

## 9. Scope & phasing

- **Phase 1** (cheap, correct core): `Overlay` protocol + seam wiring + `Decision.conviction` +
  `ConvictionGate` + `BucketFilter` + `evaluate_robust` + dashboard panel + `entry_features`
  extraction. Answers: *can filtering + the simplest learner beat baseline Sharpe, trustworthily?*
- **Phase 2**: `WinProbGate` (numpy logistic regression).
- **Phase 3**: `ParamTune` (`TunedStrategyEngine`).

**Testbed:** `mean_reversion` first (stable, clean z-score conviction). Overlays are
engine-agnostic — re-runnable on `momentum`/`linreg` with no new code.

### Out of scope (YAGNI)
- No `forward.py` / live wiring. The forward rule-strategy path is vectorized and bypasses
  `decide`; wiring overlays there is a separate effort, only worthwhile if a variant proves out in
  backtest first.
- No hyperparameter search over the overlays' own parameters.
- No upsizing in WinProbGate Phase 2 (gate-only).
- No real-money path (unchanged; `LIVE=false`, no broker token).

## 10. Testing

Each non-trivial unit leaves one runnable check (assert-based `demo()`/`__main__` or a small
`test_*.py`), consistent with the repo's style:

- **ConvictionGate**: low-`|conviction|` bar → `0.0`; high → passthrough; `None` conviction → passthrough.
- **BucketFilter**: synthetic `closed_trades` with one heavily-losing bucket → candidate in that
  bucket vetoed; candidate in a clean bucket passes; cold start passes.
- **WinProbGate**: numpy logit recovers a separable toy problem (P(win) monotone in a feature);
  cold start passes.
- **Walk-forward leak test** (the critical one): assert every row in the `closed_trades` an overlay
  receives has `exit_ts < today`'s bar timestamp — run through `PaperTrader` and check no overlay
  ever sees a same-day or future close.
- **evaluate_robust**: a known return series yields the known point Sharpe; bootstrap CI brackets
  it; deflated Sharpe ≤ point Sharpe.

Full existing suite (171 tests) must still pass — the `Decision` field add and `entry_features`
extraction are the only touches to existing code paths.

## 11. Files

- New: `src/rhagent/overlay.py`, `src/rhagent/evaluate_robust.py`, tests.
- Modified: `src/rhagent/engine.py` (`Decision.conviction`, set it in `StrategyEngine`),
  `src/rhagent/papertrade.py` (overlay param + seam call + `--overlay` CLI + feature extraction),
  `scripts/make_dashboard.py` (bake-off panel).
- Phase 2/3 add to `overlay.py` / a `tuned_engine.py`.
