# Out-of-Sample Gate + Multiple-Testing Correction — Design

Date: 2026-07-12
Status: Approved (brainstorming)

Sub-project 3 of 3 — the final gate in the quant strategy-search framework. It
consumes sub-project 2's search (`src/rhagent/search/`) and sub-project 1's
IC/ICIR machinery (`src/rhagent/factor/`), and is the first and only place the
locked out-of-sample slice is read.

## Problem

A config that looks good in-sample may just be the luckiest of many tries. The
antidote, per the quant loop, is a strict gate on data never used during the
search, with the significance bar raised for how many configs were tried. This
sub-project runs the in-sample search, then tests each surviving config on the
held-out slice and applies two multiple-testing corrections. Only configs that
clear every gate are stamped viable.

## Goals

- Orchestrate the full pipeline: run sub-project 2's search in-sample → take its
  survivors and the count of configs tested → evaluate each survivor on the
  out-of-sample slice → apply the gates → produce a viability verdict.
- Recompute each survivor's ICIR and IC half-life on the OOS slice.
- Compute two multiple-testing corrections keyed to N = configs tested: a
  **Bonferroni**-adjusted p-value and a **Deflated Sharpe Ratio**, with no scipy
  (a pure-Python normal CDF/inverse-CDF).
- A **strict** verdict: viable only if OOS ICIR holds (≥ 50% of in-sample, same
  sign, in-sample positive) AND OOS decay holds AND both corrections pass.
- A CLI that runs the whole thing and prints the per-survivor verdict table.

## Non-goals

- New strategies, new gates in the search, or portfolio P&L.
- Changing sub-project 1's `factor/` package. Sub-project 2 gets one small
  additive field (`SearchResult.all_scores`) — see below.
- Beating the small-sample reality: on the current cached data the OOS slice is
  tiny and almost nothing will pass. That is the correct, honest outcome, not a
  defect to engineer around.

## Architecture

New package `src/rhagent/gate/`. Consumes sub-project 2
(`search.loop.run_search`, `search.score.ConfigScore`) and sub-project 1
(`factor.universe.load_universe`, `factor.split.oos_cutoff`,
`factor.signals.signal_panel`, `factor.ic.{ic_series, icir, ic_decay, half_life}`).

```
stats.py    ── normal CDF/inverse-CDF, Bonferroni p-value, Deflated Sharpe (pure, no scipy)
oos.py      ── recompute a config's ICIR/decay on the OOS slice; the strict verdict
gate.py     ── orchestrate search → OOS eval → corrections → GateResult
__main__.py ── CLI: run the whole pipeline, print the verdict table
```

### 0. Sub-project 2 extension (prerequisite)

The Deflated Sharpe Ratio needs the spread of ICIRs across **all** tested configs
(their variance), not just the survivors. `run_search` already builds a
cumulative `scored_all` dict; expose it:

- Add `all_scores: list` to `search.loop.SearchResult` (every distinct
  `ConfigScore` scored across all rounds; `len(all_scores) == n_tested`).
- Populate it from `scored_all.values()` in `run_search`. Purely additive; no
  behavior change to the search or its other outputs.

### 1. Statistics — `stats.py` (pure, no I/O, no scipy)

- `norm_cdf(x: float) -> float`: `0.5 * math.erfc(-x / sqrt(2))`.
- `norm_ppf(p: float) -> float`: the inverse normal CDF via Acklam's rational
  approximation (accurate to ~1e-9 on `0 < p < 1`); raises `ValueError` for
  `p <= 0` or `p >= 1`.
- `bonferroni(icir: float, n_eff: int, n_tested: int, alpha: float = 0.05) -> tuple[float, float, bool]`:
  returns `(p_value, threshold, passed)`. The two-sided p-value of the IC-series
  mean is `p = 2 * (1 - norm_cdf(abs(icir) * sqrt(n_eff)))` (the t-stat of a
  mean over `n_eff` samples is `mean/std * sqrt(n_eff) = ICIR * sqrt(n_eff)`).
  `threshold = alpha / max(n_tested, 1)`; `passed = p_value < threshold`.
- `deflated_sharpe(sr, n_eff, skew, kurt_excess, n_trials, var_trials) -> float`:
  the López de Prado Deflated Sharpe Ratio, a probability in `[0, 1]`.
  - Expected max Sharpe of `n_trials` trials:
    `sr0 = sqrt(var_trials) * ((1 - γ) * norm_ppf(1 - 1/n_trials) + γ * norm_ppf(1 - 1/(n_trials * e)))`,
    with `γ = 0.5772156649` (Euler-Mascheroni), `e = math.e`. If `n_trials < 2`
    or `var_trials <= 0`, `sr0 = 0`.
  - `γ4 = kurt_excess + 3` (pandas kurtosis is excess; convert to non-excess).
    `denom = 1 - skew * sr + (γ4 - 1) / 4 * sr**2`. If `n_eff <= 1` or
    `denom <= 0`, return `0.0` (fail).
  - `dsr = norm_cdf((sr - sr0) * sqrt(n_eff - 1) / sqrt(denom))`.

`n_eff` is the effective independent sample size = `max(oos_ic_obs // horizon, 1)`
— the overlapping-window discount (consecutive daily ICs at horizon `h` share
`h-1` forward days), consistent with the caveat sub-project 1 already reports.

### 2. Out-of-sample evaluation + verdict — `oos.py`

- `evaluate_oos(strategy, params, bars_by_symbol, close, cutoff, horizon, min_names) -> dict`:
  build the strategy's signal panel over the **full** history
  (`signal_panel(strat, bars_by_symbol, close.index)` — signals are causal and
  need in-sample warmup), restrict to OOS dates (`>= cutoff`), and score IC
  against the OOS close panel (`close_oos = close.loc[close.index >= cutoff]`):
  `ic = ic_series(panel_oos, close_oos, horizon, min_names)`. Returns
  `{"oos_icir": icir(ic), "oos_half_life": half_life(ic_decay(panel_oos, close_oos, min_names=min_names)), "oos_ic": ic, "n_obs": len(ic)}`.
  This is the only read of the OOS slice in the whole framework.
- `icir_holds(is_icir, oos_icir, retention=0.5) -> bool`: `is_icir > 0` AND
  `oos_icir > 0` AND `oos_icir >= retention * is_icir`.
- `decay_holds(oos_half_life, floor) -> bool`: reuse the half-life rule (`">"`
  string or int `>= floor` passes; `None` or int `< floor` fails).
- `verdict(is_icir, oos_icir, oos_half_life, bonf_pass, dsr_pass, half_life_floor) -> tuple[bool, str]`:
  checks in order — ICIR holds, decay holds, Bonferroni, DSR — and returns
  `(viable, reason)` where `reason` is `"viable"` or the first failing gate
  (`"icir_did_not_hold"`, `"decay_did_not_hold"`, `"failed_bonferroni"`,
  `"failed_deflated_sharpe"`).

### 3. Orchestration — `gate.py`

```python
@dataclass(frozen=True)
class GateRow:
    params: dict
    is_icir: float
    oos_icir: float
    oos_half_life: object
    bonf_p: float
    bonf_threshold: float
    bonf_pass: bool
    dsr: float
    dsr_pass: bool
    viable: bool
    reason: str

@dataclass(frozen=True)
class GateResult:
    strategy: str
    n_tested: int
    rows: list          # GateRow per survivor
    viable: list         # GateRows that passed
```

- `run_gate(strategy, bars_by_symbol, close, *, horizon=5, min_names=10, oos_frac=0.25, rounds=4, icir_floor=0.3, half_life_floor=5, alpha=0.05, dsr_threshold=0.95) -> GateResult`:
  1. `cutoff = oos_cutoff(close.index, oos_frac)`; `close_is = close[close.index < cutoff]`.
  2. `search = run_search(strategy, bars_by_symbol, close_is, horizon=horizon, min_names=min_names, max_rounds=rounds, gates=Gates(icir_floor=icir_floor, half_life_floor=half_life_floor))`.
  3. `var_trials = variance of [s.icir for s in search.all_scores]` (population).
  4. For each survivor in `search.survivors`: `evaluate_oos`; `n_eff = max(oos n_obs // horizon, 1)`; `bonferroni(oos_icir, n_eff, search.n_tested, alpha)`; `deflated_sharpe(oos_icir, n_eff, oos_ic.skew(), oos_ic.kurt(), search.n_tested, var_trials)` with `dsr_pass = dsr > dsr_threshold`; `verdict(...)`; assemble a `GateRow`.
  5. Return `GateResult(strategy, search.n_tested, rows, [r for r in rows if r.viable])`.

  Skew/kurtosis of a short OOS IC series can be `NaN`; treat `NaN` skew/kurt as
  `0.0` (Gaussian) before calling `deflated_sharpe`.

### 4. CLI — `python -m rhagent.gate`

```
python -m rhagent.gate --strategy mean_reversion [--horizon 5] [--min-names 10]
    [--oos-frac 0.25] [--rounds 4] [--icir-floor 0.3] [--half-life-floor 5]
    [--alpha 0.05] [--dsr-threshold 0.95] [--days 400] [--symbols A,B,...]
    → load the universe, run the full gate, and print:
        strategy, universe size, configs tested (N), the correction thresholds,
        a per-survivor row (params, IS ICIR, OOS ICIR, OOS half-life,
        Bonferroni p vs alpha/N, DSR, VIABLE/reason),
        and the count of viable configs.
```

`--strategy` maps through `strategies.REGISTRY`. `close` is the full universe
panel; `run_gate` performs the in-sample/OOS split internally.

## Error handling

Boundary validation only: unknown strategy; empty universe; an `oos_frac`/`days`
leaving no in-sample or no OOS days → clear `ValueError`/`KeyError`. `norm_ppf`
raises on `p <= 0` or `p >= 1`. No defensive handling of impossible states.

## Testing (TDD)

- **stats**: `norm_cdf`/`norm_ppf` match known values (`norm_cdf(0)=0.5`,
  `norm_cdf(1.96)≈0.975`, `norm_ppf(0.975)≈1.96`, round-trip); `bonferroni`
  computes the right p-value and threshold and flips `passed` at the boundary;
  `deflated_sharpe` → near 1 for a strong SR with few trials, near 0 for a weak
  SR with many trials, and handles `var_trials=0` (sr0=0) and `denom<=0`/`n_eff<=1`
  (returns 0.0).
- **oos**: `evaluate_oos` on constructed data returns an OOS ICIR of the expected
  sign and uses only OOS dates; `icir_holds`/`decay_holds` accept/reject at the
  right boundaries; `verdict` returns the correct first-failing reason for each
  gate and `"viable"` only when all pass.
- **gate**: `run_gate` end-to-end with an injectable search/scorer or crafted
  data — a survivor that holds OOS and passes both corrections is marked viable;
  one that decays OOS is rejected with the right reason; `var_trials` is drawn
  from `all_scores`; `n_tested` flows through.
- **SP2 extension**: `SearchResult.all_scores` is present and `len(all_scores)
  == n_tested`; existing search tests still pass.
- **CLI smoke**: on cached data, `python -m rhagent.gate --strategy mean_reversion`
  runs end-to-end, prints the verdict table and "viable: N" (N is likely 0 on the
  thin universe — that's expected), returns 0.

## Increments

1. Sub-project 2 extension: `SearchResult.all_scores` + test.
2. `stats.py` (norm cdf/ppf, bonferroni, deflated_sharpe) + tests.
3. `oos.py` (evaluate_oos, icir_holds, decay_holds, verdict) + tests.
4. `gate.py` (run_gate, GateRow, GateResult) + tests.
5. CLI `__main__.py` + smoke test.

## Risks / honest notes

- **Tiny OOS sample.** ~25% of ~275 bars ≈ 68 days; at horizon 5 the effective
  independent sample is ~13. A strict Bonferroni across dozens of trials on ~13
  points is nearly impossible to pass, and the Deflated Sharpe is equally
  demanding. Expect **zero viable** on the current cached data. This is the gate
  working — it is telling you there isn't enough evidence — not a bug. The gate
  becomes meaningful with the full ~65-name universe and more history.
- **`norm_ppf` is a rational approximation** (no scipy); accurate to ~1e-9, ample
  for these thresholds.
- **DSR inputs from a short series.** Skew/kurtosis of a ~13-point IC series are
  themselves noisy; `NaN` is coerced to Gaussian. The DSR is a guardrail, not a
  precise probability, at this sample size.
- **Requiring both corrections is deliberately strict.** Per the approved
  verdict, a config must satisfy Bonferroni AND the Deflated Sharpe; either alone
  is not enough. This minimizes false "edges" at the cost of rejecting almost
  everything on thin data.
