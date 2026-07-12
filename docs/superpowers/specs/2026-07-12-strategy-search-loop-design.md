# Strategy Search Loop — Design

Date: 2026-07-12
Status: Approved (brainstorming)

Sub-project 2 of 3 in the quant strategy-search framework. Builds on sub-project 1
(the cross-sectional IC/ICIR evaluator, `src/rhagent/factor/`). Sub-project 3 (the
out-of-sample gate + multiple-testing correction) consumes this loop's survivors
and its count of configs tested.

## Problem

A single backtest confirms bias: you find a pattern in noise and mistake it for an
edge. The quant answer is a *loop* — generate config variants, score them, read why
the losers failed, refine, and repeat under a round cap, keeping only survivors that
pass rigorous gates. This sub-project builds that loop for our strategies' parameter
configurations, scored by the in-sample ICIR from sub-project 1, entirely on
in-sample data (the out-of-sample slice stays locked for the sub-project-3 gate).

## Goals

- Adaptive **coarse-to-fine** parameter search per strategy: a coarse grid in round 1,
  then finer grids concentrated around survivors each later round; round cap ≤ 5.
- Score each config by in-sample ICIR (reusing `factor.ic`), plus the measures the
  four survival gates need.
- Enforce four gates each round: **ICIR floor**, **half-life floor**, **parameter
  robustness**, **sign stability**.
- Emit a ranked list of surviving configs, a per-round log (tried / survived / why
  the rest failed), and the **total count of distinct configs evaluated** across all
  rounds — sub-project 3 needs that count for its multiple-testing correction.
- A CLI to run the loop for a strategy and print all of the above.

## Non-goals

- The out-of-sample gate and multiple-testing correction — sub-project 3. This loop
  is strictly in-sample and never reads the OOS slice.
- New strategies or new indicators. The search is over the *existing* strategies'
  parameters only.
- Portfolio construction / P&L. Selection is by signal quality (ICIR), not returns.
- LLM-proposed variants. The refinement is deterministic coarse-to-fine.
- Cross-strategy pooling. The loop runs per strategy; compare strategies by running
  it on each.

## Architecture

New package `src/rhagent/search/`. It consumes sub-project 1 unchanged
(`factor.universe.load_universe`, `factor.split.oos_cutoff`, `factor.signals.signal_panel`,
`factor.ic.{ic_series,icir,ic_decay,half_life}`) and `strategies.build`.

```
space.py    ── per-strategy parameter grids + coarse→fine refinement + neighbors
score.py    ── score one config: ICIR, half-life, sub-period mean-IC signs
loop.py     ── round loop, the four gates, survivor selection, N-tested, round log
__main__.py ── CLI: run the loop for a strategy, print survivors + log + N-tested
```

### 1. Search space — `space.py`

- `SEARCH_SPACES: dict[str, dict[str, list]]` — the coarse grid per strategy:
  - `mean_reversion`: `lookback` `[10, 20, 40, 60]`, `entry` `[0.5, 1.0, 1.5, 2.0]`,
    `exit` `[0.0, 0.5, 1.0]`.
  - `momentum`: `lookback` `[20, 40, 60, 90, 120]`.
  - `linreg`: `min_train` `[30, 60, 90, 120]`.
- `coarse_grids(strategy) -> dict[str, list]` — returns a copy of the strategy's coarse grid.
- `configs_from_grids(grids) -> list[dict]` — the Cartesian product of the per-parameter
  grids as a list of param dicts, in a deterministic order.
- `neighbors(config, grids) -> list[dict]` — the configs differing from `config` by one
  step in exactly one parameter (adjacent value in that parameter's grid list). Used by
  the robustness gate.
- `refine_grids(grids, survivors, max_values=6) -> dict[str, list]` — for each parameter,
  take the survivor values and insert midpoints between each survivor value and its
  adjacent values in the *current* grid; round to int and dedupe for integer parameters;
  keep at most `max_values` values per parameter (those nearest survivor values). This is
  the coarse-to-fine tightening.

`configs_from_grids` is capped by the caller (`loop.py`) via `max_configs` to bound
compute; with the default grids and `max_values=6` the products stay small.

### 2. Config scoring — `score.py`

```python
@dataclass(frozen=True)
class ConfigScore:
    strategy: str
    params: dict
    icir: float
    half_life: int | str | None
    subperiod_ic_signs: tuple[int, ...]   # sign of mean IC in each in-sample sub-period
    n_obs: int                            # IC observations
```

- `score_config(strategy, params, bars_by_symbol, close_is, horizon, min_names, n_subperiods=3) -> ConfigScore`:
  build `strategy` with `params` via `strategies.build`; `panel = signal_panel(strat,
  bars_by_symbol, close_is.index)`; `ic = ic_series(panel, close_is, horizon, min_names)`;
  `icir = icir(ic)`; `half_life = half_life(ic_decay(panel, close_is, min_names=min_names))`;
  split the IC series' time index into `n_subperiods` contiguous equal chunks and record
  the sign (`+1/-1/0`) of each chunk's mean IC. `close_is` is already the in-sample close
  panel (`< cutoff`), so forward returns never cross the boundary. Pure over its panel
  inputs; no I/O.

### 3. The loop and gates — `loop.py`

Gate rules (each toggleable; all on by default):

- **ICIR floor** (`icir_floor`, default 0.3): keep if `icir >= icir_floor`.
- **Half-life floor** (`half_life_floor`, default 5): keep if `half_life == ">50"` or
  (`half_life` is an int and `half_life >= half_life_floor`); reject if `half_life` is
  `None` or an int `< half_life_floor`.
- **Sign stability**: keep if every sub-period mean-IC sign is `+1` (consistently
  positive; a zero or negative sub-period fails).
- **Robustness**: keep if every in-grid neighbor (`space.neighbors`) that was scored this
  round has `icir >= icir_floor`. A config with no scored neighbors fails robustness
  (an isolated point cannot be shown robust). In a full-grid round every interior config
  has scored neighbors.

```python
@dataclass(frozen=True)
class RoundLog:
    round: int
    n_scored: int
    survivors: list[ConfigScore]
    rejected: list[tuple[dict, str]]   # (params, first failing gate)

@dataclass(frozen=True)
class SearchResult:
    strategy: str
    survivors: list[ConfigScore]       # ranked by icir desc
    rounds: list[RoundLog]
    n_tested: int                      # distinct configs scored across all rounds
    best: ConfigScore | None
```

- `run_search(strategy, bars_by_symbol, close_is, *, horizon=5, min_names=10, max_rounds=4,
  top_k=8, max_configs=128, gates=<all on>) -> SearchResult`:
  1. Round 0: `configs_from_grids(coarse_grids(strategy))`, capped at `max_configs`; score
     each; apply gates; survivors = gate-passers ranked by `icir`, truncated to `top_k`.
  2. Rounds 1..`max_rounds`-1: `refine_grids` around the survivors' params;
     `configs_from_grids` minus already-scored configs; score; gate; new survivors.
     Stop early if a round yields no survivors or no config improves on the best `icir`.
  3. `n_tested` = number of distinct configs scored across all rounds (this is the honest
     count sub-project 3 corrects for). `best` = top survivor overall, or `None`.

### 4. CLI — `python -m rhagent.search`

```
python -m rhagent.search --strategy mean_reversion [--horizon 5] [--rounds 4]
    [--icir-floor 0.3] [--half-life-floor 5] [--min-names 10] [--oos-frac 0.25]
    [--days 400] [--symbols A,B,...]
    → load the universe, derive the in-sample close panel (close < oos_cutoff),
      run the loop, and print:
        per-round log (n scored, survivors, sample of why others failed),
        the final ranked survivors (params, ICIR, half-life, sub-period signs),
        N configs tested (for the sub-project-3 correction).
```

`--strategy` maps through `strategies.REGISTRY`/`build`. The in-sample panel is
`close.loc[close.index < oos_cutoff(close.index, oos_frac)]`, mirroring sub-project 1's
CLI so the OOS slice is never touched.

## Error handling

Boundary validation only, matching the existing convention: unknown strategy; a strategy
with no entry in `SEARCH_SPACES`; an empty universe; an `oos_frac`/`days` that leaves no
in-sample days → a clear `ValueError`/`KeyError` and abort. No defensive handling of
impossible internal states.

## Testing (TDD)

- **space**: `configs_from_grids` yields the deterministic Cartesian product; `neighbors`
  returns exactly the one-step-adjacent configs (interior vs edge counts); `refine_grids`
  inserts midpoints around survivors, rounds/dedupes ints, and caps per-parameter size.
- **score**: on a crafted `bars_by_symbol`/`close_is`, `score_config` returns the expected
  ICIR sign and the expected number of sub-period signs; a config whose signal is
  anti-correlated with returns yields negative ICIR and negative sub-period signs.
- **gates**: each gate accepts/rejects the right `ConfigScore` (ICIR below floor rejected;
  `half_life=None` and `< floor` rejected while `">50"` and `>= floor` pass; a config with
  a failing neighbor rejected by robustness; a sign-flipping config rejected).
- **loop**: with a deterministic fake scorer (injected), round 0 scores the coarse grid,
  survivors carry into a refined round, `n_tested` counts distinct configs across rounds,
  and the loop stops at `max_rounds` or when survivors run out. `RoundLog.rejected` records
  the first failing gate.
- **no-lookahead / OOS**: the CLI slices `close_is = close[< cutoff]` before any scoring; a
  test poisons the full close panel's post-cutoff rows, runs the CLI path, and confirms the
  survivors are unchanged — proving nothing after the cutoff influences the search
  (`score_config` itself only ever receives the pre-sliced `close_is`).
- **CLI smoke**: on a seeded small universe, `python -m rhagent.search --strategy momentum`
  runs end-to-end, prints survivors / round log / N-tested, returns 0.

## Increments

1. `space.py` (SEARCH_SPACES, coarse_grids, configs_from_grids, neighbors, refine_grids) + tests.
2. `score.py` (ConfigScore, score_config) + tests.
3. `loop.py` gates (ICIR floor, half-life floor, sign stability, robustness) + tests.
4. `loop.py` `run_search` round loop + N-tested + RoundLog/SearchResult + tests (fake scorer).
5. CLI `__main__.py` + smoke test on cached data.

## Risks / honest notes

- **In-sample search overfits — by design.** Coarse-to-fine tightening around survivors
  makes the best in-sample ICIR optimistic; the more configs tried, the more luck creeps
  in. That is exactly why `n_tested` is a first-class output and why the OOS slice is
  never touched here: sub-project 3's out-of-sample gate + multiple-testing correction is
  the real defense. This loop's job is to *nominate* candidates, not bless them.
- **Compute.** `linreg` recomputes a per-day rolling OLS for every symbol on every config;
  a multi-round search over it is slow. `momentum` and `mean_reversion` are cheap. Runs on
  `linreg` should expect minutes; keep its grid small.
- **Small sample.** Sub-period sign stability splits an already-short in-sample IC series
  (~150–190 days) into three, and those ICs use overlapping forward windows, so each
  sub-period sign is noisy. The gate is a coarse filter, not a precise test.
- **Gate thresholds are conventions**, not laws (ICIR 0.3, half-life 5). They are CLI flags
  so they can be tuned, and their defaults are documented.
