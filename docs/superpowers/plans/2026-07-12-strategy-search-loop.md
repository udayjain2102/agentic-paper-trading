# Strategy Search Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An adaptive coarse-to-fine parameter search that nominates strategy configs by in-sample ICIR, enforcing four overfitting gates per round, and reports the surviving configs plus the honest count of configs tested (for the sub-project-3 correction).

**Architecture:** New package `src/rhagent/search/`. `space.py` holds parameter grids and coarse→fine refinement; `score.py` scores one config by reusing sub-project 1's `factor.ic`; `loop.py` holds the four gates and the round loop; `__main__.py` is the CLI. Strictly in-sample — the out-of-sample slice is never read.

**Tech Stack:** Python 3, pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-12-strategy-search-loop-design.md`

## Global Constraints

- Search is per strategy, over the existing strategies' parameters only; deterministic coarse-to-fine (no LLM, no randomness).
- Reuses sub-project 1 unchanged: `factor.universe.load_universe`, `factor.split.oos_cutoff`, `factor.signals.signal_panel`, `factor.ic.{ic_series, icir, ic_decay, half_life}`, and `strategies.build`.
- Config ordering is deterministic: `configs_from_grids` iterates parameters in `sorted(grids)` key order; a config's identity key is `tuple(sorted(params.items()))`.
- Four gates (all on by default): **ICIR floor** (default 0.3, keep if `icir >= floor`); **half-life floor** (default 5, keep if `half_life` is `">50"`/any `">"` string, or an int `>= floor`; reject if `None` or int `< floor`); **sign stability** (keep only if every sub-period mean-IC sign is `+1`); **robustness** (keep only if every in-grid neighbor scored this round has `icir >= floor`; a config with no scored neighbors fails).
- Round cap `max_rounds` default 4 (≤5); `top_k` survivors carried forward default 8; `max_configs` per round default 128.
- `n_tested` = number of distinct configs scored across all rounds — a first-class output for sub-project 3.
- In-sample only: the CLI derives `close_is = close[close.index < oos_cutoff(...)]`; the OOS slice is never read.
- Error boundaries only: unknown strategy / no `SEARCH_SPACES` entry / empty universe / no in-sample days → clear `ValueError`/`KeyError`. No defensive handling of impossible states.
- Style: `from __future__ import annotations`, module docstrings, small focused files. Tests under `tests/search/`, run with `.venv/bin/python -m pytest`.

---

### Task 1: Parameter search space (`search/space.py`)

**Files:**
- Create: `src/rhagent/search/__init__.py`, `src/rhagent/search/space.py`
- Test: `tests/search/__init__.py`, `tests/search/test_space.py`

**Interfaces:**
- Produces:
  - `SEARCH_SPACES: dict[str, dict[str, list]]`.
  - `coarse_grids(strategy: str) -> dict[str, list]` — a copy of the strategy's coarse grid; unknown strategy raises `KeyError`.
  - `configs_from_grids(grids: dict[str, list]) -> list[dict]` — Cartesian product as param dicts, parameters in `sorted(grids)` order.
  - `neighbors(config: dict, grids: dict[str, list]) -> list[dict]` — configs one adjacent step away in exactly one parameter.
  - `refine_grids(grids: dict[str, list], survivors: list[dict], max_values: int = 6) -> dict[str, list]` — denser grid around survivor values (midpoints to adjacent grid values; ints rounded/deduped; capped per parameter).

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_space.py
from rhagent.search.space import (
    SEARCH_SPACES, coarse_grids, configs_from_grids, neighbors, refine_grids,
)
import pytest


def test_coarse_grids_are_copies():
    g = coarse_grids("mean_reversion")
    g["lookback"].append(999)
    assert 999 not in SEARCH_SPACES["mean_reversion"]["lookback"]


def test_coarse_grids_unknown_strategy_raises():
    with pytest.raises(KeyError):
        coarse_grids("nope")


def test_configs_from_grids_cartesian_deterministic():
    cfgs = configs_from_grids({"lookback": [10, 20], "entry": [1.0, 2.0]})
    # keys iterated in sorted order ["entry","lookback"]; product varies the
    # last key (lookback) fastest
    assert cfgs == [
        {"entry": 1.0, "lookback": 10},
        {"entry": 1.0, "lookback": 20},
        {"entry": 2.0, "lookback": 10},
        {"entry": 2.0, "lookback": 20},
    ]


def test_neighbors_interior_and_edge():
    grids = {"lookback": [10, 20, 40, 60]}
    # interior value 20 -> neighbors 10 and 40
    nb = neighbors({"lookback": 20}, grids)
    assert {"lookback": 10} in nb and {"lookback": 40} in nb and len(nb) == 2
    # edge value 10 -> only neighbor 20
    assert neighbors({"lookback": 10}, grids) == [{"lookback": 20}]


def test_neighbors_multi_param():
    grids = {"a": [1, 2, 3], "b": [10, 20]}
    nb = neighbors({"a": 2, "b": 10}, grids)
    assert {"a": 1, "b": 10} in nb
    assert {"a": 3, "b": 10} in nb
    assert {"a": 2, "b": 20} in nb
    assert len(nb) == 3


def test_refine_grids_inserts_midpoints_int():
    refined = refine_grids({"lookback": [10, 20, 40, 60]}, [{"lookback": 20}])
    # around 20: midpoints to 10 (->15) and 40 (->30)
    assert refined["lookback"] == [15, 20, 30]


def test_refine_grids_float_param():
    refined = refine_grids({"entry": [0.5, 1.0, 1.5, 2.0]}, [{"entry": 1.0}])
    assert refined["entry"] == [0.75, 1.0, 1.25]


def test_refine_grids_caps_values():
    # many survivors across a wide grid -> capped to max_values nearest the survivor mean
    grids = {"x": [0, 10, 20, 30, 40, 50, 60]}
    survivors = [{"x": v} for v in [10, 20, 30, 40, 50]]
    refined = refine_grids(grids, survivors, max_values=4)
    assert len(refined["x"]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/search/test_space.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.search'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/search/__init__.py
"""Adaptive coarse-to-fine strategy-parameter search (quant loop sub-project 2)."""
```

```python
# src/rhagent/search/space.py
"""Parameter grids and coarse-to-fine refinement for the search loop.

A strategy's search space is a small grid per parameter. The loop scores the
Cartesian product of the grids, then refines the grids around the survivors
(inserting midpoints) so later rounds concentrate where the edge appears to be.
"""

from __future__ import annotations

from itertools import product

SEARCH_SPACES: dict[str, dict[str, list]] = {
    "mean_reversion": {
        "lookback": [10, 20, 40, 60],
        "entry": [0.5, 1.0, 1.5, 2.0],
        "exit": [0.0, 0.5, 1.0],
    },
    "momentum": {"lookback": [20, 40, 60, 90, 120]},
    "linreg": {"min_train": [30, 60, 90, 120]},
}


def coarse_grids(strategy: str) -> dict[str, list]:
    if strategy not in SEARCH_SPACES:
        raise KeyError(f"no search space for strategy {strategy!r}")
    return {k: list(v) for k, v in SEARCH_SPACES[strategy].items()}


def configs_from_grids(grids: dict[str, list]) -> list[dict]:
    keys = sorted(grids)
    return [dict(zip(keys, combo)) for combo in product(*(grids[k] for k in keys))]


def neighbors(config: dict, grids: dict[str, list]) -> list[dict]:
    out: list[dict] = []
    for k, vals in grids.items():
        if config.get(k) not in vals:
            continue
        i = vals.index(config[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                nb = dict(config)
                nb[k] = vals[j]
                out.append(nb)
    return out


def refine_grids(grids: dict[str, list], survivors: list[dict], max_values: int = 6) -> dict[str, list]:
    refined: dict[str, list] = {}
    for k, vals in grids.items():
        is_int = all(isinstance(v, int) for v in vals)
        svals = sorted({s[k] for s in survivors if k in s})
        cand: set = set(svals)
        for v in svals:
            if v in vals:
                i = vals.index(v)
                for j in (i - 1, i + 1):
                    if 0 <= j < len(vals):
                        mid = (v + vals[j]) / 2
                        cand.add(round(mid) if is_int else mid)
        merged = sorted(cand)
        if len(merged) > max_values and svals:
            center = sum(svals) / len(svals)
            merged = sorted(sorted(merged, key=lambda x: abs(x - center))[:max_values])
        refined[k] = merged
    return refined
```

- [ ] **Step 4: Create the test package init and run**

```python
# tests/search/__init__.py
```

Run: `.venv/bin/python -m pytest tests/search/test_space.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/search/__init__.py src/rhagent/search/space.py tests/search/
git commit -m "feat: search parameter grids + coarse-to-fine refinement"
```

---

### Task 2: Config scoring (`search/score.py`)

**Files:**
- Create: `src/rhagent/search/score.py`
- Test: `tests/search/test_score.py`

**Interfaces:**
- Consumes: `strategies.build`; `factor.signals.signal_panel`; `factor.ic.{ic_series, icir, ic_decay, half_life}`.
- Produces:
  - `ConfigScore` frozen dataclass: `strategy: str`, `params: dict`, `icir: float`, `half_life: int | str | None`, `subperiod_ic_signs: tuple[int, ...]`, `n_obs: int`.
  - `score_config(strategy, params, bars_by_symbol, close_is, horizon=5, min_names=10, n_subperiods=3) -> ConfigScore`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_score.py
import numpy as np
import pandas as pd

from rhagent.search.score import ConfigScore, score_config


def _panel(n_syms=12, n_bars=60, ascending=True, seed=3):
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="D", name="date")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.002, size=(n_bars, n_syms))
    bars, closes = {}, {}
    for i in range(n_syms):
        k = i if ascending else (n_syms - 1 - i)
        rate = 1 + k / 200
        c = [100.0 * (rate ** t) * (1 + noise[t][i]) for t in range(n_bars)]
        idxc = idx
        bars[f"S{i}"] = pd.DataFrame(
            {"open": c, "high": c, "low": c, "close": c, "volume": [1e6] * n_bars},
            index=idxc,
        )
        closes[f"S{i}"] = c
    return bars, pd.DataFrame(closes, index=idx)


def test_score_config_positive_when_signal_predicts():
    bars, close = _panel(ascending=True)
    sc = score_config("momentum", {"lookback": 40}, bars, close, horizon=1, min_names=5)
    assert isinstance(sc, ConfigScore)
    assert sc.strategy == "momentum" and sc.params == {"lookback": 40}
    assert sc.icir > 0
    assert sc.subperiod_ic_signs == (1, 1, 1)
    assert sc.n_obs > 0


def test_score_config_negative_when_signal_anti_predicts():
    # descending rates: higher trailing return -> lower forward return -> negative IC
    bars, close = _panel(ascending=False)
    sc = score_config("momentum", {"lookback": 40}, bars, close, horizon=1, min_names=5)
    assert sc.icir < 0
    assert all(s == -1 for s in sc.subperiod_ic_signs)


def test_score_config_subperiod_count():
    bars, close = _panel()
    sc = score_config("momentum", {"lookback": 40}, bars, close, horizon=1,
                      min_names=5, n_subperiods=4)
    assert len(sc.subperiod_ic_signs) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/search/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.search.score'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/search/score.py
"""Score one strategy config by its in-sample cross-sectional IC.

Reuses sub-project 1: build the strategy, form its signal panel over the
universe, and compute ICIR, half-life, and the sign of mean IC in each of a few
in-sample sub-periods (for the sign-stability gate). close_is is already the
in-sample close panel, so nothing here reads across the out-of-sample boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..factor.ic import ic_decay, ic_series, half_life, icir
from ..factor.signals import signal_panel
from ..strategies import build


@dataclass(frozen=True)
class ConfigScore:
    strategy: str
    params: dict
    icir: float
    half_life: object  # int | str | None
    subperiod_ic_signs: tuple
    n_obs: int


def _subperiod_signs(ic: pd.Series, k: int) -> tuple:
    ic = ic.dropna().sort_index()
    n = len(ic)
    if n == 0:
        return tuple([0] * k)
    bounds = [round(i * n / k) for i in range(k + 1)]
    signs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        chunk = ic.iloc[a:b]
        if len(chunk) == 0:
            signs.append(0)
        else:
            m = float(chunk.mean())
            signs.append(1 if m > 0 else (-1 if m < 0 else 0))
    return tuple(signs)


def score_config(
    strategy, params, bars_by_symbol, close_is, horizon=5, min_names=10, n_subperiods=3
) -> ConfigScore:
    strat = build(strategy, params)
    panel = signal_panel(strat, bars_by_symbol, close_is.index)
    ic = ic_series(panel, close_is, horizon, min_names)
    icir_val = icir(ic)
    hl = half_life(ic_decay(panel, close_is, min_names=min_names))
    signs = _subperiod_signs(ic, n_subperiods)
    return ConfigScore(strategy, dict(params), icir_val, hl, signs, len(ic))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/search/test_score.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/search/score.py tests/search/test_score.py
git commit -m "feat: score a strategy config by in-sample ICIR + sub-period IC signs"
```

---

### Task 3: The four gates (`search/loop.py` part 1)

**Files:**
- Create: `src/rhagent/search/loop.py`
- Test: `tests/search/test_gates.py`

**Interfaces:**
- Consumes: `ConfigScore` (Task 2); `neighbors` (Task 1).
- Produces:
  - `Gates` frozen dataclass: `icir_floor=0.3`, `half_life_floor=5`, `use_icir=True`, `use_half_life=True`, `use_sign=True`, `use_robustness=True`.
  - `config_key(params: dict) -> tuple` — `tuple(sorted(params.items()))`.
  - `first_failing_gate(score, scored_by_key, grids, gates) -> str | None` — the name of the first gate a config fails, or `None` if it passes all.
  - `apply_gates(scores: list[ConfigScore], grids, gates) -> tuple[list[ConfigScore], list[tuple[dict, str]]]` — `(survivors ranked by icir desc, rejections as (params, failing_gate))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_gates.py
from rhagent.search.loop import Gates, apply_gates, config_key, first_failing_gate
from rhagent.search.score import ConfigScore


def _cs(params, icir=0.4, hl=10, signs=(1, 1, 1)):
    return ConfigScore("momentum", params, icir, hl, signs, 100)


def test_config_key_is_order_independent_hashable():
    assert config_key({"a": 1, "b": 2}) == config_key({"b": 2, "a": 1})


def test_icir_floor_gate():
    grids = {"lookback": [40]}
    low = _cs({"lookback": 40}, icir=0.1)
    assert first_failing_gate(low, {config_key(low.params): low}, grids,
                              Gates(use_robustness=False, use_sign=False,
                                    use_half_life=False)) == "icir_floor"


def test_half_life_gate():
    g = Gates(use_robustness=False, use_sign=False, use_icir=False)
    grids = {"lookback": [40]}
    none_hl = _cs({"lookback": 40}, hl=None)
    short_hl = _cs({"lookback": 40}, hl=3)
    ok_int = _cs({"lookback": 40}, hl=20)
    ok_str = _cs({"lookback": 40}, hl=">50")
    assert first_failing_gate(none_hl, {}, grids, g) == "half_life_floor"
    assert first_failing_gate(short_hl, {}, grids, g) == "half_life_floor"
    assert first_failing_gate(ok_int, {}, grids, g) is None
    assert first_failing_gate(ok_str, {}, grids, g) is None


def test_sign_stability_gate():
    g = Gates(use_robustness=False, use_icir=False, use_half_life=False)
    flip = _cs({"lookback": 40}, signs=(1, -1, 1))
    zero = _cs({"lookback": 40}, signs=(1, 0, 1))
    good = _cs({"lookback": 40}, signs=(1, 1, 1))
    assert first_failing_gate(flip, {}, {"lookback": [40]}, g) == "sign_stability"
    assert first_failing_gate(zero, {}, {"lookback": [40]}, g) == "sign_stability"
    assert first_failing_gate(good, {}, {"lookback": [40]}, g) is None


def test_robustness_gate():
    g = Gates(use_icir=False, use_half_life=False, use_sign=False)  # robustness on
    grids = {"lookback": [20, 40, 60]}
    mid = _cs({"lookback": 40}, icir=0.5)
    lo = _cs({"lookback": 20}, icir=0.5)
    hi = _cs({"lookback": 60}, icir=0.1)   # weak neighbor
    by_key = {config_key(c.params): c for c in (mid, lo, hi)}
    # 40's neighbors are 20 (0.5, ok) and 60 (0.1, below floor) -> fails robustness
    assert first_failing_gate(mid, by_key, grids, g) == "robustness"
    # make both neighbors strong
    hi2 = _cs({"lookback": 60}, icir=0.5)
    by_key2 = {config_key(c.params): c for c in (mid, lo, hi2)}
    assert first_failing_gate(mid, by_key2, grids, g) is None


def test_robustness_no_scored_neighbors_fails():
    g = Gates(use_icir=False, use_half_life=False, use_sign=False)
    grids = {"lookback": [20, 40, 60]}
    lonely = _cs({"lookback": 40}, icir=0.9)
    assert first_failing_gate(lonely, {config_key(lonely.params): lonely}, grids, g) == "robustness"


def test_apply_gates_splits_and_ranks():
    grids = {"lookback": [20, 40, 60]}
    a = _cs({"lookback": 20}, icir=0.6)
    b = _cs({"lookback": 40}, icir=0.8)
    c = _cs({"lookback": 60}, icir=0.1)  # below floor
    survivors, rejected = apply_gates([a, b, c], grids,
                                      Gates(use_robustness=False, use_sign=False,
                                            use_half_life=False))
    assert [s.params["lookback"] for s in survivors] == [40, 20]  # ranked by icir desc
    assert ({"lookback": 60}, "icir_floor") in rejected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/search/test_gates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.search.loop'`

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/search/loop.py
"""The search loop: four survival gates and the coarse-to-fine round loop.

Gates (all on by default): ICIR floor, half-life floor, sign stability across
in-sample sub-periods, and parameter robustness (a config's grid-neighbors must
also clear the ICIR floor, so a lone lucky setting cannot survive). The loop is
strictly in-sample; the count of distinct configs it scores is reported for the
sub-project-3 multiple-testing correction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .score import ConfigScore, score_config
from .space import coarse_grids, configs_from_grids, neighbors, refine_grids


@dataclass(frozen=True)
class Gates:
    icir_floor: float = 0.3
    half_life_floor: int = 5
    use_icir: bool = True
    use_half_life: bool = True
    use_sign: bool = True
    use_robustness: bool = True


def config_key(params: dict) -> tuple:
    return tuple(sorted(params.items()))


def _half_life_ok(hl, floor) -> bool:
    if hl is None:
        return False
    if isinstance(hl, str):  # ">50"
        return True
    return hl >= floor


def first_failing_gate(score: ConfigScore, scored_by_key: dict, grids, gates: Gates):
    if gates.use_icir and score.icir < gates.icir_floor:
        return "icir_floor"
    if gates.use_half_life and not _half_life_ok(score.half_life, gates.half_life_floor):
        return "half_life_floor"
    if gates.use_sign and not (
        score.subperiod_ic_signs and all(s == 1 for s in score.subperiod_ic_signs)
    ):
        return "sign_stability"
    if gates.use_robustness:
        scored_nb = [
            scored_by_key[config_key(n)]
            for n in neighbors(score.params, grids)
            if config_key(n) in scored_by_key
        ]
        if not scored_nb or any(s.icir < gates.icir_floor for s in scored_nb):
            return "robustness"
    return None


def apply_gates(scores: list[ConfigScore], grids, gates: Gates):
    by_key = {config_key(s.params): s for s in scores}
    survivors, rejected = [], []
    for s in scores:
        fail = first_failing_gate(s, by_key, grids, gates)
        if fail is None:
            survivors.append(s)
        else:
            rejected.append((s.params, fail))
    survivors.sort(key=lambda s: s.icir, reverse=True)
    return survivors, rejected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/search/test_gates.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/search/loop.py tests/search/test_gates.py
git commit -m "feat: four search survival gates (ICIR, half-life, sign, robustness)"
```

---

### Task 4: The round loop (`search/loop.py` part 2)

**Files:**
- Modify: `src/rhagent/search/loop.py` (append)
- Test: `tests/search/test_loop.py`

**Interfaces:**
- Consumes: `Gates`, `config_key`, `apply_gates` (Task 3); `coarse_grids`, `configs_from_grids`, `refine_grids` (Task 1); `score_config` (Task 2).
- Produces:
  - `RoundLog` frozen dataclass: `round: int`, `n_scored: int`, `survivors: list`, `rejected: list`.
  - `SearchResult` frozen dataclass: `strategy: str`, `survivors: list`, `rounds: list`, `n_tested: int`, `best`.
  - `run_search(strategy, bars_by_symbol, close_is, *, horizon=5, min_names=10, max_rounds=4, top_k=8, max_configs=128, gates=None, scorer=None) -> SearchResult` — `scorer` is an optional injectable `params -> ConfigScore` (defaults to `score_config`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_loop.py
from rhagent.search.loop import Gates, RoundLog, SearchResult, run_search
from rhagent.search.score import ConfigScore


def _fake_scorer(good_lookbacks):
    def scorer(params):
        icir = 0.5 if params["lookback"] in good_lookbacks else 0.1
        return ConfigScore("momentum", dict(params), icir, 10, (1, 1, 1), 100)
    return scorer


def test_round0_scores_coarse_grid():
    res = run_search("momentum", None, None, max_rounds=1,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40, 60}))
    assert isinstance(res, SearchResult)
    # momentum coarse grid has 5 configs
    assert res.rounds[0].n_scored == 5
    assert res.n_tested == 5
    assert {s.params["lookback"] for s in res.rounds[0].survivors} == {40, 60}


def test_refined_round_scores_new_configs_and_counts_distinct():
    res = run_search("momentum", None, None, max_rounds=2,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40, 60}))
    assert len(res.rounds) == 2
    # round1 refines around {40,60}: midpoints 30,50,75 (20/40 -> 30, 40/60 ->50, 60/90 ->75)
    round1_tested = {tuple(sorted(p.items())) for p in
                     [s.params for s in res.rounds[1].survivors]}
    assert res.n_tested == 8  # 5 coarse + 3 new refined
    # those refined configs are not "good" -> round1 has no survivors
    assert res.rounds[1].survivors == []


def test_best_is_top_icir_survivor():
    res = run_search("momentum", None, None, max_rounds=2,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40, 60}))
    assert res.best is not None
    assert res.best.icir == 0.5
    assert res.best.params["lookback"] in {40, 60}


def test_stops_when_no_survivors():
    res = run_search("momentum", None, None, max_rounds=4,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer(set()))
    # nothing passes -> round 0 has no survivors -> loop stops after round 0
    assert len(res.rounds) == 1
    assert res.best is None
    assert res.survivors == []


def test_rejected_records_failing_gate():
    res = run_search("momentum", None, None, max_rounds=1,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40}))
    gates_hit = {gate for _, gate in res.rounds[0].rejected}
    assert "icir_floor" in gates_hit
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/search/test_loop.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_search'`

- [ ] **Step 3: Write the implementation (append to `src/rhagent/search/loop.py`)**

```python
# append to src/rhagent/search/loop.py


@dataclass(frozen=True)
class RoundLog:
    round: int
    n_scored: int
    survivors: list
    rejected: list


@dataclass(frozen=True)
class SearchResult:
    strategy: str
    survivors: list
    rounds: list
    n_tested: int
    best: object


def run_search(
    strategy,
    bars_by_symbol,
    close_is,
    *,
    horizon=5,
    min_names=10,
    max_rounds=4,
    top_k=8,
    max_configs=128,
    gates=None,
    scorer=None,
):
    gates = gates or Gates()
    if scorer is None:
        def scorer(params):
            return score_config(
                strategy, params, bars_by_symbol, close_is,
                horizon=horizon, min_names=min_names,
            )

    grids = coarse_grids(strategy)
    seen: set = set()
    rounds: list = []
    all_survivors: list = []
    current: list = []
    prev_best = None

    for r in range(max_rounds):
        if r > 0:
            grids = refine_grids(grids, [s.params for s in current])
        configs = [c for c in configs_from_grids(grids) if config_key(c) not in seen]
        configs = configs[:max_configs]
        if not configs:
            break
        scores = []
        for c in configs:
            seen.add(config_key(c))
            scores.append(scorer(c))
        survivors, rejected = apply_gates(scores, grids, gates)
        survivors = survivors[:top_k]
        rounds.append(RoundLog(r, len(scores), survivors, rejected))
        all_survivors.extend(survivors)
        if not survivors:
            break
        round_best = survivors[0]
        if prev_best is not None and r > 0 and round_best.icir <= prev_best.icir:
            break
        prev_best = round_best
        current = survivors

    best_by_key: dict = {}
    for s in all_survivors:
        k = config_key(s.params)
        if k not in best_by_key or s.icir > best_by_key[k].icir:
            best_by_key[k] = s
    ranked = sorted(best_by_key.values(), key=lambda s: s.icir, reverse=True)
    best = ranked[0] if ranked else None
    return SearchResult(strategy, ranked, rounds, len(seen), best)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/search/test_loop.py tests/search/test_gates.py -v`
Expected: all pass (5 loop + 7 gate)

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/search/loop.py tests/search/test_loop.py
git commit -m "feat: coarse-to-fine search round loop with N-tested tracking"
```

---

### Task 5: CLI (`search/__main__.py`) + smoke test

**Files:**
- Create: `src/rhagent/search/__main__.py`
- Test: `tests/search/test_search_cli.py`

**Interfaces:**
- Consumes: `strategies.REGISTRY`; `factor.universe.{UNIVERSE, load_universe}`; `factor.split.oos_cutoff`; `run_search`, `Gates` (Tasks 3–4).
- Produces: `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_search_cli.py
import pandas as pd
import pytest

from rhagent.search.__main__ import main


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_runs_search(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    for k in range(6):
        closes = [100.0 + k + (0.5 * k + 1) * i for i in range(140)]
        _seed(cache, f"S{k}", closes)
    rc = main([
        "--strategy", "momentum", "--symbols", "S0,S1,S2,S3,S4,S5",
        "--rounds", "2", "--min-names", "3", "--days", "200",
        "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "round 0" in out.lower()
    assert "configs tested" in out.lower()


def test_cli_unknown_strategy_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--strategy", "nope", "--symbols", "S0", "--cache-dir", str(tmp_path)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/search/test_search_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhagent.search.__main__'` (or ImportError for `main`)

- [ ] **Step 3: Write the implementation**

```python
# src/rhagent/search/__main__.py
"""CLI: run the coarse-to-fine strategy search for one strategy, in-sample only.

    python -m rhagent.search --strategy mean_reversion [--horizon 5] [--rounds 4]
        [--icir-floor 0.3] [--half-life-floor 5] [--min-names 10] [--oos-frac 0.25]
        [--days 400] [--symbols A,B,...]

Prints the per-round log, the ranked surviving configs, and the total number of
configs tested (which the sub-project-3 gate corrects for). The out-of-sample
slice (dates >= the cutoff) is never read.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from ..factor.split import oos_cutoff
from ..factor.universe import UNIVERSE, load_universe
from ..strategies import REGISTRY
from .loop import Gates, run_search


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="rhagent.search")
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--icir-floor", type=float, default=0.3)
    p.add_argument("--half-life-floor", type=int, default=5)
    p.add_argument("--min-names", type=int, default=10)
    p.add_argument("--oos-frac", type=float, default=0.25)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--symbols", help="comma-separated override of the default universe")
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else UNIVERSE
    )
    end = date.today()
    start = end - timedelta(days=args.days)
    bars_by_symbol, close = load_universe(
        symbols, start.isoformat(), end.isoformat(), cache_dir=args.cache_dir
    )
    cutoff = oos_cutoff(close.index, args.oos_frac)
    close_is = close.loc[close.index < cutoff]
    if len(close_is) == 0:
        p.error("no in-sample days after the out-of-sample split")

    gates = Gates(icir_floor=args.icir_floor, half_life_floor=args.half_life_floor)
    result = run_search(
        args.strategy, bars_by_symbol, close_is,
        horizon=args.horizon, min_names=args.min_names,
        max_rounds=args.rounds, gates=gates,
    )

    print(f"strategy: {result.strategy}   universe: {len(bars_by_symbol)} names   "
          f"in-sample days: {len(close_is)}")
    for rl in result.rounds:
        print(f"\nround {rl.round}: {rl.n_scored} scored, {len(rl.survivors)} survived")
        for s in rl.survivors[:5]:
            print(f"  survive  {s.params}  ICIR={s.icir:+.3f}  half_life={s.half_life}  "
                  f"signs={s.subperiod_ic_signs}")
        for params, gate in rl.rejected[:5]:
            print(f"  reject   {params}  ({gate})")
    print(f"\ntop survivors (ranked by ICIR):")
    if not result.survivors:
        print("  none passed all gates")
    for s in result.survivors[:10]:
        print(f"  {s.params}  ICIR={s.icir:+.3f}  half_life={s.half_life}")
    print(f"\nconfigs tested: {result.n_tested}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (existing suite + the new search tests).

- [ ] **Step 5: End-to-end smoke on cached data**

Run: `PYTHONPATH=src .venv/bin/python -m rhagent.search --strategy mean_reversion --symbols AAPL,MSFT,NVDA,SPY --min-names 2 --rounds 2 --days 400`
Expected: prints the per-round log, any surviving configs (possibly none on 4 names — that's fine), and "configs tested: N". Only 4 cached names, so `--min-names 2`; this proves the loop runs end-to-end, per the spec's small-sample caveat.

- [ ] **Step 6: Commit**

```bash
git add src/rhagent/search/__main__.py tests/search/test_search_cli.py
git commit -m "feat: search CLI - run coarse-to-fine loop, print survivors + N-tested"
```

---

## Self-Review Notes

- **Spec coverage:** coarse-to-fine grids + refinement (T1); config scoring via SP1 (T2); the four gates (T3); the round loop + N-tested + round log + stop conditions (T4); CLI + in-sample-only split + smoke (T5). All spec sections map to a task.
- **In-sample / OOS discipline:** the CLI derives `close_is = close[< oos_cutoff(...)]` (T5) and passes only that into `run_search`/`score_config`; the OOS slice is never read. `n_tested` is surfaced for sub-project 3.
- **Type consistency:** `ConfigScore(strategy, params, icir, half_life, subperiod_ic_signs, n_obs)`, `config_key(params) -> tuple`, `Gates(...)`, `first_failing_gate(score, scored_by_key, grids, gates) -> str|None`, `apply_gates(scores, grids, gates) -> (survivors, rejected)`, `run_search(...) -> SearchResult`, `score_config(strategy, params, bars_by_symbol, close_is, horizon, min_names, n_subperiods)` — consistent across tasks and the CLI.
- **Determinism note for the refined-round test (T4):** momentum coarse grid `[20,40,60,90,120]`; survivors `{40,60}` → `refine_grids` inserts midpoints 30 (20/40), 50 (40/60), 75 (60/90) → new configs `{30,50,75}`, so `n_tested == 8`. This is exercised by `test_refined_round_scores_new_configs_and_counts_distinct`.
