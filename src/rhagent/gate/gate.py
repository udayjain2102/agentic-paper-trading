"""Orchestrate the full out-of-sample gate.

Run sub-project 2's search in-sample, then for each surviving config recompute
its ICIR/decay on the locked out-of-sample slice and apply the two
multiple-testing corrections. The Deflated Sharpe's trial variance comes from
every config the search scored (SearchResult.all_scores). Only configs that pass
every gate are viable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..factor.split import oos_cutoff
from ..search.loop import Gates, run_search
from .oos import evaluate_oos, verdict
from .stats import bonferroni, deflated_sharpe


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
    rows: list
    viable: list


def _nan_to_zero(x) -> float:
    x = float(x)
    return 0.0 if x != x else x  # NaN != NaN


def run_gate(strategy, bars_by_symbol, close, *, horizon=5, min_names=10,
             oos_frac=0.25, rounds=4, icir_floor=0.3, half_life_floor=5,
             alpha=0.05, dsr_threshold=0.95) -> GateResult:
    cutoff = oos_cutoff(close.index, oos_frac)
    close_is = close.loc[close.index < cutoff]

    search = run_search(
        strategy, bars_by_symbol, close_is,
        horizon=horizon, min_names=min_names, max_rounds=rounds,
        gates=Gates(icir_floor=icir_floor, half_life_floor=half_life_floor),
    )
    icirs = [s.icir for s in search.all_scores]
    var_trials = float(pd.Series(icirs).var(ddof=0)) if len(icirs) > 1 else 0.0

    rows = []
    for s in search.survivors:
        ev = evaluate_oos(strategy, s.params, bars_by_symbol, close, cutoff,
                          horizon, min_names)
        n_eff = max(ev["n_obs"] // horizon, 1)
        bonf_p, bonf_thr, bonf_pass = bonferroni(
            ev["oos_icir"], n_eff, search.n_tested, alpha
        )
        skew = _nan_to_zero(ev["oos_ic"].skew())
        kurt = _nan_to_zero(ev["oos_ic"].kurt())
        dsr = deflated_sharpe(
            ev["oos_icir"], n_eff, skew, kurt, search.n_tested, var_trials
        )
        dsr_pass = dsr > dsr_threshold
        viable, reason = verdict(
            s.icir, ev["oos_icir"], ev["oos_half_life"],
            bonf_pass, dsr_pass, half_life_floor,
        )
        rows.append(GateRow(
            s.params, s.icir, ev["oos_icir"], ev["oos_half_life"],
            bonf_p, bonf_thr, bonf_pass, dsr, dsr_pass, viable, reason,
        ))

    return GateResult(strategy, search.n_tested, rows, [r for r in rows if r.viable])
