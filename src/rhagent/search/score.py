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
