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
