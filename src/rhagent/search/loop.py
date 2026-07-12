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


def apply_gates(scores: list[ConfigScore], grids, gates: Gates, scored_by_key=None):
    by_key = dict(scored_by_key) if scored_by_key is not None else {}
    for s in scores:
        by_key[config_key(s.params)] = s
    survivors, rejected = [], []
    for s in scores:
        fail = first_failing_gate(s, by_key, grids, gates)
        if fail is None:
            survivors.append(s)
        else:
            rejected.append((s.params, fail))
    survivors.sort(key=lambda s: s.icir, reverse=True)
    return survivors, rejected


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
    all_scores: list


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
    scored_all: dict = {}
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
            sc = scorer(c)
            scores.append(sc)
            scored_all[config_key(c)] = sc
        survivors, rejected = apply_gates(scores, grids, gates, scored_by_key=scored_all)
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
    return SearchResult(strategy, ranked, rounds, len(seen), best,
                        list(scored_all.values()))
