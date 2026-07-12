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
