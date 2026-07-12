"""Out-of-sample evaluation and the strict viability verdict.

evaluate_oos is the only place the locked out-of-sample slice is read: it builds
a config's signal panel over the full history (signals need in-sample warmup),
restricts scoring to dates on/after the cutoff, and recomputes ICIR + decay
there. verdict combines the OOS ICIR-retention and decay checks with the two
multiple-testing corrections into a single pass/fail with a reason.
"""

from __future__ import annotations

from ..factor.ic import ic_decay, ic_series, half_life, icir
from ..factor.signals import signal_panel
from ..strategies import build


def evaluate_oos(strategy, params, bars_by_symbol, close, cutoff, horizon=5, min_names=10) -> dict:
    strat = build(strategy, params)
    panel_full = signal_panel(strat, bars_by_symbol, close.index)
    oos_mask = close.index >= cutoff
    panel_oos = panel_full.loc[oos_mask]
    close_oos = close.loc[oos_mask]
    ic = ic_series(panel_oos, close_oos, horizon, min_names)
    return {
        "oos_icir": icir(ic),
        "oos_half_life": half_life(ic_decay(panel_oos, close_oos, min_names=min_names)),
        "oos_ic": ic,
        "n_obs": len(ic),
    }


def icir_holds(is_icir, oos_icir, retention: float = 0.5) -> bool:
    return is_icir > 0 and oos_icir > 0 and oos_icir >= retention * is_icir


def decay_holds(oos_half_life, floor) -> bool:
    if oos_half_life is None:
        return False
    if isinstance(oos_half_life, str):
        return True
    return oos_half_life >= floor


def verdict(is_icir, oos_icir, oos_half_life, bonf_pass, dsr_pass,
            half_life_floor, retention: float = 0.5):
    if not icir_holds(is_icir, oos_icir, retention):
        return False, "icir_did_not_hold"
    if not decay_holds(oos_half_life, half_life_floor):
        return False, "decay_did_not_hold"
    if not bonf_pass:
        return False, "failed_bonferroni"
    if not dsr_pass:
        return False, "failed_deflated_sharpe"
    return True, "viable"
