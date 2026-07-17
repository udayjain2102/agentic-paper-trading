"""Decision overlays: a walk-forward layer between a strategy's raw target and
the position actually taken. Each overlay sees only trades that closed on prior
bars (`closed_trades`), so no overlay can peek at the future.

    final_target = overlay.adjust(symbol, history, decision, closed_trades)

Return semantics: 0.0 vetoes the trade, a fraction downsizes it, and a value
equal to decision.target passes it through unchanged.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Protocol

import numpy as np
import pandas as pd

from .engine import Decision


class Overlay(Protocol):
    name: str
    def adjust(
        self,
        symbol: str,
        history: pd.DataFrame,
        decision: Decision,
        closed_trades: pd.DataFrame,
    ) -> float: ...


class IdentityOverlay:
    name = "none"

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        return decision.target


class ConvictionGate:
    """Veto entries whose |conviction| is below a rolling percentile of the
    symbol's own past |conviction|. Stateful and walk-forward: only convictions
    from bars already seen inform the threshold."""

    name = "conviction"

    def __init__(self, pctile: float = 0.60, window: int = 120) -> None:
        self.pctile = pctile
        self.window = window
        self._hist: dict[str, list[float]] = defaultdict(list)

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        c = decision.conviction
        if c is None or (isinstance(c, float) and math.isnan(c)):
            return decision.target
        past = self._hist[symbol]
        # threshold from history BEFORE recording today's value (no self-inclusion)
        if len(past) < self.window:
            result = decision.target  # cold start: not enough history to gate
        else:
            thresh = float(np.percentile(np.abs(past[-self.window:]), self.pctile * 100))
            result = decision.target if abs(c) > thresh else 0.0
        past.append(abs(c))
        return result


def apply_conviction(positions: pd.Series, signal: pd.Series,
                     pctile: float = 0.60, window: int = 120) -> pd.Series:
    """Vectorized twin of ConvictionGate for the forward path.

    Zeroes out any position bar whose ``|signal|`` does not strictly exceed the
    trailing-``window`` ``pctile`` percentile of prior ``|signal|`` values. This
    reproduces the bar-by-bar gate exactly (same rule, prior-bar-only threshold),
    but on a whole position/signal series at once — the shape forward.py needs.
    Cold-start bars (fewer than ``window`` prior non-NaN signals) and NaN-signal
    bars pass through unchanged, matching the gate's early-return behavior.
    """
    a = signal.abs()
    valid = a.dropna()
    # threshold from the prior `window` non-NaN values only (shift(1) excludes today)
    thr = valid.shift(1).rolling(window).quantile(pctile).reindex(a.index)
    keep = a > thr                         # strict, matches gate's `abs(c) > thresh`
    keep = keep.where(thr.notna(), other=True)   # cold start -> keep
    keep = keep.where(a.notna(), other=True)     # NaN signal -> pass through
    return positions.where(keep, other=0.0)


# BucketFilter and WinProbGate (loss-bucket veto, logit win-probability gate)
# lost the 2026-07 bake-off to ConvictionGate and were removed — see
# md/AUDIT-2026-07-17.md; they live in git history if ever needed again.
def build_overlay(name: str) -> Overlay:
    if name == "none":
        return IdentityOverlay()
    if name == "conviction":
        return ConvictionGate()
    raise KeyError(f"unknown overlay {name!r}")
