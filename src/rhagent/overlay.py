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
from .evaluate import failure_buckets
from .features import entry_features, flatten_trades


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


def _snap_size(worst_share: float, min_size: float) -> float:
    """Coarse-size the down-size fraction to 0.25 steps so a bucket's loss
    share drifting by a few points doesn't open/close a trade every bar."""
    return max(min_size, round((1.0 - worst_share) / 0.25) * 0.25)


class BucketFilter:
    """Veto/down-size entries whose setup bucket has been the worst loser in
    closed trades so far. Deterministic and inspectable."""

    name = "bucket"

    def __init__(self, veto_share=0.25, veto_wr=0.40, min_n=20, min_size=0.3) -> None:
        self.veto_share = veto_share
        self.veto_wr = veto_wr
        self.min_n = min_n
        self.min_size = min_size

    def _candidate_labels(self, history, side, closed) -> dict:
        """The candidate's bucket in each dimension (vol/gap/side)."""
        f = entry_features(history)
        # vol tercile boundaries from the population of closed trades
        vol_lab = "all"
        vols = closed["feat_vol20"].astype(float)
        if vols.nunique() >= 3:
            lo, hi = np.percentile(vols, [33.333, 66.667])
            v = f["vol20"]
            vol_lab = "low" if v <= lo else ("high" if v > hi else "med")
        gap = f["gap"]
        gap_lab = "flat"
        if gap < -0.005:
            gap_lab = "down"
        elif gap > 0.005:
            gap_lab = "up"
        return {"vol": vol_lab, "gap": gap_lab, "side": side}

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        target = decision.target
        if target == 0.0 or len(closed_trades) < self.min_n:
            return target
        closed_trades = flatten_trades(closed_trades)
        fb = failure_buckets(closed_trades)
        if len(fb) == 0:
            return target
        side = "long" if target > 0 else "short"
        labels = self._candidate_labels(history, side, closed_trades)
        worst_share = 0.0
        for dim, bucket in labels.items():
            row = fb[(fb["dimension"] == dim) & (fb["bucket"] == str(bucket))]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            if r["n_trades"] >= self.min_n and r["loss_share"] >= self.veto_share \
                    and r["win_rate"] <= self.veto_wr:
                return 0.0  # veto: this bucket is bleeding
            worst_share = max(worst_share, float(r["loss_share"]))
        # soft down-size proportional to the worst bucket's loss share, snapped
        # to coarse 0.25 levels so it only moves on a real regime change
        # instead of churning a new trade almost every bar.
        size = _snap_size(worst_share, self.min_size)
        return target * size


def build_overlay(name: str) -> Overlay:
    if name == "none":
        return IdentityOverlay()
    if name == "conviction":
        return ConvictionGate()
    if name == "bucket":
        return BucketFilter()
    raise KeyError(f"unknown overlay {name!r}")
