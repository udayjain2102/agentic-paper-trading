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


def _predict_logit(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.clip(X @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logit(X: np.ndarray, y: np.ndarray, iters: int = 25, l2: float = 1.0) -> np.ndarray:
    """Ridge-regularized logistic regression via IRLS. X includes a bias column.
    l2 on the diagonal keeps the Hessian invertible under separable/degenerate
    data (and when y is all one class)."""
    n, k = X.shape
    beta = np.zeros(k)
    ridge = l2 * np.eye(k)
    for _ in range(iters):
        p = _predict_logit(beta, X)
        w = np.clip(p * (1.0 - p), 1e-6, None)      # IRLS weights, floored
        # Hessian = X^T W X + ridge ; gradient = X^T (y - p) - l2*beta
        H = X.T @ (w[:, None] * X) + ridge
        g = X.T @ (y - p) - l2 * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


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


class WinProbGate:
    """Veto entries whose predicted win-probability is below `thresh`. Fits a
    ridge logistic model on closed trades' entry features (+ a smoothed
    per-symbol win-rate encoding); walk-forward and refit on a cadence."""

    name = "winprob"
    _FEATS = ["feat_vol20", "feat_gap", "feat_trend5"]

    def __init__(self, thresh=0.52, refit_every=20, min_train=50, l2=1.0) -> None:
        self.thresh = thresh
        self.refit_every = refit_every
        self.min_train = min_train
        self.l2 = l2
        self._beta = None
        self._sym_wr: dict[str, float] = {}
        self._prior = 0.5
        self._calls = 0

    def _design(self, feats: pd.DataFrame, sym_wr: np.ndarray) -> np.ndarray:
        base = feats[self._FEATS].to_numpy(dtype=float)
        bias = np.ones((len(feats), 1))
        return np.column_stack([bias, base, sym_wr.reshape(-1, 1)])

    def _refit(self, closed: pd.DataFrame) -> None:
        df = flatten_trades(closed)
        y = (df["outcome"].to_numpy() == "win").astype(float)
        self._prior = float(y.mean()) if len(y) else 0.5
        # smoothed target-mean encoding per symbol
        a = 5.0
        self._sym_wr = {}
        for sym, grp in df.groupby("symbol"):
            w = (grp["outcome"] == "win").sum()
            self._sym_wr[sym] = float((w + a * self._prior) / (len(grp) + a))
        sym_wr = df["symbol"].map(lambda s: self._sym_wr.get(s, self._prior)).to_numpy(float)
        X = self._design(df, sym_wr)
        self._beta = _fit_logit(X, y, l2=self.l2)

    def adjust(self, symbol, history, decision, closed_trades) -> float:
        target = decision.target
        if target == 0.0 or len(closed_trades) < self.min_train:
            return target
        if self._beta is None or self._calls % self.refit_every == 0:
            self._refit(closed_trades)
        self._calls += 1
        f = entry_features(history)
        row = pd.DataFrame([{ "feat_vol20": f["vol20"], "feat_gap": f["gap"],
                              "feat_trend5": f["trend5"] }])
        sym_wr = np.array([self._sym_wr.get(symbol, self._prior)])
        p = float(_predict_logit(self._beta, self._design(row, sym_wr))[0])
        return target if p >= self.thresh else 0.0


def build_overlay(name: str) -> Overlay:
    if name == "none":
        return IdentityOverlay()
    if name == "conviction":
        return ConvictionGate()
    if name == "bucket":
        return BucketFilter()
    if name == "winprob":
        return WinProbGate()
    raise KeyError(f"unknown overlay {name!r}")
