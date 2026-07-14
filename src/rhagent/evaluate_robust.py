"""Noise-robust ranking of paper-trade runs: fold Sharpe, bootstrap CI, and a
deflated Sharpe that penalizes multiple-testing. numpy/pandas only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import result_from_returns
from .evaluate import load_run

_ANN = 252


def _sharpe(x: np.ndarray) -> float:
    sd = x.std()
    return float(x.mean() / sd * math.sqrt(_ANN)) if sd > 0 else 0.0


def _phi(z: float) -> float:
    """Standard normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def fold_sharpe(net: pd.Series, fold: int = 60, step: int = 30) -> tuple[float, float]:
    x = net.to_numpy(dtype=float)
    srs = [_sharpe(x[i:i + fold]) for i in range(0, max(len(x) - fold + 1, 1), step)]
    srs = [s for s in srs if np.isfinite(s)]
    if not srs:
        return 0.0, 0.0
    return float(np.mean(srs)), float(np.std(srs))


def bootstrap_sharpe_ci(net: pd.Series, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    x = net.to_numpy(dtype=float)
    if len(x) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    srs = np.array([_sharpe(rng.choice(x, size=len(x), replace=True)) for _ in range(n)])
    return float(np.percentile(srs, 2.5)), float(np.percentile(srs, 97.5))


def deflated_sharpe(observed_sr: float, all_srs: list[float], net: pd.Series) -> float:
    """Probabilistic Sharpe vs a benchmark inflated for M trials (Bailey & Lopez
    de Prado). Higher = more likely the Sharpe is real, not multiple-testing luck."""
    x = net.to_numpy(dtype=float)
    T = len(x)
    if T < 3:
        return 0.0
    sd = x.std()
    if sd == 0:
        return 0.0
    z = (x - x.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # non-excess
    M = max(len(all_srs), 1)
    if M <= 1:
        # A single run has no multiple-testing inflation to correct for
        # (1 - 1/M = 0 sends the benchmark to -inf); fall back to plain PSR
        # vs a zero-Sharpe benchmark instead of reporting maximal significance.
        sr_benchmark = 0.0
    else:
        var_sr = float(np.var(all_srs))
        gamma = 0.5772156649  # Euler-Mascheroni
        e_max = (1 - gamma) * _phi_inv(1 - 1.0 / M) + gamma * _phi_inv(1 - 1.0 / (M * math.e))
        sr_benchmark = math.sqrt(var_sr) * e_max
    # daily-scale Sharpe (deflate uses per-observation SR, not annualized)
    sr_daily = observed_sr / math.sqrt(_ANN)
    denom = math.sqrt(max(1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2, 1e-9))
    num = (sr_daily - sr_benchmark / math.sqrt(_ANN)) * math.sqrt(T - 1)
    return float(_phi(num / denom))


def _group_key(row: dict) -> tuple:
    return (row["engine"], tuple(sorted(row["symbols"])))


def _baseline_by_group(rows: list[dict]) -> dict[tuple, float]:
    """Per-(engine, symbols) baseline: the best overlay=='none' point_sharpe in
    that group. Groups without a 'none' run have no baseline entry."""
    baselines: dict[tuple, float] = {}
    for r in rows:
        if r["overlay"] != "none":
            continue
        key = _group_key(r)
        baselines[key] = max(baselines.get(key, -math.inf), r["point_sharpe"])
    return baselines


def robust_table(base_dir: str | Path) -> pd.DataFrame:
    base_dir = Path(base_dir)
    runs = []
    for meta_path in sorted(base_dir.glob("*/run.json")):
        meta, trades, net = load_run(meta_path.parent)
        res = result_from_returns(net)
        runs.append({
            "run_id": meta["run_id"], "engine": meta.get("engine", ""),
            "symbols": meta.get("symbols", []),
            "overlay": meta.get("overlay", "none"),
            "point_sharpe": res.sharpe, "net": net,
        })
    if not runs:
        return pd.DataFrame()
    all_srs = [r["point_sharpe"] for r in runs]
    baselines = _baseline_by_group(runs)
    rows = []
    for r in runs:
        fm, fs = fold_sharpe(r["net"])
        lo, hi = bootstrap_sharpe_ci(r["net"])
        d = deflated_sharpe(r["point_sharpe"], all_srs, r["net"])
        baseline_sr = baselines.get(_group_key(r))
        rows.append({
            "run_id": r["run_id"], "engine": r["engine"], "overlay": r["overlay"],
            "point_sharpe": r["point_sharpe"], "fold_mean": fm, "fold_std": fs,
            "ci_lo": lo, "ci_hi": hi, "deflated": d,
            "beats_baseline": bool(baseline_sr is not None and lo > baseline_sr),
        })
    return pd.DataFrame(rows).sort_values("deflated", ascending=False).reset_index(drop=True)
