"""Statistics for the out-of-sample gate — pure, no scipy.

A normal CDF (via math.erfc) and inverse CDF (Acklam's rational approximation),
the Bonferroni-adjusted p-value of an information ratio, and the Deflated Sharpe
Ratio (Bailey & Lopez de Prado) which asks: given that N configs were tried,
how probable is it that an ICIR this high is real rather than the luckiest draw.
"""

from __future__ import annotations

import math

_EULER_GAMMA = 0.5772156649015329

# Acklam's inverse-normal-CDF coefficients.
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
               ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q / \
               (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
            ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)


def bonferroni(icir: float, n_eff: int, n_tested: int, alpha: float = 0.05):
    t = abs(icir) * math.sqrt(n_eff)
    p_value = 2.0 * (1.0 - norm_cdf(t))
    threshold = alpha / max(n_tested, 1)
    return p_value, threshold, p_value < threshold


def deflated_sharpe(sr, n_eff, skew, kurt_excess, n_trials, var_trials):
    if n_eff <= 1:
        return 0.0
    if n_trials >= 2 and var_trials > 0:
        sr0 = math.sqrt(var_trials) * (
            (1 - _EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
            + _EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e))
        )
    else:
        sr0 = 0.0
    gamma4 = kurt_excess + 3.0
    denom = 1.0 - skew * sr + (gamma4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr0) * math.sqrt(n_eff - 1) / math.sqrt(denom)
    return norm_cdf(z)
