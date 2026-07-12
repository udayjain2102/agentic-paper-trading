import numpy as np
import pandas as pd

from rhagent.gate.oos import decay_holds, evaluate_oos, icir_holds, verdict


def _panel(n_syms=12, n_bars=200, seed=3):
    # Return-based construction: each symbol has a per-name drift (ranked by i)
    # plus daily noise. The drift makes the momentum signal rank by i (positive
    # IC); the noise guarantees the daily IC varies (no degenerate std=0 -> ICIR
    # is finite and positive, not the perfect-rank tie trap).
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="D", name="date")
    rng = np.random.default_rng(seed)
    bars, closes = {}, {}
    for i in range(n_syms):
        rets = i * 0.003 + rng.normal(0, 0.006, size=n_bars)
        c = 100.0 * np.cumprod(1 + rets)
        bars[f"S{i}"] = pd.DataFrame(
            {"open": c, "high": c, "low": c, "close": c, "volume": [1e6] * n_bars},
            index=idx,
        )
        closes[f"S{i}"] = c
    return bars, pd.DataFrame(closes, index=idx)


def test_evaluate_oos_predictive_positive_icir():
    bars, close = _panel()
    cutoff = close.index[int(len(close) * 0.75)]
    ev = evaluate_oos("momentum", {"lookback": 40}, bars, close, cutoff,
                      horizon=1, min_names=5)
    assert ev["n_obs"] > 0
    assert ev["oos_icir"] > 0
    assert isinstance(ev["oos_ic"], pd.Series)


def test_evaluate_oos_uses_only_oos_dates():
    bars, close = _panel()
    cutoff = close.index[int(len(close) * 0.75)]
    ev = evaluate_oos("momentum", {"lookback": 40}, bars, close, cutoff,
                      horizon=1, min_names=5)
    # every scored date is on/after the cutoff
    assert (ev["oos_ic"].index >= cutoff).all()


def test_icir_holds_boundaries():
    assert icir_holds(0.4, 0.25) is True         # 0.25 >= 0.5*0.4
    assert icir_holds(0.4, 0.15) is False        # below 50% retention
    assert icir_holds(0.4, -0.3) is False        # OOS negative
    assert icir_holds(-0.4, -0.3) is False        # in-sample not positive


def test_decay_holds_rules():
    assert decay_holds(None, 5) is False
    assert decay_holds(3, 5) is False
    assert decay_holds(20, 5) is True
    assert decay_holds(">50", 5) is True


def test_verdict_reasons_in_order():
    # ICIR fails first
    assert verdict(0.4, 0.1, 20, True, True, 5) == (False, "icir_did_not_hold")
    # ICIR holds, decay fails
    assert verdict(0.4, 0.3, 2, True, True, 5) == (False, "decay_did_not_hold")
    # ICIR + decay hold, bonferroni fails
    assert verdict(0.4, 0.3, 20, False, True, 5) == (False, "failed_bonferroni")
    # bonferroni passes, DSR fails
    assert verdict(0.4, 0.3, 20, True, False, 5) == (False, "failed_deflated_sharpe")
    # all pass
    assert verdict(0.4, 0.3, 20, True, True, 5) == (True, "viable")
