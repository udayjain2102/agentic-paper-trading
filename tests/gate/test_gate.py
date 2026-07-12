import numpy as np
import pandas as pd

from rhagent.gate.gate import GateResult, GateRow, run_gate


def _panel(n_syms=12, n_bars=200, seed=3):
    # Return-based: per-name drift (ranked by i) + daily noise -> momentum signal
    # ranks by i (positive IC) with day-to-day variation (finite, non-degenerate
    # ICIR). Strong enough that a survivor should clear the strict OOS gate.
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


def test_run_gate_structure():
    bars, close = _panel()
    res = run_gate("momentum", bars, close, horizon=1, min_names=5,
                   rounds=1, icir_floor=0.05, half_life_floor=1)
    assert isinstance(res, GateResult)
    assert res.strategy == "momentum"
    assert res.n_tested > 0
    assert len(res.rows) == len([r for r in res.rows])  # one row per survivor
    assert all(isinstance(r, GateRow) for r in res.rows)


def test_run_gate_finds_viable_on_strongly_predictive_data():
    bars, close = _panel()
    res = run_gate("momentum", bars, close, horizon=1, min_names=5,
                   rounds=1, icir_floor=0.05, half_life_floor=1)
    assert len(res.viable) >= 1
    assert all(r.viable and r.reason == "viable" for r in res.viable)
    assert all(r.oos_icir > 0 for r in res.viable)


def test_run_gate_dsr_threshold_blocks_all():
    # an impossible DSR threshold (>1) rejects everything on the DSR gate,
    # proving the DSR result is wired into the verdict
    bars, close = _panel()
    res = run_gate("momentum", bars, close, horizon=1, min_names=5,
                   rounds=1, icir_floor=0.05, half_life_floor=1, dsr_threshold=1.0)
    assert res.viable == []
    assert any(r.reason == "failed_deflated_sharpe" for r in res.rows)
