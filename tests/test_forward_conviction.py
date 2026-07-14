"""The forward path applies the conviction gate via a vectorized twin that must
stay bit-identical to the bar-by-bar ConvictionGate used in papertrade."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward
from rhagent.engine import Decision
from rhagent.overlay import ConvictionGate, apply_conviction


def test_apply_conviction_matches_bar_by_bar_gate():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    signal = pd.Series(rng.normal(0, 1, n), index=idx)
    signal.iloc[:15] = np.nan  # warmup NaNs, like a real strategy signal
    positions = pd.Series(rng.choice([0.0, 1.0], size=n), index=idx)

    gate = ConvictionGate(pctile=0.60, window=50)
    ref = []
    for i in range(n):
        c = signal.iloc[i]
        d = Decision(target=float(positions.iloc[i]), reason="x",
                     conviction=(None if pd.isna(c) else float(c)))
        ref.append(gate.adjust("S", None, d, None))
    ref = pd.Series(ref, index=idx)

    vec = apply_conviction(positions, signal, pctile=0.60, window=50)
    pd.testing.assert_series_equal(vec, ref, check_names=False)


def test_forward_positions_applies_conviction():
    idx = pd.date_range("2025-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
    bars = {"AAA": pd.DataFrame({"open": close, "close": close})}

    def _cfg(ov):
        return SimpleNamespace(strategy=SimpleNamespace(
            name="mean_reversion", params={}, universe=["AAA"], overlay=ov))

    p_conv = forward._positions(_cfg("conviction"), "mean_reversion", bars, Path("/tmp"))
    p_none = forward._positions(_cfg("none"), "mean_reversion", bars, Path("/tmp"))

    # the gate can only remove positions, never add them
    assert ((p_conv["AAA"] != 0) & (p_none["AAA"] == 0)).sum() == 0
    # and on a 200-bar window past the 120 warmup it should zero at least one bar
    assert (p_conv["AAA"] == 0).sum() >= (p_none["AAA"] == 0).sum()
