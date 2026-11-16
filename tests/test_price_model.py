"""Tests for rhagent.price_model — the price-only model engine.

The load-bearing property is lookahead-freedom: a decision for date d must be
identical whether computed from the full history or from history truncated at
d. Everything else (feature math, logistic fit, selection) is secondary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhagent.price_model import (FEATURES, _fit_logistic, _predict_proba,
                                 build_panel, positions_on,
                                 walkforward_positions)


def _bars(n: int = 300, seed: int = 0, drift: float = 0.0005) -> dict:
    """Synthetic universe: `drift` daily mean return plus noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    out = {}
    for i, sym in enumerate(["AAA", "BBB", "CCC"]):
        rets = rng.normal(drift + 0.0002 * i, 0.01, n)
        close = 100 * np.exp(np.cumsum(rets))
        out[sym] = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1_000_000.0}, index=dates)
    return out


def test_build_panel_labels_and_features():
    bars = _bars(60)
    panel = build_panel(bars, cost_bps=7.0)
    # 60 bars - 20d feature warmup = 40 rows/symbol; the last bar is kept
    # (features resolved, label pending) so the live decision date is scorable
    assert len(panel) == 3 * 40
    assert set(FEATURES) <= set(panel.columns)
    # y is exactly "next close-to-close return beats 7bps"; recompute fwd
    # independently: fwd_ret is close[t+1]/close[t]-1. The last bar has no
    # resolved label, so check only rows with fwd_ret.
    a = panel[panel.symbol == "AAA"].sort_values("date")
    av = a[a["fwd_ret"].notna()]
    close = bars["AAA"]["close"]
    fwd = close.shift(-1) / close - 1.0
    assert np.allclose(av["fwd_ret"].to_numpy(),
                       fwd.reindex(av["date"]).to_numpy())
    assert ((av["fwd_ret"] > 7 / 1e4).astype(float) == av["y"]).all()
    assert a["y"].isna().sum() == 1  # only the unresolved last bar
    # label_date is the next bar's date
    nxt = pd.Series(bars["AAA"].index[1:], index=bars["AAA"].index[:-1])
    assert (av["label_date"] == av["date"].map(nxt)).all()


def test_build_panel_horizon5_label():
    bars = _bars(60)
    panel = build_panel(bars, cost_bps=7.0, horizon=5)
    assert len(panel) == 3 * 40
    a = panel[panel.symbol == "AAA"].sort_values("date")
    close = bars["AAA"]["close"]
    fwd = close.shift(-5) / close - 1.0
    av = a[a["fwd_ret"].notna()]
    assert np.allclose(av["fwd_ret"].to_numpy(),
                       fwd.reindex(av["date"]).to_numpy())
    # last 5 bars have unresolved labels -> NaN fwd_ret and NaN y
    assert a["fwd_ret"].isna().sum() == 5
    assert a["y"].isna().sum() == 5
    # label_date resolves 5 bars later
    nxt = pd.Series(list(bars["AAA"].index[5:]),
                    index=bars["AAA"].index[:-5])
    assert (av["label_date"] == av["date"].map(nxt)).all()


def test_positions_on_is_truncation_invariant():
    """Deciding date d from full bars must equal deciding it from bars cut
    at d — any dependence on later data is lookahead."""
    bars = _bars(300)
    d = bars["AAA"].index[250]
    full = positions_on(bars, [d], cost_bps=7.0, threshold=0.4)
    trunc = {s: df.loc[:d] for s, df in bars.items()}
    part = positions_on(trunc, [d], cost_bps=7.0, threshold=0.4)
    assert full[d].keys() == part[d].keys()
    for sym in full[d]:
        assert full[d][sym][0] == part[d][sym][0]


def test_positions_on_only_uses_resolved_labels():
    """A huge up-move AFTER the decision date must not change the decision."""
    bars = _bars(300)
    d = bars["AAA"].index[200]
    base = positions_on(bars, [d], threshold=0.4)
    shocked = {s: df.copy() for s, df in bars.items()}
    for s in shocked:
        shocked[s].iloc[210:, shocked[s].columns.get_loc("close")] *= 3.0
    after = positions_on(shocked, [d], threshold=0.4)
    assert base[d] == after[d]


def test_logistic_recovers_a_real_signal():
    rng = np.random.default_rng(1)
    n = 5000
    X = rng.normal(size=(n, 3))
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    w, mu, sd = _fit_logistic(X, y)
    p = _predict_proba(X, w, mu, sd)
    # predictions should rank-correlate with the true logit strongly
    assert np.corrcoef(p, logit)[0, 1] > 0.9


def test_walkforward_respects_min_train_and_is_flat_early():
    bars = _bars(400)
    pos = walkforward_positions(bars, min_train_rows=10**9)
    assert all((s == 0).all() for s in pos.values())

    pos = walkforward_positions(bars, min_train_rows=100,
                                threshold=0.4, fold="M")
    held = pd.DataFrame(pos)
    assert (held.sum(axis=1) > 0).any()  # eventually trades
    assert held.sum(axis=1).max() <= len(bars)
