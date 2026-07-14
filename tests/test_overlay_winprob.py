import numpy as np, pandas as pd
from rhagent.overlay import WinProbGate
from rhagent.engine import Decision

def _closed(vols, gaps, trends, wins, symbol="NVDA"):
    n = len(vols)
    return pd.DataFrame({
        "symbol": [symbol]*n, "side": ["long"]*n,
        "feat_vol20": vols, "feat_gap": gaps, "feat_trend5": trends,
        "holding_bars": [3]*n,
        "outcome": ["win" if w else "loss" for w in wins],
        "pnl_abs": [100.0 if w else -100.0 for w in wins],
    })

def _hist(vol=0.02, gap=0.0):
    # entry_features' gap == open[-1]/close[-2]-1 == `gap` exactly (a linspace-to-
    # target close makes the actual last-bar gap tiny, so build it directly).
    # Also give the history's realized vol the same magnitude as `gap`, since in
    # this synthetic training set feat_vol20 is (by construction) set equal to
    # feat_gap's discriminative signal -- matching that keeps the candidate's
    # feature row consistent with what the model was fit on.
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    n = 30
    steps = np.where(np.arange(n - 1) % 2 == 0, gap, -gap)
    rets = np.concatenate([[0.0], steps])
    close = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    opens = close.copy()
    opens.iloc[-1] = close.iloc[-2] * (1 + gap)
    return pd.DataFrame({"open": opens, "close": close}, index=idx)

def test_cold_start_passes():
    g = WinProbGate(min_train=50)
    d = Decision(target=1.0, reason="x")
    assert g.adjust("NVDA", _hist(), d, _closed([0.02]*10, [0.0]*10, [1.0]*10, [True]*10)) == 1.0

def test_low_winprob_setup_vetoed_high_passes():
    # Losses cluster at high gap; wins at low gap. Model should veto a high-gap candidate.
    rng = np.random.default_rng(0)
    gaps = np.concatenate([rng.uniform(-0.001, 0.001, 60), rng.uniform(0.02, 0.03, 60)])
    wins = np.array([True]*60 + [False]*60)   # low-gap win, high-gap lose
    vols = [0.02]*120; trends = [0.0]*120
    closed = _closed(list(gaps), list(gaps*0+0.0), trends, list(wins))
    # note: gap is the discriminative feature -> put it in feat_gap
    closed["feat_gap"] = list(gaps)
    g = WinProbGate(thresh=0.5, min_train=50, refit_every=1)
    good = g.adjust("NVDA", _hist(gap=0.0), Decision(target=1.0, reason="x"), closed)     # low gap
    bad = g.adjust("NVDA", _hist(gap=0.025), Decision(target=1.0, reason="x"), closed)    # high gap
    assert good == 1.0
    assert bad == 0.0

def test_zero_target_passthrough():
    g = WinProbGate()
    assert g.adjust("NVDA", _hist(), Decision(target=0.0, reason="x"), _closed([0.02]*60,[0.0]*60,[1.0]*60,[True]*30+[False]*30)) == 0.0
