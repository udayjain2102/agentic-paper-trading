import numpy as np, pandas as pd
from rhagent.overlay import ConvictionGate
from rhagent.engine import Decision

def _feed(gate, sym, convictions, target=1.0):
    """Feed a sequence of convictions; return the list of adjusted targets."""
    out = []
    for c in convictions:
        d = Decision(target=target, reason="x", conviction=c)
        out.append(gate.adjust(sym, pd.DataFrame({"close": [1]}), d, pd.DataFrame()))
    return out

def test_low_conviction_vetoed_high_passes():
    gate = ConvictionGate(pctile=0.60, window=40)
    # 40 small |conviction| then measure a small vs a large one
    outs = _feed(gate, "NVDA", [0.1] * 40)
    assert all(o in (0.0, 1.0) for o in outs)  # cold start passes, then gating begins
    small = gate.adjust("NVDA", pd.DataFrame({"close": [1]}),
                        Decision(target=1.0, reason="x", conviction=0.1), pd.DataFrame())
    big = gate.adjust("NVDA", pd.DataFrame({"close": [1]}),
                      Decision(target=1.0, reason="x", conviction=5.0), pd.DataFrame())
    assert small == 0.0   # below the 60th pctile of past |conviction|
    assert big == 1.0     # well above threshold

def test_none_conviction_passes_through():
    gate = ConvictionGate()
    d = Decision(target=-1.0, reason="x", conviction=None)
    assert gate.adjust("NVDA", pd.DataFrame({"close": [1]}), d, pd.DataFrame()) == -1.0

def test_nan_conviction_passes_through():
    gate = ConvictionGate()
    d = Decision(target=1.0, reason="x", conviction=float("nan"))
    assert gate.adjust("NVDA", pd.DataFrame({"close": [1]}), d, pd.DataFrame()) == 1.0

def test_cold_start_passes_before_window_fills():
    gate = ConvictionGate(window=30)
    outs = _feed(gate, "NVDA", [0.1] * 10)  # fewer than window
    assert outs == [1.0] * 10  # not enough history to gate yet
