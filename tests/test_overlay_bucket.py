import numpy as np, pandas as pd
from rhagent.overlay import BucketFilter
from rhagent.engine import Decision

def _closed(side, feat_vol20, feat_gap, pnl, n):
    """Build n identical closed trades in one bucket."""
    return pd.DataFrame({
        "symbol": ["NVDA"] * n, "side": [side] * n,
        "feat_vol20": [feat_vol20] * n, "feat_gap": [feat_gap] * n,
        "feat_trend5": [0.0] * n, "holding_bars": [3] * n,
        "pnl_abs": [pnl] * n, "outcome": ["loss" if pnl < 0 else "win"] * n,
    })

def _hist(vol20=0.02, gap=0.0):
    # a history whose entry_features produce roughly vol20/gap is hard to force;
    # BucketFilter recomputes features from history, so build a matching history.
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    close = pd.Series(np.linspace(100, 100 * (1 + gap), 30), index=idx)
    return pd.DataFrame({"open": close, "close": close}, index=idx)

def test_bleeding_short_bucket_vetoed():
    # 40 losing shorts => side=short bucket has 100% loss share, 0% win rate
    closed = _closed("short", 0.02, 0.0, pnl=-100.0, n=40)
    bf = BucketFilter(min_n=20)
    d = Decision(target=-1.0, reason="x", conviction=None)  # candidate is a short
    out = bf.adjust("NVDA", _hist(), d, closed)
    assert out == 0.0

def test_clean_bucket_passes():
    closed = _closed("long", 0.02, 0.0, pnl=+100.0, n=40)  # all winners
    bf = BucketFilter(min_n=20)
    d = Decision(target=1.0, reason="x", conviction=None)
    out = bf.adjust("NVDA", _hist(), d, closed)
    assert out == 1.0

def test_cold_start_passes():
    bf = BucketFilter(min_n=20)
    d = Decision(target=1.0, reason="x", conviction=None)
    assert bf.adjust("NVDA", _hist(), d, pd.DataFrame()) == 1.0
