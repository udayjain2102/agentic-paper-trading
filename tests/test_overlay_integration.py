"""Regression test: BucketFilter run through the real PaperTrader loop.

PaperTrader.run() passes overlay.adjust(...) a DataFrame built straight from
trade dicts, where features live in a nested `entry_features` dict column
(not flattened `feat_*` columns). BucketFilter (and evaluate.failure_buckets)
expect the flattened schema that evaluate.load_run produces. Unit tests in
test_overlay_bucket.py hand-build already-flattened frames and so never catch
this mismatch; this test drives the real integration seam.
"""
import numpy as np
import pandas as pd

from rhagent.overlay import BucketFilter
from rhagent.papertrade import PaperTrader
from rhagent.engine import StrategyEngine
from rhagent.strategies import build


def _oscillating_frame(periods=150, phase=0.0):
    idx = pd.date_range("2025-01-01", periods=periods, freq="D")
    close = pd.Series(100 + np.sin(np.arange(periods) / 3.0 + phase) * 5, index=idx)
    return pd.DataFrame({"open": close, "close": close}, index=idx)


class _MultiSymbolSource:
    def __init__(self, frames):
        self._frames = frames

    def bars(self):
        return self._frames


def test_bucket_filter_survives_real_papertrade_loop(tmp_path):
    frames = {
        "NVDA": _oscillating_frame(phase=0.0),
        "SPY": _oscillating_frame(phase=1.0),
        "AAPL": _oscillating_frame(phase=2.0),
    }
    trader = PaperTrader(
        engine=StrategyEngine(build("mean_reversion", {})),
        source=_MultiSymbolSource(frames),
        out_dir=str(tmp_path),
        overlay=BucketFilter(min_n=5),
    )
    run_dir = trader.run()
    assert run_dir.exists()
    trades_path = run_dir / "trades.jsonl"
    n_trades = sum(1 for line in trades_path.read_text().splitlines() if line.strip())
    assert n_trades > 5, f"expected enough closed trades to exercise BucketFilter, got {n_trades}"
