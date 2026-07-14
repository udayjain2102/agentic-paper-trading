"""Headless refresh batches the universe by 10 and merges each batch into the
cache. The MCP network call is injected so this runs offline."""

import pandas as pd

from rhagent import refresh


def _payload(symbols, close=10.0):
    return {"data": {"results": [
        {"symbol": s, "bars": [
            {"begins_at": "2026-07-06T00:00:00Z", "open_price": "10", "high_price": "11",
             "low_price": "9", "close_price": str(close), "volume": 1000},
            {"begins_at": "2026-07-07T00:00:00Z", "open_price": "10", "high_price": "11",
             "low_price": "9", "close_price": str(close + 1), "volume": 1200},
        ]} for s in symbols]}}


def test_batches_by_ten_and_writes_all(tmp_path):
    symbols = [f"S{i}" for i in range(23)]           # 23 -> batches of 10,10,3
    calls = []

    def fake_fetch(batch, start, end):
        calls.append(list(batch))
        return _payload(batch)

    counts = refresh.fetch_and_update(cache_dir=tmp_path, symbols=symbols,
                                      fetch_raw=fake_fetch)

    assert [len(c) for c in calls] == [10, 10, 3]     # per-call cap respected
    assert set(counts) == set(symbols)                # every symbol written
    assert all(n == 2 for n in counts.values())       # both bars kept
    assert (tmp_path / "S0.csv").exists()
    df = pd.read_csv(tmp_path / "S22.csv")
    assert len(df) == 2
