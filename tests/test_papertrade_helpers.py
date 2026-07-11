import re
from datetime import datetime, timezone

import pandas as pd

from rhagent.papertrade import CloseFill, HistoricalSource, entry_features, new_run_id


def _bars(closes, opens=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    opens = opens or closes
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_run_id_format_and_determinism():
    now = datetime(2026, 7, 11, 14, 22, 3, tzinfo=timezone.utc)
    rid = new_run_id(now=now, suffix="a1b2c3d4")
    assert rid == "2026-07-11T14-22-03Z-a1b2c3d4"


def test_run_id_random_suffix_matches_format():
    rid = new_run_id()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z-[0-9a-f]{8}", rid)


def test_close_fill_fills_at_close():
    bar = pd.Series({"open": 10.0, "close": 12.5})
    assert CloseFill().fill("AAPL", 1.0, bar) == 12.5


def test_historical_source_reads_cached_csv(tmp_path):
    df = _bars([1.0, 2.0])
    df.to_csv(tmp_path / "AAPL.csv")
    src = HistoricalSource(["AAPL"], "2026-01-01", "2026-01-02", cache_dir=tmp_path)
    out = src.bars()
    assert list(out) == ["AAPL"]
    assert out["AAPL"]["close"].tolist() == [1.0, 2.0]


def test_entry_features_keys_and_values():
    closes = [100.0] * 25
    opens = list(closes)
    opens[-1] = 102.0  # 2% gap up vs prev close 100
    hist = _bars(closes, opens)
    f = entry_features(hist)
    assert set(f) == {"vol20", "gap", "trend5"}
    assert f["vol20"] == 0.0          # flat closes -> zero vol
    assert abs(f["gap"] - 0.02) < 1e-9
    assert f["trend5"] == 0.0         # flat -> no trend


def test_entry_features_trend_sign():
    hist = _bars([100, 100, 100, 100, 100, 101, 102, 103, 104, 105])
    assert entry_features(hist)["trend5"] == 1.0


def test_entry_features_short_history_is_nan_free():
    f = entry_features(_bars([100.0, 101.0]))
    assert all(v == v for v in f.values())  # no NaNs leak into the ledger
