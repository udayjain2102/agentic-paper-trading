import pandas as pd

from rhagent.data import get_bars, rows_to_df


FIXTURE = {
    "AAPL": [
        {"date": "2025-01-03", "open": 1, "high": 2, "low": 1, "close": 191.0, "volume": 10},
        {"date": "2025-01-02", "open": 1, "high": 2, "low": 1, "close": 190.0, "volume": 10},
    ]
}


def test_rows_to_df_sorts_and_indexes_by_date():
    df = rows_to_df(FIXTURE["AAPL"])
    assert df.index.name == "date"
    assert list(df["close"]) == [190.0, 191.0]  # sorted ascending by date


def test_get_bars_fetches_then_caches(tmp_path):
    calls = []

    def fake_fetch(symbols, start, end):
        calls.append(list(symbols))
        return FIXTURE

    out = get_bars(["AAPL"], "2025-01-01", "2025-02-01", fetch=fake_fetch, cache_dir=tmp_path)
    assert out["AAPL"]["close"].iloc[-1] == 191.0
    assert (tmp_path / "AAPL.csv").exists()

    # Second call is served from cache — fetch not invoked again.
    out2 = get_bars(["AAPL"], "2025-01-01", "2025-02-01", fetch=fake_fetch, cache_dir=tmp_path)
    assert out2["AAPL"]["close"].iloc[-1] == 191.0
    assert calls == [["AAPL"]]  # only the first call fetched


def test_get_bars_cache_read_preserves_float_dtype(tmp_path):
    """OHLCV columns read back from a cached CSV must be float, mirroring the
    dtype contract enforced by rows_to_df on the write path.

    Pre-seed a cache CSV with integer-looking values (no decimal points) — the
    kind of file that pandas' dtype inference alone would read back as int64,
    silently violating the module's float-OHLCV contract. This bypasses
    rows_to_df's float cast on write, isolating the read-path guarantee.
    """
    csv_path = tmp_path / "AAPL.csv"
    csv_path.write_text(
        "date,open,high,low,close,volume\n"
        "2025-01-02,190,191,189,190,1000000\n"
        "2025-01-03,191,193,190,192,2000000\n"
    )

    def fake_fetch(symbols, start, end):
        raise AssertionError("fetch should not be called; cache file already exists")

    out = get_bars(["AAPL"], "2025-01-01", "2025-02-01", fetch=fake_fetch, cache_dir=tmp_path)
    df = out["AAPL"]
    for col in ["open", "high", "low", "close", "volume"]:
        assert str(df[col].dtype) == "float64", f"{col} dtype drifted: {df[col].dtype}"
