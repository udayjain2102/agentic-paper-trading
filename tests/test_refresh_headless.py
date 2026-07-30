"""Headless refresh batches the universe by 10 and merges each batch into the
cache. The MCP network call is injected so this runs offline."""

from datetime import date

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


def test_yahoo_payload_mapping():
    import json

    # 2026-07-06 / 07 / 08 at 13:30 UTC (9:30 ET); middle bar is a null pad
    body = json.dumps({"chart": {"result": [{
        "timestamp": [1783344600, 1783431000, 1783517400],
        "indicators": {"quote": [{
            "open": [10, None, 10.5], "high": [11, None, 12],
            "low": [9, None, 10], "close": [10.5, None, 11.8],
            "volume": [1000, None, 2000]}]}}]}}).encode()
    urls = []

    def fake_urlopen(url):
        urls.append(url)
        if "BOGUS" in url:
            raise OSError("HTTP 404")
        return body

    payload = refresh._fetch_yahoo(["NVDA", "BOGUS"], "2026-07-01", "2026-07-08",
                                   urlopen=fake_urlopen)
    assert "chart/NVDA?" in urls[0] and "interval=1d" in urls[0]
    results = payload["data"]["results"]
    assert [r["symbol"] for r in results] == ["NVDA"]   # failing ticker skipped
    bars = results[0]["bars"]
    assert len(bars) == 2                               # null-padded bar dropped
    assert bars[0]["begins_at"] == "2026-07-06T00:00:00Z"
    assert float(bars[1]["close_price"]) == 11.8
    assert bars[1]["volume"] == 2000


def test_todays_unsettled_bar_is_dropped_yesterdays_is_kept(tmp_path, capsys):
    # 2026-07-27 incident: a mid-session run cached a still-open bar as if
    # closed, and the forward record scored the previous day against it.
    # A bar dated "today" must never reach the cache; a bar dated "yesterday"
    # (already closed) must merge in as usual. No real clock involved --
    # `today` is injected straight into fetch_and_update, which threads it
    # through to update_cache's guard.
    payload = {"data": {"results": [{"symbol": "AAPL", "bars": [
        {"begins_at": "2026-07-27T00:00:00Z", "open_price": "10", "high_price": "11",
         "low_price": "9", "close_price": "10.5", "volume": 1000},   # yesterday: settled
        {"begins_at": "2026-07-28T00:00:00Z", "open_price": "10.5", "high_price": "10.6",
         "low_price": "10.4", "close_price": "10.55", "volume": 4442695},  # today: mid-session
    ]}]}}

    counts = refresh.fetch_and_update(
        cache_dir=tmp_path, symbols=["AAPL"], today=date(2026, 7, 28),
        fetch_raw=lambda batch, start, end: payload,
    )

    assert counts == {"AAPL": 1}                      # only the settled bar kept
    df = pd.read_csv(tmp_path / "AAPL.csv")
    assert list(df["date"]) == ["2026-07-27"]
    err = capsys.readouterr().err
    assert "!! refresh: dropping unsettled AAPL bar dated 2026-07-28" in err


def test_payload_of_only_unsettled_bars_does_not_crash_the_run(tmp_path):
    # A mid-session fetch whose only bar for a symbol is today's (a newly-added
    # name, or a partial payload) filters down to nothing. rows_to_df([]) raises
    # KeyError 'date', and refresh is the FIRST step in paper_cron.sh under
    # `set -euo pipefail` -- so that crash would kill the run before any tick and
    # throw away days that were already computable. Report 0 bars, keep going,
    # and leave any existing cache untouched.
    payload = {"data": {"results": [{"symbol": "NEW", "bars": [
        {"begins_at": "2026-07-28T00:00:00Z", "open_price": "10", "high_price": "11",
         "low_price": "9", "close_price": "10.5", "volume": 1000},
    ]}]}}

    counts = refresh.fetch_and_update(
        cache_dir=tmp_path, symbols=["NEW"], today=date(2026, 7, 28),
        fetch_raw=lambda batch, start, end: payload,
    )

    assert counts == {"NEW": 0}
    assert not (tmp_path / "NEW.csv").exists()


def test_fetch_defaults_to_yahoo_without_token(tmp_path, monkeypatch):
    # no ROBINHOOD_MCP_TOKEN in the environment -> source "auto" must pick
    # yahoo, never try to open an MCP session
    monkeypatch.delenv("ROBINHOOD_MCP_TOKEN", raising=False)
    called = {}

    def fake_yahoo(batch, start, end, urlopen=None):
        called["batch"] = list(batch)
        return {"data": {"results": []}}

    monkeypatch.setattr(refresh, "_fetch_yahoo", fake_yahoo)
    refresh.fetch_and_update(cache_dir=tmp_path, symbols=["NVDA"])
    assert called["batch"] == ["NVDA"]
