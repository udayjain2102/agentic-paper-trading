"""Refresh the daily-bar CSV cache with fresh MCP historicals.

`get_bars` is cache-first and never refetches an existing CSV, so a live loop
must update the cache itself. This takes the raw `get_equity_historicals`
payload (the dict the Robinhood MCP returns), merges the new bars into each
`data/<SYM>.csv`, dedupes by date, and drops degenerate snapshot rows
(volume==0, the "current price" placeholder the last populate left behind).

Two ways in:
  * Interactive: Claude fetches via its MCP session and pipes the raw payload
    here on stdin (the Mon-Fri hands-on loop).
  * Headless: ``--fetch`` pulls the whole config universe itself, for an
    unattended cron on an always-on box. Source is the MCP when
    ROBINHOOD_MCP_URL/TOKEN are set, otherwise Yahoo (keyless) — the MCP
    only ever authenticates inside an interactive Claude session, so Yahoo is
    what makes a truly unattended run possible.

Usage:
    ... | PYTHONPATH=src python -m rhagent.refresh --cache-dir data   # stdin
    PYTHONPATH=src python -m rhagent.refresh --fetch --cache-dir data # headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import _normalize, _read_csv, rows_to_df


def update_cache(payload: dict, cache_dir: str | Path = "data", today=None) -> dict[str, int]:
    """Merge MCP historicals `payload` into the CSV cache. Returns {symbol: n_rows}.

    A bar with volume==0 is treated as a placeholder snapshot and dropped, so an
    intraday snapshot never overwrites a real settled bar for the same date.

    A bar dated `today` is dropped too, for a costlier reason than volume==0:
    the 2026-07-27 incident. A cron run landed mid-session (14:20 UTC, three
    hours late) and cached AAPL's still-open 07-27 bar (volume 4.4M) as if
    closed; the *next* run's settled 07-27 bar (volume 49.6M, an 11x gap) then
    silently overwrote it, but by then forward.py had already scored 07-24
    against the half-finished price and baked a wrong return into the
    permanent record. `--days 10` means the cache itself self-heals on the
    very next run once the session closes -- the damage isn't a bad cache
    entry, it's whatever day already got SCORED against one. So don't
    conclude from a healthy-looking cache that this guard is redundant.
    `today` defaults to a UTC clock, not local: comparing a US-session bar's
    date against UTC is safe (at 01:00 UTC on the 30th, New York is still the
    29th evening, so the settled 07-29 bar, dated < 07-30, survives), and
    GitHub Actions runners are UTC anyway -- but an explicit UTC default
    keeps this correct even if this is ever invoked from a non-UTC box (e.g.
    a laptop cron), rather than relying on the caller's local clock. On the
    normal pre-open run today's bar doesn't exist in the payload yet, so this
    is a no-op.
    """
    from datetime import datetime, timezone

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if today is None:
        today = datetime.now(timezone.utc).date()
    today_str = today.isoformat() if hasattr(today, "isoformat") else str(today)

    results = payload.get("data", payload).get("results", []) or []
    symbols = [r["symbol"] for r in results if r.get("symbol")]
    fresh = _normalize(payload, symbols)

    out: dict[str, int] = {}
    for sym, rows in fresh.items():
        settled = []
        for r in rows:
            if r.get("volume", 0) == 0:
                continue
            if r["date"] >= today_str:
                print(
                    f"!! refresh: dropping unsettled {sym} bar dated {r['date']} "
                    f"(today={today_str} UTC) -- mid-session fetch, not a closed bar",
                    file=sys.stderr,
                )
                continue
            settled.append(r)
        if not settled:
            # Every row filtered out (all volume==0, or a payload whose only bar
            # is today's -- a mid-session run on a newly-added symbol). Leave the
            # existing cache alone rather than calling rows_to_df([]), which
            # raises KeyError 'date' on an empty frame. That crash would be
            # expensive out of proportion to the cause: refresh is the FIRST step
            # in paper_cron.sh under `set -euo pipefail`, so it would kill the
            # run before any tick, throwing away days that were computable.
            out[sym] = 0
            continue
        new = rows_to_df(settled)
        path = cache_dir / f"{sym}.csv"
        if path.exists():
            old = _read_csv(path)
            old = old[old["volume"] != 0]  # scrub any prior degenerate rows
            merged = pd.concat([old, new])
        else:
            merged = new
        # keep the last row for each date (fresh wins), sorted
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(path)
        out[sym] = len(merged)
    return out


def _fetch_yahoo(symbols, start, end, urlopen=None) -> dict:
    """Keyless daily bars from Yahoo's v8 chart API, in the MCP payload shape.

    Split-adjusted OHLCV (same adjustment the MCP path requests). One request
    per symbol; a symbol Yahoo can't serve is skipped with a warning. `urlopen`
    is injectable so the payload mapping can be tested offline.
    """
    import urllib.request
    from datetime import datetime, timezone

    def _default_urlopen(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(req, timeout=30).read()

    urlopen = urlopen or _default_urlopen
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()) + 86400
    results = []
    for sym in symbols:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
               f"?period1={p1}&period2={p2}&interval=1d&events=split")
        try:
            res = json.loads(urlopen(url))["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
        except Exception as e:  # feed is a trust boundary: skip, don't crash the tick
            print(f"yahoo: no data for {sym} ({e}), skipped", file=sys.stderr)
            continue
        bars = []
        for i, t in enumerate(ts):
            if q["close"][i] is None:  # yahoo pads holidays/halts with nulls
                continue
            day = datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()
            bars.append({"begins_at": f"{day}T00:00:00Z",
                         "open_price": q["open"][i], "high_price": q["high"][i],
                         "low_price": q["low"][i], "close_price": q["close"][i],
                         "volume": q["volume"][i] or 0})
        results.append({"symbol": sym, "bars": bars})
    return {"data": {"results": results}}


def _fetch_raw(session, symbols, start, end) -> dict:
    """One MCP get_equity_historicals call -> its raw structured payload."""
    import anyio

    from .broker import _structured

    result = anyio.from_thread.run(
        session.call_tool,
        "get_equity_historicals",
        {
            "symbols": list(symbols),
            "start_time": f"{start}T00:00:00Z",
            "end_time": f"{end}T00:00:00Z",
            "interval": "day",
            "adjustment_type": "split",
        },
    )
    return _structured(result)


def fetch_and_update(cache_dir="data", symbols=None, days=10, today=None,
                     fetch_raw=None, source="auto") -> dict[str, int]:
    """Headless refresh: fetch recent bars for the universe and merge into the
    cache. Batches by 10 (the MCP per-call symbol cap). Returns {symbol: n_rows}.

    `source`: "mcp" (needs ROBINHOOD_MCP_URL/TOKEN), "yahoo" (keyless), or
    "auto" — mcp when a token is configured, yahoo otherwise. This is what lets
    the daily tick run on a box with no interactive Claude session.

    `fetch_raw(batch, start, end) -> raw payload` is injectable so the batching
    and merge can be exercised without a live MCP (the network call is the only
    part that can't run offline).

    `today` defaults to UTC (see update_cache's docstring for why) and is
    reused as the update_cache unsettled-bar cutoff below, so injecting it
    here also drives the guard in tests without touching the real clock.
    """
    import contextlib
    from datetime import datetime, timedelta, timezone

    from .config import load

    cfg = load()
    symbols = list(symbols or cfg.strategy.universe)
    today = today or datetime.now(timezone.utc).date()
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()

    if fetch_raw is None and source == "auto":
        source = "mcp" if cfg.mcp_token else "yahoo"
    if fetch_raw is None and source == "yahoo":
        fetch_raw = _fetch_yahoo
    if fetch_raw is None:
        from .mcp_session import mcp_session
        session_cm = mcp_session(cfg.mcp_url, cfg.mcp_token)
    else:
        session_cm = contextlib.nullcontext(None)

    counts: dict[str, int] = {}
    with session_cm as session:
        get = fetch_raw or (lambda batch, s, e: _fetch_raw(session, batch, s, e))
        for i in range(0, len(symbols), 10):
            counts.update(update_cache(get(symbols[i:i + 10], start, end), cache_dir, today=today))
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.refresh")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--payload", help="JSON file; omit to read stdin")
    p.add_argument("--fetch", action="store_true",
                   help="fetch the config universe headlessly over the MCP "
                        "(needs ROBINHOOD_MCP_URL/TOKEN) instead of reading a payload")
    p.add_argument("--days", type=int, default=10, help="--fetch lookback window")
    p.add_argument("--source", choices=["auto", "mcp", "yahoo"], default="auto",
                   help="--fetch data source (auto: mcp if a token is set, else yahoo)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    if args.fetch:
        counts = fetch_and_update(args.cache_dir, days=args.days, source=args.source)
    else:
        raw = Path(args.payload).read_text() if args.payload else sys.stdin.read()
        counts = update_cache(json.loads(raw), args.cache_dir)
    for sym, n in sorted(counts.items()):
        print(f"{sym}: {n} bars")
    return 0


def _selfcheck() -> None:
    import tempfile

    payload = {"data": {"results": [{"symbol": "TEST", "bars": [
        {"begins_at": "2026-07-06T00:00:00Z", "open_price": "10", "high_price": "11",
         "low_price": "9", "close_price": "10.5", "volume": 1000},
        # degenerate snapshot for the same day, later -> must be dropped
        {"begins_at": "2026-07-07T00:00:00Z", "open_price": "10.5", "high_price": "10.5",
         "low_price": "10.5", "close_price": "10.5", "volume": 0},
    ]}]}}
    with tempfile.TemporaryDirectory() as d:
        n1 = update_cache(payload, d)
        assert n1 == {"TEST": 1}, n1  # volume-0 row dropped
        # a later fetch adds a real 07-07 bar; must merge, not duplicate
        payload2 = {"data": {"results": [{"symbol": "TEST", "bars": [
            {"begins_at": "2026-07-07T00:00:00Z", "open_price": "10.5", "high_price": "12",
             "low_price": "10", "close_price": "11.8", "volume": 2000}]}]}}
        n2 = update_cache(payload2, d)
        assert n2 == {"TEST": 2}, n2
        df = _read_csv(Path(d) / "TEST.csv")
        assert list(df["close"]) == [10.5, 11.8], list(df["close"])
    print("refresh selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selfcheck"]:
        _selfcheck()
    else:
        sys.exit(main())
