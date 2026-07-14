"""Refresh the daily-bar CSV cache with fresh MCP historicals.

`get_bars` is cache-first and never refetches an existing CSV, so a live loop
must update the cache itself. This takes the raw `get_equity_historicals`
payload (the dict the Robinhood MCP returns), merges the new bars into each
`data/<SYM>.csv`, dedupes by date, and drops degenerate snapshot rows
(volume==0, the "current price" placeholder the last populate left behind).

Kept separate from data.py because the fetch runs through Claude's MCP session,
not the headless `mcp_fetch` (which needs ROBINHOOD_MCP_TOKEN). Claude fetches,
hands the payload here.

Usage (payload piped as JSON on stdin):
    ... | PYTHONPATH=src python -m rhagent.refresh --cache-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import _normalize, _read_csv, rows_to_df


def update_cache(payload: dict, cache_dir: str | Path = "data") -> dict[str, int]:
    """Merge MCP historicals `payload` into the CSV cache. Returns {symbol: n_rows}.

    A bar with volume==0 is treated as a placeholder snapshot and dropped, so an
    intraday snapshot never overwrites a real settled bar for the same date.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = payload.get("data", payload).get("results", []) or []
    symbols = [r["symbol"] for r in results if r.get("symbol")]
    fresh = _normalize(payload, symbols)

    out: dict[str, int] = {}
    for sym, rows in fresh.items():
        new = rows_to_df([r for r in rows if r.get("volume", 0) != 0])
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.refresh")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--payload", help="JSON file; omit to read stdin")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

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
