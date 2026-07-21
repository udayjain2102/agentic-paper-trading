"""Historical price data: fetch from the Robinhood MCP, cache to CSV.

Cache-first: if ``<cache_dir>/<SYMBOL>.csv`` exists it is read; otherwise bars are
fetched, normalized, and written. This keeps backtests reproducible and offline,
and confines the live-MCP shape to ``mcp_fetch`` (a thin integration point, like
``McpBroker``). Tests inject a fake ``fetch`` or pre-seed the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_COLUMNS = ["open", "high", "low", "close", "volume"]


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "date"
    for col in _COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index.name = "date"
    for col in _COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def get_bars(symbols, start, end, *, fetch=None, cache_dir="data") -> dict[str, pd.DataFrame]:
    fetch = fetch or fallback_fetch(mcp_fetch, yf_fetch)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, pd.DataFrame] = {}
    missing = []
    for s in symbols:
        path = cache_dir / f"{s}.csv"
        if path.exists():
            out[s] = _read_csv(path)
        else:
            missing.append(s)

    if missing:
        fetched = fetch(missing, start, end)
        for s, rows in fetched.items():
            df = rows_to_df(rows)
            df.to_csv(cache_dir / f"{s}.csv")
            out[s] = df
    return out


def fallback_fetch(*fetchers):
    """Chain price sources: try each fetcher in order, keep whatever it returns,
    and pass only the still-missing symbols to the next one.

    A fetcher that raises (rate-limited, no session, network error) or returns no
    rows for a symbol is skipped for that symbol; the next source gets a shot.
    This is what lets a throttled RH MCP degrade to Yahoo instead of failing the
    whole run.

    ponytail: linear try-in-order chain, no per-source health tracking or
    backoff — add that only if one source starts flapping mid-run.
    """

    def _fetch(symbols, start, end) -> dict[str, list[dict]]:
        remaining = list(symbols)
        out: dict[str, list[dict]] = {}
        for f in fetchers:
            if not remaining:
                break
            try:
                got = f(remaining, start, end) or {}
            except Exception as e:  # noqa: BLE001 — any source failure falls through
                print(
                    f"!! data source {getattr(f, '__name__', f)!r} failed: {e}",
                    file=sys.stderr,
                )
                continue
            for s, rows in got.items():
                if rows:
                    out[s] = rows
            remaining = [s for s in remaining if s not in out]
        return out

    return _fetch


def yf_fetch(symbols, start, end) -> dict[str, list[dict]]:
    """Daily bars from Yahoo Finance (yfinance). No API key. Second link in the
    default fallback chain, used when the RH MCP is unavailable or throttled.
    """
    import yfinance as yf

    data = yf.download(
        list(symbols),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    out: dict[str, list[dict]] = {}
    multi = isinstance(data.columns, pd.MultiIndex)
    for s in symbols:
        try:
            sub = data[s] if multi else data
        except KeyError:
            continue
        sub = sub.dropna(subset=["Close"])
        rows = [
            {
                "date": ts.strftime("%Y-%m-%d"),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": float(r["Volume"]),
            }
            for ts, r in sub.iterrows()
        ]
        if rows:
            out[s] = rows
    return out


def mcp_fetch(symbols, start, end) -> dict[str, list[dict]]:
    """Fetch daily bars from the RH MCP. Integration point — confirm field names.

    Requires a configured MCP session (ROBINHOOD_MCP_TOKEN). Raises if unavailable
    so that offline runs rely on the CSV cache instead.
    """
    from .config import load
    from .mcp_session import mcp_session

    cfg = load()
    with mcp_session(cfg.mcp_url, cfg.mcp_token) as session:
        import anyio

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
    from .broker import _structured

    data = _structured(result)
    return _normalize(data, symbols)


def _normalize(data: dict, symbols) -> dict[str, list[dict]]:
    """Map the RH historicals payload to per-symbol normalized row lists.

    Confirmed live shape (2026-07-06):
        {"data": {"results": [
            {"symbol": "AAPL", "interval": "day", "bars": [
                {"begins_at": "2026-06-22T00:00:00Z",
                 "open_price": "297.31", "close_price": "297.01",
                 "high_price": "302.42", "low_price": "296.76",
                 "volume": 44879914, "session": "reg"}, ...]}]},
         "guide": "..."}
    Prices are strings; results are nested under the top-level "data" key.
    """
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    payload = data.get("data", data)  # tolerate either wrapped or bare
    for entry in payload.get("results", []) or []:
        sym = entry.get("symbol")
        if sym not in out:
            continue
        for bar in entry.get("bars", []) or []:
            out[sym].append(
                {
                    "date": bar["begins_at"][:10],
                    "open": float(bar["open_price"]),
                    "high": float(bar["high_price"]),
                    "low": float(bar["low_price"]),
                    "close": float(bar["close_price"]),
                    "volume": float(bar["volume"]),
                }
            )
    return out
