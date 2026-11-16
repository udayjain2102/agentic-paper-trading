"""Deepen the daily-bar cache to ~N years via Yahoo's keyless chart API.

``data/<SYM>.csv`` is cache-first and normally only ever holds the recent
window the daily tick needs. The price-only model engine trains on whatever
depth is cached, so this pulls a decade of split-adjusted daily bars (the
same adjustment the MCP path requests) and merges them in through
``refresh.update_cache`` — deduped by date, volume==0 and unsettled-today
rows dropped, exactly like the daily refresh.

    PYTHONPATH=src python scripts/fetch_long_history.py --years 10

Safe to re-run: existing bars win nothing and lose nothing — fresh rows
simply extend the history backward.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhagent.config import load as load_config  # noqa: E402
from rhagent.refresh import _fetch_yahoo, update_cache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fetch_long_history")
    p.add_argument("--years", type=float, default=10.0)
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    universe = list(load_config().strategy.universe)
    today = date.today()
    start = (today - timedelta(days=int(args.years * 365.25))).isoformat()
    print(f"fetching {len(universe)} symbols, {start} -> {today} (yahoo)")

    payload = _fetch_yahoo(universe, start, today.isoformat())
    got = {r["symbol"] for r in payload.get("data", {}).get("results", [])}
    missing = [s for s in universe if s not in got]
    if missing:
        print(f"!! no data for: {', '.join(missing)}", file=sys.stderr)

    counts = update_cache(payload, args.cache_dir, today=today)
    for sym, n in sorted(counts.items()):
        print(f"{sym}: {n} bars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
