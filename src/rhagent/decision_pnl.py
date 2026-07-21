"""Grade accepted buy decisions against forward price moves.

The agent journals every guardrail-passed intent as an ``order_intended``
event in journal/runs.jsonl, but nothing ever checks whether those decisions
would have made money. This module scores each one against N-day forward
returns using the same lookahead-free technique as ``memory.recent_outcomes``:
for a decision on date d, find the first bar with date >= d (the entry close),
then read close[i+N] -- only prices at or after the decision date are ever
touched.

Usage: python -m rhagent.decision_pnl
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import get_bars

HORIZONS = (1, 5)


def _load_decisions(runs_path) -> list[dict]:
    p = Path(runs_path)
    if not p.exists():
        return []
    decisions = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "order_intended":
                decisions.append(row)
    return decisions


def _forward_return(close: pd.Series, date_str: str, n: int) -> float | None:
    """close[i+n]/close[i] - 1, where i is the first bar with date >= date_str.
    None if there's no bar on/after date_str, or fewer than n bars follow it."""
    i = close.index.searchsorted(pd.Timestamp(date_str))
    if i >= len(close) or i + n >= len(close):
        return None
    return float(close.iloc[i + n] / close.iloc[i] - 1.0)


def _empty_summary() -> dict:
    summary = {"n_decisions": 0, "rows": [], "per_symbol": {}}
    for n in HORIZONS:
        summary[f"n_scored_{n}d"] = 0
        summary[f"hit_rate_{n}d"] = 0.0
        summary[f"avg_ret_{n}d"] = 0.0
    return summary


def score_decisions(runs_path="journal/runs.jsonl", cache_dir="data") -> dict:
    decisions = _load_decisions(runs_path)
    if not decisions:
        return _empty_summary()

    symbols = sorted({d["symbol"] for d in decisions})
    dates = [d["ts"][:10] for d in decisions]
    # Offline scorer: read only already-cached bars, never fetch. Surprise
    # network I/O during a reporting run would be wrong, and a fresh fetch
    # bounded by the last decision date wouldn't even hold the forward bars a
    # late decision needs. Uncached symbols are honestly left unscored.
    bars = get_bars(symbols, min(dates), max(dates), cache_dir=cache_dir,
                    fetch=lambda *a, **k: {})

    rows = []
    for d in decisions:
        sym = d["symbol"]
        date_str = d["ts"][:10]
        sign = 1.0 if d.get("side", "buy") == "buy" else -1.0
        row = {"date": date_str, "symbol": sym, "side": d.get("side", "buy"),
               "ret_1d": None, "ret_5d": None}
        if sym in bars:
            close = bars[sym]["close"]
            for n in HORIZONS:
                r = _forward_return(close, date_str, n)
                if r is not None:
                    row[f"ret_{n}d"] = sign * r
        rows.append(row)

    summary = {"n_decisions": len(decisions), "rows": rows}
    for n in HORIZONS:
        vals = [r[f"ret_{n}d"] for r in rows if r[f"ret_{n}d"] is not None]
        summary[f"n_scored_{n}d"] = len(vals)
        summary[f"hit_rate_{n}d"] = (sum(v > 0 for v in vals) / len(vals)) if vals else 0.0
        summary[f"avg_ret_{n}d"] = (sum(vals) / len(vals)) if vals else 0.0

    per_symbol = {}
    for sym in symbols:
        sym_rows = [r for r in rows if r["symbol"] == sym]
        vals = [r["ret_5d"] for r in sym_rows if r["ret_5d"] is not None]
        per_symbol[sym] = {
            "n": len(sym_rows),
            "avg_ret_5d": (sum(vals) / len(vals)) if vals else 0.0,
            "hit_rate_5d": (sum(v > 0 for v in vals) / len(vals)) if vals else 0.0,
        }
    summary["per_symbol"] = per_symbol
    return summary


def _fmt(ret: float | None) -> str:
    return f"{ret:+.2%}" if ret is not None else "n/a"


def _print_report(summary: dict) -> None:
    if summary["n_decisions"] == 0:
        print("no accepted decisions to score")
        return
    print(f"{'date':<12}{'symbol':<8}{'ret_1d':>10}{'ret_5d':>10}")
    for r in summary["rows"]:
        print(f"{r['date']:<12}{r['symbol']:<8}{_fmt(r['ret_1d']):>10}{_fmt(r['ret_5d']):>10}")
    print(
        f"\nn_decisions={summary['n_decisions']} "
        f"scored_1d={summary['n_scored_1d']} hit_1d={summary['hit_rate_1d']:.0%} "
        f"avg_1d={summary['avg_ret_1d']:+.2%} "
        f"scored_5d={summary['n_scored_5d']} hit_5d={summary['hit_rate_5d']:.0%} "
        f"avg_5d={summary['avg_ret_5d']:+.2%}"
    )


if __name__ == "__main__":
    _print_report(score_decisions())
