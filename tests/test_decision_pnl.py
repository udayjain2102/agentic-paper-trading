import json

import pytest

from rhagent.decision_pnl import score_decisions


def _write_runs(path, lines):
    with path.open("w") as fh:
        for row in lines:
            fh.write(json.dumps(row) + "\n")


def _write_bars(cache_dir, sym, closes, start="2025-01-02"):
    """Sequential trading days (Mon-Fri only skipped for simplicity here we
    just use consecutive calendar days -- get_bars/data doesn't care)."""
    import datetime

    d0 = datetime.date.fromisoformat(start)
    lines = ["date,open,high,low,close,volume"]
    for i, c in enumerate(closes):
        d = d0 + datetime.timedelta(days=i)
        lines.append(f"{d.isoformat()},{c},{c},{c},{c},1000")
    (cache_dir / f"{sym}.csv").write_text("\n".join(lines) + "\n")


def test_score_decisions_computes_forward_returns_and_ignores_noise(tmp_path):
    runs = tmp_path / "runs.jsonl"
    cache = tmp_path / "data"
    cache.mkdir()

    # AAPL closes 100,101,...,109 starting 2025-01-02. Decision on day 0 (100):
    # ret_1d = 101/100-1 = 1%, ret_5d = 105/100-1 = 5%.
    _write_bars(cache, "AAPL", [100 + i for i in range(10)])

    _write_runs(runs, [
        {"event": "run_start", "ts": "2025-01-01T00:00:00Z"},
        {"event": "order_intended", "symbol": "AAPL", "side": "buy",
         "notional_usd": 250.0, "ts": "2025-01-02T05:00:00Z"},
        {"event": "order_rejected", "symbol": "MSFT", "side": "buy",
         "ts": "2025-01-02T06:00:00Z"},
        # Decision near the end of the data: only 1 bar follows (day index 8
        # of 0..9), so ret_1d is scorable but ret_5d is not (lookahead-free
        # boundary handling -- must not fabricate a value).
        {"event": "order_intended", "symbol": "AAPL", "side": "buy",
         "notional_usd": 250.0, "ts": "2025-01-10T05:00:00Z"},
    ])

    summary = score_decisions(runs_path=runs, cache_dir=cache)

    assert summary["n_decisions"] == 2  # non-order_intended events ignored
    row0, row1 = summary["rows"]
    assert row0["ret_1d"] == pytest.approx(1 / 100)
    assert row0["ret_5d"] == pytest.approx(5 / 100)
    assert row1["ret_1d"] == pytest.approx(1 / 108)  # day index 8 -> 9: 109/108-1
    assert row1["ret_5d"] is None  # only 1 bar left, not enough for N=5

    assert summary["n_scored_1d"] == 2
    assert summary["n_scored_5d"] == 1
    assert summary["hit_rate_1d"] == 1.0
    assert summary["avg_ret_1d"] == pytest.approx((row0["ret_1d"] + row1["ret_1d"]) / 2)
    assert summary["avg_ret_5d"] == pytest.approx(row0["ret_5d"])  # only scorable value
    assert summary["hit_rate_5d"] == 1.0

    assert summary["per_symbol"]["AAPL"]["n"] == 2
    assert summary["per_symbol"]["AAPL"]["avg_ret_5d"] == pytest.approx(row0["ret_5d"])


def test_score_decisions_skips_symbol_missing_from_cache(tmp_path):
    runs = tmp_path / "runs.jsonl"
    cache = tmp_path / "data"
    cache.mkdir()
    _write_bars(cache, "AAPL", [100, 99, 98, 97, 96, 95])  # falling: neg ret

    _write_runs(runs, [
        {"event": "order_intended", "symbol": "AAPL", "side": "buy",
         "ts": "2025-01-02T05:00:00Z"},
        # NVDA has no bars cached at all -> unscorable, still counted.
        {"event": "order_intended", "symbol": "NVDA", "side": "buy",
         "ts": "2025-01-02T05:00:00Z"},
    ])

    summary = score_decisions(runs_path=runs, cache_dir=cache)
    assert summary["n_decisions"] == 2
    nvda_row = next(r for r in summary["rows"] if r["symbol"] == "NVDA")
    assert nvda_row["ret_1d"] is None
    assert nvda_row["ret_5d"] is None
    assert summary["n_scored_1d"] == 1  # only AAPL scored
    aapl_row = next(r for r in summary["rows"] if r["symbol"] == "AAPL")
    assert aapl_row["ret_1d"] < 0  # falling prices -> negative forward return
    assert summary["hit_rate_1d"] == 0.0


def test_score_decisions_empty_when_no_accepted_decisions(tmp_path):
    runs = tmp_path / "runs.jsonl"
    cache = tmp_path / "data"
    cache.mkdir()
    _write_runs(runs, [{"event": "run_start", "ts": "2025-01-01T00:00:00Z"}])

    summary = score_decisions(runs_path=runs, cache_dir=cache)
    assert summary == {
        "n_decisions": 0, "rows": [], "per_symbol": {},
        "n_scored_1d": 0, "hit_rate_1d": 0.0, "avg_ret_1d": 0.0,
        "n_scored_5d": 0, "hit_rate_5d": 0.0, "avg_ret_5d": 0.0,
    }
