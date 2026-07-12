import json
from pathlib import Path

import pandas as pd

from rhagent.learn import lessons_from_runs


def _trade(tid, pnl_abs, outcome, vol, gap=0.0, holding=3, symbol="A", side="long"):
    return {
        "trade_id": tid, "run_id": tid.split("#")[0], "symbol": symbol,
        "side": side, "qty": 1.0, "entry_ts": "2026-01-02", "entry_price": 100.0,
        "entry_reason": "r", "exit_ts": "2026-01-05", "exit_price": 101.0,
        "exit_reason": "r", "pnl_pct": pnl_abs / 10000.0, "pnl_abs": pnl_abs,
        "holding_bars": holding, "outcome": outcome,
        "entry_features": {"vol20": vol, "gap": gap, "trend5": 0.0},
    }


def _write_run(run_dir: Path, trades):
    run_dir.mkdir(parents=True)
    rid = run_dir.name
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": rid, "engine": "scripted", "symbols": ["A"],
        "start": "2026-01-01", "end": "2026-01-10",
    }))
    with (run_dir / "trades.jsonl").open("w") as fh:
        for t in trades:
            fh.write(json.dumps(t) + "\n")
    nets = [0.0] * len(trades)
    idx = pd.date_range("2026-01-01", periods=len(nets) or 1, freq="D")
    pd.DataFrame({"date": idx, "net": nets or [0.0]}).to_csv(
        run_dir / "returns.csv", index=False)


def test_lessons_flags_high_vol(tmp_path):
    rid = "2026-07-11T00-00-00Z-aaaaaaaa"
    # High-vol trades are big losers; low-vol trades win.
    trades = [
        _trade(f"{rid}#0001", -500.0, "loss", vol=0.05),
        _trade(f"{rid}#0002", -400.0, "loss", vol=0.05),
        _trade(f"{rid}#0003", -300.0, "loss", vol=0.04),
        _trade(f"{rid}#0004", 100.0, "win", vol=0.001),
        _trade(f"{rid}#0005", 100.0, "win", vol=0.001),
        _trade(f"{rid}#0006", 100.0, "win", vol=0.002),
        _trade(f"{rid}#0007", 100.0, "win", vol=0.002),
        _trade(f"{rid}#0008", 100.0, "win", vol=0.003),
    ]
    _write_run(tmp_path / rid, trades)
    s = lessons_from_runs(tmp_path, min_trades=1)
    assert s and "vol" in s
    assert "high" in s


def test_empty_dir_returns_blank(tmp_path):
    assert lessons_from_runs(tmp_path) == ""
