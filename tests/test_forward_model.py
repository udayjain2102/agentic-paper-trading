"""forward._model_positions: the price-only engine's forward path.

Mirrors the agent's freeze-to-disk contract: history is seeded legacy/flat,
each new bar is decided once (status "ok"), decisions.jsonl gets a row per
(symbol, date), and a second tick over unchanged bars decides nothing new.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rhagent import forward


def _bars(n: int = 80, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for i, sym in enumerate(["AAA", "BBB"]):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004 * i, 0.01, n)))
        out[sym] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1e6}, index=idx)
    return out


def _decisions(eval_dir) -> list[dict]:
    log = eval_dir / "decisions.jsonl"
    if not log.exists():
        return []
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def test_model_positions_seeds_legacy_and_decides_last_bar(tmp_path):
    bars = _bars()
    pos, excluded = forward._model_positions(tmp_path, bars)

    assert set(pos) == {"AAA", "BBB"}
    # Seeded history is legacy, not "ok" — excluded from returns exactly like
    # the agent's seeded bars. Only genuinely-decided dates stay in.
    assert excluded == set(bars["AAA"].index[:-1])

    last = bars["AAA"].index[-1]
    prev = pd.read_csv(tmp_path / "pos_AAA.csv", parse_dates=["date"]) \
        .set_index("date")
    assert prev.loc[last, "status"] == "ok"
    assert (prev["status"][:-1] == "legacy").all()
    assert set(prev["pos"].dropna().unique()) <= {0.0, 1.0}

    rows = _decisions(tmp_path)
    assert len(rows) == 2  # one per symbol, for the last bar only
    assert all(r["status"] == "ok" and r["date"] == str(last.date())
               for r in rows)
    assert all(r["reason"].startswith("model: p=") for r in rows)


def test_model_positions_is_frozen_across_ticks(tmp_path):
    bars = _bars()
    forward._model_positions(tmp_path, bars)
    n_rows = len(_decisions(tmp_path))

    # Same bars again: nothing new to decide.
    forward._model_positions(tmp_path, bars)
    assert len(_decisions(tmp_path)) == n_rows

    # One new bar: exactly one new decision per symbol, on that bar only.
    grown = {}
    for s, df in bars.items():
        new_idx = df.index[-1] + pd.offsets.BDay(1)
        close = df["close"].iloc[-1] * 1.01
        row = pd.DataFrame({"open": [close], "high": [close * 1.01],
                            "low": [close * 0.99], "close": [close],
                            "volume": [1e6]}, index=[new_idx])
        grown[s] = pd.concat([df, row])
    forward._model_positions(tmp_path, grown)
    rows = _decisions(tmp_path)
    assert len(rows) == n_rows + 2
    assert rows[-1]["date"] == str(grown["AAA"].index[-1].date())
